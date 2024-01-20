# %%
import time
from pathlib import Path
from datetime import datetime
import gymnasium as gym
import json
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.nn.functional import mse_loss
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import torch.multiprocessing as mp

# %%
class Args:
    pass

args = Args()
args.env_id = "HalfCheetah-v4"
args.total_timesteps = 10_000_000
args.num_envs = 16
args.num_steps = 5
args.learning_rate = 5e-4
args.actor_layers = [64, 64]
args.critic_layers  = [64, 64]
args.gamma = 0.99
args.gae = 1.0
args.value_coef = 0.5
args.entropy_coef = 0.01
args.clip_grad_norm = 0.5
args.seed = 0

args.batch_size = int(args.num_envs * args.num_steps)
args.num_updates = int(args.total_timesteps // args.batch_size)

# %%
def make_env(env_id, capture_video=False, run_dir="."):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env=env,
                video_folder=f"{run_dir}/videos",
                episode_trigger=lambda x: x,
                disable_logger=True,
            )
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda state: np.clip(state, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

        return env

    return thunk

# %%
def compute_advantages(rewards, flags, values, last_value, args):
    advantages = torch.zeros((args.num_steps, args.num_envs))
    adv = torch.zeros(args.num_envs)

    for i in reversed(range(args.num_steps)):
        returns = rewards[i] + args.gamma * flags[i] * last_value
        delta = returns - values[i]

        adv = delta + args.gamma * args.gae * flags[i] * adv
        advantages[i] = adv

        last_value = values[i]

    return advantages

# %%
class RolloutBuffer:
    def __init__(self, num_steps, num_envs, observation_shape, action_shape):
        self.states = np.zeros((num_steps, num_envs, *observation_shape), dtype=np.float32)
        self.actions = np.zeros((num_steps, num_envs, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.flags = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)

        self.step = 0
        self.num_steps = num_steps

    def push(self, state, action, reward, flag, value):
        self.states[self.step] = state
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.flags[self.step] = flag
        self.values[self.step] = value

        self.step = (self.step + 1) % self.num_steps

    def get(self):
        return (
            torch.from_numpy(self.states),
            torch.from_numpy(self.actions),
            torch.from_numpy(self.rewards),
            torch.from_numpy(self.flags),
            torch.from_numpy(self.values),
        )

# %%
class ActorCriticNet(nn.Module):
    def __init__(self, observation_shape, action_dim, actor_layers, critic_layers):
        super().__init__()

        self.actor_net = self._build_net(observation_shape, actor_layers)
        self.critic_net = self._build_net(observation_shape, critic_layers)

        self.actor_net.append(self._build_linear(actor_layers[-1], action_dim, std=0.01))
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        self.critic_net.append(self._build_linear(critic_layers[-1], 1, std=1.0))

    def _build_linear(self, in_size, out_size, apply_init=True, std=np.sqrt(2), bias_const=0.0):
        layer = nn.Linear(in_size, out_size)

        if apply_init:
            torch.nn.init.orthogonal_(layer.weight, std)
            torch.nn.init.constant_(layer.bias, bias_const)

        return layer

    def _build_net(self, observation_shape, hidden_layers):
        layers = nn.Sequential()
        in_size = np.prod(observation_shape)

        for out_size in hidden_layers:
            layers.append(self._build_linear(in_size, out_size))
            layers.append(nn.Tanh())
            in_size = out_size

        return layers

    def forward(self, state):
        mean = self.actor_net(state)
        std = self.actor_logstd.exp()
        distribution = Normal(mean, std)

        action = distribution.sample()

        value = self.critic_net(state).squeeze(-1)

        return action, value

    def evaluate(self, states, actions):
        mean = self.actor_net(states)
        std = self.actor_logstd.exp()
        distribution = Normal(mean, std)

        log_probs = distribution.log_prob(actions).sum(-1)
        entropy = distribution.entropy().sum(-1)

        values = self.critic_net(states).squeeze(-1)

        return log_probs, values, entropy

    def critic(self, state):
        return self.critic_net(state).squeeze(-1)

# %%
class ThreadLogger:
    def __init__(self, run_dir, thread_id):
        self.run_dir = run_dir
        self.thread_id = thread_id

    def log(self, text):
        with open(f"{self.run_dir}/thread_{self.thread_id}.log", "a") as f:
            f.write(text + "\n")

# %%
import threading
import torch.multiprocessing as mp

def worker(global_policy, global_optimizer, global_step, args, run_dir):
    try:
        print(f"Worker {threading.get_ident()} started")
        env = make_env(args.env_id, run_dir=run_dir)()

        local_policy = ActorCriticNet(
            env.observation_space.shape,
            env.action_space.shape[0],
            args.actor_layers,
            args.critic_layers,
        )

        rollout_buffer = RolloutBuffer(
            args.num_steps,
            args.num_envs,
            env.observation_space.shape,
            env.action_space.shape,
        )

        state, _ = env.reset()

        while global_step.value < args.total_timesteps:
            local_policy.load_state_dict(global_policy.state_dict())

            for _ in range(args.num_steps):
                global_step.value += 1

                with torch.no_grad():
                    action, value = local_policy(torch.from_numpy(state).float())

                action = action.detach().numpy()
                action = action.squeeze(0)
                next_state, reward, terminated, truncated, infos = env.step(action)

                flag = 1.0 - np.logical_or(terminated, truncated)
                value = value.detach().numpy()

                rollout_buffer.push(state, action, reward, flag, value)

                state = next_state

            states, actions, rewards, flags, values = rollout_buffer.get()

            with torch.no_grad():
                last_value = local_policy.critic(torch.from_numpy(state).float()).detach().numpy()

            advantages = compute_advantages(rewards, flags, values, last_value, args)
            td_target = advantages + values

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            log_probs, td_predict, entropy = local_policy.evaluate(states, actions)

            actor_loss = -(log_probs * advantages).mean()
            critic_loss = mse_loss(td_target, td_predict)
            entropy_loss = entropy.mean()

            loss = actor_loss + args.value_coef * critic_loss - args.entropy_coef * entropy_loss

            local_policy.zero_grad()
            loss.backward()

            clip_grad_norm_(local_policy.parameters(), args.clip_grad_norm)

            with global_optimizer.get_lock():
                for local_param, global_param in zip(local_policy.parameters(), global_policy.parameters()):
                    global_param._grad = local_param.grad

                global_optimizer.step()

            if global_step.value % 100 == 0:
                print(f"Worker {threading.get_ident()} step {global_step.value} loss {loss.item()} Last 5 rewards mean {rewards[-5:].mean()}")
    except:
        import traceback
        traceback.print_exc()
        raise


class Worker(mp.Process):
    def __init__(self, global_policy, global_optimizer, global_step, args, run_dir):
        super().__init__()
        self.global_policy = global_policy
        self.global_optimizer = global_optimizer
        self.global_step = global_step
        self.args = args
        self.run_dir = run_dir

    def run(self):
        worker(self.global_policy, self.global_optimizer, self.global_step, self.args, self.run_dir)

# %%
import torch.optim as optim
from multiprocessing import Lock

class GlobalOptimizer:
    def __init__(self, global_policy, args):
        self.global_policy = global_policy
        self.optimizer = optim.RMSprop(global_policy.parameters(), lr=args.learning_rate, alpha=0.99, eps=1e-5)

        self.lock = Lock()

    def step(self):
        self.optimizer.step()

    def get_lock(self):
        return self.lock
    
    def zero_grad(self):
        self.optimizer.zero_grad()

# %%
def train(args):
    run_dir = Path(f"runs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    run_dir.mkdir(parents=True)

    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    env = make_env(args.env_id, capture_video=True, run_dir=run_dir)()

    global_policy = ActorCriticNet(
        env.observation_space.shape,
        env.action_space.shape[0],
        args.actor_layers,
        args.critic_layers,
    )

    global_policy.share_memory()

    global_optimizer = GlobalOptimizer(global_policy, args)

    global_step = mp.Value('i', 0)

    workers = [Worker(global_policy, global_optimizer, global_step, args, run_dir) for _ in range(args.num_envs)]

    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()


if __name__ == '__main__':
    train(args)
