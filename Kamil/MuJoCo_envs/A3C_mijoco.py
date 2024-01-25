from pathlib import Path
from datetime import datetime
import gymnasium as gym
import json
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.nn.functional import mse_loss
import torch.multiprocessing as mp
import torch.optim as optim


class Args:
    pass

args = Args()
args.env_id = "HalfCheetah-v4"
args.total_timesteps = 10_000_000
args.num_envs = mp.cpu_count()
args.num_steps = 500
args.learning_rate = 1e-4
args.hidden_size =[256, 256]
args.gamma = 0.99
args.gae = 1.0
args.value_coef = 0.5
args.entropy_coef = 0.01
args.clip_grad_norm = 0.5
args.seed = 0


def make_env(env_id, capture_video=False, run_dir="."):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array", max_episode_steps=1000)
            env = gym.wrappers.RecordVideo(
                env=env,
                video_folder=f"{run_dir}/videos",
                episode_trigger=lambda x: x,
                disable_logger=True,
            )
        else:
            env = gym.make(env_id, max_episode_steps=1000)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda state: np.clip(state, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

        return env

    return thunk



class ActorCriticNet(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_size=[256, 256]):
        super(ActorCriticNet, self).__init__()

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(observation_dim, hidden_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], action_dim)
        )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(observation_dim, hidden_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], 1)
        )

        # Standard deviation for action distribution
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        mean = self.actor(state)
        std = torch.exp(self.log_std)
        distribution = Normal(mean, std)

        # Sample an action from the distribution
        action = distribution.sample()

        # Compute the log probability of the action
        log_prob = distribution.log_prob(action).sum(-1)

        # Compute the entropy of the distribution
        entropy = distribution.entropy().sum(-1)

        # Compute the value estimate
        value = self.critic(state)

        return action, log_prob, entropy, value

    def evaluate(self, state, action):
        mean = self.actor(state)
        std = torch.exp(self.log_std)
        distribution = Normal(mean, std)

        log_prob = distribution.log_prob(action).sum(-1)
        entropy = distribution.entropy().sum(-1)
        value = self.critic(state)

        return log_prob, entropy, value



def worker(worker_id, global_policy, global_optimizer, args, global_step, run_dir):
    # Initialize the environment
    env = gym.make(args.env_id, max_episode_steps=1000)
    # env.seed(args.seed + worker_id)
    print(f"Worker {worker_id} started")

    # Initialize local policy (copy of the global policy)
    observation_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.shape[0]
    local_policy = ActorCriticNet(observation_dim, action_dim, args.hidden_size)
    local_policy.load_state_dict(global_policy.state_dict())

    state, _ = env.reset()
    
    done = True

    while global_step.value < args.total_timesteps:
        local_policy.load_state_dict(global_policy.state_dict())

        # Initialize variables for storing rollout data
        states, actions, rewards, log_probs, values, entropies = [], [], [], [], [], []
        state, _ = env.reset()
        for _ in range(args.num_steps):
            # Run the policy
            state_tensor = torch.from_numpy(state).float()
            action, log_prob, entropy, value = local_policy(state_tensor)

            next_state, reward, terminated, truncated, _ = env.step(action.numpy())
            done = np.logical_or(terminated, truncated)

            with global_step.get_lock():
                global_step.value += 1

            # Store the experience
            states.append(state_tensor)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            entropies.append(entropy)

            state = next_state

            if done:
                state, _ = env.reset()
                break

        # Compute the last value for advantage calculation
        state_tensor = torch.from_numpy(state).float()
        _, _, _, last_value = local_policy(state_tensor)

        # Compute returns and advantages
        returns, advantages = compute_returns_and_advantages(rewards, values, last_value, done, args)

        # Convert lists to tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        log_probs = torch.stack(log_probs)
        returns = torch.tensor(returns)
        advantages = torch.tensor(advantages)
        values = torch.stack(values).squeeze()

        # Calculate losses
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = mse_loss(returns, values)
        entropy_loss = torch.stack(entropies).mean()

        total_loss = actor_loss + args.value_coef * critic_loss - args.entropy_coef * entropy_loss

        # Update global policy
        global_optimizer.zero_grad()
        total_loss.backward()
        for local_param, global_param in zip(local_policy.parameters(), global_policy.parameters()):
            global_param._grad = local_param.grad
        global_optimizer.step()

        print(f"Worker {worker_id} step {global_step.value} loss {total_loss.item()} Last 5 rewards mean {np.mean(rewards[-5:])}")
        if global_step.value % 100 == 0:
            torch.save(global_policy.state_dict(), f"{run_dir}/policy.pt")

def compute_returns_and_advantages(rewards, values, last_value, done, args):
    n_steps = len(rewards)
    returns = torch.zeros(n_steps)
    advantages = torch.zeros(n_steps)

    R = last_value if not done else 0
    A = 0  # Advantage

    for step in reversed(range(n_steps)):
        next_value = 0 if step == n_steps - 1 else values[step + 1]

        R = rewards[step] + args.gamma * R
        td_error = rewards[step] + args.gamma * next_value - values[step]
        A = td_error + args.gamma * args.gae * A

        returns[step] = R
        advantages[step] = A

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

    return returns, advantages



class Worker(mp.Process):
    def __init__(self, global_policy, global_optimizer, global_step, args, run_dir, id):
        super().__init__()
        self.global_policy = global_policy
        self.global_optimizer = global_optimizer
        self.global_step = global_step
        self.args = args
        self.run_dir = run_dir
        self.worker_id = id

    def run(self):
        worker(self.worker_id, self.global_policy, self.global_optimizer, self.args, self.global_step, self.run_dir)



class GlobalOptimizer:
    def __init__(self, parameters, lr=1e-4, alpha=0.99, eps=1e-5):

        self.optimizer = optim.RMSprop(parameters, lr=lr, alpha=alpha, eps=eps)

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def share_memory(self):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.share_memory_()





def train(args):
    run_dir = Path(f"runs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    run_dir.mkdir(parents=True)

    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    env = make_env(args.env_id, capture_video=False, run_dir=run_dir)()
    observation_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.shape[0]
    global_policy = ActorCriticNet(observation_dim,action_dim, args.hidden_size)

    global_policy.share_memory()

    global_optimizer = GlobalOptimizer(global_policy.parameters(), args.learning_rate)
    global_optimizer.share_memory()

    global_step = mp.Value('i', 0)

    workers = [Worker(global_policy, global_optimizer, global_step, args, run_dir, id) for id in range(args.num_envs)]

    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()


if __name__ == '__main__':
    train(args)
