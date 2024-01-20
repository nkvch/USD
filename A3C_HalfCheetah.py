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
import wandb

import gym
import torch as T
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical





def make_env(env_id, capture_video=False, run_dir="."):
 
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




class SharedAdam(T.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
            weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps,
                weight_decay=weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = T.zeros(1)
                state['exp_avg'] = T.zeros_like(p.data)
                state['exp_avg_sq'] = T.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

class SharedRMSProp(T.optim.RMSprop):
    def __init__(self, params, lr=1e-3, alpha=0.99, eps=1e-5, 
                 weight_decay=0, momentum=0, centered=False):
        super(SharedRMSProp, self).__init__(params, lr=lr, alpha=alpha, eps=eps,
                weight_decay=weight_decay, momentum=momentum, centered=centered)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = T.zeros(1)
                state['square_avg'] = T.zeros_like(p.data)
                state['square_avg'].share_memory_()


class ActorCriticNet(nn.Module):
    def __init__(self, observation_shape, action_shape, action_dim, actor_layers, critic_layers):
        super().__init__()

        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.action_shape = action_shape

        self.actor_net = self._build_net(observation_shape, actor_layers)
        self.critic_net = self._build_net(observation_shape, critic_layers)

        self.actor_net.append(self._build_linear(actor_layers[-1], self.action_dim, std=0.01))
        self.actor_logstd = nn.Parameter(torch.zeros(self.action_dim))

        self.critic_net.append(self._build_linear(critic_layers[-1], 1, std=1.0))



        self.rewards = []
        self.actions = []
        self.states = []

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
    
    
    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def forward(self, state):
        mean = self.actor_net(state)
        std = self.actor_logstd.expand_as(mean).exp()
        distribution = Normal(mean, std)

        action = distribution.sample()

        value = self.critic_net(state).squeeze(-1)

        return action, value
    
    def compute_advantages(self, done, values):

        advantages = torch.zeros(T_MAX)
        adv = torch.zeros(1)

        # print('v', v.shape)
        

        for i in reversed(range(T_MAX)):
            returns = self.rewards[i] + args.gamma * int(done) * values[-1]
            delta = returns - values[i]

            adv = delta + args.gamma * args.gae * int(done) * adv
            advantages[i] = adv

            last_value = values[i]
        
        # print("advantages", advantages.shape, advantages)

        return advantages
    
    # def calc_R(self, done):
    #     states = T.tensor(self.states, dtype=T.float)
    #     _, v = self.forward(states)

    #     R = v[-1]*(1-int(done))

    #     batch_return = []
    #     print("rewards --- ", self.rewards)
    #     for reward in self.rewards[::-1]:
            
    #         R = reward + args.gamma*R
    #         batch_return.append(R)
    #     batch_return.reverse()
    #     batch_return = T.tensor(batch_return, dtype=T.float)

    #     return batch_return
    
    def calc_loss(self, done):
        states = T.tensor(self.states, dtype=T.float)
        actions = T.tensor(self.actions, dtype=T.float)

        pi, values = self.forward(states)

        advantages = self.compute_advantages(done, values)
        td_target = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute losses
        log_probs, td_predict, entropy = self.evaluate(states, actions)

        actor_loss = (-log_probs * advantages).mean()
        critic_loss = mse_loss(td_target, td_predict)
        entropy_loss = entropy.mean()

        loss = actor_loss + critic_loss * args.value_coef - entropy_loss * args.entropy_coef

        # print("actor_loss", actor_loss.shape, actor_loss) # should be [steps]
        # print("critic_loss", critic_loss.shape, critic_loss) # should be [steps]
        # print("entropy_loss", entropy_loss.shape, entropy_loss)
        return loss
    

    def evaluate(self, states, actions):
        mean = self.actor_net(states)
        std = self.actor_logstd.expand_as(mean).exp()
        distribution = Normal(mean, std)

        log_probs = distribution.log_prob(actions).sum(-1)
        entropy = distribution.entropy().sum(-1)

        values = self.critic_net(states).squeeze()

        # print('states', states.shape)
        # print('values', values.shape)
        # print('log_probs', log_probs.shape)
        # print('entropy', entropy.shape)

        return log_probs, values, entropy


    def critic(self, state):
        return self.critic_net(state).squeeze(-1)


    def choose_action(self, observation):
        action, value = self.forward(torch.from_numpy(observation).float())

        return action
    

class Agent(mp.Process):
    def __init__(self, global_actor_critic, optimizer, observation_shape, action_shape, actor_layers, critic_layers, name, global_ep_idx, env_id):
        super(Agent, self).__init__()
        self.local_actor_critic = ActorCriticNet(observation_shape, action_shape, action_dim, actor_layers, critic_layers)
        self.global_actor_critic = global_actor_critic
        self.name = 'w%02i' % name
        self.episode_idx = global_ep_idx
        self.env = gym.make(env_id)
        self.optimizer = optimizer

    def run(self):
        t_step = 1
        while self.episode_idx.value < N_GAMES:
            done = False
            if args.seed:
                torch.manual_seed(args.seed)
                observation = self.env.reset(seed=args.seed)
            else:
                observation = self.env.reset(seed=args.seed)

            score = 0
            self.local_actor_critic.clear_memory()
            while not done:
                action = self.local_actor_critic.choose_action(observation)
                action = action.detach().numpy()

                observation_, reward, done, info = self.env.step(action)
                score += reward
                self.local_actor_critic.remember(observation, action, reward)
                if t_step % T_MAX == 0 or done:
                    loss = self.local_actor_critic.calc_loss(done)
                    self.optimizer.zero_grad()
                    loss.backward()
                    for local_param, global_param in zip(self.local_actor_critic.parameters(),
                                                         self.global_actor_critic.parameters()):
                        global_param._grad = local_param.grad
                    self.optimizer.step()
                    self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())
                    self.local_actor_critic.clear_memory()

                t_step += 1
                observation = observation_
            with self.episode_idx.get_lock():
                self.episode_idx.value += 1
            print(self.name, 'episode ', self.episode_idx.value, 'reward %.1f' % score)

N_GAMES = 1000
T_MAX = 10

class Args:
    pass

args = Args()
args.env_id = "HalfCheetah-v4"
# args.total_timesteps = 1_000_000
# args.num_steps = 5
args.lr = 1e-4
args.actor_layers = [64, 64]
args.critic_layers  = [64, 64]
args.gamma = 0.99
args.gae = 1.0
args.value_coef = 1
args.entropy_coef = 0.01
# args.clip_grad_norm = 0.5
args.seed = 3

if __name__=="__main__":

    env = gym.make(id=args.env_id)

    observation_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    action_dim = np.prod(action_shape)

    del env

    print(observation_shape, action_shape)

    global_actor_critic = ActorCriticNet(observation_shape, action_shape, action_dim, args.actor_layers, args.critic_layers)
    global_actor_critic.share_memory()
    # optim = SharedAdam(global_actor_critic.parameters(), lr=args.lr, betas = (0.92, 0.999))
    optimizer = SharedRMSProp(global_actor_critic.parameters(), lr=args.lr, alpha=0.99, eps=1e-5)
    global_ep = mp.Value('i', 0)

    workers = [Agent(global_actor_critic,
                     optimizer,
                     observation_shape,
                     action_dim,
                     args.actor_layers,
                     args.critic_layers,
                     name=i,
                     global_ep_idx=global_ep,
                     env_id=args.env_id) for i in range(mp.cpu_count())]

    [w.start() for w in workers]
    [w.join() for w in workers]
