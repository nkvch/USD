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
from tqdm import tqdm
import torch.multiprocessing as mp
import torch.optim as optim
from multiprocessing import Lock

from a3c_mjuco import ActorCriticNet, make_env

import json

class Args:
    pass

def read_config(file_path):
    args = Args()
    with open(file_path, 'r') as file:
        config = json.load(file)
        for key, value in config.items():
            setattr(args, key, value)
    return args



def eval_and_render(run_dir):
    """
    Run 30 episodes and record them
    """
    # Create environment

    file_path = f'{run_dir}/args.json'  # Replace with the actual file path
    args = read_config(file_path)

    env = make_env(args.env_id, capture_video=True, run_dir=run_dir)()
    # Load policy
    observation_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.shape[0]
    policy = ActorCriticNet(observation_dim,action_dim, args.hidden_size)

    filename = f"{run_dir}/policy.pt"
    print(f"reading {filename}...")
    policy.load_state_dict(torch.load(filename))
    policy.eval()

    count_episodes = 0
    list_rewards = []

    state, _ = env.reset()

    # Run episodes
    while count_episodes < 30:
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float()
            action, log_prob, entropy, value = policy(state_tensor)

        state, reward, terminated, truncated, infos = env.step(action.numpy())

        if terminated or truncated:
            returns = infos["episode"]["r"][0]
            count_episodes += 1
            list_rewards.append(returns)
            print(f"-> Episode {count_episodes}: {returns} returns")
            state, _ = env.reset()

    env.close()

    return np.mean(list_rewards)

if __name__=="__main__":
    # eval_and_render(args, run_dir="runs\\2024-01-25_17-54-11")
    eval_and_render(run_dir="runs\\2024-01-25_18-11-15")



