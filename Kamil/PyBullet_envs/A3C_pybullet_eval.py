#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import gym
import numpy as np
import pybullet_envs
import time
import torch
import time
import json
import numpy as np
import torch
import gym

from A3C_pybullet import ActorCriticNet

import json


class Args:
    pass

# args.batch_size = int(args.num_envs * args.num_steps)
# args.num_updates = int(args.total_timesteps // args.batch_size)

def make_env(env_id, capture_video=False, run_dir="."):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render=False, max_episode_steps=1000)
            # env.reset()
            # env.render(mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env=env,
                video_folder=f"{run_dir}/videos",
                episode_trigger=lambda x: x,
            )
        else:
            env = gym.make(env_id, render=False, max_episode_steps=1000)
            env.reset()
            # env.render(mode="human")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda state: np.clip(state, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

        return env

    return thunk


def read_config(file_path):
    args = Args()
    with open(file_path, 'r') as file:
        config = json.load(file)
        for key, value in config.items():
            setattr(args, key, value)
    return args



def run_simulation(run_dir):
    file_path = f'{run_dir}/args.json'  # Replace with the actual file path
    args = read_config(file_path)

    env = gym.make(args.env_id, render=True, max_episode_steps=200)
    env.reset()
    env.render(mode="human")

    count_episodes = 0



    observation_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.shape[0]
    policy = ActorCriticNet(observation_dim,action_dim, args.hidden_size)

    i = 0
    while count_episodes < 30:
        frame = 0
        score = 0
        restart_delay = 0
        state= env.reset()
        count_episodes += 1

        while 1:
            time.sleep(1. / 60.)
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float()
                action, log_prob, entropy, value = policy(state_tensor)
            action = action.numpy()
            state, r, done, infos = env.step(action)
 
            score += r
            frame += 1
            print(frame)
            still_open = env.render(mode="human")

            if still_open == False:
                return
            if not done: continue
            if restart_delay == 0:
                print("score=%0.2f in %i frames" % (score, frame))
                restart_delay = 60 * 2    # 2 sec at 60 fps
            else:
                restart_delay -= 1
                if restart_delay == 0: break




if __name__ == "__main__":
    # run_simulation(run_dir="runs\\2024-01-25_19-23-32")
    run_simulation(run_dir="runs\\2024-01-25_21-58-51")