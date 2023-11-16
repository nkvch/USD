import gymnasium as gym

# Create the environment
env = gym.make('HalfCheetah-v4')

print(f'Observation space: {env.observation_space}')
print(f'Action space: {env.action_space}')

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space

print(num_inputs, num_actions)

n_episodes = 10

for i_episode in range(n_episodes):
    observation = env.reset()
    for t in range(100):
        # Choose a random action
        action = env.action_space.sample()

        print(f'Aciton is {action}')

        observation, reward, done, truncated, info = env.step(action)

        print(f'Observation is {observation}')

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
