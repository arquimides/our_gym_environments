import gymnasium as gym
import our_gym_environments
import pygame
from gymnasium.utils.play import PlayPlot, play

env_ids = ["CoffeeTaskEnv-v0", "TaxiSmallEnv-v0", "TaxiBigEnv-v0"]
selected_env = env_ids[0] # Change this value to select the environment you want to test

#env = gym.make(selected_env, render_mode="rgb_array", env_type = "stochastic", render_fps=64)

# def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
#      return [rew, ]
#
#
# mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1, (pygame.K_UP,): 2, (pygame.K_DOWN,): 3}
# plotter = PlayPlot(callback, 30 * 5, ["reward"])
#
# play(env, keys_to_action=mapping, callback=plotter.callback)


env = gym.make(selected_env, render_mode="human", env_type = "stochastic", render_fps=64)
observation, info = env.reset(options={'state_index': 0})

for i in range(10000000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()