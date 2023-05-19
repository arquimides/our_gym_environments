import gymnasium as gym
import our_gym_environments
import pygame
from gymnasium.utils.play import PlayPlot, play
import numpy as np

env_ids = ["our_gym_environments/CoffeeTaskEnv-v0", "our_gym_environments/TaxiSmallEnv-v0", "our_gym_environments/TaxiBigEnv-v0"]
selected_env = env_ids[0] # Change this value to select the environment you want to test

mode = ["manual_play", "random_action"]
selected_mode = mode[1] # Change this value to select the mode you want to test

mappings = {"our_gym_environments/CoffeeTaskEnv-v0": {(pygame.K_g,): 0, (pygame.K_u,): 1, (pygame.K_b,): 2, (pygame.K_d,): 3},
            "our_gym_environments/TaxiSmallEnv-v0": {(pygame.K_DOWN,): 0, (pygame.K_UP,): 1, (pygame.K_RIGHT,): 2, (pygame.K_LEFT,): 3, (pygame.K_p,): 4, (pygame.K_d,): 5},
            "our_gym_environments/TaxiBigEnv-v0": {(pygame.K_DOWN,): 0, (pygame.K_UP,): 1, (pygame.K_RIGHT,): 2, (pygame.K_LEFT,): 3, (pygame.K_p,): 4, (pygame.K_d,): 5}
            }

if selected_mode == "manual_play":

    env = gym.make(selected_env, render_mode="rgb_array", env_type = "deterministic", render_fps=1)

    def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
        return [rew, ]

    #plotter = PlayPlot(callback, 30 * 5, ["reward"])

    play(env, keys_to_action=mappings[selected_env], noop=1000, callback = callback)

elif selected_mode == "random_action":
    env = gym.make(selected_env, render_mode="human", env_type="stochastic", render_fps=64)
    observation, info = env.reset(options={'state_index': 0})

    for i in range(10000000):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()