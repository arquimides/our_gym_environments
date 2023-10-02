import gymnasium as gym
import our_gym_environments
import pygame
from gymnasium.utils.play import PlayPlot, play
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
cv2.ocl.setUseOpenCL(False)

env_ids = ["our_gym_environments/CoffeeTaskEnv-v0", "our_gym_environments/TaxiSmallEnv-v0", "our_gym_environments/TaxiBigEnv-v0", "our_gym_environments/TaxiAtariSmallEnv-v0"]
print("HOla")
selected_env = env_ids[3] # Change this value to select the environment you want to test

mode = ["manual_play", "random_action", "image_generation"]
selected_mode = mode[2] # Change this value to select the mode you want to test

mappings = {"our_gym_environments/CoffeeTaskEnv-v0": {(pygame.K_g,): 0, (pygame.K_u,): 1, (pygame.K_b,): 2, (pygame.K_d,): 3},
            "our_gym_environments/TaxiSmallEnv-v0": {(pygame.K_DOWN,): 0, (pygame.K_UP,): 1, (pygame.K_RIGHT,): 2, (pygame.K_LEFT,): 3, (pygame.K_p,): 4, (pygame.K_d,): 5},
            "our_gym_environments/TaxiBigEnv-v0": {(pygame.K_DOWN,): 0, (pygame.K_UP,): 1, (pygame.K_RIGHT,): 2, (pygame.K_LEFT,): 3, (pygame.K_p,): 4, (pygame.K_d,): 5}
            }

if selected_mode == "manual_play":

    env = gym.make(selected_env, render_mode="rgb_array", env_type = "deterministic", reward_type = "new", render_fps=1)

    def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
        return [rew, ]

    #plotter = PlayPlot(callback, 30 * 5, ["reward"])

    play(env, keys_to_action=mappings[selected_env], noop=1000, callback = callback)

elif selected_mode == "random_action":
    env = gym.make(selected_env, render_mode="human", env_type="stochastic", reward_type = "new", render_fps=64)
    observation, info = env.reset(options={'state_index': 0, 'state_type': "original"})

    for i in range(10000000):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

elif selected_mode == "image_generation":
    env = gym.make(selected_env, render_mode="rgb_array", env_type="deterministic", reward_type = "original", render_fps=64)

    # Specify the folder where you want to save the image
    save_folder = 'env_images'

    for i in range(env.num_states):
        observation, info = env.reset(options={'state_index': i, 'state_type': "original"}, return_info = True)

        # Resize the image to 84,84
        image = cv2.resize(observation, (84, 84), interpolation=cv2.INTER_LINEAR)
        # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Add a channel dimension to make it (84, 84, 1)
        image = np.expand_dims(image, axis=-1)

        # Check if the folder exists, and create it if not
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # Specify the image file name
        image_filename = 'state_{}.png'.format(info['integer_state'])  # You can customize the file name

        # Specify the complete path to save the image
        save_path = os.path.join(save_folder, image_filename)

        # Save the image to the specified folder
        cv2.imwrite(save_path, image)

        # Optionally, you can print the path where the image is saved
        print("Image saved at:", save_path)


    env.close()