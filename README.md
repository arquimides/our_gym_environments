# our_gym_environments
Custom OpenAI Gymnasium environments designed for experiments on Causal-RL (See the following link: https://github.com/arquimides/carl)
There are three environments with the following ids:
1. "our_gym_environments/CoffeeTaskEnv-v0" : Custom implementation of the Coffee-Task[^1] with some render for fun visualisation
2. "our_gym_environments/TaxiSmallEnv-v0": Custom implementation of the Taxi-Task (https://gymnasium.farama.org/environments/toy_text/taxi/) with some extra-walls added.
3. "our_gym_environments/TaxiEnv-v0": Custom implementation of the Taxi-Task (https://gymnasium.farama.org/environments/toy_text/taxi/) with higher map , more walls and more possible passenger origin-destinations.

You can see the detailed info (states, actions, rewards, ect) of each environment in their corresponding python file at `our_gym_environments/our_gym_environments/envs/` folder.

### Installation
You can use this project as standalone to build over it or integrated with some other project you have. To start, clone/download the current repository to your preferred local directory.

STANDALONE PROJECT: We recommend you to create a python virtual environment to install all the dependencies (using conda or any other).

INTEGRATED INSIDE ANOTHER PROJECT:
If you want to use "our_gym_environments" inside already working project be sure to activate the corresponding project virtual-environment first.

Once the corresponding virtual-environment is created and activated, just navigate to the path of `our_gym_environments/` folder in your local computer and execute the following command.

```
pip install -e .
```
That will install all the dependencies ['gymnasium', 'pygame', 'matplotlib', 'numpy'] and the environment package[our-gym-environments]. So you are ready to use the environments.

### Installing Tkinter for Dynamic plotting
In Linux you may need to install Tkinker package manually:
```
apt-get install python-tk
```
### Testing Installation

To test your installation you can edit the `test.py` script. We create a keyboard mapping to agent actions for each environment so you can play the environment or run an agent performing random actions.
Just edit the lines 8 and 11 to select the environment id and mode in `test.py` and run the program.

```
import gymnasium as gym
import our_gym_environments
import pygame
from gymnasium.utils.play import PlayPlot, play
import numpy as np

env_ids = ["our_gym_environments/CoffeeTaskEnv-v0", "our_gym_environments/TaxiSmallEnv-v0", "our_gym_environments/TaxiBigEnv-v0"]
selected_env = env_ids[0] #Change this value to select the environment you want to test

mode = ["manual_play", "random_action"]
selected_mode = mode[1] #Change this value to select the mode you want to test

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
```

### References

[^1]: Author, Year, "Title of the Work"
