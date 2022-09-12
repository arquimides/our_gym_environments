import gym
import coffee_task_gym
import pygame
from gym.utils.play import play
from gym.utils.play import PlayPlot

def callback(obs_t, obs_tp1, action, rew, done, info):
     return [rew,]


mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1, (pygame.K_UP,): 2, (pygame.K_DOWN,): 3}
plotter = PlayPlot(callback, 30 * 5, ["reward"])

env = gym.make("CoffeeTaskEnv-v0")
play(env, keys_to_action=mapping, callback=plotter.callback )

# env = gym.make("CoffeeTaskEnv-v0", env_type = "stochastic")
# env.reset()
# for _ in range(1000):
#     env.step(env.action_space.sample())
#     env.render()
# env.close()