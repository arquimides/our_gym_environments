import gym
import coffee_task_gym

env = gym.make("CoffeeTaskEnv-v0", env_type = "stochastic")
env.reset()
for _ in range(1000):
    env.step(env.action_space.sample())
    env.render()
env.close()