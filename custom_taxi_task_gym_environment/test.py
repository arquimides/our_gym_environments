import gym
import our_taxi_gym

env = gym.make("OurTaxiEnv-v0")
env.reset()
for _ in range(1000):
    env.step(env.action_space.sample())
    env.render()
env.close()