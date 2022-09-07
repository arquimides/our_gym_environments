# custom_taxi_task_gym_environment
Custom Taxi task RL environment for experiments on Causal-RL.

### Installing Gym

`pip install -e .`

### Testing Installation
```
import gym
import our_taxi_gym

env = gym.make("OurTaxiEnv-v0")
env.reset()
for _ in range(1000):
    env.step(env.action_space.sample())
env.render()
env.close()
```
