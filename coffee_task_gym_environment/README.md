# coffee_task_gym_environment
Coffee task RL environment for experiments on Causal-RL.

### Installing Gym

`pip install -e .`

### Installing Tkinter for Dynamic plotting
apt-get install python-tk

### Testing Installation
```
import gym
import coffee_task__gym

env = gym.make("CoffeeTaskEnv-v0")
env.reset()
for _ in range(1000):
    env.step(env.action_space.sample())
env.render()
env.close()
```
