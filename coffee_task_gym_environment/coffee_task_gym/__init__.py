from gym.envs.registration import register

from .coffee_task_env import CoffeeTaskEnv

environments = [['CoffeeTaskEnv', 'v0']]

for environment in environments:
    register(
        id='{}-{}'.format(environment[0], environment[1]),
        entry_point='coffee_task_gym:{}'.format(environment[0]),
        nondeterministic=True
    )
