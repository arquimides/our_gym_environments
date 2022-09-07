from gym.envs.registration import register

from .our_taxi_gym_env import OurTaxiEnv

environments = [['OurTaxiEnv', 'v0']]

for environment in environments:
    register(
        id='{}-{}'.format(environment[0], environment[1]),
        entry_point='our_taxi_gym:{}'.format(environment[0]),
        nondeterministic=True
    )
