from gym.envs.registration import register

register(
    id="CoffeeTaskEnv-v0",
    entry_point="our_gym_environments.envs:CoffeeTaskEnv",
    nondeterministic=True,
)

register(
    id="TaxiSmallEnv-v0",
    entry_point="our_gym_environments.envs:TaxiSmallEnv",
    nondeterministic=True,
)

register(
    id="TaxiBigEnv-v0",
    entry_point="our_gym_environments.envs:TaxiBigEnv",
    nondeterministic=True,
)


