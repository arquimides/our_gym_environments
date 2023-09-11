from gymnasium.envs.registration import register

register(
    id="our_gym_environments/CoffeeTaskEnv-v0",
    entry_point="our_gym_environments.envs:CoffeeTaskEnv",
    nondeterministic=True,
)

register(
    id="our_gym_environments/TaxiSmallEnv-v0",
    entry_point="our_gym_environments.envs:TaxiSmallEnv",
    nondeterministic=True,
)

# The new Taxi as Atari game
register(
    id="our_gym_environments/TaxiAtariSmallEnv-v0",
    entry_point="our_gym_environments.envs:TaxiAtariSmallEnv",
    nondeterministic=True,
)

register(
    id="our_gym_environments/TaxiBigEnv-v0",
    entry_point="our_gym_environments.envs:TaxiBigEnv",
    nondeterministic=True,
)


