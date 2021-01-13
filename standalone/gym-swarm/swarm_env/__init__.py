from gym.envs import register

register(
    id='humanswarm-v0',
    entry_point='swarm_env.envs:HumanSwarmEnv',
)