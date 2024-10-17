from gymnasium.envs.registration import register

register(
     id="FIRSTenv/FIRSTenv_v0",
     entry_point="FIRSTenv.envs.FIRSTenv_v0:FIRSTenv",
     max_episode_steps=500,
)
