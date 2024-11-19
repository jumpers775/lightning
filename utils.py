import torch
import gymnasium as gym
import numpy as np
import random
from typing import Callable
from collections import deque

class HistoryWrapper(gym.Wrapper):
    def __init__(self, env, history_length=1):
        super().__init__(env)
        self.history_length = history_length
        # Use a deque to store the observation history up to the specified length
        self.history = deque(maxlen=history_length)
        # Expand the observation space to accommodate a sequence of observations for the history
        self.observation_space = gym.spaces.Box(
            low=np.repeat(env.observation_space.low[np.newaxis, ...], history_length, axis=0),
            high=np.repeat(env.observation_space.high[np.newaxis, ...], history_length, axis=0),
            dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Clear history upon environment reset and append the initial observation
        self.history.clear()
        self.history.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Append the new observation to history after taking a step
        self.history.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        # Convert history to numpy array, pad if history is shorter than required length
        obs = np.array(self.history)
        if len(self.history) < self.history_length:
            obs = np.pad(obs, ((0, self.history_length - len(self.history)), (0, 0)))
        return obs

class CombinedObservationEnv(gym.Wrapper):
    # Wrapper to combine multiple observation types from the environment into one observation
    def __init__(self, env_id, ):
        self.obs_types = ["ram", "rgb"]
        # Create separate environments for each observation type to collect different modalities
        envs = [gym.make(env_id, obs_type=obs_type, render_mode=None)
                for obs_type in self.obs_types]

        super().__init__(envs[0])
        self.envs = envs

        self.observation_space = gym.spaces.Dict({
            obs_type: env.observation_space
            for obs_type, env in zip(self.obs_types, self.envs)
        })

    def reset(self, seed=None, options=None):
        observations = []
        infos = []

        # Use the same seed for all environments to ensure consistency across observations
        seed = seed or random.randint(0, 2**32 - 1)

        # Collect observations and infos from all environments to combine them
        for env in self.envs:
            obs, info = env.reset(seed=seed, options=options)
            observations.append(obs)
            infos.append(info)

        combined_obs = dict(zip(self.obs_types, observations))
        combined_info = {k: v for d in infos for k, v in d.items()}
        return combined_obs, combined_info

    def step(self, action):
        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []

        # Collect observations and other outputs from each environment
        for env in self.envs:
            obs, reward, terminated, truncated, info = env.step(action)
            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)

        # Ensure that all environments return the same reward and termination flags
        assert all(r == rewards[0] for r in rewards), "Rewards from environments don't match"
        assert all(t == terminateds[0] for t in terminateds), "Terminated flags from environments don't match"
        assert all(t == truncateds[0] for t in truncateds), "Truncated flags from environments don't match"

        combined_obs = dict(zip(self.obs_types, observations))
        combined_info = {k: v for d in infos for k, v in d.items()}
        return combined_obs, rewards[0], terminateds[0], truncateds[0], combined_info

    def render(self):
        # Render using the 'rgb' observation environment if available
        return self.envs[self.obs_types.index("rgb")].render() if "rgb" in self.obs_types else None

    def close(self):
        # Close all the environments when done
        for env in self.envs:
            env.close()
