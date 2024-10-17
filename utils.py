import torch
import gymnasium as gym
import numpy as np
from typing import Callable
from collections import deque



class HistoryWrapper(gym.Wrapper):
    def __init__(self, env, history_length=1):
        super().__init__(env)
        self.history_length = history_length
        self.history = deque(maxlen=history_length)
        self.observation_space = gym.spaces.Box(
            low=np.repeat(env.observation_space.low[np.newaxis, ...], history_length, axis=0),
            high=np.repeat(env.observation_space.high[np.newaxis, ...], history_length, axis=0),
            dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.history.clear()
        self.history.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.history.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        obs = np.array(self.history)
        if len(self.history) < self.history_length:
            obs = np.pad(obs, ((0, self.history_length - len(self.history)), (0, 0)))
        return obs
