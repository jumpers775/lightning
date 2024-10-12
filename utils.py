import torch
import gymnasium as gym
import numpy as np
from typing import Callable
from stable_baselines3.common.policies import ActorCriticPolicy
from collections import deque
from model import Lightning


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        self.action_dims = action_space.n
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )



    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = Lightning(self.features_dim//500, last_layer_dim_pi=self.action_dims, last_layer_dim_vf=1)

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
