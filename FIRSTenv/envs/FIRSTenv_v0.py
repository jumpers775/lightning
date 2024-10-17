import gymnasium as gym
import numpy as np

class FIRSTenv(gym.Wrapper):
    def __init__(self, size: int = 5):
        self.size = size

        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # forward left, forward right, back left, back right, arm, grabber, elevator 1, elevator 2, bucket
        self.action_space = gym.spaces.Box(low=np.array([-1., -1., -1., -1., -1., -1., -1., -1., -1.]), high=np.array([1., 1., 1., 1., 1., 1., 1., 1., 1.]), dtype=np.float32)
