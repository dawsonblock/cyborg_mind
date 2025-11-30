"""Standard Replay Buffer for off-policy algorithms."""

import numpy as np
import torch
from typing import Dict, Tuple, Optional


class ReplayBuffer:
    """
    Experience Replay Buffer.
    
    Stores transitions (obs, action, reward, next_obs, done) for off-policy training.
    """

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        """
        Initialize replay buffer.

        Args:
            buffer_size: Max number of transitions.
            obs_dim: Observation dimension.
            action_dim: Action dimension.
            device: Torch device.
        """
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        self.obs_buf = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.action_buf = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.reward_buf = np.zeros(buffer_size, dtype=np.float32)
        self.done_buf = np.zeros(buffer_size, dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition."""
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.done_buf[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions."""
        idxs = np.random.randint(0, self.size, size=batch_size)

        return {
            "obs": torch.as_tensor(self.obs_buf[idxs], device=self.device),
            "next_obs": torch.as_tensor(self.next_obs_buf[idxs], device=self.device),
            "actions": torch.as_tensor(self.action_buf[idxs], device=self.device),
            "rewards": torch.as_tensor(self.reward_buf[idxs], device=self.device),
            "dones": torch.as_tensor(self.done_buf[idxs], device=self.device),
        }

    def __len__(self) -> int:
        return self.size
