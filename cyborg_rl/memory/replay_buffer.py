"""Replay buffer for experience storage."""

from typing import Dict, Optional, Tuple
import numpy as np
import torch


class ReplayBuffer:
    """
    Simple replay buffer for storing and sampling transitions.

    Supports both on-policy (PPO) and off-policy (DQN/SAC) algorithms.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        is_discrete: bool = True,
    ) -> None:
        """
        Initialize the replay buffer.

        Args:
            capacity: Maximum number of transitions to store.
            obs_dim: Observation dimension.
            action_dim: Action dimension.
            device: Torch device for tensors.
            is_discrete: Whether actions are discrete.
        """
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.is_discrete = is_discrete

        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)

        if is_discrete:
            self.actions = np.zeros((capacity,), dtype=np.int64)
        else:
            self.actions = np.zeros((capacity, action_dim), dtype=np.float32)

        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

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
        """
        Add a transition to the buffer.

        Args:
            obs: Current observation.
            action: Action taken.
            reward: Reward received.
            next_obs: Next observation.
            done: Whether episode ended.
        """
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dict[str, torch.Tensor]: Batch of transitions.
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        action_dtype = torch.long if self.is_discrete else torch.float32

        return {
            "observations": torch.from_numpy(self.observations[indices]).to(self.device),
            "actions": torch.from_numpy(self.actions[indices]).to(self.device, action_dtype),
            "rewards": torch.from_numpy(self.rewards[indices]).to(self.device),
            "next_observations": torch.from_numpy(self.next_observations[indices]).to(self.device),
            "dones": torch.from_numpy(self.dones[indices]).to(self.device),
        }

    def clear(self) -> None:
        """Clear the buffer."""
        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size

    def save(self, path: str) -> None:
        """
        Save buffer to disk.

        Args:
            path: Save path.
        """
        np.savez(
            path,
            observations=self.observations[:self.size],
            actions=self.actions[:self.size],
            rewards=self.rewards[:self.size],
            next_observations=self.next_observations[:self.size],
            dones=self.dones[:self.size],
        )

    def load(self, path: str) -> None:
        """
        Load buffer from disk.

        Args:
            path: Load path.
        """
        data = np.load(path)
        n = len(data["observations"])

        self.observations[:n] = data["observations"]
        self.actions[:n] = data["actions"]
        self.rewards[:n] = data["rewards"]
        self.next_observations[:n] = data["next_observations"]
        self.dones[:n] = data["dones"]

        self.size = n
        self.ptr = n % self.capacity
