"""Rollout buffer for on-policy algorithms like PPO."""

from typing import Generator, Dict, Optional
import numpy as np
import torch


class RolloutBuffer:
    """
    Rollout buffer for storing trajectories and computing advantages.

    Implements Generalized Advantage Estimation (GAE) for variance reduction.
    """

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        is_discrete: bool = True,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """
        Initialize the rollout buffer.

        Args:
            buffer_size: Number of steps to store.
            obs_dim: Observation dimension.
            action_dim: Action dimension.
            device: Torch device.
            is_discrete: Whether actions are discrete.
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter.
        """
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.is_discrete = is_discrete
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        
        if is_discrete:
            self.actions = np.zeros((buffer_size,), dtype=np.int64)
        else:
            self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)

        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.values = np.zeros((buffer_size,), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)

        self.advantages = np.zeros((buffer_size,), dtype=np.float32)
        self.returns = np.zeros((buffer_size,), dtype=np.float32)

        self.ptr = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        """
        Add a transition to the buffer.

        Args:
            obs: Observation.
            action: Action taken.
            reward: Reward received.
            value: Estimated value.
            log_prob: Log probability of action.
            done: Whether episode ended.
        """
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = float(done)

        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True

    def compute_returns_and_advantages(
        self,
        last_value: float,
        last_done: bool,
    ) -> None:
        """
        Compute returns and advantages using GAE.

        Args:
            last_value: Value estimate for the last state.
            last_done: Whether last state is terminal.
        """
        last_gae = 0.0
        
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]

    def get(
        self,
        batch_size: int,
        normalize_advantage: bool = True,
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Generate batches for training.

        Args:
            batch_size: Batch size.
            normalize_advantage: Whether to normalize advantages.

        Yields:
            Dict[str, torch.Tensor]: Batch of data.
        """
        indices = np.random.permutation(self.ptr)

        advantages = self.advantages[:self.ptr].copy()
        if normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        action_dtype = torch.long if self.is_discrete else torch.float32

        for start in range(0, self.ptr, batch_size):
            end = min(start + batch_size, self.ptr)
            batch_indices = indices[start:end]

            yield {
                "observations": torch.from_numpy(
                    self.observations[batch_indices]
                ).to(self.device),
                "actions": torch.from_numpy(
                    self.actions[batch_indices]
                ).to(self.device, action_dtype),
                "old_log_probs": torch.from_numpy(
                    self.log_probs[batch_indices]
                ).to(self.device),
                "advantages": torch.from_numpy(
                    advantages[batch_indices]
                ).to(self.device),
                "returns": torch.from_numpy(
                    self.returns[batch_indices]
                ).to(self.device),
                "old_values": torch.from_numpy(
                    self.values[batch_indices]
                ).to(self.device),
            }

    def reset(self) -> None:
        """Reset the buffer."""
        self.ptr = 0
        self.full = False

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.ptr
