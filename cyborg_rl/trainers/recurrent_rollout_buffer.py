#!/usr/bin/env python3
"""
recurrent_rollout_buffer.py

Rollout buffer that stores recurrent states per step for honest RNN/PMM PPO.

Intended for use with:
    train.recurrent_mode = "burn_in"

For each time step, we store:
    - obs, actions, rewards, dones, values, log_probs
    - recurrent_state  (Python object; e.g., dict with hidden, memory, etc.)
"""

from typing import Any, Dict, List, Optional, Generator
import numpy as np
import torch


class RecurrentRolloutBuffer:
    """
    Simple, index-based rollout buffer with per-step recurrent state.

    This is deliberately similar to the existing RolloutBuffer, but:
        - It keeps a list of recurrent states (Python objects).
        - It supports GAE and standard PPO-style batching.
        - State storage allows for proper recurrent PPO updates.

    Shapes:
        obs:          [buffer_size, obs_dim]
        actions:      [buffer_size, ...] (discrete: [buffer_size], continuous: [buffer_size, action_dim])
        rewards:      [buffer_size]
        dones:        [buffer_size]
        values:       [buffer_size]
        log_probs:    [buffer_size]
        advantages:   [buffer_size]
        returns:      [buffer_size]
        states:       list of length buffer_size; each element is the recurrent state
                      *before* taking the stored action.
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
        Initialize the recurrent rollout buffer.

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

        # Python-level state storage (list of arbitrary recurrent states)
        self.states: List[Optional[Dict[str, Any]]] = [None for _ in range(buffer_size)]

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
        recurrent_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store a single time step.

        Args:
            obs: Observation (numpy array).
            action: Action taken.
            reward: Scalar reward.
            value: Value estimate.
            log_prob: Log probability of action.
            done: Whether episode ended.
            recurrent_state: dict or other Python object containing the RNN/PMM state
                             BEFORE this action was taken.
        """
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = float(done)

        # Store state as Python object (no grad)
        self.states[self.ptr] = recurrent_state

        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True

    def compute_returns_and_advantages(
        self,
        last_value: float,
        last_done: bool,
    ) -> None:
        """
        Standard GAE computation in flat buffer order.

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
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Yield mini-batches of indices and tensors.

        The caller can access recurrent states via the yielded batch dict
        which includes 'batch_indices' for lookup in self.states.

        Yields:
            Dict with:
                - observations, actions, old_log_probs, returns, advantages, old_values: torch.Tensor
                - batch_indices: list of int (for looking up states)
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
                "batch_indices": batch_indices.tolist(),  # For state lookup
            }

    def get_states(self, indices: List[int]) -> List[Dict[str, Any]]:
        """
        Retrieve stored recurrent states for given indices.

        Args:
            indices: List of buffer indices.

        Returns:
            List of recurrent state dicts.
        """
        return [self.states[i] for i in indices]

    def reset(self) -> None:
        """Reset the buffer."""
        self.ptr = 0
        self.full = False
        # Clear states
        self.states = [None for _ in range(self.buffer_size)]

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.ptr
