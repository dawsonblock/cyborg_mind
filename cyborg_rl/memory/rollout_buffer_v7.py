#!/usr/bin/env python3
"""
RolloutBufferV7 - Batched Rollout Buffer with Burn-In

Features:
- Global batched buffer: [T, N, ...] layout
- Stores: obs, actions, values, log_probs, dones, hidden, memory
- sample_sequences(seq_len, burn_in) with gradient masking
- Efficient sequence sampling for recurrent PPO
"""

from typing import Any, Dict, Generator, List, Optional, Tuple
import numpy as np
import torch


class RolloutBufferV7:
    """
    Batched rollout buffer for V7 training system.

    Stores transitions in [T, N, ...] format where T=horizon, N=num_envs.
    Supports sequence sampling with burn-in for honest recurrent PPO.
    """

    def __init__(
        self,
        horizon: int,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        memory_slots: int,
        memory_dim: int,
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        is_discrete: bool = True,
    ) -> None:
        """
        Args:
            horizon: Rollout length (T)
            num_envs: Number of parallel environments (N)
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dim: Encoder hidden state dimension
            memory_slots: Number of memory slots
            memory_dim: Memory vector dimension
            device: Torch device
            gamma: Discount factor
            gae_lambda: GAE lambda
            is_discrete: Whether actions are discrete
        """
        self.horizon = horizon
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.memory_slots = memory_slots
        self.memory_dim = memory_dim
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.is_discrete = is_discrete

        # Allocate storage
        self._allocate()

        # Pointer
        self.ptr = 0
        self.full = False

    def _allocate(self) -> None:
        """Allocate buffer storage."""
        T, N = self.horizon, self.num_envs

        # Core
        self.obs = torch.zeros(T, N, self.obs_dim, device=self.device)
        if self.is_discrete:
            self.actions = torch.zeros(T, N, dtype=torch.long, device=self.device)
        else:
            self.actions = torch.zeros(T, N, self.action_dim, device=self.device)
        self.rewards = torch.zeros(T, N, device=self.device)
        self.values = torch.zeros(T, N, device=self.device)
        self.log_probs = torch.zeros(T, N, device=self.device)
        self.dones = torch.zeros(T, N, device=self.device)

        # Recurrent states
        self.hidden = torch.zeros(T, N, self.hidden_dim, device=self.device)
        self.memory = torch.zeros(T, N, self.memory_slots, self.memory_dim, device=self.device)

        # Computed
        self.advantages = torch.zeros(T, N, device=self.device)
        self.returns = torch.zeros(T, N, device=self.device)

    def reset(self) -> None:
        """Reset buffer for new rollout."""
        self.ptr = 0
        self.full = False

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        done: torch.Tensor,
        hidden: torch.Tensor,
        memory: torch.Tensor,
    ) -> None:
        """
        Add a transition to the buffer.

        Args:
            obs: (N, obs_dim)
            action: (N,) or (N, action_dim)
            reward: (N,)
            value: (N,)
            log_prob: (N,)
            done: (N,)
            hidden: (N, hidden_dim)
            memory: (N, num_slots, memory_dim)
        """
        if self.ptr >= self.horizon:
            raise RuntimeError("Buffer is full")

        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        self.hidden[self.ptr] = hidden
        self.memory[self.ptr] = memory

        self.ptr += 1
        if self.ptr == self.horizon:
            self.full = True

    def compute_gae(
        self,
        last_value: torch.Tensor,
        last_done: torch.Tensor,
    ) -> None:
        """
        Compute GAE advantages and returns.

        Args:
            last_value: (N,) bootstrap value
            last_done: (N,) whether last state is terminal
        """
        gae = 0.0
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_value = last_value
                next_done = last_done
            else:
                next_value = self.values[t + 1]
                next_done = self.dones[t + 1]

            delta = (
                self.rewards[t]
                + self.gamma * next_value * (1 - next_done)
                - self.values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * (1 - next_done) * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

    def normalize_advantages(self) -> None:
        """Normalize advantages to zero mean, unit variance."""
        adv = self.advantages[:self.ptr]
        self.advantages[:self.ptr] = (adv - adv.mean()) / (adv.std() + 1e-8)

    def sample_sequences(
        self,
        seq_len: int,
        burn_in: int = 0,
        batch_size: int = 16,
        drop_last: bool = True,
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Sample sequences with burn-in for recurrent training.

        Args:
            seq_len: Sequence length for training
            burn_in: Burn-in length (gradients masked)
            batch_size: Number of sequences per batch
            drop_last: Drop last incomplete batch

        Yields:
            Batched dict with:
                - obs: (B, burn_in + seq_len, obs_dim)
                - actions: (B, burn_in + seq_len)
                - values: (B, seq_len) - post-burn-in only
                - log_probs: (B, seq_len)
                - advantages: (B, seq_len)
                - returns: (B, seq_len)
                - dones: (B, burn_in + seq_len)
                - hidden_init: (B, hidden_dim) - initial hidden state
                - memory_init: (B, num_slots, memory_dim) - initial memory
                - grad_mask: (B, burn_in + seq_len) - 0 for burn-in, 1 for training
        """
        total_len = burn_in + seq_len
        num_valid_starts = max(0, self.ptr - total_len + 1)

        if num_valid_starts == 0:
            return

        # Generate all valid (env, start) pairs
        indices = []
        for env in range(self.num_envs):
            for start in range(num_valid_starts):
                indices.append((env, start))

        # Shuffle
        np.random.shuffle(indices)

        # Batch
        num_batches = len(indices) // batch_size
        if not drop_last and len(indices) % batch_size > 0:
            num_batches += 1

        for b in range(num_batches):
            batch_indices = indices[b * batch_size : (b + 1) * batch_size]
            if len(batch_indices) == 0:
                continue

            B = len(batch_indices)

            # Preallocate
            batch = {
                "obs": torch.zeros(B, total_len, self.obs_dim, device=self.device),
                "actions": torch.zeros(B, total_len, dtype=self.actions.dtype, device=self.device),
                "values": torch.zeros(B, seq_len, device=self.device),
                "log_probs": torch.zeros(B, seq_len, device=self.device),
                "advantages": torch.zeros(B, seq_len, device=self.device),
                "returns": torch.zeros(B, seq_len, device=self.device),
                "dones": torch.zeros(B, total_len, device=self.device),
                "hidden_init": torch.zeros(B, self.hidden_dim, device=self.device),
                "memory_init": torch.zeros(B, self.memory_slots, self.memory_dim, device=self.device),
                "grad_mask": torch.zeros(B, total_len, device=self.device),
            }

            # Fill batch
            for i, (env, start) in enumerate(batch_indices):
                end = start + total_len

                batch["obs"][i] = self.obs[start:end, env]
                batch["actions"][i] = self.actions[start:end, env]
                batch["dones"][i] = self.dones[start:end, env]

                # Post-burn-in data
                batch["values"][i] = self.values[start + burn_in : end, env]
                batch["log_probs"][i] = self.log_probs[start + burn_in : end, env]
                batch["advantages"][i] = self.advantages[start + burn_in : end, env]
                batch["returns"][i] = self.returns[start + burn_in : end, env]

                # Initial states (from before sequence)
                if start > 0:
                    batch["hidden_init"][i] = self.hidden[start - 1, env]
                    batch["memory_init"][i] = self.memory[start - 1, env]

                # Gradient mask: 0 for burn-in, 1 for training
                batch["grad_mask"][i, burn_in:] = 1.0

            yield batch

    def get_all_flat(self) -> Dict[str, torch.Tensor]:
        """Get all data flattened for non-recurrent training."""
        T = self.ptr
        N = self.num_envs

        return {
            "obs": self.obs[:T].reshape(T * N, -1),
            "actions": self.actions[:T].reshape(T * N, -1) if not self.is_discrete else self.actions[:T].reshape(T * N),
            "values": self.values[:T].reshape(T * N),
            "log_probs": self.log_probs[:T].reshape(T * N),
            "advantages": self.advantages[:T].reshape(T * N),
            "returns": self.returns[:T].reshape(T * N),
        }

    def __len__(self) -> int:
        return self.ptr * self.num_envs

    def __repr__(self) -> str:
        return (
            f"RolloutBufferV7(horizon={self.horizon}, num_envs={self.num_envs}, "
            f"ptr={self.ptr}, full={self.full})"
        )
