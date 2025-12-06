#!/usr/bin/env python3
"""
recurrent_rollout_buffer.py

Rollout buffer storage for Recurrent PPO with honest memory (PMM).
"""

from typing import Any, Dict, List, Optional, Generator
import numpy as np
import torch


class RecurrentRolloutBuffer:
    """
    Rollout buffer that stores per-step recurrent states.
    
    Supports both Truncated BPTT (via chunks) and Full Sequence BPTT.
    States are stored as Python objects (Any) to support arbitrary Encoder states
    (e.g., GRU hidden, Mamba inference_params, PMM memory, or tuples of these).
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

        # Store recurrent states (Any type: Tensor, Dict, Tuple, etc.)
        self.states: List[Optional[Any]] = [None for _ in range(buffer_size)]

        self.ptr = 0
        self.full = False

    
    def reset(self) -> None:
        """Reset the buffer pointers."""
        self.ptr = 0
        self.full = False
        self.states = [None for _ in range(self.buffer_size)]

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        recurrent_state: Optional[Any] = None,
    ) -> None:
        if self.ptr >= self.buffer_size:
            # Buffer overflow safety (should be handled by caller)
            return

        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = float(done)
        self.states[self.ptr] = recurrent_state

        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True

    def compute_returns_and_advantages(
        self,
        last_value: float,
        last_done: bool,
    ) -> None:
        last_gae = 0.0
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]

    def get_sampler(
        self,
        batch_size: int,
        seq_len: int,
        burn_in: int = 0,
        normalize_advantage: bool = True,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Yields batches of sequences for Recurrent PPO.
        
        If burn_in > 0, fetches (seq_len + burn_in) steps, but only computes loss on the last seq_len steps.
        """
        # Valid starting indices:
        # We need a sequence of length (burn_in + seq_len)
        # So index i such that i + burn_in + seq_len <= ptr
        # AND NO DONE flags inside the sequence (usually we handle masking inside the loop, 
        # but simpler is to sample valid segments).
        # Actually standard PPO just masks the loss if done occurs, but RNN state resetting is tricky.
        # "Honest Memory" usually implies we handle done by resetting state in the forward pass.
        # So we can sample ANY segment. 
        # But we must be careful if the segment crosses episode boundaries.
        # Our PMM implementation takes a mask.
        
        total_len = burn_in + seq_len
        valid_indices = np.arange(self.ptr - total_len + 1)
        np.random.shuffle(valid_indices)
        
        # Advantages normalization
        advantages = self.advantages[:self.ptr].copy()
        if normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Batching
        for start_i in range(0, len(valid_indices), batch_size):
            batch_idxs = valid_indices[start_i : start_i + batch_size] # Starting indices of sequences
            
            # Construct batch tensors: (B, T, ...)
            # We want [Batch, Time, Dim]
            
            b_obs, b_actions, b_logprobs, b_adv, b_ret, b_val, b_dones = [], [], [], [], [], [], []
            b_states = []
            
            for start_idx in batch_idxs:
                end_idx = start_idx + total_len
                b_obs.append(self.observations[start_idx:end_idx])
                b_actions.append(self.actions[start_idx:end_idx])
                b_logprobs.append(self.log_probs[start_idx:end_idx])
                b_adv.append(advantages[start_idx:end_idx])
                b_ret.append(self.returns[start_idx:end_idx])
                b_val.append(self.values[start_idx:end_idx])
                b_dones.append(self.dones[start_idx:end_idx])
                
                # Initial state for this sequence is the stored state at start_idx
                b_states.append(self.states[start_idx])

            # Convert to tensors
            # Obs: (B, T, D)
            t_obs = torch.from_numpy(np.stack(b_obs)).to(self.device).float()
            
            if self.is_discrete:
                t_actions = torch.from_numpy(np.stack(b_actions)).to(self.device).long()
            else:
                t_actions = torch.from_numpy(np.stack(b_actions)).to(self.device).float()
                
            t_logprobs = torch.from_numpy(np.stack(b_logprobs)).to(self.device).float()
            t_adv = torch.from_numpy(np.stack(b_adv)).to(self.device).float()
            t_ret = torch.from_numpy(np.stack(b_ret)).to(self.device).float()
            t_val = torch.from_numpy(np.stack(b_val)).to(self.device).float()
            t_dones = torch.from_numpy(np.stack(b_dones)).to(self.device).float()
            
            yield {
                "obs": t_obs,
                "actions": t_actions,
                "log_probs": t_logprobs,
                "advantages": t_adv,
                "returns": t_ret,
                "values": t_val,
                "dones": t_dones, # Terminations within the sequence
                "states": b_states # List of initial states for each batch element
            }
