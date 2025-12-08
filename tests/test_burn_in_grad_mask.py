#!/usr/bin/env python3
"""Tests for burn-in gradient masking in recurrent training."""

import pytest
import torch

from cyborg_rl.memory.rollout_buffer_v7 import RolloutBufferV7


class TestBurnInGradMask:
    """Test gradient masking for burn-in sequences."""

    @pytest.fixture
    def filled_buffer(self):
        """Create a filled buffer for testing."""
        buf = RolloutBufferV7(
            horizon=64,
            num_envs=2,
            obs_dim=16,
            action_dim=4,
            hidden_dim=8,
            memory_slots=4,
            memory_dim=8,
            device=torch.device("cpu"),
        )
        
        for _ in range(64):
            buf.add(
                obs=torch.randn(2, 16),
                action=torch.randint(0, 4, (2,)),
                reward=torch.randn(2),
                value=torch.randn(2),
                log_prob=torch.randn(2),
                done=torch.zeros(2),
                hidden=torch.randn(2, 8),
                memory=torch.randn(2, 4, 8),
            )
        
        buf.compute_gae(torch.zeros(2), torch.zeros(2))
        return buf

    def test_grad_mask_shape(self, filled_buffer):
        """Test grad_mask has correct shape."""
        batches = list(filled_buffer.sample_sequences(
            seq_len=16,
            burn_in=8,
            batch_size=4,
        ))
        
        batch = batches[0]
        # Total length = burn_in + seq_len = 8 + 16 = 24
        assert batch["grad_mask"].shape == (4, 24)

    def test_grad_mask_values(self, filled_buffer):
        """Test grad_mask has correct 0/1 pattern."""
        batches = list(filled_buffer.sample_sequences(
            seq_len=16,
            burn_in=8,
            batch_size=4,
        ))
        
        batch = batches[0]
        burn_in = 8
        
        # First burn_in steps should be 0 (masked)
        assert batch["grad_mask"][:, :burn_in].sum() == 0
        
        # Remaining steps should be 1 (training)
        assert batch["grad_mask"][:, burn_in:].sum() == 4 * 16  # B * seq_len

    def test_values_only_in_training_region(self, filled_buffer):
        """Test that values/advantages are only for training region."""
        seq_len = 16
        burn_in = 8
        
        batches = list(filled_buffer.sample_sequences(
            seq_len=seq_len,
            burn_in=burn_in,
            batch_size=4,
        ))
        
        batch = batches[0]
        
        # Values should have shape (B, seq_len), NOT (B, burn_in + seq_len)
        assert batch["values"].shape == (4, seq_len)
        assert batch["advantages"].shape == (4, seq_len)
        assert batch["returns"].shape == (4, seq_len)

    def test_obs_actions_full_sequence(self, filled_buffer):
        """Test that obs/actions include burn-in region."""
        seq_len = 16
        burn_in = 8
        total_len = burn_in + seq_len
        
        batches = list(filled_buffer.sample_sequences(
            seq_len=seq_len,
            burn_in=burn_in,
            batch_size=4,
        ))
        
        batch = batches[0]
        
        # Obs and actions should include burn-in
        assert batch["obs"].shape == (4, total_len, 16)
        assert batch["actions"].shape == (4, total_len)
