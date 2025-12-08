#!/usr/bin/env python3
"""Tests for RolloutBufferV7 sequence sampling."""

import pytest
import torch

from cyborg_rl.memory.rollout_buffer_v7 import RolloutBufferV7


class TestRolloutBufferV7:
    """Test rollout buffer functionality."""

    @pytest.fixture
    def buffer(self):
        return RolloutBufferV7(
            horizon=64,
            num_envs=4,
            obs_dim=32,
            action_dim=8,
            hidden_dim=16,
            memory_slots=4,
            memory_dim=16,
            device=torch.device("cpu"),
        )

    def test_add_transitions(self, buffer):
        """Test adding transitions to buffer."""
        for i in range(10):
            buffer.add(
                obs=torch.randn(4, 32),
                action=torch.randint(0, 8, (4,)),
                reward=torch.randn(4),
                value=torch.randn(4),
                log_prob=torch.randn(4),
                done=torch.zeros(4),
                hidden=torch.randn(4, 16),
                memory=torch.randn(4, 4, 16),
            )
        
        assert buffer.ptr == 10
        assert len(buffer) == 10 * 4

    def test_compute_gae(self, buffer):
        """Test GAE computation."""
        # Fill buffer
        for i in range(20):
            buffer.add(
                obs=torch.randn(4, 32),
                action=torch.randint(0, 8, (4,)),
                reward=torch.ones(4),
                value=torch.ones(4) * 0.5,
                log_prob=torch.randn(4),
                done=torch.zeros(4),
                hidden=torch.randn(4, 16),
                memory=torch.randn(4, 4, 16),
            )
        
        buffer.compute_gae(
            last_value=torch.ones(4) * 0.5,
            last_done=torch.zeros(4),
        )
        
        # Advantages should be computed
        assert buffer.advantages[:20].abs().sum() > 0
        # Returns = advantages + values
        assert buffer.returns[:20].abs().sum() > 0

    def test_sample_sequences(self, buffer):
        """Test sequence sampling with burn-in."""
        # Fill buffer
        for i in range(64):
            buffer.add(
                obs=torch.randn(4, 32),
                action=torch.randint(0, 8, (4,)),
                reward=torch.ones(4),
                value=torch.randn(4),
                log_prob=torch.randn(4),
                done=torch.zeros(4),
                hidden=torch.randn(4, 16),
                memory=torch.randn(4, 4, 16),
            )
        
        buffer.compute_gae(torch.zeros(4), torch.zeros(4))
        
        batches = list(buffer.sample_sequences(
            seq_len=16,
            burn_in=8,
            batch_size=4,
        ))
        
        assert len(batches) > 0
        
        batch = batches[0]
        assert batch["obs"].shape == (4, 24, 32)  # burn_in + seq_len
        assert batch["values"].shape == (4, 16)  # seq_len only
        assert batch["grad_mask"][:, :8].sum() == 0  # burn-in masked
        assert batch["grad_mask"][:, 8:].sum() > 0  # training region

    def test_reset(self, buffer):
        """Test buffer reset."""
        buffer.add(
            obs=torch.randn(4, 32),
            action=torch.randint(0, 8, (4,)),
            reward=torch.randn(4),
            value=torch.randn(4),
            log_prob=torch.randn(4),
            done=torch.zeros(4),
            hidden=torch.randn(4, 16),
            memory=torch.randn(4, 4, 16),
        )
        
        assert buffer.ptr == 1
        
        buffer.reset()
        
        assert buffer.ptr == 0
        assert not buffer.full
