#!/usr/bin/env python3
"""Tests for PMM read/write operations."""

import pytest
import torch

from cyborg_rl.memory.memory_v7 import PMMV7, create_memory


class TestPMMV7:
    """Test PMM V7 functionality."""

    @pytest.fixture
    def pmm(self):
        # memory_dim=64, num_heads=4 -> head_dim=16 (valid)
        return PMMV7(memory_dim=64, num_slots=8, num_heads=4)

    def test_initial_state(self, pmm):
        """Test initial state creation."""
        state = pmm.get_initial_state(4, torch.device("cpu"))
        
        assert "memory" in state
        assert "usage" in state
        assert "age" in state
        assert state["memory"].shape == (4, 8, 64)
        assert state["usage"].shape == (4, 8)

    def test_forward_step(self, pmm):
        """Test single-step forward."""
        state = pmm.get_initial_state(4, torch.device("cpu"))
        hidden = torch.randn(4, 64)
        
        read, new_state, logs = pmm.forward_step(hidden, state)
        
        assert read.shape == (4, 64)
        assert "write_strength" in logs
        assert "overwrite_conflict" in logs
        assert "gate_usage_low" in logs

    def test_write_updates_memory(self, pmm):
        """Test that writes update memory content."""
        state = pmm.get_initial_state(2, torch.device("cpu"))
        hidden = torch.ones(2, 64)
        
        # Initial memory is zero
        assert state["memory"].abs().sum() == 0
        
        # After write
        _, new_state, _ = pmm.forward_step(hidden, state)
        
        # Memory should have content
        assert new_state["memory"].abs().sum() > 0

    def test_read_retrieves_memory(self, pmm):
        """Test that reads retrieve stored content."""
        state = pmm.get_initial_state(2, torch.device("cpu"))
        
        # Write something
        write_content = torch.randn(2, 64)
        _, state, _ = pmm.forward_step(write_content, state)
        
        # Read it back
        query = write_content  # Same query
        read, _, _ = pmm.forward_step(query, state)
        
        # Read should be non-zero
        assert read.abs().sum() > 0

    def test_forward_sequence(self, pmm):
        """Test sequence processing."""
        state = pmm.get_initial_state(2, torch.device("cpu"))
        hidden_seq = torch.randn(2, 10, 64)
        masks = torch.ones(2, 10)
        
        read_seq, final_state, logs = pmm.forward_sequence(hidden_seq, state, masks)
        
        assert read_seq.shape == (2, 10, 64)
        assert "avg_write_strength" in logs


class TestMemoryFactory:
    """Test memory factory function."""

    @pytest.mark.parametrize("memory_type", ["pmm", "slot", "kv", "ring"])
    def test_create_all_types(self, memory_type):
        """Test creating all memory types."""
        memory = create_memory(
            memory_type=memory_type,
            memory_dim=64,
            num_slots=8,
        )
        
        state = memory.get_initial_state(2, torch.device("cpu"))
        hidden = torch.randn(2, 64)
        
        read, new_state, logs = memory.forward_step(hidden, state)
        assert read.shape == (2, 64)
