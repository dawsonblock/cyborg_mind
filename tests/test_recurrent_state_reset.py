#!/usr/bin/env python3
"""Tests for V7 recurrent state management."""

import pytest
import torch

from cyborg_rl.models.encoder_v7 import EncoderV7


class TestEncoderV7StateReset:
    """Test encoder state reset on done episodes."""

    @pytest.fixture
    def encoder(self):
        return EncoderV7(
            mode="gru",
            input_dim=64,
            hidden_dim=32,
            latent_dim=16,
            num_layers=1,
        )

    def test_get_initial_state(self, encoder):
        """Test initial state creation."""
        state = encoder.get_initial_state(batch_size=4, device=torch.device("cpu"))
        assert "gru" in state
        assert state["gru"].shape == (1, 4, 32)
        assert state["gru"].sum() == 0

    def test_reset_states_partial(self, encoder):
        """Test partial state reset (some envs done)."""
        state = encoder.get_initial_state(4, torch.device("cpu"))
        
        # Fill with non-zero values
        state["gru"] = torch.ones_like(state["gru"])
        
        # Reset env 0 and 2
        mask = torch.tensor([0.0, 1.0, 0.0, 1.0])
        new_state = encoder.reset_states(state, mask)
        
        # Check env 0 and 2 are zeroed
        assert new_state["gru"][:, 0, :].abs().sum() == 0
        assert new_state["gru"][:, 2, :].abs().sum() == 0
        
        # Check env 1 and 3 are preserved
        assert new_state["gru"][:, 1, :].sum() > 0
        assert new_state["gru"][:, 3, :].sum() > 0

    def test_detach_states(self, encoder):
        """Test state detachment for TBPTT."""
        state = encoder.get_initial_state(4, torch.device("cpu"))
        state["gru"].requires_grad = True
        
        detached = encoder.detach_states(state)
        assert not detached["gru"].requires_grad


class TestEncoderV7Modes:
    """Test all encoder modes."""

    @pytest.mark.parametrize("mode", ["gru", "mamba", "hybrid", "fusion"])
    def test_forward_all_modes(self, mode):
        """Test forward pass for all encoder modes."""
        encoder = EncoderV7(
            mode=mode,
            input_dim=64,
            hidden_dim=32,
            latent_dim=16,
            num_layers=2,
            use_cuda_mamba=False,  # Use PseudoMamba
        )
        
        x = torch.randn(4, 8, 64)  # (B, L, D)
        state = encoder.get_initial_state(4, torch.device("cpu"))
        
        output, new_state = encoder(x, state)
        
        assert output.shape == (4, 8, 16)

    def test_single_step_input(self):
        """Test encoder with single-step input (B, D)."""
        encoder = EncoderV7(mode="gru", input_dim=64, hidden_dim=32, latent_dim=16)
        
        x = torch.randn(4, 64)  # (B, D)
        state = encoder.get_initial_state(4, torch.device("cpu"))
        
        output, _ = encoder(x, state)
        assert output.shape == (4, 16)
