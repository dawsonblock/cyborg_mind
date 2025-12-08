#!/usr/bin/env python3
"""Tests for Mamba/PseudoMamba state caching."""

import pytest
import torch

from cyborg_rl.models.encoder_v7 import EncoderV7, PseudoMambaBlock


class TestMambaStateCaching:
    """Test Mamba state caching behavior."""

    def test_pseudo_mamba_state_init(self):
        """Test PseudoMamba initializes state correctly."""
        block = PseudoMambaBlock(d_model=32, d_state=16)
        x = torch.randn(2, 4, 32)  # (B, L, D)
        
        # First call - no state
        out, state = block(x, None)
        
        assert "h" in state
        # GRU hidden state: (num_layers=1, B, d_inner)
        assert state["h"].shape == (1, 2, 64)  # d_inner = d_model * expand = 32 * 2

    def test_pseudo_mamba_state_persistence(self):
        """Test state persists correctly across calls."""
        block = PseudoMambaBlock(d_model=32, d_state=16)
        
        # First sequence
        x1 = torch.randn(2, 4, 32)
        out1, state1 = block(x1, None)
        
        # Second sequence with carried state
        x2 = torch.randn(2, 4, 32)
        out2, state2 = block(x2, state1)
        
        # State should have changed
        assert not torch.allclose(state1["h"], state2["h"])

    def test_encoder_mamba_state_reset(self):
        """Test encoder resets mamba states on done."""
        encoder = EncoderV7(
            mode="mamba",
            input_dim=32,
            hidden_dim=16,
            latent_dim=8,
            num_layers=2,
            use_cuda_mamba=False,
        )
        
        x = torch.randn(2, 4, 32)
        state = encoder.get_initial_state(2, torch.device("cpu"))
        
        # Forward pass
        out, new_state = encoder(x, state)
        
        # Partial reset (env 0 done, env 1 continue)
        mask = torch.tensor([0.0, 1.0])
        reset_state = encoder.reset_states(new_state, mask)
        
        # Check mamba states
        for layer_state in reset_state["mamba"]:
            if "h" in layer_state:
                h = layer_state["h"]
                # Env 0 should be zeroed, env 1 preserved
                assert h[:, 0, :].abs().sum() == 0  # Env 0 reset
                assert h[:, 1, :].abs().sum() > 0   # Env 1 preserved

    def test_encoder_state_detach(self):
        """Test state detachment for TBPTT."""
        encoder = EncoderV7(
            mode="mamba",
            input_dim=32,
            hidden_dim=16,
            latent_dim=8,
            use_cuda_mamba=False,
        )
        
        x = torch.randn(2, 4, 32)
        state = encoder.get_initial_state(2, torch.device("cpu"))
        out, new_state = encoder(x, state)
        
        # Detach
        detached = encoder.detach_states(new_state)
        
        for layer_state in detached["mamba"]:
            if "h" in layer_state:
                assert not layer_state["h"].requires_grad

    @pytest.mark.parametrize("mode", ["gru", "mamba", "hybrid", "fusion"])
    def test_all_modes_state_consistency(self, mode):
        """Test state handling consistency across all modes."""
        encoder = EncoderV7(
            mode=mode,
            input_dim=32,
            hidden_dim=16,
            latent_dim=8,
            use_cuda_mamba=False,
        )
        
        B, L = 2, 4
        x = torch.randn(B, L, 32)
        
        # Initial state
        state0 = encoder.get_initial_state(B, torch.device("cpu"))
        
        # Forward
        out1, state1 = encoder(x, state0)
        
        # Reset partial
        mask = torch.tensor([0.0, 1.0])
        state_reset = encoder.reset_states(state1, mask)
        
        # Forward again
        out2, state2 = encoder(x, state_reset)
        
        # Outputs should differ due to state reset
        assert not torch.allclose(out1, out2)
