"""Tests for BrainCyborgMind."""

import pytest
import torch

from cyborg_mind_v2.capsule_brain.policy.brain_cyborg_mind import BrainCyborgMind


class TestBrainCyborgMind:
    """Test brain architecture."""

    @pytest.fixture
    def brain(self):
        """Create a brain instance."""
        return BrainCyborgMind(
            scalar_dim=20,
            goal_dim=4,
            thought_dim=32,
            emotion_dim=8,
            workspace_dim=64,
            vision_dim=512,
            emb_dim=256,
            hidden_dim=512,
            mem_dim=128,
            num_actions=20,
            start_slots=64,  # Smaller for testing
        )

    def test_brain_initialization(self, brain):
        """Test brain initializes correctly."""
        assert brain is not None
        assert brain.scalar_dim == 20
        assert brain.goal_dim == 4
        assert brain.thought_dim == 32
        assert brain.emotion_dim == 8
        assert brain.workspace_dim == 64

    def test_brain_forward_pass(self, brain):
        """Test brain forward pass."""
        batch_size = 2
        device = torch.device("cpu")

        brain = brain.to(device)
        brain.eval()

        # Create inputs
        pixels = torch.randn(batch_size, 3, 128, 128)
        scalars = torch.randn(batch_size, 20)
        goal = torch.randn(batch_size, 4)
        thought = torch.zeros(batch_size, 32)
        emotion = torch.zeros(batch_size, 8)
        workspace = torch.zeros(batch_size, 64)

        # Forward pass
        with torch.no_grad():
            output = brain(
                pixels=pixels,
                scalars=scalars,
                goal=goal,
                thought=thought,
                emotion=emotion,
                workspace=workspace,
            )

        # Check outputs
        assert "action_logits" in output
        assert "value" in output
        assert "thought" in output
        assert "emotion" in output
        assert "workspace" in output
        assert "hidden_h" in output
        assert "hidden_c" in output
        assert "pressure" in output

        # Check shapes
        assert output["action_logits"].shape == (batch_size, 20)
        assert output["value"].shape == (batch_size, 1)
        assert output["thought"].shape == (batch_size, 32)
        assert output["emotion"].shape == (batch_size, 8)
        assert output["workspace"].shape == (batch_size, 64)

    def test_brain_memory_expansion(self, brain):
        """Test memory can expand."""
        initial_slots = brain.pmm.mem_slots

        # Manually trigger expansion
        success = brain.pmm.expand(factor=2)

        assert success
        assert brain.pmm.mem_slots == initial_slots * 2

    def test_brain_thought_clipping(self, brain):
        """Test thought vector is clipped."""
        batch_size = 1
        device = torch.device("cpu")

        brain = brain.to(device)
        brain.train()  # Training mode for anchoring

        # Create inputs with extreme thought values
        pixels = torch.randn(batch_size, 3, 128, 128)
        scalars = torch.randn(batch_size, 20)
        goal = torch.randn(batch_size, 4)
        thought = torch.tensor([[100.0] * 32])  # Extreme values
        emotion = torch.zeros(batch_size, 8)
        workspace = torch.zeros(batch_size, 64)

        # Forward pass
        with torch.no_grad():
            output = brain(
                pixels=pixels,
                scalars=scalars,
                goal=goal,
                thought=thought,
                emotion=emotion,
                workspace=workspace,
            )

        # Check thought is clipped
        assert output["thought"].abs().max() <= 3.0

    def test_brain_parameter_count(self, brain):
        """Test brain has expected parameter count."""
        total_params = sum(p.numel() for p in brain.parameters())

        # Should be around 2-3M parameters
        assert 1_000_000 < total_params < 5_000_000

    def test_brain_memory_write(self, brain):
        """Test memory write operation."""
        batch_size = 2

        brain.eval()

        # Create inputs
        pixels = torch.randn(batch_size, 3, 128, 128)
        scalars = torch.randn(batch_size, 20)
        goal = torch.randn(batch_size, 4)
        thought = torch.zeros(batch_size, 32)

        # Forward pass (triggers memory write)
        with torch.no_grad():
            output = brain(
                pixels=pixels,
                scalars=scalars,
                goal=goal,
                thought=thought,
            )

        # Check memory write vector exists
        assert "mem_write" in output
        assert output["mem_write"].shape == (batch_size, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
