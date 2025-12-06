"""Unit tests for Predictive Memory Module."""

import pytest
import torch

from cyborg_rl.memory.pmm import PredictiveMemoryModule


class TestPMM:
    """Tests for PredictiveMemoryModule."""

    @pytest.fixture
    def device(self) -> torch.device:
        """Get test device."""
        return torch.device("cpu")

    @pytest.fixture
    def pmm(self, device: torch.device) -> PredictiveMemoryModule:
        """Create PMM instance."""
        return PredictiveMemoryModule(
            input_dim=64,
            memory_size=32,
            memory_dim=16,
            num_read_heads=2,
            num_write_heads=1,
            sharp_factor=1.0,
        ).to(device)

    def test_initialization(self, pmm: PredictiveMemoryModule) -> None:
        """Test PMM initialization."""
        assert pmm.input_dim == 64
        assert pmm.memory_size == 32
        assert pmm.memory_dim == 16
        assert pmm.num_read_heads == 2
        assert pmm.num_write_heads == 1

    def test_init_memory(self, pmm: PredictiveMemoryModule, device: torch.device) -> None:
        """Test memory initialization."""
        batch_size = 4
        memory = pmm.init_memory(batch_size, device)

        assert memory.shape == (batch_size, 32, 16)
        assert memory.device == device

    def test_read(self, pmm: PredictiveMemoryModule, device: torch.device) -> None:
        """Test memory read operation."""
        batch_size = 4
        latent = torch.randn(batch_size, 64, device=device)
        memory = pmm.init_memory(batch_size, device)

        read_vectors, read_weights = pmm.read(latent, memory)

        assert read_vectors.shape == (batch_size, 2 * 16)  # num_read_heads * memory_dim
        assert read_weights.shape == (batch_size, 2, 32)  # num_read_heads, memory_size
        # Weights should sum to 1 along memory dimension
        assert torch.allclose(read_weights.sum(dim=-1), torch.ones(batch_size, 2, device=device), atol=1e-5)

    def test_write(self, pmm: PredictiveMemoryModule, device: torch.device) -> None:
        """Test memory write operation."""
        batch_size = 4
        latent = torch.randn(batch_size, 64, device=device)
        memory = pmm.init_memory(batch_size, device)

        new_memory, write_weights = pmm.write(latent, memory)

        assert new_memory.shape == memory.shape
        assert write_weights.shape == (batch_size, 1, 32)  # num_write_heads, memory_size
        # Memory should be modified
        assert not torch.allclose(new_memory, memory)

    def test_forward(self, pmm: PredictiveMemoryModule, device: torch.device) -> None:
        """Test full forward pass."""
        batch_size = 4
        latent = torch.randn(batch_size, 64, device=device)

        # Without initial memory
        augmented, new_memory, info = pmm(latent)

        assert augmented.shape == (batch_size, 64)
        assert new_memory.shape == (batch_size, 32, 16)
        assert "read_weights" in info
        assert "write_weights" in info
        assert "memory_usage" in info

    def test_forward_with_memory(self, pmm: PredictiveMemoryModule, device: torch.device) -> None:
        """Test forward pass with existing memory."""
        batch_size = 4
        latent = torch.randn(batch_size, 64, device=device)
        memory = pmm.init_memory(batch_size, device)

        augmented, new_memory, info = pmm(latent, memory)

        assert augmented.shape == (batch_size, 64)
        assert new_memory.shape == memory.shape

    def test_memory_stats(self, pmm: PredictiveMemoryModule, device: torch.device) -> None:
        """Test memory statistics computation."""
        batch_size = 4
        memory = pmm.init_memory(batch_size, device)

        stats = pmm.get_memory_stats(memory)

        assert "memory_saturation" in stats
        assert "memory_mean_norm" in stats
        assert "memory_max_norm" in stats
        assert "memory_std" in stats
        assert 0.0 <= stats["memory_saturation"] <= 1.0

    def test_gradient_flow(self, pmm: PredictiveMemoryModule, device: torch.device) -> None:
        """Test gradient flow through PMM."""
        batch_size = 4
        latent = torch.randn(batch_size, 64, device=device, requires_grad=True)

        augmented, new_memory, _ = pmm(latent)
        loss = augmented.sum()
        loss.backward()

        assert latent.grad is not None
        assert latent.grad.shape == latent.shape
        assert not torch.isnan(latent.grad).any()

    def test_memory_persistence(self, pmm: PredictiveMemoryModule, device: torch.device) -> None:
        """Test that memory persists information across steps."""
        batch_size = 2
        
        # Initial memory
        memory = pmm.init_memory(batch_size, device)
        
        # Write some information
        latent1 = torch.randn(batch_size, 64, device=device)
        _, memory, _ = pmm(latent1, memory)
        
        # Write more information
        latent2 = torch.randn(batch_size, 64, device=device)
        _, memory2, _ = pmm(latent2, memory)
        
        # Memory should have changed
        assert not torch.allclose(memory, memory2)

    def test_attention_sharpening(self, device: torch.device) -> None:
        """Test that sharp_factor affects attention distribution."""
        batch_size = 2
        
        pmm_soft = PredictiveMemoryModule(
            input_dim=64, memory_size=32, memory_dim=16,
            num_read_heads=1, sharp_factor=0.1
        ).to(device)
        
        pmm_sharp = PredictiveMemoryModule(
            input_dim=64, memory_size=32, memory_dim=16,
            num_read_heads=1, sharp_factor=10.0
        ).to(device)
        
        latent = torch.randn(batch_size, 64, device=device)
        memory = torch.randn(batch_size, 32, 16, device=device)
        
        _, weights_soft = pmm_soft.read(latent, memory)
        _, weights_sharp = pmm_sharp.read(latent, memory)
        
        # Sharp attention should have higher max values (more peaked)
        assert weights_sharp.max() >= weights_soft.max()
