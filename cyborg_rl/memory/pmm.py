"""Predictive Memory Module (PMM) implementation."""

from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictiveMemoryModule(nn.Module):
    """
    Differentiable memory module with read/write heads.

    Features:
    - Content-based addressing
    - Multiple read/write heads
    - Memory pressure metrics
    - Entropy-based attention sharpening
    """

    def __init__(
        self,
        input_dim: int,
        memory_size: int = 128,
        memory_dim: int = 64,
        num_read_heads: int = 4,
        num_write_heads: int = 1,
        sharp_factor: float = 1.0,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        self.sharp_factor = sharp_factor

        # Read keys generator
        self.read_keys = nn.Linear(input_dim, num_read_heads * memory_dim)

        # Write keys and values generator
        self.write_keys = nn.Linear(input_dim, num_write_heads * memory_dim)
        self.write_values = nn.Linear(input_dim, num_write_heads * memory_dim)
        self.erase_vectors = nn.Linear(input_dim, num_write_heads * memory_dim)

        # Output projection (combines input + read vectors)
        self.output_proj = nn.Linear(input_dim + num_read_heads * memory_dim, input_dim)

        self.layer_norm = nn.LayerNorm(input_dim)

    def init_memory(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize zero memory state."""
        return torch.zeros(batch_size, self.memory_size, self.memory_dim, device=device)

    def _addressing(self, keys: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Compute attention weights based on cosine similarity.

        Args:
            keys: [B, Heads, D]
            memory: [B, Slots, D]

        Returns:
            weights: [B, Heads, Slots]
        """
        # Normalize for cosine similarity
        keys_norm = F.normalize(keys, dim=-1)
        mem_norm = F.normalize(memory, dim=-1)

        # Similarity: [B, Heads, Slots]
        scores = torch.matmul(keys_norm, mem_norm.transpose(-2, -1))

        # Sharpen and softmax
        weights = F.softmax(scores * self.sharp_factor, dim=-1)
        return weights

    def read(self, latent: torch.Tensor, memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read from memory."""
        batch_size = latent.shape[0]

        # Generate keys: [B, Heads, D]
        keys = self.read_keys(latent).view(batch_size, self.num_read_heads, self.memory_dim)

        # Get weights: [B, Heads, Slots]
        weights = self._addressing(keys, memory)

        # Read: [B, Heads, D]
        read_vectors = torch.matmul(weights, memory)

        # Flatten: [B, Heads * D]
        read_flat = read_vectors.view(batch_size, -1)

        return read_flat, weights

    def write(
        self, latent: torch.Tensor, memory: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Write to memory with optimized single-head path."""
        batch_size = latent.shape[0]

        # Generate keys, values, erase: [B, Heads, D]
        keys = self.write_keys(latent).view(batch_size, self.num_write_heads, self.memory_dim)
        values = self.write_values(latent).view(batch_size, self.num_write_heads, self.memory_dim)
        erase = torch.sigmoid(
            self.erase_vectors(latent).view(batch_size, self.num_write_heads, self.memory_dim)
        )

        # Get weights: [B, Heads, Slots]
        weights = self._addressing(keys, memory)

        # Expand for broadcasting
        # weights: [B, Heads, Slots, 1]
        w = weights.unsqueeze(-1)
        # erase: [B, Heads, 1, D]
        e = erase.unsqueeze(-2)
        # values: [B, Heads, 1, D]
        v = values.unsqueeze(-2)

        # Fast path for single write head (common case)
        if self.num_write_heads == 1:
            # Vectorized single-head write
            erase_gate = 1 - w[:, 0] * e[:, 0]  # [B, Slots, D]
            new_memory = memory * erase_gate + w[:, 0] * v[:, 0]
            return new_memory, weights

        # Multi-head write (sequential, as each head may affect the next)
        # Standard NTM implementation:
        prev_mem = memory
        for h in range(self.num_write_heads):
            # Erase
            erase_gate = 1 - w[:, h] * e[:, h]
            prev_mem = prev_mem * erase_gate
            # Add
            add_gate = w[:, h] * v[:, h]
            prev_mem = prev_mem + add_gate

        return prev_mem, weights

    def forward(
        self, latent: torch.Tensor, memory: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass: Read -> Output -> Write.
        """
        if memory is None:
            memory = self.init_memory(latent.shape[0], latent.device)

        # 1. Read
        read_vecs, read_weights = self.read(latent, memory)

        # 2. Combine
        combined = torch.cat([latent, read_vecs], dim=-1)
        output = self.layer_norm(self.output_proj(combined) + latent)  # Residual

        # 3. Write
        updated_memory, write_weights = self.write(latent, memory)

        # 4. Metrics
        # Pressure: Fraction of slots with high norm
        slot_norms = updated_memory.norm(dim=-1)
        pressure = (slot_norms > 0.5).float().mean(dim=-1)

        # Entropy of read weights
        read_entropy = -(read_weights * (read_weights + 1e-8).log()).sum(dim=-1).mean(dim=-1)

        # Resonance (max attention)
        resonance = read_weights.max(dim=-1)[0].mean(dim=-1)

        info = {
            "read_weights": read_weights,
            "write_weights": write_weights,
            "memory_usage": (updated_memory.abs().sum(-1) > 1e-6).float().mean(),
            "pressure": pressure,
            "read_entropy": read_entropy,
            "resonance": resonance,
        }

        return output, updated_memory, info

    def get_memory_stats(self, memory: torch.Tensor) -> Dict[str, float]:
        """Compute statistics for monitoring."""
        norms = memory.norm(dim=-1)
        return {
            "memory_saturation": (norms > 0.1).float().mean().item(),
            "memory_mean_norm": norms.mean().item(),
            "memory_max_norm": norms.max().item(),
            "memory_std": memory.std().item(),
        }
