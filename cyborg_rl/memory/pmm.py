"""Predictive Memory Module (PMM) for augmenting RL agents with external memory."""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PredictiveMemoryModule(nn.Module):
    """
    Predictive Memory Module (PMM).

    Implements a differentiable external memory with content-based addressing,
    read/write operations, and memory-augmented state generation.

    This module can be integrated into RL agents to provide long-term
    memory capabilities beyond the hidden state of recurrent networks.
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
        """
        Initialize the PMM.

        Args:
            input_dim: Dimension of input latent state.
            memory_size: Number of memory slots (N).
            memory_dim: Dimension of each memory slot (M).
            num_read_heads: Number of read heads (R).
            num_write_heads: Number of write heads (W).
            sharp_factor: Sharpening factor for attention weights.
        """
        super().__init__()

        self.input_dim = input_dim
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        self.sharp_factor = sharp_factor

        self.read_key_proj = nn.Linear(input_dim, num_read_heads * memory_dim)
        self.read_beta_proj = nn.Linear(input_dim, num_read_heads)

        self.write_key_proj = nn.Linear(input_dim, num_write_heads * memory_dim)
        self.write_beta_proj = nn.Linear(input_dim, num_write_heads)
        self.write_erase_proj = nn.Linear(input_dim, num_write_heads * memory_dim)
        self.write_add_proj = nn.Linear(input_dim, num_write_heads * memory_dim)

        self.output_proj = nn.Linear(num_read_heads * memory_dim, input_dim)

        self.layer_norm = nn.LayerNorm(input_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with proper scaling."""
        for module in [
            self.read_key_proj,
            self.write_key_proj,
            self.write_erase_proj,
            self.write_add_proj,
            self.output_proj,
        ]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        nn.init.zeros_(self.read_beta_proj.weight)
        nn.init.ones_(self.read_beta_proj.bias)
        nn.init.zeros_(self.write_beta_proj.weight)
        nn.init.ones_(self.write_beta_proj.bias)

    def init_memory(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """
        Initialize memory state.

        Args:
            batch_size: Batch size.
            device: Target device.

        Returns:
            torch.Tensor: Initial memory state [B, N, M].
        """
        memory = torch.zeros(
            batch_size, self.memory_size, self.memory_dim,
            device=device, dtype=torch.float32
        )
        nn.init.xavier_uniform_(memory)
        return memory * 0.01

    def _content_addressing(
        self,
        keys: torch.Tensor,
        betas: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute content-based attention weights.

        Args:
            keys: Query keys [B, H, M].
            betas: Sharpening factors [B, H].
            memory: Memory state [B, N, M].

        Returns:
            torch.Tensor: Attention weights [B, H, N].
        """
        keys_norm = F.normalize(keys, dim=-1)
        memory_norm = F.normalize(memory, dim=-1)

        similarity = torch.einsum("bhm,bnm->bhn", keys_norm, memory_norm)

        betas = F.softplus(betas).unsqueeze(-1) * self.sharp_factor
        weights = F.softmax(similarity * betas, dim=-1)

        return weights

    def read(
        self,
        latent: torch.Tensor,
        memory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read from memory.

        Args:
            latent: Latent state [B, D].
            memory: Memory state [B, N, M].

        Returns:
            Tuple of (read_vectors [B, R*M], read_weights [B, R, N]).
        """
        batch_size = latent.shape[0]

        keys = self.read_key_proj(latent)
        keys = rearrange(keys, "b (h m) -> b h m", h=self.num_read_heads)

        betas = self.read_beta_proj(latent)

        weights = self._content_addressing(keys, betas, memory)

        read_vectors = torch.einsum("bhn,bnm->bhm", weights, memory)
        read_vectors = rearrange(read_vectors, "b h m -> b (h m)")

        return read_vectors, weights

    def write(
        self,
        latent: torch.Tensor,
        memory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Write to memory.

        Args:
            latent: Latent state [B, D].
            memory: Memory state [B, N, M].

        Returns:
            Tuple of (updated_memory [B, N, M], write_weights [B, W, N]).
        """
        batch_size = latent.shape[0]

        keys = self.write_key_proj(latent)
        keys = rearrange(keys, "b (h m) -> b h m", h=self.num_write_heads)

        betas = self.write_beta_proj(latent)
        weights = self._content_addressing(keys, betas, memory)

        erase = self.write_erase_proj(latent)
        erase = rearrange(erase, "b (h m) -> b h m", h=self.num_write_heads)
        erase = torch.sigmoid(erase)

        add = self.write_add_proj(latent)
        add = rearrange(add, "b (h m) -> b h m", h=self.num_write_heads)

        erase_term = torch.einsum("bhn,bhm->bnm", weights, erase)
        memory = memory * (1 - erase_term)

        add_term = torch.einsum("bhn,bhm->bnm", weights, add)
        memory = memory + add_term

        return memory, weights

    def forward(
        self,
        latent: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Full PMM forward pass: read then write.

        Args:
            latent: Latent state from encoder [B, D].
            memory: Optional memory state [B, N, M]. Initialized if None.

        Returns:
            Tuple of:
                - memory_augmented_state [B, D]: Original latent + memory read
                - updated_memory [B, N, M]: Memory after write
                - info dict with read/write weights
        """
        batch_size = latent.shape[0]
        device = latent.device

        if memory is None:
            memory = self.init_memory(batch_size, device)

        read_vectors, read_weights = self.read(latent, memory)

        read_output = self.output_proj(read_vectors)
        memory_augmented_state = self.layer_norm(latent + read_output)

        updated_memory, write_weights = self.write(latent, memory)

        info = {
            "read_weights": read_weights,
            "write_weights": write_weights,
            "memory_usage": (memory.abs().sum(-1) > 1e-6).float().mean(),
        }

        return memory_augmented_state, updated_memory, info

    def get_memory_stats(self, memory: torch.Tensor) -> dict:
        """
        Compute memory statistics for monitoring.

        Args:
            memory: Memory state [B, N, M].

        Returns:
            dict: Memory statistics.
        """
        memory_norm = memory.norm(dim=-1)
        return {
            "memory_saturation": (memory_norm > 0.1).float().mean().item(),
            "memory_mean_norm": memory_norm.mean().item(),
            "memory_max_norm": memory_norm.max().item(),
            "memory_std": memory.std().item(),
        }
