#!/usr/bin/env python3
"""
MemoryV7 - Unified Memory Module Suite

Includes:
- PMMV7: Enhanced Predictive Memory Module with eviction, decay, multi-head
- SlotMemoryV7: Simple slot-based memory
- KVMemV7: Key-value store with learned indexing
- RingMemoryV7: Circular buffer for temporal tasks

All modules share a unified interface:
    read_vector, new_state, logs = memory.forward_step(hidden, state)
"""

from typing import Any, Dict, Optional, Tuple, Union
from abc import ABC, abstractmethod
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== BASE MEMORY ====================
class BaseMemoryV7(ABC, nn.Module):
    """Abstract base class for V7 memory modules."""

    @abstractmethod
    def get_initial_state(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Create initial memory state."""
        pass

    @abstractmethod
    def forward_step(
        self,
        hidden: torch.Tensor,
        state: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Single-step memory forward.

        Args:
            hidden: Hidden state (B, D)
            state: Memory state
            mask: Optional mask for done episodes (B,) where 1=keep, 0=reset

        Returns:
            read_vector: Retrieved memory (B, D)
            new_state: Updated memory state
            logs: Diagnostic dict
        """
        pass

    def forward_sequence(
        self,
        hidden_seq: torch.Tensor,
        initial_state: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Process sequence of hidden states.

        Args:
            hidden_seq: (B, T, D)
            initial_state: Initial memory state
            masks: (B, T) done masks

        Returns:
            read_seq: (B, T, D)
            final_state: Final memory state
            logs: Aggregated diagnostics
        """
        B, T, D = hidden_seq.shape
        state = initial_state
        reads = []
        all_logs = []

        for t in range(T):
            h_t = hidden_seq[:, t]
            mask_t = masks[:, t] if masks is not None else None
            read, state, logs = self.forward_step(h_t, state, mask_t)
            reads.append(read)
            all_logs.append(logs)

        read_seq = torch.stack(reads, dim=1)

        # Aggregate logs
        agg_logs = {}
        for key in all_logs[0].keys():
            if isinstance(all_logs[0][key], (int, float)):
                agg_logs[f"avg_{key}"] = sum(l[key] for l in all_logs) / T
            elif isinstance(all_logs[0][key], torch.Tensor):
                agg_logs[key] = all_logs[-1][key]  # Keep last

        return read_seq, state, agg_logs

    def reset_state(
        self, state: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Reset state for done episodes."""
        # Default: multiply by mask
        mask_expanded = mask.view(-1, *([1] * (state.dim() - 1)))
        return state * mask_expanded


# ==================== PMM V7 ====================
class PMMV7(BaseMemoryV7):
    """
    Enhanced Predictive Memory Module V7.

    Improvements over V5:
    - Usage-based eviction
    - Temporal decay
    - Multi-head addressing
    - Write conflict penalty
    """

    def __init__(
        self,
        memory_dim: int = 256,
        num_slots: int = 16,
        num_heads: int = 4,
        write_rate_target_inv: int = 2000,
        decay_rate: float = 0.99,
        gate_type: str = "soft",
        temperature: float = 1.0,
        sharpness: float = 2.0,
    ) -> None:
        super().__init__()

        self.memory_dim = memory_dim
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.head_dim = memory_dim // num_heads
        self.decay_rate = decay_rate
        self.gate_type = gate_type
        self.temperature = temperature
        self.sharpness = sharpness

        # Query/Key/Value projections
        self.query_proj = nn.Linear(memory_dim, memory_dim)
        self.key_proj = nn.Linear(memory_dim, memory_dim)
        self.value_proj = nn.Linear(memory_dim, memory_dim)

        # Write gate
        self.write_gate = nn.Sequential(
            nn.Linear(memory_dim, memory_dim // 2),
            nn.ReLU(),
            nn.Linear(memory_dim // 2, 1),
            nn.Sigmoid() if gate_type == "soft" else nn.Identity(),
        )

        # Output projection
        self.output_proj = nn.Linear(memory_dim, memory_dim)

        # Sparsity regularization
        self.write_rate_target_inv = write_rate_target_inv

        # Layer norm
        self.norm = nn.LayerNorm(memory_dim)

    def get_initial_state(
        self, batch_size: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Returns dict with 'memory' and 'usage' tensors."""
        return {
            "memory": torch.zeros(batch_size, self.num_slots, self.memory_dim, device=device),
            "usage": torch.zeros(batch_size, self.num_slots, device=device),
            "age": torch.zeros(batch_size, self.num_slots, device=device),
        }

    def forward_step(
        self,
        hidden: torch.Tensor,
        state: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Single-step PMM forward.

        Args:
            hidden: (B, D)
            state: Dict with 'memory', 'usage', 'age'
            mask: (B,) done mask

        Returns:
            read_vector: (B, D)
            new_state: Updated state dict
            logs: Diagnostics
        """
        B = hidden.size(0)
        memory = state["memory"]
        usage = state["usage"]
        age = state["age"]

        # Reset state for done episodes
        if mask is not None:
            mask_mem = mask.view(B, 1, 1)
            mask_slot = mask.view(B, 1)
            memory = memory * mask_mem
            usage = usage * mask_slot
            age = age * mask_slot

        # ==================== READ ====================
        # Multi-head attention
        query = self.query_proj(hidden).view(B, self.num_heads, self.head_dim)
        keys = self.key_proj(memory).view(B, self.num_slots, self.num_heads, self.head_dim)

        # Attention scores
        attn = torch.einsum("bhd,bshd->bhs", query, keys) / math.sqrt(self.head_dim)
        attn = F.softmax(attn * self.temperature, dim=1)

        # Read with sharpness
        attn_sharp = attn ** self.sharpness
        attn_sharp = attn_sharp / (attn_sharp.sum(dim=1, keepdim=True) + 1e-8)

        # Read values
        values = self.value_proj(memory).view(B, self.num_slots, self.num_heads, self.head_dim)
        read = torch.einsum("bhs,bshd->bhd", attn_sharp, values)
        read = read.reshape(B, self.memory_dim)
        read = self.output_proj(read)

        # ==================== WRITE ====================
        # Write gate
        gate = self.write_gate(hidden)  # (B, 1)

        # Usage-based eviction: write to least used slot
        usage_score = usage + age * 0.1
        write_weights = F.softmax(-usage_score * 10, dim=-1)  # (B, S)

        # Update memory
        write_content = hidden.unsqueeze(1).expand(-1, self.num_slots, -1)
        write_weights_expanded = write_weights.unsqueeze(-1) * gate.unsqueeze(1)  # (B, S, 1) * (B, 1, 1)
        new_memory = memory * (1 - write_weights_expanded) + write_content * write_weights_expanded

        # Temporal decay
        new_memory = new_memory * self.decay_rate + memory * (1 - self.decay_rate)

        # Update usage and age
        new_usage = usage * 0.99 + attn_sharp.mean(dim=-1)  # Read usage
        new_usage = new_usage + write_weights * gate.view(B, 1)  # Write usage  (B, S) * (B, 1)
        new_age = age + 1

        # ==================== DIAGNOSTICS ====================
        # Write conflict: overlap between read and write attention
        read_attn = attn_sharp.mean(dim=-1)  # (B, S)
        write_conflict = (read_attn * write_weights).sum(dim=-1).mean().item()

        # Gate usage histogram
        gate_val = gate.mean().item()
        gate_low = (gate < 0.33).float().mean().item()
        gate_mid = ((gate >= 0.33) & (gate < 0.67)).float().mean().item()
        gate_high = (gate >= 0.67).float().mean().item()

        # Sparsity loss
        target = 1.0 / self.write_rate_target_inv
        sparsity_loss = (gate.mean() - target).pow(2)

        logs = {
            "write_strength": gate_val,
            "read_entropy": -(read_attn * (read_attn + 1e-8).log()).sum(-1).mean().item(),
            "gate_usage_low": gate_low,
            "gate_usage_mid": gate_mid,
            "gate_usage_high": gate_high,
            "overwrite_conflict": write_conflict,
            "sparsity_loss": sparsity_loss,
            "usage_mean": new_usage.mean().item(),
            "age_mean": new_age.mean().item(),
        }

        new_state = {
            "memory": new_memory,
            "usage": new_usage,
            "age": new_age,
        }

        return read, new_state, logs

    def reset_state(
        self, state: Dict[str, torch.Tensor], mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Reset state for done episodes."""
        mask_mem = mask.view(-1, 1, 1)
        mask_slot = mask.view(-1, 1)
        return {
            "memory": state["memory"] * mask_mem,
            "usage": state["usage"] * mask_slot,
            "age": state["age"] * mask_slot,
        }


# ==================== SLOT MEMORY V7 ====================
class SlotMemoryV7(BaseMemoryV7):
    """Simple slot-based memory with fixed addressing."""

    def __init__(
        self,
        memory_dim: int = 256,
        num_slots: int = 8,
    ) -> None:
        super().__init__()
        self.memory_dim = memory_dim
        self.num_slots = num_slots

        self.address_net = nn.Linear(memory_dim, num_slots)
        self.write_net = nn.Linear(memory_dim, memory_dim)
        self.read_net = nn.Linear(memory_dim * num_slots, memory_dim)

    def get_initial_state(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        return torch.zeros(batch_size, self.num_slots, self.memory_dim, device=device)

    def forward_step(
        self,
        hidden: torch.Tensor,
        state: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        B = hidden.size(0)

        if mask is not None:
            state = state * mask.view(B, 1, 1)

        # Address
        addr = F.softmax(self.address_net(hidden), dim=-1)

        # Read
        read = (state * addr.unsqueeze(-1)).sum(dim=1)

        # Write
        write_content = self.write_net(hidden)
        new_state = state + addr.unsqueeze(-1) * write_content.unsqueeze(1)

        logs = {"address_entropy": -(addr * (addr + 1e-8).log()).sum(-1).mean().item()}

        return read, new_state, logs


# ==================== KV MEMORY V7 ====================
class KVMemV7(BaseMemoryV7):
    """Key-Value memory with learned indexing."""

    def __init__(
        self,
        memory_dim: int = 256,
        num_slots: int = 32,
        key_dim: int = 64,
    ) -> None:
        super().__init__()
        self.memory_dim = memory_dim
        self.num_slots = num_slots
        self.key_dim = key_dim

        self.key_proj = nn.Linear(memory_dim, key_dim)
        self.value_proj = nn.Linear(memory_dim, memory_dim)
        self.query_proj = nn.Linear(memory_dim, key_dim)
        self.output_proj = nn.Linear(memory_dim, memory_dim)

    def get_initial_state(
        self, batch_size: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        return {
            "keys": torch.zeros(batch_size, self.num_slots, self.key_dim, device=device),
            "values": torch.zeros(batch_size, self.num_slots, self.memory_dim, device=device),
            "write_ptr": torch.zeros(batch_size, dtype=torch.long, device=device),
        }

    def forward_step(
        self,
        hidden: torch.Tensor,
        state: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]:
        B = hidden.size(0)
        keys = state["keys"]
        values = state["values"]
        write_ptr = state["write_ptr"]

        if mask is not None:
            mask_kv = mask.view(B, 1, 1)
            keys = keys * mask_kv
            values = values * mask_kv
            write_ptr = write_ptr * mask.long()

        # Query and retrieve
        query = self.query_proj(hidden)
        attn = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)).squeeze(1)
        attn = F.softmax(attn / math.sqrt(self.key_dim), dim=-1)
        read = torch.bmm(attn.unsqueeze(1), values).squeeze(1)
        read = self.output_proj(read)

        # Write to current slot
        new_key = self.key_proj(hidden)
        new_value = self.value_proj(hidden)

        # Scatter update
        idx = write_ptr.view(B, 1, 1).expand(-1, 1, self.key_dim)
        new_keys = keys.scatter(1, idx, new_key.unsqueeze(1))
        idx_v = write_ptr.view(B, 1, 1).expand(-1, 1, self.memory_dim)
        new_values = values.scatter(1, idx_v, new_value.unsqueeze(1))

        # Advance pointer
        new_ptr = (write_ptr + 1) % self.num_slots

        logs = {"read_attn_max": attn.max(dim=-1).values.mean().item()}

        return read, {"keys": new_keys, "values": new_values, "write_ptr": new_ptr}, logs

    def reset_state(
        self, state: Dict[str, torch.Tensor], mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        B = mask.size(0)
        mask_kv = mask.view(B, 1, 1)
        return {
            "keys": state["keys"] * mask_kv,
            "values": state["values"] * mask_kv,
            "write_ptr": state["write_ptr"] * mask.long(),
        }


# ==================== RING MEMORY V7 ====================
class RingMemoryV7(BaseMemoryV7):
    """Circular buffer memory for temporal tasks."""

    def __init__(
        self,
        memory_dim: int = 256,
        buffer_size: int = 64,
    ) -> None:
        super().__init__()
        self.memory_dim = memory_dim
        self.buffer_size = buffer_size

        self.compress = nn.Linear(memory_dim, memory_dim)
        self.query = nn.Linear(memory_dim, memory_dim)
        self.output = nn.Linear(memory_dim, memory_dim)

    def get_initial_state(
        self, batch_size: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        return {
            "buffer": torch.zeros(batch_size, self.buffer_size, self.memory_dim, device=device),
            "ptr": torch.zeros(batch_size, dtype=torch.long, device=device),
        }

    def forward_step(
        self,
        hidden: torch.Tensor,
        state: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]:
        B = hidden.size(0)
        buffer = state["buffer"]
        ptr = state["ptr"]

        if mask is not None:
            buffer = buffer * mask.view(B, 1, 1)
            ptr = ptr * mask.long()

        # Query entire buffer
        q = self.query(hidden)
        attn = torch.bmm(q.unsqueeze(1), buffer.transpose(1, 2)).squeeze(1)
        attn = F.softmax(attn / math.sqrt(self.memory_dim), dim=-1)
        read = torch.bmm(attn.unsqueeze(1), buffer).squeeze(1)
        read = self.output(read)

        # Write to ring buffer
        compressed = self.compress(hidden)
        idx = ptr.view(B, 1, 1).expand(-1, 1, self.memory_dim)
        new_buffer = buffer.scatter(1, idx, compressed.unsqueeze(1))
        new_ptr = (ptr + 1) % self.buffer_size

        logs = {"buffer_fill": (new_ptr.float() / self.buffer_size).mean().item()}

        return read, {"buffer": new_buffer, "ptr": new_ptr}, logs

    def reset_state(
        self, state: Dict[str, torch.Tensor], mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        B = mask.size(0)
        return {
            "buffer": state["buffer"] * mask.view(B, 1, 1),
            "ptr": state["ptr"] * mask.long(),
        }


# ==================== FACTORY ====================
def create_memory(
    memory_type: str = "pmm",
    memory_dim: int = 256,
    num_slots: int = 16,
    **kwargs,
) -> BaseMemoryV7:
    """Factory function to create memory modules."""
    
    if memory_type == "pmm":
        return PMMV7(memory_dim=memory_dim, num_slots=num_slots, **kwargs)
    elif memory_type == "slot":
        return SlotMemoryV7(memory_dim=memory_dim, num_slots=num_slots)
    elif memory_type == "kv":
        key_dim = kwargs.get("key_dim", 64)
        return KVMemV7(memory_dim=memory_dim, num_slots=num_slots, key_dim=key_dim)
    elif memory_type == "ring":
        buffer_size = kwargs.get("buffer_size", num_slots)
        return RingMemoryV7(memory_dim=memory_dim, buffer_size=buffer_size)
    else:
        raise ValueError(f"Unknown memory type '{memory_type}'. Choose from ['pmm', 'slot', 'kv', 'ring']")

