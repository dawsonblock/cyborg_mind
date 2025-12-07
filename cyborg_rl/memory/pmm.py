"""
Honest Memory Engine v2.0 (PMM)

Production-grade Predictive Memory Module with:
- Controlled Write Gate (target sparsity)
- Zero Leakage (strict reset on done)
- Deterministic Addressing (cosine, softmax)
- Comprehensive Diagnostics (gate_usage, write_strength, overwrite_conflicts)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any


class PMM(nn.Module):
    """
    Honest Memory Engine v2.0 (PMM)

    Features:
    - Controlled Write Gate with learnable sparsity
    - Zero Leakage on episode boundaries
    - Cosine similarity addressing
    - Full diagnostic logging
    """

    def __init__(
        self,
        memory_dim: int,
        num_slots: int,
        write_rate_target_inv: int = 2000,
        gate_type: str = "soft",
        temperature: float = 1.0,
        sharpness: float = 2.0,
    ) -> None:
        """
        Initialize PMM.

        Args:
            memory_dim: Dimension of each memory slot
            num_slots: Number of memory slots
            write_rate_target_inv: Inverse of target write rate (1/N)
            gate_type: "soft" or "hard" gating
            temperature: Temperature for gate sigmoid
            sharpness: Sharpness for attention softmax
        """
        super().__init__()
        self.memory_dim = memory_dim
        self.num_slots = num_slots
        self.write_rate_target = 1.0 / write_rate_target_inv
        self.gate_type = gate_type
        self.temperature = temperature
        self.sharpness = sharpness

        # ==================== READ HEAD ====================
        self.query_proj = nn.Linear(memory_dim, memory_dim)

        # ==================== WRITE HEAD ====================
        self.write_key_proj = nn.Linear(memory_dim, memory_dim)
        self.write_val_proj = nn.Linear(memory_dim, memory_dim)
        self.gate_proj = nn.Linear(memory_dim, 1)

        # ==================== DIAGNOSTICS ====================
        self._total_writes = 0
        self._total_steps = 0
        self._overwrite_accumulator = 0.0

    # ==================== STATE MANAGEMENT ====================

    def get_initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create zeroed initial memory state."""
        return torch.zeros(batch_size, self.num_slots, self.memory_dim, device=device)

    def reset_state(self, memory: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Reset memory for done episodes (zero leakage).

        Args:
            memory: Current memory (B, Slots, Dim)
            mask: Done mask (B, 1) - 1 = keep, 0 = reset

        Returns:
            Memory with done episodes zeroed
        """
        mask_expanded = mask.unsqueeze(-1).expand_as(memory)
        return memory * mask_expanded

    def detach_state(self, memory: torch.Tensor) -> torch.Tensor:
        """Detach memory from computation graph for TBPTT."""
        return memory.detach()

    # ==================== FORWARD PASS ====================

    def forward(
        self,
        current_memory: torch.Tensor,
        hidden_state: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Single step forward.

        Args:
            current_memory: (B, Slots, Dim)
            hidden_state: (B, Dim)
            mask: (B, 1) - 1 = valid, 0 = done/reset

        Returns:
            read_vector: (B, Dim)
            next_memory: (B, Slots, Dim)
            logs: Diagnostic dictionary
        """
        return self._forward_step(current_memory, hidden_state, mask)

    def _forward_step(
        self,
        current_memory: torch.Tensor,
        hidden_state: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Core forward step implementation."""
        batch_size = hidden_state.shape[0]

        # ==================== 1. READ ====================
        query = self.query_proj(hidden_state)
        query_norm = F.normalize(query, p=2, dim=-1)
        memory_norm = F.normalize(current_memory, p=2, dim=-1)

        # Cosine similarity: (B, Slots)
        similarity = torch.sum(memory_norm * query_norm.unsqueeze(1), dim=-1)
        read_weights = F.softmax(similarity * self.sharpness, dim=-1)

        # Weighted read: (B, Dim)
        read_vector = torch.matmul(read_weights.unsqueeze(1), current_memory).squeeze(1)

        # ==================== 2. WRITE ====================
        write_key = self.write_key_proj(hidden_state)
        write_val = self.write_val_proj(hidden_state)
        gate_logit = self.gate_proj(hidden_state) / self.temperature

        # Gate computation
        if self.gate_type == "hard":
            gate_soft = torch.sigmoid(gate_logit)
            gate_open = (gate_soft > 0.5).float() - gate_soft.detach() + gate_soft
        else:
            gate_open = torch.sigmoid(gate_logit)

        # Write addressing
        write_key_norm = F.normalize(write_key, p=2, dim=-1)
        write_similarity = torch.sum(memory_norm * write_key_norm.unsqueeze(1), dim=-1)
        write_weights = F.softmax(write_similarity * self.sharpness, dim=-1)

        # ==================== OVERWRITE CONFLICT DETECTION ====================
        # Measure overlap between read and write attention
        attention_overlap = torch.sum(read_weights * write_weights, dim=-1)
        overwrite_conflict = (attention_overlap * gate_open.squeeze(-1)).mean()

        # Write operation
        write_op = gate_open.unsqueeze(1) * write_weights.unsqueeze(-1)
        next_memory = current_memory + write_op * (write_val.unsqueeze(1) - current_memory)

        # ==================== 3. ZERO LEAKAGE ====================
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(next_memory)
            next_memory = next_memory * mask_expanded

        # ==================== 4. DIAGNOSTICS ====================
        gate_mean = gate_open.mean()
        sparsity_loss = (gate_mean - self.write_rate_target) ** 2

        # Gate usage histogram (binned)
        with torch.no_grad():
            gate_values = gate_open.squeeze(-1)
            gate_low = (gate_values < 0.33).float().mean().item()
            gate_mid = ((gate_values >= 0.33) & (gate_values < 0.66)).float().mean().item()
            gate_high = (gate_values >= 0.66).float().mean().item()

            # Update counters
            self._total_steps += batch_size
            self._total_writes += (gate_values > 0.5).float().sum().item()
            self._overwrite_accumulator += overwrite_conflict.item() * batch_size

        logs = {
            "sparsity_loss": sparsity_loss,
            "write_strength": gate_mean.item(),
            "read_entropy": -torch.sum(
                read_weights * torch.log(read_weights + 1e-8), dim=-1
            ).mean().item(),
            "gate_usage_low": gate_low,
            "gate_usage_mid": gate_mid,
            "gate_usage_high": gate_high,
            "overwrite_conflict": overwrite_conflict.item(),
        }

        return read_vector, next_memory, logs

    # ==================== SEQUENCE PROCESSING ====================

    def forward_sequence(
        self,
        init_memory: torch.Tensor,
        hidden_seq: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process a sequence of hidden states.

        Args:
            init_memory: (B, Slots, Dim)
            hidden_seq: (B, L, Dim)
            masks: (B, L, 1) - Optional masks (1=valid, 0=reset/done)

        Returns:
            read_seq: (B, L, Dim)
            aux_logs: Aggregated logs
        """
        B, L, D = hidden_seq.shape
        curr_memory = init_memory

        read_vectors = []
        total_sparsity_loss = 0.0
        total_write_strength = 0.0
        total_overwrite = 0.0

        for t in range(L):
            h_t = hidden_seq[:, t, :]
            mask_t = masks[:, t, :] if masks is not None else None

            read_v, next_mem, logs = self._forward_step(curr_memory, h_t, mask_t)

            read_vectors.append(read_v)
            curr_memory = next_mem
            total_sparsity_loss += logs["sparsity_loss"]
            total_write_strength += logs["write_strength"]
            total_overwrite += logs["overwrite_conflict"]

        read_seq = torch.stack(read_vectors, dim=1)

        avg_logs = {
            "sparsity_loss": total_sparsity_loss / L,
            "avg_write_strength": total_write_strength / L,
            "avg_overwrite_conflict": total_overwrite / L,
        }

        return read_seq, avg_logs

    # ==================== GLOBAL STATS ====================

    def get_global_stats(self) -> Dict[str, float]:
        """Get cumulative statistics."""
        return {
            "total_steps": self._total_steps,
            "total_writes": self._total_writes,
            "write_rate": self._total_writes / max(self._total_steps, 1),
            "avg_overwrite_conflict": self._overwrite_accumulator / max(self._total_steps, 1),
        }

    def reset_stats(self) -> None:
        """Reset cumulative statistics."""
        self._total_writes = 0
        self._total_steps = 0
        self._overwrite_accumulator = 0.0
