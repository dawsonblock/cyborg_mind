import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

class PMM(nn.Module):
    """
    Honest Memory Engine v2.0 (PMM)
    
    Features:
    - Controlled Write Gate (target sparsity)
    - Zero Leakage (strict reset on done)
    - Deterministic Addressing (cosine, softmax)
    - Comprehensive Diagnostics
    """
    def __init__(self, memory_dim: int, num_slots: int, 
                 write_rate_target_inv: int = 2000,
                 gate_type: str = "soft",
                 temperature: float = 1.0,
                 sharpness: float = 2.0):
        super().__init__()
        self.memory_dim = memory_dim
        self.num_slots = num_slots
        self.write_rate_target = 1.0 / write_rate_target_inv
        self.gate_type = gate_type # "soft" or "hard"
        self.temperature = temperature
        self.sharpness = sharpness
        
        # Keys and Values initialized learnably? Or fixed? 
        # Usually internal memory slots are initialized to zero or learnable.
        # Here we will manage the memory state externally (passed in forward) for recurrence.
        # But we need projection layers for Read/Write heads.
        
        # Read Head
        self.query_proj = nn.Linear(memory_dim, memory_dim) # Query from hidden state
        
        # Write Head
        self.write_key_proj = nn.Linear(memory_dim, memory_dim) # Key to write
        self.write_val_proj = nn.Linear(memory_dim, memory_dim) # Value to write
        self.gate_proj = nn.Linear(memory_dim, 1) # Write Gate (scalar)
        
        # Diagnostics cache
        self.last_diagnostics = {}

    def forward(self, 
                current_memory: torch.Tensor, 
                hidden_state: torch.Tensor, 
                mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Single step forward.
        """
        # ... (implementation same as before, essentially)
        # Re-using code structure to avoid large diffs if possible, but simplest is to implement this logic.
        # See below for shared implementation.
        return self._forward_step(current_memory, hidden_state, mask)

    def _forward_step(self, current_memory, hidden_state, mask):
        # ... logic from previous forward ...
        
        # 1. READ
        query = self.query_proj(hidden_state)
        query_norm = F.normalize(query, p=2, dim=-1)
        memory_norm = F.normalize(current_memory, p=2, dim=-1)
        
        similarity = torch.sum(memory_norm * query_norm.unsqueeze(1), dim=-1)
        read_weights = F.softmax(similarity * self.sharpness, dim=-1)
        
        read_vector = torch.matmul(read_weights.unsqueeze(1), current_memory).squeeze(1)
        
        # 2. WRITE
        write_key = self.write_key_proj(hidden_state)
        write_val = self.write_val_proj(hidden_state)
        gate_logit = self.gate_proj(hidden_state)
        
        if self.gate_type == "hard":
            gate_open_soft = torch.sigmoid(gate_logit)
            gate_open = (gate_open_soft > 0.5).float() - gate_open_soft.detach() + gate_open_soft
        else:
            gate_open = torch.sigmoid(gate_logit)
            
        write_query_norm = F.normalize(write_key, p=2, dim=-1)
        write_similarity = torch.sum(memory_norm * write_query_norm.unsqueeze(1), dim=-1)
        write_weights = F.softmax(write_similarity * self.sharpness, dim=-1)
        
        write_op = gate_open.unsqueeze(1) * write_weights.unsqueeze(-1)
        next_memory = current_memory + write_op * (write_val.unsqueeze(1) - current_memory)
        
        # 3. LEAKAGE / MASK
        if mask is not None:
             mask_expanded = mask.unsqueeze(-1).expand_as(next_memory)
             next_memory = next_memory * mask_expanded
        
        # 4. LOGS
        gate_mean = gate_open.mean()
        sparsity_loss = (gate_mean - self.write_rate_target) ** 2
        
        logs = {
            "sparsity_loss": sparsity_loss,
            "write_strength": gate_mean.item(),
            "read_entropy": -torch.sum(read_weights * torch.log(read_weights + 1e-8), dim=-1).mean().item()
        }
        
        return read_vector, next_memory, logs

    def forward_sequence(self, init_memory: torch.Tensor, hidden_seq: torch.Tensor, masks: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
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
        
        # Recurrent Loop
        # This loop is Python-bound, could be optimized with TorchScript
        for t in range(L):
            h_t = hidden_seq[:, t, :]
            mask_t = masks[:, t, :] if masks is not None else None
            
            read_v, next_mem, logs = self._forward_step(curr_memory, h_t, mask_t)
            
            read_vectors.append(read_v)
            curr_memory = next_mem
            total_sparsity_loss += logs["sparsity_loss"]
            
        read_seq = torch.stack(read_vectors, dim=1) # (B, L, Dim)
        
        avg_logs = {
            "sparsity_loss": total_sparsity_loss / L
        }
        
        return read_seq, avg_logs


    def get_initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.num_slots, self.memory_dim, device=device)
