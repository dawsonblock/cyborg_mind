"""Pseudo-Mamba (Linear RNN) implementation for benchmarking.

This module implements a "Pseudo-Mamba" architecture which is essentially a 
Linear Recurrent Unit (LRU) or simplified S4/S6 layer implemented in pure PyTorch.
It serves as a baseline to compare against the optimized CUDA Mamba implementation.

Key features:
- Pure PyTorch implementation (no custom CUDA kernels)
- Linear recurrence for long-sequence modeling
- Input-dependent gating (like Mamba/S6)
"""

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from cyborg_rl.utils.logging import get_logger

logger = get_logger(__name__)


class PseudoMambaBlock(nn.Module):
    """
    A simplified Mamba-like block using pure PyTorch.
    
    Implements a gated linear recurrence:
    h_t = (1 - z_t) * h_{t-1} + z_t * x_t
    y_t = h_t * g_t
    
    Where z_t is the "forget" gate and g_t is the output gate.
    This mimics the selection mechanism of Mamba/S6 but with diagonal structure.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        
        # 1. Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # 2. Convolution (local context)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.act = nn.SiLU()
        
        # 3. State Space Model (SSM) parameters
        # A: State transition (diagonal) - initialized for stability
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # B, C: Input/Output dynamics (input-dependent in Mamba, simplified here)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)  # Predicts B and C
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True) # Predicts step size delta
        
        # 4. Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Cache for inference
        self.inference_cache = {}
        
    def forward(self, x: torch.Tensor, inference_params=None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, L, D]
            inference_params: Optional dict for state caching
            
        Returns:
            Output tensor [B, L, D]
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Project inputs
        # [B, L, D] -> [B, L, 2*D_inner]
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        
        # 2. Convolution
        # [B, L, D_inner] -> [B, D_inner, L]
        x_conv = x_inner.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = self.act(x_conv)
        # [B, D_inner, L] -> [B, L, D_inner]
        x_ssm = x_conv.transpose(1, 2)
        
        # 3. SSM (Simplified Linear Recurrence)
        # In a real Mamba, this uses a parallel scan. Here we use a sequential loop
        # for simplicity and "Pseudo" nature (benchmarking pure PyTorch speed).
        
        # Predict parameters
        # delta: [B, L, D_inner]
        dt = F.softplus(self.dt_proj(x_ssm))
        
        # B, C: [B, L, D_state]
        x_dbl = self.x_proj(x_ssm)
        B, C = x_dbl.chunk(2, dim=-1)
        
        # Discretize A
        # A: [D_inner, D_state]
        A = -torch.exp(self.A_log)
        # dA: [B, L, D_inner, D_state] = exp(dt * A)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        
        # Discretize B
        # dB: [B, L, D_inner, D_state]
        dB = B.unsqueeze(2) * dt.unsqueeze(-1)
        
        # Recurrence
        h = torch.zeros(batch_size, self.d_inner, self.d_state, device=x.device)
        ys = []
        
        # Check cache
        if inference_params is not None:
             # TODO: Implement proper caching for Pseudo-Mamba
             # For now, we just run sequential mode without cache optimization
             pass
             
        for t in range(seq_len):
            # h_t = dA_t * h_{t-1} + dB_t * x_t
            # x_t: [B, D_inner]
            xt = x_ssm[:, t, :]
            
            # Update state
            # [B, D_inner, D_state]
            h = dA[:, t] * h + dB[:, t] * xt.unsqueeze(-1)
            
            # Output
            # y_t = C_t * h_t + D * x_t
            # [B, D_inner]
            Ct = C[:, t, :]
            yt = torch.sum(h * Ct.unsqueeze(1), dim=-1) + self.D * xt
            ys.append(yt)
            
        y = torch.stack(ys, dim=1)
        
        # 4. Gating and Output
        y = y * F.silu(z)
        out = self.out_proj(y)
        out = self.dropout(out)
        
        return out + x  # Residual connection


class PseudoMambaEncoder(nn.Module):
    """
    Encoder using Pseudo-Mamba blocks.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.layers = nn.ModuleList([
            PseudoMambaBlock(
                d_model=hidden_dim,
                expand=2,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        
    def forward(self, x: torch.Tensor, hidden=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Handle non-sequential input
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = self.input_proj(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        
        # Take last step
        latent = self.output_proj(x[:, -1, :])
        
        # Return latent and None (stateless for now/managed internally)
        return latent, hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        return None
