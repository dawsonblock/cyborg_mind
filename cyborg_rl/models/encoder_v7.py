#!/usr/bin/env python3
"""
EncoderV7 - Multi-Mode Recurrent Encoder

Supports:
- Official Mamba SSM (CUDA)
- PseudoMamba fallback (CPU/MPS)
- GRU baseline
- Hybrid (Mamba → GRU residual)
- Fusion (concatenate Mamba + GRU features)

All outputs are (batch, seq, dim) with explicit state passing.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== MAMBA AVAILABILITY ====================
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("mamba-ssm not installed. Official Mamba unavailable.")


# ==================== PSEUDO MAMBA (FALLBACK) ====================
class PseudoMambaBlock(nn.Module):
    """
    Pure PyTorch Mamba approximation for non-CUDA systems.
    Uses GRU + Conv1d to approximate Mamba behavior.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Depthwise conv for local context
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        # Simple GRU for state tracking (replaces SSM scan)
        self.gru = nn.GRU(
            input_size=self.d_inner,
            hidden_size=self.d_inner,
            num_layers=1,
            batch_first=True,
        )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: Input tensor (B, L, D)
            state: Optional state dict with 'h' tensor

        Returns:
            output: (B, L, D)
            new_state: Dict with 'h' tensor
        """
        B, L, D = x.shape
        residual = x

        # Initialize state
        if state is None or "h" not in state:
            h = torch.zeros(1, B, self.d_inner, device=x.device, dtype=x.dtype)
        else:
            h = state["h"]

        # Input projection and split
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        # Conv for local context
        x_conv = x_in.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        # GRU for temporal dynamics
        y, h_new = self.gru(x_conv, h)

        # Gate and output
        y = y * F.silu(z)
        y = self.out_proj(y)
        y = self.dropout(y)

        # Residual
        output = self.norm(y + residual)

        return output, {"h": h_new}


# ==================== ENCODER V7 ====================
class EncoderV7(nn.Module):
    """
    Multi-mode recurrent encoder for V7 training system.

    Modes:
        - "gru": Standard GRU encoder
        - "mamba": Official Mamba (CUDA) or PseudoMamba (fallback)
        - "hybrid": Mamba → GRU with residual connection
        - "fusion": Concatenate Mamba + GRU features

    All modes return (batch, seq, dim) with explicit state management.
    """

    VALID_MODES = ["gru", "mamba", "hybrid", "fusion"]

    def __init__(
        self,
        mode: str = "mamba",
        input_dim: int = 512,
        hidden_dim: int = 384,
        latent_dim: int = 256,
        num_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        use_cuda_mamba: bool = True,
    ) -> None:
        """
        Args:
            mode: Encoder mode (gru, mamba, hybrid, fusion)
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            latent_dim: Output latent dimension
            num_layers: Number of recurrent layers
            d_state: SSM state dimension (Mamba)
            d_conv: Conv kernel size (Mamba)
            expand: Expansion factor (Mamba)
            dropout: Dropout rate
            use_cuda_mamba: Use official Mamba when available
        """
        super().__init__()

        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Choose from {self.VALID_MODES}")

        self.mode = mode
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.use_cuda_mamba = use_cuda_mamba and MAMBA_AVAILABLE

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Build encoder based on mode
        self._build_encoder(hidden_dim, num_layers, d_state, d_conv, expand, dropout)

        # Output projection
        output_dim = hidden_dim * 2 if mode == "fusion" else hidden_dim
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        self._init_weights()

    def _build_encoder(
        self,
        hidden_dim: int,
        num_layers: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
    ) -> None:
        """Build encoder layers based on mode."""

        if self.mode == "gru":
            self.gru = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.mamba_layers = None

        elif self.mode == "mamba":
            self.gru = None
            self.mamba_layers = nn.ModuleList()
            for _ in range(num_layers):
                if self.use_cuda_mamba:
                    self.mamba_layers.append(
                        Mamba(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
                    )
                else:
                    self.mamba_layers.append(
                        PseudoMambaBlock(hidden_dim, d_state, d_conv, expand, dropout)
                    )

        elif self.mode == "hybrid":
            # Mamba first, then GRU
            self.mamba_layers = nn.ModuleList()
            for _ in range(max(1, num_layers - 1)):
                if self.use_cuda_mamba:
                    self.mamba_layers.append(
                        Mamba(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
                    )
                else:
                    self.mamba_layers.append(
                        PseudoMambaBlock(hidden_dim, d_state, d_conv, expand, dropout)
                    )

            self.gru = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
            )
            self.residual_proj = nn.Linear(hidden_dim, hidden_dim)

        elif self.mode == "fusion":
            # Parallel Mamba + GRU, concatenate outputs
            self.mamba_layers = nn.ModuleList()
            for _ in range(num_layers):
                if self.use_cuda_mamba:
                    self.mamba_layers.append(
                        Mamba(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
                    )
                else:
                    self.mamba_layers.append(
                        PseudoMambaBlock(hidden_dim, d_state, d_conv, expand, dropout)
                    )

            self.gru = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

    def _init_weights(self) -> None:
        """Initialize weights with orthogonal initialization."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.orthogonal_(param, gain=1.0)
            elif "bias" in name:
                nn.init.zeros_(param)

    def get_initial_state(
        self, batch_size: int, device: torch.device
    ) -> Dict[str, Any]:
        """
        Create initial state for the encoder.

        Returns:
            State dict with 'gru' and/or 'mamba' states
        """
        state = {}

        if self.gru is not None:
            num_layers = self.gru.num_layers
            state["gru"] = torch.zeros(
                num_layers, batch_size, self.hidden_dim, device=device
            )

        if self.mamba_layers is not None:
            state["mamba"] = [{} for _ in range(len(self.mamba_layers))]

        return state

    def reset_states(
        self, state: Dict[str, Any], mask: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Reset states for done episodes.

        Args:
            state: Current state dict
            mask: Done mask (B,) where 1 = keep, 0 = reset

        Returns:
            New state with done episodes zeroed
        """
        new_state = {}

        if "gru" in state and state["gru"] is not None:
            # GRU state: (num_layers, B, H)
            gru_state = state["gru"]
            mask_expanded = mask.view(1, -1, 1).expand_as(gru_state)
            new_state["gru"] = gru_state * mask_expanded

        if "mamba" in state and state["mamba"] is not None:
            new_mamba = []
            for layer_state in state["mamba"]:
                if "h" in layer_state:
                    h = layer_state["h"]
                    # h shape: (B, d_inner, d_state)
                    mask_expanded = mask.view(-1, 1, 1).expand_as(h)
                    new_mamba.append({"h": h * mask_expanded})
                else:
                    new_mamba.append({})
            new_state["mamba"] = new_mamba

        return new_state

    def detach_states(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Detach states from computation graph for TBPTT."""
        new_state = {}

        if "gru" in state and state["gru"] is not None:
            new_state["gru"] = state["gru"].detach()

        if "mamba" in state and state["mamba"] is not None:
            new_mamba = []
            for layer_state in state["mamba"]:
                if "h" in layer_state:
                    new_mamba.append({"h": layer_state["h"].detach()})
                else:
                    new_mamba.append({})
            new_state["mamba"] = new_mamba

        return new_state

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through encoder.

        Args:
            x: Input tensor (B, L, D) or (B, D)
            state: Optional state dict

        Returns:
            output: Latent tensor (B, L, latent_dim)
            new_state: Updated state dict
        """
        # Handle single-step input
        squeeze_output = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_output = True

        B, L, _ = x.shape

        # Initialize state if needed
        if state is None:
            state = self.get_initial_state(B, x.device)

        # Input projection
        h = self.input_proj(x)

        # Forward based on mode
        if self.mode == "gru":
            new_state = {}
            gru_state = state.get("gru")
            h, new_gru_state = self.gru(h, gru_state)
            new_state["gru"] = new_gru_state

        elif self.mode == "mamba":
            new_state = {"mamba": []}
            mamba_states = state.get("mamba", [{} for _ in self.mamba_layers])

            for i, layer in enumerate(self.mamba_layers):
                if self.use_cuda_mamba:
                    h = layer(h)
                    new_state["mamba"].append({})
                else:
                    h, layer_state = layer(h, mamba_states[i])
                    new_state["mamba"].append(layer_state)

        elif self.mode == "hybrid":
            new_state = {"mamba": [], "gru": None}
            mamba_states = state.get("mamba", [{} for _ in self.mamba_layers])

            # Mamba layers
            residual = h
            for i, layer in enumerate(self.mamba_layers):
                if self.use_cuda_mamba:
                    h = layer(h)
                    new_state["mamba"].append({})
                else:
                    h, layer_state = layer(h, mamba_states[i])
                    new_state["mamba"].append(layer_state)

            # GRU with residual
            gru_state = state.get("gru")
            h, new_gru_state = self.gru(h, gru_state)
            h = h + self.residual_proj(residual)
            new_state["gru"] = new_gru_state

        elif self.mode == "fusion":
            new_state = {"mamba": [], "gru": None}
            mamba_states = state.get("mamba", [{} for _ in self.mamba_layers])

            # Mamba path
            h_mamba = h
            for i, layer in enumerate(self.mamba_layers):
                if self.use_cuda_mamba:
                    h_mamba = layer(h_mamba)
                    new_state["mamba"].append({})
                else:
                    h_mamba, layer_state = layer(h_mamba, mamba_states[i])
                    new_state["mamba"].append(layer_state)

            # GRU path
            gru_state = state.get("gru")
            h_gru, new_gru_state = self.gru(h, gru_state)
            new_state["gru"] = new_gru_state

            # Fuse
            h = torch.cat([h_mamba, h_gru], dim=-1)

        # Output projection
        output = self.output_proj(h)

        if squeeze_output:
            output = output.squeeze(1)

        return output, new_state

    def __repr__(self) -> str:
        return (
            f"EncoderV7(mode={self.mode}, hidden_dim={self.hidden_dim}, "
            f"latent_dim={self.latent_dim}, num_layers={self.num_layers}, "
            f"cuda_mamba={self.use_cuda_mamba})"
        )
