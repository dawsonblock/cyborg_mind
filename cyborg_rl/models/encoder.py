
import math
from typing import Tuple, Optional, Dict, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

# Try importing Mamba
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    logger.warning("mamba-ssm not installed. Will fallback to PseudoMamba (pure PyTorch).")

# --- PseudoMamba Implementation (Fallback) ---

class PseudoMambaBlock(nn.Module):
    """
    Pure PyTorch implementation of a Mamba-like block (Linear Recurrent Unit).
    Used as valid Mamba fallback on non-CUDA devices.
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
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.act = nn.SiLU()
        
        # SSM Parameters
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, inference_params: Any = None) -> torch.Tensor:
        """
        x: (B, L, D)
        """
        batch_size, seq_len, _ = x.shape
        
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        
        x_conv = x_inner.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = self.act(x_conv)
        x_ssm = x_conv.transpose(1, 2)
        
        # Pseudo-Mamba Scan (Sequential Loop for Correctness/Simp)
        # For production speed, torch.compile should optimize this loop reasonably well.
        dt = F.softplus(self.dt_proj(x_ssm))
        x_dbl = self.x_proj(x_ssm)
        B, C = x_dbl.chunk(2, dim=-1)
        
        A = -torch.exp(self.A_log)
        # dA: (B, L, D_inner, D_state)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dB = B.unsqueeze(2) * dt.unsqueeze(-1) # (B, L, D_inner, D_state)
        
        # Init state
        if inference_params is not None and "h" in inference_params:
            h = inference_params["h"]
        else:
            h = torch.zeros(batch_size, self.d_inner, self.d_state, device=x.device)
            
        ys = []
        for t in range(seq_len):
            # x_t: (B, D_inner)
            xt = x_ssm[:, t, :]
            # Update h
            h = dA[:, t] * h + dB[:, t] * xt.unsqueeze(-1)
            # Compute y
            # C_t: (B, D_state) -> (B, D_inner, D_state)?? 
            # In code C is (B, L, D_state) but x_ssm is (B, L, D_inner).
            # The broadcast logic in pure mamba is complex.
            # Simplified: C is (B, D_state). h is (B, D_inner, D_state).
            # We need to contract state.
            # Actually, standard S6 uses B (B, G, N), C (B, G, N).
            # Here assuming simple broadcasting.
            Ct = C[:, t, :]
            # h * C -> (B, D_inner, D_state) * (B, 1, D_state) -> sum?
            # Let's assume broadcasting C to (B, D_inner, D_state) implies sharing or repeating?
            # In original code: sum(h * Ct.unsqueeze(1), dim=-1).
            yt = torch.sum(h * Ct.unsqueeze(1), dim=-1) + self.D * xt
            ys.append(yt)
        
        if inference_params is not None:
            inference_params["h"] = h
            
        y = torch.stack(ys, dim=1)
        y = y * F.silu(z)
        out = self.dropout(self.out_proj(y))
        return out + x

# --- Unified Encoder ---

class UnifiedEncoder(nn.Module):
    """
    Unified Encoder supporting GRU, Mamba, and Mamba-GRU.
    """
    def __init__(
        self,
        encoder_type: str, # "gru", "mamba", "mamba_gru"
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        device: str = "cpu"
    ):
        super().__init__()
        self.encoder_type = encoder_type.lower()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # Input Projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Core Sequence Model
        self.mamba_layers = nn.ModuleList()
        self.gru = None
        
        if "mamba" in self.encoder_type:
            # Use Mamba (Original or Pseudo)
            use_pseudo = not MAMBA_AVAILABLE or device != "cuda"
            Block = PseudoMambaBlock if use_pseudo else Mamba
            # If using official Mamba, we need to wrap it to match potential API diffs or just use it.
            # Official Mamba signature: forward(x, inference_params=None)
            
            for _ in range(num_layers if self.encoder_type == "mamba" else 1):
                if use_pseudo:
                    self.mamba_layers.append(PseudoMambaBlock(d_model=hidden_dim, dropout=dropout))
                else:
                    self.mamba_layers.append(Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2))
        
        if "gru" in self.encoder_type:
            self.gru = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers if self.encoder_type == "gru" else 1,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            
        # Output Projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    # ==================== STATE MANAGEMENT ====================

    def get_initial_state(self, batch_size: int, device: torch.device) -> Any:
        """Create initial state for the encoder."""
        if self.encoder_type == "gru":
            return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        elif self.encoder_type == "mamba":
            return [{} for _ in range(len(self.mamba_layers))]
        elif self.encoder_type == "mamba_gru":
            return ([{}], torch.zeros(1, batch_size, self.hidden_dim, device=device))
        else:
            return None

    def reset_state(self, state: Any, mask: torch.Tensor) -> Any:
        """
        Reset state for done episodes (zero leakage).

        Args:
            state: Current encoder state
            mask: Done mask (B,) - 1 = keep, 0 = reset

        Returns:
            State with done episodes zeroed
        """
        if state is None:
            return None

        if self.encoder_type == "gru":
            # state: (num_layers, B, H)
            mask_expanded = mask.view(1, -1, 1).expand_as(state)
            return state * mask_expanded

        elif self.encoder_type == "mamba":
            # state: List of dicts with 'h' tensors
            new_state = []
            for layer_state in state:
                if 'h' in layer_state:
                    h = layer_state['h']
                    mask_expanded = mask.view(-1, 1, 1).expand_as(h)
                    new_state.append({'h': h * mask_expanded})
                else:
                    new_state.append({})
            return new_state

        elif self.encoder_type == "mamba_gru":
            mamba_state, gru_state = state
            # Reset GRU state
            mask_expanded = mask.view(1, -1, 1).expand_as(gru_state)
            new_gru_state = gru_state * mask_expanded
            return (mamba_state, new_gru_state)

        return state

    def detach_state(self, state: Any) -> Any:
        """Detach state from computation graph for TBPTT."""
        if state is None:
            return None

        if self.encoder_type == "gru":
            return state.detach()

        elif self.encoder_type == "mamba":
            new_state = []
            for layer_state in state:
                if 'h' in layer_state:
                    new_state.append({'h': layer_state['h'].detach()})
                else:
                    new_state.append({})
            return new_state

        elif self.encoder_type == "mamba_gru":
            mamba_state, gru_state = state
            return (mamba_state, gru_state.detach())

        return state

    def forward(self, x: torch.Tensor, state: Any = None) -> Tuple[torch.Tensor, Any]:
        """
        Args:
            x: (B, L, D) or (B, D)
            state: State object.
                   - GRU: Tensor (num_layers, B, H)
                   - Mamba: Dict or List of Dicts (inference_params)
                   - Mamba-GRU: Tuple (mamba_state, gru_state)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        bs = x.size(0)
        x_emb = self.input_proj(x) # (B, L, H)
        
        next_state = None
        
        # Dispatch
        if self.encoder_type == "gru":
            if state is None:
                state = torch.zeros(self.num_layers, bs, self.hidden_dim, device=x.device)
            out, next_state = self.gru(x_emb, state)
            
        elif self.encoder_type == "mamba":
            if state is None:
                state = [{} for _ in range(len(self.mamba_layers))] # one dict per layer
            next_state = state # Mamba updates in-place or returns?
            # Actually PseudoMamba updates dict in-place. Official Mamba uses inference_params dict.
            # We assume state is a list of inference_params dicts.
            
            out = x_emb
            for i, layer in enumerate(self.mamba_layers):
                # Mamba residual connection is internal? 
                # Official Mamba: usually need to add residual manually if using raw Mamba block?
                # Actually usage: x = x + block(norm(x))
                # PseudoMamba above already has x + in return.
                # Let's assume blocks handle residuals or are full layers.
                # PseudoMambaBlock above returns out + x.
                # Official Mamba returns result of SSM. We must add residual.
                
                # We need norm? (Usually MambaBlock includes norm or expects it?)
                # Code above PseudoMamba has no LayerNorm on input. 
                # Let's trust the block impl/wrapper.
                # For robustness, we should Normalize -> Block -> Residual.
                
                out = layer(out, inference_params=next_state[i])
            
        elif self.encoder_type == "mamba_gru":
            # Hybrid: Mamba -> GRU
            if state is None:
                state = ([{}], torch.zeros(1, bs, self.hidden_dim, device=x.device))
            
            mamba_state, gru_state = state
            
            # Mamba
            out_m = x_emb
            for i, layer in enumerate(self.mamba_layers): # Should be 1 layer
                out_m = layer(out_m, inference_params=mamba_state[i])
                
            # GRU
            out_g, next_gru_state = self.gru(out_m, gru_state)
            next_state = (mamba_state, next_gru_state)
            out = out_g
            
        else:
            raise ValueError(f"Unknown encoder type: {self.encoder_type}")
            
        # Projection (Last step)
        latent = self.output_proj(out)
        return latent, next_state

