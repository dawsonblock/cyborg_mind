"""Mamba/GRU hybrid encoder for sequence processing."""

from typing import Tuple, Optional
import torch
import torch.nn as nn

from cyborg_rl.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    logger.warning("mamba-ssm not installed. Using GRU-only encoder.")


class GRUEncoder(nn.Module):
    """
    GRU-based sequence encoder.

    Provides a recurrent encoder for processing sequential observations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        """
        Initialize the GRU encoder.

        Args:
            input_dim: Input observation dimension.
            hidden_dim: GRU hidden dimension.
            latent_dim: Output latent dimension.
            num_layers: Number of GRU layers.
            dropout: Dropout rate between layers.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor [B, T, D] or [B, D].
            hidden: Optional hidden state [num_layers, B, H].

        Returns:
            Tuple of (latent [B, L], hidden [num_layers, B, H]).
        """
        # Handle non-sequential input [B, D] -> [B, 1, D]
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch_size = x.shape[0]

        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)

        x = self.input_proj(x)
        output, hidden = self.gru(x, hidden)

        # Take the last time step output for the latent representation
        latent = self.output_proj(output[:, -1, :])

        return latent, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Initialize hidden state.

        Args:
            batch_size: Batch size.
            device: Target device.

        Returns:
            torch.Tensor: Initial hidden state.
        """
        return torch.zeros(
            self.num_layers, batch_size, self.hidden_dim,
            device=device, dtype=torch.float32
        )


class MambaBlock(nn.Module):
    """
    Mamba SSM block for efficient sequence modeling.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        """
        Initialize the Mamba block.

        Args:
            d_model: Model dimension.
            d_state: SSM state dimension.
            d_conv: Convolution kernel size.
            expand: Expansion factor.
        """
        super().__init__()

        if not MAMBA_AVAILABLE:
            raise ImportError("MambaBlock requires mamba-ssm package.")

        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor [B, T, D].

        Returns:
            torch.Tensor: Output tensor [B, T, D].
        """
        return x + self.mamba(self.norm(x))


class MambaGRUEncoder(nn.Module):
    """
    Hybrid Mamba/GRU encoder.

    Combines the efficiency of Mamba SSM with GRU for sequence processing.
    Falls back to pure GRU if Mamba is not available.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_gru_layers: int = 2,
        use_mamba: bool = True,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        """
        Initialize the hybrid encoder.

        Args:
            input_dim: Input observation dimension.
            hidden_dim: Hidden dimension.
            latent_dim: Output latent dimension.
            num_gru_layers: Number of GRU layers.
            use_mamba: Whether to use Mamba blocks.
            mamba_d_state: Mamba state dimension.
            mamba_d_conv: Mamba convolution kernel size.
            mamba_expand: Mamba expansion factor.
            dropout: Dropout rate.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.use_mamba = use_mamba and MAMBA_AVAILABLE

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        if self.use_mamba:
            self.mamba = MambaBlock(
                d_model=hidden_dim,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
            )
        else:
            self.mamba = nn.Identity()

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0.0,
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        self.num_gru_layers = num_gru_layers

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor [B, T, D] or [B, D].
            hidden: Optional hidden state [num_layers, B, H].

        Returns:
            Tuple of (latent [B, L], hidden [num_layers, B, H]).
        """
        # Handle non-sequential input [B, D] -> [B, 1, D]
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch_size = x.shape[0]

        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)

        # Project input
        x = self.input_proj(x)

        # Apply Mamba block if enabled (sequence modeling)
        if self.use_mamba:
            x = self.mamba(x)

        # Apply GRU (recurrence/memory)
        output, hidden = self.gru(x, hidden)

        # Project output to latent space (take last step)
        latent = self.output_proj(output[:, -1, :])

        return latent, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Initialize hidden state.

        Args:
            batch_size: Batch size.
            device: Target device.

        Returns:
            torch.Tensor: Initial hidden state.
        """
        return torch.zeros(
            self.num_gru_layers, batch_size, self.hidden_dim,
            device=device, dtype=torch.float32
        )
