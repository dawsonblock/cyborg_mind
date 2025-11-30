"""Value head implementation for critic networks."""

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """
    Value head for estimating state values.

    Used as the critic in actor-critic algorithms like PPO.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        """
        Initialize the value head.

        Args:
            input_dim: Input dimension (latent dim).
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with orthogonal initialization."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute state value.

        Args:
            x: Latent state [B, D].

        Returns:
            torch.Tensor: State value [B, 1].
        """
        return self.network(x)
