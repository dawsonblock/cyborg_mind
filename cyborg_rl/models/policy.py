"""Policy head implementations for discrete and continuous actions."""

from typing import Tuple
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class PolicyHead(nn.Module):
    """Base policy head with shared architecture."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        """
        Initialize the policy head base.

        Args:
            input_dim: Input dimension (latent dim).
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through shared layers."""
        return self.shared(x)


class DiscretePolicy(nn.Module):
    """Discrete action policy for environments with finite action spaces."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        """
        Initialize the discrete policy.

        Args:
            input_dim: Input dimension (latent dim).
            action_dim: Number of discrete actions.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()

        self.base = PolicyHead(input_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)

        nn.init.orthogonal_(self.action_head.weight, gain=0.01)
        nn.init.zeros_(self.action_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute action logits.

        Args:
            x: Latent state [B, D].

        Returns:
            torch.Tensor: Action logits [B, A].
        """
        features = self.base(x)
        return self.action_head(features)

    def get_distribution(self, x: torch.Tensor) -> Categorical:
        """
        Get action distribution.

        Args:
            x: Latent state [B, D].

        Returns:
            Categorical: Action distribution.
        """
        logits = self.forward(x)
        return Categorical(logits=logits)

    def sample(
        self, x: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            x: Latent state [B, D].
            deterministic: If True, return argmax action.

        Returns:
            Tuple of (action [B], log_prob [B]).
        """
        dist = self.get_distribution(x)

        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate(self, x: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of actions.

        Args:
            x: Latent state [B, D].
            action: Actions to evaluate [B].

        Returns:
            Tuple of (log_prob [B], entropy [B]).
        """
        dist = self.get_distribution(x)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy


class ContinuousPolicy(nn.Module):
    """Continuous action policy for environments with continuous action spaces."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ) -> None:
        """
        Initialize the continuous policy.

        Args:
            input_dim: Input dimension (latent dim).
            action_dim: Action dimension.
            hidden_dim: Hidden layer dimension.
            log_std_min: Minimum log standard deviation.
            log_std_max: Maximum log standard deviation.
        """
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_dim = action_dim

        self.base = PolicyHead(input_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
        nn.init.zeros_(self.log_std_head.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute action mean and log std.

        Args:
            x: Latent state [B, D].

        Returns:
            Tuple of (mean [B, A], log_std [B, A]).
        """
        features = self.base(x)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def get_distribution(self, x: torch.Tensor) -> Normal:
        """
        Get action distribution.

        Args:
            x: Latent state [B, D].

        Returns:
            Normal: Action distribution.
        """
        mean, log_std = self.forward(x)
        std = log_std.exp()
        return Normal(mean, std)

    def sample(
        self, x: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            x: Latent state [B, D].
            deterministic: If True, return mean action.

        Returns:
            Tuple of (action [B, A], log_prob [B]).
        """
        dist = self.get_distribution(x)

        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate(self, x: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of actions.

        Args:
            x: Latent state [B, D].
            action: Actions to evaluate [B, A].

        Returns:
            Tuple of (log_prob [B], entropy [B]).
        """
        dist = self.get_distribution(x)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy
