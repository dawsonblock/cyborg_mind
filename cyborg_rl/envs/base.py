"""Base environment adapter interface."""

from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict, Optional, Union
import numpy as np
import torch


class BaseEnvAdapter(ABC):
    """
    Abstract base class for environment adapters.

    Standardizes interaction between Gym, MineRL, and other environments.
    """

    def __init__(
        self,
        device: torch.device,
        normalize_obs: bool = True,
        clip_obs: float = 10.0,
    ) -> None:
        """
        Initialize base adapter.

        Args:
            device: Torch device.
            normalize_obs: Whether to normalize observations.
            clip_obs: Observation clipping range.
        """
        self.device = device
        self.normalize_obs = normalize_obs
        self.clip_obs = clip_obs
        self._obs_mean: Optional[np.ndarray] = None
        self._obs_std: Optional[np.ndarray] = None

    @property
    @abstractmethod
    def observation_dim(self) -> int:
        """Return observation dimension."""
        pass

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Return action dimension."""
        pass

    @property
    @abstractmethod
    def is_discrete(self) -> bool:
        """Return whether action space is discrete."""
        pass

    @abstractmethod
    def reset(self) -> torch.Tensor:
        """Reset environment and return initial observation tensor."""
        pass

    @abstractmethod
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, bool, Dict[str, Any]]:
        """
        Take step in environment.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close environment resources."""
        pass

    def _to_tensor(self, x: Union[np.ndarray, float, int]) -> torch.Tensor:
        """Convert numpy array or scalar to tensor on device."""
        if isinstance(x, (float, int)):
            x = np.array([x], dtype=np.float32)

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        return x.to(self.device)

    def sample_action(self) -> torch.Tensor:
        """Sample a random action (for testing)."""
        pass
