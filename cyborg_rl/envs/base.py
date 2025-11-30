"""Base environment adapter interface."""

from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict, Optional
import numpy as np
import torch


class BaseEnvAdapter(ABC):
    """
    Abstract base class for environment adapters.

    Provides a unified interface for different environment types
    with normalized, tensorized, device-aware observations.
    """

    def __init__(
        self,
        device: torch.device,
        normalize_obs: bool = True,
        clip_obs: float = 10.0,
    ) -> None:
        """
        Initialize the environment adapter.

        Args:
            device: Torch device for tensor operations.
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
        """Return the observation dimension."""
        pass

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Return the action dimension."""
        pass

    @property
    @abstractmethod
    def is_discrete(self) -> bool:
        """Return whether action space is discrete."""
        pass

    @abstractmethod
    def reset(self) -> torch.Tensor:
        """
        Reset the environment.

        Returns:
            torch.Tensor: Initial observation tensor on device.
        """
        pass

    @abstractmethod
    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action: Action tensor.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the environment."""
        pass

    def _to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """
        Convert numpy observation to tensor on device.

        Args:
            obs: Numpy observation array.

        Returns:
            torch.Tensor: Observation tensor on device.
        """
        if self.normalize_obs and self._obs_mean is not None:
            obs = (obs - self._obs_mean) / (self._obs_std + 1e-8)
            obs = np.clip(obs, -self.clip_obs, self.clip_obs)

        return torch.from_numpy(obs.astype(np.float32)).to(self.device)

    def _action_to_numpy(self, action: torch.Tensor) -> np.ndarray:
        """
        Convert action tensor to numpy for environment.

        Args:
            action: Action tensor.

        Returns:
            np.ndarray or int: Action for environment.
        """
        action_np = action.detach().cpu().numpy()
        if self.is_discrete:
            return int(action_np.item()) if action_np.ndim == 0 else int(action_np[0])
        return action_np

    def update_obs_stats(self, obs_batch: np.ndarray) -> None:
        """
        Update running observation statistics for normalization.

        Args:
            obs_batch: Batch of observations.
        """
        if self._obs_mean is None:
            self._obs_mean = np.mean(obs_batch, axis=0)
            self._obs_std = np.std(obs_batch, axis=0) + 1e-8
        else:
            batch_mean = np.mean(obs_batch, axis=0)
            batch_std = np.std(obs_batch, axis=0)
            alpha = 0.01
            self._obs_mean = (1 - alpha) * self._obs_mean + alpha * batch_mean
            self._obs_std = (1 - alpha) * self._obs_std + alpha * batch_std
