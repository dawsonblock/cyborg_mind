"""Gymnasium environment adapter."""

from typing import Any, Dict, Tuple, Optional
import numpy as np
import torch
import gymnasium as gym

from cyborg_rl.envs.base import BaseEnvAdapter
from cyborg_rl.utils.logging import get_logger

logger = get_logger(__name__)


class GymAdapter(BaseEnvAdapter):
    """
    Adapter for Gymnasium environments.

    Provides normalized, tensorized, device-aware observations
    and handles both discrete and continuous action spaces.
    """

    def __init__(
        self,
        env_name: str,
        device: torch.device,
        seed: Optional[int] = None,
        max_episode_steps: Optional[int] = None,
        normalize_obs: bool = True,
        clip_obs: float = 10.0,
    ) -> None:
        """
        Initialize the Gym adapter.

        Args:
            env_name: Gymnasium environment ID.
            device: Torch device for tensor operations.
            seed: Random seed for environment.
            max_episode_steps: Maximum steps per episode.
            normalize_obs: Whether to normalize observations.
            clip_obs: Observation clipping range.
        """
        super().__init__(device, normalize_obs, clip_obs)

        self.env_name = env_name
        self._seed = seed

        self._env = gym.make(
            env_name,
            max_episode_steps=max_episode_steps,
        )

        if seed is not None:
            self._env.action_space.seed(seed)

        self._observation_dim = self._compute_obs_dim()
        self._action_dim = self._compute_action_dim()
        self._is_discrete = isinstance(self._env.action_space, gym.spaces.Discrete)

        logger.info(
            f"Initialized GymAdapter: {env_name}, "
            f"obs_dim={self._observation_dim}, action_dim={self._action_dim}, "
            f"discrete={self._is_discrete}"
        )

    def _compute_obs_dim(self) -> int:
        """Compute observation dimension from space."""
        space = self._env.observation_space
        if isinstance(space, gym.spaces.Box):
            return int(np.prod(space.shape))
        elif isinstance(space, gym.spaces.Discrete):
            return int(space.n)
        else:
            raise ValueError(f"Unsupported observation space: {type(space)}")

    def _compute_action_dim(self) -> int:
        """Compute action dimension from space."""
        space = self._env.action_space
        if isinstance(space, gym.spaces.Discrete):
            return int(space.n)
        elif isinstance(space, gym.spaces.Box):
            return int(np.prod(space.shape))
        else:
            raise ValueError(f"Unsupported action space: {type(space)}")

    @property
    def observation_dim(self) -> int:
        """Return the observation dimension."""
        return self._observation_dim

    @property
    def action_dim(self) -> int:
        """Return the action dimension."""
        return self._action_dim

    @property
    def is_discrete(self) -> bool:
        """Return whether action space is discrete."""
        return self._is_discrete

    def reset(self) -> torch.Tensor:
        """
        Reset the environment.

        Returns:
            torch.Tensor: Initial observation tensor on device.
        """
        obs, _ = self._env.reset(seed=self._seed)
        return self._to_tensor(np.asarray(obs).flatten())

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
        action_np = self._action_to_numpy(action)
        obs, reward, terminated, truncated, info = self._env.step(action_np)

        obs_tensor = self._to_tensor(np.asarray(obs).flatten())
        return obs_tensor, float(reward), terminated, truncated, info

    def close(self) -> None:
        """Close the environment."""
        self._env.close()

    def sample_action(self) -> torch.Tensor:
        """
        Sample a random action.

        Returns:
            torch.Tensor: Random action tensor.
        """
        action = self._env.action_space.sample()
        if self._is_discrete:
            return torch.tensor([action], device=self.device, dtype=torch.long)
        return torch.tensor(action, device=self.device, dtype=torch.float32)
