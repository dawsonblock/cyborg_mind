"""Gymnasium environment adapter."""

from typing import Any, Dict, Tuple, Optional
import gymnasium as gym
import numpy as np
import torch

from cyborg_rl.envs.base import BaseEnvAdapter
from cyborg_rl.utils.logging import get_logger

logger = get_logger(__name__)


class GymAdapter(BaseEnvAdapter):
    """Adapter for standard Gymnasium environments."""

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
        Initialize Gym adapter.

        Args:
            env_name: Gym environment ID.
            device: Torch device.
            seed: Random seed.
            max_episode_steps: Optional step limit override.
            normalize_obs: Whether to normalize observations.
            clip_obs: Observation clipping range.
        """
        super().__init__(device, normalize_obs, clip_obs)

        self.env_name = env_name
        self._env = gym.make(env_name)

        if max_episode_steps:
            self._env = gym.wrappers.TimeLimit(self._env, max_episode_steps=max_episode_steps)

        if seed is not None:
            self._env.reset(seed=seed)
            self._env.action_space.seed(seed)

        self._is_discrete = isinstance(self._env.action_space, gym.spaces.Discrete)

        # Determine dimensions
        self._obs_dim = int(np.prod(self._env.observation_space.shape))
        if self._is_discrete:
            self._action_dim = int(self._env.action_space.n)
        else:
            self._action_dim = int(np.prod(self._env.action_space.shape))

        logger.info(
            f"Initialized GymAdapter: {env_name}, "
            f"obs_dim={self._obs_dim}, action_dim={self._action_dim}, "
            f"discrete={self._is_discrete}"
        )

    @property
    def observation_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def is_discrete(self) -> bool:
        return self._is_discrete

    def reset(self) -> torch.Tensor:
        obs, _ = self._env.reset()
        return self._to_tensor(obs.flatten())

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, bool, Dict[str, Any]]:
        action_np = action.detach().cpu().numpy()

        if self._is_discrete:
            if action_np.ndim > 0:
                action_val = int(action_np.item())
            else:
                action_val = int(action_np)
        else:
            action_val = action_np

        obs, reward, terminated, truncated, info = self._env.step(action_val)

        return (self._to_tensor(obs.flatten()), float(reward), terminated, truncated, info)

    def close(self) -> None:
        self._env.close()

    def sample_action(self) -> torch.Tensor:
        if self._is_discrete:
            action = self._env.action_space.sample()
            return torch.tensor([action], device=self.device, dtype=torch.long)
        else:
            action = self._env.action_space.sample()
            return torch.tensor(action, device=self.device, dtype=torch.float32)
