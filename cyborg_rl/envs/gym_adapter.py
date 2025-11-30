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
        normalize_obs: bool = True,
        clip_obs: float = 10.0,
        max_episode_steps: Optional[int] = None,
    ) -> None:
        """
        Initialize Gym adapter.

        Args:
            env_name: Environment ID.
            device: Torch device.
            seed: Random seed.
            normalize_obs: Whether to normalize observations.
            clip_obs: Clipping range.
            max_episode_steps: Optional step limit override.
        """
        super().__init__(device, normalize_obs, clip_obs)
        
        self._env = gym.make(env_name)
        
        if max_episode_steps:
            self._env = gym.wrappers.TimeLimit(self._env, max_episode_steps=max_episode_steps)
            
        self._seed = seed
        
        # Space properties
        self._obs_dim = int(np.prod(self._env.observation_space.shape))
        
        if isinstance(self._env.action_space, gym.spaces.Discrete):
            self._action_dim = int(self._env.action_space.n)
            self._is_discrete = True
        elif isinstance(self._env.action_space, gym.spaces.Box):
            self._action_dim = int(np.prod(self._env.action_space.shape))
            self._is_discrete = False
        else:
            raise ValueError(f"Unsupported action space: {self._env.action_space}")

        logger.info(f"Initialized GymAdapter: {env_name}, obs={self._obs_dim}, act={self._action_dim}")

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
        obs, _ = self._env.reset(seed=self._seed)
        return self._to_tensor(obs)

    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, float, bool, bool, Dict[str, Any]]:
        # Convert action to format expected by Gym
        action_val = action.detach().cpu().numpy()
        
        if self._is_discrete:
            if action_val.ndim > 0:
                action_val = action_val.item()
            action_val = int(action_val)
        else:
            # Handle continuous actions (squeeze batch dim if present)
            action_val = np.squeeze(action_val)

        obs, reward, terminated, truncated, info = self._env.step(action_val)
        
        return (
            self._to_tensor(obs),
            float(reward),
            terminated,
            truncated,
            info
        )

    def close(self) -> None:
        self._env.close()
        
    def sample_action(self) -> torch.Tensor:
        """Sample random action for testing."""
        if self._is_discrete:
            act = np.random.randint(0, self._action_dim)
            return torch.tensor([act], device=self.device, dtype=torch.long)
        else:
            act = np.random.randn(self._action_dim)
            return torch.tensor(act, device=self.device, dtype=torch.float32)
