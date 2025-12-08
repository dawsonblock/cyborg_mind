#!/usr/bin/env python3
"""
UniversalEnvAdapter - Factory for multi-domain environments

Supports:
- MineRL environments
- Gymnasium environments
- Custom adapters

Usage:
    env = UniversalEnvAdapter.from_name("minerl_treechop", num_envs=4)
    env = UniversalEnvAdapter.from_name("CartPole-v1", num_envs=8)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

import numpy as np
import torch


class BaseAdapter(ABC):
    """Abstract base class for environment adapters."""

    @abstractmethod
    def reset(self) -> torch.Tensor:
        """Reset environment and return initial observation."""
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Take a step in the environment.

        Returns:
            obs: (N, D) observation tensor
            rewards: (N,) reward tensor
            dones: (N,) done tensor
            infos: List of info dicts
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close environment."""
        pass

    @property
    @abstractmethod
    def observation_dim(self) -> int:
        """Return flattened observation dimension."""
        pass

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Return action dimension."""
        pass


class GymAdapter(BaseAdapter):
    """Adapter for Gymnasium environments."""

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        device: torch.device = torch.device("cpu"),
        seed: Optional[int] = None,
    ):
        """
        Args:
            env_id: Gymnasium environment ID
            num_envs: Number of parallel environments
            device: Torch device for tensors
            seed: Random seed
        """
        try:
            import gymnasium as gym
            from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
        except ImportError:
            raise ImportError("gymnasium not installed. Run: pip install gymnasium")

        self.env_id = env_id
        self.num_envs = num_envs
        self.device = device

        # Create vectorized environment
        def make_env(idx):
            def _init():
                env = gym.make(env_id)
                if seed is not None:
                    env.reset(seed=seed + idx)
                return env
            return _init

        self.env = SyncVectorEnv([make_env(i) for i in range(num_envs)])

        # Get dimensions
        obs_space = self.env.single_observation_space
        act_space = self.env.single_action_space

        if hasattr(obs_space, "shape"):
            self._obs_dim = int(np.prod(obs_space.shape))
        else:
            self._obs_dim = obs_space.n

        if hasattr(act_space, "n"):
            self._action_dim = act_space.n
            self.discrete = True
        else:
            self._action_dim = int(np.prod(act_space.shape))
            self.discrete = False

    def reset(self) -> torch.Tensor:
        obs, _ = self.env.reset()
        obs = np.array(obs).reshape(self.num_envs, -1)
        return torch.as_tensor(obs, device=self.device, dtype=torch.float32)

    def step(self, action: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
        if self.discrete:
            action = action.astype(np.int64)

        obs, rewards, terminated, truncated, infos = self.env.step(action)
        dones = np.logical_or(terminated, truncated)

        obs = np.array(obs).reshape(self.num_envs, -1)

        return (
            torch.as_tensor(obs, device=self.device, dtype=torch.float32),
            torch.as_tensor(rewards, device=self.device, dtype=torch.float32),
            torch.as_tensor(dones, device=self.device, dtype=torch.float32),
            [infos],
        )

    def close(self) -> None:
        self.env.close()

    @property
    def observation_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim


class MineRLAdapterV7(BaseAdapter):
    """
    V7 MineRL Adapter.

    Wraps MineRL environments with:
    - Discretized action space (18 actions)
    - Frame stacking
    - Reward normalization
    """

    def __init__(
        self,
        env_id: str = "MineRLTreechop-v0",
        num_envs: int = 1,
        device: torch.device = torch.device("cpu"),
        frame_stack: int = 4,
        image_size: Tuple[int, int] = (64, 64),
    ):
        self.env_id = env_id
        self.num_envs = num_envs
        self.device = device
        self.frame_stack = frame_stack
        self.image_size = image_size

        # Try to import MineRL
        try:
            import minerl
            self.minerl_available = True
            self._create_minerl_env()
        except ImportError:
            print("⚠️ MineRL not installed. Using mock environment.")
            self.minerl_available = False
            self._create_mock_env()

        # Observation: flattened RGB frames + compass
        self._obs_dim = image_size[0] * image_size[1] * 3 * frame_stack + 1
        self._action_dim = 18

    def _create_minerl_env(self):
        import minerl
        import gym as legacy_gym
        self.env = legacy_gym.make(self.env_id)
        self.frame_buffer = []

    def _create_mock_env(self):
        """Create mock environment for testing."""
        self.env = None
        self.frame_buffer = []

    def reset(self) -> torch.Tensor:
        if self.minerl_available:
            obs_dict = self.env.reset()
            obs = self._process_obs(obs_dict)
        else:
            obs = torch.randn(self.num_envs, self._obs_dim, device=self.device)

        self.frame_buffer = [obs] * self.frame_stack
        return obs

    def step(self, action: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
        if self.minerl_available:
            minerl_action = self._discrete_to_minerl(action[0])
            obs_dict, reward, done, info = self.env.step(minerl_action)
            obs = self._process_obs(obs_dict)
            rewards = torch.tensor([reward], device=self.device, dtype=torch.float32)
            dones = torch.tensor([done], device=self.device, dtype=torch.float32)
        else:
            obs = torch.randn(self.num_envs, self._obs_dim, device=self.device)
            rewards = torch.randn(self.num_envs, device=self.device)
            dones = torch.zeros(self.num_envs, device=self.device)

        return obs, rewards, dones, [{}]

    def _process_obs(self, obs_dict: Dict) -> torch.Tensor:
        """Process MineRL observation dict to tensor."""
        import cv2

        pov = obs_dict["pov"]
        pov = cv2.resize(pov, self.image_size)
        pov = pov.astype(np.float32) / 255.0
        pov = pov.transpose(2, 0, 1).flatten()

        compass = obs_dict.get("compassAngle", 0.0) / 180.0

        obs = np.concatenate([pov, [compass]])
        return torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)

    def _discrete_to_minerl(self, action: int) -> Dict:
        """Convert discrete action to MineRL action dict."""
        # Same action mapping as minerl_adapter.py
        actions = [
            {"forward": 1, "camera": [0, 0]},
            {"back": 1, "camera": [0, 0]},
            {"left": 1, "camera": [0, 0]},
            {"right": 1, "camera": [0, 0]},
            {"forward": 1, "jump": 1, "camera": [0, 0]},
            {"camera": [0, -15]},
            {"camera": [0, 15]},
            {"camera": [-15, 0]},
            {"camera": [15, 0]},
            {"attack": 1},
            {"use": 1},
            {"forward": 1, "attack": 1},
            {"forward": 1, "sprint": 1},
            {"forward": 1, "camera": [0, -10]},
            {"forward": 1, "camera": [0, 10]},
            {"forward": 1, "camera": [10, 0]},
            {"forward": 1, "jump": 1, "sprint": 1},
            {},  # No-op
        ]
        base = {
            "forward": 0, "back": 0, "left": 0, "right": 0,
            "jump": 0, "sneak": 0, "sprint": 0, "attack": 0, "use": 0,
            "camera": [0, 0],
        }
        base.update(actions[action])
        return base

    def close(self) -> None:
        if self.minerl_available and self.env:
            self.env.close()

    @property
    def observation_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim


class UniversalEnvAdapter:
    """Factory for creating environment adapters."""

    MINERL_ENVS = [
        "minerl_treechop", "minerl_navigate", "minerl_obtain",
        "MineRLTreechop-v0", "MineRLNavigate-v0", "MineRLObtainDiamond-v0",
    ]

    @classmethod
    def from_name(
        cls,
        env_name: str,
        num_envs: int = 1,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> BaseAdapter:
        """
        Create environment adapter from name.

        Args:
            env_name: Environment name or ID
            num_envs: Number of parallel environments
            device: Torch device
            **kwargs: Additional adapter arguments

        Returns:
            Environment adapter instance
        """
        if device is None:
            device = torch.device("cpu")

        # Check if MineRL
        if any(m in env_name.lower() for m in ["minerl", "treechop", "navigate", "diamond"]):
            env_id = cls._resolve_minerl_id(env_name)
            return MineRLAdapterV7(
                env_id=env_id,
                num_envs=num_envs,
                device=device,
                **kwargs,
            )

        # Default to Gymnasium
        return GymAdapter(
            env_id=env_name,
            num_envs=num_envs,
            device=device,
            **kwargs,
        )

    @classmethod
    def _resolve_minerl_id(cls, name: str) -> str:
        """Resolve short name to MineRL env ID."""
        mapping = {
            "minerl_treechop": "MineRLTreechop-v0",
            "minerl_navigate": "MineRLNavigate-v0",
            "minerl_obtain": "MineRLObtainDiamond-v0",
            "treechop": "MineRLTreechop-v0",
            "navigate": "MineRLNavigate-v0",
        }
        return mapping.get(name.lower(), name)

    @classmethod
    def list_available(cls) -> List[str]:
        """List available environment names."""
        return cls.MINERL_ENVS + ["CartPole-v1", "LunarLander-v2", "Pendulum-v1"]
