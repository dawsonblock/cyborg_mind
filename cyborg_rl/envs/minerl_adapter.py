"""MineRL environment adapter."""

from typing import Any, Dict, Tuple, Optional, List
import numpy as np
import torch

from cyborg_rl.envs.base import BaseEnvAdapter
from cyborg_rl.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import minerl
    MINERL_AVAILABLE = True
except ImportError:
    MINERL_AVAILABLE = False
    logger.warning("MineRL not installed. MineRLAdapter will not be available.")


class MineRLAdapter(BaseEnvAdapter):
    """
    Adapter for MineRL environments.

    Handles the complex observation and action spaces of MineRL
    with flattening and discretization of actions.
    """

    DISCRETE_ACTIONS: List[Dict[str, Any]] = [
        {"forward": 1, "back": 0, "left": 0, "right": 0, "jump": 0, "camera": [0, 0]},
        {"forward": 0, "back": 1, "left": 0, "right": 0, "jump": 0, "camera": [0, 0]},
        {"forward": 0, "back": 0, "left": 1, "right": 0, "jump": 0, "camera": [0, 0]},
        {"forward": 0, "back": 0, "left": 0, "right": 1, "jump": 0, "camera": [0, 0]},
        {"forward": 1, "back": 0, "left": 0, "right": 0, "jump": 1, "camera": [0, 0]},
        {"forward": 1, "back": 0, "left": 0, "right": 0, "jump": 0, "camera": [0, -15]},
        {"forward": 1, "back": 0, "left": 0, "right": 0, "jump": 0, "camera": [0, 15]},
        {"forward": 1, "back": 0, "left": 0, "right": 0, "jump": 0, "camera": [-15, 0]},
        {"forward": 1, "back": 0, "left": 0, "right": 0, "jump": 0, "camera": [15, 0]},
        {"forward": 0, "back": 0, "left": 0, "right": 0, "jump": 0, "camera": [0, 0]},
    ]

    def __init__(
        self,
        env_name: str = "MineRLNavigateDense-v0",
        device: torch.device = torch.device("cpu"),
        seed: Optional[int] = None,
        normalize_obs: bool = True,
        clip_obs: float = 10.0,
        image_size: Tuple[int, int] = (64, 64),
    ) -> None:
        """
        Initialize the MineRL adapter.

        Args:
            env_name: MineRL environment ID.
            device: Torch device for tensor operations.
            seed: Random seed for environment.
            normalize_obs: Whether to normalize observations.
            clip_obs: Observation clipping range.
            image_size: Target image size for POV observations.
        """
        if not MINERL_AVAILABLE:
            raise ImportError("MineRL is not installed. Install with: pip install minerl")

        super().__init__(device, normalize_obs, clip_obs)

        self.env_name = env_name
        self._seed = seed
        self.image_size = image_size

        self._env = minerl.make(env_name)
        self._observation_dim = self._compute_obs_dim()
        self._action_dim = len(self.DISCRETE_ACTIONS)

        logger.info(
            f"Initialized MineRLAdapter: {env_name}, "
            f"obs_dim={self._observation_dim}, action_dim={self._action_dim}"
        )

    def _compute_obs_dim(self) -> int:
        """Compute flattened observation dimension."""
        h, w = self.image_size
        pov_dim = h * w * 3
        compass_dim = 1
        return pov_dim + compass_dim

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
        return True

    def _process_obs(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Process MineRL observation to flat array.

        Args:
            obs: MineRL observation dict.

        Returns:
            np.ndarray: Flattened observation.
        """
        pov = obs["pov"].astype(np.float32) / 255.0

        if pov.shape[:2] != self.image_size:
            import cv2
            pov = cv2.resize(pov, self.image_size)

        pov_flat = pov.flatten()

        compass = obs.get("compass", {}).get("angle", 0.0)
        compass_normalized = np.array([compass / np.pi], dtype=np.float32)

        return np.concatenate([pov_flat, compass_normalized])

    def _action_index_to_minerl(self, action_idx: int) -> Dict[str, Any]:
        """
        Convert discrete action index to MineRL action dict.

        Args:
            action_idx: Discrete action index.

        Returns:
            Dict[str, Any]: MineRL action dictionary.
        """
        base_action = self.DISCRETE_ACTIONS[action_idx].copy()
        base_action["camera"] = np.array(base_action["camera"], dtype=np.float32)
        return base_action

    def reset(self) -> torch.Tensor:
        """
        Reset the environment.

        Returns:
            torch.Tensor: Initial observation tensor on device.
        """
        obs = self._env.reset()
        processed = self._process_obs(obs)
        return self._to_tensor(processed)

    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action: Action tensor (discrete index).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        action_idx = int(action.detach().cpu().item())
        action_dict = self._action_index_to_minerl(action_idx)

        obs, reward, done, info = self._env.step(action_dict)

        obs_tensor = self._to_tensor(self._process_obs(obs))
        return obs_tensor, float(reward), done, False, info

    def close(self) -> None:
        """Close the environment."""
        self._env.close()

    def sample_action(self) -> torch.Tensor:
        """
        Sample a random action.

        Returns:
            torch.Tensor: Random action tensor.
        """
        action_idx = np.random.randint(0, self._action_dim)
        return torch.tensor([action_idx], device=self.device, dtype=torch.long)
