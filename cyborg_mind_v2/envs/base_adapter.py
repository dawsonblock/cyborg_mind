"""
Base Environment Adapter Protocol for CyborgMind V2

This module defines the universal interface that all environment adapters
must implement. The BrainEnvAdapter protocol allows CyborgMind to work
with any environment (MineRL, Gym, custom simulators, etc.) by converting
environment-specific observations and actions into the brain's unified format.

Design Philosophy:
- Environment-agnostic: Brain only sees pixels, scalars, and goals
- Action abstraction: Brain outputs action indices, adapter maps to env actions
- Minimal interface: Only essential methods (reset, step, render)
- Type-safe: Uses Protocol for compile-time checking

Example Usage:
    from cyborg_mind_v2.envs import BrainEnvAdapter, MineRLAdapter
    from cyborg_mind_v2.integration import CyborgMindController

    # Create adapter for specific environment
    adapter = MineRLAdapter("MineRLTreechop-v0")

    # Controller uses adapter automatically
    controller = CyborgMindController()

    # Unified interface regardless of environment
    obs = adapter.reset()
    pixels, scalars, goal = obs
    action_idx = controller.step(pixels, scalars, goal)
    next_obs, reward, done, info = adapter.step(action_idx)
"""

from typing import Protocol, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import torch
import numpy as np


@dataclass
class BrainInputs:
    """
    Unified observation format for the CyborgMind brain.

    All environment adapters must convert their native observations into
    this standardized format. The brain processes only these three inputs:

    Attributes:
        pixels: RGB image observation [3, H, W] or [B, 3, H, W]
                Typically resized to 128x128 for the vision encoder
        scalars: Numeric state features [scalar_dim] or [B, scalar_dim]
                 Examples: health, inventory counts, position, velocity
        goal: Task directive vector [goal_dim] or [B, goal_dim]
              Examples: one-hot task ID, target coordinates, reward shaping
    """
    pixels: torch.Tensor  # [3, H, W] RGB image
    scalars: torch.Tensor  # [scalar_dim] numeric features
    goal: torch.Tensor  # [goal_dim] task objective


class BrainEnvAdapter(Protocol):
    """
    Protocol defining the interface for environment adapters.

    Any environment can be used with CyborgMind by implementing this protocol.
    The adapter handles all environment-specific logic:
    - Observation preprocessing and normalization
    - Action space mapping
    - Reward shaping (optional)
    - Episode termination logic

    Methods must follow these contracts:
    - reset() initializes episode and returns BrainInputs
    - step(action_idx) takes brain action and returns (obs, reward, done, info)
    - render() provides visualization (optional, can be no-op)

    Type Signatures:
        reset() -> BrainInputs
        step(action_idx: int) -> Tuple[BrainInputs, float, bool, Dict[str, Any]]
        render() -> None

    Implementation Notes:
    - Adapters should handle both single and batched observations
    - Pixel preprocessing should normalize to [0, 1] or [-1, 1]
    - Scalar features should be normalized (e.g., z-score or min-max)
    - Action mapping should be deterministic and documented
    """

    @property
    def action_space_size(self) -> int:
        """
        Number of discrete actions available to the brain.

        The brain outputs action logits over this many actions.
        The adapter maps action indices [0, action_space_size) to
        environment-specific actions.

        Returns:
            int: Number of discrete actions
        """
        ...

    @property
    def observation_shape(self) -> Tuple[int, int, int]:
        """
        Shape of pixel observations (C, H, W).

        Returns:
            Tuple[int, int, int]: (channels, height, width)
        """
        ...

    @property
    def scalar_dim(self) -> int:
        """
        Dimensionality of scalar feature vector.

        Returns:
            int: Number of scalar features
        """
        ...

    @property
    def goal_dim(self) -> int:
        """
        Dimensionality of goal vector.

        Returns:
            int: Number of goal features
        """
        ...

    def reset(self) -> BrainInputs:
        """
        Reset the environment to initial state.

        Called at the start of each episode. Should initialize the
        environment and return the first observation in BrainInputs format.

        Returns:
            BrainInputs: Initial observation (pixels, scalars, goal)

        Raises:
            RuntimeError: If environment fails to reset
        """
        ...

    def step(self, action_idx: int) -> Tuple[BrainInputs, float, bool, Dict[str, Any]]:
        """
        Execute one environment step with the given action.

        The action_idx is an integer in [0, action_space_size) produced by
        the brain. The adapter maps this to the environment's native action
        format and executes it.

        Args:
            action_idx: Discrete action index from brain

        Returns:
            Tuple containing:
                - BrainInputs: Next observation
                - float: Reward signal
                - bool: Episode termination flag
                - Dict[str, Any]: Additional info (for debugging/logging)

        Raises:
            ValueError: If action_idx is out of bounds
            RuntimeError: If environment step fails
        """
        ...

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render the environment for visualization.

        Optional method for debugging and evaluation. Can be a no-op
        for headless training.

        Args:
            mode: Rendering mode ("human" for display, "rgb_array" for pixels)

        Returns:
            Optional[np.ndarray]: Rendered frame if mode="rgb_array", else None
        """
        ...

    def close(self) -> None:
        """
        Clean up environment resources.

        Called when the environment is no longer needed. Should release
        any held resources (file handles, display windows, etc.).
        """
        ...


class BaseEnvAdapter:
    """
    Abstract base class for environment adapters.

    Provides common utilities for implementing BrainEnvAdapter:
    - Image preprocessing (resize, normalize)
    - Scalar normalization
    - Action mapping helpers

    Subclasses should implement the abstract methods and customize
    preprocessing as needed for their specific environment.
    """

    def __init__(
        self,
        env_name: str,
        image_size: Tuple[int, int] = (128, 128),
        device: str = "cuda",
    ):
        """
        Initialize base adapter.

        Args:
            env_name: Name/ID of the environment
            image_size: Target size for pixel observations (H, W)
            device: PyTorch device for tensor operations
        """
        self.env_name = env_name
        self.image_size = image_size
        self.device = torch.device(device)
        self._episode_step = 0
        self._episode_reward = 0.0

    def preprocess_pixels(
        self,
        pixels: np.ndarray,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Preprocess pixel observations into brain format.

        Steps:
        1. Convert to float32
        2. Resize to target image_size
        3. Normalize to [0, 1]
        4. Transpose to CHW format if needed
        5. Convert to torch tensor

        Args:
            pixels: Raw pixel array (H, W, C) or (C, H, W)
            normalize: Whether to normalize to [0, 1]

        Returns:
            torch.Tensor: Preprocessed pixels [3, H, W]
        """
        import cv2

        # Ensure numpy array
        if isinstance(pixels, torch.Tensor):
            pixels = pixels.cpu().numpy()

        # Handle different input shapes
        if pixels.ndim == 4:  # Batched
            pixels = pixels[0]

        if pixels.ndim == 2:  # Grayscale
            pixels = np.stack([pixels] * 3, axis=-1)

        # HWC -> CHW if needed
        if pixels.shape[-1] in [1, 3, 4]:
            pixels = np.transpose(pixels, (2, 0, 1))

        # Take RGB channels only
        if pixels.shape[0] > 3:
            pixels = pixels[:3]

        # CHW -> HWC for OpenCV
        pixels = np.transpose(pixels, (1, 2, 0))

        # Resize
        if pixels.shape[:2] != self.image_size:
            pixels = cv2.resize(
                pixels,
                self.image_size,
                interpolation=cv2.INTER_LINEAR
            )

        # Back to CHW
        pixels = np.transpose(pixels, (2, 0, 1))

        # Normalize
        if normalize:
            pixels = pixels.astype(np.float32) / 255.0

        # To tensor
        tensor = torch.from_numpy(pixels).float().to(self.device)
        return tensor

    def normalize_scalars(
        self,
        scalars: np.ndarray,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        Normalize scalar features.

        Args:
            scalars: Raw scalar array
            mean: Per-feature mean (if None, no normalization)
            std: Per-feature std (if None, no normalization)

        Returns:
            torch.Tensor: Normalized scalars
        """
        scalars = np.asarray(scalars, dtype=np.float32)

        if mean is not None and std is not None:
            scalars = (scalars - mean) / (std + 1e-8)

        tensor = torch.from_numpy(scalars).float().to(self.device)
        return tensor

    def reset_episode_stats(self) -> None:
        """Reset per-episode statistics."""
        self._episode_step = 0
        self._episode_reward = 0.0

    def update_episode_stats(self, reward: float) -> None:
        """Update per-episode statistics."""
        self._episode_step += 1
        self._episode_reward += reward

    @property
    def episode_stats(self) -> Dict[str, Any]:
        """Get current episode statistics."""
        return {
            "episode_step": self._episode_step,
            "episode_reward": self._episode_reward,
        }


def create_adapter(
    adapter_type: str,
    env_name: str,
    **kwargs
) -> BrainEnvAdapter:
    """
    Factory function to create environment adapters.

    Args:
        adapter_type: Type of adapter ("minerl", "gym", "synthetic")
        env_name: Environment name/ID
        **kwargs: Additional arguments passed to adapter constructor

    Returns:
        BrainEnvAdapter: Instantiated adapter

    Raises:
        ValueError: If adapter_type is unknown

    Example:
        >>> adapter = create_adapter("minerl", "MineRLTreechop-v0")
        >>> adapter = create_adapter("gym", "CartPole-v1")
        >>> adapter = create_adapter("synthetic", "test-env")
    """
    adapter_type = adapter_type.lower()

    if adapter_type == "minerl":
        from .minerl_adapter import MineRLAdapter
        return MineRLAdapter(env_name, **kwargs)

    elif adapter_type == "gym" or adapter_type == "gymnasium":
        from .gym_adapter import GymAdapter
        return GymAdapter(env_name, **kwargs)

    elif adapter_type == "synthetic":
        # Synthetic adapter will be imported from data module
        from cyborg_mind_v2.data.synthetic_dataset import SyntheticDataset
        # Return a wrapper that acts as an adapter
        raise NotImplementedError(
            "Synthetic adapter integration pending. "
            "Use SyntheticDataset directly for now."
        )

    else:
        raise ValueError(
            f"Unknown adapter type: {adapter_type}. "
            f"Available: minerl, gym, synthetic"
        )
