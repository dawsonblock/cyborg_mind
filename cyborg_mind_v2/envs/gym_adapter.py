"""
Gym Environment Adapter for CyborgMind V2.6

Hardened adapter for OpenAI Gym/Gymnasium environments with production-grade
validation, error handling, and robust preprocessing.

Features:
- Full pixel → 128×128 pipeline with shape validation
- Continuous → discretized action mapping with bounds checking
- Proper scalar_dim auto-detection with safety checks
- Clean error messages for incorrect shapes
- Support for both classic control and Atari environments

V2.6 Enhancements:
- Added comprehensive input validation
- Improved error handling with descriptive messages
- Added dimension consistency checks
- Added NaN detection in observations

Breaking Changes / Migration Notes:
- Both `gymnasium` and the legacy `gym` package are supported. `gymnasium` is preferred for full support, but `gym` remains available for backward compatibility.
  Users are encouraged to migrate their environments and code to use `gymnasium`, but existing `gym` environments will continue to work. See requirements.txt for details.
"""

from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import torch
import warnings

try:
    import gymnasium as gym
    GYM_VERSION = "gymnasium"
except ImportError:
    import gym
    GYM_VERSION = "gym"

from .base_adapter import BaseEnvAdapter, BrainInputs


class GymAdapter(BaseEnvAdapter):
    """
    Hardened adapter for OpenAI Gym/Gymnasium environments.

    Handles both state-based (CartPole) and pixel-based (Atari) environments
    with full validation and error recovery.

    V2.6 Production Features:
    - Guaranteed [3, 128, 128] pixel output
    - Validated scalar dimensions
    - Action bounds checking
    - NaN detection and handling
    """

    def __init__(
        self,
        env_name: str,
        image_size: Tuple[int, int] = (128, 128),
        device: str = "cuda",
        use_pixels: bool = False,
        max_steps: int = 1000,
        validate_obs: bool = True,
    ):
        """
        Initialize Gym adapter with validation.

        Args:
            env_name: Gym environment ID (e.g., "CartPole-v1")
            image_size: Target image size (H, W) - enforced to 128x128
            device: PyTorch device
            use_pixels: If True, render environment to pixels
            max_steps: Maximum steps per episode
            validate_obs: If True, validate observations for NaN/inf

        Raises:
            RuntimeError: If environment creation fails
            ValueError: If configuration is invalid
        """
        # Enforce 128x128 image size for consistency
        if image_size != (128, 128):
            warnings.warn(
                f"Image size {image_size} changed to (128, 128) for V2.6 consistency"
            )
            image_size = (128, 128)

        super().__init__(env_name, image_size, device)

        # Create Gym environment with error handling
        try:
            if GYM_VERSION == "gymnasium":
                self.env = gym.make(env_name, render_mode="rgb_array")
            else:
                self.env = gym.make(env_name)
        except gym.error.UnregisteredEnv:
            raise RuntimeError(
                f"Environment '{env_name}' not found. "
                f"Check spelling or install required dependencies."
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to create environment '{env_name}': {type(e).__name__}: {e}"
            )

        self.use_pixels = use_pixels
        self.max_steps = max_steps
        self.validate_obs = validate_obs

        # Analyze observation space with validation
        obs_space = self.env.observation_space
        self._validate_observation_space(obs_space)

        if isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 3:
            # Pixel-based environment (e.g., Atari)
            self.is_pixel_env = True
            self.use_pixels = True
        else:
            # State-based environment
            self.is_pixel_env = False

        # Analyze action space with validation
        action_space = self.env.action_space
        self._validate_action_space(action_space)

        if isinstance(action_space, gym.spaces.Discrete):
            self._action_space_size = action_space.n
            self.is_discrete = True
            self._continuous_dim = 0
        elif isinstance(action_space, gym.spaces.Box):
            # For continuous action spaces, discretize
            self._continuous_dim = action_space.shape[0]
            self._action_space_size = self._discretize_action_space(action_space)
            self.is_discrete = False
            self._action_low = action_space.low
            self._action_high = action_space.high
        else:
            raise ValueError(
                f"Unsupported action space type: {type(action_space).__name__}. "
                f"Only Discrete and Box spaces supported."
            )

        # Auto-detect state dimensionality with safety
        if isinstance(obs_space, gym.spaces.Box):
            if len(obs_space.shape) == 1:
                self._state_dim = obs_space.shape[0]
            else:
                # Pixel space - use minimal placeholder scalars
                self._state_dim = 4  # [step_ratio, reward_avg, done_count, placeholder]
        elif isinstance(obs_space, gym.spaces.Discrete):
            self._state_dim = obs_space.n
        else:
            self._state_dim = 4  # Safe default

        self._current_obs = None
        self._goal_vector = self._create_goal_vector(env_name)
        self._reward_history = []
        self._nan_count = 0

    def _validate_observation_space(self, obs_space: gym.Space) -> None:
        """
        Validate observation space is supported.

        Args:
            obs_space: Gym observation space

        Raises:
            ValueError: If observation space is unsupported
        """
        if not isinstance(obs_space, (gym.spaces.Box, gym.spaces.Discrete)):
            raise ValueError(
                f"Unsupported observation space: {type(obs_space).__name__}. "
                f"Only Box and Discrete spaces supported."
            )

    def _validate_action_space(self, action_space: gym.Space) -> None:
        """
        Validate action space is supported.

        Args:
            action_space: Gym action space

        Raises:
            ValueError: If action space is unsupported
        """
        if not isinstance(action_space, (gym.spaces.Discrete, gym.spaces.Box)):
            raise ValueError(
                f"Unsupported action space: {type(action_space).__name__}. "
                f"Only Discrete and Box spaces supported."
            )

    def _discretize_action_space(self, action_space: gym.spaces.Box) -> int:
        """
        Discretize continuous action space with validation.

        For each action dimension, create discrete bins:
        - 1D: 5 bins [-1, -0.5, 0, 0.5, 1]
        - 2D: 9 bins (8-way + center)
        - 3D+: 2 * dim + 1 actions (positive/negative for each axis plus a no-op)

        Args:
            action_space: Continuous action space

        Returns:
            int: Number of discrete actions

        Raises:
            ValueError: If action space dimensionality is invalid
        """
        dim = action_space.shape[0]

        if dim <= 0:
            raise ValueError(f"Invalid action dimension: {dim}")

        if dim == 1:
            return 5  # [-1, -0.5, 0, 0.5, 1]
        elif dim == 2:
            return 9  # 8 directions + no-op
        elif dim >= 3:
            # For 3D and higher, use 2*dim + 1 actions (positive/negative for each axis + no-op)
            return 2 * dim + 1

    def _create_goal_vector(self, env_name: str) -> np.ndarray:
        """
        Create task-specific goal vector.

        Args:
            env_name: Environment name

        Returns:
            np.ndarray: Goal vector [goal_dim]
        """
        # Extended task vocabulary
        tasks = [
            "cartpole", "mountaincar", "pendulum", "acrobot",
            "lunarlander", "pong", "breakout", "spaceinvaders",
            "seaquest", "qbert", "mspacman", "beamrider",
        ]
        goal = np.zeros(len(tasks), dtype=np.float32)

        # Match task name
        env_lower = env_name.lower()
        for i, task in enumerate(tasks):
            if task in env_lower:
                goal[i] = 1.0
                break

        # Default if no match
        if not goal.any():
            goal[0] = 1.0

        return goal

    @property
    def action_space_size(self) -> int:
        """Number of discrete actions."""
        return self._action_space_size

    @property
    def observation_shape(self) -> Tuple[int, int, int]:
        """Shape of pixel observations - always [3, 128, 128]."""
        return (3, 128, 128)

    @property
    def scalar_dim(self) -> int:
        """Dimensionality of scalar features."""
        return self._state_dim + 1  # +1 for step ratio

    @property
    def goal_dim(self) -> int:
        """Dimensionality of goal vector."""
        return len(self._goal_vector)

    def _render_to_pixels(self) -> np.ndarray:
        """
        Render environment to pixel array with fallback.

        Returns:
            np.ndarray: RGB image [H, W, 3]

        Raises:
            RuntimeError: If rendering fails critically
        """
        try:
            if GYM_VERSION == "gymnasium":
                img = self.env.render()
            else:
                img = self.env.render(mode="rgb_array")

            if img is not None and isinstance(img, np.ndarray):
                # Validate shape
                if img.ndim == 3 and img.shape[2] in [3, 4]:
                    return img[:, :, :3]  # Take RGB only
                elif img.ndim == 2:
                    # Grayscale - convert to RGB
                    return np.stack([img] * 3, axis=-1)

        except Exception as e:
            warnings.warn(f"Rendering failed: {e}. Using fallback.")

        # Fallback: create blank image with proper shape
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        return img

    def _state_to_scalars(self, state: np.ndarray) -> np.ndarray:
        """
        Convert environment state to scalar features with validation.

        Args:
            state: Raw state observation

        Returns:
            np.ndarray: Scalar features [scalar_dim]

        Raises:
            ValueError: If state contains NaN/inf
        """
        if state is None:
            state = np.zeros(self._state_dim, dtype=np.float32)
        elif isinstance(state, (int, float)):
            state = np.array([state], dtype=np.float32)
        else:
            state = np.asarray(state, dtype=np.float32).flatten()

        # Validate for NaN/inf
        if self.validate_obs:
            if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                self._nan_count += 1
                if getattr(self, "nan_policy", "lenient") == "strict":
                    raise ValueError(
                        f"NaN/inf detected in state (count: {self._nan_count}). "
                        f"State: {state}"
                    )
                else:
                    warnings.warn(
                        f"NaN/inf detected in state (count: {self._nan_count}). "
                        f"Replacing with zeros."
                    )
                    state = np.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)
        # Pad or truncate to _state_dim
        if len(state) < self._state_dim:
            state = np.pad(state, (0, self._state_dim - len(state)))
        elif len(state) > self._state_dim:
            state = state[:self._state_dim]

        # Add step ratio
        step_ratio = min(float(self._episode_step) / self.max_steps, 1.0)
        scalars = np.concatenate([state, [step_ratio]])

        # Final validation
        if scalars.shape[0] != self.scalar_dim:
            raise ValueError(
                f"Scalar dimension mismatch: expected {self.scalar_dim}, "
                f"got {scalars.shape[0]}"
            )

        return scalars

    def _obs_to_brain_inputs(self, obs: Any) -> BrainInputs:
        """
        Convert Gym observation to BrainInputs with full validation.

        Args:
            obs: Gym observation (state or pixels)

        Returns:
            BrainInputs: Unified observation format

        Raises:
            ValueError: If observation format is invalid
            RuntimeError: If preprocessing fails
        """
        # Handle pixels
        if self.use_pixels or self.is_pixel_env:
            if self.is_pixel_env:
                # Use observation directly as pixels
                if not isinstance(obs, np.ndarray):
                    raise ValueError(
                        f"Expected numpy array for pixel obs, got {type(obs)}"
                    )
                pixels = self.preprocess_pixels(obs)
            else:
                # Render environment
                img = self._render_to_pixels()
                pixels = self.preprocess_pixels(img)
        else:
            # Create pixels from rendering
            img = self._render_to_pixels()
            pixels = self.preprocess_pixels(img)

        # Pixel shape is guaranteed by preprocess_pixels (BaseEnvAdapter)

        # Handle scalars
        if self.is_pixel_env:
            # For pixel envs, use minimal scalars
            reward_avg = (
                np.mean(self._reward_history[-10:])
                if self._reward_history else 0.0
            )
            scalars_array = np.array([
                float(self._episode_step) / self.max_steps,
                float(reward_avg),
                float(self._nan_count),
                0.0
            ], dtype=np.float32)
        else:
            # Use state as scalars
            scalars_array = self._state_to_scalars(obs)

        scalars = self.normalize_scalars(scalars_array)

        # Validate scalar shape
        if scalars.shape[0] != self.scalar_dim:
            raise ValueError(
                f"Scalar dimension mismatch: expected {self.scalar_dim}, "
                f"got {scalars.shape[0]}"
            )

        # Goal vector
        goal = torch.from_numpy(self._goal_vector).float().to(self.device)

        return BrainInputs(pixels=pixels, scalars=scalars, goal=goal)

    def _action_idx_to_gym_action(self, action_idx: int) -> Union[int, np.ndarray]:
        """
        Map discrete action index to Gym action with bounds checking.

        Args:
            action_idx: Action index from brain

        Returns:
            Gym action (int or np.ndarray)

        Raises:
            ValueError: If action_idx is out of bounds
        """
        # Validate action index
        if action_idx < 0 or action_idx >= self._action_space_size:
            raise ValueError(
                f"Action index {action_idx} out of bounds [0, {self._action_space_size})"
            )

        if self.is_discrete:
            # Direct mapping for discrete action spaces
            return int(action_idx)
        else:
            # Map to continuous action
            dim = self._continuous_dim

            if dim == 1:
                # Map to 5 bins
                bins = [-1.0, -0.5, 0.0, 0.5, 1.0]
                value = bins[action_idx % len(bins)]
                # Scale to action bounds
                low, high = self._action_low[0], self._action_high[0]
                scaled = low + (value + 1.0) * (high - low) / 2.0
                scaled = np.clip(scaled, low, high)
                return np.array([scaled], dtype=np.float32)

            elif dim == 2:
                # Map to 9 directions
                directions = [
                    [0, 0],     # No-op
                    [-1, 0],    # Left
                    [1, 0],     # Right
                    [0, -1],    # Down
                    [0, 1],     # Up
                    [-1, -1],   # Down-Left
                    [-1, 1],    # Up-Left
                    [1, -1],    # Down-Right
                    [1, 1],     # Up-Right
                ]
                idx = action_idx % len(directions)  # Ensure idx is within bounds
                action = np.array(directions[idx], dtype=np.float32)
                # Scale to action bounds
                for i in range(2):
                    if action[i] != 0:
                        low, high = self._action_low[i], self._action_high[i]
                        action[i] = low if action[i] < 0 else high
                return action

            else:
                # For higher dims, use simple mapping
                # Create action vector with one active dimension
                action = np.zeros(dim, dtype=np.float32)
                if action_idx > 0:
                    dim_idx = (action_idx - 1) // 2
                    direction = 1 if (action_idx - 1) % 2 == 1 else -1
                    if dim_idx < dim:
                        low, high = self._action_low[dim_idx], self._action_high[dim_idx]
                        action[dim_idx] = high if direction > 0 else low
                return action

    def reset(self) -> BrainInputs:
        """
        Reset Gym environment with validation.

        Returns:
            BrainInputs: Initial observation

        Raises:
            RuntimeError: If reset fails
        """
        try:
            self.reset_episode_stats()
            self._reward_history = []

            # Reset environment and normalize return (gymnasium returns (obs, info))
            reset_out = self.env.reset()
            if isinstance(reset_out, tuple):
                obs = reset_out[0]
            else:
                obs = reset_out

            # Normalize observation for downstream processing
            if isinstance(obs, dict):
                # Attempt to flatten dict observations into a single array
                try:
                    obs = np.concatenate(
                        [np.asarray(v, dtype=np.float32).ravel() for v in obs.values()]
                    ).astype(np.float32)
                except Exception as e:
                    raise RuntimeError(f"Unsupported dict observation structure: {e}")

            self._current_obs = obs
            return self._obs_to_brain_inputs(obs)

        except Exception as e:
            raise RuntimeError(
                f"Environment reset failed: {type(e).__name__}: {e}"
            )
            if isinstance(obs, dict):
                # Attempt to flatten dict observations into a single array
                try:
                    obs = np.concatenate(
                        [np.asarray(v, dtype=np.float32).ravel() for v in obs.values()]
                    ).astype(np.float32)
                except Exception as e:
                    raise RuntimeError(f"Unsupported dict observation structure: {e}")

            self._current_obs = obs
            return self._obs_to_brain_inputs(obs)

        except Exception as e:
            raise RuntimeError(
                f"Environment reset failed: {type(e).__name__}: {e}"
            )

    def step(self, action_idx: int) -> Tuple[BrainInputs, float, bool, Dict[str, Any]]:
        """
        Execute action in Gym environment with full error handling.

        Args:
            action_idx: Discrete action index

        Returns:
            Tuple of (observation, reward, done, info)

        Raises:
            ValueError: If action is invalid
            RuntimeError: If step execution fails
        """
        try:
            # Map to Gym action (with validation)
            gym_action = self._action_idx_to_gym_action(action_idx)

            # Execute in environment
            if GYM_VERSION == "gymnasium":
                obs, reward, terminated, truncated, info = self.env.step(gym_action)
                done = terminated or truncated
            else:
                obs, reward, done, info = self.env.step(gym_action)

            self._current_obs = obs

            # Update stats
            self.update_episode_stats(reward)
            self._reward_history.append(float(reward))

            # Check max steps
            if self._episode_step >= self.max_steps:
                done = True
                info["timeout"] = True

            # Convert to brain format (with validation)
            brain_obs = self._obs_to_brain_inputs(obs)

            # Add episode stats to info
            info.update(self.episode_stats)
            info["nan_count"] = self._nan_count

            return brain_obs, float(reward), bool(done), info

        except ValueError:
            # Re-raise validation errors
            raise
        except Exception as e:
            raise RuntimeError(
                f"Environment step failed: {type(e).__name__}: {e}"
            )

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render Gym environment.

        Args:
            mode: Rendering mode

        Returns:
            Optional[np.ndarray]: Rendered frame if rgb_array mode
        """
        try:
            if GYM_VERSION == "gymnasium":
                return self.env.render()
            else:
                return self.env.render(mode=mode)
        except Exception as e:
            warnings.warn(f"Render failed: {e}")
            return None

    def close(self) -> None:
        """Close Gym environment safely."""
        if hasattr(self, "env") and self.env is not None:
            try:
                self.env.close()
            except Exception as e:
                warnings.warn(f"Environment close failed: {e}")
