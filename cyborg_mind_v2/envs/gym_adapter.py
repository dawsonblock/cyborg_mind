"""
Gym Environment Adapter for CyborgMind V2

Adapts OpenAI Gym environments to the unified BrainEnvAdapter interface.
Supports both classic control tasks (CartPole, MountainCar) and Atari games.

Gym Specifics:
- Observation: Box (continuous) or Discrete
- Action: Discrete or Continuous
- Handles both pixel-based and state-based environments

This adapter:
1. Renders environment to pixels for pixel obs (or uses state as "pixels")
2. Uses state directly as scalars
3. Creates simple goal vectors
4. Maps brain action indices to gym actions
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import gym

from .base_adapter import BaseEnvAdapter, BrainInputs


class GymAdapter(BaseEnvAdapter):
    """
    Adapter for OpenAI Gym environments.

    Handles both state-based (CartPole) and pixel-based (Atari) environments.
    For state-based envs, renders a simple visualization as "pixels".
    """

    def __init__(
        self,
        env_name: str,
        image_size: Tuple[int, int] = (128, 128),
        device: str = "cuda",
        use_pixels: bool = False,
        max_steps: int = 1000,
    ):
        """
        Initialize Gym adapter.

        Args:
            env_name: Gym environment ID (e.g., "CartPole-v1")
            image_size: Target image size (H, W)
            device: PyTorch device
            use_pixels: If True, render environment to pixels; else use state
            max_steps: Maximum steps per episode
        """
        super().__init__(env_name, image_size, device)

        # Create Gym environment
        try:
            self.env = gym.make(env_name)
        except Exception as e:
            raise RuntimeError(f"Failed to create Gym env {env_name}: {e}")

        self.use_pixels = use_pixels
        self.max_steps = max_steps

        # Determine if environment has pixel observations
        obs_space = self.env.observation_space
        if isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 3:
            # Pixel-based environment (e.g., Atari)
            self.is_pixel_env = True
            self.use_pixels = True
        else:
            # State-based environment
            self.is_pixel_env = False

        # Determine action space size
        action_space = self.env.action_space
        if isinstance(action_space, gym.spaces.Discrete):
            self._action_space_size = action_space.n
            self.is_discrete = True
        elif isinstance(action_space, gym.spaces.Box):
            # For continuous action spaces, discretize
            self._action_space_size = self._discretize_action_space(action_space)
            self.is_discrete = False
        else:
            raise ValueError(f"Unsupported action space: {type(action_space)}")

        # State dimensionality
        if isinstance(obs_space, gym.spaces.Box):
            if len(obs_space.shape) == 1:
                self._state_dim = obs_space.shape[0]
            else:
                # Pixel space
                self._state_dim = 4  # Use minimal placeholder scalars
        elif isinstance(obs_space, gym.spaces.Discrete):
            self._state_dim = obs_space.n
        else:
            self._state_dim = 4  # Default

        self._current_obs = None
        self._goal_vector = self._create_goal_vector(env_name)

    def _discretize_action_space(self, action_space: gym.spaces.Box) -> int:
        """
        Discretize continuous action space.

        For each action dimension, create 3 discrete values: [-1, 0, 1]
        Total discrete actions = 3^dim

        Args:
            action_space: Continuous action space

        Returns:
            int: Number of discrete actions
        """
        dim = action_space.shape[0]
        # Limit to prevent combinatorial explosion
        if dim > 3:
            # For high-dim spaces, use a fixed set of useful actions
            return 9  # 8 directions + no-op
        return 3 ** dim

    def _create_goal_vector(self, env_name: str) -> np.ndarray:
        """
        Create task-specific goal vector.

        Args:
            env_name: Environment name

        Returns:
            np.ndarray: Goal vector
        """
        # Simple one-hot encoding of common task types
        tasks = ["cartpole", "mountaincar", "pendulum", "acrobot",
                 "lunarlander", "pong", "breakout", "spaceinvaders"]
        goal = np.zeros(len(tasks), dtype=np.float32)

        for i, task in enumerate(tasks):
            if task in env_name.lower():
                goal[i] = 1.0
                break

        # If no match, use first element
        if not goal.any():
            goal[0] = 1.0

        return goal

    @property
    def action_space_size(self) -> int:
        """Number of discrete actions."""
        return self._action_space_size

    @property
    def observation_shape(self) -> Tuple[int, int, int]:
        """Shape of pixel observations."""
        return (3, self.image_size[0], self.image_size[1])

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
        Render environment to pixel array.

        Returns:
            np.ndarray: RGB image [H, W, 3]
        """
        try:
            # Try rgb_array mode
            img = self.env.render(mode="rgb_array")
            if img is not None:
                return img
        except:
            pass

        # Fallback: create blank image
        img = np.zeros((*self.image_size, 3), dtype=np.uint8)
        return img

    def _state_to_scalars(self, state: np.ndarray) -> np.ndarray:
        """
        Convert environment state to scalar features.

        Args:
            state: Raw state observation

        Returns:
            np.ndarray: Scalar features
        """
        if state is None:
            state = np.zeros(self._state_dim, dtype=np.float32)
        elif isinstance(state, (int, float)):
            state = np.array([state], dtype=np.float32)
        else:
            state = np.asarray(state, dtype=np.float32).flatten()

        # Pad or truncate to _state_dim
        if len(state) < self._state_dim:
            state = np.pad(state, (0, self._state_dim - len(state)))
        elif len(state) > self._state_dim:
            state = state[:self._state_dim]

        # Add step ratio
        step_ratio = self._episode_step / self.max_steps
        scalars = np.concatenate([state, [step_ratio]])

        return scalars

    def _obs_to_brain_inputs(self, obs: Any) -> BrainInputs:
        """
        Convert Gym observation to BrainInputs.

        Args:
            obs: Gym observation (state or pixels)

        Returns:
            BrainInputs: Unified observation format
        """
        # Handle pixels
        if self.use_pixels or self.is_pixel_env:
            if self.is_pixel_env:
                # Use observation directly as pixels
                pixels = self.preprocess_pixels(obs)
            else:
                # Render environment
                img = self._render_to_pixels()
                pixels = self.preprocess_pixels(img)
        else:
            # Create dummy pixels from state (visualization)
            img = self._render_to_pixels()
            pixels = self.preprocess_pixels(img)

        # Handle scalars
        if self.is_pixel_env:
            # For pixel envs, use minimal scalars
            scalars_array = np.array([
                float(self._episode_step) / self.max_steps,
                0.0, 0.0, 0.0
            ], dtype=np.float32)
        else:
            # Use state as scalars
            scalars_array = self._state_to_scalars(obs)

        scalars = self.normalize_scalars(scalars_array)

        # Goal vector
        goal = torch.from_numpy(self._goal_vector).float().to(self.device)

        return BrainInputs(pixels=pixels, scalars=scalars, goal=goal)

    def _action_idx_to_gym_action(self, action_idx: int) -> Any:
        """
        Map discrete action index to Gym action.

        Args:
            action_idx: Action index from brain

        Returns:
            Gym action (int or np.ndarray)
        """
        if self.is_discrete:
            # Direct mapping for discrete action spaces
            return action_idx
        else:
            # Map to continuous action
            action_space = self.env.action_space
            dim = action_space.shape[0]

            if dim == 1:
                # Map to [-1, 0, 1]
                actions = [-1.0, 0.0, 1.0]
                idx = action_idx % len(actions)
                return np.array([actions[idx]], dtype=np.float32)
            elif dim == 2:
                # Map to 9 directions (8-way + no-op)
                actions = [
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
                idx = action_idx % len(actions)
                return np.array(actions[idx], dtype=np.float32)
            else:
                # For higher dims, use a simple mapping
                # This is a simplification; real apps may need better discretization
                return np.zeros(dim, dtype=np.float32)

    def reset(self) -> BrainInputs:
        """
        Reset Gym environment.

        Returns:
            BrainInputs: Initial observation
        """
        self.reset_episode_stats()
        obs = self.env.reset()
        self._current_obs = obs
        return self._obs_to_brain_inputs(obs)

    def step(self, action_idx: int) -> Tuple[BrainInputs, float, bool, Dict[str, Any]]:
        """
        Execute action in Gym environment.

        Args:
            action_idx: Discrete action index

        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Map to Gym action
        gym_action = self._action_idx_to_gym_action(action_idx)

        # Execute in environment
        obs, reward, done, info = self.env.step(gym_action)
        self._current_obs = obs

        # Update stats
        self.update_episode_stats(reward)

        # Check max steps
        if self._episode_step >= self.max_steps:
            done = True
            info["timeout"] = True

        # Convert to brain format
        brain_obs = self._obs_to_brain_inputs(obs)

        # Add episode stats to info
        info.update(self.episode_stats)

        return brain_obs, float(reward), bool(done), info

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render Gym environment.

        Args:
            mode: Rendering mode

        Returns:
            Optional[np.ndarray]: Rendered frame if rgb_array mode
        """
        try:
            return self.env.render(mode=mode)
        except:
            return None

    def close(self) -> None:
        """Close Gym environment."""
        if hasattr(self, "env"):
            self.env.close()
