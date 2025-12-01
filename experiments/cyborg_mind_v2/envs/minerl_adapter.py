"""
MineRL Environment Adapter for CyborgMind V2

Adapts MineRL environments (TreeChop, Navigate, etc.) to the unified
BrainEnvAdapter interface. Handles complex observation spaces, action
mapping, and reward shaping specific to Minecraft tasks.

MineRL Specifics:
- Observation: Dict with 'pov' (pixels), 'inventory', 'compassAngle', etc.
- Action: Dict with 'camera', 'forward', 'attack', etc.
- Reward: Sparse (wood collected, distance traveled, etc.)

This adapter:
1. Extracts POV image as pixels
2. Flattens inventory/compass into scalars
3. Creates goal vector from task type
4. Maps discrete action indices to MineRL action dicts
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import gym

from .base_adapter import BaseEnvAdapter, BrainInputs


class MineRLAdapter(BaseEnvAdapter):
    """
    Adapter for MineRL environments.

    Converts MineRL's complex dict observations/actions into the brain's
    unified format: (pixels, scalars, goal) -> action_idx
    """

    # MineRL action space mapping
    # We define a discrete action space by enumerating useful action combinations
    ACTION_MAP = [
        # No-op
        {"camera": [0, 0], "forward": 0, "back": 0, "left": 0, "right": 0,
         "jump": 0, "sneak": 0, "sprint": 0, "attack": 0},

        # Movement
        {"camera": [0, 0], "forward": 1, "back": 0, "left": 0, "right": 0,
         "jump": 0, "sneak": 0, "sprint": 0, "attack": 0},
        {"camera": [0, 0], "forward": 0, "back": 1, "left": 0, "right": 0,
         "jump": 0, "sneak": 0, "sprint": 0, "attack": 0},
        {"camera": [0, 0], "forward": 0, "back": 0, "left": 1, "right": 0,
         "jump": 0, "sneak": 0, "sprint": 0, "attack": 0},
        {"camera": [0, 0], "forward": 0, "back": 0, "left": 0, "right": 1,
         "jump": 0, "sneak": 0, "sprint": 0, "attack": 0},

        # Forward + Attack (chopping trees)
        {"camera": [0, 0], "forward": 1, "back": 0, "left": 0, "right": 0,
         "jump": 0, "sneak": 0, "sprint": 0, "attack": 1},

        # Attack only
        {"camera": [0, 0], "forward": 0, "back": 0, "left": 0, "right": 0,
         "jump": 0, "sneak": 0, "sprint": 0, "attack": 1},

        # Jump
        {"camera": [0, 0], "forward": 1, "back": 0, "left": 0, "right": 0,
         "jump": 1, "sneak": 0, "sprint": 0, "attack": 0},

        # Camera movements (looking around)
        {"camera": [-15, 0], "forward": 0, "back": 0, "left": 0, "right": 0,
         "jump": 0, "sneak": 0, "sprint": 0, "attack": 0},  # Look up
        {"camera": [15, 0], "forward": 0, "back": 0, "left": 0, "right": 0,
         "jump": 0, "sneak": 0, "sprint": 0, "attack": 0},  # Look down
        {"camera": [0, -15], "forward": 0, "back": 0, "left": 0, "right": 0,
         "jump": 0, "sneak": 0, "sprint": 0, "attack": 0},  # Look left
        {"camera": [0, 15], "forward": 0, "back": 0, "left": 0, "right": 0,
         "jump": 0, "sneak": 0, "sprint": 0, "attack": 0},  # Look right

        # Combined: Forward + Look up
        {"camera": [-10, 0], "forward": 1, "back": 0, "left": 0, "right": 0,
         "jump": 0, "sneak": 0, "sprint": 0, "attack": 0},
        # Combined: Forward + Look down
        {"camera": [10, 0], "forward": 1, "back": 0, "left": 0, "right": 0,
         "jump": 0, "sneak": 0, "sprint": 0, "attack": 0},
        # Combined: Forward + Look left
        {"camera": [0, -10], "forward": 1, "back": 0, "left": 0, "right": 0,
         "jump": 0, "sneak": 0, "sprint": 0, "attack": 0},
        # Combined: Forward + Look right
        {"camera": [0, 10], "forward": 1, "back": 0, "left": 0, "right": 0,
         "jump": 0, "sneak": 0, "sprint": 0, "attack": 0},

        # Combined: Forward + Attack + Look down (optimal for trees)
        {"camera": [10, 0], "forward": 1, "back": 0, "left": 0, "right": 0,
         "jump": 0, "sneak": 0, "sprint": 0, "attack": 1},

        # Sprint forward
        {"camera": [0, 0], "forward": 1, "back": 0, "left": 0, "right": 0,
         "jump": 0, "sneak": 0, "sprint": 1, "attack": 0},

        # Sneak (for careful movement)
        {"camera": [0, 0], "forward": 1, "back": 0, "left": 0, "right": 0,
         "jump": 0, "sneak": 1, "sprint": 0, "attack": 0},
    ]

    # Inventory items to track (for TreeChop)
    INVENTORY_ITEMS = ["log", "planks", "stick", "crafting_table", "dirt", "stone"]

    def __init__(
        self,
        env_name: str,
        image_size: Tuple[int, int] = (128, 128),
        device: str = "cuda",
        max_steps: int = 8000,
    ):
        """
        Initialize MineRL adapter.

        Args:
            env_name: MineRL environment ID (e.g., "MineRLTreechop-v0")
            image_size: Target image size (H, W)
            device: PyTorch device
            max_steps: Maximum steps per episode
        """
        super().__init__(env_name, image_size, device)

        # Create MineRL environment
        try:
            self.env = gym.make(env_name)
        except Exception as e:
            raise RuntimeError(f"Failed to create MineRL env {env_name}: {e}")

        self.max_steps = max_steps
        self._current_obs = None

        # Determine goal based on task
        self._goal_vector = self._create_goal_vector(env_name)

    def _create_goal_vector(self, env_name: str) -> np.ndarray:
        """
        Create task-specific goal vector.

        Different MineRL tasks have different objectives:
        - TreeChop: Collect wood
        - Navigate: Reach target
        - etc.

        Args:
            env_name: Environment name

        Returns:
            np.ndarray: Goal vector [goal_dim]
        """
        # Simple one-hot encoding of task type
        tasks = ["treechop", "navigate", "obtain", "survival"]
        goal = np.zeros(len(tasks), dtype=np.float32)

        for i, task in enumerate(tasks):
            if task in env_name.lower():
                goal[i] = 1.0
                break

        return goal

    @property
    def action_space_size(self) -> int:
        """Number of discrete actions."""
        return len(self.ACTION_MAP)

    @property
    def observation_shape(self) -> Tuple[int, int, int]:
        """Shape of pixel observations."""
        return (3, self.image_size[0], self.image_size[1])

    @property
    def scalar_dim(self) -> int:
        """Dimensionality of scalar features."""
        # Inventory items + compass angle + step count
        return len(self.INVENTORY_ITEMS) + 1 + 1

    @property
    def goal_dim(self) -> int:
        """Dimensionality of goal vector."""
        return len(self._goal_vector)

    def _extract_scalars(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Extract scalar features from MineRL observation.

        Features:
        - Inventory counts for tracked items
        - Compass angle (normalized)
        - Step count (normalized)

        Args:
            obs: MineRL observation dict

        Returns:
            np.ndarray: Scalar feature vector
        """
        scalars = []

        # Inventory
        inventory = obs.get("inventory", {})
        for item in self.INVENTORY_ITEMS:
            count = inventory.get(item, 0)
            scalars.append(float(count))

        # Compass angle (normalize to [-1, 1])
        compass = obs.get("compassAngle", 0.0)
        scalars.append(float(compass) / 180.0)

        # Step count (normalize to [0, 1])
        step_ratio = self._episode_step / self.max_steps
        scalars.append(float(step_ratio))

        return np.array(scalars, dtype=np.float32)

    def _obs_to_brain_inputs(self, obs: Dict[str, Any]) -> BrainInputs:
        """
        Convert MineRL observation to BrainInputs.

        Args:
            obs: MineRL observation dict

        Returns:
            BrainInputs: Unified observation format
        """
        # Extract POV image
        pov = obs["pov"]  # [H, W, 3]
        pixels = self.preprocess_pixels(pov)

        # Extract scalars
        scalars_array = self._extract_scalars(obs)
        scalars = self.normalize_scalars(scalars_array)

        # Goal vector
        goal = torch.from_numpy(self._goal_vector).float().to(self.device)

        return BrainInputs(pixels=pixels, scalars=scalars, goal=goal)

    def _action_idx_to_minerl_action(self, action_idx: int) -> Dict[str, Any]:
        """
        Map discrete action index to MineRL action dict.

        Args:
            action_idx: Action index from brain

        Returns:
            Dict[str, Any]: MineRL action dict

        Raises:
            ValueError: If action_idx out of bounds
        """
        if not 0 <= action_idx < len(self.ACTION_MAP):
            raise ValueError(
                f"Action index {action_idx} out of bounds [0, {len(self.ACTION_MAP)})"
            )

        action = self.ACTION_MAP[action_idx].copy()

        # Convert to numpy arrays where needed
        action["camera"] = np.array(action["camera"], dtype=np.float32)

        return action

    def reset(self) -> BrainInputs:
        """
        Reset MineRL environment.

        Returns:
            BrainInputs: Initial observation
        """
        self.reset_episode_stats()
        obs = self.env.reset()
        self._current_obs = obs
        return self._obs_to_brain_inputs(obs)

    def step(self, action_idx: int) -> Tuple[BrainInputs, float, bool, Dict[str, Any]]:
        """
        Execute action in MineRL environment.

        Args:
            action_idx: Discrete action index

        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Map to MineRL action
        minerl_action = self._action_idx_to_minerl_action(action_idx)

        # Execute in environment
        obs, reward, done, info = self.env.step(minerl_action)
        self._current_obs = obs
        
        # Apply reward shaping
        shaped_reward = self._shape_reward(reward, obs, info)

        # Update stats
        self.update_episode_stats(shaped_reward)

        # Check max steps
        if self._episode_step >= self.max_steps:
            done = True
            info["timeout"] = True

        # Convert to brain format
        brain_obs = self._obs_to_brain_inputs(obs)

        # Add episode stats to info
        info.update(self.episode_stats)

        return brain_obs, float(shaped_reward), bool(done), info

    def _shape_reward(self, raw_reward: float, obs: Dict[str, Any], info: Dict[str, Any]) -> float:
        """
        Shape reward for better learning signal.
        
        Adds dense rewards for:
        - Movement (small penalty to encourage efficiency, or bonus for exploration)
        - Inventory gain (dense reward for collecting items)
        - Camera movement (small penalty to discourage jitter)
        """
        reward = raw_reward
        
        # 1. Inventory gain reward
        # We track previous inventory to detect changes
        current_inventory = obs.get("inventory", {})
        if not hasattr(self, "_prev_inventory"):
            self._prev_inventory = current_inventory
            
        for item in self.INVENTORY_ITEMS:
            curr_count = current_inventory.get(item, 0)
            prev_count = self._prev_inventory.get(item, 0)
            if curr_count > prev_count:
                # Bonus for collecting items
                reward += (curr_count - prev_count) * 1.0
                
        self._prev_inventory = current_inventory
        
        # 2. Exploration reward (optional, based on distance)
        # MineRL usually provides this in raw reward if configured, but we can add more.
        
        return reward

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render MineRL environment.

        Args:
            mode: Rendering mode

        Returns:
            Optional[np.ndarray]: Rendered frame if rgb_array mode
        """
        if mode == "rgb_array" and self._current_obs is not None:
            return self._current_obs["pov"]
        elif mode == "human":
            # MineRL doesn't have native render, just return POV
            return self._current_obs["pov"] if self._current_obs else None
        return None

    def close(self) -> None:
        """Close MineRL environment."""
        if hasattr(self, "env"):
            self.env.close()

    def get_action_name(self, action_idx: int) -> str:
        """
        Get human-readable action name.

        Args:
            action_idx: Action index

        Returns:
            str: Action description
        """
        if not 0 <= action_idx < len(self.ACTION_MAP):
            return "INVALID"

        action = self.ACTION_MAP[action_idx]

        # Build description
        parts = []
        if action["forward"]:
            parts.append("Forward")
        if action["back"]:
            parts.append("Back")
        if action["left"]:
            parts.append("Left")
        if action["right"]:
            parts.append("Right")
        if action["jump"]:
            parts.append("Jump")
        if action["sprint"]:
            parts.append("Sprint")
        if action["sneak"]:
            parts.append("Sneak")
        if action["attack"]:
            parts.append("Attack")
        if action["camera"][0] != 0 or action["camera"][1] != 0:
            parts.append(f"Camera{action['camera']}")

        return " + ".join(parts) if parts else "No-op"
