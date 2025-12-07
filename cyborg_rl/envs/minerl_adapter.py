"""
MineRL Environment Adapter v2.0

Production-grade adapter for MineRL environments with:
- Reward normalization (running mean/std)
- Sticky-action mode (action repeat with probability)
- Configurable frame stacking (4/8 frames)
- Full type hints and documentation
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import cv2

from cyborg_rl.envs.base import BaseEnvAdapter
from cyborg_rl.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import minerl
    MINERL_AVAILABLE = True
except ImportError:
    MINERL_AVAILABLE = False
    logger.warning("MineRL not installed. MineRLAdapter will not be available.")


class RunningMeanStd:
    """Running mean and standard deviation tracker for reward normalization."""
    
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x: np.ndarray) -> None:
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = x.shape[0] if x.ndim > 0 else 1
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(self, batch_mean: float, batch_var: float, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m_2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
    
    @property
    def std(self) -> float:
        return np.sqrt(self.var + 1e-8)


class MineRLAdapter(BaseEnvAdapter):
    """
    Production adapter for MineRL environments.

    Features:
    - Discretized action space (10 actions)
    - Frame stacking with configurable depth
    - Reward normalization (running stats)
    - Sticky-action mode for exploration
    - Efficient image preprocessing
    """

    # ==================== DISCRETE ACTION MAPPING ====================
    DISCRETE_ACTIONS: List[Dict[str, Any]] = [
        {"forward": 1, "back": 0, "left": 0, "right": 0, "jump": 0, "attack": 0, "camera": [0, 0]},    # 0: Forward
        {"forward": 0, "back": 1, "left": 0, "right": 0, "jump": 0, "attack": 0, "camera": [0, 0]},    # 1: Back
        {"forward": 0, "back": 0, "left": 1, "right": 0, "jump": 0, "attack": 0, "camera": [0, 0]},    # 2: Left
        {"forward": 0, "back": 0, "left": 0, "right": 1, "jump": 0, "attack": 0, "camera": [0, 0]},    # 3: Right
        {"forward": 1, "back": 0, "left": 0, "right": 0, "jump": 1, "attack": 0, "camera": [0, 0]},    # 4: Jump Forward
        {"forward": 1, "back": 0, "left": 0, "right": 0, "jump": 0, "attack": 0, "camera": [0, -15]},  # 5: Look Left
        {"forward": 1, "back": 0, "left": 0, "right": 0, "jump": 0, "attack": 0, "camera": [0, 15]},   # 6: Look Right
        {"forward": 1, "back": 0, "left": 0, "right": 0, "jump": 0, "attack": 0, "camera": [-15, 0]},  # 7: Look Up
        {"forward": 1, "back": 0, "left": 0, "right": 0, "jump": 0, "attack": 0, "camera": [15, 0]},   # 8: Look Down
        {"forward": 1, "back": 0, "left": 0, "right": 0, "jump": 0, "attack": 1, "camera": [0, 0]},    # 9: Attack
    ]

    def __init__(
        self,
        env_name: str = "MineRLTreechop-v0",
        device: torch.device = torch.device("cpu"),
        seed: Optional[int] = None,
        # Observation settings
        normalize_obs: bool = True,
        clip_obs: float = 10.0,
        image_size: Tuple[int, int] = (64, 64),
        frame_stack: int = 4,
        crop_center: bool = False,
        # Reward settings
        normalize_rewards: bool = True,
        reward_clip: float = 10.0,
        # Action settings
        sticky_action_prob: float = 0.0,
        # Environment settings
        num_envs: int = 1,
        max_steps: Optional[int] = None,
    ) -> None:
        """
        Initialize MineRL adapter.

        Args:
            env_name: MineRL environment ID
            device: Torch device for tensors
            seed: Random seed
            normalize_obs: Whether to normalize observations
            clip_obs: Observation clipping range
            image_size: Target image size (H, W)
            frame_stack: Number of frames to stack (4 or 8)
            crop_center: Whether to crop center of image
            normalize_rewards: Whether to normalize rewards
            reward_clip: Reward clipping range after normalization
            sticky_action_prob: Probability of repeating previous action
            num_envs: Number of parallel environments
            max_steps: Maximum steps per episode
        """
        if not MINERL_AVAILABLE:
            raise ImportError("MineRL is not installed. Install with: pip install minerl")

        super().__init__(device, normalize_obs, clip_obs)

        # ==================== CONFIGURATION ====================
        self.env_name = env_name
        self._seed = seed
        self.image_size = image_size
        self.frame_stack = frame_stack
        self.crop_center = crop_center
        self.num_envs = num_envs
        self.max_steps = max_steps

        # Reward normalization
        self.normalize_rewards = normalize_rewards
        self.reward_clip = reward_clip
        self._reward_rms = RunningMeanStd() if normalize_rewards else None

        # Sticky action
        self.sticky_action_prob = sticky_action_prob
        self._last_actions: List[int] = [0] * num_envs

        # ==================== ENVIRONMENT CREATION ====================
        self._envs = [minerl.make(env_name) for _ in range(num_envs)]
        self._step_counts = [0] * num_envs

        # ==================== OBSERVATION SPACE ====================
        self._observation_dim = self._compute_obs_dim()
        self._action_dim = len(self.DISCRETE_ACTIONS)

        # Frame buffers: (num_envs, frame_stack, 3, H, W)
        self._frame_buffers = [
            np.zeros((frame_stack, 3, *image_size), dtype=np.float32)
            for _ in range(num_envs)
        ]

        logger.info(
            f"Initialized MineRLAdapter: {env_name}, num_envs={num_envs}, "
            f"obs_dim={self._observation_dim}, action_dim={self._action_dim}, "
            f"frame_stack={frame_stack}, sticky_prob={sticky_action_prob}"
        )

    # ==================== PROPERTIES ====================

    def _compute_obs_dim(self) -> int:
        """Compute flattened observation dimension."""
        c = 3 * self.frame_stack
        h, w = self.image_size
        # Stacked POV + compass
        return (c * h * w) + 1

    @property
    def observation_dim(self) -> int:
        return self._observation_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def is_discrete(self) -> bool:
        return True

    # ==================== OBSERVATION PROCESSING ====================

    def _process_obs(self, obs_dict: Dict[str, Any], env_idx: int) -> np.ndarray:
        """
        Process raw MineRL observation into flattened tensor.

        Args:
            obs_dict: Raw observation dictionary from MineRL
            env_idx: Environment index for frame buffer

        Returns:
            Flattened observation array
        """
        # 1. Extract and preprocess POV
        if "pov" not in obs_dict:
            pov = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.float32)
        else:
            pov = obs_dict["pov"].astype(np.float32) / 255.0

        # 2. Center crop if enabled
        if self.crop_center:
            h, w, _ = pov.shape
            short_dim = min(h, w)
            cy, cx = h // 2, w // 2
            r = short_dim // 2
            pov = pov[cy - r : cy + r, cx - r : cx + r]

        # 3. Resize to target size
        if pov.shape[:2] != self.image_size:
            pov = cv2.resize(pov, self.image_size)

        # 4. Convert to CHW format
        pov = pov.transpose(2, 0, 1)  # (H, W, 3) -> (3, H, W)

        # 5. Update frame buffer (shift and insert new frame)
        self._frame_buffers[env_idx] = np.roll(self._frame_buffers[env_idx], -1, axis=0)
        self._frame_buffers[env_idx][-1] = pov

        # 6. Flatten stacked frames
        stacked_flat = self._frame_buffers[env_idx].flatten()

        # 7. Extract compass (normalized)
        compass = obs_dict.get("compass", {}).get("angle", 0.0)
        if compass is None:
            compass = 0.0
        compass_normalized = np.array([compass / 180.0], dtype=np.float32)

        return np.concatenate([stacked_flat, compass_normalized])

    def _action_index_to_minerl(self, action_idx: int) -> Dict[str, Any]:
        """Convert discrete action index to MineRL action dictionary."""
        return self.DISCRETE_ACTIONS[action_idx].copy()

    # ==================== REWARD PROCESSING ====================

    def _normalize_reward(self, rewards: np.ndarray) -> np.ndarray:
        """Normalize rewards using running statistics."""
        if not self.normalize_rewards or self._reward_rms is None:
            return rewards

        self._reward_rms.update(rewards)
        normalized = rewards / self._reward_rms.std
        return np.clip(normalized, -self.reward_clip, self.reward_clip)

    # ==================== ENVIRONMENT INTERFACE ====================

    def reset(self) -> torch.Tensor:
        """
        Reset all environments.

        Returns:
            Initial observations tensor (num_envs, obs_dim)
        """
        obs_list = []

        for i, env in enumerate(self._envs):
            obs = env.reset()

            # Reset frame buffer
            self._frame_buffers[i] = np.zeros_like(self._frame_buffers[i])
            self._step_counts[i] = 0
            self._last_actions[i] = 0

            # Process observation (fills last frame slot)
            obs_flat = self._process_obs(obs, i)

            # Fill buffer with initial frame (repeat for all slots)
            current_frame = self._frame_buffers[i][-1].copy()
            for j in range(self.frame_stack):
                self._frame_buffers[i][j] = current_frame

            # Re-extract flattened observation with full buffer
            stacked_flat = self._frame_buffers[i].flatten()
            compass = obs.get("compass", {}).get("angle", 0.0) or 0.0
            compass_normalized = np.array([compass / 180.0], dtype=np.float32)
            obs_flat = np.concatenate([stacked_flat, compass_normalized])

            obs_list.append(obs_flat)

        return self._to_tensor(np.stack(obs_list))

    def step(
        self, action: np.ndarray
    ) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Step all environments.

        Args:
            action: Action indices (num_envs,)

        Returns:
            (observations, rewards, dones, infos)
        """
        obs_list = []
        rew_list = []
        done_list = []
        info_list = []

        for i, env in enumerate(self._envs):
            # ==================== STICKY ACTION ====================
            act_idx = int(action[i])
            if self.sticky_action_prob > 0 and np.random.random() < self.sticky_action_prob:
                act_idx = self._last_actions[i]
            self._last_actions[i] = act_idx

            # Convert to MineRL action
            act_dict = self._action_index_to_minerl(act_idx)

            # ==================== STEP ====================
            obs, rew, done, info = env.step(act_dict)
            self._step_counts[i] += 1

            # Check max steps
            if self.max_steps and self._step_counts[i] >= self.max_steps:
                done = True
                info["truncated"] = True

            # ==================== AUTO-RESET ====================
            if done:
                obs = env.reset()
                self._frame_buffers[i] = np.zeros_like(self._frame_buffers[i])
                self._step_counts[i] = 0
                self._last_actions[i] = 0

            # Process observation
            processed = self._process_obs(obs, i)

            obs_list.append(processed)
            rew_list.append(rew)
            done_list.append(done)
            info_list.append(info)

        # Stack and normalize rewards
        rewards = np.array(rew_list, dtype=np.float32)
        rewards = self._normalize_reward(rewards)

        return (
            self._to_tensor(np.stack(obs_list)),
            rewards,
            np.array(done_list, dtype=bool),
            info_list,
        )

    def close(self) -> None:
        """Close all environments."""
        for env in self._envs:
            try:
                env.close()
            except Exception as e:
                logger.warning(f"Error closing environment: {e}")

    def sample_action(self) -> torch.Tensor:
        """Sample random actions for all environments."""
        actions = np.random.randint(0, self._action_dim, size=self.num_envs)
        return torch.tensor(actions, device=self.device, dtype=torch.long)

    # ==================== DIAGNOSTICS ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics for logging."""
        stats = {
            "env_name": self.env_name,
            "num_envs": self.num_envs,
            "frame_stack": self.frame_stack,
            "sticky_action_prob": self.sticky_action_prob,
        }
        if self._reward_rms:
            stats["reward_mean"] = float(self._reward_rms.mean)
            stats["reward_std"] = float(self._reward_rms.std)
        return stats
