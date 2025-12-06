"""MineRL environment adapter."""

from typing import Any, Dict, Tuple, Optional, List
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
        frame_stack: int = 4,
        crop_center: bool = False,
        num_envs: int = 1,
        max_steps: Optional[int] = None
    ) -> None:
        if not MINERL_AVAILABLE:
            raise ImportError("MineRL is not installed. Install with: pip install minerl")

        super().__init__(device, normalize_obs, clip_obs)

        self.env_name = env_name
        self._seed = seed
        self.image_size = image_size
        self.frame_stack = frame_stack
        self.crop_center = crop_center
        self.num_envs = num_envs

        if num_envs > 1:
            # Vectorized env
            # MineRL currently doesn't support built-in vectorization well in single process?
            # We usually use AsyncVectorEnv from Gym/Gymnasium.
            # Here we assume user might wrap this adapter in a VectorEnv OR we handle it if needed.
            # But the Trainer uses this Adapter as the Env.
            # If num_envs > 1, we should probably return a VectorEnv?
            # However, MineRL is heavy. Launching 32 instances on one machine might OOM.
            # For this implementation, we assume Single Env unless we use a wrapper.
            # The prompt asked for "Vectorized MineRL environments".
            # Python 'gymnasium.vector.AsyncVectorEnv' or 'SyncVectorEnv' is standard.
            # But here we implement a Single Adapter which might be vectorized externally 
            # OR we implement simple serial vectorization internally if requested.
            # Given constraints, I will implement SERIAL vectorization loop here if num_envs > 1 
            # to be robust without extra libs, or rely on gym.vector.
            # Let's assume EXTERNAL wrapping or simple list of envs.
            # Wait, the Trainer calls `self.env.step(action_cpu)`. `action_cpu` is (B,).
            # So this Adapter MUST handle vectorization.
            self._envs = [minerl.make(env_name) for _ in range(num_envs)]
        else:
            self._envs = [minerl.make(env_name)]

        self._observation_dim = self._compute_obs_dim()
        self._action_dim = len(self.DISCRETE_ACTIONS)
        
        # Frame Stacking Buffers: List of deques per env?
        # Or just concatenate last N frames.
        # Simple: Keep last N frames in a buffer using numpy.
        self._frame_buffers = [np.zeros((frame_stack, 3, *image_size), dtype=np.float32) for _ in range(num_envs)]
        
        logger.info(
            f"Initialized MineRLAdapter: {env_name}, num_envs={num_envs}, "
            f"obs_dim={self._observation_dim}, action_dim={self._action_dim}"
        )

    def _compute_obs_dim(self) -> Tuple[int, int, int]:
        # Return (C * Stack, H, W) + compass?
        # If we return flattened, it's different.
        # Trainer expects flattened dim?
        # Trainer: `flattened_obs_dim = int(np.prod(self.obs_dim))`
        # If we stack frames, we increase channels.
        # POV: (3 * stack, H, W).
        # Compass: 1 * stack? Or just current compass? Usually current.
        # But flattening mixes them.
        # Let's stack POV on channels.
        c = 3 * self.frame_stack
        h, w = self.image_size
        # Compass is separate.
        # If we flat, it's (C*H*W) + 1.
        # Trainer uses Linear.
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

    def _process_obs(self, obs_dict: Dict[str, Any], env_idx: int) -> np.ndarray:
        # 1. Image Processing
        if "pov" not in obs_dict:
            pov = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.float32)
        else:
            pov = obs_dict["pov"].astype(np.float32) / 255.0

        # Crop Center
        if self.crop_center:
            h, w, _ = pov.shape
            # Take center 50%? or fixed size? 
            # Treechop: focus on center square.
            # If resizing to 64x64, maybe crop to 64x64 from center if larger?
            # MineRL native is 64x64 or higher.
            # Let's crop center square then resize.
            short_dim = min(h, w)
            cy, cx = h // 2, w // 2
            r = short_dim // 2
            pov = pov[cy-r:cy+r, cx-r:cx+r]
            
        if pov.shape[:2] != self.image_size:
            pov = cv2.resize(pov, self.image_size)
            
        # (H, W, 3) -> (3, H, W)
        pov = pov.transpose(2, 0, 1)

        # Update Buffer
        # Shift buffer
        # buffer shape: (Stack, 3, H, W)
        self._frame_buffers[env_idx] = np.roll(self._frame_buffers[env_idx], -1, axis=0)
        self._frame_buffers[env_idx][-1] = pov
        
        # Flatten Stacked Frames
        # (Stack, 3, H, W) -> (Stack*3, H, W) -> Flat
        stacked_pov = self._frame_buffers[env_idx].reshape(-1, *self.image_size)
        stacked_flat = stacked_pov.flatten()        

        # 2. Compass
        compass = obs_dict.get("compass", {}).get("angle", 0.0)
        if compass is None: compass = 0.0
        compass_normalized = np.array([compass / 180.0], dtype=np.float32)

        return np.concatenate([stacked_flat, compass_normalized])

    def reset(self) -> torch.Tensor:
        obs_list = []
        for i, env in enumerate(self._envs):
            obs = env.reset()
            # Reset buffer
            self._frame_buffers[i] = np.zeros_like(self._frame_buffers[i])
            # Pre-fill buffer with initial frame?
            # Yes, usually repeat initial frame.
            if "pov" in obs:
               # ... process initial frame ...
               # Simplified: just call process once, it fills last slot, others zero.
               # Better: Repeat N times.
               # Let's just process once.
               pass
            obs_flat = self._process_obs(obs, i)
            # Fill buffer with this frame by running process N times?
            # Hack: Manually duplicate last frame to all slots
            # Extract the last frame (last 3*H*W elements of pov part)
            # Actually _process_obs updates buffer.
            # We want to fill it.
            current_frame = self._frame_buffers[i][-1]
            for j in range(self.frame_stack - 1):
                self._frame_buffers[i][j] = current_frame
            
            # Re-process to get full stack
            # Efficient: just stack current_frame N times
            # logic inside _process_obs calls flatten on buffer.
            # So calling it again returns correctly stacked.
            obs_flat = self._process_obs(obs, i) # Wait, this shifts again!
            # Correct logic: Manually set buffer, then get result.
            # The buffer is already set by first _process_obs.
            # So just fix the buffer, then manually stack.
            # To avoid complexity, standard is zero-init or repeat.
            # Zero-init is fine (conceptually "just turned on").
            
            obs_list.append(obs_flat)
            
        return self._to_tensor(np.stack(obs_list)) # (B, D)

    def step(self, action: np.ndarray) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, List[Dict]]:
        # action is (B,) numpy
        obs_list = []
        rew_list = []
        done_list = []
        info_list = []
        
        for i, env in enumerate(self._envs):
            act_idx = int(action[i])
            act_dict = self._action_index_to_minerl(act_idx)
            
            obs, rew, done, info = env.step(act_dict)
            
            if done:
                obs = env.reset()
                self._frame_buffers[i] = np.zeros_like(self._frame_buffers[i])
            
            processed = self._process_obs(obs, i)
            
            obs_list.append(processed)
            rew_list.append(rew)
            done_list.append(done)
            info_list.append(info)
            
        return (
            self._to_tensor(np.stack(obs_list)), 
            np.array(rew_list, dtype=np.float32), 
            np.array(done_list, dtype=bool), 
            info_list
        )

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
