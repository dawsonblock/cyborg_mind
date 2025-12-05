"""Delayed-Cue vectorized environment for memory benchmarking.

This environment tests long-horizon memory by presenting a cue, 
injecting a variable-length delay, then requiring the agent to recall the cue.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional


class DelayedCueEnv(gym.Env):
    """
    Delayed-Cue Memory Task (Vectorized).
    
    Episode Structure:
    1. Cue Phase (1 step): Present one-hot cue indicating target direction
    2. Delay Phase (horizon steps): Neutral observations (zeros)
    3. Query Phase (1 step): Agent must recall and execute correct action
    
    Reward:
    - +1.0 for correct action at query step
    - 0.0 otherwise
    
    Args:
        num_cues: Number of possible cues/actions (default 4)
        horizon: Length of delay phase (default 100)
        obs_dim: Observation dimension (default = num_cues)
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        num_cues: int = 4,
        horizon: int = 100,
        obs_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.num_cues = num_cues
        self.horizon = horizon
        self.obs_dim = obs_dim or num_cues
        
        # Observation: either cue (one-hot) or zeros during delay
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )
        
        # Action: continuous vector (will be argmax'd)
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_cues,), dtype=np.float32
        )
        
        # Episode state
        self.current_cue = 0
        self.step_count = 0
        self.total_steps = horizon + 2  # cue + delay + query
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment and present new cue."""
        super().reset(seed=seed)
        
        # Sample random cue
        self.current_cue = self.np_random.integers(0, self.num_cues)
        self.step_count = 0
        
        # Return one-hot cue observation
        obs = self._get_observation()
        info = {
            "cue": self.current_cue,
            "phase": "cue",
            "step": self.step_count,
        }
        
        return obs, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation based on phase."""
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        
        if self.step_count == 0:
            # Cue phase: one-hot encoding
            obs[self.current_cue] = 1.0
        elif self.step_count <= self.horizon:
            # Delay phase: zeros (neutral)
            pass
        else:
            # Query phase: zeros (agent must recall from memory)
            pass
            
        return obs
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute action and return result."""
        self.step_count += 1
        
        # Determine reward
        reward = 0.0
        success = False
        
        if self.step_count == self.total_steps - 1:
            # Query phase: check if action matches cue
            # Convert continuous action to discrete choice via argmax
            action_arr = np.asarray(action)
            action_val = int(np.argmax(action_arr))
            if action_val == self.current_cue:
                reward = 10.0  # Stronger reward signal
                success = True
            else:
                reward = -1.0  # Penalty for wrong answer
        
        # Determine phase
        if self.step_count == 0:
            phase = "cue"
        elif self.step_count <= self.horizon:
            phase = "delay"
        else:
            phase = "query"
        
        # Check if episode is done
        terminated = self.step_count >= self.total_steps - 1
        truncated = False
        
        obs = self._get_observation()
        info = {
            "cue": self.current_cue,
            "phase": phase,
            "step": self.step_count,
            "success": success,
        }
        
        return obs, reward, terminated, truncated, info


def make_delayed_cue_env(num_cues=4, horizon=100, obs_dim=None, seed=None):
    """Factory function for DelayedCueEnv."""
    def _init():
        env = DelayedCueEnv(num_cues=num_cues, horizon=horizon, obs_dim=obs_dim)
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init


def VectorizedDelayedCueEnv(num_envs: int, num_cues: int = 4, horizon: int = 100, obs_dim: Optional[int] = None):
    """
    Create a vectorized DelayedCueEnv using SyncVectorEnv.
    
    Args:
        num_envs: Number of parallel environments
        num_cues: Number of cues/actions
        horizon: Delay length
        obs_dim: Observation dimension
        
    Returns:
        gym.vector.VectorEnv: Vectorized environment
    """
    envs = [
        make_delayed_cue_env(num_cues=num_cues, horizon=horizon, obs_dim=obs_dim, seed=i)
        for i in range(num_envs)
    ]
    
    return gym.vector.SyncVectorEnv(envs)
