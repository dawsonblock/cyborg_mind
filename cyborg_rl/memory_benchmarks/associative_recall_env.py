"""Associative Recall task environment (stub implementation).

This environment tests the agent's ability to learn and recall key-value associations.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Dict


class AssociativeRecallEnv(gym.Env):
    """
    Associative Recall Task (Simplified Stub).
    
    Episode Structure:
    1. Training Phase: Agent observes key-value pairs
    2. Query Phase: Given a key, agent must produce the associated value
    
    This is a STUB implementation with simplified mechanics.
    
    Args:
        num_keys: Number of key-value pairs (default 5)
        key_dim: Dimension of key vectors (default 4)
        value_dim: Number of possible values (default 4)
        num_queries: Number of queries per episode (default 3)
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        num_keys: int = 5,
        key_dim: int = 4,
        value_dim: int = 4,
        num_queries: int = 3,
    ):
        super().__init__()
        
        self.num_keys = num_keys
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_queries = num_queries
        
        # Observation: key vector + phase indicator
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(key_dim + 1,), dtype=np.float32
        )
        
        # Action: value choice (discrete)
        self.action_space = spaces.Discrete(value_dim)
        
        # Episode state
        self.associations: Dict[int, int] = {}  # key_idx -> value
        self.key_vectors = []
        self.current_query_key = None
        self.step_count = 0
        self.query_count = 0
        self.total_steps = num_keys + num_queries
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset and generate new associations."""
        super().reset(seed=seed)
        
        # Generate random key-value associations
        self.associations = {
            i: self.np_random.integers(0, self.value_dim)
            for i in range(self.num_keys)
        }
        
        # Generate random key vectors (one-hot for simplicity)
        self.key_vectors = []
        for i in range(self.num_keys):
            key = np.zeros(self.key_dim, dtype=np.float32)
            if i < self.key_dim:
                key[i] = 1.0  # One-hot
            else:
                # Random for additional keys
                key[self.np_random.integers(0, self.key_dim)] = 1.0
            self.key_vectors.append(key)
        
        self.step_count = 0
        self.query_count = 0
        self.current_query_key = None
        
        obs = self._get_observation()
        info = {"associations": self.associations, "phase": "training"}
        
        return obs, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs = np.zeros(self.key_dim + 1, dtype=np.float32)
        
        if self.step_count < self.num_keys:
            # Training phase: show key-value pair
            key_idx = self.step_count
            obs[:self.key_dim] = self.key_vectors[key_idx]
            # Value is encoded in the reward signal during training
            obs[-1] = 0.0  # Training phase indicator
        else:
            # Query phase: show key, expect value
            obs[:self.key_dim] = self.key_vectors[self.current_query_key]
            obs[-1] = 1.0  # Query phase indicator
            
        return obs
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute action."""
        reward = 0.0
        
        if self.step_count >= self.num_keys:
            # Query phase: check if action matches expected value
            expected_value = self.associations[self.current_query_key]
            if action == expected_value:
                reward = 1.0
            
            self.query_count += 1
            
            # Sample next query key (if more queries remain)
            if self.query_count < self.num_queries:
                self.current_query_key = self.np_random.integers(0, self.num_keys)
        else:
            # Training phase: just observe, no action needed
            # Sample first query key when training ends
            if self.step_count == self.num_keys - 1:
                self.current_query_key = self.np_random.integers(0, self.num_keys)
        
        self.step_count += 1
        
        terminated = self.step_count >= self.total_steps
        truncated = False
        
        obs = self._get_observation()
        
        phase = "training" if self.step_count <= self.num_keys else "query"
        info = {"associations": self.associations, "phase": phase}
        
        return obs, reward, terminated, truncated, info
