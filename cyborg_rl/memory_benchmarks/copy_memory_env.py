"""Copy-Memory task environment (stub implementation).

This environment tests the agent's ability to memorize and reproduce sequences.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional


class CopyMemoryEnv(gym.Env):
    """
    Copy-Memory Task (Simplified Stub).
    
    Episode Structure:
    1. Input Phase: Agent observes a sequence of symbols (one per step)
    2. Delay Phase: Short delay with neutral observations
    3. Output Phase: Agent must reproduce the sequence
    
    This is a STUB implementation with simplified mechanics for demonstration.
    
    Args:
        num_symbols: Number of distinct symbols (default 4)
        sequence_length: Length of sequence to memorize (default 3)
        delay_length: Delay between input and output (default 5)
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        num_symbols: int = 4,
        sequence_length: int = 3,
        delay_length: int = 5,
    ):
        super().__init__()
        
        self.num_symbols = num_symbols
        self.sequence_length = sequence_length
        self.delay_length = delay_length
        
        # Observation: one-hot symbol or zeros during delay/output
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(num_symbols + 1,), dtype=np.float32
        )
        
        # Action: symbol choice + "no-op" action
        self.action_space = spaces.Discrete(num_symbols + 1)
        
        # Episode state
        self.sequence = []
        self.step_count = 0
        self.output_idx = 0
        self.total_steps = sequence_length + delay_length + sequence_length
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset and generate new sequence."""
        super().reset(seed=seed)
        
        # Generate random sequence
        self.sequence = [
            self.np_random.integers(0, self.num_symbols)
            for _ in range(self.sequence_length)
        ]
        self.step_count = 0
        self.output_idx = 0
        
        obs = self._get_observation()
        info = {"sequence": self.sequence, "phase": "input"}
        
        return obs, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs = np.zeros(self.num_symbols + 1, dtype=np.float32)
        
        if self.step_count < self.sequence_length:
            # Input phase: show symbol
            symbol = self.sequence[self.step_count]
            obs[symbol] = 1.0
        elif self.step_count < self.sequence_length + self.delay_length:
            # Delay phase: neutral (all zeros)
            pass
        else:
            # Output phase: neutral, agent must produce from memory
            obs[-1] = 1.0  # Signal output phase
            
        return obs
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute action."""
        reward = 0.0
        
        # Check if in output phase
        if self.step_count >= self.sequence_length + self.delay_length:
            # Verify action matches expected symbol
            if action == self.sequence[self.output_idx]:
                reward = 1.0 / self.sequence_length  # Partial credit
            self.output_idx += 1
        
        self.step_count += 1
        
        terminated = self.step_count >= self.total_steps
        truncated = False
        
        obs = self._get_observation()
        
        if self.step_count < self.sequence_length:
            phase = "input"
        elif self.step_count < self.sequence_length + self.delay_length:
            phase = "delay"
        else:
            phase = "output"
            
        info = {"sequence": self.sequence, "phase": phase}
        
        return obs, reward, terminated, truncated, info
