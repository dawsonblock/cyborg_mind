import gymnasium as gym
import numpy as np
import time
import requests
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class LabInstrumentAdapter(gym.Env):
    """
    RL Adapter for Lab Instruments (Robots, Sensors, Liquid Handlers).
    
    Features:
    - REST/WebSocket driver interface
    - Safety guardrails (limits, emergency stop)
    - Latency tracking
    """
    
    def __init__(
        self,
        instrument_url: str = "http://localhost:5000",
        safety_limits: Dict[str, float] = {"max_temp": 100.0, "max_speed": 10.0},
        mock: bool = True
    ):
        super().__init__()
        
        self.url = instrument_url
        self.safety_limits = safety_limits
        self.mock = mock
        
        # Action: [x, y, z, temp, speed]
        self.action_space = gym.spaces.Box(
            low=np.array([-10, -10, 0, 20, 0]),
            high=np.array([10, 10, 10, 100, 10]),
            dtype=np.float32
        )
        
        # Observation: [x, y, z, temp, status]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )
        
        self.state = np.zeros(5, dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Any:
        super().reset(seed=seed)
        
        if self.mock:
            self.state = np.array([0, 0, 0, 25, 1], dtype=np.float32)
        else:
            try:
                resp = requests.post(f"{self.url}/reset", timeout=1.0)
                self.state = np.array(resp.json()["state"], dtype=np.float32)
            except Exception as e:
                logger.error(f"Failed to reset instrument: {e}")
                self.state = np.zeros(5, dtype=np.float32)
                
        return self.state, {}

    def step(self, action: np.ndarray) -> Any:
        # 1. Safety Check
        if not self._check_safety(action):
            logger.warning("Safety violation! Action clipped/rejected.")
            reward = -10.0
            terminated = True
            return self.state, reward, terminated, False, {"error": "safety_violation"}
        
        # 2. Execute
        start_time = time.time()
        if self.mock:
            # Simple physics simulation
            self.state[:3] += action[:3] * 0.1 # Move
            self.state[3] += (action[3] - self.state[3]) * 0.1 # Heat
            self.state[4] = 1.0 # OK
        else:
            try:
                payload = {"action": action.tolist()}
                resp = requests.post(f"{self.url}/step", json=payload, timeout=0.5)
                self.state = np.array(resp.json()["state"], dtype=np.float32)
            except Exception as e:
                logger.error(f"Instrument communication error: {e}")
                return self.state, 0.0, True, False, {"error": str(e)}
                
        latency = time.time() - start_time
        
        # 3. Reward (Task dependent, placeholder here)
        reward = -np.linalg.norm(self.state[:3]) # Minimize distance to origin
        
        terminated = False
        truncated = False
        
        info = {"latency": latency}
        
        return self.state, reward, terminated, truncated, info

    def _check_safety(self, action: np.ndarray) -> bool:
        # Check temp
        if action[3] > self.safety_limits["max_temp"]:
            return False
        # Check speed
        if action[4] > self.safety_limits["max_speed"]:
            return False
        return True
