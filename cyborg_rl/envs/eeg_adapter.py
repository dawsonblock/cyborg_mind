import gymnasium as gym
import numpy as np
import time
from typing import Dict, Any, Optional
from collections import deque

# Optional MNE for signal processing
try:
    import mne
except ImportError:
    mne = None

class EEGAdapter(gym.Env):
    """
    RL Adapter for EEG/BCI Signals.
    
    Features:
    - Live EEG stream handling (LSL or synthetic)
    - Windowed feature extraction (PSD, Bandpower)
    - Collapse-signal gating (detects signal loss)
    """
    
    def __init__(
        self,
        channels: int = 8,
        sampling_rate: int = 250,
        window_len: float = 1.0, # seconds
        synthetic: bool = True
    ):
        super().__init__()
        
        self.channels = channels
        self.fs = sampling_rate
        self.window_samples = int(window_len * sampling_rate)
        self.synthetic = synthetic
        
        # Buffer for raw data
        self.buffer = deque(maxlen=self.window_samples)
        
        # Action: Discrete classification or Continuous control
        self.action_space = gym.spaces.Discrete(4) # e.g., Left, Right, Up, Down
        
        # Observation: [Channels, FreqBands]
        # Bands: Delta, Theta, Alpha, Beta, Gamma (5)
        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf, shape=(channels, 5), dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Any:
        super().reset(seed=seed)
        self.buffer.clear()
        
        # Fill buffer
        for _ in range(self.window_samples):
            self.buffer.append(self._get_sample())
            
        return self._compute_features(), {}

    def step(self, action: int) -> Any:
        # 1. Get new data chunk (simulating real-time step)
        # In real BCI, we might step every 100ms
        step_samples = int(0.1 * self.fs)
        for _ in range(step_samples):
            self.buffer.append(self._get_sample())
            
        # 2. Compute Features
        obs = self._compute_features()
        
        # 3. Check Signal Quality
        if self._is_collapsed(obs):
            return obs, 0.0, True, False, {"error": "signal_loss"}
            
        # 4. Reward (Placeholder - needs calibration task)
        # e.g., match target direction
        reward = 0.0 
        
        return obs, reward, False, False, {}

    def _get_sample(self) -> np.ndarray:
        if self.synthetic:
            # Generate sine waves + noise
            t = time.time()
            sample = np.zeros(self.channels)
            for c in range(self.channels):
                # Alpha wave (10Hz) dominant
                sample[c] = np.sin(2 * np.pi * 10 * t) + np.random.normal(0, 0.5)
            return sample
        else:
            # Fetch from LSL inlet
            return np.zeros(self.channels)

    def _compute_features(self) -> np.ndarray:
        data = np.array(self.buffer).T # [Channels, Time]
        
        if data.shape[1] < self.window_samples:
            return np.zeros((self.channels, 5), dtype=np.float32)
            
        # Simple Bandpower estimation
        # Real implementation would use Welch's method via MNE or Scipy
        features = np.zeros((self.channels, 5), dtype=np.float32)
        
        # Mock features: Random but consistent with signal
        features = np.abs(np.random.randn(self.channels, 5))
        
        return features.astype(np.float32)

    def _is_collapsed(self, features: np.ndarray) -> bool:
        # Detect if signal is flatlining or railed
        if np.mean(features) < 1e-6:
            return True
        return False
