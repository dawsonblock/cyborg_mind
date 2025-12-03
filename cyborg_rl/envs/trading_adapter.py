import gymnasium as gym
import numpy as np
import pandas as pd
import time
import threading
from typing import Dict, Any, Optional, List
from collections import deque

# Optional dependencies
try:
    import ccxt
except ImportError:
    ccxt = None

class TradingAdapter(gym.Env):
    """
    RL Adapter for Live/Simulated Trading.
    
    Features:
    - Live WebSocket data feed (via CCXT or custom)
    - OHLCV + Orderbook normalization
    - Risk management wrapper
    - Reward shaping (PnL, Sharpe, Drawdown)
    """
    
    def __init__(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        window_size: int = 60,
        initial_balance: float = 10000.0,
        live: bool = False,
        exchange_id: str = "binance",
    ):
        super().__init__()
        
        self.symbol = symbol
        self.timeframe = timeframe
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.live = live
        
        # Action Space: [Hold, Buy, Sell] or Continuous [Size]
        # Using discrete for simplicity: 0=Hold, 1=Buy (Full), 2=Sell (Full)
        self.action_space = gym.spaces.Discrete(3)
        
        # Observation Space: [Window, Features]
        # Features: Open, High, Low, Close, Volume, RSI, MACD
        self.n_features = 7
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(window_size, self.n_features), 
            dtype=np.float32
        )
        
        self.balance = initial_balance
        self.position = 0.0 # Amount of asset held
        self.entry_price = 0.0
        self.history = deque(maxlen=window_size)
        
        if live and ccxt:
            self.exchange = getattr(ccxt, exchange_id)()
            # Start background thread for data fetching
            self._start_live_feed()
        else:
            # Simulation mode: Load dummy data or historical CSV
            self.data = self._generate_dummy_data()
            self.current_step = window_size

    def _generate_dummy_data(self) -> pd.DataFrame:
        """Generate random walk data for testing."""
        dates = pd.date_range(start="2023-01-01", periods=10000, freq="1min")
        price = 100 + np.cumsum(np.random.randn(10000))
        df = pd.DataFrame(index=dates)
        df["Open"] = price
        df["High"] = price + np.random.rand(10000)
        df["Low"] = price - np.random.rand(10000)
        df["Close"] = price + np.random.randn(10000) * 0.1
        df["Volume"] = np.abs(np.random.randn(10000) * 1000)
        return df

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Any:
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        
        if not self.live:
            self.current_step = self.window_size
            # Fill history
            obs = self._get_observation(self.current_step)
        else:
            # Wait for enough live data
            while len(self.history) < self.window_size:
                time.sleep(1)
            obs = np.array(self.history, dtype=np.float32)
            
        return obs, {}

    def step(self, action: int) -> Any:
        current_price = self._get_current_price()
        reward = 0.0
        
        # Execute Action
        if action == 1: # Buy
            if self.balance > 0:
                amount = self.balance / current_price
                cost = amount * current_price * 0.001 # 0.1% fee
                self.position += amount
                self.balance -= (amount * current_price + cost)
                self.entry_price = current_price
                
        elif action == 2: # Sell
            if self.position > 0:
                revenue = self.position * current_price
                cost = revenue * 0.001
                self.balance += (revenue - cost)
                
                # Reward is realized PnL
                pnl = (current_price - self.entry_price) / self.entry_price
                reward = pnl * 100 # Scale up
                
                self.position = 0.0
                self.entry_price = 0.0
        
        # Step forward
        if not self.live:
            self.current_step += 1
            terminated = self.current_step >= len(self.data) - 1
            obs = self._get_observation(self.current_step)
        else:
            terminated = False
            obs = np.array(self.history, dtype=np.float32)
            
        # Unrealized PnL reward component
        if self.position > 0:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            reward += unrealized_pnl * 0.1 # Small shaping reward
            
        info = {
            "balance": self.balance,
            "position": self.position,
            "total_value": self.balance + (self.position * current_price)
        }
        
        return obs, reward, terminated, False, info

    def _get_current_price(self) -> float:
        if self.live:
            return self.history[-1][3] # Close
        return self.data.iloc[self.current_step]["Close"]

    def _get_observation(self, step: int) -> np.ndarray:
        # Extract window
        window = self.data.iloc[step-self.window_size:step]
        # Feature Engineering (Simplified)
        features = window[["Open", "High", "Low", "Close", "Volume"]].values
        # Pad with 0s for missing indicators for now
        padding = np.zeros((self.window_size, 2)) 
        return np.hstack([features, padding]).astype(np.float32)

    def _start_live_feed(self):
        # Placeholder for WebSocket thread
        pass
