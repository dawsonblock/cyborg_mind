"""
CyborgMind RL - Production-grade Reinforcement Learning with PMM Memory.

This package provides a complete RL pipeline with:
- Gym/MineRL environment adapters
- Mamba/GRU hybrid models with PMM memory integration
- PPO trainer with GAE and proper batching
- Prometheus metrics for monitoring
"""

__version__ = "1.0.0"
__author__ = "CyborgMind Team"

from cyborg_rl.config import Config
from cyborg_rl.utils.device import get_device
from cyborg_rl.utils.logging import get_logger
from cyborg_rl.utils.seeding import set_seed

__all__ = ["Config", "get_device", "set_seed", "get_logger", "__version__"]
