"""Environment adapters for CyborgMind RL."""

from cyborg_rl.envs.base import BaseEnvAdapter
from cyborg_rl.envs.gym_adapter import GymAdapter

__all__ = ["BaseEnvAdapter", "GymAdapter"]

try:
    from cyborg_rl.envs.minerl_adapter import MineRLAdapter
    __all__.append("MineRLAdapter")
except ImportError:
    pass  # MineRL not installed
