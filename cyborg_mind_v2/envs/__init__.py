"""
Environment Adapters for CyborgMind V2

This module provides universal environment adapters that allow CyborgMind
to interface with any RL environment (MineRL, Gym, Synthetic).

The adapter system provides a unified observation/action interface:
- Observations: (pixels, scalars, goal) -> brain-compatible format
- Actions: brain action indices -> environment-specific actions

Available Adapters:
- BrainEnvAdapter: Protocol/interface for all adapters
- MineRLAdapter: Minecraft environments (MineRL)
- GymAdapter: OpenAI Gym/Gymnasium environments
- SyntheticAdapter: Generated synthetic trajectories for testing

Usage:
    from cyborg_mind_v2.envs import create_adapter

    # Create adapter for any environment
    adapter = create_adapter("minerl", "MineRLTreechop-v0")

    # Unified interface
    obs = adapter.reset()
    action_idx = brain.select_action(obs.pixels, obs.scalars, obs.goal)
    next_obs, reward, done, info = adapter.step(action_idx)
"""

from .base_adapter import (
    BrainEnvAdapter,
    BrainInputs,
    BaseEnvAdapter,
    create_adapter,
)
from .minerl_adapter import MineRLAdapter
from .gym_adapter import GymAdapter

# Legacy adapters (for backward compatibility)
try:
    from .minerl_obs_adapter import MineRLObsAdapter
except ImportError:
    MineRLObsAdapter = None

try:
    from .action_mapping import ActionMapper
except ImportError:
    ActionMapper = None

__all__ = [
    # Core protocol and types
    "BrainEnvAdapter",
    "BrainInputs",
    "BaseEnvAdapter",

    # Adapter implementations
    "MineRLAdapter",
    "GymAdapter",

    # Factory function
    "create_adapter",

    # Legacy (if available)
    "MineRLObsAdapter",
    "ActionMapper",
]
