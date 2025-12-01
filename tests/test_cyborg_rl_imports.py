import torch
import pytest

# Simple test to verify cyborg_rl is importable
def test_imports():
    """Test that cyborg_rl modules can be imported."""
    from cyborg_rl.memory.pmm import PredictiveMemoryModule
    from cyborg_rl.agents.ppo_agent import PPOAgent
    from cyborg_rl.config import Config
    
    assert PredictiveMemoryModule is not None
    assert PPOAgent is not None
    assert Config is not None
