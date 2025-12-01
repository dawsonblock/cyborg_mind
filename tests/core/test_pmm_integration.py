import torch
import pytest
from experiments.cyborg_mind_v2.capsule_brain.policy.brain_cyborg_mind import BrainCyborgMind
from experiments.cyborg_mind_v2.capsule_brain.memory.pmm import PredictiveMemoryModule

def test_brain_forward_pass():
    """Test that the brain forward pass works and outputs expected shapes."""
    batch_size = 2
    
    # Init brain
    brain = BrainCyborgMind(
        scalar_dim=20,
        goal_dim=4,
        thought_dim=32,
        emotion_dim=8,
        workspace_dim=64,
        vision_dim=512,
        emb_dim=256,
        hidden_dim=512,
        mem_dim=64,
        num_actions=10,
        start_slots=10
    )
    
    # Create dummy inputs
    pixels = torch.randn(batch_size, 3, 128, 128)
    scalars = torch.randn(batch_size, 20)
    goal = torch.randn(batch_size, 4)
    thought = torch.randn(batch_size, 32)
    emotion = torch.randn(batch_size, 8)
    workspace = torch.randn(batch_size, 64)
    hidden = (
        torch.randn(1, batch_size, 512),
        torch.randn(1, batch_size, 512)
    )
    
    # Forward
    output = brain(
        pixels=pixels,
        scalars=scalars,
        goal=goal,
        thought=thought,
        emotion=emotion,
        workspace=workspace,
        hidden=hidden
    )
    
    # Check outputs
    assert output["action_logits"].shape == (batch_size, 10)
    assert output["value"].shape == (batch_size, 1)
    assert output["thought"].shape == (batch_size, 32)
    assert output["emotion"].shape == (batch_size, 8)
    assert output["workspace"].shape == (batch_size, 64)
    assert output["hidden_h"].shape == (1, batch_size, 512)
    assert output["hidden_c"].shape == (1, batch_size, 512)
    assert "pressure" in output

def test_pmm_write():
    """Test that PMM write updates internal state."""
    brain = BrainCyborgMind(mem_dim=64, start_slots=10)
    
    # Initial state
    initial_usage = brain.pmm.usage.clone()
    
    # Create inputs that trigger write
    # The brain writes automatically in forward()
    pixels = torch.randn(1, 3, 128, 128)
    scalars = torch.randn(1, 20)
    goal = torch.randn(1, 4)
    thought = torch.randn(1, 32)
    
    brain(pixels, scalars, goal, thought)
    
    # Check usage updated (usage decays and adds attention, write also resets usage)
    # It's hard to check exact values but we can check if it changed
    assert not torch.allclose(brain.pmm.usage, initial_usage)
