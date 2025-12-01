"""
Checkpoint persistence test for Cyborg Mind v2.0.

This test trains a ``CyborgMindController`` for a short run, saves
its brain state to disk and loads it into a new controller.  It then
verifies that memory slots match and that a single forward pass
produces identical action logits (within a tolerance) when given
identical inputs.  The test runs on CPU for determinism and
convenience.
"""

import os
import torch

from experiments.cyborg_mind_v2.integration.cyborg_mind_controller import CyborgMindController


def test_checkpoint_persistence(tmp_path: str = "test_ckpt.pt"):
    # Train for 1 000 steps on dummy data
    controller1 = CyborgMindController(device="cpu")
    agent_ids = ["0"]
    for _ in range(1_000):
        pixels = torch.randn(1, 3, 128, 128)
        scalars = torch.randn(1, controller1.brain.scalar_dim)
        goals = torch.randn(1, controller1.brain.goal_dim)
        controller1.step(agent_ids, pixels, scalars, goals)
    # Save checkpoint
    ckpt_path = tmp_path
    torch.save(controller1.brain.state_dict(), ckpt_path)
    # Load into new controller
    controller2 = CyborgMindController(ckpt_path=ckpt_path, device="cpu")
    # Verify memory slots match
    assert controller2.brain.pmm.mem_slots == controller1.brain.pmm.mem_slots
    # Verify outputs match
    pixels = torch.randn(1, 3, 128, 128)
    scalars = torch.randn(1, controller1.brain.scalar_dim)
    goals = torch.randn(1, controller1.brain.goal_dim)
    aid = ["test"]
    out1_actions = controller1.step(aid, pixels, scalars, goals)
    out2_actions = controller2.step(aid, pixels, scalars, goals)
    assert out1_actions == out2_actions, "Actions mismatch after reload"
    # Cleanup
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)


if __name__ == "__main__":
    test_checkpoint_persistence()