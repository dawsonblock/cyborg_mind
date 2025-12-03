"""
Stress test for Cyborg Mind v2.0 controller.

This test runs 1 000 agents for 72 000 steps (equivalent to one hour
at 20 ticks per second) and asserts that no NaNs appear in the
actions and that memory pressure remains below 1.0.  Progress is
printed every 1 000 steps.  Because this test is computationally
intensive it should be run on a GPU.
"""

import torch

from experiments.cyborg_mind_v2.integration.cyborg_mind_controller import CyborgMindController


def test_1000_agents_1_hour():
    controller = CyborgMindController(device="cuda")
    agent_ids = [str(i) for i in range(1000)]
    for step in range(72_000):
        pixels = torch.randn(1000, 3, 128, 128, device="cuda")
        scalars = torch.randn(1000, controller.brain.scalar_dim, device="cuda")
        goals = torch.randn(1000, controller.brain.goal_dim, device="cuda")
        actions = controller.step(agent_ids, pixels, scalars, goals)
        # Check for anomalies
        assert all(not torch.isnan(torch.tensor(a, dtype=torch.float32)) for a in actions), f"NaN at step {step}"
        pressure = controller.brain.pmm.get_pressure()
        assert pressure < 1.0, f"Pressure overflow at step {step}"
        if step % 1000 == 0:
            print(f"Step {step}/72000: OK (pressure={pressure:.2%})")


if __name__ == "__main__":
    test_1000_agents_1_hour()