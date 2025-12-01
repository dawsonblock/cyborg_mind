"""
Unit test to validate memory expansion behaviour in Cyborg Mind v2.

The test instantiates a ``CyborgMindController`` on CPU, repeatedly
performs dummy inference steps and observes whether the underlying
PMM expands when the pressure threshold is exceeded.  It prints the
pressure and slot count every 100 steps and asserts that at least
one expansion occurred after 10 000 steps.
"""

import torch

from experiments.cyborg_mind_v2.integration.cyborg_mind_controller import CyborgMindController


def test_expansion():
    controller = CyborgMindController(device="cpu")
    print(f"Initial slots: {controller.brain.pmm.mem_slots}")
    # Use simple agent IDs
    agent_ids = [str(i) for i in range(8)]
    for step in range(10_000):
        pixels = torch.randn(8, 3, 128, 128)
        scalars = torch.randn(8, controller.brain.scalar_dim)
        goals = torch.randn(8, controller.brain.goal_dim)
        controller.step(agent_ids, pixels, scalars, goals)
        if step % 100 == 0:
            pressure = controller.brain.pmm.get_pressure()
            print(f"Step {step}: Pressure={pressure:.2f}, Slots={controller.brain.pmm.mem_slots}")
    print(f"Final slots: {controller.brain.pmm.mem_slots}")
    assert controller.brain.pmm.mem_slots > 256, "Memory should have expanded"


if __name__ == "__main__":
    test_expansion()