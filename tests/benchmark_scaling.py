"""
Benchmark scaling for Cyborg Mind v2.0.

This script measures the throughput (updates per second), latency
per step and GPU memory usage when running the Cyborg Mind
controller with varying numbers of agents.  It is intended to
approximate Phase 4 of the HTDE plan.  You can run this script
manually to produce a simple report printed to stdout.
"""

import time
import torch

from experiments.cyborg_mind_v2.integration.cyborg_mind_controller import CyborgMindController


def benchmark_agents(num_agents: int, num_steps: int = 100, device: str = "cuda") -> tuple[float, float]:
    print("\n" + "=" * 50)
    print(f"BENCHMARK: {num_agents} agents on {device}")
    print("=" * 50)
    controller = CyborgMindController(device=device)
    pixels = torch.randn(num_agents, 3, 128, 128, device=device)
    scalars = torch.randn(num_agents, controller.brain.scalar_dim, device=device)
    goals = torch.randn(num_agents, controller.brain.goal_dim, device=device)
    agent_ids = [str(i) for i in range(num_agents)]
    # Warmup to compile kernels and build memory structures
    for _ in range(10):
        controller.step(agent_ids, pixels, scalars, goals)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(num_steps):
        controller.step(agent_ids, pixels, scalars, goals)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    fps = (num_agents * num_steps) / elapsed
    if device == "cuda":
        mem_gb = torch.cuda.max_memory_allocated() / 1e9
    else:
        mem_gb = 0.0
    print("Results:")
    print(f"  Updates/sec: {fps:,.0f}")
    print(f"  Latency: {elapsed / num_steps * 1000:.2f} ms/step")
    print(f"  Memory: {mem_gb:.2f} GB")
    print(f"  Slots: {controller.brain.pmm.mem_slots}")
    return fps, mem_gb


if __name__ == "__main__":
    for N in [100, 500, 1_000, 2_000, 5_000]:
        try:
            fps, mem = benchmark_agents(N)
            if mem > 10:
                print(f"⚠️  WARNING: {N} agents exceeds 10GB VRAM")
                break
            if fps < 30 * N:
                print(f"⚠️  WARNING: {N} agents below 30 FPS/agent")
        except RuntimeError as e:
            print(f"❌ FAILED at {N} agents: {e}")
            break