#!/usr/bin/env python3
"""
run_memory_sweep.py

Run a sweep over tasks, backbones, and horizons using the
canonical MemorySuiteConfig + run_single_experiment().

Usage example:

    python scripts/run_memory_sweep.py \
        --tasks delayed_cue copy_memory associative_recall \
        --backbones gru pseudo_mamba mamba_gru \
        --horizons 10 100 1000 \
        --total-timesteps 200000 \
        --device cuda \
        --output results/memory_sweep.csv
"""

import argparse
import csv
import os
import time
from typing import List, Dict, Any

from cyborg_rl.memory_benchmarks.pseudo_mamba_memory_suite import (
    MemorySuiteConfig,
    run_single_experiment,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Memory Benchmark Sweep Runner")

    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["delayed_cue"],
        choices=["delayed_cue", "copy_memory", "associative_recall"],
        help="Tasks to include in the sweep.",
    )
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=["gru", "pseudo_mamba", "mamba_gru"],
        choices=["gru", "pseudo_mamba", "mamba_gru"],
        help="Backbones to include in the sweep.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[10, 100, 1000],
        help="Horizons to test.",
    )

    parser.add_argument(
        "--num-envs", type=int, default=64, help="Number of parallel envs."
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=200_000,
        help="Total timesteps per run.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device string (cpu or cuda)."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Base seed. Will be offset per run."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="experiments/runs",
        help="Base directory for ExperimentRegistry.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=64,
        help="Number of evaluation episodes per run.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/memory_sweep.csv",
        help="CSV file to store sweep results.",
    )

    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output)

    tasks: List[str] = args.tasks
    backbones: List[str] = args.backbones
    horizons: List[int] = args.horizons

    fieldnames = [
        "task",
        "backbone",
        "horizon",
        "run_name",
        "success_rate",
        "mean_reward",
        "steps_per_second",
        "total_updates",
        "total_seconds",
        "final_memory_saturation",
        "final_memory_mean_norm",
        "num_envs",
        "total_timesteps",
        "device",
        "seed",
    ]

    rows: List[Dict[str, Any]] = []

    run_idx = 0
    for task in tasks:
        for backbone in backbones:
            for horizon in horizons:
                run_idx += 1
                run_seed = args.seed + run_idx
                run_name = f"{task}_{backbone}_H{horizon}_T{args.total_timesteps}_S{run_seed}"

                print(
                    f"\n=== RUN {run_idx} / {len(tasks)*len(backbones)*len(horizons)} "
                    f"task={task} backbone={backbone} horizon={horizon} seed={run_seed} ==="
                )

                msc = MemorySuiteConfig(
                    task=task,
                    backbone=backbone,
                    horizon=horizon,
                    run_name=run_name,
                    num_envs=args.num_envs,
                    total_timesteps=args.total_timesteps,
                    device=args.device,
                    seed=run_seed,
                    base_dir=args.base_dir,
                    eval_episodes=args.eval_episodes,
                )

                t0 = time.time()
                summary = run_single_experiment(msc)
                t1 = time.time()

                row = {
                    "task": task,
                    "backbone": backbone,
                    "horizon": horizon,
                    "run_name": run_name,
                    "success_rate": summary.get("success_rate", float("nan")),
                    "mean_reward": summary.get("mean_reward", float("nan")),
                    "steps_per_second": summary.get("steps_per_second", float("nan")),
                    "total_updates": summary.get("total_updates", 0),
                    "total_seconds": summary.get("total_seconds", t1 - t0),
                    "final_memory_saturation": summary.get(
                        "final_memory_saturation", float("nan")
                    ),
                    "final_memory_mean_norm": summary.get(
                        "final_memory_mean_norm", float("nan")
                    ),
                    "num_envs": args.num_envs,
                    "total_timesteps": args.total_timesteps,
                    "device": args.device,
                    "seed": run_seed,
                }

                rows.append(row)

                # Append incrementally to avoid losing data on crash
                with open(args.output, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if f.tell() == 0:
                        writer.writeheader()
                    writer.writerow(row)

                print(
                    f"Completed: success_rate={row['success_rate']:.4f}, "
                    f"mean_reward={row['mean_reward']:.4f}, "
                    f"mem_sat={row['final_memory_saturation']:.4f}"
                )

    print(f"\nSweep finished. Results written to: {args.output}")


if __name__ == "__main__":
    main()
