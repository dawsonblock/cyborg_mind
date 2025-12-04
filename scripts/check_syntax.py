#!/usr/bin/env python3
"""
Simple syntax and import check for critical files.
This doesn't require torch/gym to be installed.
"""

import sys
import ast
from pathlib import Path

def check_syntax(filepath):
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)

def main():
    """Check syntax of all critical files."""
    files_to_check = [
        "cyborg_rl/config.py",
        "cyborg_rl/trainers/ppo_trainer.py",
        "cyborg_rl/trainers/rollout_buffer.py",
        "cyborg_rl/trainers/recurrent_rollout_buffer.py",
        "cyborg_rl/agents/ppo_agent.py",
        "cyborg_rl/memory/pmm.py",
        "cyborg_rl/memory_benchmarks/pseudo_mamba_memory_suite.py",
        "cyborg_rl/memory_benchmarks/delayed_cue_env.py",
        "cyborg_rl/memory_benchmarks/copy_memory_env.py",
        "cyborg_rl/memory_benchmarks/associative_recall_env.py",
        "scripts/run_memory_sweep.py",
    ]

    print("=" * 80)
    print("SYNTAX CHECK")
    print("=" * 80)

    all_passed = True
    for filepath in files_to_check:
        path = Path(filepath)
        if not path.exists():
            print(f"✗ {filepath}: FILE NOT FOUND")
            all_passed = False
            continue

        passed, error = check_syntax(path)
        if passed:
            print(f"✓ {filepath}")
        else:
            print(f"✗ {filepath}: {error}")
            all_passed = False

    print("=" * 80)
    if all_passed:
        print("✓ ALL SYNTAX CHECKS PASSED")
        return 0
    else:
        print("✗ SOME SYNTAX CHECKS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
