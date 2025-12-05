#!/usr/bin/env python3
"""
Comprehensive setup verification for CyborgMind v4.0

Checks:
- Core dependencies (torch, numpy, gymnasium)
- Optional dependencies (minerl, mamba-ssm)
- API dependencies (fastapi, uvicorn)
- Model components (encoders, memory modules)
- Environment adapters
"""

import sys
from pathlib import Path

# Add current directory to path for development mode
sys.path.insert(0, str(Path(__file__).parent.parent))

# Color codes for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


def check_import(module_name, package_name=None, optional=False):
    """Check if a module can be imported."""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"{GREEN}✓{RESET} {package_name}")
        return True
    except ImportError as e:
        if optional:
            print(f"{YELLOW}⚠{RESET} {package_name} (optional) - {e}")
        else:
            print(f"{RED}✗{RESET} {package_name} - {e}")
        return False


def main():
    print(f"\n{BOLD}CyborgMind v4.0 Setup Verification{RESET}\n")
    
    all_ok = True
    
    # Core dependencies
    print(f"{BOLD}Core Dependencies:{RESET}")
    all_ok &= check_import("torch")
    all_ok &= check_import("numpy")
    all_ok &= check_import("gymnasium")
    all_ok &= check_import("tqdm")
    all_ok &= check_import("pandas")
    all_ok &= check_import("matplotlib")
    
    # RL dependencies
    print(f"\n{BOLD}RL Dependencies:{RESET}")
    all_ok &= check_import("minigrid")
    check_import("minerl", optional=True)  # Optional
    
    # Model dependencies
    print(f"\n{BOLD}Model Dependencies:{RESET}")
    all_ok &= check_import("einops")
    check_import("mamba_ssm", "mamba-ssm", optional=True)  # Optional
    check_import("causal_conv1d", "causal-conv1d", optional=True)  # Optional
    
    # API dependencies
    print(f"\n{BOLD}API Dependencies:{RESET}")
    all_ok &= check_import("fastapi")
    all_ok &= check_import("uvicorn")
    all_ok &= check_import("pydantic")
    all_ok &= check_import("slowapi")
    all_ok &= check_import("jwt", "PyJWT")
    
    # Monitoring
    print(f"\n{BOLD}Monitoring:{RESET}")
    all_ok &= check_import("prometheus_client")
    check_import("wandb", optional=True)  # Optional
    
    # Testing
    print(f"\n{BOLD}Testing:{RESET}")
    all_ok &= check_import("pytest")
    
    # CyborgMind modules
    print(f"\n{BOLD}CyborgMind Modules:{RESET}")
    all_ok &= check_import("cyborg_rl")
    all_ok &= check_import("cyborg_rl.agents")
    all_ok &= check_import("cyborg_rl.envs")
    all_ok &= check_import("cyborg_rl.models")
    all_ok &= check_import("cyborg_rl.trainers")
    all_ok &= check_import("cyborg_rl.memory")
    all_ok &= check_import("cyborg_rl.memory_benchmarks")
    
    # Check key files
    print(f"\n{BOLD}Key Files:{RESET}")
    key_files = [
        "configs/treechop_ppo.yaml",
        "scripts/run_treechop_pipeline.py",
        "scripts/run_api_server.py",
        "cyborg_rl/server.py",
    ]
    
    for file_path in key_files:
        full_path = Path(file_path)
        if full_path.exists():
            print(f"{GREEN}✓{RESET} {file_path}")
        else:
            print(f"{RED}✗{RESET} {file_path} - Not found")
            all_ok = False
    
    # Summary
    print(f"\n{BOLD}{'='*60}{RESET}")
    if all_ok:
        print(f"{GREEN}{BOLD}✓ All required dependencies are installed!{RESET}")
        print(f"\nYou can now:")
        print(f"  1. Run memory benchmarks: python -m cyborg_rl.memory_benchmarks.pseudo_mamba_memory_suite --help")
        print(f"  2. Train MineRL agent: python scripts/run_treechop_pipeline.py --help")
        print(f"  3. Start API server: python scripts/run_api_server.py --help")
        return 0
    else:
        print(f"{RED}{BOLD}✗ Some required dependencies are missing!{RESET}")
        print(f"\nTo install:")
        print(f"  - Core dependencies: pip install -r requirements.txt")
        print(f"  - MineRL (optional): ./setup_minerl.sh")
        print(f"  - Mamba GPU (optional): ./setup_mamba_gpu.sh")
        return 1


if __name__ == "__main__":
    sys.exit(main())
