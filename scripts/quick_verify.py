#!/usr/bin/env python3
"""
Quick verification script for CyborgMind environment.
Hard fails on core modules, soft warns on optional ones.
"""

import sys
import importlib

def check_module(name: str, optional: bool = False) -> bool:
    try:
        importlib.import_module(name)
        print(f"✅ {name}")
        return True
    except ImportError as e:
        if optional:
            print(f"⚠️  {name} (Optional) - Missing: {e}")
            return True
        else:
            print(f"❌ {name} (CRITICAL) - Missing: {e}")
            return False

def main():
    print("=== CyborgMind Quick Verify ===")
    
    # Core Dependencies
    core_ok = True
    core_ok &= check_module("torch")
    core_ok &= check_module("gymnasium")
    core_ok &= check_module("cyborg_rl")
    
    # Optional Dependencies
    check_module("transformers", optional=True)
    check_module("minerl", optional=True)
    check_module("mamba_ssm", optional=True)

    if not core_ok:
        print("\n❌ Critical dependencies missing. Setup failed.")
        sys.exit(1)
    
    print("\n✅ Core system verified.")
    sys.exit(0)

if __name__ == "__main__":
    main()
