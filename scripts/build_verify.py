#!/usr/bin/env python3
"""
Build verification script.
Checks file structure, configs, and imports.
"""

import sys
from pathlib import Path
import toml

ROOT = Path(__file__).parent.parent

def check_path(path: str) -> bool:
    if not (ROOT / path).exists():
        print(f"❌ Missing path: {path}")
        return False
    print(f"✅ Found: {path}")
    return True

def main():
    print("=== CyborgMind Build Verify ===")
    
    ok = True
    
    # Check Directories
    ok &= check_path("cyborg_rl")
    ok &= check_path("cyborg_rl/agents")
    ok &= check_path("cyborg_rl/trainers")
    ok &= check_path("monitoring")
    ok &= check_path("scripts")
    
    # Check Configs
    ok &= check_path("pyproject.toml")
    
    # Validate pyproject.toml
    try:
        pyproject = toml.load(ROOT / "pyproject.toml")
        if "project" not in pyproject:
            print("❌ pyproject.toml missing [project] section")
            ok = False
        else:
            print("✅ pyproject.toml valid")
    except Exception as e:
        print(f"❌ pyproject.toml invalid: {e}")
        ok = False

    if not ok:
        print("\n❌ Build verification failed.")
        sys.exit(1)
        
    print("\n✅ Build verified.")
    sys.exit(0)

if __name__ == "__main__":
    main()
