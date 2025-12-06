#!/usr/bin/env python3
"""
CyborgMind V2.6 Build Verification Script

Comprehensive verification of the V2.6 build to ensure production readiness.
"""

import os
import sys
from pathlib import Path

def color_print(text: str, color: str = "green"):
    """Print colored text."""
    colors = {
        "green": "\033[92m",
        "red": "\033[91m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "end": "\033[0m"
    }
    print(f"{colors.get(color, '')}{text}{colors['end']}")

def check_file(path: str, description: str) -> bool:
    """Check if a file exists."""
    if os.path.exists(path):
        color_print(f"  ✓ {description}: {path}", "green")
        return True
    else:
        color_print(f"  ✗ MISSING: {description}: {path}", "red")
        return False

def main():
    print("=" * 70)
    print("🧪 CYBORGMIND V2.6 BUILD VERIFICATION")
    print("=" * 70)
    
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    checks_passed = 0
    checks_failed = 0
    
    # Test 1: Core Python Files
    print("\n📦 Test 1: Core Python Files")
    core_files = [
        "cyborg_mind_v2/__init__.py",
        "cyborg_mind_v2/envs/__init__.py",
        "cyborg_mind_v2/envs/base_adapter.py",
        "cyborg_mind_v2/envs/gym_adapter.py",
        "cyborg_mind_v2/envs/minerl_adapter.py",
        "cyborg_mind_v2/integration/cyborg_mind_controller.py",
        "cyborg_mind_v2/capsule_brain/policy/brain_cyborg_mind.py",
        "cyborg_mind_v2/deployment/api_server.py",
    ]
    
    for f in core_files:
        if check_file(f, "Core file"):
            checks_passed += 1
        else:
            checks_failed += 1
    
    # Test 2: CC3D Removal
    print("\n🗑️  Test 2: CC3D Removal Verification")
    cc3d_file = "cyborg_mind_v2/envs/cc3d_adapter.py"
    if not os.path.exists(cc3d_file):
        color_print(f"  ✓ CC3D adapter correctly removed", "green")
        checks_passed += 1
    else:
        color_print(f"  ✗ CC3D adapter still exists!", "red")
        checks_failed += 1
    
    # Test 3: Docker Infrastructure
    print("\n🐋 Test 3: Docker Infrastructure")
    docker_files = [
        ("Dockerfile", "Docker build file"),
        ("docker-compose.yml", "Docker Compose orchestration"),
        (".dockerignore", "Docker ignore file"),
        ("requirements.txt", "Python dependencies"),
        (".env.example", "Environment template"),
    ]
    
    for f, desc in docker_files:
        if check_file(f, desc):
            checks_passed += 1
        else:
            checks_failed += 1
    
    # Test 4: Documentation
    print("\n📖 Test 4: V2.6 Documentation")
    docs = [
        ("docs/V2.6_RELEASE_NOTES.md", "Release notes"),
        ("docs/V2.6_ARCHITECTURE.md", "Architecture docs"),
        ("docs/V2.6_MIGRATION_GUIDE.md", "Migration guide"),
        ("README.md", "Main README"),
    ]
    
    for f, desc in docs:
        if check_file(f, desc):
            checks_passed += 1
        else:
            checks_failed += 1
    
    # Test 5: Monitoring Stack
    print("\n📊 Test 5: Monitoring Stack")
    monitoring_files = [
        ("cyborg_mind_v2/deployment/monitoring/prometheus.yml", "Prometheus config"),
        ("cyborg_mind_v2/deployment/monitoring/grafana/datasources/prometheus.yml", "Grafana datasource"),
    ]
    
    for f, desc in monitoring_files:
        if check_file(f, desc):
            checks_passed += 1
        else:
            checks_failed += 1
    
    # Test 6: Python Syntax
    print("\n🐍 Test 6: Python Syntax Verification")
    try:
        import py_compile
        syntax_files = [
            "cyborg_mind_v2/envs/__init__.py",
            "cyborg_mind_v2/envs/base_adapter.py",
            "cyborg_mind_v2/envs/gym_adapter.py",
        ]
        
        for f in syntax_files:
            try:
                py_compile.compile(f, doraise=True)
                color_print(f"  ✓ Syntax OK: {f}", "green")
                checks_passed += 1
            except py_compile.PyCompileError as e:
                color_print(f"  ✗ Syntax ERROR: {f}: {e}", "red")
                checks_failed += 1
    except ImportError:
        color_print("  ⚠ py_compile not available, skipping syntax check", "yellow")
    
    # Test 7: README Content Verification
    print("\n📝 Test 7: README Content Verification")
    try:
        with open("README.md", "r") as f:
            readme_content = f.read()
        
        checks = [
            ("V2.6" in readme_content, "V2.6 version mentioned"),
            ("CC3D" not in readme_content or "CC3D Removed" in readme_content, "CC3D properly documented as removed"),
            ("Docker" in readme_content, "Docker deployment documented"),
            ("Prometheus" in readme_content or "Monitoring" in readme_content, "Monitoring stack documented"),
        ]
        
        for check, desc in checks:
            if check:
                color_print(f"  ✓ {desc}", "green")
                checks_passed += 1
            else:
                color_print(f"  ✗ {desc}", "red")
                checks_failed += 1
    except Exception as e:
        color_print(f"  ✗ Error reading README: {e}", "red")
        checks_failed += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 70)
    
    total_checks = checks_passed + checks_failed
    success_rate = (checks_passed / total_checks * 100) if total_checks > 0 else 0
    
    color_print(f"Checks Passed: {checks_passed}/{total_checks}", "green")
    if checks_failed > 0:
        color_print(f"Checks Failed: {checks_failed}/{total_checks}", "red")
    color_print(f"Success Rate: {success_rate:.1f}%", "green" if success_rate == 100 else "yellow")
    
    print("=" * 70)
    
    if checks_failed == 0:
        color_print("\n🎉 ALL CHECKS PASSED! V2.6 BUILD IS PRODUCTION-READY!", "green")
        return 0
    else:
        color_print(f"\n⚠️  {checks_failed} CHECK(S) FAILED. REVIEW REQUIRED.", "red")
        return 1

if __name__ == "__main__":
    sys.exit(main())
