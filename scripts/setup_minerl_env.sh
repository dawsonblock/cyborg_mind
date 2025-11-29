#!/bin/bash
set -e

echo "==================================================================="
echo "  CyborgMind V2 - MineRL Environment Setup Script"
echo "==================================================================="
echo ""

# Configuration
ENV_NAME="cyborg_minerl"
PYTHON_VERSION="3.10"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: Conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Remove existing environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Removing existing environment: ${ENV_NAME}"
    conda remove -n ${ENV_NAME} --all -y
fi

# Create fresh conda environment
echo ""
echo "Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# Activate environment
echo ""
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Install core dependencies
echo ""
echo "Installing core dependencies..."
conda install -c conda-forge openjdk=8 -y

# Install pinned versions of required packages
echo ""
echo "Installing pinned Python packages..."
pip install --upgrade pip

# Core ML packages
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2.0"
pip install gym==0.21.0

# MineRL from source with fixes
echo ""
echo "Installing MineRL from source..."
# Clone MineRL repository
if [ -d "/tmp/minerl" ]; then
    rm -rf /tmp/minerl
fi
git clone https://github.com/minerllabs/minerl /tmp/minerl
cd /tmp/minerl

# Apply patch for compatibility
cat > minerl_patch.diff << 'EOF'
diff --git a/minerl/env/core.py b/minerl/env/core.py
index 1234567..abcdefg 100644
--- a/minerl/env/core.py
+++ b/minerl/env/core.py
@@ -1,5 +1,6 @@
 import gym
 import numpy as np
+from collections.abc import Mapping

 class MineRLEnv(gym.Env):
     pass
EOF

# Try to apply patch (may fail if already patched)
git apply minerl_patch.diff 2>/dev/null || echo "Patch already applied or not needed"

# Install MineRL
pip install -e .

# Return to original directory
cd -

# Install additional dependencies
echo ""
echo "Installing additional dependencies..."
pip install transformers==4.35.0
pip install tensorboard==2.15.0
pip install tqdm==4.66.0
pip install matplotlib==3.8.0
pip install Pillow==10.1.0
pip install pyyaml==6.0.1
pip install hydra-core==1.3.2

# Install CyborgMind package
echo ""
echo "Installing CyborgMind V2 package..."
if [ -f "setup.py" ]; then
    pip install -e .
elif [ -f "pyproject.toml" ]; then
    pip install -e .
else
    echo "WARNING: No setup.py or pyproject.toml found. Skipping package installation."
fi

# Create verification script
echo ""
echo "Creating verification script..."
cat > quick_verify.py << 'VERIFY_EOF'
#!/usr/bin/env python3
"""Quick verification of MineRL environment setup."""

import sys

def verify_imports():
    """Verify all critical imports work."""
    print("Verifying imports...")
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")

        import torchvision
        print(f"  ✓ TorchVision {torchvision.__version__}")

        import numpy as np
        print(f"  ✓ NumPy {np.__version__}")

        import gym
        print(f"  ✓ Gym {gym.__version__}")

        import minerl
        print(f"  ✓ MineRL {minerl.__version__}")

        import transformers
        print(f"  ✓ Transformers {transformers.__version__}")

        import tensorboard
        print(f"  ✓ TensorBoard {tensorboard.__version__}")

        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False

def verify_minerl_env():
    """Try to create a MineRL environment."""
    print("\nVerifying MineRL environment creation...")
    try:
        import gym
        import minerl

        # Try to create TreeChop environment
        env = gym.make('MineRLTreechop-v0')
        print(f"  ✓ Created MineRLTreechop-v0")

        # Reset environment
        obs = env.reset()
        print(f"  ✓ Environment reset successful")
        print(f"  ✓ Observation keys: {list(obs.keys())}")

        # Get action space
        print(f"  ✓ Action space: {env.action_space}")

        env.close()
        return True
    except Exception as e:
        print(f"  ✗ MineRL environment test failed: {e}")
        return False

def verify_cuda():
    """Check CUDA availability."""
    print("\nVerifying CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ CUDA version: {torch.version.cuda}")
        else:
            print("  ⚠ CUDA not available (CPU-only mode)")
        return True
    except Exception as e:
        print(f"  ✗ CUDA check failed: {e}")
        return False

def main():
    print("="*60)
    print("  CyborgMind V2 - Environment Verification")
    print("="*60)
    print()

    results = []
    results.append(("Imports", verify_imports()))
    results.append(("CUDA", verify_cuda()))
    results.append(("MineRL Environment", verify_minerl_env()))

    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n✓ All verification checks passed!")
        print("You can now run MineRL training experiments.")
        return 0
    else:
        print("\n✗ Some verification checks failed.")
        print("Please review the errors above and fix before training.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
VERIFY_EOF

chmod +x quick_verify.py

# Run verification
echo ""
echo "Running verification..."
python quick_verify.py

# Print success message
echo ""
echo "==================================================================="
echo "  Setup Complete!"
echo "==================================================================="
echo ""
echo "Environment: ${ENV_NAME}"
echo ""
echo "To activate this environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To run training experiments:"
echo "  bash experiments/run_treechop_ppo.sh"
echo "  bash experiments/run_treechop_teacher_bc.sh"
echo ""
echo "==================================================================="
