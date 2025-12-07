#!/bin/bash
# ============================================================================
# cleanup_v5.sh - CyborgMind v5.0.0 Repository Standardization Script
# ============================================================================
#
# This script performs:
# 1. Legacy code purge
# 2. Version unification to 5.0.0
# 3. Dependency sanitization
# 4. Entry point standardization
# 5. Config cleanup
#
# Usage:
#   chmod +x cleanup_v5.sh
#   ./cleanup_v5.sh
#
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

VERSION="5.0.0"

echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}  CyborgMind v${VERSION} Repository Cleanup Script${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""

# ============================================================================
# 1. PURGE LEGACY CODE
# ============================================================================
echo -e "${YELLOW}[1/5] Purging Legacy Code...${NC}"

# Delete legacy/ directory
if [ -d "legacy" ]; then
    echo "  Removing legacy/ directory..."
    rm -rf legacy/
    echo -e "  ${GREEN}✓ legacy/ removed${NC}"
else
    echo -e "  ${GREEN}✓ legacy/ already removed${NC}"
fi

# Remove experimental adapters (non-MineRL, non-Gym)
# Note: Currently envs/ only has minerl_adapter.py and base.py - no cleanup needed
ADAPTERS_TO_REMOVE=(
    "cyborg_rl/envs/trading_adapter.py"
    "cyborg_rl/envs/eeg_adapter.py"
    "cyborg_rl/envs/lab_adapter.py"
    "cyborg_rl/envs/trading/"
    "cyborg_rl/envs/eeg/"
    "cyborg_rl/envs/lab/"
)

for adapter in "${ADAPTERS_TO_REMOVE[@]}"; do
    if [ -e "$adapter" ]; then
        echo "  Removing $adapter..."
        rm -rf "$adapter"
        echo -e "  ${GREEN}✓ $adapter removed${NC}"
    fi
done

# Remove non-MineRL/PPO configs
CONFIGS_TO_REMOVE=(
    "configs/trading*.yaml"
    "configs/eeg*.yaml"
    "configs/lab*.yaml"
    "configs/atari*.yaml"
)

for pattern in "${CONFIGS_TO_REMOVE[@]}"; do
    for file in $pattern; do
        if [ -f "$file" ]; then
            echo "  Removing $file..."
            rm -f "$file"
            echo -e "  ${GREEN}✓ $file removed${NC}"
        fi
    done
done

echo ""

# ============================================================================
# 2. UNIFY VERSIONING TO 5.0.0
# ============================================================================
echo -e "${YELLOW}[2/5] Unifying Version to ${VERSION}...${NC}"

# Update pyproject.toml
if [ -f "pyproject.toml" ]; then
    sed -i.bak 's/version = "[0-9]\+\.[0-9]\+\.[0-9]\+"/version = "'"${VERSION}"'"/' pyproject.toml
    rm -f pyproject.toml.bak
    echo -e "  ${GREEN}✓ pyproject.toml updated${NC}"
fi

# Update cyborg_rl/__init__.py
if [ -f "cyborg_rl/__init__.py" ]; then
    sed -i.bak 's/__version__ = "[0-9]\+\.[0-9]\+\.[0-9]\+"/__version__ = "'"${VERSION}"'"/' cyborg_rl/__init__.py
    rm -f cyborg_rl/__init__.py.bak
    echo -e "  ${GREEN}✓ cyborg_rl/__init__.py updated${NC}"
fi

# Update README.md version badges/references
if [ -f "README.md" ]; then
    sed -i.bak 's/v[0-9]\+\.[0-9]\+\.[0-9]\+/v'"${VERSION}"'/g' README.md
    rm -f README.md.bak
    echo -e "  ${GREEN}✓ README.md updated${NC}"
fi

# Update docs/API.md
if [ -f "docs/API.md" ]; then
    sed -i.bak 's/"version": "[0-9]\+\.[0-9]\+\.[0-9]\+"/"version": "'"${VERSION}"'"/' docs/API.md
    rm -f docs/API.md.bak
    echo -e "  ${GREEN}✓ docs/API.md updated${NC}"
fi

echo ""

# ============================================================================
# 3. SANITIZE DEPENDENCIES
# ============================================================================
echo -e "${YELLOW}[3/5] Sanitizing Dependencies...${NC}"

# Create requirements-core.txt
cat > requirements-core.txt << 'EOF'
# CyborgMind Core Dependencies (Lean ML Stack)
# Install: pip install -r requirements-core.txt

# PyTorch (core ML)
torch>=2.1.0
torchvision>=0.16.0

# Numerics
numpy>=1.24.0
einops>=0.7.0

# API & Server
fastapi>=0.109.0
uvicorn>=0.27.0
pydantic>=2.5.0
python-multipart>=0.0.6
slowapi>=0.1.9
PyJWT>=2.8.0

# Monitoring & Logging
wandb>=0.16.0
tensorboard>=2.15.0
prometheus-client>=0.19.0
tqdm>=4.66.0

# Config
pyyaml>=6.0

# Mamba SSM (optional - uncomment for GPU systems)
# causal-conv1d>=1.1.0
# mamba-ssm>=1.1.1
EOF
echo -e "  ${GREEN}✓ requirements-core.txt created${NC}"

# Create requirements-minerl.txt
cat > requirements-minerl.txt << 'EOF'
# CyborgMind MineRL Dependencies (Heavy Environment Stack)
# Install: pip install -r requirements-minerl.txt
#
# Prerequisites:
# - Java 8 (OpenJDK 8): brew install openjdk@8 (macOS) or apt install openjdk-8-jdk (Ubuntu)
# - Set JAVA_HOME to JDK 8 path
#
# Note: MineRL is sensitive to dependency versions. Use exactly as specified.

# Include core first
-r requirements-core.txt

# MineRL Environment
minerl>=0.4.4

# Image processing for observations
opencv-python-headless>=4.8.0
Pillow>=10.0.0

# Legacy gym compatibility (required by MineRL 0.4.x)
gym==0.21.0
# Note: gym 0.21.0 conflicts with gymnasium. 
# MineRL adapter handles the compatibility layer.
EOF
echo -e "  ${GREEN}✓ requirements-minerl.txt created${NC}"

# Create requirements-dev.txt
cat > requirements-dev.txt << 'EOF'
# CyborgMind Development Dependencies
# Install: pip install -r requirements-dev.txt

-r requirements-core.txt

# Testing
pytest>=8.0.0
pytest-cov>=4.1.0
pytest-asyncio>=0.23.0

# Linting & Formatting
black>=24.1.0
ruff>=0.1.0
mypy>=1.8.0
isort>=5.13.0

# Notebooks
jupyter>=1.0.0
ipython>=8.18.0
matplotlib>=3.8.0

# Documentation
mkdocs>=1.5.0
mkdocs-material>=9.5.0
EOF
echo -e "  ${GREEN}✓ requirements-dev.txt created${NC}"

echo ""

# ============================================================================
# 4. STANDARDIZE ENTRY POINTS
# ============================================================================
echo -e "${YELLOW}[4/5] Standardizing Entry Points...${NC}"

# Rename mine_rl_train.py to train.py
if [ -f "mine_rl_train.py" ]; then
    mv mine_rl_train.py train.py
    echo -e "  ${GREEN}✓ mine_rl_train.py → train.py${NC}"
    
    # Ensure shebang is correct
    if ! head -1 train.py | grep -q "#!/usr/bin/env python3"; then
        sed -i.bak '1s|^|#!/usr/bin/env python3\n|' train.py
        rm -f train.py.bak
        echo -e "  ${GREEN}✓ Added shebang to train.py${NC}"
    fi
fi

# Make scripts executable
chmod +x train.py 2>/dev/null || true
chmod +x train_advanced.py 2>/dev/null || true
chmod +x train_production.py 2>/dev/null || true
chmod +x evaluate_minerl_agent.py 2>/dev/null || true
echo -e "  ${GREEN}✓ Entry point scripts made executable${NC}"

echo ""

# ============================================================================
# 5. CONFIG CLEANUP
# ============================================================================
echo -e "${YELLOW}[5/5] Cleaning Configs...${NC}"

# Check for Hydra/OmegaConf usage in non-legacy files
HYDRA_USAGE=$(grep -r "hydra" --include="*.py" --exclude-dir=legacy --exclude-dir=.git . 2>/dev/null | grep -v "# hydra" || true)
OMEGACONF_USAGE=$(grep -r "omegaconf" --include="*.py" --exclude-dir=legacy --exclude-dir=.git . 2>/dev/null | grep -v "# omegaconf" || true)

if [ -z "$HYDRA_USAGE" ] && [ -z "$OMEGACONF_USAGE" ]; then
    echo -e "  ${GREEN}✓ No Hydra/OmegaConf dependencies in core code${NC}"
else
    echo -e "  ${YELLOW}⚠ Found Hydra/OmegaConf usage (may need manual review):${NC}"
    echo "$HYDRA_USAGE"
    echo "$OMEGACONF_USAGE"
fi

# Verify config.py exists
if [ -f "cyborg_rl/utils/config.py" ]; then
    echo -e "  ${GREEN}✓ cyborg_rl/utils/config.py exists${NC}"
else
    echo -e "  ${RED}✗ cyborg_rl/utils/config.py not found${NC}"
fi

echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo -e "${BLUE}============================================================================${NC}"
echo -e "${GREEN}  Cleanup Complete! Repository is now at v${VERSION}${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo "Changes made:"
echo "  • Removed legacy/ directory"
echo "  • Updated all version strings to ${VERSION}"
echo "  • Created requirements-core.txt (lean ML stack)"
echo "  • Created requirements-minerl.txt (MineRL environment)"
echo "  • Created requirements-dev.txt (development tools)"
echo "  • Renamed mine_rl_train.py → train.py"
echo ""
echo "Next steps:"
echo "  1. Review changes: git diff"
echo "  2. Commit: git add . && git commit -m 'Refactor: v${VERSION} cleanup'"
echo "  3. Push: git push"
echo ""
echo -e "${GREEN}Done!${NC}"
