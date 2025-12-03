#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Setting up CyborgMind Mamba+GPU Environment ===${NC}"

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}Warning: nvcc (CUDA) not found. Mamba requires CUDA for efficient kernels.${NC}"
    echo "Proceeding anyway, but Mamba installation might fail or fall back to slow path."
fi

# Create venv
if [ ! -d "venv_mamba" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv_mamba
else
    echo "Virtual environment already exists."
fi

# Activate
source venv_mamba/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA support (adjust version if needed)
echo "Installing PyTorch..."
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# Install Mamba dependencies
echo "Installing Mamba dependencies..."
pip install causal-conv1d>=1.1.0
pip install mamba-ssm>=1.1.1

# Install rest of requirements
echo "Installing core requirements..."
pip install -r requirements.txt

# Verify
echo "Verifying installation..."
python3 quick_verify.py

echo -e "${GREEN}=== Setup Complete! Activate with: source venv_mamba/bin/activate ===${NC}"
