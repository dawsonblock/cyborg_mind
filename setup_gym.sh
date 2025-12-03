#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}=== Setting up CyborgMind Gym Environment ===${NC}"

# Check for python3
if ! command -v python3 &> /dev/null; then
    echo "Python 3 could not be found"
    exit 1
fi

# Create venv
if [ ! -d "venv_gym" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv_gym
else
    echo "Virtual environment already exists."
fi

# Activate
source venv_gym/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Verify
echo "Verifying installation..."
python3 quick_verify.py

echo -e "${GREEN}=== Setup Complete! Activate with: source venv_gym/bin/activate ===${NC}"
