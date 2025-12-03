#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}=== Setting up CyborgMind MineRL Environment ===${NC}"

# Create venv
if [ ! -d "venv_minerl" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv_minerl
else
    echo "Virtual environment already exists."
fi

# Activate
source venv_minerl/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Java (required for MineRL)
if ! command -v java &> /dev/null; then
    echo "Java not found. Please install JDK 8 or higher."
    # On Ubuntu: sudo apt install openjdk-8-jdk
    # On Mac: brew install openjdk@8
fi

# Install MineRL specific deps
echo "Installing MineRL..."
pip install minerl==0.4.4
pip install gym==0.19.0  # MineRL often needs older gym

# Install core requirements (excluding conflicting gym)
# We filter out gymnasium/gym from requirements.txt for this env
grep -v "gymnasium" requirements.txt > requirements_minerl.txt
pip install -r requirements_minerl.txt
rm requirements_minerl.txt

# Verify
echo "Verifying installation..."
python3 -c "import minerl; import gym; print('MineRL + Gym imported successfully')"

echo -e "${GREEN}=== Setup Complete! Activate with: source venv_minerl/bin/activate ===${NC}"
