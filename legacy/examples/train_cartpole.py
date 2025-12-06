"""
Example: Training CyborgMind on CartPole-v1
"""
import os
import sys

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from launcher import main as launcher_main
from unittest.mock import patch
import sys

if __name__ == "__main__":
    # Simulate CLI arguments
    test_args = ["launcher.py", "train", "--env", "CartPole-v1", "--steps", "100000"]
    with patch.object(sys, 'argv', test_args):
        launcher_main()
