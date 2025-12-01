"""
Example: Training CyborgMind on MineRLTreechop-v0
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
    # Note: MineRL requires a display or headless setup.
    test_args = ["launcher.py", "train", "--env", "MineRLTreechop-v0", "--steps", "1000000"]
    with patch.object(sys, 'argv', test_args):
        launcher_main()
