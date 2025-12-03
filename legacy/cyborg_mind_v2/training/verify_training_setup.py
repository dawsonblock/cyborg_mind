#!/usr/bin/env python3
# cyborg_mind_v2/training/verify_training_setup.py

"""
Verification script for training setup.
Checks all prerequisites before running actual training.

Run this before attempting any training to catch configuration issues early.
"""

import sys
import os
import subprocess
from typing import Dict, List, Tuple


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print a section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")


def check_java() -> bool:
    """Check if Java is installed and accessible"""
    print_header("1. Checking Java Installation")
    try:
        result = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Java version is printed to stderr
            version_output = result.stderr.split('\n')[0]
            print_success(f"Java found: {version_output}")
            return True
        else:
            print_error("Java command failed")
            return False
    except FileNotFoundError:
        print_error("Java not found in PATH")
        print_warning("MineRL requires Java to be installed and accessible")
        print_warning("Download from: https://www.oracle.com/java/technologies/downloads/")
        return False
    except subprocess.TimeoutExpired:
        print_error("Java command timed out")
        return False


def check_python_packages() -> Dict[str, bool]:
    """Check if required Python packages are installed with correct versions"""
    print_header("2. Checking Python Packages")
    
    required_packages = {
        "torch": None,  # Any version
        "torchvision": None,
        "numpy": None,
        "cv2": "opencv-python",  # cv2 is imported from opencv-python
        "gym": "0.21.0",  # Specific version required
        "minerl": "0.4.4",  # Specific version required
        "transformers": None,
    }
    
    results = {}
    
    for package, expected_version in required_packages.items():
        import_name = package if package != "cv2" else "cv2"
        package_name = required_packages[package] if package == "cv2" else package
        
        try:
            if package == "cv2":
                import cv2
                version = cv2.__version__
            elif package == "gym":
                import gym
                version = gym.__version__
            elif package == "minerl":
                import minerl
                version = minerl.__version__
            elif package == "torch":
                import torch
                version = torch.__version__
            elif package == "torchvision":
                import torchvision
                version = torchvision.__version__
            elif package == "numpy":
                import numpy
                version = numpy.__version__
            elif package == "transformers":
                import transformers
                version = transformers.__version__
            else:
                exec(f"import {package}")
                version = "unknown"
            
            if expected_version and version != expected_version:
                print_warning(f"{package}: version {version} (expected {expected_version})")
                results[package] = False
            else:
                print_success(f"{package}: {version}")
                results[package] = True
                
        except ImportError:
            print_error(f"{package}: NOT INSTALLED")
            if expected_version:
                print_warning(f"  Install with: pip install {package_name}=={expected_version}")
            else:
                print_warning(f"  Install with: pip install {package_name}")
            results[package] = False
    
    return results


def check_cuda() -> bool:
    """Check CUDA availability"""
    print_header("3. Checking CUDA/GPU Support")
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            print_success(f"CUDA available: {device_count} device(s)")
            
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                print_success(f"  Device {i}: {device_name}")
            
            # Test basic CUDA operation
            try:
                x = torch.tensor([1.0]).cuda()
                print_success("CUDA test tensor created successfully")
                return True
            except Exception as e:
                print_error(f"CUDA test failed: {e}")
                return False
        else:
            print_warning("CUDA not available - training will use CPU (very slow)")
            print_warning("Check NVIDIA drivers and PyTorch CUDA installation")
            return False
            
    except ImportError:
        print_error("PyTorch not installed")
        return False


def check_project_structure() -> Dict[str, bool]:
    """Check if required project files exist"""
    print_header("4. Checking Project Structure")
    
    required_files = [
        "cyborg_mind_v2/envs/action_mapping.py",
        "cyborg_mind_v2/envs/minerl_obs_adapter.py",
        "cyborg_mind_v2/training/real_teacher.py",
        "cyborg_mind_v2/capsule_brain/policy/brain_cyborg_mind.py",
    ]
    
    results = {}
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print_success(f"{file_path}")
            results[file_path] = True
        else:
            print_error(f"{file_path} - NOT FOUND")
            results[file_path] = False
    
    return results


def check_brain_api() -> bool:
    """Check if BrainCyborgMind has the expected API"""
    print_header("5. Checking BrainCyborgMind API")
    
    try:
        from experiments.cyborg_mind_v2.capsule_brain.policy.brain_cyborg_mind import BrainCyborgMind
        import torch
        
        brain = BrainCyborgMind()
        
        # Create dummy inputs
        B = 2
        pixels = torch.zeros(B, 3, 128, 128)
        scalars = torch.zeros(B, 20)
        goal = torch.zeros(B, 4)
        thought = torch.zeros(B, 32)
        emotion = torch.zeros(B, 8)
        workspace = torch.zeros(B, 64)
        h0 = torch.zeros(1, B, 512)
        c0 = torch.zeros(1, B, 512)
        
        # Test forward pass
        with torch.no_grad():
            out = brain(
                pixels=pixels,
                scalars=scalars,
                goal=goal,
                thought=thought,
                emotion=emotion,
                workspace=workspace,
                hidden=(h0, c0),
            )
        
        # Check required keys
        required_keys = [
            "action_logits", "value", "mem_write", "thought",
            "emotion", "workspace", "hidden_h", "hidden_c", "pressure"
        ]
        
        missing_keys = []
        for key in required_keys:
            if key not in out:
                missing_keys.append(key)
        
        if missing_keys:
            print_error(f"Missing output keys: {missing_keys}")
            return False
        
        # Check shapes
        if out["action_logits"].shape[0] != B:
            print_error(f"action_logits batch size mismatch: {out['action_logits'].shape}")
            return False
        
        if out["value"].shape[0] != B:
            print_error(f"value batch size mismatch: {out['value'].shape}")
            return False
        
        if out["hidden_h"].shape != (1, B, 512):
            print_error(f"hidden_h shape mismatch: {out['hidden_h'].shape}")
            return False
        
        if out["hidden_c"].shape != (1, B, 512):
            print_error(f"hidden_c shape mismatch: {out['hidden_c'].shape}")
            return False
        
        print_success("BrainCyborgMind API is correct")
        print_success(f"  action_logits shape: {out['action_logits'].shape}")
        print_success(f"  value shape: {out['value'].shape}")
        print_success(f"  hidden_h shape: {out['hidden_h'].shape}")
        print_success(f"  hidden_c shape: {out['hidden_c'].shape}")
        return True
        
    except ImportError as e:
        print_error(f"Failed to import BrainCyborgMind: {e}")
        return False
    except Exception as e:
        print_error(f"API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_action_mapping() -> bool:
    """Check action mapping consistency"""
    print_header("6. Checking Action Mapping")
    
    try:
        from experiments.cyborg_mind_v2.envs.action_mapping import NUM_ACTIONS, index_to_minerl_action
        
        print_success(f"NUM_ACTIONS = {NUM_ACTIONS}")
        
        # Test a few action indices
        test_indices = [0, 1, NUM_ACTIONS - 1]
        for idx in test_indices:
            action = index_to_minerl_action(idx)
            print_success(f"  Action {idx}: {list(action.keys())}")
        
        return True
        
    except ImportError as e:
        print_error(f"Failed to import action_mapping: {e}")
        return False
    except Exception as e:
        print_error(f"Action mapping test failed: {e}")
        return False


def check_env_adapter() -> bool:
    """Check observation adapter"""
    print_header("7. Checking Observation Adapter")
    
    try:
        from experiments.cyborg_mind_v2.envs.minerl_obs_adapter import obs_to_brain
        import numpy as np
        
        # Create dummy MineRL observation
        dummy_obs = {
            "pov": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        }
        
        pixels, scalars, goal = obs_to_brain(dummy_obs)
        
        if pixels.shape != (3, 128, 128):
            print_error(f"pixels shape incorrect: {pixels.shape} (expected (3, 128, 128))")
            return False
        
        if scalars.shape != (20,):
            print_error(f"scalars shape incorrect: {scalars.shape} (expected (20,))")
            return False
        
        if goal.shape != (4,):
            print_error(f"goal shape incorrect: {goal.shape} (expected (4,))")
            return False
        
        if pixels.max() > 1.0 or pixels.min() < 0.0:
            print_error(f"pixels not normalized: min={pixels.min()}, max={pixels.max()}")
            return False
        
        print_success("obs_to_brain working correctly")
        print_success(f"  pixels: {pixels.shape}, dtype={pixels.dtype}, range=[{pixels.min():.3f}, {pixels.max():.3f}]")
        print_success(f"  scalars: {scalars.shape}, dtype={scalars.dtype}")
        print_success(f"  goal: {goal.shape}, dtype={goal.dtype}")
        return True
        
    except ImportError as e:
        print_error(f"Failed to import obs_to_brain: {e}")
        return False
    except Exception as e:
        print_error(f"Observation adapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_minerl_env() -> bool:
    """Test basic MineRL environment creation"""
    print_header("8. Testing MineRL Environment")
    
    try:
        import gym
        
        print("Attempting to create MineRLTreechop-v0 environment...")
        print("(This may take a moment and might download data)")
        
        env = gym.make("MineRLTreechop-v0")
        print_success("Environment created successfully")
        
        print("Testing env.reset()...")
        obs = env.reset()
        print_success(f"env.reset() successful, obs keys: {list(obs.keys())}")
        
        env.close()
        print_success("Environment test complete")
        return True
        
    except Exception as e:
        print_error(f"MineRL environment test failed: {e}")
        print_warning("This is often caused by Java issues or MineRL installation problems")
        import traceback
        traceback.print_exc()
        return False


def print_summary(results: Dict[str, bool]):
    """Print summary of all checks"""
    print_header("VERIFICATION SUMMARY")
    
    all_passed = all(results.values())
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total} checks\n")
    
    if all_passed:
        print_success("All checks passed! You're ready to start training.")
        print(f"\n{Colors.BOLD}Next steps:{Colors.RESET}")
        print("1. Run BC training:")
        print("   python -m cyborg_mind_v2.training.train_real_teacher_bc --epochs 1")
        print("\n2. Monitor with TensorBoard:")
        print("   tensorboard --logdir runs/real_teacher_bc")
        print("\n3. Run PPO training:")
        print("   python -m cyborg_mind_v2.training.train_cyborg_mind_ppo")
    else:
        print_error("Some checks failed. Please fix the issues above before training.")
        print(f"\n{Colors.BOLD}Common fixes:{Colors.RESET}")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Install correct gym version: pip install gym==0.21.0")
        print("- Install correct minerl version: pip install minerl==0.4.4")
        print("- Install Java if missing")
        print("- Check CUDA/PyTorch installation if GPU not detected")


def main():
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║     Cyborg Mind v2 Training Setup Verification            ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(Colors.RESET)
    
    results = {}
    
    # Run all checks
    results["Java"] = check_java()
    
    package_results = check_python_packages()
    results.update(package_results)
    
    results["CUDA"] = check_cuda()
    
    structure_results = check_project_structure()
    results["Project Structure"] = all(structure_results.values())
    
    results["Brain API"] = check_brain_api()
    results["Action Mapping"] = check_action_mapping()
    results["Obs Adapter"] = check_env_adapter()
    results["MineRL Environment"] = test_minerl_env()
    
    # Print summary
    print_summary(results)
    
    # Return exit code
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
