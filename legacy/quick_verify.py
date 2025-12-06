import sys
import torch
import importlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QuickVerify")

def check_import(module_name):
    try:
        importlib.import_module(module_name)
        logger.info(f"✅ {module_name} imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Failed to import {module_name}: {e}")
        return False

def main():
    logger.info("=== CyborgMind Environment Verification ===")
    
    # 1. System Info
    logger.info(f"Python: {sys.version}")
    logger.info(f"PyTorch: {torch.__version__}")
    
    # 2. CUDA Check
    if torch.cuda.is_available():
        logger.info(f"✅ CUDA Available: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("⚠️ CUDA NOT Available. Running on CPU.")

    # 3. Critical Imports
    modules = [
        "numpy",
        "gymnasium",
        "wandb",
        "fastapi",
        "prometheus_client",
        "mamba_ssm",  # Optional but good to check
    ]
    
    all_good = True
    for mod in modules:
        if mod == "mamba_ssm":
            try:
                import mamba_ssm
                logger.info(f"✅ {mod} imported successfully")
            except ImportError:
                logger.warning(f"⚠️ {mod} not found (OK if CPU-only)")
            except Exception as e:
                logger.warning(f"⚠️ {mod} import error: {e}")
        else:
            if not check_import(mod):
                all_good = False

    # 4. Basic Tensor Test
    try:
        x = torch.randn(10, 10)
        if torch.cuda.is_available():
            x = x.cuda()
        logger.info("✅ Tensor operations working")
    except Exception as e:
        logger.error(f"❌ Tensor test failed: {e}")
        all_good = False

    if all_good:
        logger.info("\n🎉 Environment looks good!")
        sys.exit(0)
    else:
        logger.error("\n❌ Environment has issues. Check logs.")
        sys.exit(1)

if __name__ == "__main__":
    main()
