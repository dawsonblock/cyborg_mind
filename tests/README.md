# CyborgMind v3.0 Test Suite

This directory contains the production test suite for CyborgMind v3.0.

## Running Tests

```bash
# Install dependencies
pip install -e .

# Run all tests
pytest

# Run specific test file
pytest tests/test_v3_smoke.py

# Run with coverage
pytest --cov=cyborg_rl --cov-report=html

# Run only fast tests (exclude slow integration tests)
pytest -m "not slow"
```

## Test Organization

### Core v3 Tests (Active)

**Integration/Smoke Tests:**
- `test_v3_smoke.py` - Comprehensive end-to-end test validating the full v3 pipeline
  - Config loading (dict/YAML)
  - PPOAgent initialization and forward pass
  - RolloutBuffer operations
  - PPOTrainer setup
  - Save/load functionality
  - Gradient flow validation

**Component Tests:**
- `test_memory_pmm.py` - Predictive Memory Module (PMM)
  - Read/write operations
  - Memory statistics
  - Attention sharpening
  - Gradient flow

- `test_model_forward.py` - Neural network components
  - MambaGRUEncoder
  - DiscretePolicy / ContinuousPolicy
  - ValueHead
  - PPOAgent forward pass

- `test_trainer_step.py` - Training components
  - RolloutBuffer
  - PPOTrainer

- `test_env_adapters.py` - Environment adapters
  - GymAdapter
  - Reset, step, reward handling

**Infrastructure Tests:**
- `test_api_server.py` - FastAPI server endpoints
  - /health, /reset, /step, /metrics
  - Authentication validation
  - Request/response validation

- `test_experiment_registry.py` - Experiment tracking
  - Metrics logging
  - Checkpoint saving
  - Config persistence

- `test_cyborg_rl_imports.py` - Basic import validation

### Legacy v2 Tests (Archived)

The `legacy_v2/` directory contains tests from the old CyborgMind v2.x architecture. These are preserved for reference but are **not executed** by CI.

Files in `legacy_v2/`:
- `test_brain.py` - BrainCyborgMind architecture (deprecated)
- `test_adapters.py` - v2 adapter interface (replaced by cyborg_rl.envs)
- `test_stress.py`, `test_checkpoint_persistence.py`, etc.
- `core/test_pmm_integration.py` - v2 PMM integration
- `training/test_train_loop.py` - v2 trainer

**Do not modify** legacy tests. If you need to test similar functionality, create new v3 tests.

## Test Coverage Goals

Target coverage: **>80%** for core modules:
- `cyborg_rl/agents/`
- `cyborg_rl/trainers/`
- `cyborg_rl/memory/`
- `cyborg_rl/models/`
- `cyborg_rl/config.py`
- `cyborg_rl/server.py`

## Adding New Tests

1. **Unit tests** → Add to existing test files or create `test_<module>.py`
2. **Integration tests** → Add to `test_v3_smoke.py` or create new integration file
3. **Slow tests** → Mark with `@pytest.mark.slow`
4. **Requires GPU** → Mark with `@pytest.mark.gpu` and add conditional skip

Example:
```python
import pytest

@pytest.mark.slow
def test_full_training_run():
    # Long-running test
    pass

@pytest.mark.gpu
def test_cuda_performance():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # GPU-specific test
    pass
```

## CI Integration

The test suite runs automatically on:
- Pull requests
- Pushes to main
- Manual workflow dispatch

See `.github/workflows/main.yml` for CI configuration.

## Troubleshooting

**Import errors:**
- Ensure you've installed the package: `pip install -e .`
- Check Python path: `python -c "import cyborg_rl; print(cyborg_rl.__file__)"`

**Missing dependencies:**
- Install dev dependencies: `pip install -e .[dev]`
- For MineRL tests: `pip install minerl`

**Tests failing after code changes:**
1. Run specific failing test: `pytest tests/test_foo.py::test_bar -v`
2. Check if mocks/fixtures need updating
3. Verify config changes are reflected in test fixtures

---

For questions about the test suite, see `docs/CONTRIBUTING.md` or open an issue.
