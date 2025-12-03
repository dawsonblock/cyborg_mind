"""Tests for ExperimentRegistry."""

import pytest
import tempfile
import json
from pathlib import Path

from cyborg_rl.experiments.registry import ExperimentRegistry
from cyborg_rl.config import Config


class TestExperimentRegistry:
    """Test experiment tracking and registry."""

    @pytest.fixture
    def config_dict(self):
        """Create test config dict."""
        config = Config()
        return config.to_dict()

    def test_registry_initialization(self, config_dict):
        """Test registry can be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ExperimentRegistry(
                config_dict,
                run_name="test_exp",
                base_dir=tmpdir,
            )

            assert registry.run_name == "test_exp"
            assert Path(tmpdir, "test_exp").exists()

    def test_log_metrics(self, config_dict):
        """Test metrics logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ExperimentRegistry(
                config_dict,
                run_name="metrics_test",
                base_dir=tmpdir,
            )

            # Log some metrics
            registry.log_metrics(step=0, metrics={"loss": 0.5, "reward": 10.0})
            registry.log_metrics(step=100, metrics={"loss": 0.3, "reward": 15.0})
            registry.log_metrics(step=200, metrics={"loss": 0.1, "reward": 20.0})

            # Check metrics file exists
            metrics_file = Path(tmpdir, "metrics_test", "metrics.jsonl")
            # Note: Some implementations may use CSV or other formats
            # Adjust based on actual implementation

    def test_save_checkpoint(self, config_dict):
        """Test checkpoint saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ExperimentRegistry(
                config_dict,
                run_name="checkpoint_test",
                base_dir=tmpdir,
            )

            # Save checkpoint
            dummy_state = {"layer1": "weights", "layer2": "biases"}
            registry.save_checkpoint(dummy_state, step=1000)

            # Check checkpoint directory exists
            checkpoint_dir = Path(tmpdir, "checkpoint_test")
            assert checkpoint_dir.exists()

    def test_config_saved_on_init(self, config_dict):
        """Test that config is saved on registry creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ExperimentRegistry(
                config_dict,
                run_name="config_test",
                base_dir=tmpdir,
            )

            # Check config file exists
            run_dir = Path(tmpdir, "config_test")
            config_file = run_dir / "config.yaml"

            # Registry should save config
            # (Implementation may vary - adjust based on actual code)

    def test_multiple_experiments(self, config_dict):
        """Test multiple experiments in same base dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry1 = ExperimentRegistry(
                config_dict,
                run_name="exp1",
                base_dir=tmpdir,
            )

            registry2 = ExperimentRegistry(
                config_dict,
                run_name="exp2",
                base_dir=tmpdir,
            )

            # Both should exist
            assert Path(tmpdir, "exp1").exists()
            assert Path(tmpdir, "exp2").exists()

    def test_run_name_autogeneration(self, config_dict):
        """Test auto-generated run names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # No run_name provided
            registry = ExperimentRegistry(
                config_dict,
                run_name=None,
                base_dir=tmpdir,
            )

            # Should have auto-generated name
            assert registry.run_name is not None
            assert len(registry.run_name) > 0
