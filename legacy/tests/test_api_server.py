"""Tests for FastAPI server endpoints."""

import pytest
from fastapi.testclient import TestClient
import torch

from cyborg_rl.config import Config
from cyborg_rl.agents.ppo_agent import PPOAgent
from cyborg_rl.server import CyborgServer


class TestAPIServer:
    """Test FastAPI server endpoints."""

    @pytest.fixture
    def config(self) -> Config:
        """Create test config."""
        config = Config()
        config.api.auth_token = "test-token-12345"
        return config

    @pytest.fixture
    def agent(self, config: Config) -> PPOAgent:
        """Create test agent."""
        return PPOAgent(
            obs_dim=4,
            action_dim=2,
            config=config,
            is_discrete=True,
            device=torch.device("cpu"),
        )

    @pytest.fixture
    def server(self, agent: PPOAgent, config: Config) -> CyborgServer:
        """Create server instance."""
        return CyborgServer(agent, config)

    @pytest.fixture
    def client(self, server: CyborgServer) -> TestClient:
        """Create test client."""
        return TestClient(server.app)

    def test_health_endpoint(self, client: TestClient):
        """Test /health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_reset_endpoint(self, client: TestClient, config: Config):
        """Test /reset endpoint."""
        headers = {"Authorization": f"Bearer {config.api.auth_token}"}
        response = client.post("/reset", headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert "state" in data
        assert "message" in data

    def test_reset_unauthorized(self, client: TestClient):
        """Test /reset without auth fails."""
        response = client.post("/reset")
        assert response.status_code == 403

    def test_reset_wrong_token(self, client: TestClient):
        """Test /reset with wrong token fails."""
        headers = {"Authorization": "Bearer wrong-token"}
        response = client.post("/reset", headers=headers)
        assert response.status_code == 403

    def test_step_endpoint(self, client: TestClient, config: Config):
        """Test /step endpoint."""
        headers = {"Authorization": f"Bearer {config.api.auth_token}"}

        # First reset
        client.post("/reset", headers=headers)

        # Then step
        step_data = {
            "observation": [0.1, 0.2, 0.3, 0.4],
        }
        response = client.post("/step", json=step_data, headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert "action" in data
        assert "value" in data
        assert "log_prob" in data

    def test_step_invalid_observation(self, client: TestClient, config: Config):
        """Test /step with invalid observation."""
        headers = {"Authorization": f"Bearer {config.api.auth_token}"}

        # Reset first
        client.post("/reset", headers=headers)

        # Wrong dimension
        step_data = {
            "observation": [0.1, 0.2],  # Should be 4 dims
        }
        response = client.post("/step", json=step_data, headers=headers)
        assert response.status_code in [400, 422]

    def test_metrics_endpoint(self, client: TestClient, config: Config):
        """Test /metrics endpoint (Prometheus format)."""
        response = client.get("/metrics")
        assert response.status_code == 200
        # Prometheus format is plain text
        assert "text/plain" in response.headers.get("content-type", "")

    def test_multiple_steps(self, client: TestClient, config: Config):
        """Test multiple sequential steps."""
        headers = {"Authorization": f"Bearer {config.api.auth_token}"}

        # Reset
        client.post("/reset", headers=headers)

        # Multiple steps
        for i in range(5):
            step_data = {"observation": [0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i]}
            response = client.post("/step", json=step_data, headers=headers)
            assert response.status_code == 200
            data = response.json()
            assert "action" in data
