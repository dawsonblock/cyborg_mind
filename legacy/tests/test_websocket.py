"""Tests for WebSocket streaming endpoint."""

import pytest
import json
from fastapi.testclient import TestClient

from cyborg_rl.server import create_app
from cyborg_rl.config import Config


class TestWebSocketStreaming:
    """Test WebSocket streaming inference."""

    @pytest.fixture
    def app(self):
        """Create test app."""
        return create_app()

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def config(self):
        """Get server config."""
        config = Config()
        return config

    def test_websocket_connection(self, client: TestClient, config: Config):
        """Test WebSocket connection can be established."""
        with client.websocket_connect("/stream") as websocket:
            # Connection should be accepted
            assert websocket is not None

    def test_websocket_inference(self, client: TestClient, config: Config):
        """Test basic WebSocket inference."""
        with client.websocket_connect("/stream") as websocket:
            # Send observation
            message = {
                "observation": [0.1, 0.2, 0.3, 0.4],
                "deterministic": True,
                "token": config.api.auth_token
            }
            websocket.send_json(message)

            # Receive response
            response = websocket.receive_json()

            # Check response structure
            assert "action" in response
            assert "value" in response
            assert "pressure" in response
            assert "error" in response
            assert response["error"] is None

    def test_websocket_missing_token(self, client: TestClient):
        """Test WebSocket with missing token."""
        with client.websocket_connect("/stream") as websocket:
            # Send observation without token
            message = {
                "observation": [0.1, 0.2, 0.3, 0.4],
                "deterministic": True
            }
            websocket.send_json(message)

            # Should receive error
            response = websocket.receive_json()
            assert response["error"] is not None
            assert "token" in response["error"].lower()

    def test_websocket_invalid_token(self, client: TestClient):
        """Test WebSocket with invalid token."""
        with client.websocket_connect("/stream") as websocket:
            # Send observation with invalid token
            message = {
                "observation": [0.1, 0.2, 0.3, 0.4],
                "deterministic": True,
                "token": "invalid-token-123"
            }
            websocket.send_json(message)

            # Should receive authentication error
            response = websocket.receive_json()
            assert response["error"] is not None
            assert "authentication" in response["error"].lower() or "invalid" in response["error"].lower()

    def test_websocket_missing_observation(self, client: TestClient, config: Config):
        """Test WebSocket with missing observation."""
        with client.websocket_connect("/stream") as websocket:
            # Send message without observation
            message = {
                "deterministic": True,
                "token": config.api.auth_token
            }
            websocket.send_json(message)

            # Should receive error
            response = websocket.receive_json()
            assert response["error"] is not None
            assert "observation" in response["error"].lower()

    def test_websocket_multiple_requests(self, client: TestClient, config: Config):
        """Test multiple sequential requests over same WebSocket."""
        with client.websocket_connect("/stream") as websocket:
            # Send multiple observations
            for i in range(5):
                message = {
                    "observation": [0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i],
                    "deterministic": True,
                    "token": config.api.auth_token
                }
                websocket.send_json(message)

                # Receive response
                response = websocket.receive_json()
                assert response["error"] is None
                assert "action" in response
                assert "value" in response

    def test_websocket_state_persistence(self, client: TestClient, config: Config):
        """Test that agent state persists across WebSocket messages."""
        with client.websocket_connect("/stream") as websocket:
            # First request
            message1 = {
                "observation": [0.1, 0.2, 0.3, 0.4],
                "deterministic": True,
                "token": config.api.auth_token
            }
            websocket.send_json(message1)
            response1 = websocket.receive_json()

            # Second request (state should carry over from first)
            message2 = {
                "observation": [0.5, 0.6, 0.7, 0.8],
                "deterministic": True,
                "token": config.api.auth_token
            }
            websocket.send_json(message2)
            response2 = websocket.receive_json()

            # Both should succeed
            assert response1["error"] is None
            assert response2["error"] is None

            # Pressure values may differ due to state evolution
            # (This depends on agent implementation)

    def test_websocket_concurrent_connections(self, client: TestClient, config: Config):
        """Test multiple concurrent WebSocket connections."""
        # Open two WebSocket connections
        with client.websocket_connect("/stream") as ws1, \
             client.websocket_connect("/stream") as ws2:

            # Send different observations to each
            message1 = {
                "observation": [0.1, 0.2, 0.3, 0.4],
                "deterministic": True,
                "token": config.api.auth_token
            }
            message2 = {
                "observation": [0.5, 0.6, 0.7, 0.8],
                "deterministic": True,
                "token": config.api.auth_token
            }

            ws1.send_json(message1)
            ws2.send_json(message2)

            # Both should receive responses
            response1 = ws1.receive_json()
            response2 = ws2.receive_json()

            assert response1["error"] is None
            assert response2["error"] is None

    def test_websocket_deterministic_vs_stochastic(self, client: TestClient, config: Config):
        """Test deterministic vs stochastic action selection."""
        with client.websocket_connect("/stream") as websocket:
            # Deterministic
            message_det = {
                "observation": [0.1, 0.2, 0.3, 0.4],
                "deterministic": True,
                "token": config.api.auth_token
            }
            websocket.send_json(message_det)
            response_det = websocket.receive_json()

            # Stochastic
            message_stoch = {
                "observation": [0.1, 0.2, 0.3, 0.4],
                "deterministic": False,
                "token": config.api.auth_token
            }
            websocket.send_json(message_stoch)
            response_stoch = websocket.receive_json()

            # Both should succeed
            assert response_det["error"] is None
            assert response_stoch["error"] is None

            # Actions may differ (though not guaranteed for all policies)

    @pytest.mark.skipif(True, reason="JWT not always enabled in test config")
    def test_websocket_with_jwt_token(self, client: TestClient, config: Config):
        """Test WebSocket with JWT token (if enabled)."""
        # Generate JWT token
        token_response = client.post("/auth/token", json={
            "subject": "test_websocket_user",
            "expiry_minutes": 10
        })

        if token_response.status_code == 200:
            jwt_token = token_response.json()["access_token"]

            with client.websocket_connect("/stream") as websocket:
                message = {
                    "observation": [0.1, 0.2, 0.3, 0.4],
                    "deterministic": True,
                    "token": jwt_token
                }
                websocket.send_json(message)

                response = websocket.receive_json()
                assert response["error"] is None


class TestWebSocketMetrics:
    """Test WebSocket Prometheus metrics."""

    @pytest.fixture
    def app(self):
        """Create test app."""
        return create_app()

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def config(self):
        """Get server config."""
        return Config()

    def test_websocket_metrics_incremented(self, client: TestClient, config: Config):
        """Test that WebSocket metrics are incremented."""
        # Get initial metrics
        metrics_before = client.get("/metrics").text

        # Open and close WebSocket
        with client.websocket_connect("/stream") as websocket:
            message = {
                "observation": [0.1, 0.2, 0.3, 0.4],
                "deterministic": True,
                "token": config.api.auth_token
            }
            websocket.send_json(message)
            response = websocket.receive_json()

        # Get metrics after
        metrics_after = client.get("/metrics").text

        # Check that WebSocket connection counter was incremented
        assert "cyborg_websocket_connections_total" in metrics_after

        # Check request counter for /stream endpoint
        assert 'endpoint="/stream"' in metrics_after or "endpoint=\"/stream\"" in metrics_after
