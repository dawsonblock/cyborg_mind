"""Tests for JWT authentication system."""

import pytest
import time
from datetime import timedelta

from cyborg_rl.utils.jwt_auth import JWTAuth, create_jwt_handler


class TestJWTAuth:
    """Test JWT authentication handler."""

    @pytest.fixture
    def jwt_handler(self) -> JWTAuth:
        """Create JWT handler with HS256."""
        return JWTAuth(
            secret_key="test-secret-key",
            algorithm="HS256",
            issuer="cyborg-test",
            audience="test-clients",
            token_expiry_minutes=60,
            static_token="static-test-token",
        )

    def test_generate_token(self, jwt_handler: JWTAuth):
        """Test token generation."""
        token = jwt_handler.generate_token(subject="test_user")

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    def test_verify_valid_token(self, jwt_handler: JWTAuth):
        """Test verification of valid JWT token."""
        token = jwt_handler.generate_token(subject="test_user")

        is_valid, payload, error = jwt_handler.verify_token(token)

        assert is_valid is True
        assert payload is not None
        assert payload["sub"] == "test_user"
        assert payload["iss"] == "cyborg-test"
        assert payload["aud"] == "test-clients"
        assert error is None

    def test_verify_static_token(self, jwt_handler: JWTAuth):
        """Test verification of static bearer token."""
        is_valid, payload, error = jwt_handler.verify_token("static-test-token")

        assert is_valid is True
        assert payload is not None
        assert payload["sub"] == "static_user"
        assert payload["type"] == "static"
        assert error is None

    def test_verify_invalid_token(self, jwt_handler: JWTAuth):
        """Test verification of invalid token."""
        is_valid, payload, error = jwt_handler.verify_token("invalid-token")

        assert is_valid is False
        assert payload is None
        assert error is not None
        assert "Invalid token" in error

    def test_verify_expired_token(self):
        """Test verification of expired token."""
        # Create handler with 0 minute expiry
        handler = JWTAuth(
            secret_key="test-secret",
            algorithm="HS256",
            token_expiry_minutes=0,
        )

        token = handler.generate_token(subject="test_user")

        # Wait a bit to ensure expiry
        time.sleep(0.1)

        is_valid, payload, error = handler.verify_token(token)

        assert is_valid is False
        assert payload is None
        assert "expired" in error.lower()

    def test_verify_wrong_issuer(self):
        """Test token with wrong issuer."""
        # Generate token with one issuer
        handler1 = JWTAuth(
            secret_key="test-secret",
            algorithm="HS256",
            issuer="issuer-1",
        )
        token = handler1.generate_token(subject="test_user")

        # Verify with different issuer
        handler2 = JWTAuth(
            secret_key="test-secret",
            algorithm="HS256",
            issuer="issuer-2",
        )

        is_valid, payload, error = handler2.verify_token(token)

        assert is_valid is False
        assert "issuer" in error.lower()

    def test_verify_wrong_audience(self):
        """Test token with wrong audience."""
        handler1 = JWTAuth(
            secret_key="test-secret",
            algorithm="HS256",
            audience="audience-1",
        )
        token = handler1.generate_token(subject="test_user")

        handler2 = JWTAuth(
            secret_key="test-secret",
            algorithm="HS256",
            audience="audience-2",
        )

        is_valid, payload, error = handler2.verify_token(token)

        assert is_valid is False
        assert "audience" in error.lower()

    def test_token_with_additional_claims(self, jwt_handler: JWTAuth):
        """Test token with custom claims."""
        token = jwt_handler.generate_token(
            subject="test_user",
            additional_claims={"role": "admin", "permissions": ["read", "write"]},
        )

        is_valid, payload, error = jwt_handler.verify_token(token)

        assert is_valid is True
        assert payload["role"] == "admin"
        assert payload["permissions"] == ["read", "write"]

    def test_custom_expiry(self, jwt_handler: JWTAuth):
        """Test token with custom expiry."""
        # 2 hour expiry
        token = jwt_handler.generate_token(
            subject="test_user",
            expires_delta=timedelta(hours=2),
        )

        expiry = jwt_handler.get_token_expiry(token)

        assert expiry is not None
        # Should be approximately 2 hours from now
        from datetime import datetime
        time_until_expiry = (expiry - datetime.utcnow()).total_seconds()
        assert 7100 < time_until_expiry < 7300  # ~2 hours (with some tolerance)

    def test_decode_unsafe(self, jwt_handler: JWTAuth):
        """Test unsafe decoding (no verification)."""
        token = jwt_handler.generate_token(subject="test_user")

        payload = jwt_handler.decode_token_unsafe(token)

        assert payload is not None
        assert payload["sub"] == "test_user"

    def test_is_token_expired(self):
        """Test expiry checking."""
        handler = JWTAuth(
            secret_key="test-secret",
            token_expiry_minutes=0,
        )

        token = handler.generate_token(subject="test_user")
        time.sleep(0.1)

        assert handler.is_token_expired(token) is True

    def test_create_jwt_handler_factory(self):
        """Test factory function for creating JWT handler."""
        handler = create_jwt_handler(
            jwt_enabled=True,
            jwt_secret="my-secret",
            jwt_algorithm="HS256",
            jwt_issuer="test-issuer",
            jwt_audience="test-audience",
            jwt_expiry_minutes=30,
            static_token="fallback-token",
        )

        assert handler is not None
        assert handler.secret_key == "my-secret"
        assert handler.algorithm == "HS256"
        assert handler.issuer == "test-issuer"
        assert handler.static_token == "fallback-token"

    def test_create_jwt_handler_disabled(self):
        """Test factory with JWT disabled."""
        handler = create_jwt_handler(
            jwt_enabled=False,
            static_token="my-static-token",
        )

        assert handler is not None
        # Should still work with static token
        is_valid, _, _ = handler.verify_token("my-static-token")
        assert is_valid is True

    def test_create_jwt_handler_missing_secret(self):
        """Test factory raises error when JWT enabled but no secret."""
        with pytest.raises(ValueError, match="JWT secret key is required"):
            create_jwt_handler(jwt_enabled=True, jwt_secret=None)

    def test_no_issuer_or_audience(self):
        """Test JWT without issuer/audience validation."""
        handler = JWTAuth(
            secret_key="test-secret",
            algorithm="HS256",
            issuer=None,
            audience=None,
        )

        token = handler.generate_token(subject="test_user")

        is_valid, payload, error = handler.verify_token(token)

        assert is_valid is True
        assert "iss" not in payload
        assert "aud" not in payload


class TestJWTIntegrationWithAPI:
    """Integration tests with API config."""

    def test_jwt_with_config(self):
        """Test JWT handler created from API config."""
        from cyborg_rl.config import APIConfig

        config = APIConfig(
            jwt_enabled=True,
            jwt_secret="production-secret-key",
            jwt_algorithm="HS256",
            jwt_issuer="cyborg-api",
            jwt_audience="cyborg-clients",
            jwt_expiry_minutes=120,
            auth_token="fallback-static-token",
        )

        handler = create_jwt_handler(
            jwt_enabled=config.jwt_enabled,
            jwt_secret=config.jwt_secret,
            jwt_algorithm=config.jwt_algorithm,
            jwt_issuer=config.jwt_issuer,
            jwt_audience=config.jwt_audience,
            jwt_expiry_minutes=config.jwt_expiry_minutes,
            static_token=config.auth_token,
        )

        # Test JWT generation and verification
        token = handler.generate_token(subject="user123")
        is_valid, payload, _ = handler.verify_token(token)

        assert is_valid is True
        assert payload["sub"] == "user123"
        assert payload["iss"] == "cyborg-api"
        assert payload["aud"] == "cyborg-clients"

        # Test static token fallback
        is_valid_static, _, _ = handler.verify_token("fallback-static-token")
        assert is_valid_static is True
