"""JWT Authentication utilities for CyborgMind API.

Provides JWT token generation, validation, and verification with:
- HS256 (symmetric) and RS256 (asymmetric) algorithms
- Token expiry, issuer, and audience validation
- Backward compatibility with static bearer tokens
"""

import jwt
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class JWTAuth:
    """JWT authentication handler with dual-mode support (JWT + static token)."""

    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        issuer: Optional[str] = None,
        audience: Optional[str] = None,
        token_expiry_minutes: int = 60,
        static_token: Optional[str] = None,
        public_key_path: Optional[str] = None,
    ):
        """
        Initialize JWT authentication.

        Args:
            secret_key: Secret key for HS256 or path to private key for RS256
            algorithm: JWT algorithm (HS256 or RS256)
            issuer: Token issuer (e.g., "cyborg-api")
            audience: Token audience (e.g., "cyborg-clients")
            token_expiry_minutes: Token validity duration in minutes
            static_token: Optional static bearer token for backward compatibility
            public_key_path: Path to public key file for RS256 verification
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.issuer = issuer
        self.audience = audience
        self.token_expiry_minutes = token_expiry_minutes
        self.static_token = static_token

        # Load public key for RS256
        self.public_key = None
        if algorithm == "RS256" and public_key_path:
            try:
                with open(public_key_path, "r") as f:
                    self.public_key = f.read()
                logger.info(f"Loaded RSA public key from {public_key_path}")
            except Exception as e:
                logger.warning(f"Failed to load public key: {e}")

    def generate_token(
        self,
        subject: str,
        additional_claims: Optional[Dict[str, Any]] = None,
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        """
        Generate a JWT token.

        Args:
            subject: Token subject (typically user ID or agent ID)
            additional_claims: Extra claims to include in the token
            expires_delta: Custom expiry time (overrides default)

        Returns:
            JWT token string
        """
        now = datetime.utcnow()
        expire = now + (expires_delta or timedelta(minutes=self.token_expiry_minutes))

        payload = {
            "sub": subject,
            "iat": now,
            "exp": expire,
        }

        if self.issuer:
            payload["iss"] = self.issuer

        if self.audience:
            payload["aud"] = self.audience

        if additional_claims:
            payload.update(additional_claims)

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token

    def verify_token(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Verify a JWT token or static token.

        Args:
            token: JWT token string or static bearer token

        Returns:
            Tuple of (is_valid, decoded_payload, error_message)
        """
        # First check if it's a static token (backward compatibility)
        if self.static_token and token == self.static_token:
            logger.debug("Static token validated")
            return True, {"sub": "static_user", "type": "static"}, None

        # Try to decode as JWT
        try:
            # Use public key for RS256, secret key for HS256
            key = self.public_key if self.algorithm == "RS256" and self.public_key else self.secret_key

            # Prepare validation options
            options = {}
            if not self.issuer:
                options["verify_iss"] = False
            if not self.audience:
                options["verify_aud"] = False

            payload = jwt.decode(
                token,
                key,
                algorithms=[self.algorithm],
                issuer=self.issuer,
                audience=self.audience,
                options=options,
            )

            logger.debug(f"JWT token validated for subject: {payload.get('sub')}")
            return True, payload, None

        except jwt.ExpiredSignatureError:
            return False, None, "Token has expired"
        except jwt.InvalidIssuerError:
            return False, None, "Invalid token issuer"
        except jwt.InvalidAudienceError:
            return False, None, "Invalid token audience"
        except jwt.InvalidTokenError as e:
            return False, None, f"Invalid token: {str(e)}"
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return False, None, f"Token verification failed: {str(e)}"

    def decode_token_unsafe(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Decode token without verification (for debugging only).

        Args:
            token: JWT token string

        Returns:
            Decoded payload or None
        """
        try:
            return jwt.decode(token, options={"verify_signature": False})
        except Exception:
            return None

    def get_token_expiry(self, token: str) -> Optional[datetime]:
        """
        Get the expiry time of a token.

        Args:
            token: JWT token string

        Returns:
            Expiry datetime or None
        """
        payload = self.decode_token_unsafe(token)
        if payload and "exp" in payload:
            return datetime.fromtimestamp(payload["exp"])
        return None

    def is_token_expired(self, token: str) -> bool:
        """
        Check if token is expired.

        Args:
            token: JWT token string

        Returns:
            True if expired, False otherwise
        """
        expiry = self.get_token_expiry(token)
        if expiry:
            return datetime.utcnow() > expiry
        return True  # Assume expired if can't decode


def create_jwt_handler(
    jwt_enabled: bool = False,
    jwt_secret: Optional[str] = None,
    jwt_algorithm: str = "HS256",
    jwt_issuer: Optional[str] = None,
    jwt_audience: Optional[str] = None,
    jwt_expiry_minutes: int = 60,
    static_token: Optional[str] = None,
    jwt_public_key_path: Optional[str] = None,
) -> JWTAuth:
    """
    Factory function to create JWT auth handler.

    Args:
        jwt_enabled: Whether JWT authentication is enabled
        jwt_secret: Secret key for JWT (required if jwt_enabled)
        jwt_algorithm: JWT algorithm (HS256 or RS256)
        jwt_issuer: Token issuer
        jwt_audience: Token audience
        jwt_expiry_minutes: Token validity in minutes
        static_token: Static bearer token for backward compatibility
        jwt_public_key_path: Path to RSA public key (for RS256)

    Returns:
        JWTAuth instance

    Raises:
        ValueError: If JWT is enabled but secret_key is not provided
    """
    if jwt_enabled and not jwt_secret:
        raise ValueError("JWT secret key is required when JWT authentication is enabled")

    # Use static token as secret if JWT not enabled
    secret = jwt_secret if jwt_enabled else (static_token or "default-secret")

    return JWTAuth(
        secret_key=secret,
        algorithm=jwt_algorithm,
        issuer=jwt_issuer,
        audience=jwt_audience,
        token_expiry_minutes=jwt_expiry_minutes,
        static_token=static_token,
        public_key_path=jwt_public_key_path,
    )
