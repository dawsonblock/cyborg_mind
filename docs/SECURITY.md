# Security Policy

## Authentication
- **Bearer Tokens**: All API access is protected via Bearer tokens.
- **Rotation**: Tokens should be rotated periodically via the K8s secret or config.

## Network Security
- **TLS/SSL**: Production deployments MUST use HTTPS. The provided Nginx config enforces HTTPS redirect.
- **Firewall**: Only expose port 443 (HTTPS) to the public. Port 8000 should be internal.

## Rate Limiting
- **DoS Protection**: The API server implements rate limiting (default 100 req/s per IP) to prevent abuse.

## Input Validation
- **Pydantic**: All inputs are strictly validated against schemas.
- **NaN Guards**: The model guards against NaN/Inf values to prevent crashes or poisoning.

## Container Security
- **Non-Root**: Docker containers should ideally run as non-root (TODO for v3.1).
- **Minimal Base**: We use `python:3.10-slim` to minimize attack surface.
