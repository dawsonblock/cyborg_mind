# API Reference

CyborgMind v3.0 provides a production-ready FastAPI server for real-time RL agent inference.

## Quick Start

### 1. Start the Server

```bash
# Start server with default settings
uvicorn cyborg_rl.server:create_app --reload

# Or with custom host/port
uvicorn cyborg_rl.server:create_app --host 0.0.0.0 --port 8080
```

### 2. Test Health Endpoint

```bash
curl http://localhost:8000/health
```

---

## Authentication

CyborgMind v3.0 supports **two authentication modes**: Static Bearer Tokens (default) and JWT Tokens (production).

### Mode 1: Static Bearer Token (Default)

Simple bearer token authentication for development and internal deployments.

**Configuration:**
```yaml
api:
  host: "0.0.0.0"
  port: 8000
  auth_token: "your-secret-token-here"
  jwt_enabled: false  # Static token mode
```

**Default token:** `cyborg-secret-v2`

**Usage:**
```bash
curl -X POST http://localhost:8000/step \
  -H "Authorization: Bearer cyborg-secret-v2" \
  -H "Content-Type: application/json" \
  -d '{"observation": [0.1, 0.2, 0.3, 0.4]}'
```

⚠️ **Security Note:** Static tokens don't expire and have no scope/claims. Suitable for:
- Development environments
- Internal deployments with network isolation
- Trusted client scenarios

---

### Mode 2: JWT Authentication (Production)

JSON Web Tokens with expiry, issuer, and audience validation for production deployments.

**Configuration:**
```yaml
api:
  host: "0.0.0.0"
  port: 8000
  auth_token: "fallback-static-token"  # Fallback for backward compatibility

  # JWT Settings
  jwt_enabled: true
  jwt_secret: "your-production-secret-key-min-32-chars"
  jwt_algorithm: "HS256"  # or RS256 for asymmetric
  jwt_issuer: "cyborg-api"
  jwt_audience: "cyborg-clients"
  jwt_expiry_minutes: 60
```

#### Generating JWT Tokens

**Request:**
```bash
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "user123",
    "expiry_minutes": 120
  }'
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_at": "2025-12-04T02:00:00",
  "subject": "user123"
}
```

#### Using JWT Tokens

```bash
curl -X POST http://localhost:8000/step \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  -H "Content-Type: application/json" \
  -d '{"observation": [0.1, 0.2, 0.3, 0.4]}'
```

#### JWT Features

✅ **Token Expiry:** Tokens automatically expire after configured duration  
✅ **Issuer Validation:** Ensures tokens come from trusted source  
✅ **Audience Validation:** Restricts tokens to specific clients  
✅ **Custom Claims:** Add role, permissions, or other metadata  
✅ **Algorithm Support:** HS256 (symmetric) or RS256 (asymmetric)  
✅ **Backward Compatible:** Falls back to static token if provided

#### Asymmetric JWT (RS256)

For distributed systems or when you can't share secrets:

**Configuration:**
```yaml
api:
  jwt_enabled: true
  jwt_secret: "/path/to/private_key.pem"  # Private key for signing
  jwt_algorithm: "RS256"
  jwt_public_key_path: "/path/to/public_key.pem"  # Public key for verification
  jwt_issuer: "cyborg-api"
  jwt_audience: "external-clients"
```

**Generate RSA keys:**
```bash
# Generate private key
openssl genrsa -out private_key.pem 2048

# Extract public key
openssl rsa -in private_key.pem -pubout -out public_key.pem
```

#### JWT Error Responses

**Expired Token:**
```json
{
  "detail": "Token has expired"
}
```
**Status:** `403 Forbidden`

**Invalid Issuer:**
```json
{
  "detail": "Invalid token issuer"
}
```
**Status:** `403 Forbidden`

**Invalid Audience:**
```json
{
  "detail": "Invalid token audience"
}
```
**Status:** `403 Forbidden`

---

### Dual-Mode Authentication

When JWT is enabled, the server accepts **both** JWT and static tokens:

```python
# JWT token (with expiry, validation)
headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."}

# Static token (fallback, no expiry)
headers = {"Authorization": "Bearer cyborg-secret-v2"}
```

This allows gradual migration from static to JWT authentication.

---


## Rate Limiting

The API uses [SlowAPI](https://github.com/laurents/slowapi) for rate limiting.

**Default limits:**
- Global: 100 requests/minute per IP
- Per-endpoint limits may vary

**Rate limit headers:**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 99
X-RateLimit-Reset: 1638360000
```

When rate limited, you'll receive:
```json
{
  "detail": "Rate limit exceeded: 100 per 1 minute"
}
```
**Status:** `429 Too Many Requests`

---

## Endpoints

### `GET /health`

Health check endpoint (no auth required).

**Response:**
```json
{
  "status": "healthy",
  "version": "3.0.0",
  "timestamp": "2025-12-03T23:59:59.123Z"
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

---

### `POST /reset`

Reset agent state (hidden state + memory) for a specific agent ID.

**Auth:** Required

**Request Body:**
```json
{
  "agent_id": "default"
}
```

**Parameters:**
- `agent_id` (optional): Agent identifier (default: `"default"`)

**Response:**
```json
{
  "message": "Agent state reset",
  "agent_id": "default",
  "state": {
    "hidden": "initialized",
    "memory": "initialized"
  }
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/reset \
  -H "Authorization: Bearer cyborg-secret-v2" \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "session_abc123"}'
```

---

### `POST /step`

Single-step inference. Get action for one observation.

**Auth:** Required

**Request Body:**
```json
{
  "observation": [0.1, -0.5, 0.0, 1.2],
  "agent_id": "default",
  "deterministic": true
}
```

**Parameters:**
- `observation` (required): List of floats matching agent's `obs_dim`
- `agent_id` (optional): Agent identifier (default: `"default"`)
- `deterministic` (optional): Use deterministic policy (default: `false`)

**Response (Discrete Action):**
```json
{
  "action": 1,
  "value": 0.4523,
  "log_prob": -0.6931,
  "agent_id": "default"
}
```

**Response (Continuous Action):**
```json
{
  "action": [0.234, -0.567],
  "value": 0.3421,
  "log_prob": -1.2345,
  "agent_id": "default"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/step \
  -H "Authorization: Bearer cyborg-secret-v2" \
  -H "Content-Type: application/json" \
  -d '{
    "observation": [0.1, 0.2, 0.3, 0.4],
    "agent_id": "cartpole_agent",
    "deterministic": true
  }'
```

**Error Responses:**

Wrong observation dimension:
```json
{
  "detail": "Expected observation of length 4, got 2"
}
```
**Status:** `400 Bad Request`

Agent not initialized:
```json
{
  "detail": "Agent 'unknown_agent' not initialized. Call /reset first."
}
```
**Status:** `400 Bad Request`

---

### `POST /step_batch`

Batch inference for multiple observations.

**Auth:** Required

**Request Body:**
```json
{
  "observations": [
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8]
  ],
  "agent_ids": ["agent_1", "agent_2"],
  "deterministic": true
}
```

**Parameters:**
- `observations` (required): List of observation arrays
- `agent_ids` (optional): List of agent IDs (must match length of observations)
- `deterministic` (optional): Use deterministic policy (default: `false`)

**Response:**
```json
{
  "actions": [1, 0],
  "values": [0.45, 0.32],
  "log_probs": [-0.69, -0.81],
  "agent_ids": ["agent_1", "agent_2"]
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/step_batch \
  -H "Authorization: Bearer cyborg-secret-v2" \
  -H "Content-Type: application/json" \
  -d '{
    "observations": [
      [0.1, 0.2, 0.3, 0.4],
      [0.5, 0.6, 0.7, 0.8],
      [0.9, 1.0, 1.1, 1.2]
    ],
    "deterministic": true
  }'
```

---

### `GET /metrics`

Prometheus metrics endpoint (no auth required).

**Response:** Plain text (Prometheus exposition format)

```
# HELP request_count_total Total requests by endpoint
# TYPE request_count_total counter
request_count_total{endpoint="/step",status="success"} 1234
request_count_total{endpoint="/step",status="error"} 5

# HELP request_latency_seconds Request latency
# TYPE request_latency_seconds histogram
request_latency_seconds_bucket{endpoint="/step",le="0.005"} 45
request_latency_seconds_bucket{endpoint="/step",le="0.01"} 120
...
```

**Example:**
```bash
curl http://localhost:8000/metrics
```

**Prometheus Scrape Config:**
```yaml
scrape_configs:
  - job_name: 'cyborg_api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
```


---

### `POST /auth/token`

Generate a JWT access token (only available when JWT authentication is enabled).

**Auth:** Not required

**Request Body:**
```json
{
  "subject": "user123",
  "expiry_minutes": 120
}
```

**Parameters:**
- `subject` (required): Token subject (user ID, agent ID, or identifier)
- `expiry_minutes` (optional): Custom token validity duration (overrides default)

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyMTIzIiwiaWF0IjoxNzAxNjQwMDAwLCJleHAiOjE3MDE2NDcyMDAsImlzcyI6ImN5Ym9yZy1hcGkiLCJhdWQiOiJjeWJvcmctY2xpZW50cyJ9.signature",
  "token_type": "bearer",
  "expires_at": "2025-12-04T02:00:00",
  "subject": "user123"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "api_client_001",
    "expiry_minutes": 60
  }'
```

**Error Response (JWT Disabled):**
```json
{
  "detail": "JWT authentication is not enabled. Set api.jwt_enabled=true in config."
}
```
**Status:** `501 Not Implemented`

**Use Case:**
- Generate tokens for API clients dynamically
- Set custom expiry for different use cases (short-lived for sensitive operations, long-lived for batch jobs)
- Track token subjects for audit logs

**Note:** This endpoint is intentionally unprotected to allow initial token generation. In production, consider:
- Protecting with API key or IP whitelist
- Implementing rate limiting
- Using a separate authentication service

---

### `WS /stream`

WebSocket streaming endpoint for continuous real-time inference.

**Auth:** Required (via token in message payload)

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/stream');
```

**Message Format:**

**Client → Server:**
```json
{
  "observation": [0.1, 0.2, 0.3, 0.4],
  "deterministic": true,
  "token": "cyborg-secret-v2"
}
```

**Server → Client:**
```json
{
  "action": 1,
  "value": 0.4523,
  "pressure": 0.12,
  "error": null
}
```

**Parameters:**
- `observation` (required): Observation vector
- `deterministic` (optional): Use deterministic policy (default: `true`)
- `token` (required): Bearer token for authentication

**Response Fields:**
- `action`: Selected action (int for discrete, array for continuous)
- `value`: State value estimate
- `pressure`: PMM memory pressure
- `error`: Error message (null if successful)

**Example (JavaScript):**
```javascript
const ws = new WebSocket('ws://localhost:8000/stream');

ws.onopen = () => {
  // Send observation
  ws.send(JSON.stringify({
    observation: [0.1, 0.2, 0.3, 0.4],
    deterministic: true,
    token: 'cyborg-secret-v2'
  }));
};

ws.onmessage = (event) => {
  const response = JSON.parse(event.data);
  if (response.error) {
    console.error('Error:', response.error);
  } else {
    console.log('Action:', response.action);
    console.log('Value:', response.value);
  }
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('WebSocket closed');
};
```

**Example (Python):**
```python
import websockets
import json
import asyncio

async def stream_inference():
    uri = "ws://localhost:8000/stream"
    async with websockets.connect(uri) as websocket:
        # Send observation
        message = {
            "observation": [0.1, 0.2, 0.3, 0.4],
            "deterministic": True,
            "token": "cyborg-secret-v2"
        }
        await websocket.send(json.dumps(message))

        # Receive response
        response = await websocket.recv()
        data = json.loads(response)

        if data["error"]:
            print(f"Error: {data['error']}")
        else:
            print(f"Action: {data['action']}")
            print(f"Value: {data['value']}")

asyncio.run(stream_inference())
```

**Use Cases:**
- Real-time game control
- High-frequency trading agents
- Continuous sensor data processing
- Interactive robotics control

**Benefits:**
- Lower latency than HTTP (no connection overhead)
- Maintains agent state across requests
- Bidirectional communication
- Efficient for high-frequency inference

**State Management:**
- Each WebSocket connection maintains its own agent state
- State persists for the duration of the connection
- State is automatically cleaned up on disconnect

**Error Handling:**
- Authentication errors return `{"error": "Authentication failed: ..."}`
- Missing fields return descriptive error messages
- Connection errors trigger WebSocket close

---

## Error Codes

| Status Code | Meaning |
|-------------|---------|
| `200 OK` | Request succeeded |
| `400 Bad Request` | Invalid input (wrong obs dim, missing agent) |
| `403 Forbidden` | Invalid or missing auth token |
| `422 Unprocessable Entity` | Malformed JSON or validation error |
| `429 Too Many Requests` | Rate limit exceeded |
| `500 Internal Server Error` | Server error (logged for debugging) |

---

## Python Client Example

```python
import requests

API_URL = "http://localhost:8000"
AUTH_TOKEN = "cyborg-secret-v2"
HEADERS = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json"
}

# Reset agent
response = requests.post(
    f"{API_URL}/reset",
    json={"agent_id": "my_session"},
    headers=HEADERS
)
print(response.json())

# Get action
observation = [0.1, 0.2, 0.3, 0.4]
response = requests.post(
    f"{API_URL}/step",
    json={
        "observation": observation,
        "agent_id": "my_session",
        "deterministic": True
    },
    headers=HEADERS
)
result = response.json()
action = result["action"]
value = result["value"]

print(f"Action: {action}, Value: {value}")
```

---

## Server Lifecycle

### Starting the Server

**Development (with auto-reload):**
```bash
uvicorn cyborg_rl.server:create_app --reload --log-level debug
```

**Production:**
```bash
uvicorn cyborg_rl.server:create_app --host 0.0.0.0 --port 8000 --workers 4
```

**With Gunicorn (production):**
```bash
gunicorn cyborg_rl.server:create_app \
  -k uvicorn.workers.UvicornWorker \
  -w 4 \
  -b 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile -
```

### Docker Deployment

```bash
# Build image
docker build -t cyborg-api:v3 .

# Run container
docker run -d \
  -p 8000:8000 \
  -e AUTH_TOKEN=your-production-token \
  --name cyborg-api \
  cyborg-api:v3
```

---

## Security Best Practices

1. **Change the default token:**
   ```yaml
   api:
     auth_token: "use-strong-random-token-here"
   ```

2. **Use HTTPS in production:**
   - Put behind Nginx/Traefik with SSL
   - Use Let's Encrypt for certificates

3. **Rate limit tuning:**
   - Adjust per your deployment needs
   - Monitor `/metrics` for abuse patterns

4. **Network isolation:**
   - Run in private VPC
   - Expose only through API gateway

5. **Logging:**
   - Enable structured logging
   - Monitor for auth failures
   - Alert on 5xx errors

---

## Monitoring

### Grafana Dashboard

Import the provided dashboard: `monitoring/grafana/cyborg_api.json`

**Metrics tracked:**
- Request rate (by endpoint)
- Latency percentiles (p50, p95, p99)
- Error rate
- Auth failure rate
- Inference throughput

### Logs

Structured JSON logs (configurable):
```json
{
  "timestamp": "2025-12-03T23:59:59.123Z",
  "level": "INFO",
  "endpoint": "/step",
  "agent_id": "session_abc",
  "latency_ms": 12.3,
  "status": "success"
}
```

---

## Troubleshooting

**Server won't start:**
```
Port 8000 is already in use
```
**Fix:** Change port: `uvicorn cyborg_rl.server:create_app --port 8080`

**403 Forbidden:**
```json
{"detail": "Invalid authentication token"}
```
**Fix:** Check Authorization header and token value

**Observation dimension mismatch:**
```json
{"detail": "Expected observation of length 4, got 3"}
```
**Fix:** Ensure observation matches agent's `obs_dim`

---

For training agents, see [docs/HOW_TO_TRAIN.md](HOW_TO_TRAIN.md).
For deployment guides, see [docs/DEPLOYMENT.md](DEPLOYMENT.md).
