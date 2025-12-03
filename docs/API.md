# API Reference

## Authentication

All endpoints require a Bearer Token.
Header: `Authorization: Bearer <YOUR_TOKEN>`

## Endpoints

### `POST /step`
Single-step inference.

**Request:**
```json
{
  "observation": [0.1, -0.5, 0.0, 1.2],
  "agent_id": "session_123",
  "deterministic": true
}
```

**Response:**
```json
{
  "action": 1,
  "value": 0.45,
  "thought": "Memory pressure: 0.12",
  "pressure": 0.12
}
```

### `POST /step_batch`
Batch inference for multiple agents.

**Request:**
```json
{
  "observations": [[...], [...]],
  "agent_ids": ["agent_1", "agent_2"]
}
```

### `POST /reset`
Reset agent state (hidden/memory).

**Request:**
```json
{ "agent_id": "session_123" }
```

### `GET /health`
Server health status.

### `GET /metrics`
Prometheus metrics.
