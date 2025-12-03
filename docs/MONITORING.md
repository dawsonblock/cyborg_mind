# Monitoring & Observability

## Stack
- **Prometheus**: Scrapes metrics from the API server.
- **Grafana**: Visualizes metrics via dashboards.

## Metrics

### API Metrics
- `cyborg_api_requests_total`: Counter of requests by endpoint/status.
- `cyborg_api_latency_seconds`: Histogram of response times.

### Agent Metrics
- `cyborg_agent_memory_pressure`: Histogram of PMM memory usage.

### Training Metrics
- `cyborg_training_steps_total`: Total steps.
- `cyborg_training_loss_*`: Loss components.

## Dashboards

JSON definitions are located in `monitoring/grafana/dashboards/`.
- **Training Dashboard**: FPS, Losses, Entropy.
- **Inference Dashboard**: RPS, Latency, Memory Pressure.
