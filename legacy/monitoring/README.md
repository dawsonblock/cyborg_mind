# CyborgMind Monitoring Stack

Production-ready monitoring setup with Prometheus and Grafana for CyborgMind API.

## Quick Start

### 1. Start the Stack

```bash
docker-compose up -d
```

This launches:
- **CyborgAPI** on port `8000`
- **Prometheus** on port `9090`
- **Grafana** on port `3000`

### 2. Access Dashboards

**Grafana Dashboard:**
- URL: http://localhost:3000
- Default credentials: `admin` / `admin`
- Pre-configured dashboard: "CyborgMind API Dashboard"

**Prometheus UI:**
- URL: http://localhost:9090
- Query metrics directly

**API Health:**
- URL: http://localhost:8000/health

---

## Configuration

### Environment Variables

Create `.env` file (see `.env.example`):

```bash
# API
AUTH_TOKEN=your-production-token
JWT_ENABLED=true
JWT_SECRET=your-jwt-secret
DEVICE=cuda  # or cpu

# Grafana
GRAFANA_USER=admin
GRAFANA_PASSWORD=secure-password
```

### Custom Prometheus Config

Edit `monitoring/prometheus/prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'cyborg-api'
    static_configs:
      - targets: ['cyborg-api:8000']
    scrape_interval: 10s  # Adjust as needed
```

---

## Metrics Tracked

### Request Metrics
- `request_count_total{endpoint, status}`: Total requests by endpoint and status
- `request_latency_seconds{endpoint}`: Request latency histogram

### API Performance
- Request rate (req/s)
- Latency percentiles (p50, p95, p99)
- Error rate
- Auth failure count
- Inference throughput

---

## Grafana Dashboard

Pre-configured panels:

1. **Request Rate by Endpoint**: Success vs error rates
2. **Inference Latency**: p50, p95, p99 percentiles
3. **Error Rate**: Real-time error percentage
4. **Auth Failures**: Authentication failure count (5m window)
5. **Inference Throughput**: Single vs batch inference rates
6. **Requests by Status**: Status code distribution
7. **Total Requests**: All-time request counts

---

## Operations

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f cyborg-api
docker-compose logs -f prometheus
docker-compose logs -f grafana
```

### Stop Stack

```bash
docker-compose down
```

### Stop and Remove Data

```bash
docker-compose down -v
```

### Restart Service

```bash
docker-compose restart cyborg-api
```

### Scale API (Multiple Replicas)

```bash
docker-compose up -d --scale cyborg-api=3
```

Then update Prometheus config with:
```yaml
scrape_configs:
  - job_name: 'cyborg-api'
    dns_sd_configs:
      - names: ['cyborg-api']
        type: 'A'
        port: 8000
```

---

## Troubleshooting

### Grafana Can't Connect to Prometheus

Check datasource configuration:
```bash
docker-compose exec grafana cat /etc/grafana/provisioning/datasources/prometheus.yml
```

Ensure Prometheus URL is `http://prometheus:9090`

### No Metrics in Prometheus

1. Check API is exposing metrics:
   ```bash
   curl http://localhost:8000/metrics
   ```

2. Check Prometheus targets:
   - Go to http://localhost:9090/targets
   - Ensure `cyborg-api:8000` is UP

3. Check Prometheus logs:
   ```bash
   docker-compose logs prometheus
   ```

### Dashboard Not Showing Data

1. Verify time range (top-right in Grafana)
2. Check Prometheus is receiving data:
   ```
   http://localhost:9090/graph
   Query: request_count_total
   ```
3. Refresh dashboard or restart Grafana:
   ```bash
   docker-compose restart grafana
   ```

---

## Production Deployment

### Persistent Storage

Volumes are automatically created:
- `prometheus-data`: Prometheus time-series database
- `grafana-data`: Grafana dashboards and config

### Security

1. **Change default passwords:**
   ```bash
   export GRAFANA_PASSWORD=<strong-password>
   docker-compose up -d
   ```

2. **Use HTTPS:**
   Add Nginx reverse proxy:
   ```yaml
   services:
     nginx:
       image: nginx:alpine
       ports:
         - "443:443"
       volumes:
         - ./nginx.conf:/etc/nginx/nginx.conf
         - ./ssl:/etc/nginx/ssl
   ```

3. **Network isolation:**
   Remove public ports for Prometheus/Grafana:
   ```yaml
   prometheus:
     # Remove 'ports:' section
     expose:
       - "9090"
   ```

### Backup

```bash
# Backup Prometheus data
docker run --rm -v cyborg_mind_prometheus-data:/data \
  -v $(pwd)/backup:/backup alpine \
  tar czf /backup/prometheus-backup.tar.gz /data

# Backup Grafana data
docker run --rm -v cyborg_mind_grafana-data:/data \
  -v $(pwd)/backup:/backup alpine \
  tar czf /backup/grafana-backup.tar.gz /data
```

---

## Custom Dashboards

### Import Additional Dashboards

1. Go to Grafana → Dashboards → Import
2. Enter dashboard ID or upload JSON
3. Select Prometheus datasource

### Export Current Dashboard

1. Dashboard → Settings → JSON Model
2. Copy JSON
3. Save to `monitoring/grafana/dashboards/`

---

## Alerting (Advanced)

Add Prometheus alerting rules:

**prometheus/alerts.yml:**
```yaml
groups:
  - name: cyborg_api
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: rate(request_count_total{status="error"}[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} req/s"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(request_latency_seconds_bucket[5m])) > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High API latency"
          description: "p95 latency is {{ $value }}s"
```

Update `prometheus.yml`:
```yaml
rule_files:
  - /etc/prometheus/alerts.yml

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
```

---

For API reference, see [../docs/API.md](../docs/API.md).
For training guide, see [../docs/HOW_TO_TRAIN.md](../docs/HOW_TO_TRAIN.md).
