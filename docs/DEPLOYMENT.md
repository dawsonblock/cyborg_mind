# Deployment Guide

Production deployment strategies for CyborgMind v4.0 API server.

## Table of Contents

- [Quick Start](#quick-start)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Production Checklist](#production-checklist)
- [Monitoring](#monitoring)
- [Scaling](#scaling)
- [Security](#security)

---

## Quick Start

### Local Development

```bash
# Start server with auto-reload
python scripts/run_api_server.py \
    --config configs/treechop_ppo.yaml \
    --checkpoint artifacts/minerl_treechop/run_v1/best_model.pt \
    --reload
```

### Production (Single Server)

```bash
# Start with Gunicorn + Uvicorn workers
gunicorn cyborg_rl.server:create_app \
    -k uvicorn.workers.UvicornWorker \
    -w 4 \
    -b 0.0.0.0:8000 \
    --access-logfile - \
    --error-logfile - \
    --log-level info
```

---

## Docker Deployment

### Build Image

```bash
# Standard build
docker build -t cyborg-mind:v4.0 .

# GPU-enabled build
docker build -f Dockerfile.gpu -t cyborg-mind:v4.0-gpu .
```

### Run Container

```bash
# CPU version
docker run -d \
    --name cyborg-api \
    -p 8000:8000 \
    -v $(pwd)/artifacts:/app/artifacts:ro \
    -v $(pwd)/configs:/app/configs:ro \
    -e CYBORG_CONFIG_PATH=/app/configs/treechop_ppo.yaml \
    -e CYBORG_CHECKPOINT_PATH=/app/artifacts/minerl_treechop/run_v1/best_model.pt \
    -e CYBORG_DEVICE=cpu \
    cyborg-mind:v4.0

# GPU version
docker run -d \
    --name cyborg-api-gpu \
    --gpus all \
    -p 8000:8000 \
    -v $(pwd)/artifacts:/app/artifacts:ro \
    -v $(pwd)/configs:/app/configs:ro \
    -e CYBORG_CONFIG_PATH=/app/configs/treechop_ppo.yaml \
    -e CYBORG_CHECKPOINT_PATH=/app/artifacts/minerl_treechop/run_v1/best_model.pt \
    -e CYBORG_DEVICE=cuda \
    cyborg-mind:v4.0-gpu
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  cyborg-api:
    image: cyborg-mind:v4.0
    ports:
      - "8000:8000"
    volumes:
      - ./artifacts:/app/artifacts:ro
      - ./configs:/app/configs:ro
    environment:
      - CYBORG_CONFIG_PATH=/app/configs/treechop_ppo.yaml
      - CYBORG_CHECKPOINT_PATH=/app/artifacts/minerl_treechop/run_v1/best_model.pt
      - CYBORG_DEVICE=cpu
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - ./monitoring/grafana:/etc/grafana/provisioning:ro
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    restart: unless-stopped
```

**Start stack:**
```bash
docker-compose up -d
```

---

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.20+)
- kubectl configured
- Container registry access

### Deploy to Kubernetes

```bash
# Apply deployment
kubectl apply -f deploy/k8s/deployment.yaml

# Check status
kubectl get pods -l app=cyborg-api
kubectl get svc cyborg-api

# View logs
kubectl logs -f deployment/cyborg-api
```

### Example Deployment YAML

```yaml
# deploy/k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cyborg-api
  labels:
    app: cyborg-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cyborg-api
  template:
    metadata:
      labels:
        app: cyborg-api
    spec:
      containers:
      - name: api
        image: cyborg-mind:v4.0
        ports:
        - containerPort: 8000
        env:
        - name: CYBORG_CONFIG_PATH
          value: "/app/configs/treechop_ppo.yaml"
        - name: CYBORG_CHECKPOINT_PATH
          value: "/app/artifacts/best_model.pt"
        - name: CYBORG_DEVICE
          value: "cpu"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        volumeMounts:
        - name: models
          mountPath: /app/artifacts
          readOnly: true
        - name: configs
          mountPath: /app/configs
          readOnly: true
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: cyborg-models-pvc
      - name: configs
        configMap:
          name: cyborg-configs

---
apiVersion: v1
kind: Service
metadata:
  name: cyborg-api
spec:
  selector:
    app: cyborg-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Horizontal Pod Autoscaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: cyborg-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: cyborg-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## Production Checklist

### Security

- [ ] Change default authentication token
- [ ] Enable JWT authentication (`api.jwt_enabled: true`)
- [ ] Use strong JWT secret (min 32 characters)
- [ ] Enable HTTPS/TLS (via reverse proxy)
- [ ] Configure CORS allowed origins
- [ ] Set up rate limiting
- [ ] Enable request logging
- [ ] Implement API key rotation

### Performance

- [ ] Tune worker count (CPU cores × 2-4)
- [ ] Configure connection pooling
- [ ] Enable response caching (if applicable)
- [ ] Set appropriate timeout values
- [ ] Monitor memory usage
- [ ] Profile inference latency

### Reliability

- [ ] Set up health checks
- [ ] Configure auto-restart on failure
- [ ] Implement graceful shutdown
- [ ] Set resource limits (CPU/memory)
- [ ] Enable request retries (client-side)
- [ ] Set up backup/failover

### Monitoring

- [ ] Enable Prometheus metrics
- [ ] Set up Grafana dashboards
- [ ] Configure alerting rules
- [ ] Monitor error rates
- [ ] Track inference latency
- [ ] Monitor memory pressure

### Operations

- [ ] Document deployment process
- [ ] Set up CI/CD pipeline
- [ ] Implement blue-green deployment
- [ ] Configure log aggregation
- [ ] Set up backup strategy
- [ ] Create runbook for incidents

---

## Monitoring

### Prometheus Metrics

The API exposes metrics at `/metrics`:

```bash
curl http://localhost:8000/metrics
```

**Key metrics:**
- `cyborg_api_requests_total` - Total requests by endpoint and status
- `cyborg_api_latency_seconds` - Request latency histogram
- `cyborg_agent_memory_pressure` - Agent memory pressure
- `cyborg_websocket_connections_total` - WebSocket connections

### Grafana Dashboard

Import the provided dashboard:

```bash
# Import dashboard
curl -X POST http://localhost:3000/api/dashboards/db \
    -H "Content-Type: application/json" \
    -d @monitoring/grafana/cyborg_api.json
```

### Alerting Rules

Example Prometheus alert rules:

```yaml
# alerts.yml
groups:
- name: cyborg_api
  rules:
  - alert: HighErrorRate
    expr: rate(cyborg_api_requests_total{status="error"}[5m]) > 0.05
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} errors/sec"

  - alert: HighLatency
    expr: histogram_quantile(0.95, cyborg_api_latency_seconds_bucket) > 1.0
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High API latency"
      description: "P95 latency is {{ $value }}s"

  - alert: ServiceDown
    expr: up{job="cyborg_api"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "CyborgMind API is down"
```

---

## Scaling

### Vertical Scaling

Increase resources for single instance:

```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "2000m"
  limits:
    memory: "8Gi"
    cpu: "4000m"
```

### Horizontal Scaling

Add more replicas:

```bash
# Manual scaling
kubectl scale deployment cyborg-api --replicas=5

# Auto-scaling (see HPA section above)
kubectl autoscale deployment cyborg-api --min=2 --max=10 --cpu-percent=70
```

### Load Balancing

Use Nginx as reverse proxy:

```nginx
# /etc/nginx/sites-available/cyborg-api
upstream cyborg_backend {
    least_conn;
    server 10.0.1.10:8000;
    server 10.0.1.11:8000;
    server 10.0.1.12:8000;
}

server {
    listen 80;
    server_name api.cyborg.example.com;

    location / {
        proxy_pass http://cyborg_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## Security

### HTTPS/TLS

Use Let's Encrypt with Certbot:

```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d api.cyborg.example.com

# Auto-renewal
sudo certbot renew --dry-run
```

### JWT Configuration

Production JWT settings:

```yaml
api:
  jwt_enabled: true
  jwt_secret: "your-production-secret-min-32-chars-random"
  jwt_algorithm: "HS256"
  jwt_issuer: "cyborg-api-prod"
  jwt_audience: "cyborg-clients"
  jwt_expiry_minutes: 60
```

### Network Security

- Use VPC/private networks
- Configure security groups/firewall rules
- Enable DDoS protection
- Use API gateway for additional security layer

### Secrets Management

Use Kubernetes secrets or external secret managers:

```bash
# Create secret
kubectl create secret generic cyborg-secrets \
    --from-literal=jwt-secret='your-secret-here' \
    --from-literal=auth-token='your-token-here'

# Reference in deployment
env:
- name: JWT_SECRET
  valueFrom:
    secretKeyRef:
      name: cyborg-secrets
      key: jwt-secret
```

---

## Troubleshooting

### Common Issues

**Port already in use:**
```bash
# Find process using port 8000
lsof -i :8000
# Kill process
kill -9 <PID>
```

**Out of memory:**
- Reduce worker count
- Increase container memory limits
- Enable swap (not recommended for production)

**High latency:**
- Check model size and inference time
- Profile with `cProfile` or `py-spy`
- Consider GPU acceleration
- Implement request batching

**Connection timeouts:**
- Increase timeout values in reverse proxy
- Check network connectivity
- Monitor server load

---

## Best Practices

1. **Use environment-specific configs** - Separate dev/staging/prod configs
2. **Implement health checks** - Both liveness and readiness probes
3. **Enable structured logging** - JSON logs for easy parsing
4. **Monitor everything** - Metrics, logs, traces
5. **Automate deployments** - CI/CD pipeline with automated tests
6. **Plan for failure** - Implement retries, circuit breakers, fallbacks
7. **Document everything** - Runbooks, architecture diagrams, API docs
8. **Test at scale** - Load testing before production
9. **Secure by default** - Enable authentication, use HTTPS, rotate secrets
10. **Keep it simple** - Start small, scale as needed

---

## Additional Resources

- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Uvicorn Deployment](https://www.uvicorn.org/deployment/)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/)
- [Prometheus Monitoring](https://prometheus.io/docs/introduction/overview/)
- [Grafana Dashboards](https://grafana.com/docs/grafana/latest/dashboards/)

---

For training guides, see [HOW_TO_TRAIN.md](HOW_TO_TRAIN.md).  
For API reference, see [API.md](API.md).
