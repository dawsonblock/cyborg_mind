# Deployment Guide

## Docker

Build the production image:
```bash
docker build -t cyborg-brain:latest .
```

Run locally:
```bash
docker run -p 8000:8000 cyborg-brain:latest
```

## Kubernetes

Deploy to a K8s cluster using the provided manifests.

1. **Create Secret** (for API Token):
```bash
kubectl create secret generic cyborg-secrets --from-literal=api-token=supersecret123
```

2. **Apply Deployment**:
```bash
kubectl apply -f deploy/k8s/deployment.yaml
```

## Nginx Reverse Proxy

For bare-metal or VM deployments, use Nginx as a reverse proxy with SSL.
Copy `deploy/nginx.conf` to `/etc/nginx/nginx.conf`.

## Scaling

- **Horizontal**: Increase `replicas` in `deployment.yaml`.
- **Vertical**: Adjust CPU/Memory limits in `deployment.yaml`.
