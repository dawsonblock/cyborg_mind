# CyborgMind V2 - Deployment Guide

## Deployment Options

1. **Local Development**: Run on workstation
2. **Docker Container**: Reproducible deployment
3. **Cloud Service**: AWS/GCP/Azure
4. **Edge Device**: Raspberry Pi / Jetson

---

## Quick Start

```bash
# 1. Install
pip install -e .

# 2. Start API server
uvicorn cyborg_mind_v2.deployment.api_server:app --host 0.0.0.0 --port 8000

# 3. Open visualizer
cd frontend/demo && python -m http.server 8080
```

Visit `http://localhost:8080`

---

## Docker Deployment

**Build Image**:
```bash
docker build -t cyborgmind:latest .
```

**Run Container**:
```bash
docker run -d \
  --name cyborgmind \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  cyborgmind:latest
```

**Docker Compose**:
```yaml
version: '3.8'
services:
  api:
    image: cyborgmind:latest
    ports:
      - "8000:8000"
    volumes:
      - ./checkpoints:/app/checkpoints
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## Production Configuration

**Environment Variables**:
```bash
export CYBORGMIND_DEVICE=cuda  # or cpu
export CYBORGMIND_CHECKPOINT=/app/checkpoints/treechop_brain.pt
export CYBORGMIND_LOG_LEVEL=INFO
export CYBORGMIND_MAX_AGENTS=100
```

**API Server Options**:
```bash
uvicorn cyborg_mind_v2.deployment.api_server:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-level info \
  --access-log
```

---

## Monitoring

**Health Check**:
```bash
curl http://localhost:8000/health
```

**Metrics**:
```bash
curl http://localhost:8000/metrics
```

**TensorBoard**:
```bash
tensorboard --logdir=logs --port=6006
```

---

## Scaling

**Horizontal Scaling**:
- Deploy multiple API instances behind load balancer
- Use Redis for shared agent state (future)

**Vertical Scaling**:
- Batch multiple agents in single forward pass
- Use larger GPU for more parallel agents

**Benchmark**:
```bash
# Load test
ab -n 1000 -c 10 -p request.json -T application/json http://localhost:8000/step
```

---

## Security

**API Key Auth**:
```python
# Add to api_server.py
from fastapi.security import APIKeyHeader
api_key_header = APIKeyHeader(name="X-API-Key")

@app.post("/step")
async def step(request: StepRequest, api_key: str = Depends(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(401, "Invalid API key")
    ...
```

**HTTPS**:
```bash
uvicorn cyborg_mind_v2.deployment.api_server:app \
  --ssl-keyfile=key.pem \
  --ssl-certfile=cert.pem
```

---

## Integration Examples

**Unity Game**:
```csharp
using UnityEngine.Networking;
using Newtonsoft.Json;

public class CyborgMindClient {
    string apiURL = "http://localhost:8000";

    public async Task<int> GetAction(float[] pixels, float[] scalars, float[] goal) {
        var request = new {
            agent_id = "player_1",
            pixels = pixels,
            scalars = scalars,
            goal = goal
        };

        string json = JsonConvert.SerializeObject(request);
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(json);

        var webRequest = new UnityWebRequest($"{apiURL}/step", "POST");
        webRequest.uploadHandler = new UploadHandlerRaw(bodyRaw);
        webRequest.downloadHandler = new DownloadHandlerBuffer();
        webRequest.SetRequestHeader("Content-Type", "application/json");

        await webRequest.SendWebRequest();

        var response = JsonConvert.DeserializeObject<StepResponse>(
            webRequest.downloadHandler.text
        );

        return response.action;
    }
}
```

**Python Client**:
```python
import requests
import numpy as np

class CyborgMindClient:
    def __init__(self, api_url="http://localhost:8000", agent_id="agent_0"):
        self.api_url = api_url
        self.agent_id = agent_id
        self.reset()

    def reset(self):
        response = requests.post(
            f"{self.api_url}/reset",
            json={"agent_id": self.agent_id}
        )
        return response.json()

    def step(self, pixels, scalars, goal):
        response = requests.post(
            f"{self.api_url}/step",
            json={
                "agent_id": self.agent_id,
                "pixels": pixels.tolist() if isinstance(pixels, np.ndarray) else pixels,
                "scalars": scalars,
                "goal": goal,
            }
        )
        return response.json()
```

---

## Cloud Deployment

**AWS EC2** (GPU instance):
```bash
# 1. Launch p3.2xlarge instance (V100 GPU)
# 2. Install CUDA + Docker
# 3. Clone repo and build image
# 4. Run container
# 5. Expose via ELB
```

**GCP Compute Engine**:
```bash
gcloud compute instances create cyborgmind-server \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=50GB \
  --metadata-from-file startup-script=startup.sh
```

**Azure**:
```bash
az vm create \
  --resource-group CyborgMind \
  --name cyborgmind-vm \
  --image UbuntuLTS \
  --size Standard_NC6 \
  --admin-username azureuser
```

---

## Troubleshooting

**GPU Not Detected**:
```bash
# Check CUDA
nvidia-smi

# Check PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

**OOM Error**:
- Reduce batch size
- Use CPU mode
- Decrease PMM max_slots

**Slow Inference**:
- Check GPU utilization
- Profile with `torch.profiler`
- Use TorchScript compilation

---

## Backup & Recovery

**Checkpoint Backup**:
```bash
# Periodic backup
rsync -avz checkpoints/ backup/checkpoints_$(date +%Y%m%d)/

# S3 backup
aws s3 sync checkpoints/ s3://my-bucket/cyborgmind/checkpoints/
```

**Disaster Recovery**:
1. Keep checkpoints in cloud storage
2. Use infrastructure-as-code (Terraform)
3. Automate deployment with CI/CD
4. Monitor with alerting (Grafana + Prometheus)

---

For development guide, see `docs/DEVELOPMENT.md`
For API reference, see `docs/API.md`
