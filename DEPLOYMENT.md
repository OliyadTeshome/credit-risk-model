# Deployment Guide - Credit Risk Model

This guide covers deploying the Credit Risk Model API in various environments, from local development to production cloud platforms.

## 🏠 Local Development

### Prerequisites
- Python 3.10+
- Docker and Docker Compose
- Git

### Quick Start
```bash
# Clone and setup
git clone <your-repo-url>
cd credit-risk-model

# Install dependencies
pip install -r requirements.txt

# Train the model
python src/train.py

# Run API with Docker
docker-compose up --build
```

### Development Workflow
```bash
# Start development environment
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild after code changes
docker-compose up --build
```

## 🚀 Production Deployment

### Option 1: Docker Standalone

#### 1. Build Production Image
```bash
# Build optimized image
docker build -t credit-risk-api:latest .

# Tag for registry (if using)
docker tag credit-risk-api:latest your-registry/credit-risk-api:latest
```

#### 2. Run Production Container
```bash
# Basic run
docker run -d -p 8000:8000 --name credit-risk-api credit-risk-api:latest

# With environment variables
docker run -d \
  -p 8000:8000 \
  -e ENVIRONMENT=production \
  -e LOG_LEVEL=info \
  --name credit-risk-api \
  credit-risk-api:latest

# With volume mounts for logs
docker run -d \
  -p 8000:8000 \
  -v /var/log/credit-risk:/app/logs \
  --name credit-risk-api \
  credit-risk-api:latest
```

#### 3. Docker Compose Production
Create `docker-compose.prod.yml`:
```yaml
version: '3.8'
services:
  api:
    build: .
    container_name: credit-risk-api-prod
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=info
    restart: unless-stopped
    volumes:
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Option 2: Cloud Platforms

#### AWS Deployment

##### AWS ECS (Elastic Container Service)
```bash
# 1. Create ECR repository
aws ecr create-repository --repository-name credit-risk-api

# 2. Build and push image
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag credit-risk-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/credit-risk-api:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/credit-risk-api:latest

# 3. Deploy to ECS (using AWS Console or CLI)
```

##### AWS Lambda (Serverless)
```python
# Create lambda_function.py
import json
from src.api.main import app
from mangum import Mangum

handler = Mangum(app)
```

#### Google Cloud Platform

##### Google Cloud Run
```bash
# 1. Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT-ID/credit-risk-api

# 2. Deploy to Cloud Run
gcloud run deploy credit-risk-api \
  --image gcr.io/PROJECT-ID/credit-risk-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8000
```

#### Azure

##### Azure Container Instances
```bash
# 1. Build and push to Azure Container Registry
az acr build --registry <registry-name> --image credit-risk-api .

# 2. Deploy to Container Instances
az container create \
  --resource-group <resource-group> \
  --name credit-risk-api \
  --image <registry-name>.azurecr.io/credit-risk-api:latest \
  --ports 8000 \
  --dns-name-label credit-risk-api
```

### Option 3: Kubernetes

#### Basic Kubernetes Deployment
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: credit-risk-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: credit-risk-api
  template:
    metadata:
      labels:
        app: credit-risk-api
    spec:
      containers:
      - name: api
        image: credit-risk-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
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
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: credit-risk-api-service
spec:
  selector:
    app: credit-risk-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

```bash
# Deploy to Kubernetes
kubectl apply -f k8s-deployment.yaml
```

## 🔧 Configuration

### Environment Variables
```bash
# Application
ENVIRONMENT=production
LOG_LEVEL=info
DEBUG=false

# Model
MODEL_PATH=/app/models/credit_risk_model.joblib

# API
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Security
API_KEY=your-secret-key
CORS_ORIGINS=https://yourdomain.com
```

### Production Dockerfile Optimizations
```dockerfile
# Multi-stage build for smaller image
FROM python:3.10-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.10-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY src ./src
COPY models ./models

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

## 📊 Monitoring & Observability

### Health Checks
```python
# Add to src/api/main.py
@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/ready")
def readiness_check():
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "ready"}
```

### Logging
```python
# Add structured logging
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()
    response = await call_next(request)
    duration = datetime.now() - start_time
    
    logger.info(
        f"{request.method} {request.url.path} - {response.status_code} - {duration.total_seconds():.3f}s"
    )
    return response
```

### Metrics (Prometheus)
```python
# Add to requirements.txt: prometheus-client
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

## 🔒 Security

### API Authentication
```python
# Add to src/api/main.py
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != "your-secret-token":
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

@app.post("/predict", dependencies=[Depends(verify_token)])
def predict_risk_endpoint(request: RiskRequest):
    # ... existing code
```

### CORS Configuration
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## 🚨 Troubleshooting

### Common Production Issues

1. **High Memory Usage**
   ```bash
   # Monitor container resources
   docker stats credit-risk-api
   
   # Adjust memory limits
   docker run -m 1g credit-risk-api:latest
   ```

2. **Slow Response Times**
   ```bash
   # Check logs for bottlenecks
   docker logs credit-risk-api
   
   # Monitor CPU usage
   docker stats --no-stream credit-risk-api
   ```

3. **Model Loading Issues**
   ```bash
   # Verify model file exists
   docker exec credit-risk-api ls -la /app/models/
   
   # Check model file size
   docker exec credit-risk-api du -h /app/models/credit_risk_model.joblib
   ```

### Performance Optimization

1. **Enable Gunicorn with multiple workers**
   ```bash
   # Update Dockerfile CMD
   CMD ["gunicorn", "src.api.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
   ```

2. **Add caching for predictions**
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=1000)
   def cached_predict(features_tuple):
       return model.predict_proba(np.array([features_tuple]))[0, 1]
   ```

## 📈 Scaling

### Horizontal Scaling
```bash
# Docker Swarm
docker service create --name credit-risk-api --replicas 3 -p 8000:8000 credit-risk-api:latest

# Kubernetes
kubectl scale deployment credit-risk-api --replicas=5
```

### Load Balancing
```nginx
# Nginx configuration
upstream credit_risk_api {
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
}

server {
    listen 80;
    location / {
        proxy_pass http://credit_risk_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

**For additional support or questions about deployment, please refer to the main README.md or open an issue on [GitHub](https://github.com/OliyadTeshome/credit-risk-model.git).** 