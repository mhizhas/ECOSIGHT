# 🚀 EcoSight Deployment Guide

## Cloud Deployment & Production Evaluation

This guide covers deploying the EcoSight Wildlife Monitoring system to production and evaluating its performance under real-world conditions.

---

## 📋 Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Local Testing](#local-testing)
3. [Cloud Platform Deployment](#cloud-platform-deployment)
4. [Production Configuration](#production-configuration)
5. [Monitoring & Logging](#monitoring--logging)
6. [Load Testing in Production](#load-testing-in-production)
7. [Scaling Strategies](#scaling-strategies)
8. [Troubleshooting](#troubleshooting)

---

## ✅ Pre-Deployment Checklist

Before deploying to production, ensure:

- [ ] Model artifacts are trained and validated
- [ ] All dependencies are in `requirements.txt`
- [ ] Dockerfiles build successfully
- [ ] Environment variables are configured
- [ ] Health check endpoints are working
- [ ] API documentation is up-to-date
- [ ] Security credentials are not hardcoded
- [ ] Load tests pass locally
- [ ] Backup strategy is defined
- [ ] Monitoring tools are configured

---

## 🧪 Local Testing

### 1. Test with Docker Compose

```bash
# Build and start all services
docker-compose up -d

# Check container status
docker-compose ps

# View logs
docker-compose logs -f api

# Test API health
curl http://localhost:8000/health

# Test prediction endpoint
curl -X POST http://localhost:8000/predict \
  -F "file=@test_audio.wav"

# Stop all services
docker-compose down
```

### 2. Test with Different Scales

```bash
# Single container
docker-compose up -d --scale api=1
locust -f locustfile.py --host=http://localhost:80 \
  --users=50 --spawn-rate=5 --run-time=2m --headless

# Multiple containers (3)
docker-compose up -d --scale api=3
locust -f locustfile.py --host=http://localhost:80 \
  --users=150 --spawn-rate=15 --run-time=2m --headless

# High scale (5 containers)
docker-compose up -d --scale api=5
locust -f locustfile.py --host=http://localhost:80 \
  --users=250 --spawn-rate=25 --run-time=2m --headless
```

**Expected Results:**

| Scale | Avg Latency | P95 Latency | RPS | Error Rate |
|-------|-------------|-------------|-----|------------|
| 1     | ~500ms      | ~800ms      | 50  | <1%        |
| 3     | ~200ms      | ~350ms      | 150 | <0.5%      |
| 5     | ~150ms      | ~250ms      | 250 | <0.1%      |

---

## ☁️ Cloud Platform Deployment

### Option 1: AWS (Elastic Container Service)

#### Step 1: Create ECR Repository

```bash
# Create repository
aws ecr create-repository --repository-name ecosight-api --region us-east-1

# Authenticate Docker to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  <account-id>.dkr.ecr.us-east-1.amazonaws.com
```

#### Step 2: Build and Push Image

```bash
# Build image
docker build -t ecosight-api .

# Tag image
docker tag ecosight-api:latest \
  <account-id>.dkr.ecr.us-east-1.amazonaws.com/ecosight-api:latest

# Push to ECR
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/ecosight-api:latest
```

#### Step 3: Create ECS Cluster

```bash
# Create cluster
aws ecs create-cluster --cluster-name ecosight-cluster

# Create task definition (save as task-def.json)
aws ecs register-task-definition --cli-input-json file://task-def.json
```

**task-def.json:**
```json
{
  "family": "ecosight-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [
    {
      "name": "ecosight-api",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/ecosight-api:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "essential": true,
      "environment": [
        {
          "name": "PYTHONUNBUFFERED",
          "value": "1"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/ecosight-api",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 10,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

#### Step 4: Create Application Load Balancer

```bash
# Create load balancer
aws elbv2 create-load-balancer \
  --name ecosight-alb \
  --subnets subnet-xxx subnet-yyy \
  --security-groups sg-xxx

# Create target group
aws elbv2 create-target-group \
  --name ecosight-tg \
  --protocol HTTP \
  --port 8000 \
  --vpc-id vpc-xxx \
  --target-type ip \
  --health-check-path /health
```

#### Step 5: Create ECS Service

```bash
aws ecs create-service \
  --cluster ecosight-cluster \
  --service-name ecosight-api-service \
  --task-definition ecosight-api \
  --desired-count 3 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx,subnet-yyy],securityGroups=[sg-xxx],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:...,containerName=ecosight-api,containerPort=8000"
```

#### Step 6: Setup Auto-Scaling

```bash
# Register scalable target
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/ecosight-cluster/ecosight-api-service \
  --min-capacity 2 \
  --max-capacity 10

# Create scaling policy
aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/ecosight-cluster/ecosight-api-service \
  --policy-name cpu-scaling-policy \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration file://scaling-policy.json
```

**scaling-policy.json:**
```json
{
  "TargetValue": 70.0,
  "PredefinedMetricSpecification": {
    "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
  },
  "ScaleInCooldown": 300,
  "ScaleOutCooldown": 60
}
```

---

### Option 2: Google Cloud Platform (Cloud Run)

```bash
# Enable APIs
gcloud services enable run.googleapis.com containerregistry.googleapis.com

# Build and submit
gcloud builds submit --tag gcr.io/PROJECT_ID/ecosight-api

# Deploy to Cloud Run
gcloud run deploy ecosight-api \
  --image gcr.io/PROJECT_ID/ecosight-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --min-instances 2 \
  --max-instances 10 \
  --concurrency 80 \
  --timeout 300

# Get service URL
gcloud run services describe ecosight-api --format='value(status.url)'
```

---

### Option 3: Azure (Container Instances)

```bash
# Create resource group
az group create --name ecosight-rg --location eastus

# Create Azure Container Registry
az acr create --resource-group ecosight-rg \
  --name ecosightacr --sku Basic

# Build and push image
az acr build --registry ecosightacr \
  --image ecosight-api:latest .

# Deploy container instance
az container create \
  --resource-group ecosight-rg \
  --name ecosight-api \
  --image ecosightacr.azurecr.io/ecosight-api:latest \
  --cpu 2 \
  --memory 4 \
  --registry-login-server ecosightacr.azurecr.io \
  --registry-username <username> \
  --registry-password <password> \
  --dns-name-label ecosight-api \
  --ports 8000
```

---

## ⚙️ Production Configuration

### Environment Variables

Create `.env` file (DO NOT commit to git):

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Model Configuration
MODEL_PATH=/app/models/yamnet_classifier.keras
CLASS_NAMES_PATH=/app/models/class_names.json
YAMNET_MODEL_URL=https://tfhub.dev/google/yamnet/1

# Storage Configuration
UPLOAD_DIR=/app/uploads
AUGMENTED_AUDIO_DIR=/app/augmented_audio
MODELS_DIR=/app/models

# Performance Settings
MAX_UPLOAD_SIZE=50MB
REQUEST_TIMEOUT=60
PREDICTION_BATCH_SIZE=32

# Security
API_KEY=your-secret-api-key-here
CORS_ORIGINS=["https://yourdomain.com"]

# Monitoring
LOG_LEVEL=INFO
ENABLE_METRICS=true
METRICS_PORT=9090

# Cloud Storage (optional)
AWS_S3_BUCKET=ecosight-models
GCS_BUCKET=ecosight-models
AZURE_STORAGE_ACCOUNT=ecosight
```

### Docker Compose Production

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  api:
    build: .
    image: ecosight-api:latest
    ports:
      - "8000:8000"
    volumes:
      - model-data:/app/models:ro
      - upload-data:/app/uploads
    environment:
      - PYTHONUNBUFFERED=1
      - LOG_LEVEL=INFO
    restart: always
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.prod.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - api
    restart: always

volumes:
  model-data:
    driver: local
  upload-data:
    driver: local
```

---

## 📊 Monitoring & Logging

### CloudWatch (AWS)

```bash
# Create log group
aws logs create-log-group --log-group-name /ecs/ecosight-api

# View logs in real-time
aws logs tail /ecs/ecosight-api --follow

# Create dashboard
aws cloudwatch put-dashboard \
  --dashboard-name EcoSight \
  --dashboard-body file://cloudwatch-dashboard.json
```

**cloudwatch-dashboard.json:**
```json
{
  "widgets": [
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["AWS/ECS", "CPUUtilization", {"stat": "Average"}],
          [".", "MemoryUtilization", {"stat": "Average"}]
        ],
        "period": 300,
        "stat": "Average",
        "region": "us-east-1",
        "title": "ECS Service Metrics"
      }
    }
  ]
}
```

### Application Insights (Azure)

```bash
# Create Application Insights
az monitor app-insights component create \
  --app ecosight-insights \
  --location eastus \
  --resource-group ecosight-rg

# Get instrumentation key
az monitor app-insights component show \
  --app ecosight-insights \
  --resource-group ecosight-rg \
  --query instrumentationKey
```

### Stackdriver (GCP)

```bash
# View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ecosight-api" --limit 50

# Create log-based metric
gcloud logging metrics create prediction_errors \
  --description="Count of prediction errors" \
  --log-filter='resource.type="cloud_run_revision" AND severity="ERROR"'
```

---

## 🧪 Load Testing in Production

### Pre-Production Stress Test

```bash
# Gradual ramp-up test
locust -f locustfile.py --host=https://your-domain.com \
  --users=1000 \
  --spawn-rate=50 \
  --run-time=10m \
  --headless \
  --csv=production_test_results
```

### Test Different Scenarios

#### Scenario 1: Normal Load
```bash
locust -f locustfile.py --host=https://your-domain.com \
  --users=100 --spawn-rate=10 --run-time=5m --headless
```

#### Scenario 2: Peak Load
```bash
locust -f locustfile.py --host=https://your-domain.com \
  --users=500 --spawn-rate=50 --run-time=10m --headless
```

#### Scenario 3: Stress Test
```bash
locust -f locustfile.py --host=https://your-domain.com \
  --users=1000 --spawn-rate=100 --run-time=15m --headless
```

### Recording Results

Test results will show:
- **Response times** (min, max, median, P95, P99)
- **Throughput** (requests per second)
- **Error rate** (percentage of failed requests)
- **Concurrent users** supported

**Example Results:**

```
Type     Name                      # reqs    Avg     Min     Max   Median  P95    P99  Fail %
------------------------------------------------------------------------
POST     /predict                   45234   342ms    89ms   2145ms   298ms  567ms  892ms  0.34%
GET      /status                    12456    45ms    12ms    234ms    38ms   78ms  123ms  0.00%
GET      /metrics                    6789    67ms    23ms    456ms    54ms  112ms  189ms  0.00%
------------------------------------------------------------------------
         Aggregated                 64479   256ms    12ms   2145ms   189ms  487ms  756ms  0.21%
```

---

## 📈 Scaling Strategies

### Horizontal Scaling (Recommended)

**AWS ECS:**
```bash
# Update service desired count
aws ecs update-service \
  --cluster ecosight-cluster \
  --service ecosight-api-service \
  --desired-count 5
```

**Docker Compose:**
```bash
docker-compose up -d --scale api=5
```

**Kubernetes:**
```bash
kubectl scale deployment ecosight-api --replicas=5
```

### Auto-Scaling Rules

1. **CPU-based:** Scale up when CPU > 70%, scale down when CPU < 30%
2. **Memory-based:** Scale up when Memory > 80%
3. **Request-based:** Scale up when RPS > 100 per instance
4. **Schedule-based:** Scale up during peak hours (e.g., 8 AM - 8 PM)

---

## 🐛 Troubleshooting

### Common Issues

#### Issue 1: High Latency

**Symptoms:** Response time > 1 second

**Solutions:**
1. Check container CPU/Memory usage
2. Increase number of replicas
3. Enable caching for model predictions
4. Optimize model inference (batch predictions)

#### Issue 2: Out of Memory

**Symptoms:** Container crashes with OOM error

**Solutions:**
1. Increase container memory limit
2. Implement request queuing
3. Add memory limits per request
4. Monitor memory leaks

#### Issue 3: Connection Timeouts

**Symptoms:** 504 Gateway Timeout errors

**Solutions:**
1. Increase timeout settings in load balancer
2. Optimize model loading time
3. Implement health check grace period
4. Check network connectivity

### Debug Commands

```bash
# Check container logs
docker logs -f <container-id>

# Check resource usage
docker stats

# Execute command in container
docker exec -it <container-id> bash

# Check API health
curl -v http://localhost:8000/health

# Test prediction locally
curl -X POST http://localhost:8000/predict \
  -F "file=@test.wav" -v
```

---

## 📞 Support

For deployment issues:
- Create an issue on GitHub
- Email: devops@ecosight.org
- Slack: #ecosight-deployment

---

**Happy Deploying! 🚀**
