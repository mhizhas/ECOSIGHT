# EcoSight Wildlife Monitoring - Render Deployment Guide

## Overview
This guide walks you through deploying EcoSight to Render with both API and UI services.

## Prerequisites
- GitHub account with EcoSight repository pushed
- Render account (free tier works)
- Your model files committed to the repository

## Deployment Steps

### 1. Prepare Your Repository

Ensure these files are committed:
```bash
git add render.yaml deployment/Dockerfile deployment/Dockerfile.streamlit
git add models/yamnet_classifier_v2.keras models/class_names.json models/model_metadata.json
git commit -m "Add Render deployment configuration"
git push origin main
```

### 2. Connect to Render

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **"New +"** → **"Blueprint"**
3. Connect your GitHub repository
4. Select the repository containing EcoSight
5. Render will detect `render.yaml` automatically

### 3. Configure Services

Render will create two services from `render.yaml`:

**ecosight-api** (FastAPI Backend)
- Runtime: Docker
- Region: Oregon (change in render.yaml if needed)
- Plan: Starter ($7/month) or Free
- Disk: 10GB persistent storage
- Health check: `/status`

**ecosight-ui** (Streamlit Dashboard)
- Runtime: Docker
- Region: Oregon
- Plan: Starter ($7/month) or Free
- Health check: `/_stcore/health`
- Auto-connected to API service

### 4. Environment Variables (Auto-configured)

The `render.yaml` sets these automatically:
- `API_URL`: Auto-linked from ecosight-api service
- `PORT`: 8000 (API), 8501 (UI)
- `PYTHONUNBUFFERED`: 1
- Streamlit server configs

### 5. Deploy

1. Click **"Apply"** to create both services
2. Render will:
   - Build Docker images
   - Deploy API service
   - Deploy UI service
   - Link services together

**Build time**: ~5-10 minutes per service

### 6. Access Your Application

After deployment completes:
- **API**: `https://ecosight-api.onrender.com`
- **UI**: `https://ecosight-ui.onrender.com`

Test the API:
```bash
curl https://ecosight-api.onrender.com/status
```

## Monitoring

### View Logs
1. Go to Render Dashboard
2. Click on service (ecosight-api or ecosight-ui)
3. Navigate to **"Logs"** tab

### Health Checks
Render automatically monitors:
- API: `GET /status` every 30s
- UI: `GET /_stcore/health` every 30s

## Persistent Storage

The API service includes a 10GB disk for:
- Uploaded audio files (`/app/uploads`)
- Augmented audio (`/app/augmented_audio`)
- Model files (`/app/models`)

**Note**: Free tier doesn't include persistent disks. Upgrade to Starter plan for storage.

## Updating Your Deployment

### Auto-Deploy (Enabled by default)
Push to main branch:
```bash
git add .
git commit -m "Update model/code"
git push origin main
```
Render auto-deploys within minutes.

### Manual Deploy
1. Go to service in Render Dashboard
2. Click **"Manual Deploy"** → **"Deploy latest commit"**

## Troubleshooting

### Build Fails
- Check Render build logs
- Verify all files are committed (models/, config/, src/)
- Ensure requirements.txt is complete

### API/UI Can't Connect
- Check environment variables in Render Dashboard
- Verify `API_URL` is set correctly in ecosight-ui
- Check service logs for connection errors

### Out of Memory
- Upgrade to larger Render plan
- Reduce model complexity
- Optimize Docker image size

### Model Files Too Large
If model files exceed GitHub limits (>100MB):

**Option 1: Use Git LFS**
```bash
git lfs install
git lfs track "models/*.keras"
git add .gitattributes models/
git commit -m "Add models with LFS"
git push
```

**Option 2: Download during build**
Add to Dockerfile:
```dockerfile
RUN curl -o models/yamnet_classifier_v2.keras https://your-storage-url/model.keras
```

## Cost Estimate

### Free Tier
- API: Free (spins down after inactivity)
- UI: Free (spins down after inactivity)
- **Limitation**: No persistent storage, cold starts

### Starter Plan
- API: $7/month (always on)
- UI: $7/month (always on)
- Disk: $0.25/GB/month (10GB = $2.50/month)
- **Total**: ~$16.50/month

## Scaling

### Increase Resources
In `render.yaml`, change:
```yaml
plan: starter  # → standard, pro
disk:
  sizeGB: 10   # → 20, 50, 100
```

### Add Workers
For retraining jobs, add background worker:
```yaml
- type: worker
  name: ecosight-worker
  runtime: docker
  dockerfilePath: ./deployment/Dockerfile
  dockerContext: .
  startCommand: python retrain_model.py
```

## Custom Domain

1. Go to service settings
2. Click **"Custom Domain"**
3. Add your domain (e.g., `ecosight.yourdomain.com`)
4. Update DNS records as instructed
5. Render provides free SSL

## Security

### API Keys (Recommended)
Add to `render.yaml`:
```yaml
envVars:
  - key: API_KEY
    generateValue: true
```

Update API to require authentication:
```python
from fastapi import Header, HTTPException

@app.post("/predict")
async def predict(api_key: str = Header(None)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(401, "Invalid API key")
    # ... prediction logic
```

## Next Steps

1. ✅ Deploy to Render
2. Set up custom domain (optional)
3. Configure API authentication
4. Set up monitoring/alerts
5. Schedule automated retraining
6. Add MongoDB for audio storage (see MongoDB guide)

## Support

- Render Docs: https://render.com/docs
- EcoSight Issues: https://github.com/mangaorphy/ECOSIGHT/issues
- Render Community: https://community.render.com
