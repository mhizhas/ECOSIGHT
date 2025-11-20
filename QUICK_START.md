# 🚀 EcoSight Quick Start Guide

## Start the System

```bash
export DOCKER_HOST=unix://$HOME/.docker/run/docker.sock
cd /Users/cococe/Desktop/EcoSight
docker compose -f deployment/docker-compose.yml up -d
```

**Access Points:**
- Dashboard: http://localhost:8501
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

---

## Stop the System

```bash
export DOCKER_HOST=unix://$HOME/.docker/run/docker.sock
docker compose -f deployment/docker-compose.yml down
```

---

## View Logs

```bash
export DOCKER_HOST=unix://$HOME/.docker/run/docker.sock
docker compose -f deployment/docker-compose.yml logs -f
```

---

## Retrain the Model

### Option 1: Via API
```bash
curl -X POST http://localhost:8000/retrain \
  -H "Content-Type: application/json" \
  -d '{"trigger_reason": "Manual retraining"}'
```

### Option 2: Run Script Directly
```bash
conda activate ecosight
python scripts/retrain_model.py
```

### Option 3: Inside Docker
```bash
export DOCKER_HOST=unix://$HOME/.docker/run/docker.sock
docker compose -f deployment/docker-compose.yml exec api python ../scripts/retrain_model.py
```

---

## Add New Training Data

```bash
# Add audio files to appropriate directories
cp /path/to/gunshot.wav augmented_audio/gun_shot/
cp /path/to/dog.wav augmented_audio/dog_bark/
cp /path/to/engine.wav augmented_audio/engine_idling/

# Check file counts
find augmented_audio/ -name "*.wav" | wc -l
```

---

## Useful Commands

### Check System Status
```bash
export DOCKER_HOST=unix://$HOME/.docker/run/docker.sock
docker compose -f deployment/docker-compose.yml ps
```

### View Model Performance
```bash
cat models/performance_metrics.json | python -m json.tool
```

### Check Retraining History
```bash
cat models/retraining_log.json | python -m json.tool
```

### Rebuild Containers
```bash
export DOCKER_HOST=unix://$HOME/.docker/run/docker.sock
docker compose -f deployment/docker-compose.yml up -d --build
```

---

## Troubleshooting

### Can't connect to Docker?
```bash
# Start Docker Desktop manually, then:
export DOCKER_HOST=unix://$HOME/.docker/run/docker.sock
docker ps
```

### API not responding?
```bash
export DOCKER_HOST=unix://$HOME/.docker/run/docker.sock
docker compose -f deployment/docker-compose.yml restart api
docker compose -f deployment/docker-compose.yml logs api --tail=50
```

### UI showing errors?
```bash
export DOCKER_HOST=unix://$HOME/.docker/run/docker.sock
docker compose -f deployment/docker-compose.yml restart ui
docker compose -f deployment/docker-compose.yml logs ui --tail=50
```

---

**For detailed documentation, see:**
- `docs/RETRAINING_GUIDE.md` - Model retraining
- `docs/DEPLOYMENT.md` - Deployment guide
- `STRUCTURE.md` - Project structure

