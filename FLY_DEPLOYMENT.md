# 🚀 Fly.io Deployment Guide for EcoSight

## Quick Start (Automated)

```bash
# Make deployment script executable
chmod +x deploy-fly.sh

# Run automated deployment
./deploy-fly.sh
```

This script will:
1. ✅ Install flyctl (if needed)
2. ✅ Authenticate with Fly.io
3. ✅ Create persistent volume for models
4. ✅ Deploy API service
5. ✅ Deploy UI service
6. ✅ Configure automatic scaling

---

## Manual Step-by-Step Deployment

### Prerequisites

```bash
# Install Fly.io CLI
brew install flyctl

# Sign up/login
flyctl auth signup  # or flyctl auth login
```

### Step 1: Deploy the API

```bash
# Create app (first time only)
flyctl launch --config fly.toml --name ecosight-api --no-deploy

# Create persistent volume for models
flyctl volumes create ecosight_models --region iad --size 1

# Deploy
flyctl deploy --config fly.toml
```

### Step 2: Upload Model Files

```bash
# SSH into the machine
flyctl ssh console --config fly.toml

# From another terminal, copy model files
flyctl ssh sftp shell --config fly.toml
> put models/yamnet_classifier_v2.keras /app/models/
> put models/class_names.json /app/models/
> put models/model_metadata.json /app/models/
> put models/performance_metrics.json /app/models/
> exit
```

### Step 3: Deploy the UI

```bash
# Get API URL first
flyctl status --config fly.toml

# Update fly.streamlit.toml with your API URL
# Then deploy UI
flyctl launch --config fly.streamlit.toml --name ecosight-ui --no-deploy
flyctl deploy --config fly.streamlit.toml
```

---

## Fly.io Commands Reference

### Deployment & Monitoring

```bash
# Deploy/update
flyctl deploy --config fly.toml

# View logs
flyctl logs --config fly.toml

# Check status
flyctl status --config fly.toml

# Open in browser
flyctl open --config fly.toml
```

### Scaling

```bash
# Scale memory (shared-cpu-1x = 256MB, shared-cpu-2x = 512MB, etc.)
flyctl scale memory 512 --config fly.toml

# Scale instances
flyctl scale count 2 --config fly.toml

# Auto-scale (min/max instances)
flyctl autoscale set min=1 max=3 --config fly.toml
```

### SSH & Debugging

```bash
# SSH into machine
flyctl ssh console --config fly.toml

# Execute command
flyctl ssh console --config fly.toml -C "ls -la /app/models"

# SFTP for file transfers
flyctl ssh sftp shell --config fly.toml
```

### Secrets Management

```bash
# Set environment variables
flyctl secrets set API_KEY=your_key --config fly.toml

# List secrets
flyctl secrets list --config fly.toml

# Remove secret
flyctl secrets unset API_KEY --config fly.toml
```

---

## Free Tier Limits

Fly.io free tier includes:
- ✅ Up to 3 shared-cpu-1x VMs (256MB RAM each)
- ✅ 3GB persistent storage
- ✅ 160GB outbound data transfer/month
- ✅ Automatic SSL certificates
- ✅ Auto-scaling to zero (save costs when idle)

**Perfect for EcoSight!**

---

## Optimization Tips

### 1. **Enable Auto-Sleep** (Already configured)
```toml
[http_service]
  auto_stop_machines = "stop"
  auto_start_machines = true
  min_machines_running = 0
```

### 2. **Monitor Costs**
```bash
# View usage
flyctl dashboard

# Check billing
flyctl billing
```

### 3. **Optimize Docker Image**

Add to `.dockerignore`:
```
__pycache__/
*.pyc
.git/
.gitignore
*.md
tests/
.env
augmented_audio/
uploads/
```

### 4. **Use Shared Volumes**
Models are stored in persistent volume (already configured) to avoid rebuilding.

---

## Troubleshooting

### Issue: "Not enough memory"

**Solution:**
```bash
# Upgrade to 512MB
flyctl scale memory 512 --config fly.toml
```

### Issue: "Health check failing"

**Solution:**
```bash
# Check logs
flyctl logs --config fly.toml

# SSH and test manually
flyctl ssh console --config fly.toml
curl http://localhost:8000/health
```

### Issue: "Model files not found"

**Solution:**
```bash
# Upload via SFTP
flyctl ssh sftp shell --config fly.toml
> put models/yamnet_classifier_v2.keras /app/models/
```

---

## Production Checklist

- [ ] Models uploaded to persistent volume
- [ ] Health checks passing
- [ ] SSL certificates active (automatic)
- [ ] Auto-scaling configured
- [ ] Logs monitoring setup
- [ ] Secrets configured (if any)
- [ ] Custom domain configured (optional)

---

## Custom Domain (Optional)

```bash
# Add custom domain
flyctl certs add yourdomain.com --config fly.toml

# Get DNS configuration
flyctl ips list --config fly.toml

# Add to your DNS:
# A record: @ -> <IPv4>
# AAAA record: @ -> <IPv6>
```

---

## Costs Estimate

**Free Tier (Default):**
- API: shared-cpu-1x (256MB) - $0/month
- UI: shared-cpu-1x (256MB) - $0/month
- Storage: 1GB volume - $0/month (within 3GB free)
- **Total: $0/month** (within free tier limits)

**Production Setup:**
- API: shared-cpu-2x (4GB) - ~$20/month
- UI: shared-cpu-1x (512MB) - ~$5/month
- Storage: 1GB - $0.15/month
- **Total: ~$25/month**

---

## Next Steps After Deployment

1. **Test the API:**
   ```bash
   curl https://ecosight-api.fly.dev/health
   curl https://ecosight-api.fly.dev/status
   ```

2. **Test the UI:**
   Open `https://ecosight-ui.fly.dev` in browser

3. **Upload test audio:**
   Use the UI or API to test predictions

4. **Monitor logs:**
   ```bash
   flyctl logs --config fly.toml -f
   ```

5. **Set up GitHub Actions** for CI/CD (optional):
   Create `.github/workflows/deploy.yml`

---

## Support

- Fly.io Docs: https://fly.io/docs
- Community: https://community.fly.io
- Status: https://status.flyio.net

---

**Ready to deploy? Run `./deploy-fly.sh` or follow manual steps!** 🚀
