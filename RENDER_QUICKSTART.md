# Quick Start - Render Deployment

## 1. Push to GitHub
```bash
cd /Users/cococe/Desktop/EcoSight
git add .
git commit -m "Add Render deployment configuration"
git push origin main
```

## 2. Deploy on Render
1. Go to https://dashboard.render.com/
2. Click **"New +"** → **"Blueprint"**
3. Connect GitHub → Select **ECOSIGHT** repository
4. Click **"Apply"**

## 3. Wait for Build (5-10 minutes)
Render will create:
- `ecosight-api` - FastAPI backend
- `ecosight-ui` - Streamlit dashboard

## 4. Access Your App
- **Dashboard**: `https://ecosight-ui.onrender.com`
- **API**: `https://ecosight-api.onrender.com/status`

## Configuration Files Created
- ✅ `render.yaml` - Service definitions
- ✅ `deployment/Dockerfile.streamlit` - Updated UI Dockerfile
- ✅ `docs/RENDER_DEPLOYMENT.md` - Full deployment guide

## Important Notes

### Model Files
If your model files are >100MB, you need Git LFS:
```bash
git lfs install
git lfs track "models/*.keras"
git add .gitattributes
git commit -m "Track models with LFS"
git push
```

### Free Tier Limitations
- Services spin down after 15 min inactivity
- No persistent storage
- Cold start takes 30-60 seconds

### Upgrade to Starter ($7/month per service)
- Always on
- Persistent storage
- No cold starts

## Next Steps
1. Push code to GitHub
2. Deploy via Render Blueprint
3. Test your deployed app
4. (Optional) Add custom domain
5. (Optional) Set up MongoDB

See `docs/RENDER_DEPLOYMENT.md` for detailed guide.
