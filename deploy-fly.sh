#!/bin/bash
# EcoSight Fly.io Deployment Script

set -e

echo "🚀 EcoSight Deployment to Fly.io"
echo "================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Check if flyctl is installed
if ! command -v flyctl &> /dev/null; then
    echo "❌ flyctl not found. Installing..."
    brew install flyctl
fi

echo -e "${GREEN}✅ flyctl installed${NC}"

# Step 2: Login check
echo -e "${YELLOW}Checking Fly.io authentication...${NC}"
if ! flyctl auth whoami &> /dev/null; then
    echo "Please login to Fly.io:"
    flyctl auth login
fi

echo -e "${GREEN}✅ Authenticated${NC}"

# Step 3: Create volume for model storage (one-time only)
echo -e "${YELLOW}Creating persistent volume for models...${NC}"
flyctl volumes create ecosight_models --region iad --size 1 || echo "Volume might already exist"

# Step 4: Deploy API
echo -e "${YELLOW}Deploying EcoSight API...${NC}"
flyctl deploy --config fly.toml --ha=false

# Get API URL
API_URL=$(flyctl status --config fly.toml | grep "Hostname" | awk '{print $2}')
echo -e "${GREEN}✅ API deployed at: https://${API_URL}${NC}"

# Step 5: Update Streamlit config with API URL
echo -e "${YELLOW}Updating Streamlit configuration...${NC}"
sed -i.bak "s|API_URL = \".*\"|API_URL = \"https://${API_URL}\"|" fly.streamlit.toml
rm fly.streamlit.toml.bak

# Step 6: Deploy Streamlit UI
echo -e "${YELLOW}Deploying EcoSight UI...${NC}"
flyctl deploy --config fly.streamlit.toml --ha=false

# Get UI URL
UI_URL=$(flyctl status --config fly.streamlit.toml | grep "Hostname" | awk '{print $2}')
echo -e "${GREEN}✅ UI deployed at: https://${UI_URL}${NC}"

echo ""
echo "================================"
echo "🎉 Deployment Complete!"
echo "================================"
echo ""
echo "📍 Your EcoSight applications:"
echo "   API:  https://${API_URL}"
echo "   Docs: https://${API_URL}/docs"
echo "   UI:   https://${UI_URL}"
echo ""
echo "📊 Monitoring:"
echo "   API logs:  flyctl logs --config fly.toml"
echo "   UI logs:   flyctl logs --config fly.streamlit.toml"
echo ""
echo "📈 Scaling (if needed):"
echo "   flyctl scale memory 4096 --config fly.toml"
echo "   flyctl scale count 2 --config fly.toml"
echo ""
echo "💰 Free tier includes:"
echo "   - Up to 3 shared-cpu-1x VMs"
echo "   - 160GB outbound data transfer"
echo "   - Automatic SSL certificates"
echo ""
