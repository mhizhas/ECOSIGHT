#!/bin/bash
# Quick deployment to Fly.io

echo "🚀 EcoSight Quick Deploy to Fly.io"
echo "===================================="
echo ""

# Install flyctl
if ! command -v flyctl &> /dev/null; then
    echo "📦 Installing Fly.io CLI..."
    brew install flyctl
fi

# Login
echo "🔐 Please login to Fly.io..."
flyctl auth login

echo ""
echo "📋 Choose deployment option:"
echo "  1) Deploy API only"
echo "  2) Deploy UI only"
echo "  3) Deploy both (recommended)"
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo "🚀 Deploying API..."
        flyctl launch --config fly.toml --name ecosight-api
        ;;
    2)
        echo "🚀 Deploying UI..."
        flyctl launch --config fly.streamlit.toml --name ecosight-ui
        ;;
    3)
        echo "🚀 Deploying API..."
        flyctl launch --config fly.toml --name ecosight-api
        
        echo ""
        echo "🚀 Deploying UI..."
        flyctl launch --config fly.streamlit.toml --name ecosight-ui
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "✅ Deployment initiated!"
echo ""
echo "📖 Next steps:"
echo "  1. Upload model files via SFTP"
echo "  2. Check deployment: flyctl status"
echo "  3. View logs: flyctl logs"
echo ""
echo "📚 Full guide: See FLY_DEPLOYMENT.md"
