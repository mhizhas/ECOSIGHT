#!/bin/bash

# EcoSight Quick Start Script
# This script helps you get started with the EcoSight Wildlife Monitoring system

set -e

echo "=================================================="
echo "🦜 EcoSight Wildlife Monitoring - Quick Start"
echo "=================================================="
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "Checking prerequisites..."

if ! command_exists python3; then
    echo "❌ Python 3 is not installed. Please install Python 3.10+"
    exit 1
fi
echo "✅ Python 3 found: $(python3 --version)"

if ! command_exists docker; then
    echo "⚠️  Docker is not installed. Docker deployment will not be available."
    DOCKER_AVAILABLE=false
else
    echo "✅ Docker found: $(docker --version)"
    DOCKER_AVAILABLE=true
fi

echo ""
echo "=================================================="
echo "What would you like to do?"
echo "=================================================="
echo "1. Setup local development environment"
echo "2. Run API server"
echo "3. Run Streamlit UI"
echo "4. Run both API and UI"
echo "5. Docker deployment (single container)"
echo "6. Docker deployment (scaled - 3 containers)"
echo "7. Run load tests"
echo "8. View API documentation"
echo "9. Exit"
echo ""

read -p "Enter your choice (1-9): " choice

case $choice in
    1)
        echo ""
        echo "Setting up local development environment..."
        
        # Create virtual environment
        if [ ! -d "venv" ]; then
            echo "Creating virtual environment..."
            python3 -m venv venv
        fi
        
        # Activate virtual environment
        echo "Activating virtual environment..."
        source venv/bin/activate
        
        # Install dependencies
        echo "Installing dependencies..."
        pip install -r requirements.txt
        
        # Create directories
        echo "Creating necessary directories..."
        mkdir -p models uploads augmented_audio features
        
        echo ""
        echo "✅ Setup complete!"
        echo ""
        echo "Next steps:"
        echo "1. Copy your trained model files to the models/ directory:"
        echo "   - yamnet_classifier.keras"
        echo "   - class_names.json"
        echo "   - model_metadata.json"
        echo "   - performance_metrics.json"
        echo ""
        echo "2. Run the API: ./start.sh (option 2)"
        echo "3. Run the UI: ./start.sh (option 3)"
        ;;
    
    2)
        echo ""
        echo "Starting API server..."
        
        # Activate venv if exists
        if [ -d "venv" ]; then
            source venv/bin/activate
        fi
        
        echo "API will be available at: http://localhost:8000"
        echo "API documentation: http://localhost:8000/docs"
        echo ""
        python api.py
        ;;
    
    3)
        echo ""
        echo "Starting Streamlit UI..."
        
        # Activate venv if exists
        if [ -d "venv" ]; then
            source venv/bin/activate
        fi
        
        echo "UI will be available at: http://localhost:8501"
        echo ""
        streamlit run app.py
        ;;
    
    4)
        echo ""
        echo "Starting both API and UI..."
        echo ""
        
        # Activate venv if exists
        if [ -d "venv" ]; then
            source venv/bin/activate
        fi
        
        echo "Starting API server in background..."
        python api.py &
        API_PID=$!
        
        sleep 3
        
        echo "Starting Streamlit UI..."
        echo ""
        echo "Services available at:"
        echo "  - API: http://localhost:8000"
        echo "  - UI: http://localhost:8501"
        echo "  - Docs: http://localhost:8000/docs"
        echo ""
        streamlit run app.py
        
        # Kill API when streamlit exits
        kill $API_PID
        ;;
    
    5)
        if [ "$DOCKER_AVAILABLE" = false ]; then
            echo "❌ Docker is not available. Please install Docker first."
            exit 1
        fi
        
        echo ""
        echo "Starting Docker deployment (single container)..."
        docker-compose up -d
        
        echo ""
        echo "✅ Deployment complete!"
        echo ""
        echo "Services available at:"
        echo "  - API: http://localhost:8000"
        echo "  - UI: http://localhost:8501"
        echo "  - Load Balancer: http://localhost:80"
        echo ""
        echo "View logs: docker-compose logs -f"
        echo "Stop: docker-compose down"
        ;;
    
    6)
        if [ "$DOCKER_AVAILABLE" = false ]; then
            echo "❌ Docker is not available. Please install Docker first."
            exit 1
        fi
        
        echo ""
        echo "Starting Docker deployment (scaled - 3 containers)..."
        docker-compose up -d --scale api=3
        
        echo ""
        echo "✅ Deployment complete!"
        echo ""
        echo "Services:"
        echo "  - API containers: 3"
        echo "  - Load Balancer: http://localhost:80"
        echo "  - UI: http://localhost:8501"
        echo ""
        echo "View logs: docker-compose logs -f api"
        echo "Stop: docker-compose down"
        ;;
    
    7)
        echo ""
        echo "Load Testing Options:"
        echo "1. Light load (10 users, 2 min)"
        echo "2. Medium load (100 users, 5 min)"
        echo "3. Heavy load (500 users, 10 min)"
        echo "4. Custom (Web UI)"
        echo ""
        
        read -p "Enter your choice (1-4): " load_choice
        
        # Activate venv if exists
        if [ -d "venv" ]; then
            source venv/bin/activate
        fi
        
        case $load_choice in
            1)
                echo "Running light load test..."
                locust -f locustfile.py --host=http://localhost:8000 \
                    --users=10 --spawn-rate=2 --run-time=2m --headless
                ;;
            2)
                echo "Running medium load test..."
                locust -f locustfile.py --host=http://localhost:8000 \
                    --users=100 --spawn-rate=10 --run-time=5m --headless
                ;;
            3)
                echo "Running heavy load test..."
                locust -f locustfile.py --host=http://localhost:8000 \
                    --users=500 --spawn-rate=50 --run-time=10m --headless
                ;;
            4)
                echo "Starting Locust Web UI..."
                echo "Open http://localhost:8089 in your browser"
                locust -f locustfile.py --host=http://localhost:8000
                ;;
            *)
                echo "Invalid choice"
                ;;
        esac
        ;;
    
    8)
        echo ""
        echo "Opening API documentation..."
        echo ""
        echo "Make sure the API is running (option 2 or 5)"
        echo "Then visit: http://localhost:8000/docs"
        echo ""
        
        # Try to open in default browser
        if command_exists open; then
            open http://localhost:8000/docs
        elif command_exists xdg-open; then
            xdg-open http://localhost:8000/docs
        else
            echo "Please open http://localhost:8000/docs in your browser"
        fi
        ;;
    
    9)
        echo "Goodbye! 👋"
        exit 0
        ;;
    
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=================================================="
echo "Done!"
echo "=================================================="
