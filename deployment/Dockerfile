# Dockerfile for EcoSight Wildlife Monitoring API
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY config/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy API code
COPY src/api.py .
COPY scripts/retrain_model.py .
COPY models/ ./models/

# Create necessary directories
RUN mkdir -p uploads augmented_audio

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/status')" || exit 1

# Run the API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
