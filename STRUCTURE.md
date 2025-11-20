# 📁 EcoSight Project Structure

## Organized Directory Layout

```
EcoSight/
├── 📂 src/                          # Source Code
│   ├── api.py                       # FastAPI backend server
│   └── app.py                       # Streamlit dashboard UI
│
├── 📂 deployment/                   # Deployment Configurations
│   ├── Dockerfile                   # API container definition
│   ├── Dockerfile.streamlit         # UI container definition
│   ├── docker-compose.yml           # Multi-container orchestration
│   ├── fly.toml                     # Fly.io API configuration
│   ├── fly.streamlit.toml           # Fly.io UI configuration
│   └── nginx.conf                   # Load balancer configuration
│
├── 📂 config/                       # Configuration Files
│   ├── requirements.txt             # Python dependencies
│   └── environment.yml              # Conda environment specification
│
├── 📂 scripts/                      # Utility Scripts
│   ├── deploy-fly.sh                # Automated Fly.io deployment
│   ├── quick-deploy-fly.sh          # Quick interactive deployment
│   └── start.sh                     # Local development startup
│
├── 📂 docs/                         # Documentation
│   ├── DEPLOYMENT.md                # General deployment guide
│   ├── FLY_DEPLOYMENT.md            # Fly.io specific deployment
│   ├── RETRAINING_EXPLAINED.md      # Retraining system documentation
│   ├── PROJECT_SUMMARY.md           # Complete project overview
│   ├── CONDA_INSTALL.md             # Conda installation guide
│   └── FILES.md                     # File descriptions
│
├── 📂 tests/                        # Testing & Utilities
│   ├── locustfile.py                # Load testing with Locust
│   └── apply_weights.py             # Model weight conversion utility
│
├── 📂 models/                       # Model Artifacts
│   ├── yamnet_classifier_v2.keras   # Trained classifier (99.30% accuracy)
│   ├── yamnet_classifier.keras      # Original model
│   ├── class_names.json             # Class label mappings
│   ├── model_metadata.json          # Training metadata
│   ├── performance_metrics.json     # Evaluation metrics
│   └── training_history.pkl         # Training history data
│
├── 📂 .github/workflows/            # CI/CD Pipeline
│   └── deploy-fly.yml               # GitHub Actions deployment workflow
│
├── 📂 augmented_audio/              # Training Data (gitignored)
│   ├── gun_shot/
│   ├── dog_bark/
│   ├── engine_idling/
│   └── clips/
│
├── 📂 extracted_audio/              # Raw Audio Data (gitignored - 1.7GB)
│   └── clips/
│
├── 📂 uploads/                      # Temporary Uploads (gitignored)
│
├── 📄 acoustic_togetherso_(1).ipynb # Training Jupyter Notebook
├── 📄 README.md                     # Main project documentation
├── 📄 STRUCTURE.md                  # This file
├── 📄 .gitignore                    # Git ignore patterns
└── 📄 .dockerignore                 # Docker ignore patterns
```

---

## 🎯 Quick Navigation

### For Development
- **Start coding:** `src/api.py` or `src/app.py`
- **Install dependencies:** `config/requirements.txt` or `config/environment.yml`
- **Run locally:** `scripts/start.sh`

### For Deployment
- **Docker:** `deployment/docker-compose.yml`
- **Fly.io:** `deployment/fly.toml` and `deployment/fly.streamlit.toml`
- **Deploy script:** `scripts/deploy-fly.sh`

### For Documentation
- **Learn deployment:** `docs/DEPLOYMENT.md` or `docs/FLY_DEPLOYMENT.md`
- **Understand retraining:** `docs/RETRAINING_EXPLAINED.md`
- **Full overview:** `docs/PROJECT_SUMMARY.md`

### For Testing
- **Load testing:** `tests/locustfile.py`
- **Model utilities:** `tests/apply_weights.py`

---

## 📊 File Sizes

| Directory | Size | Notes |
|-----------|------|-------|
| models/ | ~11 MB | Model weights and metadata |
| extracted_audio/ | 1.7 GB | Raw audio data (gitignored) |
| augmented_audio/ | Varies | Augmented training data (gitignored) |
| src/ | < 1 MB | Source code |
| deployment/ | < 1 MB | Configuration files |
| docs/ | < 1 MB | Documentation |

---

## 🚀 Common Commands

### Development
```bash
# Start API
python src/api.py

# Start UI
streamlit run src/app.py

# Run tests
locust -f tests/locustfile.py
```

### Docker
```bash
# Build and run
docker-compose -f deployment/docker-compose.yml up -d

# View logs
docker-compose -f deployment/docker-compose.yml logs -f

# Stop
docker-compose -f deployment/docker-compose.yml down
```

### Deployment
```bash
# Deploy to Fly.io
cd scripts && ./deploy-fly.sh

# View logs
flyctl logs --config deployment/fly.toml
```

---

**Last Updated:** November 18, 2025
