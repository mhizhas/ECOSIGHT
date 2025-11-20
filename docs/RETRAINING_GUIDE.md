# 🔄 EcoSight Model Retraining Guide

## Overview

The EcoSight system supports automated model retraining to improve accuracy as new audio data is collected. The retraining pipeline uses the same YAMNet architecture and augmented audio data used in the original training.

---

## 📋 Prerequisites

- Docker containers running (API + UI)
- New audio samples uploaded to `augmented_audio/` directory
- Minimum 100 new samples recommended for retraining

---

## 🚀 Retraining Methods

### Method 1: Via API Endpoint (Recommended)

Trigger retraining through the API:

```bash
curl -X POST http://localhost:8000/retrain \
  -H "Content-Type: application/json" \
  -d '{"trigger_reason": "Manual retraining request"}'
```

Or via the Streamlit UI:
1. Navigate to the **Training** page
2. Click **"Trigger Retraining"**
3. Monitor progress in the logs

### Method 2: Run Script Directly

Run the retraining script manually:

```bash
# Make sure you're in the project directory
cd /Users/cococe/Desktop/EcoSight

# Activate conda environment
conda activate ecosight

# Run retraining script
python scripts/retrain_model.py
```

### Method 3: Inside Docker Container

Execute retraining inside the API container:

```bash
export DOCKER_HOST=unix://$HOME/.docker/run/docker.sock
docker compose -f deployment/docker-compose.yml exec api python ../scripts/retrain_model.py
```

---

## 📁 Adding New Training Data

### Step 1: Upload Audio Files

Place new audio files in the appropriate class directory:

```bash
augmented_audio/
├── gun_shot/          # Add gun shot audio here
├── dog_bark/          # Add dog bark audio here
├── engine_idling/     # Add engine audio here
└── clips/             # Add other wildlife sounds here
```

### Step 2: Verify Data

Check that files are recognized:

```bash
# Count files per class
find augmented_audio/ -name "*.wav" | wc -l

# Per class breakdown
for dir in augmented_audio/*/; do 
    echo "$(basename $dir): $(ls $dir/*.wav 2>/dev/null | wc -l) files"
done
```

### Step 3: Trigger Retraining

Use any of the methods above to start retraining.

---

## 📊 Retraining Process

The retraining pipeline performs the following steps:

1. **Data Collection**: Scans `augmented_audio/` for all `.wav` files
2. **Feature Extraction**: Extracts YAMNet embeddings from each audio file
3. **Data Split**: 70% training, 15% validation, 15% test
4. **Model Training**: Trains classifier with early stopping and learning rate reduction
5. **Evaluation**: Tests model on held-out test set
6. **Model Saving**: Saves best model to `models/yamnet_classifier_v2.keras`
7. **Metrics Logging**: Updates `retraining_log.json` with performance metrics

---

## 📈 Monitoring Retraining

### Check Retraining Status

```bash
# View retraining logs
export DOCKER_HOST=unix://$HOME/.docker/run/docker.sock
docker compose -f deployment/docker-compose.yml logs api -f
```

### View Retraining History

```bash
# Check retraining log
cat models/retraining_log.json | python -m json.tool
```

### Check Model Performance

```bash
# View latest performance metrics
cat models/performance_metrics.json | python -m json.tool
```

---

## 🎯 Retraining Triggers

Retraining is automatically triggered when:

- **Manual Trigger**: User initiates retraining via API or UI
- **Data Threshold**: 100+ new audio samples added (configurable)
- **Performance Drop**: Model accuracy falls below threshold (future feature)
- **Scheduled**: Weekly/monthly automated retraining (future feature)

---

## 📝 Configuration

Edit retraining parameters in `scripts/retrain_model.py`:

```python
SAMPLE_RATE = 16000        # Audio sample rate (YAMNet uses 16kHz)
EPOCHS = 100               # Maximum training epochs
BATCH_SIZE = 64            # Training batch size
MIN_NEW_SAMPLES = 100      # Minimum new samples to trigger retraining
```

---

## 🔧 Troubleshooting

### Issue: "Not enough new samples for retraining"

**Solution**: Add more audio files to `augmented_audio/` directories. Need at least 100 new samples.

### Issue: "Retraining already in progress"

**Solution**: Wait for current retraining to complete. Check logs:
```bash
docker compose -f deployment/docker-compose.yml logs api --tail=50
```

### Issue: "Memory error during training"

**Solution**: Reduce batch size in `scripts/retrain_model.py`:
```python
BATCH_SIZE = 32  # Lower from 64
```

### Issue: "Model not loading after retraining"

**Solution**: Check model file exists and restart API:
```bash
ls -lh models/yamnet_classifier_v2.keras
docker compose -f deployment/docker-compose.yml restart api
```

---

## 📂 Generated Files

After retraining, the following files are created/updated:

- `models/yamnet_classifier_v2.keras` - Retrained model
- `models/model_metadata.json` - Model information (date, accuracy, classes)
- `models/performance_metrics.json` - Detailed performance metrics
- `models/training_history.pkl` - Training history (loss/accuracy per epoch)
- `models/retraining_log.json` - Complete retraining history
- `models/class_names.json` - Class label mappings

---

## 🎓 Best Practices

1. **Collect Diverse Data**: Include variations in recording quality, background noise, distance
2. **Balance Classes**: Aim for similar number of samples per class
3. **Validate Results**: Check confusion matrix for misclassifications
4. **Monitor Recall**: For gun shots, prioritize high recall (don't miss detections)
5. **Regular Retraining**: Retrain weekly or monthly as new data arrives
6. **Keep History**: Don't delete `retraining_log.json` - it tracks model improvements

---

## 📞 Support

For issues or questions:
- Check logs: `docker compose logs api`
- Review retraining log: `cat models/retraining_log.json`
- Verify data: `find augmented_audio/ -name "*.wav"`

---

**Last Updated**: November 19, 2025  
**Version**: 1.0.0
