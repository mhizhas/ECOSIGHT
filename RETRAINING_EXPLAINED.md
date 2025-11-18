# Model Retraining System Explained

## Overview
Your EcoSight system has an automated retraining pipeline that allows the model to be updated when new labeled audio data is added. Here's how it all works:

---

## 🔄 The Complete Retraining Flow

### **1. Data Upload (Adding New Training Data)**

**Via API (`/upload` endpoint):**
```python
# User uploads a new audio file
POST /upload
{
  "file": audio_file.wav,
  "class_name": "dog_bark"  # or gun_shot, engine_idling, clips
}
```

**What happens:**
- File is saved to: `augmented_audio/{class_name}/{timestamp}_filename.wav`
- Class directories are created automatically if they don't exist
- Each file is timestamped to avoid name conflicts

**Via Streamlit UI:**
- Navigate to "Training" tab
- Upload audio file
- Select the correct class label
- Click "Upload"

---

### **2. Checking Retraining Trigger**

The `ModelRetrainingPipeline.check_retraining_trigger()` method monitors new data:

```python
# Counts audio files in each class directory
current_samples = {
    "dog_bark": 1200,
    "gun_shot": 950,
    "engine_idling": 1100,
    "clips": 800
}

# Compares with last training session
last_samples = {  # from retraining_log.json
    "dog_bark": 1000,
    "gun_shot": 850,
    "engine_idling": 1000,
    "clips": 700
}

# Calculates new samples
new_samples = (1200-1000) + (950-850) + (1100-1000) + (800-700)
            = 200 + 100 + 100 + 100 = 500 new samples

# Trigger decision
if new_samples >= min_new_samples (default 100):
    ✓ Trigger retraining!
else:
    ⏸️  Wait for more data
```

**Trigger Logic:**
- **Automatic threshold**: 100 new samples (configurable)
- **Manual trigger**: User can force retraining via API/UI
- **First training**: Always triggers if no previous training exists

---

### **3. Triggering Retraining**

**Method 1: Automatic (Threshold-based)**
```python
# In notebook or scheduled script
should_retrain, new_count = retraining_pipeline.check_retraining_trigger(min_new_samples=100)

if should_retrain:
    print(f"Retraining with {new_count} new samples!")
    model, metrics = retraining_pipeline.retrain_model(yamnet_model, epochs=100)
```

**Method 2: Manual via API**
```python
POST /retrain
{
  "trigger_reason": "Monthly scheduled retraining",
  "epochs": 100,
  "min_accuracy": 0.95
}
```

**Method 3: Manual via Streamlit UI**
- Go to "Training" tab
- Click "Trigger Retraining" button
- Specify reason and parameters
- Monitor progress

---

### **4. The Retraining Pipeline** 

When retraining is triggered, `retrain_model()` executes these steps:

#### **Step 1: Feature Extraction**
```python
# Scans all audio files in augmented_audio/
for each class_dir in [dog_bark, gun_shot, engine_idling, clips]:
    for each audio_file in class_dir:
        1. Load audio at 16kHz (4 seconds max)
        2. Pass through YAMNet pretrained model
        3. Extract 1024-dimensional embedding
        4. Average embeddings across time
        5. Store feature vector + class label
```

**Output:**
- `X_features`: Shape (N, 1024) - N audio samples, 1024 features each
- `y_labels`: Shape (N,) - Class indices [0, 1, 2, 3]
- `class_names`: ['clips', 'dog_bark', 'engine_idling', 'gun_shot']

#### **Step 2: Data Preprocessing**
```python
# Normalize features (zero mean, unit variance)
X_normalized = (X - X.mean()) / X.std()

# Convert labels to one-hot encoding
y_categorical = [[1,0,0,0], [0,1,0,0], ...] # for 4 classes
```

#### **Step 3: Train/Val/Test Split**
```python
# Split data into three sets
Total: 100% of data
├── Test: 15% (unseen evaluation data)
└── Temp: 85%
    ├── Validation: 15% of temp (≈13% of total)
    └── Training: 85% of temp (≈72% of total)
```

**Example with 10,000 samples:**
- Training: ~7,200 samples (72%)
- Validation: ~1,300 samples (13%)  
- Test: ~1,500 samples (15%)

#### **Step 4: Model Architecture**
```python
model = Sequential([
    # Layer 1: 1024 inputs → 512 neurons
    Dense(512, activation='relu', input_shape=(1024,)),
    BatchNormalization(),  # Stabilize training
    Dropout(0.5),          # Prevent overfitting
    
    # Layer 2: 512 → 256 neurons
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    
    # Layer 3: 256 → 128 neurons
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    # Layer 4: 128 → 64 neurons
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    # Output: 64 → 4 classes (softmax probabilities)
    Dense(4, activation='softmax')
])
```

**Total Parameters:** ~697,540 trainable weights

#### **Step 5: Training**
```python
# Training with callbacks
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,           # Maximum epochs
    batch_size=64,        # Process 64 samples at a time
    callbacks=[
        EarlyStopping(patience=15),      # Stop if no improvement for 15 epochs
        ReduceLROnPlateau(patience=7),   # Reduce learning rate if plateau
        ModelCheckpoint(save_best_only=True)  # Save best model
    ]
)
```

**Training Process:**
```
Epoch 1/100: loss: 1.2345, acc: 0.5678, val_loss: 1.1234, val_acc: 0.6543
Epoch 2/100: loss: 0.9876, acc: 0.7123, val_loss: 0.8765, val_acc: 0.7654
...
Epoch 42/100: val_loss improved → Save model checkpoint
...
Epoch 65/100: No improvement for 15 epochs → Early stop
✓ Training complete! Best epoch: 50
```

#### **Step 6: Evaluation**
```python
# Test on unseen data
test_loss, test_accuracy = model.evaluate(X_test, y_test)

# Calculate per-class metrics
precision = [0.98, 0.99, 0.97, 0.98]  # For each class
recall    = [0.97, 0.99, 0.98, 0.99]
f1_score  = [0.975, 0.99, 0.975, 0.985]
```

#### **Step 7: Save Everything**
```python
# 1. Save model
model.save("models/retrained_model_20251118_143045.keras")

# 2. Update retraining log
retraining_log.json:
{
  "retraining_history": [
    {
      "timestamp": "20251118_143045",
      "datetime": "2025-11-18 14:30:45",
      "test_accuracy": 0.9930,
      "num_classes": 4,
      "sample_counts": {
        "dog_bark": 1200,
        "gun_shot": 950,
        ...
      },
      "total_samples": 4050,
      "epochs_trained": 65
    }
  ]
}
```

---

## 📊 Real-World Example

Let's say you've been running the system for a month:

### **Day 1 - Initial State**
```
augmented_audio/
├── dog_bark/      1000 files
├── gun_shot/      850 files
├── engine_idling/ 1000 files
└── clips/         700 files
Total: 3,550 files
```

### **Week 1 - Users upload data**
```
# New uploads via API/UI
POST /upload {file: dog1.wav, class: dog_bark}
POST /upload {file: dog2.wav, class: dog_bark}
POST /upload {file: gun1.wav, class: gun_shot}
...

# Check trigger
should_retrain = check_retraining_trigger(min_new_samples=100)
→ False (only 45 new files)
```

### **Week 2 - More uploads**
```
# Total new files: 120
should_retrain = check_retraining_trigger(min_new_samples=100)
→ True! (120 >= 100)

# Automatic retraining triggered
retrain_model(yamnet_model, epochs=100)
```

### **Retraining Process**
```
1. Extract features from 3,670 audio files
   ⏱️  Time: ~30-45 minutes (depends on CPU/GPU)

2. Split data:
   - Training: 2,642 samples
   - Validation: 477 samples  
   - Test: 551 samples

3. Train model:
   Epoch 1: loss: 1.234, val_acc: 0.654
   Epoch 10: loss: 0.567, val_acc: 0.891
   Epoch 25: loss: 0.123, val_acc: 0.967
   Epoch 50: loss: 0.045, val_acc: 0.993 ← Best!
   Early stopping at epoch 65

4. Evaluate:
   Test Accuracy: 99.35%
   Test Loss: 0.025

5. Save:
   ✓ Model: retrained_model_20251118.keras
   ✓ Log updated with new metrics
```

### **Result**
```
Model improved from 99.30% → 99.35% accuracy
New data incorporated: 120 files
Ready for production deployment
```

---

## 🎯 Key Benefits

### **1. Continuous Learning**
- Model improves as more data is collected
- Adapts to new sound variations
- No manual intervention needed (if automated)

### **2. Quality Control**
- Tracks all retraining sessions in `retraining_log.json`
- Monitors accuracy trends over time
- Can rollback to previous models if needed

### **3. Scalability**
- Can run in background without affecting API
- Handles thousands of samples efficiently
- Uses batch processing for large datasets

### **4. Flexibility**
- Manual triggers for urgent updates
- Automatic triggers based on data volume
- Configurable thresholds and parameters

---

## 🚀 How to Use in Production

### **Option 1: Scheduled Retraining**
```python
# Cron job or scheduler (runs weekly)
import schedule

def weekly_retraining():
    should_retrain, count = pipeline.check_retraining_trigger(min_new_samples=50)
    if should_retrain:
        model, metrics = pipeline.retrain_model(yamnet_model)
        # Deploy new model
        
schedule.every().sunday.at("02:00").do(weekly_retraining)
```

### **Option 2: Threshold-Based**
```python
# After each upload, check threshold
@app.post("/upload")
def upload_and_check():
    # Save file
    save_file(...)
    
    # Check if we should retrain
    should_retrain, count = pipeline.check_retraining_trigger()
    if should_retrain:
        # Trigger background retraining
        background_tasks.add_task(retrain_async)
```

### **Option 3: Manual Admin Control**
```python
# Admin dashboard button
if admin_clicks_retrain_button:
    POST /retrain {
        "trigger_reason": "Admin requested retraining",
        "epochs": 100
    }
```

---

## 📁 File Locations

```
EcoSight/
├── augmented_audio/          # Training data
│   ├── dog_bark/*.wav        # New files uploaded here
│   ├── gun_shot/*.wav
│   ├── engine_idling/*.wav
│   └── clips/*.wav
│
├── models/
│   ├── yamnet_classifier_v2.keras      # Current production model
│   ├── retrained_model_*.keras         # New retrained models
│   ├── retraining_log.json            # Complete retraining history
│   ├── class_names.json
│   └── model_metadata.json
│
└── api.py                    # /upload and /retrain endpoints
```

---

## ⚡ Quick Start Commands

```bash
# Check if retraining should be triggered
python -c "
from pathlib import Path
# Run trigger check
should_retrain, count = retraining_pipeline.check_retraining_trigger(100)
print(f'Retrain: {should_retrain}, New samples: {count}')
"

# Manually trigger retraining
curl -X POST http://localhost:8000/retrain \
  -H "Content-Type: application/json" \
  -d '{"trigger_reason": "Manual test", "epochs": 50}'

# Upload new training file
curl -X POST http://localhost:8000/upload \
  -F "file=@new_dog_bark.wav" \
  -F "class_name=dog_bark"
```

---

## 🔍 Monitoring Retraining

Check the retraining log to see history:

```python
import json

with open('models/retraining_log.json', 'r') as f:
    log = json.load(f)

for session in log['retraining_history']:
    print(f"Date: {session['datetime']}")
    print(f"Accuracy: {session['test_accuracy']:.4f}")
    print(f"Samples: {session['total_samples']}")
    print(f"Epochs: {session['epochs_trained']}")
    print("-" * 40)
```

---

This system gives you full control over model updates while maintaining production stability!
