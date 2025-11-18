# Conda Installation Guide for EcoSight

## Option 1: Using environment.yml (Recommended)

```bash
# Create conda environment from file
conda env create -f environment.yml

# Activate the environment
conda activate ecosight

# Verify installation
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import streamlit; print('Streamlit installed')"
python -c "import fastapi; print('FastAPI installed')"
```

## Option 2: Manual Installation

```bash
# Create new conda environment
conda create -n ecosight python=3.10 -y

# Activate environment
conda activate ecosight

# Install conda packages first
conda install -c conda-forge numpy=1.24.3 pandas=2.1.3 scikit-learn=1.3.2 ffmpeg libsndfile -y

# Install pip packages
pip install tensorflow==2.15.0 tensorflow-hub==0.15.0
pip install librosa==0.10.1 soundfile==0.12.1
pip install fastapi==0.104.1 uvicorn[standard]==0.24.0
pip install python-multipart==0.0.6 pydantic==2.5.0
pip install streamlit==1.28.2 plotly==5.18.0
pip install resampy==0.4.2 locust==2.18.0
pip install python-dotenv==1.0.0 aiofiles==23.2.1
```

## Option 3: Install Everything via Conda (if pip issues)

```bash
# Create environment
conda create -n ecosight python=3.10 -y
conda activate ecosight

# Install from conda-forge
conda install -c conda-forge numpy pandas scikit-learn ffmpeg libsndfile -y
conda install -c conda-forge tensorflow -y
conda install -c conda-forge librosa -y
conda install -c conda-forge fastapi uvicorn -y
conda install -c conda-forge streamlit plotly -y
conda install -c conda-forge locust -y

# Install remaining via pip (if needed)
pip install tensorflow-hub python-multipart pydantic resampy python-dotenv aiofiles
```

## Troubleshooting Common Errors

### Error: "Solving environment: failed"

**Solution 1:** Update conda first
```bash
conda update -n base conda
conda update --all
```

**Solution 2:** Use mamba (faster solver)
```bash
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

# Or install mamba
conda install -c conda-forge mamba
mamba env create -f environment.yml
```

### Error: Package conflicts

**Solution:** Create fresh environment
```bash
# Remove old environment
conda env remove -n ecosight

# Create new one
conda env create -f environment.yml
```

### Error: TensorFlow installation fails

**Solution:** Install specific build
```bash
# For macOS (M1/M2)
conda install -c apple tensorflow-deps
pip install tensorflow-macos==2.15.0
pip install tensorflow-metal  # For GPU acceleration

# For Linux/Windows
pip install tensorflow==2.15.0
```

### Error: librosa/soundfile issues

**Solution:** Install system dependencies first
```bash
# macOS
brew install libsndfile ffmpeg

# Linux (Ubuntu/Debian)
sudo apt-get install libsndfile1 ffmpeg

# Then install Python packages
conda install -c conda-forge librosa soundfile
```

## Verify Installation

```bash
# Activate environment
conda activate ecosight

# Run verification script
python << EOF
import sys
print(f"Python: {sys.version}")

try:
    import tensorflow as tf
    print(f"✓ TensorFlow: {tf.__version__}")
except ImportError as e:
    print(f"✗ TensorFlow: {e}")

try:
    import tensorflow_hub as hub
    print(f"✓ TensorFlow Hub installed")
except ImportError as e:
    print(f"✗ TensorFlow Hub: {e}")

try:
    import librosa
    print(f"✓ Librosa: {librosa.__version__}")
except ImportError as e:
    print(f"✗ Librosa: {e}")

try:
    import fastapi
    print(f"✓ FastAPI: {fastapi.__version__}")
except ImportError as e:
    print(f"✗ FastAPI: {e}")

try:
    import streamlit
    print(f"✓ Streamlit: {streamlit.__version__}")
except ImportError as e:
    print(f"✗ Streamlit: {e}")

try:
    import locust
    print(f"✓ Locust installed")
except ImportError as e:
    print(f"✗ Locust: {e}")

print("\nAll core packages verified!")
EOF
```

## Quick Start After Installation

```bash
# Activate environment
conda activate ecosight

# Run API
python api.py

# Or run Streamlit UI (in another terminal)
conda activate ecosight
streamlit run app.py
```

## Managing the Environment

```bash
# List all conda environments
conda env list

# Export environment
conda env export > environment_backup.yml

# Update environment
conda env update -f environment.yml --prune

# Remove environment
conda env remove -n ecosight
```
