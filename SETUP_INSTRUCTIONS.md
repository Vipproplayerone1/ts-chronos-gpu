# Package Installation Guide

This guide provides step-by-step instructions for setting up all required packages for the Wikipedia Pageviews Forecasting project.

## Prerequisites

- Python 3.10 or higher
- Conda or Miniconda installed
- CUDA-capable GPU (for GPU acceleration)

## Step 1: Create Conda Environment

```bash
# Create a new conda environment with Python 3.10
conda create -n ts-chronos-gpu python=3.10 -y

# Activate the environment
conda activate ts-chronos-gpu
```

## Step 2: Install PyTorch with CUDA Support

```bash
# For CUDA 11.8 (adjust version based on your GPU driver)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Alternative: For CUDA 12.1
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Step 3: Install Chronos Library

**Option A: Chronos-2 (Recommended - Latest Model)**
```bash
# Install Chronos-2 forecasting library
pip install chronos-forecasting

# For development version from GitHub
# pip install git+https://github.com/amazon-science/chronos-forecasting.git
```

**Option B: AutoGluon (Includes Chronos models)**
```bash
# Install full AutoGluon package (includes Chronos)
pip install autogluon

# This automatically installs with PyTorch GPU support if CUDA is available
```

## Step 4: Install Data Science & ML Packages

```bash
# Core data science packages
pip install pandas numpy scipy

# Visualization
pip install matplotlib seaborn plotly

# Statistical modeling
pip install statsmodels

# Machine learning
pip install scikit-learn lightgbm xgboost

# Utilities
pip install tqdm requests

# Jupyter (optional, for interactive development)
pip install jupyter notebook ipykernel
```

## Step 5: Install Additional Dependencies

```bash
# For better numerical computations
pip install numba

# For time series cross-validation
pip install scikit-learn --upgrade

# For working with dates
pip install python-dateutil

# For configuration management (if needed)
pip install pyyaml
```

## Step 6: Create requirements.txt (Optional)

```bash
# Save current environment packages
pip freeze > requirements.txt
```

## Step 7: Verify Installation

**If you installed chronos-forecasting:**
```bash
python -c "
import torch
import pandas as pd
import numpy as np
from chronos import ChronosPipeline
print('All packages imported successfully!')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Chronos: Successfully imported')
"
```

**If you installed autogluon:**
```bash
python -c "
import torch
import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesPredictor
print('All packages imported successfully!')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'AutoGluon: Successfully imported')
"
```

## Alternative: Install from requirements.txt

If a `requirements.txt` file is provided, you can install all packages at once:

```bash
pip install -r requirements.txt
```

## Troubleshooting

### CUDA Not Available
- Check GPU driver compatibility: `nvidia-smi`
- Reinstall PyTorch with correct CUDA version
- Verify CUDA toolkit installation

### Import Errors
- Ensure conda environment is activated
- Try reinstalling the specific package: `pip install --upgrade <package-name>`

### Memory Issues
- Use smaller batch sizes in model training
- Close other GPU-consuming applications
- Consider using CPU version for development

## Quick Setup (All-in-One)

```bash
# Create and activate environment
conda create -n ts-chronos-gpu python=3.10 -y
conda activate ts-chronos-gpu

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install Chronos-2 and all other packages
pip install chronos-forecasting pandas numpy scipy matplotlib seaborn plotly statsmodels scikit-learn lightgbm xgboost tqdm requests jupyter

# OR install AutoGluon (which includes Chronos)
# pip install autogluon pandas numpy scipy matplotlib seaborn plotly statsmodels scikit-learn lightgbm xgboost tqdm requests jupyter

# Verify
python -c "import torch; print(f'Setup complete! CUDA: {torch.cuda.is_available()}')"
```

## Notes

- Adjust CUDA version based on your system's GPU driver
- GPU acceleration significantly speeds up Chronos model inference
- Choose between chronos-forecasting (lighter, direct) or autogluon (full suite).
- Total installation size: ~5-8 GB depending on configurations

## Additional Resources

- [Chronos-2 Documentation](https://github.com/amazon-science/chronos-forecasting)
- [AutoGluon Time Series Guide](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html)
- [Chronos-2 on Hugging Face](https://huggingface.co/amazon/chronos-2)
