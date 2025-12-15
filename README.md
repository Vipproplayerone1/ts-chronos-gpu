# Time Series Forecasting with Chronos-2: Wikipedia Pageviews

A comprehensive, reproducible time series forecasting project using the Chronos-2 foundation model with proper baseline comparisons, rolling-origin backtesting, and statistical significance testing.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Educational-green.svg)](LICENSE)

---

## 🚀 **QUICK START** (3 Steps)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify setup
python test_setup.py

# 3. Run pipeline
# Windows:
run_end_to_end.bat

# Linux/Mac:
bash run_end_to_end.sh
```

**Runtime**: 30-60 min (GPU) | **Results**: `artifacts/` | **GPU Recommended**

---

## 📋 **Table of Contents**
- [Project Overview](#-project-overview)
- [Quick Start](#-quick-start-3-steps)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Pipeline Details](#-pipeline-details)
- [Configuration](#-configuration)
- [Results](#-results)
- [Troubleshooting](#-troubleshooting)
- [Documentation](#-documentation)
- [Contributing](#-contributing)

---

## 📊 **Project Overview**

### **Goal**
Evaluate Chronos-2 foundation model on Wikipedia pageview forecasting with rigorous scientific methodology.

### **Key Specifications**
- **Foundation Model**: Chronos-2 (T5-Base, zero-shot)
- **Dataset**: Wikipedia Pageviews (Bitcoin page, 2020-2024)
- **Domain**: Information/Media
- **Frequency**: Daily
- **Forecast Horizon**: H = 30 days
- **Seasonal Period**: m = 7 (weekly)
- **Evaluation**: 5-fold rolling-origin backtesting

### **Models Compared**
1. **Seasonal Naive** (m=7) - Simple baseline
2. **ETS** (Exponential Smoothing) - Statistical model
3. **Gradient Boosting** (LightGBM) - ML with lag features
4. **Chronos-2** (T5-Base) - Foundation model

### **Key Features**
- ✅ Zero data leakage (proper time series splits)
- ✅ Rolling-origin backtesting (expanding window)
- ✅ Statistical significance testing (Wilcoxon)
- ✅ Probabilistic forecasting (quantile predictions)
- ✅ Comprehensive metrics (MAE, RMSE, sMAPE, MASE, pinball, coverage)
- ✅ Publication-quality plots (300 DPI)
- ✅ Full reproducibility (fixed seeds, version tracking)

---

## 📦 **Installation**

### **Prerequisites**
- Python 3.10 or higher
- CUDA-capable GPU (recommended, not required)
- 8GB RAM minimum (16GB recommended)
- 50GB free disk space

### **Option 1: Quick Install (Recommended)**

**Windows**:
```bash
pip install -r requirements.txt
```

**Linux/Mac with Conda**:
```bash
conda env create -f environment.yml
conda activate ts-chronos-gpu
```

### **Option 2: Manual Setup**

<details>
<summary><b>Click to expand detailed installation steps</b></summary>

#### **Step 1: Create Environment**
```bash
# Using Conda (recommended for GPU)
conda create -n ts-chronos-gpu python=3.10 -y
conda activate ts-chronos-gpu

# Or using venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

#### **Step 2: Install PyTorch with CUDA**
```bash
# For CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# For CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# For CPU only
pip install torch torchvision torchaudio
```

#### **Step 3: Install Chronos**
```bash
# Option A: Chronos-2 (recommended)
pip install chronos-forecasting

# Option B: AutoGluon (includes Chronos)
pip install autogluon
```

#### **Step 4: Install Other Packages**
```bash
pip install pandas numpy scipy matplotlib seaborn plotly statsmodels \
            scikit-learn lightgbm xgboost tqdm requests pyyaml jupyter
```

</details>

### **Verify Installation**
```bash
python test_setup.py
```

✅ Expected output: `[SUCCESS] ALL TESTS PASSED - Ready to run pipeline!`

---

## 🎯 **Usage**

### **1. Run Complete Pipeline**

**Windows (Easiest)**:
```bash
# Double-click this file, or run:
run_end_to_end.bat
```

**Linux/Mac**:
```bash
bash run_end_to_end.sh
```

**Direct Python**:
```bash
python run_pipeline.py --config configs/default.yaml
```

### **2. View Results**

**Summary**:
```bash
# Windows
type artifacts\results_summary.yaml

# Linux/Mac
cat artifacts/results_summary.yaml
```

**Plots**:
```bash
# Windows
explorer artifacts\figures

# Linux/Mac
open artifacts/figures/  # Mac
xdg-open artifacts/figures/  # Linux
```

**Interactive Analysis**:
```bash
jupyter notebook notebooks/
# Open: 03_test_eval.ipynb
```

### **3. Customize Configuration**

Edit `configs/default.yaml`:

```yaml
# Try different Wikipedia pages
data:
  page_title: "Taylor_Swift"  # or "Python_(programming_language)"

# Adjust forecast horizon
ts_params:
  horizon: 14  # days

# Use CPU instead of GPU
models:
  chronos:
    device: "cpu"
    model_name: "amazon/chronos-t5-small"  # smaller model
```

---

## 📁 **Project Structure**

```
final/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment
├── test_setup.py               # Setup verification script
├── run_pipeline.py             # Main execution script
├── run_end_to_end.bat          # Windows batch file
├── run_end_to_end.sh           # Linux/Mac shell script
│
├── configs/
│   └── default.yaml            # Configuration file
│
├── src/                        # Source code (3,500+ lines)
│   ├── config.py               # Configuration management
│   ├── data_loader.py          # Wikipedia API loader (with caching)
│   ├── preprocess.py           # Data cleaning & outlier handling
│   ├── features.py             # Feature engineering (lags, rolling)
│   ├── metrics.py              # All evaluation metrics
│   ├── backtesting.py          # Rolling-origin framework
│   ├── baselines.py            # Baseline models
│   ├── chronos_model.py        # Chronos-2 wrapper
│   ├── stats_tests.py          # Statistical tests
│   ├── plots.py                # Visualization functions
│   └── utils.py                # Utility functions
│
├── notebooks/                   # Jupyter notebooks
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_backtesting.ipynb    # Backtesting analysis
│   └── 03_test_eval.ipynb      # Test evaluation
│
├── data/                        # Cached data (created by pipeline)
│   ├── Bitcoin_2020-01-01_2024-12-31.json
│   ├── train.parquet
│   ├── val.parquet
│   └── test.parquet
│
├── artifacts/                   # Results (created by pipeline)
│   ├── predictions/            # Model predictions (.parquet)
│   ├── metrics/                # Evaluation metrics (.json)
│   ├── figures/                # Plots (.png, 300 DPI)
│   └── results_summary.yaml    # Complete results
│
└── docs/                        # Documentation
    ├── README.md               # Documentation guide
    ├── model_card.md           # Chronos-2 model card
    ├── REPORT_TEMPLATE.md      # Report template
    └── SLIDES_TEMPLATE.md      # Slides template
```

---

## 🔄 **Pipeline Details**

The pipeline executes **9 steps** automatically:

```
[1/9] Loading configuration and setting seeds...
[2/9] Downloading Wikipedia pageviews (with caching)...
[3/9] Preprocessing (missing values, outliers)...
[4/9] Creating train/val/test splits (60/20/20)...
[5/9] Running 5-fold rolling-origin backtesting...
      ├── Training Seasonal Naive...
      ├── Training ETS...
      ├── Training Gradient Boosting...
      └── Running Chronos-2 inference...
[6/9] Computing metrics (MAE, RMSE, sMAPE, MASE, pinball, coverage)...
[7/9] Performing statistical significance tests (Wilcoxon)...
[8/9] Evaluating on final test set...
[9/9] Generating plots (6+ publication-quality figures)...

✓ Results saved to artifacts/
```

**Expected Runtime**:
- With GPU: 30-60 minutes
- With CPU: 2-3 hours

---

## ⚙️ **Configuration**

Key settings in `configs/default.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `random_seed` | 42 | For reproducibility |
| `data.page_title` | "Bitcoin" | Wikipedia page |
| `ts_params.horizon` | 30 | Forecast days |
| `ts_params.seasonal_period` | 7 | Weekly seasonality |
| `backtesting.n_folds` | 5 | CV folds |
| `models.chronos.device` | "cuda" | GPU/CPU |
| `models.chronos.model_name` | "amazon/chronos-t5-base" | Model size |

**Common Customizations**:

```yaml
# Use smaller model for faster testing
models:
  chronos:
    model_name: "amazon/chronos-t5-small"
    device: "cpu"

# Use less data
data:
  end_date: "2023-12-31"

# Fewer backtesting folds
backtesting:
  n_folds: 3
```

---

## 📈 **Results**

Results are saved to `artifacts/` after pipeline completion:

### **Metrics** (`artifacts/metrics/`)
- `*_metrics.json` - Per-model metrics
- `test_metrics.yaml` - Final test performance
- `statistical_tests.csv` - Significance tests
- `results_summary.yaml` - Complete summary

### **Predictions** (`artifacts/predictions/`)
- `*_backtest.parquet` - Validation predictions
- `*_backtest_metadata.json` - Fold information

### **Figures** (`artifacts/figures/`, 300 DPI)
- `train_val_test_split.png` - Data splits
- `test_forecasts.png` - Model comparisons
- `seasonality_decomposition.png` - STL decomposition
- `calibration_curve.png` - Interval calibration
- And more...

---

## 🔧 **Troubleshooting**

### **Package not found**
```bash
# Reinstall
pip install -r requirements.txt --force-reinstall
```

### **CUDA not available**
```bash
# Check GPU
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### **Out of memory**
Edit `configs/default.yaml`:
```yaml
models:
  chronos:
    device: "cpu"  # Use CPU
    model_name: "amazon/chronos-t5-small"  # Smaller model
```

### **Pipeline hangs**
Common causes:
1. Internet connection (for data download)
2. Disk space (need ~1GB)
3. Memory (need 8GB+ RAM)

**Quick fix**: Use smaller dataset
```yaml
data:
  end_date: "2023-12-31"
```

### **Import errors**
```bash
# Check you're in project directory
cd D:\Major\Apply_Forcasting\final

# Verify Python finds modules
python -c "import sys; print('\n'.join(sys.path))"
```

---

## ❓ **FAQ**

<details>
<summary><b>How long does it take?</b></summary>
30-60 minutes with GPU, 2-3 hours with CPU
</details>

<details>
<summary><b>Can I use a different Wikipedia page?</b></summary>
Yes! Edit <code>configs/default.yaml</code> → <code>data.page_title: "Your_Page"</code>
</details>

<details>
<summary><b>Where are results saved?</b></summary>
All in <code>artifacts/</code> folder (predictions, metrics, figures)
</details>

<details>
<summary><b>Can I run without GPU?</b></summary>
Yes, set <code>device: "cpu"</code> in <code>configs/default.yaml</code>
</details>

<details>
<summary><b>How do I reproduce results?</b></summary>
Fixed seed (42), cached data, version tracking ensure reproducibility
</details>

---

## 📚 **Documentation**

### **Notebooks**
- `notebooks/01_eda.ipynb` - Exploratory analysis, seasonality
- `notebooks/02_backtesting.ipynb` - Validation results analysis
- `notebooks/03_test_eval.ipynb` - Final test evaluation

### **Technical Docs**
- `docs/model_card.md` - Chronos-2 model card
- `docs/REPORT_TEMPLATE.md` - Academic report template
- `docs/SLIDES_TEMPLATE.md` - Presentation template

### **Original Spec**
- `PROJECT_SPEC.md` - Complete project requirements

---

## 🔬 **Reproducibility**

This project ensures full reproducibility:

- ✅ **Fixed Random Seed**: 42
- ✅ **Version Pinning**: All packages in `requirements.txt`
- ✅ **Data Caching**: Raw API responses saved with timestamps
- ✅ **Model Checkpoints**: Exact Hugging Face identifiers
- ✅ **Results Tracking**: All predictions and metrics saved

**To reproduce**:
```bash
git clone https://github.com/Vipproplayerone1/ts-chronos-gpu.git
cd ts-chronos-gpu
pip install -r requirements.txt
python run_pipeline.py
```

---

## 🤝 **Contributing**

This is an academic project. For issues or questions:
- Open an issue on GitHub
- Check documentation in `docs/`
- Review `PROJECT_SPEC.md` for requirements

---

## 📄 **License**

Educational use. Data from Wikipedia (CC BY-SA 3.0).

---

## 🙏 **Acknowledgments**

- **Chronos-2**: Amazon Science ([GitHub](https://github.com/amazon-science/chronos-forecasting))
- **Data**: Wikimedia REST API
- **Libraries**: PyTorch, statsmodels, scikit-learn, LightGBM

---

## 📚 **References**

1. Ansari et al. (2024). "Chronos: Learning the Language of Time Series." arXiv:2403.07815
2. Hyndman & Athanasopoulos (2021). "Forecasting: Principles and Practice" (3rd ed.)
3. Wikipedia Pageviews API Documentation

---

## 📞 **Support**

**Having issues?**
1. Run `python test_setup.py` for diagnostics
2. Check [Troubleshooting](#-troubleshooting) section
3. Review error messages
4. Open GitHub issue with details

---

**Status**: ✅ Production Ready | **Last Updated**: December 2024

🤖 *Built with [Claude Code](https://claude.com/claude-code)*

---

### **Quick Command Reference**

| Task | Command |
|------|---------|
| Install | `pip install -r requirements.txt` |
| Test | `python test_setup.py` |
| Run (Win) | `run_end_to_end.bat` |
| Run (Unix) | `bash run_end_to_end.sh` |
| Results | `cat artifacts/results_summary.yaml` |
| Plots | `open artifacts/figures/` |
| Notebook | `jupyter notebook notebooks/` |
| Config | Edit `configs/default.yaml` |
