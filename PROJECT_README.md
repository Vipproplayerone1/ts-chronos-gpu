# Time Series Forecasting with Chronos-2: Wikipedia Pageviews

A comprehensive, reproducible time series forecasting project using the Chronos-2 foundation model with proper baseline comparisons, rolling-origin backtesting, and statistical significance testing.

---

## 🚀 **QUICK START** (3 Commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test setup
python test_setup.py

# 3. Run pipeline (Windows)
run_end_to_end.bat

# OR (Linux/Mac)
bash run_end_to_end.sh
```

**Expected Runtime**: 30-60 minutes | **GPU Recommended** | **Results**: `artifacts/`

---

## 📋 Project Overview

- **Foundation Model**: Chronos-2 (zero-shot probabilistic forecasting)
- **Dataset**: Wikipedia Pageviews (Daily) via Wikimedia API
- **Domain**: Information/Media (Bitcoin page)
- **Forecast Horizon**: H = 30 days
- **Seasonal Period**: m = 7 (weekly seasonality)
- **Evaluation**: Rolling-origin backtesting with 5 folds

## 🎯 Key Features

- ✅ **Zero Data Leakage**: Proper time series splits and rolling-origin backtesting
- ✅ **Strong Baselines**: Seasonal Naive, ETS, Gradient Boosting with lag features
- ✅ **Probabilistic Forecasting**: Quantile predictions with calibration analysis
- ✅ **Statistical Testing**: Wilcoxon signed-rank test for model comparison
- ✅ **Comprehensive Metrics**: MAE, RMSE, sMAPE, MASE, pinball loss, coverage
- ✅ **Full Reproducibility**: Fixed seeds, cached data, version tracking
- ✅ **Publication-Quality Plots**: Error analysis, calibration curves, seasonality

## 📁 Repository Structure

```
.
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── environment.yml             # Conda environment
├── run_pipeline.py             # Main execution script
├── run_end_to_end.sh          # Bash wrapper script
│
├── configs/
│   └── default.yaml           # Configuration file
│
├── src/
│   ├── config.py              # Configuration management
│   ├── data_loader.py         # Wikipedia API data loader
│   ├── preprocess.py          # Data preprocessing
│   ├── features.py            # Feature engineering
│   ├── metrics.py             # Evaluation metrics
│   ├── backtesting.py         # Rolling-origin framework
│   ├── baselines.py           # Baseline models
│   ├── chronos_model.py       # Chronos-2 wrapper
│   ├── stats_tests.py         # Statistical significance tests
│   ├── plots.py               # Visualization functions
│   └── utils.py               # Utility functions
│
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   ├── 02_backtesting.ipynb   # Backtesting analysis
│   └── 03_test_eval.ipynb     # Test evaluation
│
├── data/                       # Cached data (gitignored)
├── artifacts/
│   ├── predictions/           # Model predictions
│   ├── metrics/               # Evaluation metrics
│   ├── figures/               # Generated plots
│   └── results_summary.yaml   # Final results
│
└── docs/
    ├── report.pdf             # Project report (≤6 pages)
    ├── slides.pdf             # Presentation (6-8 slides)
    └── model_card.md          # Model card (1 page)
```

## 📦 Installation & Setup

### **Step 1: Install Dependencies**

**Windows**:
```bash
pip install -r requirements.txt
```

**Linux/Mac (with Conda)**:
```bash
conda env create -f environment.yml
conda activate ts-chronos-gpu
```

### **Step 2: Verify Setup**
```bash
python test_setup.py
```

✅ You should see: `[SUCCESS] ALL TESTS PASSED - Ready to run pipeline!`

### **Step 3: Run the Pipeline**

**Windows**:
```bash
# Double-click or run:
run_end_to_end.bat
```

**Linux/Mac**:
```bash
bash run_end_to_end.sh
```

**Or directly**:
```bash
python run_pipeline.py
```

### **Step 4: View Results**

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
ls artifacts/figures/
```

**Interactive Analysis**:
```bash
jupyter notebook notebooks/03_test_eval.ipynb
```

## 📊 Pipeline Steps

The pipeline executes the following steps:

1. **Data Loading**: Fetches Wikipedia pageviews via Wikimedia API (with caching)
2. **Preprocessing**: Handles missing values and outliers
3. **Train/Val/Test Split**: 60%/20%/20% temporal split
4. **Rolling-Origin Backtesting**: 5-fold expanding window validation
5. **Model Training**: Seasonal Naive, ETS, Gradient Boosting, Chronos-2
6. **Metrics Computation**: All point and probabilistic metrics
7. **Statistical Testing**: Wilcoxon test for significance
8. **Test Evaluation**: Final hold-out performance
9. **Visualization**: All required plots

**Expected Runtime**: ~30-60 minutes (depends on GPU availability)

## 🎓 Models

### Baselines

1. **Seasonal Naive** (m=7)
   - Forecast = value from same weekday last week
   - Simple but strong baseline for seasonal data

2. **ETS (Exponential Smoothing)**
   - Holt-Winters with additive seasonality
   - Trend and seasonal components

3. **Gradient Boosting (LightGBM)**
   - Features: lags [1, 7, 14, 28], rolling stats [7, 28], day-of-week
   - 100 trees, depth 5, learning rate 0.05

### Foundation Model

4. **Chronos-2 (amazon/chronos-t5-base)**
   - Zero-shot probabilistic forecasting
   - 20 samples for quantile estimation
   - Quantiles: [0.1, 0.5, 0.9]

## 📈 Evaluation Metrics

### Point Forecasts
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **sMAPE**: Symmetric Mean Absolute Percentage Error (%)
- **MASE**: Mean Absolute Scaled Error (primary metric)

### Probabilistic Forecasts
- **Pinball Loss**: For each quantile (0.1, 0.5, 0.9)
- **Coverage**: Empirical vs nominal interval coverage
- **Width**: Mean prediction interval width

## 🔬 Key Results

Results will be saved to `artifacts/results_summary.yaml` after running the pipeline.

Expected findings:
- Model rankings by MASE on validation set
- Statistical significance tests (Chronos vs best baseline)
- Test set performance comparison
- Error analysis by forecast horizon
- Calibration quality of prediction intervals

## 🛠️ Configuration

Edit `configs/default.yaml` to customize:

```yaml
# Change Wikipedia page
data:
  page_title: "Bitcoin"  # Try: "Taylor_Swift", "Python_(programming_language)"

# Adjust forecast horizon
ts_params:
  horizon: 30  # days

# Modify backtesting
backtesting:
  n_folds: 5
  method: "expanding"

# Configure Chronos
models:
  chronos:
    model_name: "amazon/chronos-t5-base"  # or t5-small, t5-large
    device: "cuda"  # or "cpu"
```

## 📦 Requirements

### Minimum Requirements
- Python 3.10+
- 8GB RAM (16GB recommended)
- Internet connection (for data download and model download)

### Recommended
- NVIDIA GPU with 8GB+ VRAM (for Chronos)
- CUDA 11.8 or 12.1
- 50GB free disk space

### Key Dependencies
- `chronos-forecasting==2.0.0` (Chronos-2 model)
- `torch>=2.1.0` (PyTorch)
- `pandas`, `numpy`, `scipy` (Data processing)
- `statsmodels` (Statistical models)
- `lightgbm` (Gradient boosting)
- `matplotlib`, `seaborn` (Visualization)

## 🔄 Reproducibility

This project ensures full reproducibility through:

1. **Fixed Random Seeds**: `random_seed: 42` in config
2. **Version Pinning**: Exact library versions in requirements.txt
3. **Data Caching**: Raw API responses saved with timestamps
4. **Model Checkpoints**: Exact Hugging Face model identifiers
5. **Results Tracking**: All metrics and predictions saved

To reproduce results:
```bash
git clone https://github.com/Vipproplayerone1/ts-chronos-gpu.git
cd ts-chronos-gpu
conda env create -f environment.yml
conda activate ts-chronos-gpu
bash run_end_to_end.sh
```

## 📝 Documentation

- **Report**: `docs/report.pdf` - Full technical report (≤6 pages)
- **Slides**: `docs/slides.pdf` - Presentation deck (6-8 slides)
- **Model Card**: `docs/model_card.md` - Chronos-2 model documentation

## 🔧 Troubleshooting

### **Issue: Package not found**
```bash
# Solution: Install missing package
pip install <package-name>

# Or reinstall all:
pip install -r requirements.txt --force-reinstall
```

### **Issue: CUDA not available**
```bash
# Check GPU
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### **Issue: Out of memory**
**Solution**: Edit `configs/default.yaml`:
```yaml
models:
  chronos:
    device: "cpu"  # Change from "cuda" to "cpu"
    model_name: "amazon/chronos-t5-small"  # Use smaller model
```

### **Issue: Pipeline hangs or fails**
```bash
# Check logs in terminal
# Common causes:
# 1. Internet connection (for data download)
# 2. Disk space (need ~1GB free)
# 3. Memory (need 8GB+ RAM)

# Quick fix: Run with smaller dataset
# Edit configs/default.yaml:
data:
  end_date: "2023-12-31"  # Use less data
```

### **Issue: Import errors**
```bash
# Make sure you're in the project directory
cd D:\Major\Apply_Forcasting\final

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

## ❓ FAQ

**Q: How long does the pipeline take?**
A: 30-60 minutes with GPU, 2-3 hours with CPU

**Q: Can I use a different Wikipedia page?**
A: Yes! Edit `configs/default.yaml` → `data.page_title: "Your_Page"`

**Q: Where are the results saved?**
A: All in `artifacts/` folder (predictions, metrics, figures)

**Q: Can I run this without a GPU?**
A: Yes, set `device: "cpu"` in `configs/default.yaml`, but it's slower

**Q: How do I cite this project?**
A: See references section below

## 🤝 Contributing

This is an academic project. For issues or questions:
- Open an issue on GitHub
- Check `docs/REPORT_TEMPLATE.md` for methodology details

## 📄 License

This project is for educational purposes. Data is from Wikipedia (CC BY-SA 3.0).

## 🙏 Acknowledgments

- **Chronos-2**: Amazon Science (https://github.com/amazon-science/chronos-forecasting)
- **Data Source**: Wikimedia REST API
- **Framework**: statsmodels, scikit-learn, LightGBM

## 📚 References

1. Ansari et al. (2024). "Chronos: Learning the Language of Time Series"
2. Hyndman & Athanasopoulos (2021). "Forecasting: Principles and Practice"
3. Wikipedia Pageviews API Documentation

---

## 📞 Support

**Having issues?**
1. Check [Troubleshooting](#-troubleshooting) section above
2. Run `python test_setup.py` to diagnose problems
3. Check GitHub Issues
4. Review error messages carefully

**Status**: ✅ Production Ready | **Last Updated**: December 2024

🤖 Generated with [Claude Code](https://claude.com/claude-code)
