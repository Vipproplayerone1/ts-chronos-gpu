# Time Series Forecasting with Chronos-2: Wikipedia Pageviews

A comprehensive, reproducible time series forecasting project using the Chronos-2 foundation model with proper baseline comparisons, rolling-origin backtesting, and statistical significance testing.

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

## 🚀 Quick Start

### 1. Environment Setup

**Option A: Using Conda (Recommended)**

```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate ts-chronos-gpu

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Option B: Using pip**

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Pipeline

**Single Command (Recommended)**

```bash
bash run_end_to_end.sh
```

**Or run Python script directly**

```bash
python run_pipeline.py --config configs/default.yaml
```

### 3. View Results

```bash
# Check summary
cat artifacts/results_summary.yaml

# View plots
ls artifacts/figures/

# Explore notebooks
jupyter notebook notebooks/
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

## 🤝 Contributing

This is an academic project. For issues or questions:
- Open an issue on GitHub
- Check `docs/report.pdf` for methodology details

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

**Status**: ✅ Production Ready
**Last Updated**: December 2024
**Contact**: [Your GitHub]

🤖 Generated with [Claude Code](https://claude.com/claude-code)
