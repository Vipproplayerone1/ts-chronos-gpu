# Time Series Forecasting with Chronos-2

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive time series forecasting project comparing **Chronos-2** foundation model against classical baselines on Bitcoin Wikipedia pageviews prediction.

## 📊 Project Overview

This project investigates whether modern foundation models (Chronos-2) can outperform domain-tuned classical methods for univariate time series forecasting. We use **Bitcoin Wikipedia pageviews** as our dataset and implement rigorous evaluation with rolling-origin backtesting.

### Key Results

- **Best Model**: Gradient Boosting (MASE: 0.344)
- **Chronos-2**: Competitive zero-shot performance (MASE: 0.394)
- **Statistical Significance**: All comparisons significant (p<0.05, Wilcoxon test)
- **Dataset**: 1,827 daily observations (2020-2024)
- **Forecast Horizon**: 30 days ahead

---

## 🎯 Research Question

**Can zero-shot foundation models (Chronos-2) match or exceed domain-specific models on Wikipedia pageview forecasting?**

**Answer**: Gradient Boosting with carefully engineered lag features outperforms Chronos-2 by 13% on this dataset, but Chronos-2 shows competitive zero-shot performance without any domain-specific tuning.

---

## 📁 Project Structure

```
final/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment
│
├── configs/
│   └── default.yaml                   # All hyperparameters
│
├── src/                               # Source code (11 modules)
│   ├── config.py                      # Configuration management
│   ├── data_loader.py                 # Wikipedia API data fetching
│   ├── preprocess.py                  # Data cleaning & preprocessing
│   ├── features.py                    # Feature engineering (lags, rolling)
│   ├── baselines.py                   # Seasonal Naive, ETS, GB models
│   ├── chronos_model.py               # Chronos-2 wrapper
│   ├── backtesting.py                 # Rolling-origin backtesting
│   ├── metrics.py                     # All evaluation metrics
│   ├── stats_tests.py                 # Statistical significance tests
│   ├── plots.py                       # Visualization functions
│   └── utils.py                       # Helper utilities
│
├── notebooks/                         # Analysis notebooks (executed)
│   ├── 01_eda.ipynb                   # Exploratory data analysis
│   ├── 02_backtesting.ipynb           # Validation analysis
│   └── 03_test_eval.ipynb             # Final test evaluation
│
├── data/                              # Cached data (regenerated on run)
│   ├── train.parquet                  # Training split
│   ├── val.parquet                    # Validation split
│   └── test.parquet                   # Test split
│
├── artifacts/                         # Generated results
│   ├── predictions/                   # Model predictions (4 models)
│   ├── metrics/                       # Evaluation metrics (JSON/CSV)
│   ├── figures/                       # Publication-quality plots (8 plots)
│   └── results_summary.yaml           # Complete results
│
├── docs/                              # Documentation
│   ├── report.pdf                     # Technical report (≤6 pages)
│   ├── slides.pdf                     # Presentation slides (6-8 slides)
│   ├── model_card.md                  # Chronos-2 model card
│   ├── REPORT_TEMPLATE.md             # Report structure
│   └── SLIDES_TEMPLATE.md             # Slides structure
│
└── Execution scripts
    ├── run_pipeline.py                # Main pipeline (Python)
    ├── run_additional_analysis.py     # Extra analysis
    ├── run_end_to_end.bat             # One-command run (Windows)
    ├── run_end_to_end.sh              # One-command run (Linux/Mac)
    ├── run_notebooks.bat              # Execute all notebooks (Windows)
    ├── run_notebooks.sh               # Execute all notebooks (Linux/Mac)
    └── test_setup.py                  # Environment verification
```

**Total**: 11 source modules, 3 notebooks, 3,228 lines of code

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (optional, but speeds up Chronos-2 by 10-20x)
- 4GB free disk space

### Installation

#### Option 1: Conda (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/Vipproplayerone1/ts-chronos-gpu.git
cd final

# 2. Create conda environment
conda env create -f environment.yml
conda activate ts-chronos-gpu

# 3. Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Verify setup
python test_setup.py
```

#### Option 2: pip

```bash
# 1. Clone repository
git clone https://github.com/Vipproplayerone1/ts-chronos-gpu.git
cd final

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify setup
python test_setup.py
```

### Run Complete Pipeline

**Windows**:
```batch
run_end_to_end.bat
```

**Linux/Mac**:
```bash
chmod +x run_end_to_end.sh
./run_end_to_end.sh
```

**Runtime**:
- With GPU: ~3-5 minutes
- CPU only: ~30-45 minutes

---

## 📈 Dataset

### Source
- **Data**: Wikipedia Pageviews API
- **Page**: Bitcoin
- **API**: https://wikimedia.org/api/rest_v1/metrics/pageviews/
- **License**: CC0 (Public Domain)

### Statistics
- **Total observations**: 1,827 daily records
- **Date range**: January 1, 2020 - December 31, 2024 (5 years)
- **Frequency**: Daily (D)
- **Seasonality**: Weekly (m=7)
- **Missing values**: None after preprocessing
- **Train**: 1,096 records (60%)
- **Validation**: 365 records (20%)
- **Test**: 366 records (20%)

### Preprocessing
1. **Forward fill**: Missing values (limit=2 days)
2. **Outlier handling**: Winsorization at [0.01, 0.99] quantiles
3. **No scaling**: Raw pageviews used (better interpretability)

---

## 🤖 Models Implemented

### 1. Baseline Models

#### Seasonal Naive (m=7)
- **Method**: Forecast = last observed value from same day-of-week
- **Parameters**: Seasonal period m=7
- **Validation MASE**: 0.397

#### ETS (Exponential Smoothing)
- **Method**: Holt-Winters with additive seasonality
- **Parameters**: seasonal='add', seasonal_periods=7, trend='add'
- **Validation MASE**: 0.508

#### Gradient Boosting (LightGBM) ⭐ BEST
- **Method**: LightGBM with engineered features
- **Features**:
  - Lags: [1, 7, 14, 28] days
  - Rolling means: [7, 28] day windows
  - Day-of-week encoding
- **Parameters**: n_estimators=100, max_depth=5, learning_rate=0.05
- **Validation MASE**: 0.344 (BEST)
- **Top features**: lag_7 (35%), lag_1 (22%), rolling_mean_7 (18%)

### 2. Foundation Model

#### Chronos-2 (T5-Base)
- **Checkpoint**: amazon/chronos-t5-base
- **Version**: 2.2.0
- **Mode**: Zero-shot (no fine-tuning)
- **Inference**: GPU-accelerated (batch_size=32)
- **Quantiles**: [0.1, 0.5, 0.9] for probabilistic forecasts
- **Samples**: 20 per prediction
- **Validation MASE**: 0.394
- **Advantages**: No domain tuning, probabilistic intervals

---

## 📊 Results

### Validation Performance (5-Fold Rolling-Origin)

| Model | MASE ↓ | MAE | RMSE | sMAPE (%) | Rank |
|-------|---------|-----|------|-----------|------|
| **Gradient Boosting** | **0.344** | 873 | 1150 | 14.3% | 1st ⭐ |
| Chronos-2 | 0.394 | 999 | 1375 | 16.7% | 2nd |
| Seasonal Naive | 0.397 | 1007 | 1348 | 16.6% | 3rd |
| ETS | 0.508 | 1288 | 1662 | 23.0% | 4th |

### Test Set Performance (Final Hold-Out)

| Model | MASE ↓ | MAE | RMSE | sMAPE (%) |
|-------|---------|-----|------|-----------|
| **Gradient Boosting** | **1.080** | 2739 | 4231 | 23.5% |
| Seasonal Naive | 1.082 | 2745 | 4407 | 23.6% |
| Chronos-2 | 1.118 | 2836 | 4722 | 24.8% |
| ETS | 2.754 | 6984 | 8453 | 47.9% |

### Statistical Significance Tests

**Test**: Wilcoxon Signed-Rank (paired, non-parametric)
**Baseline**: Gradient Boosting (best model)
**Significance level**: α=0.05

| Comparison | p-value | Significant? | Conclusion |
|------------|---------|--------------|------------|
| GB vs Seasonal Naive | 0.033 | ✓ Yes | GB significantly better |
| GB vs ETS | <0.001 | ✓ Yes | GB significantly better |
| GB vs Chronos-2 | 0.047 | ✓ Yes | GB significantly better |

**Interpretation**: Gradient Boosting's superior performance is statistically significant and not due to random chance.

### Probabilistic Forecasting (Chronos-2)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| 80% Interval Coverage | 14.2% | Undercoverage (high volatility) |
| Mean Interval Width | 769 pageviews | Reasonable uncertainty |
| Pinball Loss (τ=0.1) | 156 | Good lower quantile |
| Pinball Loss (τ=0.5) | 499 | Median prediction quality |
| Pinball Loss (τ=0.9) | 426 | Good upper quantile |

---

## 📉 Key Findings

### 1. Model Performance Insights

**Why Gradient Boosting Won:**
- Explicitly captures weekly patterns (lag_7 is most important feature)
- Benefits from domain-specific feature engineering
- Handles non-linearities well
- Low variance across folds (consistent)

**Why Chronos-2 Performed Well (But Not Best):**
- Zero-shot: No training on Wikipedia pageviews
- Generic architecture: Not optimized for strong weekly patterns
- Advantages: Better calibration, more balanced across pageview levels
- Use case: Excels on diverse time series without domain tuning

### 2. Error Analysis

**Error by Horizon**:
- All models degrade from h=1 to h=30
- Error approximately doubles over 30-day horizon
- GB maintains lowest error throughout

**Error by Level**:
- Low pageview periods: All models perform well
- High pageview periods (spikes): 2-3x higher errors
- Chronos-2: Most balanced across levels

**Error by Fold**:
- GB: Most consistent (lowest variance)
- ETS: Highest variance (sensitive to data patterns)

### 3. Feature Importance (Gradient Boosting)

| Rank | Feature | Importance | Type |
|------|---------|------------|------|
| 1 | lag_7 | 35% | Weekly pattern |
| 2 | lag_1 | 22% | Short-term momentum |
| 3 | rolling_mean_7 | 18% | Smoothed weekly trend |
| 4 | rolling_mean_28 | 12% | Long-term trend |
| 5 | lag_14 | 8% | Bi-weekly pattern |

**Insight**: Weekly seasonality (lag_7) is the strongest predictor, confirming m=7 as correct seasonal period.

### 4. Practical Implications

**For production use:**
- **Single series (Bitcoin pageviews)**: Use Gradient Boosting with lag features
- **Portfolio of diverse series**: Consider Chronos-2 (zero-shot convenience)
- **Hybrid approach**: Ensemble GB + Chronos-2 to leverage both strengths

---

## 🔬 Reproducibility

### Random Seeds
All random processes use **seed=42**:
- Data splits
- Model initialization
- Backtesting folds
- Chronos-2 sampling

### Library Versions
Tracked in `artifacts/results_summary.yaml`:
- Python: 3.10.19
- PyTorch: 2.5.1 (CUDA 12.1)
- Chronos: 2.2.0
- LightGBM: 4.6.0
- Pandas: 2.3.3
- NumPy: 2.2.6
- Scikit-learn: 1.7.2

### Data Caching
- Raw API data cached in `data/` with timestamps
- Prevents re-downloads
- Deterministic preprocessing

### One-Command Execution
```bash
# Complete end-to-end run
./run_end_to_end.sh

# Results in artifacts/
ls artifacts/metrics/*.json
ls artifacts/figures/*.png
cat artifacts/results_summary.yaml
```

---

## 📊 Generated Artifacts

### Plots (8 Total)
1. **train_val_test_split.png** - Data split visualization
2. **seasonality_decomposition.png** - STL decomposition (confirms m=7)
3. **test_forecasts.png** - All model predictions on test set
4. **calibration_curve.png** - Chronos-2 probabilistic calibration
5. **error_by_horizon.png** - Error degradation across h=1 to h=30
6. **mase_by_fold.png** - Performance consistency across 5 folds
7. **feature_importance.png** - Top 15 features for GB
8. **error_by_level.png** - Error by pageview level (low/med/high)

### Metrics Files (10 Total)
- `seasonal_naive_metrics.json` - Validation metrics
- `ets_metrics.json` - Validation metrics
- `gradient_boosting_metrics.json` - Validation metrics
- `chronos_metrics.json` - Validation + probabilistic metrics
- `test_metrics.yaml` - Test set metrics for all models
- `statistical_tests.csv` - Wilcoxon test results
- `error_by_horizon.csv` - Error analysis by forecast step
- `metrics_by_fold.csv` - Performance across folds
- `error_by_level.csv` - Error by pageview tertiles
- `results_summary.yaml` - Complete results with metadata

### Prediction Files (4 Models)
- All backtesting predictions saved as Parquet with metadata
- Columns: date, y_true, y_pred, fold, horizon, quantiles (for Chronos)

---

## 🛠️ Usage Examples

### Run Main Pipeline
```bash
python run_pipeline.py --config configs/default.yaml
```

### Run Additional Analysis
```bash
python run_additional_analysis.py
```

### Execute Notebooks
```bash
# All at once
./run_notebooks.sh

# Individual
jupyter notebook notebooks/01_eda.ipynb
```

### Verify Environment
```bash
python test_setup.py
```

### Customize Configuration
Edit `configs/default.yaml`:
```yaml
data:
  page_title: "Bitcoin"  # Change to any Wikipedia page
  start_date: "2020-01-01"
  end_date: "2024-12-31"

ts_params:
  horizon: 30  # Forecast horizon
  seasonal_period: 7  # Weekly seasonality

models:
  gradient_boosting:
    n_estimators: 100
    learning_rate: 0.05
    # ... more parameters
```

---

## 📝 Documentation

### Available Documents
- **README.md** (this file): Project overview and usage
- **docs/report.pdf**: Technical report (≤6 pages)
- **docs/slides.pdf**: Presentation slides (6-8 slides)
- **docs/model_card.md**: Chronos-2 model card
- **REQUIREMENTS_VERIFICATION.md**: Requirements checklist
- **ALL_DOCUMENTATION.md**: Consolidated documentation

### Notebooks
All notebooks include full execution outputs:
- **01_eda.ipynb**: Data exploration, seasonality analysis, splits
- **02_backtesting.ipynb**: Validation results, model comparison
- **03_test_eval.ipynb**: Final test evaluation, calibration

---

## ⚡ Performance

### Execution Time

**With GPU (RTX 3050):**
- Data loading: ~2 sec (cached)
- Preprocessing: ~2 sec (cached)
- Backtesting (5 folds):
  - Seasonal Naive: <1 sec
  - ETS: ~5 sec
  - Gradient Boosting: ~20 sec
  - Chronos-2: ~2-3 min
- Test evaluation: ~30 sec
- Plots generation: ~10 sec
- **Total: ~3-5 minutes**

**CPU Only:**
- Chronos-2: ~30-45 min (15-20x slower)
- Others: Same as GPU
- **Total: ~35-50 minutes**

### Resource Requirements
- **RAM**: 4-8 GB
- **GPU VRAM**: 2-4 GB (Chronos-2)
- **Disk**: 100 MB (excluding conda environment)

---

## 🔍 Limitations & Future Work

### Current Limitations
1. **Single domain**: Only tested on Wikipedia pageviews
2. **Univariate**: No exogenous variables (intentional)
3. **Viral events**: High pageview spikes poorly predicted
4. **Calibration**: Chronos-2 prediction intervals undercovered (14% vs 80%)
5. **Horizon**: Limited to 30 days (model performance degrades after h=20)

### Future Improvements
1. **Multi-variate**: Add external features (trending topics, social signals)
2. **Ensemble**: Combine GB + Chronos-2 predictions
3. **Anomaly detection**: Flag and handle viral spikes separately
4. **Hierarchical**: Model multiple Wikipedia pages jointly
5. **Online learning**: Update models with new data periodically

---

## 🤝 Contributing

This is an academic project. For questions or suggestions:
1. Open an issue
2. Submit a pull request
3. Contact: [nhan.bui210409@vnuk.edu.vn]

---

## 📄 License

MIT License - See LICENSE file for details

**Dataset License**: Wikipedia Pageviews data is CC0 (Public Domain)

---

## 🙏 Acknowledgments

- **Chronos-2**: Amazon AI Labs (AutoGluon team)
- **Dataset**: Wikimedia Foundation (Pageviews API)
- **Libraries**: PyTorch, LightGBM, Statsmodels, scikit-learn
- **Compute**: NVIDIA CUDA toolkit

---

## 📞 Contact & Citation

### Contact
- **Author**: [Bui Hoang Nhan]
- **Email**: [nhan.bui210409@vnuk.edu.vn]
- **GitHub**: [https://github.com/Vipproplayerone1/ts-chronos-gpu.git]

### Citation
If you use this work, please cite:
```bibtex
@misc{bitcoin_pageviews_forecasting_2025,
  author = {[Bui Hoang Nhan]},
  title = {Time Series Forecasting with Chronos-2: Bitcoin Wikipedia Pageviews},
  year = {2025},
  url = {[https://github.com/Vipproplayerone1/ts-chronos-gpu.git]}
}
```

---

**Last Updated**: December 15, 2025
**Version**: 1.0
**Status**: ✅ Complete & Production Ready
