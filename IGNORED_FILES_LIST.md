# Git Ignored Files - Complete List

**Generated**: December 15, 2025

This document shows all files currently being ignored by `.gitignore`.

---

## 📊 Summary

- **Total Files Ignored**: 22 files/directories
- **Total Size**: 3.33 MB
- **Why Ignored**: These are generated files that can be recreated by running the pipeline

---

## 🗂️ Ignored Files by Category

### 1. Plots (8 files - 3.05 MB)

| File | Size | Purpose |
|------|------|---------|
| `artifacts/figures/calibration_curve.png` | 168.5 KB | Chronos-2 calibration analysis |
| `artifacts/figures/error_by_horizon.png` | 468.3 KB | Error degradation h=1 to h=30 |
| `artifacts/figures/error_by_level.png` | 121.0 KB | Error by pageview level |
| `artifacts/figures/feature_importance.png` | 109.9 KB | Top 15 GB features |
| `artifacts/figures/mase_by_fold.png` | 353.7 KB | Performance across folds |
| `artifacts/figures/seasonality_decomposition.png` | 979.5 KB | STL decomposition |
| `artifacts/figures/test_forecasts.png` | 451.6 KB | All models on test set |
| `artifacts/figures/train_val_test_split.png` | 471.9 KB | Data split visualization |

### 2. Metrics (5 files - 3.9 KB)

| File | Size | Purpose |
|------|------|---------|
| `artifacts/metrics/chronos_metrics.json` | 0.7 KB | Chronos-2 validation metrics |
| `artifacts/metrics/ets_metrics.json` | 0.7 KB | ETS validation metrics |
| `artifacts/metrics/gradient_boosting_metrics.json` | 0.7 KB | GB validation metrics |
| `artifacts/metrics/seasonal_naive_metrics.json` | 0.7 KB | Seasonal Naive metrics |
| `artifacts/metrics/test_metrics.yaml` | 1.1 KB | Test set metrics (all models) |

**Note**: `artifacts/results_summary.yaml` is **NOT ignored** (kept for reproducibility)

### 3. Predictions (4 files - 34.9 KB)

| File | Size | Purpose |
|------|------|---------|
| `artifacts/predictions/chronos_backtest.parquet` | 7.8 KB | Chronos-2 backtest predictions |
| `artifacts/predictions/ets_backtest.parquet` | 10.7 KB | ETS backtest predictions |
| `artifacts/predictions/gradient_boosting_backtest.parquet` | 9.9 KB | GB backtest predictions |
| `artifacts/predictions/seasonal_naive_backtest.parquet` | 6.5 KB | Seasonal Naive predictions |

### 4. Cached Data (4 files - 164.6 KB)

| File | Size | Purpose |
|------|------|---------|
| `data/Bitcoin_2020-01-01_2024-12-31.json` | 134.6 KB | Raw Wikipedia API data |
| `data/test.parquet` | 6.6 KB | Test split (cached) |
| `data/train.parquet` | 16.9 KB | Train split (cached) |
| `data/val.parquet` | 6.5 KB | Validation split (cached) |

### 5. Python Cache (1 directory - 82.3 KB)

| File | Size | Purpose |
|------|------|---------|
| `src/__pycache__/` | 82.3 KB | Python bytecode cache |

---

## 🎯 Active .gitignore Patterns

These patterns will catch current and future files:

### Python
```
__pycache__/
*.py[cod]
*$py.class
*.so
*.egg-info/
build/
dist/
```

### Environments
```
venv/
.venv
env/
ENV/
*.env
```

### IDEs
```
.vscode/
.idea/
*.swp
*.swo
*~
```

### Jupyter
```
.ipynb_checkpoints/
```

### Data Files
```
data/*.csv
data/*.parquet
data/*.json
```

### Artifacts
```
artifacts/predictions/*.csv
artifacts/predictions/*.parquet
artifacts/metrics/*.json
artifacts/metrics/*.yaml
artifacts/metrics/*.csv
artifacts/figures/*.png
artifacts/figures/*.pdf
```

### Office/Documents
```
~$*.docx         # Word lock files
~$*.xlsx         # Excel lock files
~$*.pptx         # PowerPoint lock files
*.tmp
```

### LaTeX/Reports
```
*.aux
*.log
*.out
*.toc
*.synctex.gz
*.fdb_latexmk
*.fls
```

### Model Checkpoints
```
models/
checkpoints/
*.pt
*.pth
*.pkl
```

### OS Files
```
.DS_Store        # macOS
Thumbs.db        # Windows
```

---

## ✅ Files NOT Ignored (Tracked in Git)

These essential files ARE committed to git:

### Source Code
```
✓ src/config.py
✓ src/data_loader.py
✓ src/preprocess.py
✓ src/features.py
✓ src/baselines.py
✓ src/chronos_model.py
✓ src/backtesting.py
✓ src/metrics.py
✓ src/stats_tests.py
✓ src/plots.py
✓ src/utils.py
```

### Notebooks
```
✓ notebooks/01_eda.ipynb
✓ notebooks/02_backtesting.ipynb
✓ notebooks/03_test_eval.ipynb
```

### Configuration
```
✓ configs/default.yaml
✓ requirements.txt
✓ environment.yml
✓ .gitignore
```

### Scripts
```
✓ run_pipeline.py
✓ run_additional_analysis.py
✓ run_end_to_end.bat
✓ run_end_to_end.sh
✓ run_notebooks.bat
✓ run_notebooks.sh
✓ test_setup.py
```

### Documentation
```
✓ README.md
✓ docs/model_card.md
✓ docs/report.pdf
✓ docs/slides.pdf
✓ docs/Time Series Forecasting with Chronos-2.docx (source)
✓ docs/Time Series Forecasting with Chronos-2.pdf
✓ ALL_DOCUMENTATION.md
✓ REQUIREMENTS_VERIFICATION.md
✓ UNNECESSARY_FILES.md
```

### Important Results
```
✓ artifacts/results_summary.yaml (EXCEPTION - kept for reproducibility)
```

---

## 🔄 How to Regenerate Ignored Files

All ignored files can be recreated by running:

**Windows**:
```batch
run_end_to_end.bat
```

**Linux/Mac**:
```bash
./run_end_to_end.sh
```

**Or step by step**:
```bash
# 1. Run main pipeline
python run_pipeline.py

# 2. Run additional analysis
python run_additional_analysis.py

# 3. Execute notebooks (optional)
./run_notebooks.sh
```

**Runtime**: 3-5 minutes (with GPU), 35-50 minutes (CPU only)

---

## 💡 Why These Files Are Ignored

1. **Reproducibility**: All can be regenerated from source code
2. **Repository Size**: Keeps git repo small (~10 MB vs 13 MB)
3. **Version Control**: Source code + config is enough to recreate everything
4. **Collaboration**: Others can generate their own results
5. **Best Practice**: Don't commit generated files

---

## 🚫 To Force-Add an Ignored File (Not Recommended)

If you really need to commit an ignored file:

```bash
git add -f path/to/ignored/file
```

**Warning**: Only do this if you have a specific reason (e.g., pre-generated results for CI/CD).

---

## 📝 Optional: Ignore More Files

To also ignore verification documents (after submission), uncomment these lines in `.gitignore`:

```bash
# Change from:
# ALL_DOCUMENTATION.md
# REQUIREMENTS_VERIFICATION.md

# To:
ALL_DOCUMENTATION.md
REQUIREMENTS_VERIFICATION.md
```

---

**Last Updated**: December 15, 2025
**Total Size Saved**: 3.33 MB
**Status**: ✅ All Generated Files Properly Ignored
