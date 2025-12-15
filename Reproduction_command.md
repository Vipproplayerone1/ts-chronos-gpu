# Reproduction Commands

**Project**: Time Series Forecasting with Chronos-2
**Version**: v1.0
**Expected Runtime**: 3-5 minutes (GPU) | 35-50 minutes (CPU)
**Random Seed**: 42 (fixed for reproducibility)

---

## 🚀 Quick Start (One Command)

### Windows
```bash
git clone https://github.com/Vipproplayerone1/ts-chronos-gpu.git && cd ts-chronos-gpu && git checkout v1.0 && conda env create -f environment.yml && conda activate ts-chronos-gpu && run_end_to_end.bat
```

### Linux/Mac
```bash
git clone https://github.com/Vipproplayerone1/ts-chronos-gpu.git && cd ts-chronos-gpu && git checkout v1.0 && conda env create -f environment.yml && conda activate ts-chronos-gpu && ./run_end_to_end.sh
```

---

## 📋 Step-by-Step Instructions

### 1. Clone Repository

```bash
git clone https://github.com/Vipproplayerone1/ts-chronos-gpu.git
cd ts-chronos-gpu
```

### 2. Checkout Tagged Version

```bash
git checkout v1.0
```

**Verify you're on the correct version:**
```bash
git describe --tags
# Expected output: v1.0
```

### 3. Create Conda Environment

```bash
conda env create -f environment.yml
```

**This installs:**
- Python 3.10
- PyTorch 2.1.2 (with CUDA 12.1 support)
- chronos-forecasting 2.0.0
- LightGBM, statsmodels, pandas, numpy
- Jupyter, matplotlib, seaborn

**Environment creation time**: ~5-10 minutes (depends on internet speed)

### 4. Activate Environment

```bash
conda activate ts-chronos-gpu
```

**Verify installation:**
```bash
python test_setup.py
```

**Expected output:**
```
✓ Python version: 3.10.x
✓ PyTorch version: 2.1.2+cu121
✓ CUDA available: True (or False if CPU-only)
✓ chronos-forecasting installed
✓ All required packages installed
```

### 5. Run End-to-End Pipeline

**Windows:**
```bash
run_end_to_end.bat
```

**Linux/Mac:**
```bash
chmod +x run_end_to_end.sh
./run_end_to_end.sh
```

**What this does:**
1. Downloads Wikipedia pageviews data (if not cached)
2. Preprocesses and splits data (train/val/test)
3. Runs 5-fold rolling-origin backtesting:
   - Seasonal Naive baseline
   - ETS (Exponential Smoothing)
   - Gradient Boosting (LightGBM)
   - Chronos-2 (T5-Base) foundation model
4. Evaluates on test set
5. Performs statistical tests (Wilcoxon signed-rank)
6. Generates all plots and tables
7. Saves results to `artifacts/`

**Expected runtime:**
- **GPU (CUDA)**: 3-5 minutes
- **CPU only**: 35-50 minutes

### 6. Optional: Run Additional Analysis

```bash
python run_additional_analysis.py
```

**Generates:**
- Error by horizon analysis
- Error by level (high/low demand)
- Feature importance for Gradient Boosting
- Statistical test details

### 7. Optional: Run Jupyter Notebooks

```bash
jupyter notebook
```

**Then open:**
1. `notebooks/01_eda.ipynb` - Exploratory Data Analysis
2. `notebooks/02_backtesting.ipynb` - Validation Analysis
3. `notebooks/03_test_eval.ipynb` - Test Set Evaluation

**Or run all notebooks at once:**

**Windows:**
```bash
run_notebooks.bat
```

**Linux/Mac:**
```bash
chmod +x run_notebooks.sh
./run_notebooks.sh
```

---

## 📊 Expected Outputs

After running the pipeline, you should have:

### Predictions (Parquet files)
```
artifacts/predictions/
├── seasonal_naive_backtest.parquet
├── seasonal_naive_backtest_metadata.json
├── ets_backtest.parquet
├── ets_backtest_metadata.json
├── gradient_boosting_backtest.parquet
├── gradient_boosting_backtest_metadata.json
├── chronos_backtest.parquet
└── chronos_backtest_metadata.json
```

### Metrics (JSON/CSV/YAML)
```
artifacts/metrics/
├── seasonal_naive_metrics.json
├── ets_metrics.json
├── gradient_boosting_metrics.json
├── chronos_metrics.json
├── metrics_by_fold.csv
├── error_by_horizon.csv
├── error_by_level.csv
├── statistical_tests.csv
└── test_metrics.yaml
```

### Figures (PNG)
```
artifacts/figures/
├── train_val_test_split.png          (Data split visualization)
├── seasonality_decomposition.png      (STL decomposition)
├── mase_by_fold.png                   (Validation performance)
├── test_forecasts.png                 (Test predictions overlay)
├── error_by_horizon.png               (Metric vs. forecast step)
├── error_by_level.png                 (High/low demand comparison)
├── calibration_curve.png              (Prediction interval calibration)
└── feature_importance.png             (Gradient Boosting features)
```

### Summary Results
```
artifacts/results_summary.yaml
```

---

## 🔍 Verification

### Check Key Results

```bash
# View test metrics
cat artifacts/metrics/test_metrics.yaml
```

**Expected test MASE (±0.05 due to randomness):**
- Seasonal Naive: 1.000 (baseline)
- ETS: 0.391
- Gradient Boosting: **0.344** (best)
- Chronos-2: 0.394

### Verify Statistical Tests

```bash
# View statistical test results
cat artifacts/metrics/statistical_tests.csv
```

**Expected:**
- All p-values < 0.05 (statistically significant)
- Gradient Boosting vs. Chronos-2: Comparable performance

### Check Plots

```bash
# List generated plots
ls -lh artifacts/figures/*.png
```

**Expected:** 8 PNG files, total ~3.2 MB

### Verify Reproducibility

Run the pipeline twice and compare results:

```bash
# First run
run_end_to_end.bat  # or ./run_end_to_end.sh
cp artifacts/metrics/test_metrics.yaml test_metrics_run1.yaml

# Second run
run_end_to_end.bat  # or ./run_end_to_end.sh
cp artifacts/metrics/test_metrics.yaml test_metrics_run2.yaml

# Compare
diff test_metrics_run1.yaml test_metrics_run2.yaml
```

**Expected:** No differences (results are deterministic with seed=42)

---

## 💻 Hardware Requirements

### Minimum (CPU-only)
- CPU: 4 cores, 2.0 GHz+
- RAM: 8 GB
- Storage: 2 GB free
- Runtime: 35-50 minutes

### Recommended (GPU)
- GPU: NVIDIA with 4+ GB VRAM (e.g., RTX 3050, GTX 1650)
- CUDA: 11.8 or 12.1
- RAM: 16 GB
- Storage: 5 GB free
- Runtime: 3-5 minutes

### Tested On
- GPU: NVIDIA RTX 3050 Laptop (4GB VRAM)
- CPU: Intel i7-11800H (8 cores)
- RAM: 16 GB
- OS: Windows 11
- CUDA: 12.1

---

## 🐛 Troubleshooting

### Issue 1: CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
```python
# Edit configs/default.yaml
chronos:
  batch_size: 16  # Reduce from 32
```

### Issue 2: Conda Environment Creation Fails

**Error:** `ResolvePackageNotFound` or `CondaHTTPError`

**Solution:**
```bash
# Use pip instead
pip install -r requirements.txt
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```

### Issue 3: Wikipedia API Rate Limit

**Error:** `HTTPError: 429 Too Many Requests`

**Solution:** Data is already cached in `data/Bitcoin_2020-01-01_2024-12-31.json`. If missing, the pipeline downloads it automatically with retry logic.

### Issue 4: Chronos Model Download Fails

**Error:** `OSError: Can't load 'amazon/chronos-t5-base'`

**Solution:**
```bash
# Pre-download model
python -c "from chronos import ChronosPipeline; ChronosPipeline.from_pretrained('amazon/chronos-t5-base')"
```

### Issue 5: Permission Denied (Linux/Mac)

**Error:** `Permission denied: './run_end_to_end.sh'`

**Solution:**
```bash
chmod +x run_end_to_end.sh
chmod +x run_notebooks.sh
```

### Issue 6: Unicode Encoding Error (Windows)

**Error:** `UnicodeEncodeError: 'charmap' codec can't encode character`

**Solution:** Already fixed in notebooks (using ASCII characters). If you still see this, set:
```bash
set PYTHONIOENCODING=utf-8
```

---

## 📦 Alternative: Use Cached Results

If you cannot run the pipeline (e.g., no GPU, limited time), you can verify results using cached artifacts:

```bash
# Clone repository
git clone https://github.com/Vipproplayerone1/ts-chronos-gpu.git
cd ts-chronos-gpu
git checkout v1.0

# Artifacts are already included in the repository
# View results directly:
cat artifacts/metrics/test_metrics.yaml
ls artifacts/figures/
```

**Note:** Cached results are from the original run with seed=42 on RTX 3050 GPU.

---

## 🔄 Clean and Re-run

To start fresh:

```bash
# Remove generated artifacts (keeps cached data)
rm -rf artifacts/predictions/*
rm -rf artifacts/metrics/*
rm -rf artifacts/figures/*
rm -f artifacts/results_summary.yaml

# Re-run pipeline
run_end_to_end.bat  # or ./run_end_to_end.sh
```

To remove everything (including cached data):

```bash
# WARNING: This will re-download data from Wikipedia API
rm -rf data/*.parquet
rm -rf artifacts/predictions/*
rm -rf artifacts/metrics/*
rm -rf artifacts/figures/*
rm -f artifacts/results_summary.yaml

# Re-run pipeline
run_end_to_end.bat  # or ./run_end_to_end.sh
```

---

## 📞 Support

If you encounter issues reproducing the results:

1. **Check Python version**: `python --version` (should be 3.10+)
2. **Check PyTorch version**: `python -c "import torch; print(torch.__version__)"`
3. **Check CUDA**: `python -c "import torch; print(torch.cuda.is_available())"`
4. **Run diagnostics**: `python test_setup.py`
5. **Check logs**: Pipeline prints detailed progress; save output to file:
   ```bash
   run_end_to_end.bat > reproduction_log.txt 2>&1
   ```

**Contact:**
- GitHub Issues: https://github.com/Vipproplayerone1/ts-chronos-gpu/issues
- Email: nhan.bui210409@vnuk.edu.vn

---

## 📝 Citation

If you use this code or reproduce these results, please cite:

```bibtex
@misc{bui2024chronos,
  author = {Bui, Hoang Nhan},
  title = {Time Series Forecasting with Chronos-2: Bitcoin Wikipedia Pageviews},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Vipproplayerone1/ts-chronos-gpu}},
  version = {v1.0}
}
```

---

**Last Updated**: December 15, 2024
**Version**: v1.0
**Reproducibility Guarantee**: Results are deterministic with seed=42 on the same hardware.
