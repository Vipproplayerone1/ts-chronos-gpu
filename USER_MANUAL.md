# User Manual: Time Series Forecasting with Chronos-2

**Version**: 2.0
**Last Updated**: January 5, 2026
**Project**: Bitcoin Wikipedia Pageviews Forecasting

---

## Table of Contents

1. [Overview](#1-overview)
2. [System Requirements](#2-system-requirements)
3. [Installation Guide](#3-installation-guide)
4. [Quick Start Guide](#4-quick-start-guide)
5. [Detailed Usage](#5-detailed-usage)
6. [Configuration](#6-configuration)
7. [Understanding the Results](#7-understanding-the-results)
8. [Troubleshooting](#8-troubleshooting)
9. [Advanced Usage](#9-advanced-usage)
10. [FAQ](#10-faq)

---

## 1. Overview

### 1.1 What is This Project?

This project provides a complete framework for time series forecasting that compares:
- **Classical baselines**: Seasonal Naive, ETS, Gradient Boosting
- **Foundation models**: Chronos-2 (zero-shot and fine-tuned)

### 1.2 What Can You Do?

- Forecast Wikipedia pageviews 30 days ahead
- Compare 5 different forecasting models
- Fine-tune foundation models on your data
- Generate publication-quality visualizations
- Run statistical significance tests
- Reproduce research results

### 1.3 Key Features

✅ **5 models**: From simple baselines to advanced foundation models
✅ **Rigorous evaluation**: 5-fold cross-validation with statistical tests
✅ **Memory efficient**: Works with 4GB GPU (or CPU-only)
✅ **Fully reproducible**: Fixed seeds, version-locked dependencies
✅ **Well documented**: Code comments, docstrings, technical reports

---

## 2. System Requirements

### 2.1 Minimum Requirements

- **Operating System**: Windows 10/11, Linux, or macOS
- **Python**: 3.10 or higher
- **RAM**: 8 GB minimum, 16 GB recommended
- **Disk Space**: 4 GB free space
- **Internet**: Required for first-time model download (~500 MB)

### 2.2 Recommended Requirements

- **GPU**: NVIDIA GPU with 4GB+ VRAM (RTX 3050 or better)
- **CUDA**: 12.1 or compatible version
- **RAM**: 16 GB
- **Disk Space**: 10 GB free space

### 2.3 Software Dependencies

Automatically installed via `environment.yml` or `requirements.txt`:
- PyTorch 2.5.1
- Chronos 2.2.0
- LightGBM 4.6.0
- Pandas, NumPy, Matplotlib, Seaborn
- Statsmodels, Scikit-learn

---

## 3. Installation Guide

### 3.1 Option 1: Conda (Recommended)

**Step 1: Clone the Repository**
```bash
git clone https://github.com/Vipproplayerone1/ts-chronos-gpu.git
cd final
```

**Step 2: Create Conda Environment**
```bash
conda env create -f environment.yml
```
This creates an environment named `ts-chronos-gpu`.

**Step 3: Activate Environment**
```bash
conda activate ts-chronos-gpu
```

**Step 4: Install PyTorch with CUDA Support**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

For CPU-only (no GPU):
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

**Step 5: Install Remaining Dependencies**
```bash
pip install -r requirements.txt
```

**Step 6: Verify Installation**
```bash
python test_setup.py
```

You should see:
```
✓ Python version: 3.10.x
✓ PyTorch installed
✓ CUDA available: Yes (or No)
✓ Chronos installed
✓ All dependencies OK
```

### 3.2 Option 2: pip (Virtual Environment)

**Step 1: Clone Repository**
```bash
git clone https://github.com/Vipproplayerone1/ts-chronos-gpu.git
cd final
```

**Step 2: Create Virtual Environment**
```bash
python -m venv venv
```

**Step 3: Activate Environment**

Windows:
```cmd
venv\Scripts\activate
```

Linux/Mac:
```bash
source venv/bin/activate
```

**Step 4: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 5: Verify Installation**
```bash
python test_setup.py
```

### 3.3 Troubleshooting Installation

**Problem**: `conda env create` fails
**Solution**: Update conda: `conda update -n base -c defaults conda`

**Problem**: PyTorch CUDA not detected
**Solution**: Check CUDA version: `nvidia-smi`, then install matching PyTorch version

**Problem**: Import errors
**Solution**: Ensure environment is activated: `conda activate ts-chronos-gpu`

---

## 4. Quick Start Guide

### 4.1 Run Everything (One Command)

**Windows**:
```cmd
run_end_to_end.bat
```

**Linux/Mac**:
```bash
chmod +x run_end_to_end.sh
./run_end_to_end.sh
```

This will:
1. Load Bitcoin Wikipedia pageviews data
2. Train all 5 models (including fine-tuning Chronos-2)
3. Run 5-fold cross-validation
4. Generate evaluation metrics
5. Create visualizations
6. Run statistical tests

**Expected Time**:
- With GPU: ~35-55 minutes (includes fine-tuning)
- CPU only: ~1-2 hours

### 4.2 Run Without Fine-Tuning (Fast)

If you want to skip fine-tuning and run only 4 models:

**Step 1: Edit Configuration**

Open `configs/default.yaml` and comment out the fine-tuned model in the pipeline, or:

**Step 2: Run Pipeline Without Fine-Tuning**
```bash
python run_pipeline.py --config configs/default.yaml --skip_models chronos_finetuned
```

**Expected Time**: ~3-5 minutes with GPU

### 4.3 View Results

After running, check:

**Metrics**:
```bash
# View validation metrics for all models
cat artifacts/metrics/gradient_boosting_metrics.json
cat artifacts/metrics/chronos_finetuned_metrics.json

# View complete results summary
cat artifacts/results_summary.yaml
```

**Visualizations**:
```bash
# Open figures directory
explorer artifacts\figures          # Windows
open artifacts/figures              # Mac
xdg-open artifacts/figures          # Linux
```

**Predictions**:
```bash
# View saved predictions (Parquet format)
python -c "import pandas as pd; print(pd.read_parquet('artifacts/predictions/chronos_finetuned_backtest.parquet').head())"
```

---

## 5. Detailed Usage

### 5.1 Run Main Pipeline

**Basic Usage**:
```bash
python run_pipeline.py --config configs/default.yaml
```

**Command-Line Options**:
```bash
python run_pipeline.py --help

Options:
  --config PATH         Path to config file (default: configs/default.yaml)
  --skip_models LIST    Comma-separated models to skip (e.g., "chronos_finetuned,ets")
  --n_folds INT         Number of CV folds (default: 5)
  --output_dir PATH     Output directory (default: artifacts/)
  --verbose             Enable verbose logging
```

**Examples**:

Run with 3 folds (faster):
```bash
python run_pipeline.py --n_folds 3
```

Skip fine-tuning:
```bash
python run_pipeline.py --skip_models chronos_finetuned
```

Run only Chronos models:
```bash
python run_pipeline.py --skip_models seasonal_naive,ets,gradient_boosting
```

### 5.2 Generate Visualizations

After running the pipeline:

```bash
python visualize_model_comparison.py
```

This generates 12+ comparison plots in `artifacts/figures/detailed_comparison/`:
1. Architecture comparison (5 models)
2. Radar chart (multi-metric)
3. Performance breakdown (validation vs test)
4. Error by horizon
5. Error by pageview level
6. Fold consistency
7. Statistical significance heatmap
8. Model rankings
9. Probabilistic quality (Chronos-specific)
10. Calibration curves
11. Test forecasts
12. Feature importance (Gradient Boosting)

### 5.3 Run Statistical Tests

```bash
python run_additional_analysis.py
```

This performs:
- Wilcoxon signed-rank tests (all model pairs)
- Effect size calculations (Cohen's d)
- Chronos-2 zero-shot vs fine-tuned comparison
- Error analysis by horizon and pageview level

Output: `artifacts/metrics/statistical_tests.csv`

### 5.4 Run Jupyter Notebooks

**Execute All Notebooks**:

Windows:
```cmd
run_notebooks.bat
```

Linux/Mac:
```bash
./run_notebooks.sh
```

**Or Run Individually**:
```bash
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_backtesting.ipynb
jupyter notebook notebooks/03_test_eval.ipynb
```

**Notebooks Overview**:
- `01_eda.ipynb`: Data exploration, seasonality analysis, train/val/test splits
- `02_backtesting.ipynb`: Validation results, model comparison, fold analysis
- `03_test_eval.ipynb`: Final test evaluation, calibration, error analysis

---

## 6. Configuration

### 6.1 Configuration File Structure

All hyperparameters are in `configs/default.yaml`:

```yaml
# Random seed for reproducibility
random_seed: 42

# Data configuration
data:
  page_title: "Bitcoin"           # Wikipedia page to fetch
  start_date: "2020-01-01"        # Start date
  end_date: "2024-12-31"          # End date
  cache_dir: "data"               # Cache directory

# Time series parameters
ts_params:
  frequency: "D"                  # Daily
  horizon: 30                     # Forecast horizon (days)
  seasonal_period: 7              # Weekly seasonality

# Model configurations
models:
  seasonal_naive:
    seasonal_period: 7

  ets:
    seasonal: "add"
    seasonal_periods: 7
    trend: "add"

  gradient_boosting:
    n_estimators: 100
    learning_rate: 0.05
    max_depth: 5
    lags: [1, 7, 14, 28]
    rolling_windows: [7, 28]

  chronos:
    model_name: "amazon/chronos-t5-base"
    device: "cuda"
    batch_size: 32
    num_samples: 20

  chronos_finetuned:
    model_name: "amazon/chronos-t5-base"
    device: "cuda"
    learning_rate: 0.00003
    num_epochs: 25
    batch_size: 4
    accumulation_steps: 8
    use_mixed_precision: true

# Evaluation metrics
metrics:
  point_forecast:
    - "mae"
    - "rmse"
    - "smape"
    - "mase"
  primary_metric: "mase_mean"

# Output paths
output:
  predictions_dir: "artifacts/predictions"
  metrics_dir: "artifacts/metrics"
  figures_dir: "artifacts/figures"
```

### 6.2 Common Configuration Changes

#### Change Wikipedia Page

```yaml
data:
  page_title: "Python_(programming_language)"  # Use underscores for spaces
```

#### Change Forecast Horizon

```yaml
ts_params:
  horizon: 14  # 2 weeks instead of 30 days
```

#### Change Seasonal Period

```yaml
ts_params:
  seasonal_period: 12  # Monthly seasonality instead of weekly
```

#### Fine-Tuning Hyperparameters

```yaml
chronos_finetuned:
  num_epochs: 30              # More epochs (slower)
  learning_rate: 0.00005      # Higher learning rate
  batch_size: 2               # Smaller batch (if OOM)
  early_stopping_patience: 3  # Stop earlier
```

#### Use CPU Instead of GPU

```yaml
chronos:
  device: "cpu"  # Change from "cuda" to "cpu"

chronos_finetuned:
  device: "cpu"
```

### 6.3 Advanced Configuration

#### Gradient Boosting Feature Engineering

```yaml
gradient_boosting:
  lags: [1, 2, 7, 14, 21, 28]      # Add more lags
  rolling_windows: [3, 7, 14, 28]  # Add more rolling windows
  use_day_of_week: true            # Include day-of-week features
```

#### Backtesting Configuration

```yaml
backtesting:
  n_folds: 3      # Fewer folds (faster)
  method: "expanding"  # or "rolling"
```

#### Data Preprocessing

```yaml
preprocessing:
  outlier_method: "z_score"        # Use z-score instead of winsorization
  outlier_quantiles: [0.05, 0.95]  # Different quantiles
  missing_method: "interpolate"    # Linear interpolation
```

---

## 7. Understanding the Results

### 7.1 Metrics Explained

#### MASE (Mean Absolute Scaled Error) - PRIMARY METRIC
- **Range**: 0 to ∞ (lower is better)
- **Interpretation**:
  - MASE < 1: Better than seasonal naive baseline
  - MASE = 1: Equal to seasonal naive baseline
  - MASE > 1: Worse than seasonal naive baseline
- **Example**: MASE = 0.373 means 37.3% of baseline error

#### MAE (Mean Absolute Error)
- **Range**: 0 to ∞ (lower is better)
- **Unit**: Same as data (pageviews)
- **Interpretation**: Average absolute deviation from true values
- **Example**: MAE = 946 means on average, predictions are off by 946 pageviews

#### RMSE (Root Mean Squared Error)
- **Range**: 0 to ∞ (lower is better)
- **Unit**: Same as data (pageviews)
- **Interpretation**: Penalizes large errors more than MAE
- **Example**: RMSE = 1316 means typical prediction error is 1316 pageviews

#### sMAPE (Symmetric Mean Absolute Percentage Error)
- **Range**: 0% to 100% (lower is better)
- **Interpretation**: Percentage error, symmetric treatment of over/under predictions
- **Example**: sMAPE = 15.7% means typical error is 15.7% of actual value

#### Pinball Loss (Quantile Forecasts)
- **Range**: 0 to ∞ (lower is better)
- **Interpretation**: Quality of quantile forecasts (τ = 0.1, 0.5, 0.9)
- **Example**: Pinball loss at τ=0.1 measures lower bound quality

#### Coverage (Prediction Intervals)
- **Range**: 0% to 100%
- **Target**: Should match nominal level (e.g., 80% intervals should cover 80% of data)
- **Example**: 49% coverage for 80% intervals = undercoverage

### 7.2 Reading the Results Files

#### Model Metrics (JSON)

`artifacts/metrics/chronos_finetuned_metrics.json`:
```json
{
  "mae_mean": 946.47,      // Average MAE across 5 folds
  "mae_std": 222.54,       // Standard deviation of MAE
  "mase_mean": 0.373,      // Average MASE (PRIMARY METRIC)
  "mase_std": 0.088,       // Consistency across folds
  "coverage_80_mean": 0.493 // Prediction interval coverage
}
```

**Interpretation**:
- `_mean`: Average performance across 5 folds
- `_std`: Consistency (lower std = more reliable)

#### Results Summary (YAML)

`artifacts/results_summary.yaml`:
```yaml
model_rankings:
  1: gradient_boosting (MASE: 0.344)
  2: chronos_finetuned (MASE: 0.373)
  3: chronos (MASE: 0.394)
  4: seasonal_naive (MASE: 0.397)
  5: ets (MASE: 0.508)

statistical_tests:
  chronos_finetuned_vs_zero_shot:
    p_value: 0.028
    significant: true
    improvement: 5.3%
```

#### Statistical Tests (CSV)

`artifacts/metrics/statistical_tests.csv`:
```csv
model_1,model_2,test,p_value,significant,effect_size
gradient_boosting,chronos_finetuned,wilcoxon,0.041,True,0.68
chronos_finetuned,chronos,wilcoxon,0.028,True,0.42
```

**Interpretation**:
- p_value < 0.05: Statistically significant difference
- effect_size > 0.5: Large practical difference

### 7.3 Interpreting Visualizations

#### Architecture Comparison
- Shows 5 model architectures side-by-side
- Compare complexity, features, outputs

#### Radar Chart
- Multi-metric performance comparison
- Smaller area = better performance
- Identifies strengths/weaknesses

#### Performance Breakdown
- Validation vs test performance
- Checks for overfitting (if test >> validation)

#### Error by Horizon
- Shows how error increases from h=1 to h=30
- Identifies model degradation patterns

#### Fold Consistency (Box Plot)
- Shows performance variance across 5 folds
- Tighter boxes = more consistent model

#### Statistical Significance Heatmap
- Color-coded p-values between all model pairs
- Red = significant difference, Blue = no difference

---

## 8. Troubleshooting

### 8.1 Installation Issues

**Problem**: `conda env create` fails with conflict errors
```
Solution:
1. Update conda: conda update -n base -c defaults conda
2. Clear cache: conda clean --all
3. Retry: conda env create -f environment.yml
```

**Problem**: PyTorch CUDA not available after installation
```
Solution:
1. Check GPU: nvidia-smi
2. Check CUDA version: nvidia-smi (top right)
3. Install matching PyTorch:
   - CUDA 12.1: conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
   - CUDA 11.8: conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
4. Verify: python -c "import torch; print(torch.cuda.is_available())"
```

**Problem**: Import errors after installation
```
Solution:
1. Activate environment: conda activate ts-chronos-gpu
2. Verify: which python (should show venv/conda path)
3. Reinstall problematic package: pip install --force-reinstall <package>
```

### 8.2 Runtime Errors

**Problem**: CUDA Out of Memory (OOM)
```
Error: RuntimeError: CUDA out of memory
Solution:
1. Reduce batch size in configs/default.yaml:
   chronos_finetuned:
     batch_size: 2  # Was 4
2. Or use CPU:
   chronos_finetuned:
     device: "cpu"
```

**Problem**: Wikipedia API timeout
```
Error: ConnectionError or Timeout
Solution:
1. Check internet connection
2. Try different page: data: page_title: "Python_(programming_language)"
3. Increase timeout in src/data_loader.py
4. Use cached data if available (check data/ directory)
```

**Problem**: Model training very slow
```
Solution:
1. Check GPU usage: nvidia-smi
2. Reduce number of folds: python run_pipeline.py --n_folds 3
3. Skip fine-tuning: python run_pipeline.py --skip_models chronos_finetuned
4. Use smaller model: model_name: "amazon/chronos-t5-small"
```

**Problem**: Visualization fails to generate
```
Error: KeyError or AttributeError in visualize_model_comparison.py
Solution:
1. Ensure all models completed: ls artifacts/metrics/
2. Check for errors in pipeline: cat artifacts/logs/pipeline.log
3. Run visualization with debug: python visualize_model_comparison.py --verbose
```

### 8.3 Data Issues

**Problem**: Not enough data for backtesting
```
Error: ValueError: Insufficient data for 5 folds
Solution:
1. Use longer date range:
   data:
     start_date: "2015-01-01"  # More years
2. Or reduce folds:
   backtesting:
     n_folds: 3
```

**Problem**: High missing values
```
Warning: >10% missing values after forward fill
Solution:
1. Increase fill limit:
   preprocessing:
     forward_fill_limit: 5  # Was 2
2. Or use interpolation:
   preprocessing:
     missing_method: "interpolate"
```

### 8.4 Performance Issues

**Problem**: Models perform poorly (MASE > 1.0)
```
Solution:
1. Check seasonality: Plot ACF/PACF in notebooks/01_eda.ipynb
2. Adjust seasonal_period:
   ts_params:
     seasonal_period: 12  # Try different periods
3. Add more features (GB):
   gradient_boosting:
     lags: [1, 2, 3, 7, 14, 28]
4. Increase fine-tuning epochs:
   chronos_finetuned:
     num_epochs: 40
```

**Problem**: Fine-tuned model not better than zero-shot
```
Solution:
1. Check training loss: Ensure it's decreasing
2. Increase training time:
   chronos_finetuned:
     num_epochs: 40
     early_stopping_patience: 10
3. Adjust learning rate:
   chronos_finetuned:
     learning_rate: 0.00005  # Higher
4. Ensure sufficient data: Need 500+ training points
```

---

## 9. Advanced Usage

### 9.1 Custom Models

Add your own forecasting model:

**Step 1**: Create model file `src/my_model.py`:
```python
class MyCustomModel:
    def __init__(self, **kwargs):
        self.config = kwargs

    def fit(self, train_df):
        """Train on training data."""
        # Your training logic
        pass

    def predict(self, horizon):
        """Generate point forecasts."""
        # Your prediction logic
        return predictions

    def predict_quantiles(self, horizon, quantiles):
        """Generate probabilistic forecasts."""
        # Your quantile prediction logic
        return {q: predictions for q in quantiles}

def create_my_model(config):
    """Factory function."""
    params = config['models']['my_model']
    return (MyCustomModel, params)
```

**Step 2**: Add to `run_pipeline.py`:
```python
from src.my_model import create_my_model

# Add to models dictionary
models['my_model'] = create_my_model(config.config)
```

**Step 3**: Add config in `configs/default.yaml`:
```yaml
models:
  my_model:
    param1: value1
    param2: value2
```

**Step 4**: Run pipeline:
```bash
python run_pipeline.py
```

### 9.2 Custom Datasets

Forecast your own time series:

**Step 1**: Prepare data as Pandas DataFrame:
```python
import pandas as pd

# Your data must have columns: ['ds', 'y']
df = pd.DataFrame({
    'ds': pd.date_range('2020-01-01', periods=1000, freq='D'),
    'y': your_values  # Your time series values
})

# Save to parquet
df.to_parquet('data/custom_data.parquet')
```

**Step 2**: Modify `src/data_loader.py`:
```python
def load_custom_data(file_path):
    df = pd.read_parquet(file_path)
    # Ensure proper datetime format
    df['ds'] = pd.to_datetime(df['ds'])
    return df
```

**Step 3**: Update pipeline to use custom data:
```python
# In run_pipeline.py
from src.data_loader import load_custom_data

df = load_custom_data('data/custom_data.parquet')
```

### 9.3 Hyperparameter Tuning

Optimize model parameters:

**Step 1**: Create grid search script `tune_hyperparameters.py`:
```python
import itertools
from run_pipeline import run_pipeline

param_grid = {
    'learning_rate': [1e-5, 3e-5, 5e-5],
    'num_epochs': [20, 30, 40],
    'batch_size': [2, 4, 8]
}

best_mase = float('inf')
best_params = None

for params in itertools.product(*param_grid.values()):
    config = dict(zip(param_grid.keys(), params))

    # Run pipeline with these params
    results = run_pipeline(config)

    if results['mase'] < best_mase:
        best_mase = results['mase']
        best_params = config

print(f"Best MASE: {best_mase}")
print(f"Best params: {best_params}")
```

**Step 2**: Run tuning:
```bash
python tune_hyperparameters.py
```

### 9.4 Ensemble Models

Combine multiple models:

**Step 1**: Create ensemble script `src/ensemble.py`:
```python
import numpy as np

def simple_average_ensemble(predictions_dict):
    """Average predictions from multiple models."""
    predictions = np.array(list(predictions_dict.values()))
    return np.mean(predictions, axis=0)

def weighted_ensemble(predictions_dict, weights):
    """Weighted average based on validation performance."""
    predictions = np.array(list(predictions_dict.values()))
    weights = np.array(weights) / np.sum(weights)
    return np.sum(predictions * weights[:, None], axis=0)
```

**Step 2**: Use in pipeline:
```python
from src.ensemble import weighted_ensemble

# Get predictions from all models
preds = {
    'gb': gb_model.predict(30),
    'chronos_ft': chronos_ft_model.predict(30)
}

# Ensemble with weights based on validation MASE
weights = [1/0.344, 1/0.373]  # Inverse of MASE
ensemble_pred = weighted_ensemble(preds, weights)
```

---

## 10. FAQ

### 10.1 General Questions

**Q: Do I need a GPU?**

A: No, but highly recommended. Without GPU:
- Chronos models are 15-20x slower
- Total runtime: ~1-2 hours vs ~30-50 minutes

**Q: Can I use a different Wikipedia page?**

A: Yes! Change in `configs/default.yaml`:
```yaml
data:
  page_title: "Your_Page_Name"  # Use underscores for spaces
```

**Q: How much data do I need?**

A: Minimum requirements:
- Seasonal Naive: 1 seasonal cycle (7 days for weekly)
- ETS: 2+ seasonal cycles (~14+ days)
- Gradient Boosting: 50+ observations
- Chronos Zero-Shot: 50+ observations
- Chronos Fine-Tuned: 500+ observations recommended

**Q: Can I forecast other variables (not pageviews)?**

A: Yes! The framework works for any univariate time series. Just prepare data with columns `['ds', 'y']`.

### 10.2 Model-Specific Questions

**Q: Why does Gradient Boosting perform best?**

A: Because:
1. Explicitly captures weekly patterns (lag_7 feature)
2. Domain-optimized features
3. Lower model capacity (less overfitting)

**Q: When should I fine-tune Chronos-2?**

A: Fine-tune when:
- ✓ You have 500+ training observations
- ✓ You can afford ~30-50 minutes
- ✓ Series has learnable patterns
- ✓ You want probabilistic forecasts

**Q: Can I fine-tune with less than 4GB GPU?**

A: Yes, with adjustments:
```yaml
chronos_finetuned:
  batch_size: 1           # Smallest batch
  accumulation_steps: 16  # Keep effective batch=32
  gradient_checkpointing: true  # Enable checkpointing
```

**Q: How do I interpret prediction intervals?**

A: Example for 80% interval:
- Lower bound (10th percentile): 90% of future values should be above this
- Median (50th percentile): Most likely forecast
- Upper bound (90th percentile): 90% of future values should be below this

If 80% interval covers only 49% of actual values (undercoverage), intervals are too narrow.

### 10.3 Technical Questions

**Q: Why is fine-tuning taking longer than 50 minutes?**

A: Possible reasons:
1. CPU mode (15-20x slower) - check GPU: `nvidia-smi`
2. Large batch size - reduce in config
3. Too many epochs - reduce to 20
4. Slow disk I/O - use SSD

**Q: Can I resume interrupted training?**

A: Yes! Checkpoints are saved:
```python
# Load checkpoint and resume
model = ChronosFineTunedModel.load_from_checkpoint(
    'artifacts/checkpoints/chronos_finetuned_fold_0.pt'
)
```

**Q: How do I export results to Excel?**

A:
```bash
python -c "import pandas as pd; pd.read_csv('artifacts/metrics/statistical_tests.csv').to_excel('results.xlsx')"
```

**Q: Can I deploy models to production?**

A: Yes! Save trained models:
```python
# Save
model.save('production_model.pkl')

# Load and predict
model = load_model('production_model.pkl')
forecast = model.predict(horizon=30)
```

### 10.4 Results Questions

**Q: My MASE is > 1.0. Is that bad?**

A: MASE > 1.0 means model is worse than seasonal naive baseline. Possible fixes:
1. Check seasonal period (try different m values)
2. Increase model complexity (more features, longer training)
3. Verify data quality (outliers, missing values)

**Q: Why is test performance worse than validation?**

A: Common reasons:
1. Distribution shift (test period has different patterns)
2. Overfitting to validation folds
3. Forecast horizon too long (try shorter horizons)

**Q: How do I know if my model is good enough?**

A: Compare to baselines:
- MASE < 1.0: Better than seasonal naive ✓
- MASE < 0.5: Strong performance ✓✓
- MASE < 0.3: Excellent performance ✓✓✓

For Bitcoin pageviews:
- GB: 0.344 (excellent)
- Chronos FT: 0.373 (very good)

---

## Appendix A: Command Reference

### A.1 Main Commands

| Command | Purpose | Time |
|---------|---------|------|
| `python run_pipeline.py` | Run complete pipeline | 35-55 min (GPU) |
| `python visualize_model_comparison.py` | Generate plots | ~10 sec |
| `python run_additional_analysis.py` | Statistical tests | ~5 sec |
| `python test_setup.py` | Verify installation | ~5 sec |
| `run_end_to_end.bat/sh` | One-command execution | 35-55 min (GPU) |

### A.2 Environment Commands

| Command | Purpose |
|---------|---------|
| `conda activate ts-chronos-gpu` | Activate environment |
| `conda deactivate` | Deactivate environment |
| `conda env list` | List all environments |
| `conda env remove -n ts-chronos-gpu` | Remove environment |
| `pip list` | Show installed packages |

### A.3 Utility Commands

| Command | Purpose |
|---------|---------|
| `nvidia-smi` | Check GPU status |
| `python -c "import torch; print(torch.cuda.is_available())"` | Check CUDA |
| `ls artifacts/metrics/` | List generated metrics |
| `cat artifacts/results_summary.yaml` | View results summary |

---

## Appendix B: File Structure Reference

```
final/
├── README.md                      # Project overview
├── USER_MANUAL.md                 # This file
├── requirements.txt               # Python dependencies
├── environment.yml                # Conda environment
├── test_setup.py                  # Installation verification
│
├── configs/
│   └── default.yaml               # All hyperparameters
│
├── src/                           # Source code
│   ├── config.py                  # Configuration loader
│   ├── data_loader.py             # Data fetching
│   ├── preprocess.py              # Data preprocessing
│   ├── features.py                # Feature engineering
│   ├── baselines.py               # Baseline models
│   ├── chronos_model.py           # Chronos zero-shot
│   ├── chronos_finetuned.py       # Chronos fine-tuned
│   ├── backtesting.py             # Cross-validation
│   ├── metrics.py                 # Evaluation metrics
│   ├── stats_tests.py             # Statistical tests
│   ├── plots.py                   # Visualizations
│   └── utils.py                   # Utilities
│
├── data/                          # Cached data
│   ├── train.parquet
│   ├── val.parquet
│   └── test.parquet
│
├── artifacts/                     # Generated results
│   ├── predictions/               # Model predictions
│   ├── metrics/                   # Evaluation metrics
│   ├── figures/                   # Visualizations
│   ├── checkpoints/               # Fine-tuned weights
│   └── results_summary.yaml       # Summary
│
├── docs/                          # Documentation
│   ├── PROJECT_REPORT.md          # Technical report
│   ├── PRESENTATION_SLIDES.md     # Slides
│   └── model_card.md              # Model documentation
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_eda.ipynb
│   ├── 02_backtesting.ipynb
│   └── 03_test_eval.ipynb
│
└── Execution scripts
    ├── run_pipeline.py            # Main pipeline
    ├── run_additional_analysis.py # Extra analysis
    ├── visualize_model_comparison.py # Visualizations
    ├── run_end_to_end.bat         # Windows runner
    └── run_end_to_end.sh          # Linux/Mac runner
```

---

## Appendix C: Getting Help

### C.1 Documentation

1. **README.md**: Project overview and quick start
2. **USER_MANUAL.md**: This comprehensive guide
3. **docs/PROJECT_REPORT.md**: Technical report with methodology
4. **docs/model_card.md**: Chronos-2 model documentation
5. **Code docstrings**: In-line documentation in source files

### C.2 Support Channels

- **GitHub Issues**: https://github.com/Vipproplayerone1/ts-chronos-gpu/issues
- **Email**: nhan.bui210409@vnuk.edu.vn
- **Documentation**: Check README and this manual first

### C.3 Reporting Issues

When reporting issues, please include:
1. Error message (full traceback)
2. Command that caused the error
3. Environment info: `python test_setup.py`
4. Config file: `cat configs/default.yaml`
5. Log file: `cat artifacts/logs/pipeline.log` (if exists)

---

**End of User Manual**

*For additional help, refer to README.md or contact: nhan.bui210409@vnuk.edu.vn*
