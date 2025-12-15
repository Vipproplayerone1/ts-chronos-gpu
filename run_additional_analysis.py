"""
Additional analysis script to generate missing plots and analysis.

This script:
1. Computes error by horizon for all models
2. Computes metrics by fold for backtesting
3. Generates feature importance for Gradient Boosting
4. Performs error dissection by level
5. Generates all missing plots
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from src.config import load_config
from src.metrics import (
    compute_point_metrics,
    compute_metrics_by_horizon
)
from src.plots import (
    plot_error_by_horizon,
    plot_metrics_by_fold,
    plot_feature_importance
)
from src.baselines import GradientBoostingModel

print("="*100)
print(" ADDITIONAL ANALYSIS - Missing Plots & Analysis")
print("="*100)

# Load configuration
config = load_config("configs/default.yaml")

# Load data splits
print("\n[1/6] Loading data splits...")
train_df = pd.read_parquet("data/train.parquet")
val_df = pd.read_parquet("data/val.parquet")
test_df = pd.read_parquet("data/test.parquet")
print(f"[OK] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Load backtest results
print("\n[2/6] Loading backtest results...")
backtest_results = {}
for model_name in ['seasonal_naive', 'ets', 'gradient_boosting', 'chronos']:
    parquet_path = f"artifacts/predictions/{model_name}_backtest.parquet"
    if Path(parquet_path).exists():
        df = pd.read_parquet(parquet_path)

        # Reconstruct results structure
        backtest_results[model_name] = {'folds': []}

        # Group by fold
        for fold_num in df['fold'].unique():
            fold_df = df[df['fold'] == fold_num].sort_values('horizon')

            fold_result = {
                'fold': int(fold_num),
                'y_true': fold_df['y_true'].values,
                'y_pred': fold_df['y_pred'].values
            }

            # Add quantile predictions if available
            quantile_cols = [col for col in fold_df.columns if col.startswith('q_')]
            if quantile_cols:
                quantile_predictions = {}
                for col in quantile_cols:
                    q_value = float(col.split('_')[1])
                    quantile_predictions[q_value] = fold_df[col].values
                fold_result['quantile_predictions'] = quantile_predictions

            backtest_results[model_name]['folds'].append(fold_result)

print(f"[OK] Loaded backtest results for {len(backtest_results)} models")

# 1. COMPUTE ERROR BY HORIZON
print("\n[3/6] Computing error by horizon...")
error_by_horizon_all_models = {}

for model_name, results in backtest_results.items():
    horizon_errors = {}

    # Aggregate across folds
    for fold_result in results['folds']:
        if 'error' in fold_result:
            continue

        y_true = fold_result['y_true']
        y_pred = fold_result['y_pred']

        # Compute error for each horizon step
        for h in range(len(y_true)):
            if h not in horizon_errors:
                horizon_errors[h] = []

            error = abs(y_true[h] - y_pred[h])  # Absolute error
            horizon_errors[h].append(error)

    # Average across folds
    avg_horizon_errors = {h+1: np.mean(errors) for h, errors in horizon_errors.items()}
    error_by_horizon_all_models[model_name] = avg_horizon_errors

print(f"[OK] Computed error by horizon for {len(error_by_horizon_all_models)} models")

# Generate error by horizon plot
plot_error_by_horizon(
    error_by_horizon=error_by_horizon_all_models,
    metric_name="MAE",
    save_path="artifacts/figures/error_by_horizon.png"
)
print("[OK] Generated error_by_horizon.png")

# 2. COMPUTE METRICS BY FOLD
print("\n[4/6] Computing metrics by fold...")
metrics_by_fold_all_models = {}

for model_name, results in backtest_results.items():
    fold_mase = []

    for fold_result in results['folds']:
        if 'error' in fold_result:
            continue

        # Compute MASE for this fold
        metrics = compute_point_metrics(
            y_true=fold_result['y_true'],
            y_pred=fold_result['y_pred'],
            y_train=train_df['y'].values,
            seasonal_period=config.seasonal_period
        )
        fold_mase.append(metrics['mase'])

    metrics_by_fold_all_models[model_name] = fold_mase

print(f"[OK] Computed MASE by fold for {len(metrics_by_fold_all_models)} models")

# Generate metrics by fold plot
plot_metrics_by_fold(
    metrics_by_fold=metrics_by_fold_all_models,
    metric_name="MASE",
    save_path="artifacts/figures/mase_by_fold.png"
)
print("[OK] Generated mase_by_fold.png")

# 3. FEATURE IMPORTANCE FOR GRADIENT BOOSTING
print("\n[5/6] Generating feature importance for Gradient Boosting...")

# Retrain GB on full train+val to get feature importance
full_train = pd.concat([train_df, val_df], ignore_index=True)

from src.features import TimeSeriesFeatureEngineer

gb_params = config['models']['gradient_boosting']
feature_engineer = TimeSeriesFeatureEngineer(
    lags=gb_params['lags'],
    rolling_windows=gb_params['rolling_windows'],
    use_day_of_week=gb_params['use_day_of_week']
)

# Create features
full_train_features = feature_engineer.create_features(full_train)

# Train model
gb_model = GradientBoostingModel(
    n_estimators=gb_params['n_estimators'],
    max_depth=gb_params['max_depth'],
    learning_rate=gb_params['learning_rate'],
    lags=gb_params['lags'],
    rolling_windows=gb_params['rolling_windows'],
    use_day_of_week=gb_params['use_day_of_week']
)
gb_model.fit(full_train)

# Get feature importance
if hasattr(gb_model.model, 'feature_importances_'):
    feature_names = full_train_features.drop(columns=['ds', 'y']).columns.tolist()
    importances = gb_model.model.feature_importances_

    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Plot
    plot_feature_importance(
        importance_df=importance_df,
        top_n=15,
        save_path="artifacts/figures/feature_importance.png"
    )
    print("[OK] Generated feature_importance.png")
else:
    print("[WARN] Model does not have feature_importances_ attribute")

# 4. ERROR DISSECTION BY LEVEL
print("\n[6/6] Performing error dissection by level...")

# Load test predictions
test_predictions = {}
for model_name in ['seasonal_naive', 'ets', 'gradient_boosting', 'chronos']:
    # Use the median prediction from backtest results on test set
    # For simplicity, we'll compute fresh predictions
    pass

# Compute error by level using validation results
level_analysis = {}

for model_name, results in backtest_results.items():
    all_y_true = []
    all_y_pred = []

    for fold_result in results['folds']:
        if 'error' in fold_result:
            continue
        all_y_true.extend(fold_result['y_true'])
        all_y_pred.extend(fold_result['y_pred'])

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    # Split into tertiles (low, medium, high)
    tertiles = np.percentile(all_y_true, [33.33, 66.67])

    low_mask = all_y_true <= tertiles[0]
    med_mask = (all_y_true > tertiles[0]) & (all_y_true <= tertiles[1])
    high_mask = all_y_true > tertiles[1]

    # Compute MAE for each level
    mae_low = np.mean(np.abs(all_y_true[low_mask] - all_y_pred[low_mask]))
    mae_med = np.mean(np.abs(all_y_true[med_mask] - all_y_pred[med_mask]))
    mae_high = np.mean(np.abs(all_y_true[high_mask] - all_y_pred[high_mask]))

    level_analysis[model_name] = {
        'low': mae_low,
        'medium': mae_med,
        'high': mae_high
    }

print(f"[OK] Computed error by level for {len(level_analysis)} models")

# Plot error by level
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(3)
width = 0.2
colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']

for idx, (model_name, levels) in enumerate(level_analysis.items()):
    values = [levels['low'], levels['medium'], levels['high']]
    ax.bar(x + idx*width, values, width, label=model_name, color=colors[idx % len(colors)])

ax.set_xlabel('Pageview Level', fontsize=12)
ax.set_ylabel('MAE', fontsize=12)
ax.set_title('Error Dissection by Pageview Level', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(['Low\n(<33rd pctile)', 'Medium\n(33-66th pctile)', 'High\n(>66th pctile)'])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("artifacts/figures/error_by_level.png", dpi=300, bbox_inches='tight')
print("[OK] Generated error_by_level.png")
plt.close()

# Save analysis results
print("\n[7/6] Saving analysis results...")

# Save error by horizon
error_by_horizon_df = pd.DataFrame(error_by_horizon_all_models)
error_by_horizon_df.to_csv("artifacts/metrics/error_by_horizon.csv")
print("[OK] Saved error_by_horizon.csv")

# Save metrics by fold
metrics_by_fold_df = pd.DataFrame(metrics_by_fold_all_models)
metrics_by_fold_df.to_csv("artifacts/metrics/metrics_by_fold.csv")
print("[OK] Saved metrics_by_fold.csv")

# Save error by level
level_analysis_df = pd.DataFrame(level_analysis).T
level_analysis_df.to_csv("artifacts/metrics/error_by_level.csv")
print("[OK] Saved error_by_level.csv")

print("\n" + "="*100)
print(" ADDITIONAL ANALYSIS COMPLETE!")
print("="*100)
print("\nGenerated Plots:")
print("  - artifacts/figures/error_by_horizon.png")
print("  - artifacts/figures/mase_by_fold.png")
print("  - artifacts/figures/feature_importance.png")
print("  - artifacts/figures/error_by_level.png")
print("\nGenerated Data:")
print("  - artifacts/metrics/error_by_horizon.csv")
print("  - artifacts/metrics/metrics_by_fold.csv")
print("  - artifacts/metrics/error_by_level.csv")
