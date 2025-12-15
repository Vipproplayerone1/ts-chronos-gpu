"""
Main execution script for end-to-end time series forecasting pipeline.

This script:
1. Loads and preprocesses Wikipedia pageviews data
2. Performs EDA and seasonality analysis
3. Runs rolling-origin backtesting for all models
4. Evaluates models on test set
5. Performs statistical significance testing
6. Generates all required plots and results

Usage:
    python run_pipeline.py --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from datetime import datetime

from src.config import load_config
from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.utils import (
    create_train_val_test_split,
    save_predictions,
    save_metrics,
    save_yaml,
    get_library_versions,
    print_library_versions
)
from src.baselines import create_baseline_models
from src.chronos_model import create_chronos_model
from src.backtesting import backtest_multiple_models, save_backtest_results
from src.metrics import (
    compute_point_metrics,
    compute_probabilistic_metrics,
    aggregate_fold_metrics,
    print_metrics_summary,
    rank_models
)
from src.stats_tests import compare_all_models, print_statistical_test_results
from src.plots import (
    plot_time_series,
    plot_train_val_test_split,
    plot_forecasts_with_history,
    plot_error_by_horizon,
    plot_calibration_curve,
    plot_seasonality_decomposition
)


def main(config_path: str = "configs/default.yaml"):
    """Run the complete pipeline."""

    print("="*100)
    print(" TIME SERIES FORECASTING PIPELINE - Wikipedia Pageviews with Chronos-2")
    print("="*100)

    # 1. Load configuration
    print("\n[1/9] Loading configuration...")
    config = load_config(config_path)
    config.create_output_dirs()

    print(f"[OK] Random seed: {config.random_seed}")
    print(f"[OK] Page: {config.page_title}")
    print(f"[OK] Horizon: {config.horizon} days")
    print(f"[OK] Seasonal period: {config.seasonal_period} days")

    # Print library versions for reproducibility
    print_library_versions()

    # 2. Load data
    print("\n[2/9] Loading data from Wikipedia Pageviews API...")
    raw_df = load_data(
        page_title=config['data']['page_title'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        cache_dir=config['data']['cache_dir']
    )

    print(f"[OK] Loaded {len(raw_df)} records")
    print(f"[OK] Date range: {raw_df['ds'].min()} to {raw_df['ds'].max()}")

    # Verify minimum length
    if len(raw_df) < config['data']['min_length']:
        raise ValueError(
            f"Time series too short: {len(raw_df)} < {config['data']['min_length']} required"
        )

    # 3. Preprocess data
    print("\n[3/9] Preprocessing data...")
    clean_df, preprocessor = preprocess_data(raw_df, config.config, fit=True)

    print(f"[OK] Clean data: {len(clean_df)} records")

    # 4. Create train/val/test splits
    print("\n[4/9] Creating train/val/test splits...")
    train_df, val_df, test_df = create_train_val_test_split(
        clean_df,
        train_ratio=config['splits']['train_ratio'],
        val_ratio=config['splits']['val_ratio'],
        test_ratio=config['splits']['test_ratio']
    )

    # Save splits
    train_df.to_parquet("data/train.parquet", index=False)
    val_df.to_parquet("data/val.parquet", index=False)
    test_df.to_parquet("data/test.parquet", index=False)
    print("[OK] Saved splits to data/")

    # Plot splits
    plot_train_val_test_split(
        train_df, val_df, test_df,
        save_path="artifacts/figures/train_val_test_split.png"
    )

    # 5. Run backtesting on validation set
    print("\n[5/9] Running rolling-origin backtesting...")

    # Combine train and validation for backtesting
    backtest_data = pd.concat([train_df, val_df], ignore_index=True)

    # Create models
    models = create_baseline_models(config.config)

    # Add Chronos model
    models['chronos'] = create_chronos_model(config.config)

    print(f"[OK] Models to evaluate: {list(models.keys())}")

    # Run backtesting
    backtest_results = backtest_multiple_models(
        data=backtest_data,
        models=models,
        n_folds=config['backtesting']['n_folds'],
        horizon=config.horizon,
        method=config['backtesting']['method'],
        quantile_levels=config['metrics']['quantiles'],
        verbose=True
    )

    # Save backtest results
    save_backtest_results(backtest_results, output_dir=config['output']['predictions_dir'])

    # 6. Compute metrics from backtesting
    print("\n[6/9] Computing metrics from backtesting...")

    all_model_metrics = {}

    for model_name, results in backtest_results.items():
        fold_metrics = []

        for fold_result in results['folds']:
            if 'error' in fold_result:
                continue

            # Compute metrics for this fold
            metrics = compute_point_metrics(
                y_true=fold_result['y_true'],
                y_pred=fold_result['y_pred'],
                y_train=train_df['y'].values,
                seasonal_period=config.seasonal_period
            )

            # Add probabilistic metrics if available
            if 'quantile_predictions' in fold_result:
                prob_metrics = compute_probabilistic_metrics(
                    y_true=fold_result['y_true'],
                    quantile_predictions=fold_result['quantile_predictions']
                )
                metrics.update(prob_metrics)

            fold_metrics.append(metrics)

        # Aggregate across folds
        aggregated_metrics = aggregate_fold_metrics(fold_metrics)
        all_model_metrics[model_name] = aggregated_metrics

        # Save metrics
        save_metrics(aggregated_metrics, model_name, output_dir=config['output']['metrics_dir'])

    # Print metrics summary
    print_metrics_summary(all_model_metrics)

    # Rank models
    rankings = rank_models(all_model_metrics, primary_metric=config['metrics']['primary_metric'])
    print(f"\nModel Rankings (by {config['metrics']['primary_metric']}):")
    for rank, (model_name, score) in enumerate(rankings, 1):
        print(f"  {rank}. {model_name}: {score:.4f}")

    # 7. Statistical significance testing
    print("\n[7/9] Performing statistical significance tests...")

    # Find best baseline (non-chronos model)
    best_baseline = None
    for model_name, score in rankings:
        if model_name != 'chronos':
            best_baseline = model_name
            break

    if best_baseline is None:
        print("[WARN] No valid baseline models found for comparison. Skipping statistical tests.")
        comparison_df = None
    else:
        print(f"[OK] Best baseline: {best_baseline}")

        comparison_df = compare_all_models(
            backtest_results=backtest_results,
            baseline_model=best_baseline,
            test_type=config['stats_testing']['test_type'],
            alpha=config['stats_testing']['alpha']
        )

        print_statistical_test_results(comparison_df)

        # Save comparison results
        comparison_df.to_csv(
            f"{config['output']['metrics_dir']}/statistical_tests.csv",
            index=False
        )

    # 8. Final evaluation on test set
    print("\n[8/9] Final evaluation on test set...")

    test_predictions = {}
    test_quantile_predictions = {}

    for model_name, (model_class, model_params) in models.items():
        print(f"\nEvaluating {model_name} on test set...")

        # Train on train + val
        full_train_df = pd.concat([train_df, val_df], ignore_index=True)

        # Initialize and fit model
        model = model_class(**model_params)
        model.fit(full_train_df)

        # Predict
        if hasattr(model, 'predict_quantiles'):
            quantile_preds = model.predict_quantiles(
                horizon=len(test_df),
                quantiles=config['metrics']['quantiles']
            )
            test_predictions[model_name] = quantile_preds[0.5]  # Median as point forecast
            test_quantile_predictions[model_name] = quantile_preds
        else:
            test_predictions[model_name] = model.predict(horizon=len(test_df))

        print(f"[OK] {model_name} predictions generated")

    # Compute test metrics
    test_metrics = {}
    for model_name, preds in test_predictions.items():
        metrics = compute_point_metrics(
            y_true=test_df['y'].values,
            y_pred=preds,
            y_train=train_df['y'].values,
            seasonal_period=config.seasonal_period
        )

        if model_name in test_quantile_predictions:
            prob_metrics = compute_probabilistic_metrics(
                y_true=test_df['y'].values,
                quantile_predictions=test_quantile_predictions[model_name]
            )
            metrics.update(prob_metrics)

        test_metrics[model_name] = metrics

    print("\n=== Test Set Performance ===")
    print_metrics_summary(test_metrics)

    # Save test metrics
    save_yaml(test_metrics, f"{config['output']['metrics_dir']}/test_metrics.yaml")

    # 9. Generate plots
    print("\n[9/9] Generating visualizations...")

    # Plot forecasts
    plot_forecasts_with_history(
        history_df=pd.concat([train_df, val_df]).tail(100),
        test_df=test_df.head(config.horizon),
        predictions_dict={k: v[:config.horizon] for k, v in test_predictions.items()},
        save_path="artifacts/figures/test_forecasts.png"
    )

    # Plot seasonality decomposition
    plot_seasonality_decomposition(
        df=train_df,
        period=config.seasonal_period,
        save_path="artifacts/figures/seasonality_decomposition.png"
    )

    # Plot calibration for Chronos (if available)
    if 'chronos' in test_quantile_predictions:
        plot_calibration_curve(
            y_true=test_df['y'].values[:config.horizon],
            quantile_predictions={k: v[:config.horizon]
                                 for k, v in test_quantile_predictions['chronos'].items()},
            save_path="artifacts/figures/calibration_curve.png"
        )

    # Save final summary
    summary = {
        'execution_time': datetime.now().isoformat(),
        'config': config.config,
        'data_info': {
            'n_total': len(clean_df),
            'n_train': len(train_df),
            'n_val': len(val_df),
            'n_test': len(test_df),
            'date_range': f"{clean_df['ds'].min()} to {clean_df['ds'].max()}"
        },
        'validation_metrics': all_model_metrics,
        'test_metrics': test_metrics,
        'model_rankings': {name: float(score) for name, score in rankings},
        'library_versions': get_library_versions()
    }

    save_yaml(summary, config['output']['results_file'])

    print("\n" + "="*100)
    print(" PIPELINE COMPLETE!")
    print("="*100)
    print(f"\n[OK] Results saved to: {config['output']['results_file']}")
    print(f"[OK] Predictions saved to: {config['output']['predictions_dir']}")
    print(f"[OK] Metrics saved to: {config['output']['metrics_dir']}")
    print(f"[OK] Figures saved to: {config['output']['figures_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run time series forecasting pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )

    args = parser.parse_args()

    try:
        main(args.config)
    except Exception as e:
        print(f"\n[FAIL] Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
