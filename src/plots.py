"""Visualization functions for time series forecasting analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# Set default style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100


def plot_time_series(
    df: pd.DataFrame,
    title: str = "Time Series",
    xlabel: str = "Date",
    ylabel: str = "Value",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
):
    """
    Plot a time series.

    Args:
        df: DataFrame with columns ['ds', 'y']
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(df['ds'], df['y'], linewidth=1.5, color='#2E86AB')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved plot to {save_path}")

    plt.show()


def plot_train_val_test_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    figsize: Tuple[int, int] = (16, 6),
    save_path: Optional[str] = None
):
    """
    Plot train/validation/test splits.

    Args:
        train_df: Training data
        val_df: Validation data
        test_df: Test data
        figsize: Figure size
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(train_df['ds'], train_df['y'], label='Train', linewidth=1.5, color='#2E86AB')
    ax.plot(val_df['ds'], val_df['y'], label='Validation', linewidth=1.5, color='#A23B72')
    ax.plot(test_df['ds'], test_df['y'], label='Test', linewidth=1.5, color='#F18F01')

    ax.set_title('Train / Validation / Test Split', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved plot to {save_path}")

    plt.show()


def plot_forecasts_with_history(
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictions_dict: Dict[str, np.ndarray],
    figsize: Tuple[int, int] = (16, 8),
    save_path: Optional[str] = None
):
    """
    Plot forecasts from multiple models alongside historical data.

    Args:
        history_df: Historical data
        test_df: Test data with true values
        predictions_dict: Dictionary mapping model names to predictions
        figsize: Figure size
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot history (last 100 points for clarity)
    history_window = history_df.tail(100)
    ax.plot(history_window['ds'], history_window['y'],
            label='Historical', linewidth=2, color='#2E86AB', alpha=0.7)

    # Plot actual test values
    ax.plot(test_df['ds'], test_df['y'],
            label='Actual', linewidth=2.5, color='black', marker='o', markersize=4)

    # Plot predictions from each model
    colors = ['#A23B72', '#F18F01', '#06A77D', '#D00000', '#8338EC']

    for idx, (model_name, predictions) in enumerate(predictions_dict.items()):
        # Create dates for predictions
        pred_dates = test_df['ds'].iloc[:len(predictions)]

        ax.plot(pred_dates, predictions,
                label=model_name, linewidth=2, linestyle='--',
                marker='s', markersize=5, color=colors[idx % len(colors)])

    ax.set_title('Forecasts Comparison on Test Set', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved plot to {save_path}")

    plt.show()


def plot_metrics_by_fold(
    metrics_by_fold: Dict[str, List[float]],
    metric_name: str = "MAE",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
):
    """
    Plot metric values across folds for multiple models.

    Args:
        metrics_by_fold: Dictionary mapping model names to list of metric values per fold
        metric_name: Name of the metric
        figsize: Figure size
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    model_names = list(metrics_by_fold.keys())
    n_folds = len(metrics_by_fold[model_names[0]])
    x = np.arange(1, n_folds + 1)

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#D00000']

    for idx, (model_name, values) in enumerate(metrics_by_fold.items()):
        ax.plot(x, values, label=model_name, marker='o', linewidth=2,
                markersize=8, color=colors[idx % len(colors)])

    ax.set_title(f'{metric_name} across Backtesting Folds', fontsize=14, fontweight='bold')
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_xticks(x)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved plot to {save_path}")

    plt.show()


def plot_error_by_horizon(
    error_by_horizon: Dict[str, Dict[int, float]],
    metric_name: str = "MAE",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
):
    """
    Plot forecast error by horizon for multiple models.

    Args:
        error_by_horizon: Dictionary mapping model names to {horizon: error} dictionaries
        metric_name: Name of the metric
        figsize: Figure size
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#D00000']

    for idx, (model_name, horizon_errors) in enumerate(error_by_horizon.items()):
        horizons = sorted(horizon_errors.keys())
        errors = [horizon_errors[h] for h in horizons]

        ax.plot(horizons, errors, label=model_name, marker='o', linewidth=2.5,
                markersize=6, color=colors[idx % len(colors)])

    ax.set_title(f'{metric_name} by Forecast Horizon', fontsize=14, fontweight='bold')
    ax.set_xlabel('Forecast Horizon (days)', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved plot to {save_path}")

    plt.show()


def plot_calibration_curve(
    y_true: np.ndarray,
    quantile_predictions: Dict[float, np.ndarray],
    figsize: Tuple[int, int] = (10, 10),
    save_path: Optional[str] = None
):
    """
    Plot calibration curve for prediction intervals.

    Args:
        y_true: True values
        quantile_predictions: Dictionary mapping quantile levels to predictions
        figsize: Figure size
        save_path: Path to save figure
    """
    # Calculate empirical coverage for different nominal levels
    quantiles = sorted(quantile_predictions.keys())

    # Find pairs of symmetric quantiles
    nominal_levels = []
    empirical_coverages = []

    for i in range(len(quantiles) // 2):
        lower_q = quantiles[i]
        upper_q = quantiles[-(i+1)]

        if abs((upper_q - lower_q) - (1 - 2*lower_q)) < 0.01:  # Check if symmetric
            # Calculate coverage
            lower_bound = quantile_predictions[lower_q]
            upper_bound = quantile_predictions[upper_q]

            in_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
            empirical_coverage = np.mean(in_interval)

            nominal_level = upper_q - lower_q
            nominal_levels.append(nominal_level * 100)
            empirical_coverages.append(empirical_coverage * 100)

    if not nominal_levels:
        print("Warning: No symmetric quantile pairs found for calibration plot")
        return

    # Create calibration plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot perfect calibration line
    ax.plot([0, 100], [0, 100], 'k--', linewidth=2, label='Perfect Calibration', alpha=0.7)

    # Plot empirical calibration
    ax.plot(nominal_levels, empirical_coverages, 'o-', linewidth=3,
            markersize=10, color='#2E86AB', label='Empirical Coverage')

    # Fill area between perfect and empirical
    ax.fill_between(nominal_levels, nominal_levels, empirical_coverages,
                     alpha=0.2, color='#2E86AB')

    ax.set_title('Prediction Interval Calibration', fontsize=14, fontweight='bold')
    ax.set_xlabel('Nominal Coverage (%)', fontsize=12)
    ax.set_ylabel('Empirical Coverage (%)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved plot to {save_path}")

    plt.show()


def plot_residuals_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None
):
    """
    Plot residuals analysis (scatter, histogram, Q-Q plot).

    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Model name
        figsize: Figure size
        save_path: Path to save figure
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Residuals vs Predicted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5, color='#2E86AB')
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Predicted Values', fontsize=11)
    axes[0, 0].set_ylabel('Residuals', fontsize=11)
    axes[0, 0].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Residuals histogram
    axes[0, 1].hist(residuals, bins=30, color='#A23B72', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Residuals', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('Residuals Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Q-Q plot
    from scipy import stats as scipy_stats
    scipy_stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Residuals over time
    axes[1, 1].plot(residuals, color='#F18F01', linewidth=1.5)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Time Index', fontsize=11)
    axes[1, 1].set_ylabel('Residuals', fontsize=11)
    axes[1, 1].set_title('Residuals over Time', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f'Residuals Analysis: {model_name}', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved plot to {save_path}")

    plt.show()


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
):
    """
    Plot feature importance.

    Args:
        importance_df: DataFrame with columns ['feature', 'importance']
        top_n: Number of top features to show
        figsize: Figure size
        save_path: Path to save figure
    """
    # Select top N features
    plot_df = importance_df.head(top_n).sort_values('importance')

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(plot_df['feature'], plot_df['importance'], color='#2E86AB', edgecolor='black')
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved plot to {save_path}")

    plt.show()


def plot_seasonality_decomposition(
    df: pd.DataFrame,
    period: int = 7,
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None
):
    """
    Plot STL decomposition of time series.

    Args:
        df: DataFrame with columns ['ds', 'y']
        period: Seasonal period
        figsize: Figure size
        save_path: Path to save figure
    """
    from statsmodels.tsa.seasonal import STL

    # Perform STL decomposition
    stl = STL(df['y'].values, period=period, seasonal=13)
    result = stl.fit()

    # Create plots
    fig, axes = plt.subplots(4, 1, figsize=figsize)

    # Original
    axes[0].plot(df['ds'], df['y'], color='#2E86AB', linewidth=1.5)
    axes[0].set_ylabel('Original', fontsize=11)
    axes[0].set_title('STL Decomposition', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Trend
    axes[1].plot(df['ds'], result.trend, color='#A23B72', linewidth=1.5)
    axes[1].set_ylabel('Trend', fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # Seasonal
    axes[2].plot(df['ds'], result.seasonal, color='#F18F01', linewidth=1.5)
    axes[2].set_ylabel('Seasonal', fontsize=11)
    axes[2].grid(True, alpha=0.3)

    # Residual
    axes[3].plot(df['ds'], result.resid, color='#06A77D', linewidth=1.5)
    axes[3].set_ylabel('Residual', fontsize=11)
    axes[3].set_xlabel('Date', fontsize=11)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved plot to {save_path}")

    plt.show()


def create_all_plots(
    data: Dict,
    output_dir: str = "artifacts/figures"
):
    """
    Generate all required plots for the project.

    Args:
        data: Dictionary containing all necessary data
        output_dir: Output directory for saving plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("\n=== Generating All Plots ===")

    # Add plot generation calls here based on available data
    # This is a template that will be called from main execution script

    print(f"[OK] All plots saved to {output_dir}")
