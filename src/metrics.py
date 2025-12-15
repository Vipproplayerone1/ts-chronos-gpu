"""Evaluation metrics for time series forecasting."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAE value
    """
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        RMSE value
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error (sMAPE).

    Formula: 100 * mean(2 * |y_true - y_pred| / (|y_true| + |y_pred|))

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        sMAPE value (in percentage, 0-100)
    """
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

    # Handle division by zero
    mask = denominator != 0
    smape_values = np.zeros_like(numerator)
    smape_values[mask] = numerator[mask] / denominator[mask]

    return 100 * np.mean(smape_values)


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonal_period: int = 1
) -> float:
    """
    Mean Absolute Scaled Error (MASE).

    Formula: MAE(y_true, y_pred) / MAE_naive
    where MAE_naive is the MAE of a seasonal naive forecast on the training data.

    Args:
        y_true: True values (test set)
        y_pred: Predicted values
        y_train: Training set values (for computing scaling factor)
        seasonal_period: Seasonal period for naive forecast (m)

    Returns:
        MASE value
    """
    # Compute MAE of predictions
    mae_pred = mae(y_true, y_pred)

    # Compute MAE of seasonal naive forecast on training data
    # Seasonal naive: y_t = y_{t-m}
    if len(y_train) <= seasonal_period:
        # Fallback to simple naive if not enough data
        naive_errors = np.abs(np.diff(y_train))
    else:
        naive_forecast = y_train[:-seasonal_period]
        naive_actual = y_train[seasonal_period:]
        naive_errors = np.abs(naive_actual - naive_forecast)

    mae_naive = np.mean(naive_errors)

    # Avoid division by zero
    if mae_naive == 0:
        return np.inf if mae_pred > 0 else 0.0

    return mae_pred / mae_naive


def pinball_loss(
    y_true: np.ndarray,
    y_pred_quantile: np.ndarray,
    quantile: float
) -> float:
    """
    Pinball loss for quantile forecasts.

    Formula: mean(rho_tau(y_true - y_pred_quantile))
    where rho_tau(u) = u * (tau - I(u < 0))

    Args:
        y_true: True values
        y_pred_quantile: Predicted quantile values
        quantile: Quantile level (tau, e.g., 0.5 for median)

    Returns:
        Pinball loss value
    """
    errors = y_true - y_pred_quantile
    loss = np.where(errors >= 0, quantile * errors, (quantile - 1) * errors)
    return np.mean(loss)


def interval_coverage(
    y_true: np.ndarray,
    y_pred_lower: np.ndarray,
    y_pred_upper: np.ndarray
) -> float:
    """
    Prediction interval coverage (proportion of true values within interval).

    Args:
        y_true: True values
        y_pred_lower: Lower bound of prediction interval
        y_pred_upper: Upper bound of prediction interval

    Returns:
        Coverage (between 0 and 1)
    """
    in_interval = (y_true >= y_pred_lower) & (y_true <= y_pred_upper)
    return np.mean(in_interval)


def interval_width(
    y_pred_lower: np.ndarray,
    y_pred_upper: np.ndarray
) -> float:
    """
    Mean width of prediction intervals.

    Args:
        y_pred_lower: Lower bound of prediction interval
        y_pred_upper: Upper bound of prediction interval

    Returns:
        Mean interval width
    """
    return np.mean(y_pred_upper - y_pred_lower)


def compute_point_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonal_period: int = 7
) -> Dict[str, float]:
    """
    Compute all point forecast metrics.

    Args:
        y_true: True values
        y_pred: Predicted values
        y_train: Training data (for MASE)
        seasonal_period: Seasonal period

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'mae': mae(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'smape': smape(y_true, y_pred),
        'mase': mase(y_true, y_pred, y_train, seasonal_period)
    }
    return metrics


def compute_probabilistic_metrics(
    y_true: np.ndarray,
    quantile_predictions: Dict[float, np.ndarray]
) -> Dict[str, float]:
    """
    Compute probabilistic forecast metrics.

    Args:
        y_true: True values
        quantile_predictions: Dictionary mapping quantile levels to predictions
                            (e.g., {0.1: array, 0.5: array, 0.9: array})

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Pinball loss for each quantile
    for q, y_pred_q in quantile_predictions.items():
        metrics[f'pinball_loss_{q}'] = pinball_loss(y_true, y_pred_q, q)

    # Interval coverage and width (if we have symmetric quantiles)
    quantiles = sorted(quantile_predictions.keys())

    # Check for common interval pairs
    interval_pairs = [
        (0.1, 0.9),  # 80% interval
        (0.05, 0.95),  # 90% interval
        (0.25, 0.75)  # 50% interval
    ]

    for lower_q, upper_q in interval_pairs:
        if lower_q in quantile_predictions and upper_q in quantile_predictions:
            coverage = interval_coverage(
                y_true,
                quantile_predictions[lower_q],
                quantile_predictions[upper_q]
            )
            width = interval_width(
                quantile_predictions[lower_q],
                quantile_predictions[upper_q]
            )

            nominal_level = int((upper_q - lower_q) * 100)
            metrics[f'coverage_{nominal_level}'] = coverage
            metrics[f'width_{nominal_level}'] = width

    return metrics


def compute_metrics_by_horizon(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int,
    y_train: np.ndarray,
    seasonal_period: int = 7
) -> Dict[int, Dict[str, float]]:
    """
    Compute metrics for each forecast horizon.

    Args:
        y_true: True values (shape: [n_folds, horizon])
        y_pred: Predicted values (shape: [n_folds, horizon])
        horizon: Forecast horizon
        y_train: Training data
        seasonal_period: Seasonal period

    Returns:
        Dictionary mapping horizon step h to metrics
    """
    metrics_by_h = {}

    for h in range(1, horizon + 1):
        # Extract predictions for horizon h (index h-1)
        y_true_h = y_true[:, h-1]
        y_pred_h = y_pred[:, h-1]

        metrics_h = compute_point_metrics(
            y_true_h,
            y_pred_h,
            y_train,
            seasonal_period
        )

        metrics_by_h[h] = metrics_h

    return metrics_by_h


def aggregate_fold_metrics(
    fold_metrics: List[Dict[str, float]]
) -> Dict[str, float]:
    """
    Aggregate metrics across folds (compute mean and std).

    Args:
        fold_metrics: List of metric dictionaries, one per fold

    Returns:
        Dictionary with mean and std for each metric
    """
    if not fold_metrics:
        return {}

    # Get all metric names
    metric_names = fold_metrics[0].keys()

    aggregated = {}
    for metric_name in metric_names:
        values = [fold[metric_name] for fold in fold_metrics if metric_name in fold]
        aggregated[f'{metric_name}_mean'] = np.mean(values)
        aggregated[f'{metric_name}_std'] = np.std(values)

    return aggregated


def create_metrics_dataframe(
    model_metrics: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """
    Create a formatted DataFrame of metrics for multiple models.

    Args:
        model_metrics: Dictionary mapping model names to their metrics

    Returns:
        DataFrame with models as rows and metrics as columns
    """
    df = pd.DataFrame(model_metrics).T
    return df


def rank_models(
    model_metrics: Dict[str, Dict[str, float]],
    primary_metric: str = 'mase_mean',
    lower_is_better: bool = True
) -> List[Tuple[str, float]]:
    """
    Rank models by a primary metric.

    Args:
        model_metrics: Dictionary mapping model names to their metrics
        primary_metric: Name of metric to rank by
        lower_is_better: Whether lower values are better

    Returns:
        List of tuples (model_name, metric_value) sorted by rank
    """
    rankings = []

    for model_name, metrics in model_metrics.items():
        if primary_metric in metrics:
            rankings.append((model_name, metrics[primary_metric]))

    # Sort
    rankings.sort(key=lambda x: x[1], reverse=not lower_is_better)

    return rankings


def print_metrics_summary(model_metrics: Dict[str, Dict[str, float]]):
    """
    Print a formatted summary of metrics.

    Args:
        model_metrics: Dictionary mapping model names to their metrics
    """
    df = create_metrics_dataframe(model_metrics)

    print("\n" + "="*100)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*100)
    print(df.to_string())
    print("="*100)
