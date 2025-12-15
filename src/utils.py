"""Utility functions for the time series forecasting project."""

import pandas as pd
import numpy as np
from typing import Tuple, List
from pathlib import Path
import json
import yaml


def create_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train, validation, and test sets.

    Args:
        df: DataFrame with time series data (sorted by time)
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"\n=== Data Split ===")
    print(f"Train: {len(train_df)} records ({len(train_df)/n*100:.1f}%) | {train_df['ds'].min()} to {train_df['ds'].max()}")
    print(f"Val:   {len(val_df)} records ({len(val_df)/n*100:.1f}%) | {val_df['ds'].min()} to {val_df['ds'].max()}")
    print(f"Test:  {len(test_df)} records ({len(test_df)/n*100:.1f}%) | {test_df['ds'].min()} to {test_df['ds'].max()}")

    return train_df, val_df, test_df


def save_predictions(
    predictions: pd.DataFrame,
    model_name: str,
    output_dir: str = "artifacts/predictions"
):
    """
    Save model predictions to file.

    Args:
        predictions: DataFrame with predictions
        model_name: Name of the model
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / f"{model_name}_predictions.parquet"
    predictions.to_parquet(filepath, index=False)
    print(f"[OK] Saved predictions to {filepath}")


def load_predictions(
    model_name: str,
    output_dir: str = "artifacts/predictions"
) -> pd.DataFrame:
    """
    Load model predictions from file.

    Args:
        model_name: Name of the model
        output_dir: Output directory

    Returns:
        DataFrame with predictions
    """
    filepath = Path(output_dir) / f"{model_name}_predictions.parquet"
    return pd.read_parquet(filepath)


def save_metrics(
    metrics: dict,
    model_name: str,
    output_dir: str = "artifacts/metrics"
):
    """
    Save model metrics to file.

    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / f"{model_name}_metrics.json"
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"[OK] Saved metrics to {filepath}")


def load_metrics(
    model_name: str,
    output_dir: str = "artifacts/metrics"
) -> dict:
    """
    Load model metrics from file.

    Args:
        model_name: Name of the model
        output_dir: Output directory

    Returns:
        Dictionary of metrics
    """
    filepath = Path(output_dir) / f"{model_name}_metrics.json"
    with open(filepath, 'r') as f:
        return json.load(f)


def _convert_to_python_types(obj):
    """Recursively convert numpy types to Python native types."""
    import numpy as np

    if isinstance(obj, dict):
        return {key: _convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_python_types(item) for item in obj]
    elif isinstance(obj, np.generic):
        return obj.item()  # Convert numpy scalar to Python type
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy array to list
    else:
        return obj


def save_yaml(data: dict, filepath: str):
    """Save dictionary to YAML file, converting numpy types to Python types."""
    # Convert all numpy types to Python native types
    data_converted = _convert_to_python_types(data)

    with open(filepath, 'w') as f:
        yaml.dump(data_converted, f, default_flow_style=False, sort_keys=False)


def load_yaml(filepath: str) -> dict:
    """Load dictionary from YAML file."""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def get_library_versions() -> dict:
    """
    Get versions of key libraries for reproducibility.

    Returns:
        Dictionary of library versions
    """
    import sys
    import pandas
    import numpy
    import sklearn
    import statsmodels

    versions = {
        'python': sys.version,
        'pandas': pandas.__version__,
        'numpy': numpy.__version__,
        'scikit-learn': sklearn.__version__,
        'statsmodels': statsmodels.__version__,
    }

    # Try to get versions of optional libraries
    try:
        import torch
        versions['torch'] = str(torch.__version__)  # Convert to string explicitly
        versions['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            versions['cuda_version'] = str(torch.version.cuda)  # Convert to string
    except ImportError:
        pass

    try:
        import lightgbm
        versions['lightgbm'] = lightgbm.__version__
    except ImportError:
        pass

    try:
        import xgboost
        versions['xgboost'] = xgboost.__version__
    except ImportError:
        pass

    try:
        from chronos import __version__ as chronos_version
        versions['chronos'] = chronos_version
    except (ImportError, AttributeError):
        versions['chronos'] = 'unknown'

    return versions


def print_library_versions():
    """Print versions of key libraries."""
    versions = get_library_versions()
    print("\n=== Library Versions ===")
    for lib, version in versions.items():
        print(f"{lib}: {version}")


def ensure_dir(path: str):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def timeseries_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Time series train-test split (respects temporal order).

    Args:
        X: Features array
        y: Target array
        test_size: Number of samples for test set

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    train_size = len(X) - test_size
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, X_test, y_train, y_test


def format_metric(value: float, decimals: int = 4) -> str:
    """Format metric value for display."""
    return f"{value:.{decimals}f}"


def print_metrics_table(metrics_dict: dict, model_names: List[str] = None):
    """
    Print a formatted table of metrics.

    Args:
        metrics_dict: Dictionary mapping model names to their metrics
        model_names: Optional list of model names to include (in order)
    """
    if model_names is None:
        model_names = list(metrics_dict.keys())

    # Get all metric names
    all_metrics = set()
    for model_metrics in metrics_dict.values():
        all_metrics.update(model_metrics.keys())
    all_metrics = sorted(list(all_metrics))

    # Print header
    print("\n" + "=" * 80)
    print(f"{'Model':<20} " + " ".join([f"{m:>12}" for m in all_metrics]))
    print("=" * 80)

    # Print rows
    for model in model_names:
        if model not in metrics_dict:
            continue
        metrics = metrics_dict[model]
        row = f"{model:<20} "
        for metric in all_metrics:
            value = metrics.get(metric, np.nan)
            if isinstance(value, (int, float)) and not np.isnan(value):
                row += f"{value:>12.4f} "
            else:
                row += f"{'N/A':>12} "
        print(row)

    print("=" * 80)
