"""Rolling-origin backtesting framework for time series models."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Any, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class RollingOriginBacktester:
    """
    Implements rolling-origin (expanding window) backtesting for time series models.

    This ensures proper evaluation without data leakage by:
    - Training on increasing amounts of historical data
    - Forecasting the next H steps
    - Moving forward in time and repeating
    """

    def __init__(
        self,
        n_folds: int = 5,
        horizon: int = 30,
        method: str = "expanding"
    ):
        """
        Initialize backtester.

        Args:
            n_folds: Number of backtesting folds
            horizon: Forecast horizon (H)
            method: Window type ('expanding' or 'sliding')
        """
        self.n_folds = n_folds
        self.horizon = horizon
        self.method = method

    def create_folds(
        self,
        data: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create train/test folds for rolling-origin backtesting.

        Args:
            data: Full DataFrame with time series data

        Returns:
            List of (train_df, test_df) tuples for each fold
        """
        n = len(data)

        # Calculate fold positions
        # We need at least horizon * (n_folds + 1) points
        min_train_size = n - (self.horizon * self.n_folds) - self.horizon

        if min_train_size < 50:
            raise ValueError(
                f"Not enough data for {self.n_folds} folds with horizon {self.horizon}. "
                f"Need at least {50 + (self.horizon * self.n_folds) + self.horizon} points, "
                f"but have {n} points."
            )

        folds = []

        for fold_idx in range(self.n_folds):
            # Calculate train and test indices
            test_end_idx = n - (self.n_folds - fold_idx - 1) * self.horizon
            test_start_idx = test_end_idx - self.horizon

            if self.method == "expanding":
                # Expanding window: use all data from start
                train_start_idx = 0
                train_end_idx = test_start_idx
            elif self.method == "sliding":
                # Sliding window: use fixed-size window
                train_size = min_train_size
                train_start_idx = test_start_idx - train_size
                train_end_idx = test_start_idx
            else:
                raise ValueError(f"Unknown method: {self.method}")

            # Create fold dataframes
            train_df = data.iloc[train_start_idx:train_end_idx].copy()
            test_df = data.iloc[test_start_idx:test_end_idx].copy()

            folds.append((train_df, test_df))

        return folds

    def backtest_model(
        self,
        data: pd.DataFrame,
        model_class: Any,
        model_params: Dict = None,
        predict_quantiles: bool = False,
        quantile_levels: List[float] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run rolling-origin backtesting for a model.

        Args:
            data: Full time series data
            model_class: Model class with fit() and predict() methods
            model_params: Parameters to pass to model constructor
            predict_quantiles: Whether model supports quantile predictions
            quantile_levels: Quantile levels to predict (if supported)
            verbose: Whether to print progress

        Returns:
            Dictionary with predictions and metadata for each fold
        """
        if model_params is None:
            model_params = {}

        if quantile_levels is None:
            quantile_levels = [0.1, 0.5, 0.9]

        # Create folds
        folds = self.create_folds(data)

        if verbose:
            print(f"\n=== Backtesting: {model_class.__name__} ===")
            print(f"Folds: {self.n_folds}, Horizon: {self.horizon}, Method: {self.method}")

        results = {
            'folds': [],
            'model_name': model_class.__name__,
            'n_folds': self.n_folds,
            'horizon': self.horizon
        }

        # Run backtesting for each fold
        fold_iterator = enumerate(folds)
        if verbose:
            fold_iterator = tqdm(list(fold_iterator), desc="Backtesting folds")

        for fold_idx, (train_df, test_df) in fold_iterator:
            try:
                # Initialize model
                model = model_class(**model_params)

                # Fit model on training data
                model.fit(train_df)

                # Make predictions
                if predict_quantiles and hasattr(model, 'predict_quantiles'):
                    predictions = model.predict_quantiles(
                        horizon=self.horizon,
                        quantiles=quantile_levels
                    )
                else:
                    predictions = model.predict(horizon=self.horizon)

                # Store results for this fold
                fold_result = {
                    'fold': fold_idx,
                    'train_size': len(train_df),
                    'test_size': len(test_df),
                    'train_start': train_df['ds'].iloc[0],
                    'train_end': train_df['ds'].iloc[-1],
                    'test_start': test_df['ds'].iloc[0],
                    'test_end': test_df['ds'].iloc[-1],
                    'y_true': test_df['y'].values,
                    'y_pred': predictions if not predict_quantiles else predictions.get(0.5, predictions),
                    'predictions': predictions
                }

                if predict_quantiles and isinstance(predictions, dict):
                    fold_result['quantile_predictions'] = predictions

                results['folds'].append(fold_result)

            except Exception as e:
                if verbose:
                    print(f"\nError in fold {fold_idx}: {str(e)}")
                # Store error information
                results['folds'].append({
                    'fold': fold_idx,
                    'error': str(e)
                })

        return results

    def aggregate_predictions(
        self,
        backtest_results: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Aggregate predictions from all folds into a DataFrame.

        Args:
            backtest_results: Results from backtest_model()

        Returns:
            DataFrame with all predictions
        """
        rows = []

        for fold_result in backtest_results['folds']:
            if 'error' in fold_result:
                continue

            fold_idx = fold_result['fold']
            y_true = fold_result['y_true']
            y_pred = fold_result['y_pred']

            for h in range(self.horizon):
                row = {
                    'fold': fold_idx,
                    'horizon': h + 1,
                    'y_true': y_true[h],
                    'y_pred': y_pred[h] if h < len(y_pred) else np.nan
                }

                # Add quantile predictions if available
                if 'quantile_predictions' in fold_result:
                    for q, pred_q in fold_result['quantile_predictions'].items():
                        row[f'q_{q}'] = pred_q[h] if h < len(pred_q) else np.nan

                rows.append(row)

        return pd.DataFrame(rows)


def backtest_multiple_models(
    data: pd.DataFrame,
    models: Dict[str, Tuple[Any, Dict]],
    n_folds: int = 5,
    horizon: int = 30,
    method: str = "expanding",
    quantile_levels: List[float] = None,
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Backtest multiple models.

    Args:
        data: Time series data
        models: Dictionary mapping model names to (model_class, model_params) tuples
        n_folds: Number of folds
        horizon: Forecast horizon
        method: Backtesting method
        quantile_levels: Quantile levels for probabilistic forecasts
        verbose: Whether to print progress

    Returns:
        Dictionary mapping model names to their backtest results
    """
    backtester = RollingOriginBacktester(
        n_folds=n_folds,
        horizon=horizon,
        method=method
    )

    all_results = {}

    for model_name, (model_class, model_params) in models.items():
        if verbose:
            print(f"\n{'='*80}")
            print(f"Backtesting: {model_name}")
            print(f"{'='*80}")

        # Check if model supports quantile predictions
        predict_quantiles = hasattr(model_class, 'predict_quantiles')

        results = backtester.backtest_model(
            data=data,
            model_class=model_class,
            model_params=model_params,
            predict_quantiles=predict_quantiles,
            quantile_levels=quantile_levels,
            verbose=verbose
        )

        all_results[model_name] = results

    return all_results


def save_backtest_results(
    results: Dict[str, Dict[str, Any]],
    output_dir: str = "artifacts/predictions"
):
    """
    Save backtesting results to files.

    Args:
        results: Backtest results
        output_dir: Output directory
    """
    from pathlib import Path
    import json

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for model_name, model_results in results.items():
        # Save as parquet
        backtester = RollingOriginBacktester(
            n_folds=model_results['n_folds'],
            horizon=model_results['horizon']
        )

        df = backtester.aggregate_predictions(model_results)
        filepath = output_path / f"{model_name}_backtest.parquet"
        df.to_parquet(filepath, index=False)

        # Save metadata as JSON
        metadata = {
            'model_name': model_name,
            'n_folds': model_results['n_folds'],
            'horizon': model_results['horizon']
        }

        metadata_filepath = output_path / f"{model_name}_backtest_metadata.json"
        with open(metadata_filepath, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"✓ Saved {model_name} backtest results to {filepath}")
