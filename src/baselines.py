"""Baseline models for time series forecasting."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import lightgbm as lgb
from sklearn.base import BaseEstimator

from src.features import TimeSeriesFeatureEngineer, create_recursive_forecast_features


class SeasonalNaiveModel:
    """
    Seasonal Naive forecasting model.

    Forecast: y_t+h = y_t-m+((h-1) mod m)
    where m is the seasonal period.
    """

    def __init__(self, seasonal_period: int = 7):
        """
        Initialize Seasonal Naive model.

        Args:
            seasonal_period: Seasonal period (m), e.g., 7 for weekly
        """
        self.seasonal_period = seasonal_period
        self.y_train = None
        self.last_date = None

    def fit(self, train_df: pd.DataFrame):
        """
        Fit the model (just stores training data).

        Args:
            train_df: Training DataFrame with columns ['ds', 'y']
        """
        self.y_train = train_df['y'].values
        self.last_date = train_df['ds'].iloc[-1]

    def predict(self, horizon: int) -> np.ndarray:
        """
        Make point forecasts.

        Args:
            horizon: Number of steps to forecast

        Returns:
            Array of predictions
        """
        predictions = np.zeros(horizon)

        for h in range(1, horizon + 1):
            # Get the seasonal lag
            lag_idx = len(self.y_train) - self.seasonal_period + ((h - 1) % self.seasonal_period)

            if lag_idx >= 0 and lag_idx < len(self.y_train):
                predictions[h - 1] = self.y_train[lag_idx]
            else:
                # Fallback to last known value if not enough history
                predictions[h - 1] = self.y_train[-1]

        return predictions

    def predict_quantiles(
        self,
        horizon: int,
        quantiles: List[float] = None
    ) -> Dict[float, np.ndarray]:
        """
        Predict quantiles (naive model returns same value for all quantiles).

        Args:
            horizon: Number of steps to forecast
            quantiles: List of quantile levels

        Returns:
            Dictionary mapping quantile levels to predictions
        """
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]

        point_forecast = self.predict(horizon)

        # For naive model, all quantiles are the same (point forecast)
        return {q: point_forecast.copy() for q in quantiles}


class ETSModel:
    """
    Exponential Smoothing (ETS) model wrapper.
    Uses statsmodels implementation of Holt-Winters.
    """

    def __init__(
        self,
        seasonal: str = "add",
        seasonal_periods: int = 7,
        trend: str = "add"
    ):
        """
        Initialize ETS model.

        Args:
            seasonal: Type of seasonal component ('add' or 'mul')
            seasonal_periods: Number of periods in season
            trend: Type of trend component ('add', 'mul', or None)
        """
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.model = None
        self.fitted_model = None

    def fit(self, train_df: pd.DataFrame):
        """
        Fit the ETS model.

        Args:
            train_df: Training DataFrame with columns ['ds', 'y']
        """
        y_train = train_df['y'].values

        try:
            self.model = ExponentialSmoothing(
                y_train,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
                trend=self.trend
            )
            self.fitted_model = self.model.fit(optimized=True)
        except Exception as e:
            print(f"Warning: ETS fit failed, trying simpler model. Error: {e}")
            # Try simpler model without trend
            self.model = ExponentialSmoothing(
                y_train,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
                trend=None
            )
            self.fitted_model = self.model.fit(optimized=True)

    def predict(self, horizon: int) -> np.ndarray:
        """
        Make point forecasts.

        Args:
            horizon: Number of steps to forecast

        Returns:
            Array of predictions
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")

        predictions = self.fitted_model.forecast(steps=horizon)
        return predictions

    def predict_quantiles(
        self,
        horizon: int,
        quantiles: List[float] = None
    ) -> Dict[float, np.ndarray]:
        """
        Predict quantiles (ETS returns same point forecast for all quantiles).

        Args:
            horizon: Number of steps to forecast
            quantiles: List of quantile levels

        Returns:
            Dictionary mapping quantile levels to predictions
        """
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]

        point_forecast = self.predict(horizon)

        # Return point forecast for all quantiles
        # Note: Could add prediction intervals using simulate() for true probabilistic forecasts
        return {q: point_forecast.copy() for q in quantiles}


class GradientBoostingModel:
    """
    Gradient Boosting model with lag and rolling window features.
    Uses LightGBM for fast training.
    """

    def __init__(
        self,
        lags: List[int] = None,
        rolling_windows: List[int] = None,
        use_day_of_week: bool = True,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        n_jobs: int = -1
    ):
        """
        Initialize Gradient Boosting model.

        Args:
            lags: Lag features to use
            rolling_windows: Rolling window sizes
            use_day_of_week: Whether to use day of week feature
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate
            max_depth: Maximum tree depth
            n_jobs: Number of parallel jobs
        """
        self.lags = lags or [1, 7, 14, 28]
        self.rolling_windows = rolling_windows or [7, 28]
        self.use_day_of_week = use_day_of_week
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_jobs = n_jobs

        self.feature_engineer = TimeSeriesFeatureEngineer(
            lags=self.lags,
            rolling_windows=self.rolling_windows,
            use_day_of_week=self.use_day_of_week
        )

        self.model = None
        self.train_df = None

    def fit(self, train_df: pd.DataFrame):
        """
        Fit the model.

        Args:
            train_df: Training DataFrame with columns ['ds', 'y']
        """
        self.train_df = train_df.copy()

        # Create features
        X_train, y_train = self.feature_engineer.prepare_train_test(
            train_df,
            test_df=None
        )

        # Train LightGBM model
        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            n_jobs=self.n_jobs,
            random_state=42,
            verbose=-1
        )

        self.model.fit(X_train, y_train)

    def predict(self, horizon: int) -> np.ndarray:
        """
        Make recursive multi-step forecasts.

        Args:
            horizon: Number of steps to forecast

        Returns:
            Array of predictions
        """
        if self.model is None or self.train_df is None:
            raise ValueError("Model must be fitted before prediction")

        predictions = []
        history_df = self.train_df.copy()

        for h in range(horizon):
            # Create features for next step
            features_df = create_recursive_forecast_features(
                history_df,
                self.feature_engineer,
                horizon=1,
                predictions=predictions
            )

            # Predict next step
            X_next = features_df[self.feature_engineer.get_feature_columns()].values
            y_next = self.model.predict(X_next)[0]

            predictions.append(y_next)

            # Update history with prediction
            next_date = history_df['ds'].iloc[-1] + pd.Timedelta(days=1)
            new_row = pd.DataFrame({
                'ds': [next_date],
                'y': [y_next]
            })
            history_df = pd.concat([history_df, new_row], ignore_index=True)

        return np.array(predictions)

    def predict_quantiles(
        self,
        horizon: int,
        quantiles: List[float] = None
    ) -> Dict[float, np.ndarray]:
        """
        Predict quantiles (GB model returns point forecasts for all quantiles).

        Args:
            horizon: Number of steps to forecast
            quantiles: List of quantile levels

        Returns:
            Dictionary mapping quantile levels to predictions
        """
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]

        point_forecast = self.predict(horizon)

        # For standard GB, return point forecast for all quantiles
        # Note: Could use quantile regression for true probabilistic forecasts
        return {q: point_forecast.copy() for q in quantiles}

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model.

        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")

        importance_df = pd.DataFrame({
            'feature': self.feature_engineer.get_feature_columns(),
            'importance': self.model.feature_importances_
        })

        importance_df = importance_df.sort_values('importance', ascending=False)

        return importance_df


def create_baseline_models(config: dict) -> Dict[str, tuple]:
    """
    Create baseline models based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary mapping model names to (model_class, model_params) tuples
    """
    models = {}

    # Seasonal Naive
    models['seasonal_naive'] = (
        SeasonalNaiveModel,
        {'seasonal_period': config['models']['seasonal_naive']['seasonal_period']}
    )

    # ETS
    models['ets'] = (
        ETSModel,
        {
            'seasonal': config['models']['ets']['seasonal'],
            'seasonal_periods': config['models']['ets']['seasonal_periods'],
            'trend': config['models']['ets']['trend']
        }
    )

    # Gradient Boosting
    gb_config = config['models']['gradient_boosting']
    models['gradient_boosting'] = (
        GradientBoostingModel,
        {
            'lags': gb_config['lags'],
            'rolling_windows': gb_config['rolling_windows'],
            'use_day_of_week': gb_config['use_day_of_week'],
            'n_estimators': gb_config['n_estimators'],
            'learning_rate': gb_config['learning_rate'],
            'max_depth': gb_config['max_depth']
        }
    )

    return models
