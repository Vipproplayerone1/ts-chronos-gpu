"""Feature engineering for time series forecasting."""

import pandas as pd
import numpy as np
from typing import List, Tuple


class TimeSeriesFeatureEngineer:
    """Feature engineer for time series forecasting with lag and rolling features."""

    def __init__(
        self,
        lags: List[int] = None,
        rolling_windows: List[int] = None,
        use_day_of_week: bool = True,
        use_month: bool = False,
        use_year: bool = False
    ):
        """
        Initialize feature engineer.

        Args:
            lags: List of lag periods to create (e.g., [1, 7, 14, 28])
            rolling_windows: List of rolling window sizes for mean/std (e.g., [7, 28])
            use_day_of_week: Whether to include day of week features
            use_month: Whether to include month features
            use_year: Whether to include year features
        """
        self.lags = lags or [1, 7, 14, 28]
        self.rolling_windows = rolling_windows or [7, 28]
        self.use_day_of_week = use_day_of_week
        self.use_month = use_month
        self.use_year = use_year

    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lag features.

        Args:
            df: DataFrame with columns ['ds', 'y']

        Returns:
            DataFrame with lag features added
        """
        df = df.copy()

        for lag in self.lags:
            df[f'lag_{lag}'] = df['y'].shift(lag)

        return df

    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling window features (mean, std).

        Args:
            df: DataFrame with columns ['ds', 'y']

        Returns:
            DataFrame with rolling features added
        """
        df = df.copy()

        for window in self.rolling_windows:
            df[f'rolling_mean_{window}'] = df['y'].shift(1).rolling(window=window, min_periods=1).mean()
            df[f'rolling_std_{window}'] = df['y'].shift(1).rolling(window=window, min_periods=1).std()

        return df

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from timestamp.

        Args:
            df: DataFrame with 'ds' column

        Returns:
            DataFrame with time features added
        """
        df = df.copy()

        if self.use_day_of_week:
            df['day_of_week'] = df['ds'].dt.dayofweek  # Monday=0, Sunday=6

        if self.use_month:
            df['month'] = df['ds'].dt.month

        if self.use_year:
            df['year'] = df['ds'].dt.year

        return df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features.

        Args:
            df: DataFrame with columns ['ds', 'y']

        Returns:
            DataFrame with all features added
        """
        df = df.copy()

        # Create lag features
        df = self.create_lag_features(df)

        # Create rolling features
        df = self.create_rolling_features(df)

        # Create time features
        df = self.create_time_features(df)

        return df

    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names."""
        features = []

        # Lag features
        for lag in self.lags:
            features.append(f'lag_{lag}')

        # Rolling features
        for window in self.rolling_windows:
            features.append(f'rolling_mean_{window}')
            features.append(f'rolling_std_{window}')

        # Time features
        if self.use_day_of_week:
            features.append('day_of_week')
        if self.use_month:
            features.append('month')
        if self.use_year:
            features.append('year')

        return features

    def prepare_train_test(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame = None,
        horizon: int = 1
    ) -> Tuple:
        """
        Prepare training and test data with features.

        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame (optional)
            horizon: Forecast horizon

        Returns:
            Tuple of (X_train, y_train, X_test, y_test) or (X_train, y_train) if test_df is None
        """
        # Create features for training data
        train_with_features = self.create_features(train_df)

        # Remove rows with NaN (due to lags and rolling windows)
        max_lag = max(self.lags) if self.lags else 0
        max_window = max(self.rolling_windows) if self.rolling_windows else 0
        min_valid_idx = max(max_lag, max_window)

        train_with_features = train_with_features.iloc[min_valid_idx:]

        # Prepare X_train, y_train
        feature_cols = self.get_feature_columns()
        X_train = train_with_features[feature_cols].values
        y_train = train_with_features['y'].values

        if test_df is not None:
            # For test data, we need to concatenate with train to ensure lags are available
            combined_df = pd.concat([train_df, test_df], ignore_index=True)
            combined_with_features = self.create_features(combined_df)

            # Get test portion
            test_start_idx = len(train_df)
            test_with_features = combined_with_features.iloc[test_start_idx:]

            X_test = test_with_features[feature_cols].values
            y_test = test_with_features['y'].values

            return X_train, y_train, X_test, y_test

        return X_train, y_train


def create_recursive_forecast_features(
    history_df: pd.DataFrame,
    feature_engineer: TimeSeriesFeatureEngineer,
    horizon: int,
    predictions: List[float] = None
) -> pd.DataFrame:
    """
    Create features for recursive multi-step forecasting.

    Args:
        history_df: Historical data DataFrame
        feature_engineer: Feature engineer instance
        horizon: Number of steps to forecast
        predictions: List of already-made predictions (for recursive forecasting)

    Returns:
        DataFrame with features for the next step
    """
    # Combine history with predictions made so far
    if predictions:
        pred_df = pd.DataFrame({
            'ds': pd.date_range(
                start=history_df['ds'].iloc[-1] + pd.Timedelta(days=1),
                periods=len(predictions),
                freq='D'
            ),
            'y': predictions
        })
        combined_df = pd.concat([history_df, pred_df], ignore_index=True)
    else:
        combined_df = history_df.copy()

    # Create features
    combined_with_features = feature_engineer.create_features(combined_df)

    # Get the last row (features for next prediction)
    feature_cols = feature_engineer.get_feature_columns()
    next_features = combined_with_features[feature_cols].iloc[-1:]

    return next_features


def prepare_direct_forecast_data(
    df: pd.DataFrame,
    feature_engineer: TimeSeriesFeatureEngineer,
    horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for direct multi-step forecasting (one model per horizon).

    Args:
        df: DataFrame with time series data
        feature_engineer: Feature engineer instance
        horizon: Specific horizon to prepare for (e.g., h=1, h=7, etc.)

    Returns:
        Tuple of (X, y) where y is shifted by horizon steps
    """
    df_with_features = feature_engineer.create_features(df)

    # Shift target by horizon steps to create direct forecast target
    df_with_features['y_target'] = df_with_features['y'].shift(-horizon)

    # Remove rows with NaN
    df_with_features = df_with_features.dropna()

    feature_cols = feature_engineer.get_feature_columns()
    X = df_with_features[feature_cols].values
    y = df_with_features['y_target'].values

    return X, y
