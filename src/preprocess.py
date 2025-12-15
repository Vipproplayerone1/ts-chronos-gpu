"""Data preprocessing pipeline for time series data."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from scipy import stats


class TimeSeriesPreprocessor:
    """Preprocessor for time series data with outlier handling and missing value imputation."""

    def __init__(
        self,
        outlier_method: str = "winsorize",
        outlier_quantiles: Tuple[float, float] = (0.01, 0.99),
        missing_method: str = "forward_fill",
        forward_fill_limit: int = 2,
        z_score_threshold: float = 3.5
    ):
        """
        Initialize preprocessor.

        Args:
            outlier_method: Method for outlier handling (winsorize, z_score, or None)
            outlier_quantiles: Quantiles for winsorization
            missing_method: Method for missing value imputation (forward_fill, interpolate)
            forward_fill_limit: Maximum number of consecutive values to forward fill
            z_score_threshold: Z-score threshold for outlier detection
        """
        self.outlier_method = outlier_method
        self.outlier_quantiles = outlier_quantiles
        self.missing_method = missing_method
        self.forward_fill_limit = forward_fill_limit
        self.z_score_threshold = z_score_threshold

        self.outlier_bounds_ = None
        self.preprocessing_stats_ = {}

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the time series.

        Args:
            df: DataFrame with columns ['ds', 'y']

        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        n_missing_before = df['y'].isna().sum()

        if n_missing_before == 0:
            print("[OK] No missing values found")
            return df

        print(f"Found {n_missing_before} missing values ({n_missing_before/len(df)*100:.2f}%)")

        if self.missing_method == "forward_fill":
            df['y'] = df['y'].fillna(method='ffill', limit=self.forward_fill_limit)
            # If still missing after forward fill, use backward fill
            df['y'] = df['y'].fillna(method='bfill', limit=self.forward_fill_limit)

        elif self.missing_method == "interpolate":
            df['y'] = df['y'].interpolate(method='linear', limit_direction='both')

        else:
            raise ValueError(f"Unknown missing_method: {self.missing_method}")

        n_missing_after = df['y'].isna().sum()

        if n_missing_after > 0:
            print(f"Warning: {n_missing_after} missing values remain after imputation")
            # Drop remaining missing values
            df = df.dropna(subset=['y'])

        print(f"[OK] Imputed {n_missing_before - n_missing_after} missing values using {self.missing_method}")

        return df

    def detect_and_handle_outliers(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Detect and handle outliers.

        Args:
            df: DataFrame with columns ['ds', 'y']
            fit: Whether to fit outlier bounds (set True for training data)

        Returns:
            DataFrame with outliers handled
        """
        if self.outlier_method is None or self.outlier_method == "None":
            print("[OK] Skipping outlier handling")
            return df

        df = df.copy()

        if self.outlier_method == "winsorize":
            if fit:
                lower_q, upper_q = self.outlier_quantiles
                self.outlier_bounds_ = (
                    df['y'].quantile(lower_q),
                    df['y'].quantile(upper_q)
                )
                print(f"[OK] Outlier bounds (winsorization): [{self.outlier_bounds_[0]:.1f}, {self.outlier_bounds_[1]:.1f}]")

            if self.outlier_bounds_ is not None:
                lower_bound, upper_bound = self.outlier_bounds_
                n_outliers = ((df['y'] < lower_bound) | (df['y'] > upper_bound)).sum()

                df['y'] = df['y'].clip(lower=lower_bound, upper=upper_bound)

                print(f"[OK] Handled {n_outliers} outliers ({n_outliers/len(df)*100:.2f}%) using winsorization")

        elif self.outlier_method == "z_score":
            if fit:
                mean_y = df['y'].mean()
                std_y = df['y'].std()
                self.outlier_bounds_ = (mean_y, std_y)
                print(f"[OK] Outlier detection (z-score): mean={mean_y:.1f}, std={std_y:.1f}")

            if self.outlier_bounds_ is not None:
                mean_y, std_y = self.outlier_bounds_
                z_scores = np.abs((df['y'] - mean_y) / std_y)
                outliers = z_scores > self.z_score_threshold
                n_outliers = outliers.sum()

                # Cap outliers at threshold
                df.loc[outliers & (df['y'] > mean_y), 'y'] = mean_y + self.z_score_threshold * std_y
                df.loc[outliers & (df['y'] < mean_y), 'y'] = mean_y - self.z_score_threshold * std_y

                print(f"[OK] Handled {n_outliers} outliers ({n_outliers/len(df)*100:.2f}%) using z-score (threshold={self.z_score_threshold})")

        else:
            raise ValueError(f"Unknown outlier_method: {self.outlier_method}")

        return df

    def validate_frequency(self, df: pd.DataFrame, expected_freq: str = 'D') -> bool:
        """
        Validate that the time series has consistent daily frequency.

        Args:
            df: DataFrame with 'ds' column
            expected_freq: Expected frequency code (default: 'D' for daily)

        Returns:
            True if frequency is consistent
        """
        df = df.sort_values('ds')
        time_diffs = df['ds'].diff().dt.days

        # Check for consistent daily frequency
        if expected_freq == 'D':
            inconsistent = time_diffs[time_diffs.notna() & (time_diffs != 1)]

            if len(inconsistent) > 0:
                print(f"Warning: Found {len(inconsistent)} inconsistent gaps in daily frequency")
                print(f"Gap distribution: {inconsistent.value_counts().head()}")
                return False

            print("[OK] Validated consistent daily frequency")
            return True

        return True

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit preprocessor and transform data (use for training data).

        Args:
            df: Raw DataFrame with columns ['ds', 'y']

        Returns:
            Preprocessed DataFrame
        """
        print("\n=== Preprocessing Training Data ===")

        # Ensure datetime type
        df = df.copy()
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds').reset_index(drop=True)

        # Store original stats
        self.preprocessing_stats_['n_original'] = len(df)
        self.preprocessing_stats_['original_mean'] = df['y'].mean()
        self.preprocessing_stats_['original_std'] = df['y'].std()
        self.preprocessing_stats_['original_min'] = df['y'].min()
        self.preprocessing_stats_['original_max'] = df['y'].max()

        # Validate frequency
        self.validate_frequency(df)

        # Handle missing values
        df = self.handle_missing_values(df)

        # Handle outliers (fit=True)
        df = self.detect_and_handle_outliers(df, fit=True)

        # Store final stats
        self.preprocessing_stats_['n_final'] = len(df)
        self.preprocessing_stats_['final_mean'] = df['y'].mean()
        self.preprocessing_stats_['final_std'] = df['y'].std()
        self.preprocessing_stats_['final_min'] = df['y'].min()
        self.preprocessing_stats_['final_max'] = df['y'].max()

        print(f"\n[OK] Preprocessing complete: {len(df)} records")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted parameters (use for validation/test data).

        Args:
            df: Raw DataFrame with columns ['ds', 'y']

        Returns:
            Preprocessed DataFrame
        """
        print("\n=== Preprocessing Val/Test Data ===")

        df = df.copy()
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds').reset_index(drop=True)

        # Handle missing values
        df = self.handle_missing_values(df)

        # Handle outliers (fit=False, use training bounds)
        df = self.detect_and_handle_outliers(df, fit=False)

        print(f"[OK] Preprocessing complete: {len(df)} records")
        return df


def preprocess_data(
    df: pd.DataFrame,
    config: dict,
    fit: bool = True
) -> Tuple[pd.DataFrame, TimeSeriesPreprocessor]:
    """
    Convenience function to preprocess time series data.

    Args:
        df: Raw DataFrame
        config: Configuration dictionary
        fit: Whether to fit the preprocessor

    Returns:
        Tuple of (preprocessed_df, preprocessor)
    """
    preprocessor = TimeSeriesPreprocessor(
        outlier_method=config.get('preprocessing', {}).get('outlier_method', 'winsorize'),
        outlier_quantiles=config.get('preprocessing', {}).get('outlier_quantiles', (0.01, 0.99)),
        missing_method=config.get('preprocessing', {}).get('missing_method', 'forward_fill'),
        forward_fill_limit=config.get('preprocessing', {}).get('forward_fill_limit', 2)
    )

    if fit:
        df_clean = preprocessor.fit_transform(df)
    else:
        df_clean = preprocessor.transform(df)

    return df_clean, preprocessor
