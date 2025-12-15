"""Chronos-2 foundation model wrapper for time series forecasting."""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional
from chronos import ChronosPipeline


class ChronosModel:
    """
    Wrapper for Chronos-2 foundation model.

    Uses zero-shot inference for probabilistic time series forecasting.
    """

    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-base",
        device: str = "cuda",
        torch_dtype: str = "float32",
        num_samples: int = 20,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0
    ):
        """
        Initialize Chronos model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('cuda' or 'cpu')
            torch_dtype: Torch data type
            num_samples: Number of samples for probabilistic forecasting
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.num_samples = num_samples
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        # Set torch dtype
        if torch_dtype == "float32":
            self.torch_dtype = torch.float32
        elif torch_dtype == "float16":
            self.torch_dtype = torch.float16
        elif torch_dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float32

        self.pipeline = None
        self.context = None
        self.context_df = None

        print(f"Initializing Chronos model: {self.model_name}")
        print(f"Device: {self.device}")

    def _load_model(self):
        """Load the Chronos pipeline (lazy loading)."""
        if self.pipeline is None:
            print(f"Loading Chronos model from {self.model_name}...")
            self.pipeline = ChronosPipeline.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=self.torch_dtype
            )
            print("✓ Model loaded successfully")

    def fit(self, train_df: pd.DataFrame):
        """
        'Fit' the model (zero-shot, so just stores context).

        Args:
            train_df: Training DataFrame with columns ['ds', 'y']
        """
        # Load model if not already loaded
        self._load_model()

        # Store training context
        self.context_df = train_df.copy()
        self.context = torch.tensor(train_df['y'].values, dtype=self.torch_dtype)

        print(f"✓ Context stored: {len(train_df)} time steps")

    def predict(self, horizon: int) -> np.ndarray:
        """
        Make point forecasts (median).

        Args:
            horizon: Number of steps to forecast

        Returns:
            Array of median predictions
        """
        if self.pipeline is None or self.context is None:
            raise ValueError("Model must be fitted before prediction")

        # Generate forecasts
        with torch.no_grad():
            forecast_samples = self.pipeline.predict(
                context=self.context.unsqueeze(0),  # Add batch dimension
                prediction_length=horizon,
                num_samples=self.num_samples,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p
            )

        # Extract median (50th percentile) as point forecast
        forecast_samples = forecast_samples.squeeze(0).numpy()  # Remove batch dimension
        median_forecast = np.median(forecast_samples, axis=0)

        return median_forecast

    def predict_quantiles(
        self,
        horizon: int,
        quantiles: List[float] = None
    ) -> Dict[float, np.ndarray]:
        """
        Predict quantiles for probabilistic forecasting.

        Args:
            horizon: Number of steps to forecast
            quantiles: List of quantile levels (e.g., [0.1, 0.5, 0.9])

        Returns:
            Dictionary mapping quantile levels to predictions
        """
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]

        if self.pipeline is None or self.context is None:
            raise ValueError("Model must be fitted before prediction")

        # Generate forecast samples
        with torch.no_grad():
            forecast_samples = self.pipeline.predict(
                context=self.context.unsqueeze(0),
                prediction_length=horizon,
                num_samples=self.num_samples,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p
            )

        # Extract samples and compute quantiles
        forecast_samples = forecast_samples.squeeze(0).numpy()  # Shape: (num_samples, horizon)

        quantile_predictions = {}
        for q in quantiles:
            quantile_predictions[q] = np.quantile(forecast_samples, q, axis=0)

        return quantile_predictions

    def get_model_info(self) -> Dict[str, str]:
        """
        Get model metadata for reproducibility.

        Returns:
            Dictionary with model information
        """
        info = {
            'model_name': self.model_name,
            'device': self.device,
            'torch_dtype': str(self.torch_dtype),
            'num_samples': self.num_samples,
            'temperature': self.temperature,
            'top_k': self.top_k,
            'top_p': self.top_p,
            'inference_mode': 'zero-shot'
        }

        # Add library versions
        try:
            import chronos
            info['chronos_version'] = chronos.__version__ if hasattr(chronos, '__version__') else 'unknown'
        except (ImportError, AttributeError):
            info['chronos_version'] = 'unknown'

        info['torch_version'] = torch.__version__
        info['cuda_available'] = str(torch.cuda.is_available())

        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_name'] = torch.cuda.get_device_name(0)

        return info


def create_chronos_model(config: dict) -> tuple:
    """
    Create Chronos model from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (model_class, model_params)
    """
    chronos_config = config['models']['chronos']

    model_params = {
        'model_name': chronos_config['model_name'],
        'device': chronos_config['device'],
        'num_samples': chronos_config['num_samples'],
        'temperature': chronos_config.get('temperature', 1.0),
        'top_k': chronos_config.get('top_k', 50),
        'top_p': chronos_config.get('top_p', 1.0)
    }

    return (ChronosModel, model_params)


if __name__ == "__main__":
    # Test the Chronos model
    print("Testing Chronos model...")

    # Create synthetic data
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    values = np.sin(np.arange(100) * 2 * np.pi / 7) * 10 + 50 + np.random.randn(100) * 2

    train_df = pd.DataFrame({'ds': dates, 'y': values})

    # Initialize and test model
    model = ChronosModel(
        model_name="amazon/chronos-t5-tiny",  # Use tiny model for testing
        device="cpu",
        num_samples=10
    )

    model.fit(train_df)

    # Test predictions
    horizon = 7
    point_forecast = model.predict(horizon)
    quantile_forecast = model.predict_quantiles(horizon, quantiles=[0.1, 0.5, 0.9])

    print(f"\nPoint forecast (next {horizon} days): {point_forecast}")
    print(f"\nQuantile forecasts: {list(quantile_forecast.keys())}")

    # Model info
    print("\nModel info:")
    for k, v in model.get_model_info().items():
        print(f"  {k}: {v}")
