"""Configuration management for the time series forecasting project."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for the project."""

    def __init__(self, config_path: str = "configs/default.yaml"):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._set_seed()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _set_seed(self):
        """Set random seeds for reproducibility."""
        import random
        import numpy as np

        seed = self.config['random_seed']
        random.seed(seed)
        np.random.seed(seed)

        # Set PyTorch seed if available
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass

    def get(self, key: str, default=None):
        """Get configuration value by key (supports dot notation)."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    def __getitem__(self, key: str):
        """Get configuration value by key."""
        return self.get(key)

    def __repr__(self):
        """String representation."""
        return f"Config(config_path='{self.config_path}')"

    @property
    def random_seed(self) -> int:
        """Get random seed."""
        return self.config['random_seed']

    @property
    def horizon(self) -> int:
        """Get forecast horizon."""
        return self.config['ts_params']['horizon']

    @property
    def seasonal_period(self) -> int:
        """Get seasonal period."""
        return self.config['ts_params']['seasonal_period']

    @property
    def page_title(self) -> str:
        """Get Wikipedia page title."""
        return self.config['data']['page_title']

    def create_output_dirs(self):
        """Create output directories if they don't exist."""
        for key in ['predictions_dir', 'metrics_dir', 'figures_dir']:
            path = self.config['output'][key]
            Path(path).mkdir(parents=True, exist_ok=True)


def load_config(config_path: str = "configs/default.yaml") -> Config:
    """
    Load configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        Config object
    """
    return Config(config_path)
