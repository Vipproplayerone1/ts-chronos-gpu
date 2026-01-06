#!/usr/bin/env python
"""
End-to-End Execution Script for Time Series Forecasting with Chronos-2
Cross-platform launcher that works on Windows, Linux, and macOS.

Usage:
    python run_end_to_end.py

Or directly (if executable):
    ./run_end_to_end.py
"""

import sys
import subprocess
import platform
from pathlib import Path
import shutil

def print_header():
    """Print header banner."""
    print("=" * 70)
    print("  Time Series Forecasting with Chronos-2 - End-to-End Execution")
    print("=" * 70)
    print()

def check_python_version():
    """Verify Python version is 3.10+."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"Error: Python 3.10+ required, but found {version.major}.{version.minor}")
        return False

    print("[OK] Python version compatible")
    print()
    return True

def check_dependencies():
    """Check if required packages are installed."""
    print("Checking dependencies...")

    required = ['pandas', 'numpy', 'torch', 'statsmodels', 'lightgbm', 'matplotlib']
    missing = []

    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"Warning: Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        print("Continuing anyway - pipeline may fail if dependencies are missing")
        print()
    else:
        print("[OK] All required packages found")
        print()

    return True

def create_directories():
    """Create necessary output directories."""
    print("[1/3] Setting up directories...")

    directories = [
        'data',
        'artifacts/predictions',
        'artifacts/metrics',
        'artifacts/figures',
        'artifacts/checkpoints'
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    print("[OK] Directories created")
    print()

def run_pipeline():
    """Execute the main pipeline."""
    print("[2/3] Running main pipeline...")
    print("This will take 30-60 minutes depending on your GPU.")
    print()

    # Check if config file exists
    config_path = Path('configs/default.yaml')
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        return False

    # Check if pipeline script exists
    pipeline_path = Path('run_pipeline.py')
    if not pipeline_path.exists():
        print(f"Error: Pipeline script not found: {pipeline_path}")
        return False

    # Run the pipeline
    cmd = [sys.executable, 'run_pipeline.py', '--config', 'configs/default.yaml']

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print()
        print(f"[FAIL] Pipeline failed with exit code {e.returncode}")
        print()
        print("Common issues:")
        print("  1. Missing packages: pip install -r requirements.txt")
        print("  2. CUDA not available: Check GPU drivers")
        print("  3. Memory error: Try smaller model or CPU mode")
        print("  4. Data loading error: Check internet connection")
        return False
    except KeyboardInterrupt:
        print()
        print("[INTERRUPTED] Pipeline execution cancelled by user")
        return False

def print_results():
    """Print results summary."""
    print()
    print("[3/3] Pipeline completed successfully!")
    print()
    print("=" * 70)
    print("Results:")
    print(f"  - Summary: artifacts{Path('/').as_posix()}results_summary.yaml")
    print(f"  - Predictions: artifacts{Path('/').as_posix()}predictions/")
    print(f"  - Metrics: artifacts{Path('/').as_posix()}metrics/")
    print(f"  - Figures: artifacts{Path('/').as_posix()}figures/")
    print()
    print("To view results:")

    if platform.system() == 'Windows':
        print("  - Run: type artifacts\\results_summary.yaml")
        print("  - Open: explorer artifacts\\figures")
    else:
        print("  - Run: cat artifacts/results_summary.yaml")
        print("  - Open: ls artifacts/figures/")

    print("  - Analyze: jupyter notebook notebooks/")
    print("=" * 70)
    print()
    print("Execution Complete!")
    print()

def main():
    """Main execution function."""
    # Change to script directory
    script_dir = Path(__file__).parent
    if script_dir != Path.cwd():
        print(f"Changing directory to: {script_dir}")
        import os
        os.chdir(script_dir)
        print()

    print_header()

    # Check Python version
    if not check_python_version():
        return 1

    # Check dependencies (non-blocking)
    check_dependencies()

    # Create directories
    try:
        create_directories()
    except Exception as e:
        print(f"Error creating directories: {e}")
        return 1

    # Run pipeline
    success = run_pipeline()

    if success:
        print_results()
        return 0
    else:
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
