#!/bin/bash

# End-to-end execution script for time series forecasting project
# This script runs the complete pipeline from data loading to evaluation

echo "======================================================================"
echo "  Time Series Forecasting with Chronos-2 - End-to-End Execution"
echo "======================================================================"

# Check Python
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.10+ and try again."
    exit 1
fi

echo ""
echo "Python version:"
python --version

# Check if conda environment is activated (optional warning, not required)
if [[ ! -z "$CONDA_DEFAULT_ENV" && "$CONDA_DEFAULT_ENV" != "ts-chronos-gpu" ]]; then
    echo ""
    echo "Note: You're in conda environment '$CONDA_DEFAULT_ENV'"
    echo "Recommended: conda activate ts-chronos-gpu"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create output directories
echo ""
echo "[1/3] Setting up directories..."
mkdir -p data
mkdir -p artifacts/predictions
mkdir -p artifacts/metrics
mkdir -p artifacts/figures
echo "[OK] Directories created"

# Run main pipeline
echo ""
echo "[2/3] Running main pipeline..."
echo "This will take 30-60 minutes depending on your GPU."
echo ""
python run_pipeline.py --config configs/default.yaml

# Check if pipeline succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "[3/3] Pipeline completed successfully!"
    echo ""
    echo "======================================================================"
    echo "Results:"
    echo "  - Summary: artifacts/results_summary.yaml"
    echo "  - Predictions: artifacts/predictions/"
    echo "  - Metrics: artifacts/metrics/"
    echo "  - Figures: artifacts/figures/"
    echo ""
    echo "To view results:"
    echo "  - Run: cat artifacts/results_summary.yaml"
    echo "  - Open: ls artifacts/figures/"
    echo "  - Analyze: jupyter notebook notebooks/"
    echo "======================================================================"
    echo ""
    echo "Execution Complete!"
else
    echo ""
    echo "[FAIL] Pipeline failed. Check error messages above."
    echo ""
    echo "Common issues:"
    echo "  1. Missing packages: pip install -r requirements.txt"
    echo "  2. CUDA not available: Check GPU drivers"
    echo "  3. Memory error: Try smaller model or CPU mode"
    exit 1
fi
