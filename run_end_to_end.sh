#!/bin/bash

# End-to-end execution script for time series forecasting project
# This script runs the complete pipeline from data loading to evaluation

echo "======================================================================"
echo "  Time Series Forecasting with Chronos-2 - End-to-End Execution"
echo "======================================================================"

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "ts-chronos-gpu" ]]; then
    echo "Warning: conda environment 'ts-chronos-gpu' is not activated"
    echo "Please run: conda activate ts-chronos-gpu"
    exit 1
fi

# Create output directories
echo ""
echo "[1/3] Setting up directories..."
mkdir -p data
mkdir -p artifacts/predictions
mkdir -p artifacts/metrics
mkdir -p artifacts/figures
echo "✓ Directories created"

# Run main pipeline
echo ""
echo "[2/3] Running main pipeline..."
python run_pipeline.py --config configs/default.yaml

# Check if pipeline succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "[3/3] Pipeline completed successfully!"
    echo ""
    echo "Results:"
    echo "  - Summary: artifacts/results_summary.yaml"
    echo "  - Predictions: artifacts/predictions/"
    echo "  - Metrics: artifacts/metrics/"
    echo "  - Figures: artifacts/figures/"
    echo ""
    echo "To view results:"
    echo "  - Open notebooks/03_test_eval.ipynb for detailed analysis"
    echo "  - Check artifacts/figures/ for all plots"
else
    echo ""
    echo "❌ Pipeline failed. Check error messages above."
    exit 1
fi

echo ""
echo "======================================================================"
echo "  Execution Complete!"
echo "======================================================================"
