@echo off
REM End-to-end execution script for Windows
REM Time Series Forecasting with Chronos-2

echo ======================================================================
echo   Time Series Forecasting with Chronos-2 - End-to-End Execution
echo ======================================================================

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.10+ and try again.
    pause
    exit /b 1
)

REM Create output directories
echo.
echo [1/3] Setting up directories...
if not exist "data" mkdir data
if not exist "artifacts\predictions" mkdir artifacts\predictions
if not exist "artifacts\metrics" mkdir artifacts\metrics
if not exist "artifacts\figures" mkdir artifacts\figures
echo [OK] Directories created

REM Run main pipeline
echo.
echo [2/3] Running main pipeline...
echo This will take 30-60 minutes depending on your GPU.
echo.
python run_pipeline.py --config configs\default.yaml

REM Check if pipeline succeeded
if errorlevel 1 (
    echo.
    echo [FAIL] Pipeline failed. Check error messages above.
    pause
    exit /b 1
)

REM Success
echo.
echo [3/3] Pipeline completed successfully!
echo.
echo ======================================================================
echo Results:
echo   - Summary: artifacts\results_summary.yaml
echo   - Predictions: artifacts\predictions\
echo   - Metrics: artifacts\metrics\
echo   - Figures: artifacts\figures\
echo.
echo To view results:
echo   - Run: type artifacts\results_summary.yaml
echo   - Open: explorer artifacts\figures
echo   - Analyze: jupyter notebook notebooks\
echo ======================================================================
echo.
echo Execution Complete!
pause
