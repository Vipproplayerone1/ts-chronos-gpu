@echo off
echo ============================================
echo Running All Notebooks
echo ============================================

echo.
echo [1/4] First, run the main pipeline...
python run_pipeline.py
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] Pipeline failed
    exit /b 1
)

echo.
echo [2/4] Running 01_eda.ipynb...
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda_executed.ipynb
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] EDA notebook failed
    exit /b 1
)

echo.
echo [3/4] Running 02_backtesting.ipynb...
jupyter nbconvert --to notebook --execute notebooks/02_backtesting.ipynb --output 02_backtesting_executed.ipynb
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] Backtesting notebook failed
    exit /b 1
)

echo.
echo [4/4] Running 03_test_eval.ipynb...
jupyter nbconvert --to notebook --execute notebooks/03_test_eval.ipynb --output 03_test_eval_executed.ipynb
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] Test eval notebook failed
    exit /b 1
)

echo.
echo ============================================
echo [SUCCESS] All notebooks executed successfully!
echo ============================================
echo.
echo Executed notebooks saved with "_executed" suffix
echo Original notebooks remain unchanged
