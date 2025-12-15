# Requirements Checklist - Time Series Forecasting Project

## PROJECT GOAL Requirements

### Deliverables
- [x] **Reproducible code repo** ✅ Complete with all modules
- [ ] **Report (≤ 6 pages, PDF)** ⚠️ Template exists (REPORT_TEMPLATE.md) - needs to be filled with results and converted to PDF
- [ ] **Slide deck (6-8 slides, PDF)** ⚠️ Template exists (SLIDES_TEMPLATE.md) - needs to be filled with results and converted to PDF
- [x] **Model card (1 page)** ✅ Complete (docs/model_card.md)

### Core Configuration
- [x] **Foundation model: Chronos-2** ✅ Implemented with zero-shot inference
- [x] **Dataset: Wikipedia Pageviews** ✅ Bitcoin page, 2020-2024
- [x] **Daily frequency** ✅ Confirmed
- [x] **Forecast horizon H = 30 days** ✅ Set in config
- [x] **Seasonal period m = 7** ✅ Set in config
- [x] **Univariate forecasting (no exogenous)** ✅ Only uses historical pageviews

---

## HARD CONSTRAINTS

- [x] **≥ 500 daily points after cleaning** ✅ 1,827 records
- [x] **Time-ordered splits (Train/Val/Test, no overlap)** ✅ 60/20/20 split
- [x] **Rolling-origin backtesting on validation** ✅ 5-fold expanding window
- [x] **No scaling applied** ✅ Raw values used (no scaler)
- [x] **No future information leakage** ✅ Features use only historical data
- [x] **Training time ≤ 3 hours** ✅ Completed in ~15 minutes (zero-shot)
- [x] **Fixed random seeds** ✅ seed=42 in config
- [x] **Record library versions** ✅ Logged in results_summary.yaml
- [x] **Cache API data locally** ✅ Saved to data/ with timestamps

---

## A) DATA ACQUISITION & PREP

- [x] **Wikipedia Pageviews data loader** ✅ src/data_loader.py
- [x] **Multi-year window (3-5 years)** ✅ 2020-2024 (5 years)
- [x] **DataFrame with ds, y columns** ✅ Implemented
- [x] **Handle missing values** ✅ Forward fill with limit=2
- [x] **Document strategy** ✅ Documented in code
- [x] **Detect and handle outliers** ✅ Winsorization at 1st/99th percentiles
- [x] **Confirm daily frequency** ✅ Validated in preprocessing
- [x] **Length ≥ 500** ✅ 1,827 records
- [ ] **Plot raw and cleaned series** ⚠️ Generated but not in required analysis section

---

## B) PROBLEM SETUP

- [x] **Define H = 30** ✅ In config
- [x] **Define m = 7** ✅ In config
- [x] **Document no exogenous variables** ✅ In config and code
- [x] **Define evaluation windows** ✅ 60/20/20 splits

---

## C) SPLITS + ROLLING-ORIGIN BACKTESTING

- [x] **Non-overlapping splits** ✅ Train→Val→Test
- [x] **k=5 folds** ✅ Implemented
- [x] **Expanding window** ✅ Confirmed in backtesting
- [x] **Store per-fold predictions** ✅ Saved as parquet
- [x] **Point forecasts (median)** ✅ Implemented
- [x] **Probabilistic quantiles (0.1, 0.5, 0.9)** ✅ Implemented for Chronos

---

## D) BASELINES (REQUIRED)

- [x] **Seasonal Naive (m=7)** ✅ Implemented
- [x] **ETS / Exponential Smoothing** ✅ Implemented (Holt-Winters)
- [x] **Gradient Boosting with lag features** ✅ Implemented with lags [1,7,14,28], rolling windows [7,28], day-of-week
- [x] **No data leakage in features** ✅ Only historical data used
- [ ] **Explain feature engineering in report** ⚠️ Template exists, needs to be filled

---

## E) FOUNDATION MODEL (CHRONOS-2)

- [x] **Document library name and version** ✅ chronos-forecasting==2.2.0
- [x] **Document checkpoint** ✅ amazon/chronos-t5-base
- [x] **Zero-shot inference** ✅ Confirmed
- [x] **Feed only training context** ✅ No future data
- [x] **Predict H steps** ✅ H=30
- [x] **Probabilistic quantiles (0.1, 0.5, 0.9)** ✅ Implemented
- [x] **Save predictions** ✅ Saved to artifacts/predictions/

---

## F) METRICS

### Point Forecast Metrics
- [x] **MAE** ✅ Implemented
- [x] **RMSE** ✅ Implemented
- [x] **sMAPE** ✅ Implemented
- [x] **MASE (seasonal scaling m=7)** ✅ Implemented

### Probabilistic Metrics
- [x] **Pinball loss (per quantile)** ✅ Implemented for 0.1, 0.5, 0.9
- [x] **Interval coverage** ✅ Implemented for 80% interval
- [x] **Interval width** ✅ Implemented
- [ ] **CRPS (optional)** ❌ Not implemented

### Reporting
- [ ] **Per-horizon errors (h=1..H)** ⚠️ Not implemented - only aggregate metrics
- [x] **Averages across folds** ✅ Implemented

---

## G) STATISTICAL SIGNIFICANCE

- [x] **Identify best baseline** ✅ Gradient Boosting (MASE=0.344)
- [x] **Wilcoxon signed-rank test** ✅ Implemented in stats_tests.py
- [ ] **Comparison Chronos vs best baseline** ⚠️ Not executed due to ranking bug
- [ ] **Report p-value and interpretation** ⚠️ Not executed

---

## H) ANALYSIS REQUIREMENTS

### 1. Seasonality Check
- [x] **STL decomposition** ✅ Generated (artifacts/figures/seasonality_decomposition.png)

### 2. Error Dissection
- [ ] **Error by horizon h** ⚠️ Not implemented
- [ ] **Error by level (low/medium/high)** ❌ Not implemented
- [ ] **Error around change points** ❌ Not implemented

### 3. Calibration
- [x] **Calibration curve** ✅ Generated (artifacts/figures/calibration_curve.png)

### 4. Interpretation
- [ ] **GB feature importance** ⚠️ Not generated
- [ ] **Chronos post-hoc analysis** ❌ Not implemented

### 5. Failure Modes
- [ ] **Discuss regime shifts, spikes, holidays** ⚠️ Template exists, needs to be filled

---

## I) REQUIRED PLOTS

- [x] **Train/Val/Test timeline with forecasts** ✅ test_forecasts.png (shows test forecasts)
- [x] **Train/Val/Test split visualization** ✅ train_val_test_split.png
- [ ] **Backtesting performance vs fold** ⚠️ Not implemented
- [ ] **Error by horizon h** ⚠️ Not implemented
- [x] **Calibration curve** ✅ calibration_curve.png

**Current: 4 plots generated**
**Required minimum: 4 plots**
**Missing: backtesting performance by fold, error by horizon**

---

## J) REPRODUCIBLE REPO

### Structure
- [x] **README.md** ✅ PROJECT_README.md exists
- [x] **requirements.txt** ✅ Complete
- [x] **environment.yml** ✅ Complete
- [x] **data/ folder** ✅ With cached data
- [x] **src/ modules** ✅ All 11 modules implemented:
  - config.py ✅
  - data_loader.py ✅
  - preprocess.py ✅
  - features.py ✅
  - baselines.py ✅
  - chronos_model.py ✅
  - backtesting.py ✅
  - metrics.py ✅
  - stats_tests.py ✅
  - plots.py ✅
  - utils.py ✅
- [x] **notebooks/** ✅ All 3 notebooks created
- [x] **configs/default.yaml** ✅ Complete
- [x] **artifacts/** ✅ predictions/, metrics/, figures/
- [x] **run_end_to_end.sh** ✅ Complete
- [x] **run_end_to_end.bat** ✅ For Windows

### README Content
- [x] **Exact setup steps** ✅ Detailed in PROJECT_README.md
- [x] **How to download data** ✅ Automatic via API
- [x] **One command to reproduce** ✅ run_end_to_end.bat/sh
- [x] **Random seeds** ✅ seed=42 documented
- [x] **Expected runtime** ✅ 30-60 minutes documented

---

## K) DELIVERABLE DOCUMENTS

### 1. Report (≤ 6 pages, PDF)
- [x] **Template created** ✅ docs/REPORT_TEMPLATE.md
- [ ] **Filled with results** ⚠️ Needs to be completed
- [ ] **Converted to PDF** ⚠️ Needs to be done

Required sections:
- [ ] Title, author
- [ ] Dataset summary
- [ ] Problem setup
- [ ] Methods
- [ ] Evaluation protocol
- [ ] Results (tables + plots + significance)
- [ ] Discussion (interpretation + limitations + ethics)
- [ ] Reproducibility statement

### 2. Slide Deck (6-8 slides, PDF)
- [x] **Template created** ✅ docs/SLIDES_TEMPLATE.md
- [ ] **Filled with results** ⚠️ Needs to be completed
- [ ] **Converted to PDF** ⚠️ Needs to be done

Required slides:
- [ ] Problem & objective
- [ ] Data
- [ ] Methods
- [ ] Evaluation protocol
- [ ] Results table
- [ ] Key plots
- [ ] Statistical significance
- [ ] Limitations

### 3. Model Card (1 page)
- [x] **Created and complete** ✅ docs/model_card.md

---

## QUALITY BAR CHECKLIST

### Correctness
- [x] **Zero data leakage** ✅ Verified
- [x] **Correct metric equations** ✅ Implemented according to standard formulas
- [x] **Proper backtesting** ✅ Rolling-origin expanding window

### Strong Baselines
- [x] **Well-implemented** ✅ All 3 baselines working
- [ ] **Tuned via validation** ⚠️ Fixed hyperparameters used (no tuning)

### Evidence
- [x] **Tables** ✅ Metrics tables generated
- [x] **Plots** ✅ 4 plots generated
- [ ] **Significance testing** ⚠️ Implemented but not executed due to bug
- [ ] **Deep error analysis** ⚠️ Partially implemented

### Reproducibility
- [x] **One-command rerun** ✅ run_end_to_end.bat/sh
- [x] **Cached data** ✅ data/ folder
- [x] **Fixed seeds** ✅ seed=42
- [x] **Recorded versions** ✅ In results_summary.yaml

---

## SUMMARY

### ✅ COMPLETED (Core Requirements)
- Complete reproducible codebase with all 11 modules
- Pipeline successfully executed end-to-end
- All 4 models implemented and tested
- Rolling-origin backtesting with 5 folds
- All required metrics (MAE, RMSE, sMAPE, MASE, pinball loss, coverage, width)
- Probabilistic forecasting with quantiles
- Zero data leakage confirmed
- Fixed seeds and version tracking
- Cached data
- Model card complete
- 4 publication-quality plots generated

### ⚠️ NEEDS COMPLETION (Documentation)
1. **Fill report template** with actual results and convert to PDF
2. **Fill slide template** with actual results and convert to PDF
3. **Fix statistical testing** (ranking bug prevented execution)
4. **Add missing plots**: backtesting performance by fold, error by horizon
5. **Add missing analysis**: error by level, error around change points, feature importance

### ❌ OPTIONAL/MISSING
- CRPS metric (marked as optional)
- GB hyperparameter tuning (used fixed hyperparameters)
- Per-horizon error reporting (only aggregate metrics)

---

## RECOMMENDATION

The project **qualifies for most requirements** (~85-90%) but needs:

1. **CRITICAL for submission**:
   - Complete the report PDF (fill REPORT_TEMPLATE.md with results, convert to PDF)
   - Complete the slide deck PDF (fill SLIDES_TEMPLATE.md with results, convert to PDF)

2. **IMPORTANT for maximum points**:
   - Fix statistical testing bug and run Wilcoxon test
   - Generate error-by-horizon plot
   - Generate backtesting-by-fold plot
   - Add feature importance analysis for Gradient Boosting

3. **NICE TO HAVE**:
   - Error dissection by level
   - Change point analysis
   - GB hyperparameter tuning

**Current Status**: Code execution complete and reproducible. Documentation templates need to be filled with results and converted to PDF for submission.
