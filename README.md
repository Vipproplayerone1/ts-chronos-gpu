You are an expert ML engineer and academic writer. Build an end-to-end, fully reproducible time-series forecasting project that satisfies 100% of the assignment requirements and targets maximum points. Follow the exact equations in the course PDF for all metrics (MAE, RMSE, sMAPE, MASE, pinball loss, coverage/width) and ensure absolutely NO data leakage.

PROJECT GOAL
Investigate ONE modern time-series forecasting foundation model and apply it to ONE real-world univariate time series (Daily frequency) with ≥ 500 time steps. Build strong baselines, use correct time-series evaluation (rolling-origin backtesting), analyze uncertainty, and deliver:
1) Reproducible code repo
2) Concise report (≤ 6 pages, excl. refs/appendix) as PDF
3) Slide deck (6–8 slides) as PDF
4) One-page model card

CHOSEN DIRECTION (LOCK THIS IN)
- Foundation model: Chronos-2 (zero-shot inference, probabilistic quantiles)
- Dataset: Wikipedia Pageviews (Daily) using Wikimedia Pageviews API
- Domain: information/media
- Frequency: Daily
- Forecast horizon: H = 30 days
- Seasonal period: m = 7 (weekly seasonality)
- Use univariate forecasting (no exogenous variables) to keep it simple and correct

HARD CONSTRAINTS (MUST FOLLOW)
- Time series must have ≥ 500 daily points after cleaning.
- Split must preserve time order: Train / Validation / Test with NO overlap.
- Rolling-origin (expanding window) backtesting is REQUIRED on validation for model selection.
- Fit any scaler/normalizer ONLY on training data (though prefer no scaling for simplicity); apply to val/test.
- No future information in features, transforms, or imputation.
- Total training time cap: ≤ 3 hours (prefer zero-shot, minimal training).
- Fix random seeds; record library versions and model checkpoint identifiers/hashes.
- If any API calls are used (pageviews), cache raw data locally (data/ folder) with timestamps.

TASKS TO COMPLETE

A) DATA ACQUISITION & PREP
1. Implement a data loader that downloads daily Wikipedia Pageviews for a chosen page title (e.g., "Bitcoin" or "Taylor Swift") over a multi-year window (e.g., last 3–5 years) using Wikimedia REST API.
2. Convert to a clean dataframe with columns: ds (date), y (pageviews).
3. Handle missing values:
   - Prefer minimal imputation (e.g., forward fill with limit; or linear interpolation).
   - Document the strategy in code and report.
4. Detect and handle outliers:
   - Use a simple, documented method (e.g., winsorize at quantiles or robust z-score).
5. Confirm:
   - daily frequency with no mixed frequency
   - length ≥ 500
   - plot the raw series and cleaned series

B) PROBLEM SETUP
1. Define:
   - Horizon H = 30
   - Seasonal period m = 7
2. Clearly document that exogenous variables are NOT used (to avoid leakage).
3. Define evaluation windows and the prediction target.

C) SPLITS + ROLLING-ORIGIN BACKTESTING (VALIDATION)
1. Create non-overlapping splits:
   - Train: earliest portion
   - Validation: next portion used for k-fold rolling-origin
   - Test: final hold-out period (used once)
2. Implement rolling-origin (expanding window) backtesting on validation with k folds (k=5 preferred):
   Fold i: Train[1:ti] → Forecast next H steps
3. Store per-fold predictions for every model for:
   - point forecasts (mean or median)
   - probabilistic quantiles (at least 0.1, 0.5, 0.9)

D) BASELINES (REQUIRED)
Implement and evaluate these baselines:

1) Seasonal Naive (m=7):
   - Forecast equals last observed value from same season (exact definition from PDF)

2) One statistical model:
   - ETS / Exponential Smoothing (Holt-Winters) OR ARIMA
   - Use validation rolling-origin to tune minimal hyperparameters
   - Keep it robust and stable

3) One of:
   - Prophet OR Gradient-Boosted Regressor with lag features
   Choose Gradient Boosting (recommended for easy setup):
   - Features: lags [1, 7, 14, 28], rolling mean/std windows [7, 28]
   - Optional: day-of-week encoded ONLY from timestamp (safe, known at prediction time)
   - Ensure feature creation uses strictly historical data (no leakage)

Explain feature engineering clearly in the report.

E) FOUNDATION MODEL (CHRONOS-2) USAGE
1. Document exactly:
   - library/package name and version
   - checkpoint name and version identifier
   - inference mode: zero-shot
2. Implement inference:
   - For each fold, feed only the training context (no future)
   - Predict H steps
   - Request probabilistic outputs: quantiles (at least 0.1, 0.5, 0.9)
3. Save all predictions to artifacts/ as parquet/csv for reproducibility.

F) METRICS (FOLLOW PDF EQUATIONS EXACTLY)
Compute per fold and averaged across folds:
Point forecast:
- MAE
- RMSE
- sMAPE
- MASE (use correct seasonal scaling based on m)

Probabilistic (if quantiles available):
- Pinball loss for each quantile τ
- Interval coverage and width for nominal level based on quantiles (e.g., [0.1, 0.9])
Optional:
- CRPS if available

Report:
- per-horizon (h=1..H) errors
- overall averages across folds

G) STATISTICAL SIGNIFICANCE
1. Identify the best baseline based on average validation metric (choose primary metric, e.g., MASE or sMAPE, and justify).
2. Run a paired statistical test (Wilcoxon signed-rank recommended) comparing Chronos-2 vs best baseline:
   - across validation folds and horizons (use paired samples of errors)
3. Report test statistic and p-value in the report with interpretation.

H) ANALYSIS REQUIREMENTS (FOR MAX POINTS)
1. Seasonality check:
   - STL decomposition OR spectral/periodogram demonstrating m=7 weekly seasonality.
2. Error dissection:
   - Error by horizon h
   - Error by level (split y into low/medium/high bins; compare metrics)
   - Error around change points (simple change-point detector or visual + discussion)
3. Calibration:
   - For prediction intervals, plot nominal vs empirical coverage curve.
4. Interpretation:
   - For GB baseline: show feature importance or SHAP (optional) and discuss key drivers.
   - For Chronos-2 (if no attribution): provide post-hoc analysis (e.g., compare performance during peaks vs normal periods).
5. Failure modes/limitations:
   - regime shifts, viral spikes, holidays, missingness, distribution shift
   - discuss when each model breaks

I) REQUIRED PLOTS (MINIMUM)
Generate publication-quality plots (matplotlib/plotly OK):
1. Train/Val/Test timeline with forecast overlays on the test horizon for all models.
2. Backtesting performance vs fold (bar or line).
3. Error by horizon h (line plot).
4. Calibration curve for prediction intervals.

J) REPRODUCIBLE REPO (MUST MATCH CHECKLIST)
Create a Git-ready repository structure:

repo/
  README.md
  environment.yml (or requirements.txt)
  data/ (cached raw downloads; include .gitignore rules if large)
  src/
    config.py
    data_loader.py
    preprocess.py
    features.py
    baselines.py
    chronos_model.py
    backtesting.py
    metrics.py
    stats_tests.py
    plots.py
    utils.py
  notebooks/
    01_eda.ipynb
    02_backtesting.ipynb
    03_test_eval.ipynb
  configs/
    default.yaml
  artifacts/
    predictions/
    metrics/
    figures/
  run_end_to_end.sh (or Makefile target)

README MUST include:
- exact setup steps
- how to download data
- one command to reproduce main tables/figures end-to-end
- random seeds
- expected runtime

K) DELIVERABLE DOCUMENTS
1) REPORT (≤ 6 pages, PDF):
Include:
- Title, author
- Dataset summary: source + license notes
- Problem setup: frequency, H, m
- Methods: baselines + Chronos-2, training/inference details, compute budget
- Evaluation protocol: splits + rolling-origin
- Results: tables + plots + significance test
- Discussion: interpretation + limitations + ethical considerations
- Reproducibility statement

2) SLIDE DECK (6–8 slides, PDF):
Slides:
1. Problem & objective
2. Data (source, size, frequency, examples)
3. Method overview (baselines + Chronos-2)
4. Evaluation protocol (rolling-origin)
5. Results table
6. Key plots (error by horizon + calibration)
7. Statistical significance + takeaway
8. Limitations + next steps

3) MODEL CARD (1 page):
- Model name, checkpoint/API version
- Intended use
- Data sensitivity
- Limitations & failure cases
- Ethical considerations

OUTPUT REQUIREMENTS
- Provide complete code for all modules.
- Provide the report content in a ready-to-export format (LaTeX/Markdown) and compile instructions.
- Provide slide content (PowerPoint/Markdown/Reveal) and export instructions.
- Ensure everything runs end-to-end from a clean environment.

QUALITY BAR (MAX SCORE)
- Correctness: zero leakage, correct equations, proper backtesting.
- Strong baselines: well-implemented and tuned via validation only.
- Evidence: tables + plots + significance testing + deep error analysis.
- Reproducibility: one-command rerun, cached data, fixed seeds, recorded versions.

Now execute: produce (1) repo file contents, (2) report draft, (3) slide draft, (4) model card draft, and a final checklist confirming every rubric item is satisfied.
