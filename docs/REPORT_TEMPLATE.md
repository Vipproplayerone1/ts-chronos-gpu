# Time Series Forecasting with Chronos-2: Wikipedia Pageviews

**Author**: [Your Name]
**Date**: December 2024
**Course**: [Course Name/Number]

---

## Abstract

This study investigates the performance of Chronos-2, a foundation model for time series forecasting, on Wikipedia pageview data. We compare Chronos-2 against three strong baselines (Seasonal Naive, ETS, and Gradient Boosting) using proper rolling-origin backtesting and statistical significance testing. Our results show [TO BE FILLED AFTER PIPELINE RUN].

**Keywords**: Time series forecasting, Foundation models, Chronos-2, Wikipedia pageviews, Zero-shot learning

---

## 1. Introduction

Foundation models have revolutionized natural language processing and computer vision. Recently, Chronos [1] demonstrated that foundation models can achieve strong zero-shot performance on time series forecasting tasks. This study evaluates Chronos-2 on Wikipedia pageview data with the following objectives:

1. Compare zero-shot Chronos-2 against traditional baselines
2. Evaluate probabilistic forecast calibration
3. Assess statistical significance of performance differences
4. Analyze error patterns and failure modes

---

## 2. Dataset

### 2.1 Data Source

- **Source**: Wikimedia Pageviews API (https://wikimedia.org/api/rest_v1/)
- **Page**: Bitcoin
- **Period**: January 1, 2020 - December 31, 2024
- **Frequency**: Daily
- **License**: Wikipedia content (CC BY-SA 3.0)
- **Raw Records**: [FILL: from results_summary.yaml → data_info.n_total]

### 2.2 Preprocessing

**Missing Values**: [FILL: number and handling method]
**Outliers**: [FILL: method and number handled]
**Final Records**: [FILL: n_total after cleaning]

### 2.3 Data Split

- **Train**: 60% ([FILL: n_train] days)
- **Validation**: 20% ([FILL: n_val] days)
- **Test**: 20% ([FILL: n_test] days)

**Date ranges**: [FILL: from results]

---

## 3. Problem Setup

- **Forecast Horizon**: H = 30 days
- **Seasonal Period**: m = 7 (weekly seasonality confirmed via STL decomposition)
- **Frequency**: Daily (D)
- **Task**: Univariate forecasting (no exogenous variables)

---

## 4. Methods

### 4.1 Baseline Models

**1. Seasonal Naive (m=7)**
- Forecast equals value from same weekday last week
- Simple but strong baseline for seasonal data

**2. Exponential Smoothing (ETS)**
- Holt-Winters with additive trend and seasonality
- Optimized parameters via MLE

**3. Gradient Boosting (LightGBM)**
- Features: lags [1, 7, 14, 28], rolling mean/std [7, 28], day-of-week
- Hyperparameters: 100 trees, depth 5, learning rate 0.05
- Recursive multi-step forecasting

### 4.2 Foundation Model: Chronos-2

- **Model**: amazon/chronos-t5-base
- **Architecture**: T5 encoder-decoder
- **Inference**: Zero-shot (no fine-tuning)
- **Probabilistic**: 20 samples → quantiles [0.1, 0.5, 0.9]
- **Context**: Full training history

---

## 5. Evaluation Protocol

### 5.1 Rolling-Origin Backtesting

- **Method**: Expanding window
- **Folds**: k = 5
- **Horizon per fold**: H = 30 days
- **No data leakage**: Strict temporal ordering maintained

### 5.2 Metrics

**Point Forecast**:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- sMAPE (Symmetric Mean Absolute Percentage Error)
- MASE (Mean Absolute Scaled Error) - primary metric

**Probabilistic Forecast**:
- Pinball loss (τ = 0.1, 0.5, 0.9)
- Interval coverage (80% nominal)
- Interval width

### 5.3 Statistical Testing

- **Test**: Wilcoxon signed-rank test (paired, non-parametric)
- **Comparison**: Each model vs. best baseline
- **Alpha**: 0.05
- **Null hypothesis**: No difference in forecast errors

---

## 6. Results

### 6.1 Validation Performance (5-fold backtesting)

**Table 1: Model Performance on Validation Set**

| Model | MAE | RMSE | sMAPE (%) | MASE |
|-------|-----|------|-----------|------|
| Seasonal Naive | [FILL] | [FILL] | [FILL] | [FILL] |
| ETS | [FILL] | [FILL] | [FILL] | [FILL] |
| Gradient Boosting | [FILL] | [FILL] | [FILL] | [FILL] |
| **Chronos-2** | [FILL] | [FILL] | [FILL] | [FILL] |

*Note: MASE is the primary metric. Values are mean ± std across folds.*

### 6.2 Test Set Performance

**Table 2: Final Test Set Performance**

| Model | MAE | RMSE | sMAPE (%) | MASE |
|-------|-----|------|-----------|------|
| [Best Baseline] | [FILL] | [FILL] | [FILL] | [FILL] |
| **Chronos-2** | [FILL] | [FILL] | [FILL] | [FILL] |

### 6.3 Statistical Significance

**Table 3: Wilcoxon Test Results**

| Comparison | Test Statistic | p-value | Significant? |
|------------|---------------|---------|--------------|
| Chronos-2 vs [Baseline] | [FILL] | [FILL] | [Yes/No] |

**Interpretation**: [FILL: Is Chronos-2 significantly better/worse/equivalent?]

### 6.4 Error by Horizon

**Figure 1**: MAE increases with forecast horizon for all models. [DESCRIBE PATTERN]

### 6.5 Probabilistic Forecast Quality (Chronos-2)

- **80% Interval Coverage**: [FILL]% (expected: 80%)
- **Mean Interval Width**: [FILL]
- **Calibration**: [Well-calibrated / Under-confident / Over-confident]

---

## 7. Discussion

### 7.1 Key Findings

1. [FILL: Which model performed best?]
2. [FILL: How did Chronos-2 compare to baselines?]
3. [FILL: Was the difference statistically significant?]
4. [FILL: How well calibrated were the prediction intervals?]

### 7.2 Error Analysis

**By Horizon**: Error increases with h due to [FILL: explain]

**By Level**: Performance on [high/low/medium] pageview periods: [FILL]

**Failure Modes**: Models struggled with [FILL: viral spikes, trend changes, etc.]

### 7.3 Feature Importance (Gradient Boosting)

Top 3 most important features:
1. [FILL: e.g., lag_7]
2. [FILL: e.g., rolling_mean_28]
3. [FILL: e.g., day_of_week]

### 7.4 Limitations

1. **Data**: Single Wikipedia page, limited domain
2. **Models**: No hyperparameter tuning for Chronos-2
3. **Forecast Horizon**: H=30 only
4. **Exogenous Variables**: Not explored
5. **Computational Cost**: Chronos-2 requires GPU

---

## 8. Ethical Considerations

- **Data Privacy**: Public Wikipedia data, no PII
- **Bias**: Potential topic bias (Bitcoin vs other pages)
- **Misuse**: Should not be used for market manipulation
- **Environmental**: GPU usage has carbon footprint

---

## 9. Reproducibility

- **Code**: https://github.com/Vipproplayerone1/ts-chronos-gpu
- **Random Seed**: 42 (fixed)
- **Environment**: Python 3.10, PyTorch 2.5.1, CUDA 12.1
- **Hardware**: NVIDIA RTX 3050 (4GB VRAM)
- **Runtime**: ~[FILL] minutes for full pipeline

All data, code, and results are available in the repository.

---

## 10. Conclusion

This study evaluated Chronos-2 foundation model on Wikipedia pageview forecasting. Key conclusions:

1. [FILL: Main finding about Chronos-2 performance]
2. [FILL: Comparison to baselines]
3. [FILL: Practical implications]

**Future Work**: Extend to multiple pages, incorporate exogenous features, fine-tune Chronos-2.

---

## References

[1] Ansari, A. F., et al. (2024). "Chronos: Learning the Language of Time Series." arXiv preprint arXiv:2403.07815.

[2] Hyndman, R. J., & Athanasopoulos, G. (2021). "Forecasting: Principles and Practice" (3rd ed.).

[3] Wikimedia Foundation. "Pageviews API." https://wikimedia.org/api/rest_v1/

---

**Word Count**: [Keep ≤ 6 pages excluding references]

---

## Appendix (Optional)

### A. Additional Plots

[Include if needed and space permits]

### B. Hyperparameters

[Full configuration from default.yaml]

### C. Library Versions

[Copy from results_summary.yaml → library_versions]
