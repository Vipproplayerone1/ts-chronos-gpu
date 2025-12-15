# Presentation: Time Series Forecasting with Chronos-2

**[Your Name]**
**[Date]**
**[Course/Institution]**

---

## Slide 1: Title

# Time Series Forecasting with Chronos-2
## Evaluating Foundation Models on Wikipedia Pageviews

**[Your Name]**
[Date]

---

## Slide 2: Problem & Objective

### Research Question
Can foundation models (Chronos-2) match or exceed traditional time series methods in zero-shot forecasting?

### Objectives
- Compare Chronos-2 vs. strong baselines
- Evaluate probabilistic forecast quality
- Test statistical significance
- Analyze error patterns

### Data
- Wikipedia Pageviews (Bitcoin page)
- Daily frequency, 2020-2024
- ~1,800 time steps

---

## Slide 3: Data & Preprocessing

### Dataset Statistics
- **Source**: Wikimedia Pageviews API
- **Page**: Bitcoin
- **Period**: Jan 2020 - Dec 2024
- **Records**: [FILL: n_total] days after cleaning
- **Seasonality**: Weekly (m=7) ✓ confirmed via STL

### Preprocessing
- Missing values: [FILL: method]
- Outliers: [FILL: % handled]
- Split: 60% train / 20% val / 20% test

### [INSERT: Time series plot showing train/val/test split]

---

## Slide 4: Methods

### Baseline Models
1. **Seasonal Naive** (m=7)
2. **ETS** (Exponential Smoothing)
3. **Gradient Boosting** (LightGBM + lag features)

### Foundation Model
**Chronos-2** (T5-Base)
- Zero-shot inference (no fine-tuning)
- Probabilistic: 20 samples → quantiles
- Context: Full training history

### Evaluation
- **Protocol**: 5-fold rolling-origin backtesting
- **Horizon**: H = 30 days
- **Metrics**: MAE, RMSE, sMAPE, MASE (primary)

---

## Slide 5: Results - Performance Comparison

### Validation Performance (5-fold CV)

| Model | MASE | MAE | RMSE | sMAPE |
|-------|------|-----|------|-------|
| Seasonal Naive | [FILL] | [FILL] | [FILL] | [FILL] |
| ETS | [FILL] | [FILL] | [FILL] | [FILL] |
| Gradient Boosting | [FILL] | [FILL] | [FILL] | [FILL] |
| **Chronos-2** | **[FILL]** | **[FILL]** | **[FILL]** | **[FILL]** |

### Test Set Performance

| Model | MASE ↓ |
|-------|--------|
| [Best Baseline] | [FILL] |
| **Chronos-2** | **[FILL]** |

**Key Finding**: [FILL: Which model won? By how much?]

---

## Slide 6: Error Analysis & Calibration

### Error by Forecast Horizon

**[INSERT: Plot showing MAE increasing with horizon for all models]**

**Observations**:
- All models degrade with longer horizons
- [FILL: Which model degrades fastest/slowest?]

### Calibration (Chronos-2 only)

**[INSERT: Calibration curve plot]**

- **80% Interval Coverage**: [FILL]% (expected: 80%)
- **Calibration Quality**: [Well-calibrated / Needs improvement]

---

## Slide 7: Statistical Significance & Interpretation

### Wilcoxon Signed-Rank Test

| Comparison | p-value | Significant? |
|------------|---------|--------------|
| Chronos-2 vs [Best Baseline] | [FILL] | [Yes/No] |

**Interpretation**:
[FILL:
- If p < 0.05: "Chronos-2 significantly [better/worse] than baseline"
- If p >= 0.05: "No significant difference between models"
]

### Key Takeaways
1. [FILL: Main finding]
2. [FILL: Practical implication]
3. [FILL: When to use Chronos-2 vs baselines?]

---

## Slide 8: Limitations & Future Work

### Limitations
1. **Single domain**: Bitcoin Wikipedia page only
2. **No fine-tuning**: Zero-shot Chronos-2 only
3. **Fixed horizon**: H=30 days
4. **Computational cost**: GPU required
5. **No exogenous features**: Weather, events, etc.

### Future Directions
- Multiple Wikipedia pages (cross-domain evaluation)
- Fine-tune Chronos-2 on domain data
- Add exogenous features (news events, holidays)
- Compare to more recent foundation models
- Explore ensemble methods

### Code & Reproducibility
**GitHub**: https://github.com/Vipproplayerone1/ts-chronos-gpu
**All results reproducible with fixed seed (42)**

---

## Backup Slides (Optional)

### Feature Importance (Gradient Boosting)

Top features:
1. [FILL: e.g., lag_7 - 35%]
2. [FILL: e.g., rolling_mean_28 - 20%]
3. [FILL: e.g., day_of_week - 15%]

### Seasonality Decomposition

**[INSERT: STL decomposition plot]**

- Clear weekly pattern (m=7)
- Trend: [Increasing/Decreasing/Stable]
- Residuals: [Characteristics]

---

## Slide: Questions?

# Thank You!

**Contact**:
- GitHub: [Your GitHub]
- Email: [Your Email]

**Resources**:
- Code: https://github.com/Vipproplayerone1/ts-chronos-gpu
- Chronos Paper: Ansari et al. (2024), arXiv:2403.07815

---

**Notes for Presenter**:
- Keep to 6-8 slides (excluding backups)
- Aim for 10-15 minutes presentation time
- Focus on key results and insights
- Have backup slides ready for Q&A
- Practice explaining the calibration plot
- Be ready to discuss limitations honestly
