# Documentation

This folder contains project documentation.

## Files

### 📄 report.pdf
Technical report (≤6 pages) covering:
- Dataset description and preprocessing
- Problem setup (H=30, m=7)
- Models and methods (baselines + Chronos-2)
- Evaluation protocol (rolling-origin backtesting)
- Results and metrics tables
- Statistical significance testing
- Discussion and limitations
- Reproducibility statement

**Status**: To be generated after pipeline execution

### 📊 slides.pdf
Presentation deck (6-8 slides) covering:
1. Problem & Objective
2. Data Source & Preprocessing
3. Methods Overview
4. Evaluation Protocol
5. Results Table
6. Key Plots (error by horizon, calibration)
7. Statistical Tests & Conclusions
8. Limitations & Future Work

**Status**: To be generated after pipeline execution

### 📋 model_card.md
One-page model card for Chronos-2:
- Model details and architecture
- Intended use cases
- Performance metrics
- Limitations and failure modes
- Ethical considerations
- Reproducibility information

**Status**: ✅ Complete (see model_card.md)

## Generation Instructions

### After Running Pipeline

Once you've run `python run_pipeline.py` and have results in `artifacts/`, create the report and slides:

1. **Report Template** (LaTeX recommended):
   ```bash
   # Use results from:
   - artifacts/results_summary.yaml
   - artifacts/metrics/*.json
   - artifacts/figures/*.png
   ```

2. **Slides Template** (PowerPoint/Beamer):
   ```bash
   # Include:
   - Key metrics table from results_summary.yaml
   - Forecast comparison plot
   - Error by horizon plot
   - Calibration curve
   - Statistical test results
   ```

### Quick Stats for Report

```bash
# View summary
cat artifacts/results_summary.yaml

# Check metrics
cat artifacts/metrics/chronos_metrics.json
cat artifacts/metrics/gradient_boosting_metrics.json

# View statistical tests
cat artifacts/metrics/statistical_tests.csv
```

## Notes

- Keep report concise (≤6 pages excluding references)
- Focus on reproducibility and rigor
- Include all required plots
- Document any limitations or assumptions
- Provide clear setup instructions
