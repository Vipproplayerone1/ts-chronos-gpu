# Model Card: Chronos-2 for Wikipedia Pageviews Forecasting

## Model Details

**Model Name**: Chronos-2 (T5-Base)
**Model Type**: Foundation Model for Time Series Forecasting
**Developer**: Amazon Science
**Model Version**: chronos-forecasting 2.0.0
**Checkpoint**: `amazon/chronos-t5-base`
**Release Date**: October 2024

## Intended Use

### Primary Use Case
Zero-shot probabilistic forecasting of Wikipedia pageview time series with daily frequency and 30-day horizon.

### Intended Users
- Researchers studying time series forecasting
- Data scientists evaluating foundation models
- Academic projects on forecasting methodology

### Out-of-Scope Uses
- Real-time production systems (not optimized for latency)
- Financial trading decisions (not validated for financial data)
- Safety-critical applications (model limitations not fully characterized)

## Model Architecture

- **Base Architecture**: T5 (Text-to-Text Transfer Transformer)
- **Input**: Historical time series (context window)
- **Output**: Probabilistic forecasts (quantile predictions)
- **Inference Mode**: Zero-shot (no fine-tuning)
- **Sampling**: 20 samples for quantile estimation

## Training Data

Chronos-2 was pre-trained on a diverse collection of public time series datasets (exact composition not disclosed by developers). The model uses zero-shot inference on our Wikipedia pageviews data without any fine-tuning.

## Performance

### Dataset
- **Source**: Wikipedia Pageviews API (Bitcoin page)
- **Frequency**: Daily
- **Period**: 2020-2024 (≥500 time steps)
- **Domain**: Information/Media

### Evaluation Protocol
- **Method**: Rolling-origin backtesting (5 folds, expanding window)
- **Split**: 60% train, 20% validation, 20% test
- **Horizon**: H = 30 days
- **Seasonal Period**: m = 7 (weekly)

### Metrics (Test Set)
Results will be available after running the pipeline.

- **MAE**: [To be computed]
- **RMSE**: [To be computed]
- **sMAPE**: [To be computed]%
- **MASE**: [To be computed]
- **Coverage (80% PI)**: [To be computed]%

### Baseline Comparison
- Seasonal Naive (m=7)
- ETS (Exponential Smoothing)
- Gradient Boosting (LightGBM with lag features)

## Limitations

### Model Limitations
1. **Context Length**: Limited by T5 architecture (finite context window)
2. **Frequency Support**: Optimized for common frequencies (daily, hourly)
3. **Computational Cost**: Requires GPU for reasonable inference speed
4. **Calibration**: Prediction intervals may not be perfectly calibrated
5. **Interpretability**: Black-box model, limited explainability

### Data Limitations
1. **Domain Specificity**: Performance on Wikipedia pageviews may not generalize to other domains
2. **Seasonal Patterns**: Assumes consistent weekly seasonality
3. **Structural Breaks**: May not adapt to sudden regime changes
4. **Missing Data**: Preprocessing required for gaps

### Known Failure Modes
1. **Viral Events**: Sudden spikes (e.g., news events) are hard to predict
2. **Trend Changes**: Long-term trend shifts may be missed
3. **Low-Count Series**: Performance may degrade for very low pageview counts
4. **Holidays**: Special days may not be modeled well without exogenous features

## Ethical Considerations

### Privacy
- Uses publicly available Wikipedia data
- No personal identifiable information (PII)
- Aggregated pageview counts only

### Fairness
- Model trained on diverse time series (provider claims)
- No direct demographic or protected attribute dependencies
- Performance may vary across different Wikipedia pages/topics

### Environmental Impact
- GPU training and inference have carbon footprint
- Recommend using renewable energy for compute when possible
- Consider model size vs performance trade-offs

### Potential Misuse
- Should not be used for:
  - Manipulating Wikipedia traffic
  - Gaming recommendation systems
  - Making high-stakes decisions without human oversight

## Reproducibility

### Environment
- Python 3.10+
- PyTorch 2.1+ with CUDA support
- chronos-forecasting 2.0.0

### Seeds
- Random seed: 42 (fixed for reproducibility)
- All data splits are deterministic

### Hardware
- GPU: NVIDIA RTX 3050 Laptop (4GB VRAM)
- CUDA: 12.1
- RAM: 16GB recommended

## Citation

If using this model in research, please cite:

```bibtex
@article{ansari2024chronos,
  title={Chronos: Learning the Language of Time Series},
  author={Ansari, Abdul Fatir and others},
  journal={arXiv preprint arXiv:2403.07815},
  year={2024}
}
```

## Contact

For issues or questions:
- GitHub: [Repository URL]
- Email: [Your contact]

## Version History

- **v1.0** (December 2024): Initial model card for academic project

---

**Last Updated**: December 2024
**Status**: Experimental / Academic Use Only
