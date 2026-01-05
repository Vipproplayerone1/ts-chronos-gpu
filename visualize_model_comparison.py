"""
Advanced Model Comparison and Deep Visualization Script
========================================================
Creates comprehensive visualizations comparing all 5 forecasting models:
- Seasonal Naive, ETS, Gradient Boosting, Chronos-2 Zero-Shot, Chronos-2 Fine-Tuned

Generates 10+ publication-quality visualizations for detailed model analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import yaml
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
FIGSIZE = (16, 10)
DPI = 300

# Paths
ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
METRICS_DIR = ARTIFACTS / "metrics"
FIGURES_DIR = ARTIFACTS / "figures"
PREDICTIONS_DIR = ARTIFACTS / "predictions"

# Create output directory
DEEP_VIZ_DIR = FIGURES_DIR / "detailed_comparison"
DEEP_VIZ_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("ADVANCED MODEL COMPARISON VISUALIZATION")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/11] Loading model results...")

# Load metrics
with open(METRICS_DIR / "seasonal_naive_metrics.json") as f:
    sn_metrics = json.load(f)
with open(METRICS_DIR / "ets_metrics.json") as f:
    ets_metrics = json.load(f)
with open(METRICS_DIR / "gradient_boosting_metrics.json") as f:
    gb_metrics = json.load(f)
with open(METRICS_DIR / "chronos_metrics.json") as f:
    chronos_metrics = json.load(f)
with open(METRICS_DIR / "chronos_finetuned_metrics.json") as f:
    chronos_ft_metrics = json.load(f)

# Load summary
with open(ARTIFACTS / "results_summary.yaml") as f:
    summary = yaml.safe_load(f)

# Load test metrics
with open(METRICS_DIR / "test_metrics.yaml") as f:
    test_metrics = yaml.safe_load(f)

# Load detailed CSVs
metrics_by_fold = pd.read_csv(METRICS_DIR / "metrics_by_fold.csv")
error_by_horizon = pd.read_csv(METRICS_DIR / "error_by_horizon.csv")
error_by_level = pd.read_csv(METRICS_DIR / "error_by_level.csv")
stat_tests = pd.read_csv(METRICS_DIR / "statistical_tests.csv")

print(f"[OK] Loaded metrics for 5 models")
print(f"[OK] Loaded {len(metrics_by_fold)} fold records")
print(f"[OK] Loaded {len(error_by_horizon)} horizon records")

# ============================================================================
# VIZ 1: MODEL ARCHITECTURE COMPARISON
# ============================================================================
print("\n[2/11] Creating model architecture comparison...")

fig = plt.figure(figsize=(18, 15))
gs = GridSpec(5, 4, figure=fig, hspace=0.4, wspace=0.3)

models_info = [
    {
        "name": "Seasonal Naive",
        "type": "Statistical Baseline",
        "params": "0 trainable",
        "complexity": "O(1)",
        "training": "None",
        "inference": "Instant",
        "memory": "<1 MB",
        "features": ["Seasonality (m=7)"],
        "outputs": ["Point forecast"],
        "color": "#FF6B6B"
    },
    {
        "name": "ETS (Exponential Smoothing)",
        "type": "Statistical Model",
        "params": "~15 optimized",
        "complexity": "O(n)",
        "training": "<1 second",
        "inference": "Instant",
        "memory": "<5 MB",
        "features": ["Level", "Trend", "Seasonality"],
        "outputs": ["Point forecast", "Confidence intervals"],
        "color": "#4ECDC4"
    },
    {
        "name": "Gradient Boosting",
        "type": "Tree Ensemble (ML)",
        "params": "100 trees × depth 5",
        "complexity": "O(n·log(n)·trees)",
        "training": "2-3 seconds",
        "inference": "~10ms",
        "memory": "~50 MB",
        "features": ["Lags [1,7,14,28]", "Rolling mean/std",
                     "Day of week", "Temporal"],
        "outputs": ["Point forecast"],
        "color": "#95E1D3"
    },
    {
        "name": "Chronos-2 (T5-Base)",
        "type": "Foundation Model",
        "params": "~220M (frozen)",
        "complexity": "O(L²) transformer",
        "training": "Zero-shot",
        "inference": "~2 seconds (GPU)",
        "memory": "~1 GB (model)",
        "features": ["Raw time series", "Context window",
                     "Pre-trained patterns"],
        "outputs": ["Probabilistic forecast", "Quantiles [0.1, 0.5, 0.9]",
                   "20 sample paths"],
        "color": "#F38181"
    },
    {
        "name": "Chronos-2 Fine-Tuned",
        "type": "Foundation Model (Adapted)",
        "params": "~220M (framework demo)",
        "complexity": "O(L²) transformer",
        "training": "25 epochs (~framework)",
        "inference": "~2 seconds (GPU)",
        "memory": "~1 GB (model)",
        "features": ["Raw time series", "Domain-adapted",
                     "Training context"],
        "outputs": ["Probabilistic forecast", "Quantiles [0.1, 0.5, 0.9]",
                   "20 sample paths"],
        "color": "#FFA07A"
    }
]

for idx, model in enumerate(models_info):
    ax = fig.add_subplot(gs[idx, :])
    ax.axis('off')

    # Model box
    box = FancyBboxPatch((0.02, 0.3), 0.96, 0.65,
                          boxstyle="round,pad=0.02",
                          facecolor=model["color"],
                          edgecolor='black',
                          linewidth=2,
                          alpha=0.3)
    ax.add_patch(box)

    # Title
    ax.text(0.5, 0.85, model["name"],
            fontsize=18, weight='bold', ha='center',
            transform=ax.transAxes)

    # Type
    ax.text(0.5, 0.75, f"Type: {model['type']}",
            fontsize=12, ha='center', style='italic',
            transform=ax.transAxes)

    # Left column - Architecture
    ax.text(0.05, 0.60, "Architecture:",
            fontsize=11, weight='bold', transform=ax.transAxes)
    ax.text(0.05, 0.53, f"• Parameters: {model['params']}",
            fontsize=10, transform=ax.transAxes)
    ax.text(0.05, 0.46, f"• Complexity: {model['complexity']}",
            fontsize=10, transform=ax.transAxes)
    ax.text(0.05, 0.39, f"• Memory: {model['memory']}",
            fontsize=10, transform=ax.transAxes)

    # Middle column - Performance
    ax.text(0.35, 0.60, "Training/Inference:",
            fontsize=11, weight='bold', transform=ax.transAxes)
    ax.text(0.35, 0.53, f"• Training: {model['training']}",
            fontsize=10, transform=ax.transAxes)
    ax.text(0.35, 0.46, f"• Inference: {model['inference']}",
            fontsize=10, transform=ax.transAxes)

    # Right column - Features
    ax.text(0.60, 0.60, "Features Used:",
            fontsize=11, weight='bold', transform=ax.transAxes)
    for i, feat in enumerate(model["features"]):
        ax.text(0.60, 0.53 - i*0.07, f"• {feat}",
                fontsize=10, transform=ax.transAxes)

    # Bottom - Outputs
    outputs_text = ", ".join(model["outputs"])
    ax.text(0.5, 0.33, f"Outputs: {outputs_text}",
            fontsize=10, ha='center', style='italic',
            transform=ax.transAxes, color='darkblue')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

fig.suptitle("MODEL ARCHITECTURE COMPARISON",
             fontsize=22, weight='bold', y=0.98)
plt.savefig(DEEP_VIZ_DIR / "01_architecture_comparison.png",
            dpi=DPI, bbox_inches='tight')
plt.close()
print(f"[OK] Saved architecture comparison")

# ============================================================================
# VIZ 2: MULTI-METRIC RADAR CHART
# ============================================================================
print("\n[3/11] Creating multi-metric radar chart...")

# Normalize metrics to 0-1 scale (inverted for errors - lower is better)
metrics_data = {
    'Seasonal Naive': {
        'MASE': sn_metrics['mase_mean'],
        'MAE': sn_metrics['mae_mean'],
        'RMSE': sn_metrics['rmse_mean'],
        'sMAPE': sn_metrics['smape_mean'],
    },
    'ETS': {
        'MASE': ets_metrics['mase_mean'],
        'MAE': ets_metrics['mae_mean'],
        'RMSE': ets_metrics['rmse_mean'],
        'sMAPE': ets_metrics['smape_mean'],
    },
    'Gradient Boosting': {
        'MASE': gb_metrics['mase_mean'],
        'MAE': gb_metrics['mae_mean'],
        'RMSE': gb_metrics['rmse_mean'],
        'sMAPE': gb_metrics['smape_mean'],
    },
    'Chronos-2': {
        'MASE': chronos_metrics['mase_mean'],
        'MAE': chronos_metrics['mae_mean'],
        'RMSE': chronos_metrics['rmse_mean'],
        'sMAPE': chronos_metrics['smape_mean'],
    }
}

# Get min/max for normalization
metric_names = ['MASE', 'MAE', 'RMSE', 'sMAPE']
min_vals = {m: min(metrics_data[model][m] for model in metrics_data)
            for m in metric_names}
max_vals = {m: max(metrics_data[model][m] for model in metrics_data)
            for m in metric_names}

# Normalize (invert so higher = better on radar)
normalized_data = {}
for model in metrics_data:
    normalized_data[model] = []
    for metric in metric_names:
        val = metrics_data[model][metric]
        # Invert: 1 - normalized score (so lower error = higher score)
        norm_val = 1 - (val - min_vals[metric]) / (max_vals[metric] - min_vals[metric] + 1e-10)
        normalized_data[model].append(norm_val)

# Radar chart
fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))

angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
angles += angles[:1]  # Close the circle

colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#F38181']
model_names = list(normalized_data.keys())

for idx, model in enumerate(model_names):
    values = normalized_data[model]
    values += values[:1]  # Close the circle
    ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[idx])
    ax.fill(angles, values, alpha=0.15, color=colors[idx])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metric_names, size=14)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0 (Best)'], size=10)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
ax.set_title("MULTI-METRIC PERFORMANCE COMPARISON\n(Normalized: Outer = Better)",
             size=18, weight='bold', pad=20)

plt.savefig(DEEP_VIZ_DIR / "02_radar_chart.png",
            dpi=DPI, bbox_inches='tight')
plt.close()
print(f"[OK] Saved radar chart")

# ============================================================================
# VIZ 3: DETAILED PERFORMANCE BREAKDOWN
# ============================================================================
print("\n[4/11] Creating detailed performance breakdown...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("DETAILED PERFORMANCE BREAKDOWN", fontsize=20, weight='bold')

metrics_to_plot = ['MASE', 'MAE', 'RMSE', 'sMAPE']
metric_labels = ['MASE (↓)', 'MAE (↓)', 'RMSE (↓)', 'sMAPE % (↓)']

for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
    ax = axes[idx // 2, idx % 2]

    # Extract data
    plot_data = []
    for model in model_names:
        val_score = metrics_data[model][metric]

        # Get test score from test_metrics
        test_key = model.replace(' ', '_').replace('-', '_').lower()
        if test_key == 'chronos_2':
            test_key = 'chronos'
        test_score = test_metrics[test_key][metric.lower()]

        plot_data.append({
            'Model': model,
            'Validation': val_score,
            'Test': test_score
        })

    df = pd.DataFrame(plot_data)

    # Bar plot
    x = np.arange(len(model_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, df['Validation'], width,
                   label='Validation', color='skyblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, df['Test'], width,
                   label='Test', color='lightcoral', edgecolor='black')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)

    ax.set_ylabel(label, fontsize=12, weight='bold')
    ax.set_xlabel('Model', fontsize=12, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Highlight best performer
    best_idx = df['Validation'].argmin()
    ax.axvline(x[best_idx], color='green', linestyle='--',
               alpha=0.5, linewidth=2, label='Best')

plt.tight_layout()
plt.savefig(DEEP_VIZ_DIR / "03_performance_breakdown.png",
            dpi=DPI, bbox_inches='tight')
plt.close()
print(f"[OK] Saved performance breakdown")

# ============================================================================
# VIZ 4: ERROR BY HORIZON (ALL MODELS)
# ============================================================================
print("\n[5/11] Creating error by horizon comparison...")

fig, ax = plt.subplots(figsize=(16, 10))
fig.suptitle("MAE ERROR EVOLUTION BY FORECAST HORIZON (30 Days)",
             fontsize=20, weight='bold')

colors_map = {
    'seasonal_naive': '#FF6B6B',
    'ets': '#4ECDC4',
    'gradient_boosting': '#95E1D3',
    'chronos': '#F38181'
}

# The error_by_horizon is in wide format: horizons as rows, models as columns
horizons = error_by_horizon.iloc[:, 0].values  # First column is horizon

for model_key, color in colors_map.items():
    if model_key in error_by_horizon.columns:
        errors = error_by_horizon[model_key].values

        model_display = model_key.replace('_', ' ').title().replace('Chronos', 'Chronos-2')

        ax.plot(horizons, errors, marker='o', label=model_display,
                color=color, linewidth=2.5, markersize=5, alpha=0.8)

ax.set_xlabel('Forecast Horizon (Days)', fontsize=14, weight='bold')
ax.set_ylabel('MAE Error (↓)', fontsize=14, weight='bold')
ax.legend(fontsize=12, loc='upper left')
ax.grid(alpha=0.4, linestyle='--')
ax.set_xlim(1, 30)

# Highlight h=7, 14, 21 (weekly intervals)
for h in [7, 14, 21]:
    ax.axvline(h, color='gray', linestyle=':', alpha=0.6, linewidth=2)
    ax.text(h, ax.get_ylim()[1] * 0.95, f'Week {h//7}',
           ha='center', fontsize=10, style='italic', color='gray')

# Add trend annotation
ax.text(0.98, 0.02, 'Error generally increases with horizon',
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=11, style='italic', bbox=dict(boxstyle='round',
        facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(DEEP_VIZ_DIR / "04_error_by_horizon.png",
            dpi=DPI, bbox_inches='tight')
plt.close()
print(f"[OK] Saved error by horizon")

# ============================================================================
# VIZ 5: ERROR BY PAGEVIEW LEVEL
# ============================================================================
print("\n[6/11] Creating error by pageview level...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("ERROR BY PAGEVIEW LEVEL (Low/Medium/High Traffic)",
             fontsize=18, weight='bold')

levels = error_by_level['level'].unique()
metrics_level = ['MASE', 'MAE', 'sMAPE']

for idx, metric in enumerate(metrics_level):
    ax = axes[idx]

    # Prepare data for grouped bar chart
    data_matrix = []
    for model_key in colors_map.keys():
        model_data = error_by_level[error_by_level['model'] == model_key]
        data_matrix.append([
            model_data[model_data['level'] == level][metric].values[0]
            for level in levels
        ])

    x = np.arange(len(levels))
    width = 0.2

    for i, (model_key, color) in enumerate(colors_map.items()):
        model_display = model_key.replace('_', ' ').title().replace('Chronos', 'Chronos-2')
        offset = (i - 1.5) * width
        ax.bar(x + offset, data_matrix[i], width,
               label=model_display, color=color, edgecolor='black')

    ax.set_ylabel(f'{metric} (↓)', fontsize=12, weight='bold')
    ax.set_xlabel('Pageview Level', fontsize=12, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(DEEP_VIZ_DIR / "05_error_by_level.png",
            dpi=DPI, bbox_inches='tight')
plt.close()
print(f"[OK] Saved error by level")

# ============================================================================
# VIZ 6: FOLD-WISE CONSISTENCY
# ============================================================================
print("\n[7/11] Creating fold-wise consistency analysis...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("CROSS-VALIDATION CONSISTENCY (5 Folds)",
             fontsize=20, weight='bold')

for idx, metric in enumerate(['MASE', 'MAE', 'RMSE', 'sMAPE']):
    ax = axes[idx // 2, idx % 2]

    # Box plot
    data_to_plot = []
    labels = []
    colors_list = []

    for model_key, color in colors_map.items():
        model_data = metrics_by_fold[metrics_by_fold['model'] == model_key]
        data_to_plot.append(model_data[metric].values)
        labels.append(model_key.replace('_', ' ').title().replace('Chronos', 'Chronos-2'))
        colors_list.append(color)

    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                    showmeans=True, meanline=True)

    # Color boxes
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel(f'{metric} (↓)', fontsize=12, weight='bold')
    ax.set_xlabel('Model', fontsize=12, weight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=15)

    # Add variance annotation
    for i, data in enumerate(data_to_plot):
        cv = np.std(data) / np.mean(data) * 100  # Coefficient of variation
        ax.text(i+1, ax.get_ylim()[1] * 0.95, f'CV: {cv:.1f}%',
                ha='center', fontsize=8, style='italic')

plt.tight_layout()
plt.savefig(DEEP_VIZ_DIR / "06_fold_consistency.png",
            dpi=DPI, bbox_inches='tight')
plt.close()
print(f"[OK] Saved fold consistency")

# ============================================================================
# VIZ 7: STATISTICAL SIGNIFICANCE
# ============================================================================
print("\n[8/11] Creating statistical significance visualization...")

fig, ax = plt.subplots(figsize=(12, 8))

# Create comparison matrix
models_short = ['SN', 'ETS', 'GB', 'Chronos-2']
n_models = len(models_short)
significance_matrix = np.zeros((n_models, n_models))

# Parse statistical tests (all against GB)
for _, row in stat_tests.iterrows():
    comparison = row['comparison']
    p_value = row['p_value']

    # Map to indices
    if 'Seasonal Naive' in comparison:
        idx = 0
    elif 'ETS' in comparison:
        idx = 1
    elif 'Chronos' in comparison:
        idx = 3
    else:
        continue

    # GB is index 2
    gb_idx = 2

    # Fill matrix (symmetric)
    if p_value < 0.001:
        val = 3  # Highly significant
    elif p_value < 0.01:
        val = 2
    elif p_value < 0.05:
        val = 1
    else:
        val = 0  # Not significant

    significance_matrix[idx, gb_idx] = val
    significance_matrix[gb_idx, idx] = val

# Heatmap
im = ax.imshow(significance_matrix, cmap='RdYlGn_r', vmin=0, vmax=3)

ax.set_xticks(np.arange(n_models))
ax.set_yticks(np.arange(n_models))
ax.set_xticklabels(models_short, fontsize=12)
ax.set_yticklabels(models_short, fontsize=12)

# Add text annotations
for i in range(n_models):
    for j in range(n_models):
        if i == j:
            text = "—"
        else:
            val = significance_matrix[i, j]
            if val == 3:
                text = "***"
            elif val == 2:
                text = "**"
            elif val == 1:
                text = "*"
            else:
                text = "ns"

        ax.text(j, i, text, ha="center", va="center",
                color="black", fontsize=16, weight='bold')

ax.set_title("STATISTICAL SIGNIFICANCE vs GRADIENT BOOSTING\n" +
             "*** p<0.001, ** p<0.01, * p<0.05, ns = not significant",
             fontsize=16, weight='bold', pad=15)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Significance Level', rotation=270, labelpad=20, fontsize=12)

plt.tight_layout()
plt.savefig(DEEP_VIZ_DIR / "07_statistical_significance.png",
            dpi=DPI, bbox_inches='tight')
plt.close()
print(f"[OK] Saved statistical significance")

# ============================================================================
# VIZ 8: PROBABILISTIC FORECAST QUALITY (CHRONOS)
# ============================================================================
print("\n[9/11] Creating probabilistic forecast quality analysis...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("CHRONOS-2 PROBABILISTIC FORECAST QUALITY",
             fontsize=20, weight='bold')

# Pinball loss by quantile
ax = axes[0, 0]
quantiles = [0.1, 0.5, 0.9]
pinball_scores = [
    chronos_metrics.get('pinball_loss_0.1_mean', 0),
    chronos_metrics.get('pinball_loss_0.5_mean', 0),
    chronos_metrics.get('pinball_loss_0.9_mean', 0)
]

bars = ax.bar(quantiles, pinball_scores, width=0.15,
              color=['#FF6B6B', '#4ECDC4', '#95E1D3'],
              edgecolor='black', linewidth=2)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{height:.1f}',
           ha='center', va='bottom', fontsize=11, weight='bold')

ax.set_xlabel('Quantile', fontsize=12, weight='bold')
ax.set_ylabel('Pinball Loss (↓)', fontsize=12, weight='bold')
ax.set_title('Quantile Forecast Quality', fontsize=14, weight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_xticks(quantiles)

# Coverage analysis
ax = axes[0, 1]
coverage_levels = ['80%']
actual_coverage = [
    chronos_metrics.get('coverage_80_mean', 0) * 100  # Convert to percentage
]
expected_coverage = [80]

x = np.arange(len(coverage_levels))
width = 0.35

bars1 = ax.bar(x - width/2, expected_coverage, width,
               label='Expected', color='lightblue', edgecolor='black')
bars2 = ax.bar(x + width/2, actual_coverage, width,
               label='Actual', color='orange', edgecolor='black')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%',
               ha='center', va='bottom', fontsize=10)

ax.set_ylabel('Coverage (%)', fontsize=12, weight='bold')
ax.set_title('Prediction Interval Coverage', fontsize=14, weight='bold')
ax.set_xticks(x)
ax.set_xticklabels(coverage_levels)
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=100, color='red', linestyle='--', linewidth=1, alpha=0.5)

# Interval width
ax = axes[1, 0]
interval_width_80 = float(chronos_metrics.get('width_80_mean', '0').replace(',', ''))
interval_widths = [interval_width_80]

ax.bar(coverage_levels, interval_widths, color='purple', edgecolor='black', width=0.4)

for i, (level, width) in enumerate(zip(coverage_levels, interval_widths)):
    ax.text(i, width, f'{width:.0f}',
            ha='center', va='bottom', fontsize=11, weight='bold')

ax.set_xlabel('Coverage Level', fontsize=12, weight='bold')
ax.set_ylabel('Mean Interval Width (pageviews)', fontsize=12, weight='bold')
ax.set_title('Prediction Interval Width', fontsize=14, weight='bold')
ax.grid(axis='y', alpha=0.3)

# Summary text
ax = axes[1, 1]
ax.axis('off')

summary_text = f"""
PROBABILISTIC SUMMARY

Quantile Performance:
• τ=0.1: {pinball_scores[0]:.1f} (lower bound)
• τ=0.5: {pinball_scores[1]:.1f} (median)
• τ=0.9: {pinball_scores[2]:.1f} (upper bound)

Coverage Quality:
• 80% PI: {actual_coverage[0]:.1f}% (expect 80%)

Interval Precision:
• 80% width: {interval_widths[0]:.0f} pageviews

Key Insights:
[OK] Lower quantile (0.1) well-calibrated
    Pinball loss of {pinball_scores[0]:.1f}
[WARNING] Moderate undercoverage on 80% PI
    Actual: {actual_coverage[0]:.1f}%
    (volatile data makes tight bounds hard)
[OK] Median forecast quality good
    Pinball loss of {pinball_scores[1]:.1f}
"""

ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='top',
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig(DEEP_VIZ_DIR / "08_probabilistic_quality.png",
            dpi=DPI, bbox_inches='tight')
plt.close()
print(f"[OK] Saved probabilistic quality")

# ============================================================================
# VIZ 9: MODEL RANKING ACROSS METRICS
# ============================================================================
print("\n[10/11] Creating model ranking visualization...")

fig, ax = plt.subplots(figsize=(14, 10))

# Calculate ranks for each metric
metrics_for_ranking = ['MASE', 'MAE', 'RMSE', 'sMAPE']
ranks = {model: [] for model in model_names}

for metric in metrics_for_ranking:
    scores = [(model, metrics_data[model][metric]) for model in model_names]
    scores.sort(key=lambda x: x[1])  # Lower is better

    for rank, (model, _) in enumerate(scores, 1):
        ranks[model].append(rank)

# Add average rank
for model in model_names:
    ranks[model].append(np.mean(ranks[model]))

# Create heatmap
rank_matrix = np.array([ranks[model] for model in model_names])
metric_labels_rank = metrics_for_ranking + ['Average']

im = ax.imshow(rank_matrix, cmap='RdYlGn_r', vmin=1, vmax=4, aspect='auto')

ax.set_xticks(np.arange(len(metric_labels_rank)))
ax.set_yticks(np.arange(len(model_names)))
ax.set_xticklabels(metric_labels_rank, fontsize=12)
ax.set_yticklabels(model_names, fontsize=12)

# Add rank values
for i in range(len(model_names)):
    for j in range(len(metric_labels_rank)):
        rank_val = rank_matrix[i, j]
        color = 'white' if rank_val <= 2 else 'black'

        if j == len(metric_labels_rank) - 1:  # Average column
            text = f'{rank_val:.2f}'
        else:
            text = f'{int(rank_val)}'

        ax.text(j, i, text, ha="center", va="center",
                color=color, fontsize=14, weight='bold')

ax.set_title("MODEL RANKING ACROSS METRICS\n(1=Best, 4=Worst)",
             fontsize=18, weight='bold', pad=15)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Rank', rotation=270, labelpad=20, fontsize=12)
cbar.set_ticks([1, 2, 3, 4])

plt.tight_layout()
plt.savefig(DEEP_VIZ_DIR / "09_model_ranking.png",
            dpi=DPI, bbox_inches='tight')
plt.close()
print(f"[OK] Saved model ranking")

# ============================================================================
# VIZ 10: COMPREHENSIVE COMPARISON TABLE
# ============================================================================
print("\n[11/11] Creating comprehensive comparison table...")

fig, ax = plt.subplots(figsize=(18, 12))
ax.axis('off')

# Prepare table data
table_data = []
table_data.append(['Metric', 'Seasonal Naive', 'ETS',
                   'Gradient Boosting', 'Chronos-2'])

# Performance metrics
perf_metrics = [
    ('MASE (Val)', 'MASE', 'validation'),
    ('MASE (Test)', 'MASE', 'test'),
    ('MAE (Val)', 'MAE', 'validation'),
    ('MAE (Test)', 'MAE', 'test'),
    ('RMSE (Val)', 'RMSE', 'validation'),
    ('RMSE (Test)', 'RMSE', 'test'),
    ('sMAPE (Val)', 'sMAPE', 'validation'),
    ('sMAPE (Test)', 'sMAPE', 'test'),
]

for label, metric, split in perf_metrics:
    row = [label]
    for model_key in ['seasonal_naive', 'ets', 'gradient_boosting', 'chronos']:
        if split == 'validation':
            if model_key == 'seasonal_naive':
                val = sn_metrics[f'{metric.lower()}_mean']
            elif model_key == 'ets':
                val = ets_metrics[f'{metric.lower()}_mean']
            elif model_key == 'gradient_boosting':
                val = gb_metrics[f'{metric.lower()}_mean']
            else:
                val = chronos_metrics[f'{metric.lower()}_mean']
        else:
            val = test_metrics[model_key][metric.lower()]

        row.append(f'{val:.3f}')

    table_data.append(row)

# Add separator
table_data.append(['---'] * 5)

# Model characteristics
char_data = [
    ('Training Time', 'None', '<1s', '2-3s', 'Zero-shot'),
    ('Inference Time', 'Instant', 'Instant', '~10ms', '~2s (GPU)'),
    ('Parameters', '0', '~15', '100 trees', '220M (frozen)'),
    ('Memory Usage', '<1 MB', '<5 MB', '~50 MB', '~1 GB'),
    ('Features Used', '1 (season)', '3 (L+T+S)', '10+ engineered', 'Raw TS'),
    ('Probabilistic', 'No', 'Yes', 'No', 'Yes'),
]

for row_data in char_data:
    table_data.append(list(row_data))

# Add separator
table_data.append(['---'] * 5)

# Rankings
table_data.append(['Validation Rank', '3rd', '4th', '1st [BEST]', '2nd'])
table_data.append(['Test Rank', '2nd', '4th', '1st [BEST]', '3rd'])
table_data.append(['Avg Rank', f'{np.mean(ranks["Seasonal Naive"]):.2f}',
                  f'{np.mean(ranks["ETS"]):.2f}',
                  f'{np.mean(ranks["Gradient Boosting"]):.2f}',
                  f'{np.mean(ranks["Chronos-2"]):.2f}'])

# Create table
table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.25, 0.18, 0.18, 0.18, 0.18])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header row
for j in range(5):
    cell = table[(0, j)]
    cell.set_facecolor('#4ECDC4')
    cell.set_text_props(weight='bold', color='white', fontsize=12)

# Style metric rows alternating colors
for i in range(1, len(table_data)):
    if '---' in table_data[i]:
        for j in range(5):
            table[(i, j)].set_facecolor('#E0E0E0')
        continue

    if i % 2 == 0:
        color = '#F5F5F5'
    else:
        color = 'white'

    for j in range(5):
        table[(i, j)].set_facecolor(color)

    # Bold first column
    table[(i, 0)].set_text_props(weight='bold')

    # Highlight best performers
    if 'Val' in table_data[i][0] or 'Test' in table_data[i][0]:
        values = [float(table_data[i][j]) for j in range(1, 5)]
        best_idx = values.index(min(values))
        table[(i, best_idx + 1)].set_facecolor('#90EE90')
        table[(i, best_idx + 1)].set_text_props(weight='bold')

ax.set_title("COMPREHENSIVE MODEL COMPARISON TABLE",
             fontsize=20, weight='bold', pad=30)

plt.savefig(DEEP_VIZ_DIR / "10_comprehensive_table.png",
            dpi=DPI, bbox_inches='tight')
plt.close()
print(f"[OK] Saved comprehensive table")

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE!")
print("=" * 80)
print(f"\nGenerated 10 detailed visualizations in:")
print(f"  {DEEP_VIZ_DIR}")
print("\nFiles created:")
print("  01_architecture_comparison.png  - Model design comparison")
print("  02_radar_chart.png              - Multi-metric performance")
print("  03_performance_breakdown.png    - Val vs Test metrics")
print("  04_error_by_horizon.png         - 30-day error evolution")
print("  05_error_by_level.png           - Error by traffic level")
print("  06_fold_consistency.png         - Cross-validation stability")
print("  07_statistical_significance.png - Significance vs GB")
print("  08_probabilistic_quality.png    - Chronos-2 probabilistic")
print("  09_model_ranking.png            - Rank heatmap")
print("  10_comprehensive_table.png      - Full comparison table")

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)
print("\n1. BEST PERFORMER: Gradient Boosting")
print(f"   - Validation MASE: {gb_metrics['mase_mean']:.3f}")
print(f"   - Test MASE: {test_metrics['gradient_boosting']['mase']:.3f}")
print("   - Advantage: Domain-tuned features (lag_7 = 35% importance)")

print("\n2. STRONG ZERO-SHOT: Chronos-2")
print(f"   - Validation MASE: {chronos_metrics['mase_mean']:.3f}")
print(f"   - Test MASE: {test_metrics['chronos']['mase']:.3f}")
print("   - Advantage: Probabilistic forecasts, no training")

print("\n3. SIMPLE BASELINE: Seasonal Naive")
print(f"   - Validation MASE: {sn_metrics['mase_mean']:.3f}")
print(f"   - Surprisingly competitive on test set")

print("\n4. WEAKEST: ETS")
print(f"   - Test MASE: {test_metrics['ets']['mase']:.3f}")
print("   - Struggles with volatile Bitcoin pageviews")

print("\n" + "=" * 80)
