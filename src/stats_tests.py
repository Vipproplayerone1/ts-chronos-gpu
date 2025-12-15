"""Statistical significance testing for model comparison."""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, List


def wilcoxon_signed_rank_test(
    errors_model_a: np.ndarray,
    errors_model_b: np.ndarray,
    alternative: str = "two-sided"
) -> Tuple[float, float]:
    """
    Wilcoxon signed-rank test for paired samples.

    Tests if the errors from two models come from the same distribution.

    Args:
        errors_model_a: Errors from model A (absolute or squared)
        errors_model_b: Errors from model B
        alternative: Alternative hypothesis ('two-sided', 'less', 'greater')

    Returns:
        Tuple of (test_statistic, p_value)
    """
    # Ensure arrays are the same length
    assert len(errors_model_a) == len(errors_model_b), "Error arrays must have same length"

    # Perform Wilcoxon signed-rank test
    try:
        statistic, p_value = stats.wilcoxon(
            errors_model_a,
            errors_model_b,
            alternative=alternative,
            zero_method='wilcox'
        )
    except ValueError as e:
        # Handle case where all differences are zero
        print(f"Warning: Wilcoxon test failed: {e}")
        statistic, p_value = np.nan, np.nan

    return statistic, p_value


def paired_t_test(
    errors_model_a: np.ndarray,
    errors_model_b: np.ndarray,
    alternative: str = "two-sided"
) -> Tuple[float, float]:
    """
    Paired t-test for comparing two models.

    Args:
        errors_model_a: Errors from model A
        errors_model_b: Errors from model B
        alternative: Alternative hypothesis ('two-sided', 'less', 'greater')

    Returns:
        Tuple of (test_statistic, p_value)
    """
    statistic, p_value = stats.ttest_rel(
        errors_model_a,
        errors_model_b,
        alternative=alternative
    )

    return statistic, p_value


def diebold_mariano_test(
    errors_model_a: np.ndarray,
    errors_model_b: np.ndarray,
    horizon: int = 1
) -> Tuple[float, float]:
    """
    Diebold-Mariano test for comparing forecast accuracy.

    Args:
        errors_model_a: Forecast errors from model A
        errors_model_b: Forecast errors from model B
        horizon: Forecast horizon (for HAC adjustment)

    Returns:
        Tuple of (test_statistic, p_value)
    """
    # Compute loss differential
    d = errors_model_a ** 2 - errors_model_b ** 2

    # Mean of loss differential
    d_mean = np.mean(d)

    # Variance of loss differential (with HAC correction for autocorrelation)
    n = len(d)

    # Simple variance (no HAC correction for now)
    d_var = np.var(d, ddof=1) / n

    # DM test statistic
    dm_stat = d_mean / np.sqrt(d_var)

    # P-value (two-sided)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

    return dm_stat, p_value


def compare_models_statistical(
    predictions_dict: Dict[str, np.ndarray],
    y_true: np.ndarray,
    model_a: str,
    model_b: str,
    test_type: str = "wilcoxon",
    alpha: float = 0.05
) -> Dict[str, any]:
    """
    Compare two models using statistical significance test.

    Args:
        predictions_dict: Dictionary mapping model names to their predictions
        y_true: True values
        model_a: Name of first model
        model_b: Name of second model
        test_type: Type of test ('wilcoxon', 'paired_t', 'diebold_mariano')
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    # Compute errors
    errors_a = np.abs(y_true - predictions_dict[model_a])
    errors_b = np.abs(y_true - predictions_dict[model_b])

    # Perform test
    if test_type == "wilcoxon":
        statistic, p_value = wilcoxon_signed_rank_test(errors_a, errors_b)
        test_name = "Wilcoxon Signed-Rank Test"
    elif test_type == "paired_t":
        statistic, p_value = paired_t_test(errors_a, errors_b)
        test_name = "Paired t-test"
    elif test_type == "diebold_mariano":
        statistic, p_value = diebold_mariano_test(errors_a, errors_b)
        test_name = "Diebold-Mariano Test"
    else:
        raise ValueError(f"Unknown test_type: {test_type}")

    # Interpret results
    is_significant = p_value < alpha if not np.isnan(p_value) else False

    # Compute mean errors for comparison
    mean_error_a = np.mean(errors_a)
    mean_error_b = np.mean(errors_b)

    better_model = model_a if mean_error_a < mean_error_b else model_b

    results = {
        'test_name': test_name,
        'model_a': model_a,
        'model_b': model_b,
        'test_statistic': statistic,
        'p_value': p_value,
        'alpha': alpha,
        'is_significant': is_significant,
        'mean_error_a': mean_error_a,
        'mean_error_b': mean_error_b,
        'better_model': better_model,
        'interpretation': _interpret_test_results(
            model_a, model_b, p_value, alpha,
            mean_error_a < mean_error_b
        )
    }

    return results


def _interpret_test_results(
    model_a: str,
    model_b: str,
    p_value: float,
    alpha: float,
    a_is_better: bool
) -> str:
    """Generate interpretation text for test results."""
    if np.isnan(p_value):
        return "Test could not be performed (insufficient variation in errors)."

    if p_value < alpha:
        if a_is_better:
            return f"{model_a} significantly outperforms {model_b} (p={p_value:.4f} < {alpha})"
        else:
            return f"{model_b} significantly outperforms {model_a} (p={p_value:.4f} < {alpha})"
    else:
        return f"No significant difference between {model_a} and {model_b} (p={p_value:.4f} >= {alpha})"


def compare_all_models(
    backtest_results: Dict[str, Dict],
    baseline_model: str,
    test_type: str = "wilcoxon",
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Compare all models against a baseline using statistical tests.

    Args:
        backtest_results: Backtest results for all models
        baseline_model: Name of baseline model to compare against
        test_type: Type of statistical test
        alpha: Significance level

    Returns:
        DataFrame with comparison results
    """
    results_list = []

    # Get baseline predictions
    baseline_folds = backtest_results[baseline_model]['folds']

    for model_name, model_results in backtest_results.items():
        if model_name == baseline_model:
            continue

        # Collect predictions and true values across all folds
        y_true_all = []
        y_pred_baseline_all = []
        y_pred_model_all = []

        model_folds = model_results['folds']

        for fold_idx in range(len(baseline_folds)):
            if 'error' in baseline_folds[fold_idx] or 'error' in model_folds[fold_idx]:
                continue

            y_true_all.extend(baseline_folds[fold_idx]['y_true'])
            y_pred_baseline_all.extend(baseline_folds[fold_idx]['y_pred'])
            y_pred_model_all.extend(model_folds[fold_idx]['y_pred'])

        # Convert to arrays
        y_true = np.array(y_true_all)
        predictions_dict = {
            baseline_model: np.array(y_pred_baseline_all),
            model_name: np.array(y_pred_model_all)
        }

        # Perform statistical test
        test_results = compare_models_statistical(
            predictions_dict,
            y_true,
            baseline_model,
            model_name,
            test_type=test_type,
            alpha=alpha
        )

        results_list.append({
            'model': model_name,
            'baseline': baseline_model,
            'test': test_results['test_name'],
            'statistic': test_results['test_statistic'],
            'p_value': test_results['p_value'],
            'significant': test_results['is_significant'],
            'mean_error_model': test_results['mean_error_b'],
            'mean_error_baseline': test_results['mean_error_a'],
            'interpretation': test_results['interpretation']
        })

    return pd.DataFrame(results_list)


def print_statistical_test_results(comparison_df: pd.DataFrame):
    """
    Print formatted statistical test results.

    Args:
        comparison_df: DataFrame from compare_all_models()
    """
    print("\n" + "="*100)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*100)

    for _, row in comparison_df.iterrows():
        print(f"\n{row['model']} vs {row['baseline']}:")
        print(f"  Test: {row['test']}")
        print(f"  Statistic: {row['statistic']:.4f}")
        print(f"  P-value: {row['p_value']:.4f}")
        print(f"  Significant: {'Yes' if row['significant'] else 'No'}")
        print(f"  Mean Error ({row['model']}): {row['mean_error_model']:.4f}")
        print(f"  Mean Error ({row['baseline']}): {row['mean_error_baseline']:.4f}")
        print(f"  {row['interpretation']}")

    print("="*100)
