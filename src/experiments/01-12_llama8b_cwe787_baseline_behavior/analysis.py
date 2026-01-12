"""
Analysis Module: Bootstrap CIs and Aggregation

Provides statistical analysis for Experiment 1 results:
- Aggregate rates (Secure%, Insecure%, Other%, Refusal%)
- Bootstrap 95% confidence intervals
- Breakdowns by base_id and vuln_type
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass

from experiment_config import BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_CI_LEVEL


@dataclass
class RateWithCI:
    """Rate with confidence interval."""
    rate: float
    ci_low: float
    ci_high: float
    count: int
    total: int

    def __str__(self):
        return f"{self.rate*100:.1f}% [{self.ci_low*100:.1f}-{self.ci_high*100:.1f}%] ({self.count}/{self.total})"


def bootstrap_ci(
    data: List[int],
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for a proportion.

    Args:
        data: List of 0/1 values
        n_resamples: Number of bootstrap resamples
        ci_level: Confidence level (e.g., 0.95)

    Returns:
        (ci_low, ci_high)
    """
    if len(data) == 0:
        return (0.0, 0.0)

    data = np.array(data)
    n = len(data)

    # Generate bootstrap samples
    np.random.seed(42)  # Reproducibility
    bootstrap_means = []

    for _ in range(n_resamples):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    # Compute percentile CI
    alpha = 1 - ci_level
    ci_low = np.percentile(bootstrap_means, alpha / 2 * 100)
    ci_high = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

    return (ci_low, ci_high)


def compute_rates(results: List[Dict], scoring_mode: str = 'strict') -> Dict[str, RateWithCI]:
    """
    Compute rates with bootstrap CIs for a list of results.

    Args:
        results: List of result dictionaries
        scoring_mode: 'strict' or 'expanded'

    Returns:
        Dict with 'secure', 'insecure', 'other', 'refusal' RateWithCI objects
    """
    n = len(results)
    if n == 0:
        return {
            'secure': RateWithCI(0, 0, 0, 0, 0),
            'insecure': RateWithCI(0, 0, 0, 0, 0),
            'other': RateWithCI(0, 0, 0, 0, 0),
            'refusal': RateWithCI(0, 0, 0, 0, 0),
        }

    label_key = f'{scoring_mode}_label'

    # Count labels
    secure = [1 if r.get(label_key) == 'secure' else 0 for r in results]
    insecure = [1 if r.get(label_key) == 'insecure' else 0 for r in results]
    other = [1 if r.get(label_key) == 'other' else 0 for r in results]
    refusal = [1 if r.get('is_refusal', False) else 0 for r in results]

    return {
        'secure': RateWithCI(
            rate=sum(secure) / n,
            ci_low=bootstrap_ci(secure)[0],
            ci_high=bootstrap_ci(secure)[1],
            count=sum(secure),
            total=n,
        ),
        'insecure': RateWithCI(
            rate=sum(insecure) / n,
            ci_low=bootstrap_ci(insecure)[0],
            ci_high=bootstrap_ci(insecure)[1],
            count=sum(insecure),
            total=n,
        ),
        'other': RateWithCI(
            rate=sum(other) / n,
            ci_low=bootstrap_ci(other)[0],
            ci_high=bootstrap_ci(other)[1],
            count=sum(other),
            total=n,
        ),
        'refusal': RateWithCI(
            rate=sum(refusal) / n,
            ci_low=bootstrap_ci(refusal)[0],
            ci_high=bootstrap_ci(refusal)[1],
            count=sum(refusal),
            total=n,
        ),
    }


def aggregate_overall(results: List[Dict]) -> Dict[str, Dict[str, RateWithCI]]:
    """
    Compute overall aggregate rates for both scoring modes.

    Returns:
        {
            'strict': {'secure': RateWithCI, 'insecure': ..., 'other': ..., 'refusal': ...},
            'expanded': {'secure': RateWithCI, ...}
        }
    """
    return {
        'strict': compute_rates(results, 'strict'),
        'expanded': compute_rates(results, 'expanded'),
    }


def aggregate_by_base_id(results: List[Dict]) -> Dict[str, Dict[str, Dict[str, RateWithCI]]]:
    """
    Compute rates grouped by base_id.

    Returns:
        {
            base_id: {
                'strict': {'secure': RateWithCI, ...},
                'expanded': {'secure': RateWithCI, ...}
            }
        }
    """
    # Group by base_id
    by_base_id = defaultdict(list)
    for r in results:
        base_id = r.get('base_id', 'unknown')
        by_base_id[base_id].append(r)

    # Compute rates for each group
    aggregated = {}
    for base_id, group_results in by_base_id.items():
        aggregated[base_id] = {
            'strict': compute_rates(group_results, 'strict'),
            'expanded': compute_rates(group_results, 'expanded'),
            'n': len(group_results),
        }

    return aggregated


def aggregate_by_vuln_type(results: List[Dict]) -> Dict[str, Dict[str, Dict[str, RateWithCI]]]:
    """
    Compute rates grouped by vulnerability type.

    Returns:
        {
            vuln_type: {
                'strict': {'secure': RateWithCI, ...},
                'expanded': {'secure': RateWithCI, ...}
            }
        }
    """
    # Group by vuln_type
    by_vuln_type = defaultdict(list)
    for r in results:
        vuln_type = r.get('vulnerability_type', 'unknown')
        by_vuln_type[vuln_type].append(r)

    # Compute rates for each group
    aggregated = {}
    for vuln_type, group_results in by_vuln_type.items():
        aggregated[vuln_type] = {
            'strict': compute_rates(group_results, 'strict'),
            'expanded': compute_rates(group_results, 'expanded'),
            'n': len(group_results),
        }

    return aggregated


def format_comparison_table(
    base_results: Dict[str, Dict[str, RateWithCI]],
    expanded_results: Dict[str, Dict[str, RateWithCI]],
) -> str:
    """
    Format a comparison table for Base vs Expanded results.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("BASELINE RATES: Base vs Expanded")
    lines.append("=" * 80)
    lines.append("")

    for mode in ['strict', 'expanded']:
        lines.append(f"### {mode.upper()} SCORING")
        lines.append("-" * 60)
        lines.append(f"{'Metric':<15} {'Base':>25} {'Expanded':>25}")
        lines.append("-" * 60)

        for metric in ['secure', 'insecure', 'other', 'refusal']:
            base_rate = base_results[mode][metric]
            exp_rate = expanded_results[mode][metric]

            base_str = f"{base_rate.rate*100:5.1f}% [{base_rate.ci_low*100:4.1f}-{base_rate.ci_high*100:4.1f}%]"
            exp_str = f"{exp_rate.rate*100:5.1f}% [{exp_rate.ci_low*100:4.1f}-{exp_rate.ci_high*100:4.1f}%]"

            lines.append(f"{metric.capitalize():<15} {base_str:>25} {exp_str:>25}")

        lines.append("")

    lines.append(f"{'n (samples)':<15} {base_results['strict']['secure'].total:>25} {expanded_results['strict']['secure'].total:>25}")

    return "\n".join(lines)


def format_by_base_id_table(by_base_id: Dict, scoring_mode: str = 'strict') -> str:
    """
    Format a table showing rates by base_id.
    """
    lines = []
    lines.append("")
    lines.append(f"### BY BASE_ID ({scoring_mode.upper()} SCORING)")
    lines.append("-" * 90)
    lines.append(f"{'Base ID':<25} {'n':>5} {'Secure%':>12} {'Insecure%':>12} {'Other%':>12} {'Refusal%':>12}")
    lines.append("-" * 90)

    for base_id in sorted(by_base_id.keys()):
        data = by_base_id[base_id]
        n = data['n']
        rates = data[scoring_mode]

        lines.append(
            f"{base_id:<25} {n:>5} "
            f"{rates['secure'].rate*100:>11.1f}% "
            f"{rates['insecure'].rate*100:>11.1f}% "
            f"{rates['other'].rate*100:>11.1f}% "
            f"{rates['refusal'].rate*100:>11.1f}%"
        )

    lines.append("-" * 90)

    return "\n".join(lines)


def format_by_vuln_type_table(by_vuln_type: Dict, scoring_mode: str = 'strict') -> str:
    """
    Format a table showing rates by vulnerability type.
    """
    lines = []
    lines.append("")
    lines.append(f"### BY VULN_TYPE ({scoring_mode.upper()} SCORING)")
    lines.append("-" * 70)
    lines.append(f"{'Vuln Type':<15} {'n':>5} {'Secure%':>12} {'Insecure%':>12} {'Other%':>12} {'Refusal%':>12}")
    lines.append("-" * 70)

    for vuln_type in sorted(by_vuln_type.keys()):
        data = by_vuln_type[vuln_type]
        n = data['n']
        rates = data[scoring_mode]

        lines.append(
            f"{vuln_type:<15} {n:>5} "
            f"{rates['secure'].rate*100:>11.1f}% "
            f"{rates['insecure'].rate*100:>11.1f}% "
            f"{rates['other'].rate*100:>11.1f}% "
            f"{rates['refusal'].rate*100:>11.1f}%"
        )

    lines.append("-" * 70)

    return "\n".join(lines)


def results_to_serializable(rates: Dict[str, RateWithCI]) -> Dict[str, Any]:
    """Convert RateWithCI objects to serializable dicts."""
    return {
        k: {
            'rate': v.rate,
            'ci_low': v.ci_low,
            'ci_high': v.ci_high,
            'count': v.count,
            'total': v.total,
        }
        for k, v in rates.items()
    }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Create mock results
    mock_results = [
        {'strict_label': 'secure', 'expanded_label': 'secure', 'is_refusal': False, 'base_id': 'pair_07', 'vulnerability_type': 'sprintf'},
        {'strict_label': 'insecure', 'expanded_label': 'insecure', 'is_refusal': False, 'base_id': 'pair_07', 'vulnerability_type': 'sprintf'},
        {'strict_label': 'insecure', 'expanded_label': 'other', 'is_refusal': False, 'base_id': 'pair_07', 'vulnerability_type': 'sprintf'},
        {'strict_label': 'other', 'expanded_label': 'other', 'is_refusal': True, 'base_id': 'pair_09', 'vulnerability_type': 'strcat'},
        {'strict_label': 'secure', 'expanded_label': 'secure', 'is_refusal': False, 'base_id': 'pair_09', 'vulnerability_type': 'strcat'},
    ]

    print("Testing analysis module...")
    print("=" * 70)

    # Test overall aggregation
    overall = aggregate_overall(mock_results)
    print("\nOverall (STRICT):")
    for k, v in overall['strict'].items():
        print(f"  {k}: {v}")

    print("\nOverall (EXPANDED):")
    for k, v in overall['expanded'].items():
        print(f"  {k}: {v}")

    # Test by base_id
    by_base = aggregate_by_base_id(mock_results)
    print(format_by_base_id_table(by_base, 'strict'))

    # Test by vuln_type
    by_vuln = aggregate_by_vuln_type(mock_results)
    print(format_by_vuln_type_table(by_vuln, 'strict'))

    print("\n" + "=" * 70)
    print("Testing complete.")
