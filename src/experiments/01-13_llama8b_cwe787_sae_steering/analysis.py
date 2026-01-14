"""
Analysis Module for Experiment 3

Handles aggregation, statistics, and output generation for both:
- Part 3A: Precision Steering Head-to-Head
- Part 3B: Forced-Choice Logit Gap Validation
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from scipy import stats as scipy_stats

from experiment_config import (
    DATA_DIR, FOLD_RESULTS_DIR, FIGURES_DIR,
    OTHER_THRESHOLD, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_CI_LEVEL,
)


# =============================================================================
# PART 3A ANALYSIS
# =============================================================================

def load_3A_results(results_path: Path = None) -> Dict:
    """Load Part 3A results from JSON."""
    if results_path is None:
        # Find most recent
        results_files = list(DATA_DIR.glob("results_3A_*.json"))
        if not results_files:
            raise FileNotFoundError("No 3A results found")
        results_path = sorted(results_files)[-1]

    with open(results_path) as f:
        return json.load(f)


def load_all_fold_results(timestamp: str = None) -> List[Dict]:
    """Load all fold results from Part 3A."""
    if timestamp is None:
        # Find most recent timestamp
        fold_files = list(FOLD_RESULTS_DIR.glob("fold_*_*.json"))
        if not fold_files:
            raise FileNotFoundError("No fold results found")
        # Extract timestamp from filename
        timestamp = sorted(fold_files)[-1].stem.split('_')[-1]

    # Load all folds with this timestamp
    fold_results = []
    for fold_file in sorted(FOLD_RESULTS_DIR.glob(f"fold_*_{timestamp}.json")):
        with open(fold_file) as f:
            fold_results.append(json.load(f))

    return fold_results


def create_per_example_csv(fold_results: List[Dict], output_path: Path = None) -> pd.DataFrame:
    """
    Create per-example CSV with all generations.

    Columns: fold, method, setting, prompt_id, gen_idx, output_snippet,
             strict_label, expanded_label, is_refusal, vuln_type
    """
    rows = []

    for fold in fold_results:
        fold_id = fold['fold_id']

        for method, method_results in fold['method_results'].items():
            for setting, items in method_results.items():
                for item in items:
                    rows.append({
                        'fold': fold_id,
                        'method': method,
                        'setting': setting,
                        'prompt_id': item['id'],
                        'gen_idx': item['gen_idx'],
                        'output_snippet': item['output'][:200],
                        'strict_label': item['strict_label'],
                        'expanded_label': item['expanded_label'],
                        'is_refusal': item['is_refusal'],
                        'vuln_type': item['vulnerability_type'],
                    })

    df = pd.DataFrame(rows)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved per-example CSV: {output_path}")

    return df


def create_aggregates_csv(results: Dict, output_path: Path = None) -> pd.DataFrame:
    """
    Create aggregated CSV with summary statistics.

    Columns: method, setting, n, strict_secure_rate, strict_insecure_rate,
             strict_other_rate, expanded_secure_rate, expanded_insecure_rate,
             expanded_other_rate, refusal_rate
    """
    rows = []

    for method, method_results in results['aggregated'].items():
        for setting, stats in method_results.items():
            rows.append({
                'method': method,
                'setting': setting,
                'n': stats['n'],
                'strict_secure_rate': stats['strict_secure_rate'],
                'strict_insecure_rate': stats['strict_insecure_rate'],
                'strict_other_rate': stats['strict_other_rate'],
                'expanded_secure_rate': stats['expanded_secure_rate'],
                'expanded_insecure_rate': stats['expanded_insecure_rate'],
                'expanded_other_rate': stats['expanded_other_rate'],
                'refusal_rate': stats['refusal_rate'],
            })

    df = pd.DataFrame(rows)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved aggregates CSV: {output_path}")

    return df


def find_best_operating_points(
    aggregates_df: pd.DataFrame,
    other_threshold: float = OTHER_THRESHOLD,
    scoring: str = 'expanded',
) -> pd.DataFrame:
    """
    Find best operating point for each method under cost threshold.

    Returns DataFrame with best settings where Other% <= threshold.
    """
    other_col = f'{scoring}_other_rate'
    secure_col = f'{scoring}_secure_rate'

    # Filter by threshold
    filtered = aggregates_df[aggregates_df[other_col] <= other_threshold].copy()

    if len(filtered) == 0:
        print(f"WARNING: No settings meet Other% <= {other_threshold*100:.0f}%")
        # Fall back to minimum other rate
        filtered = aggregates_df.copy()

    # Find best per method
    best = filtered.loc[filtered.groupby('method')[secure_col].idxmax()]

    return best[['method', 'setting', 'n', secure_col, other_col, 'refusal_rate']]


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_fn,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
) -> tuple:
    """Compute bootstrap confidence interval for a statistic."""
    n = len(data)
    bootstrap_stats = []

    for _ in range(n_resamples):
        resample_idx = np.random.choice(n, size=n, replace=True)
        resample = data[resample_idx]
        bootstrap_stats.append(statistic_fn(resample))

    bootstrap_stats = np.array(bootstrap_stats)
    alpha = 1 - ci_level
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return lower, upper


def generate_3A_summary(results: Dict, output_path: Path = None) -> str:
    """Generate markdown summary for Part 3A."""
    timestamp = results.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))

    lines = [
        "# Experiment 3A Summary: Precision Steering Head-to-Head",
        "",
        f"**Timestamp**: {timestamp}",
        f"**Model**: {results['config']['model']}",
        f"**Generations per prompt**: {results['config']['generations_per_prompt']}",
        "",
        "## Methods Compared",
        "",
        "| Method | Description |",
        "|--------|-------------|",
        "| M1 | Mean-diff at L31 |",
        "| M2a | SAE L31:1895 (single feature) |",
        "| M2b | SAE L30:10391 (single feature) |",
        "| M3a | Top-5 SAE features |",
        "| M3b | Top-10 SAE features |",
        "",
        "## Aggregated Results (EXPANDED Scoring)",
        "",
        "| Method | Setting | Secure% | Insecure% | Other% | Refusal% |",
        "|--------|---------|---------|-----------|--------|----------|",
    ]

    for method in sorted(results['aggregated'].keys()):
        for setting, stats in sorted(results['aggregated'][method].items()):
            lines.append(
                f"| {method} | {setting} | "
                f"{stats['expanded_secure_rate']*100:.1f}% | "
                f"{stats['expanded_insecure_rate']*100:.1f}% | "
                f"{stats['expanded_other_rate']*100:.1f}% | "
                f"{stats['refusal_rate']*100:.1f}% |"
            )

    lines.extend([
        "",
        "## Best Operating Points (Other% ≤ 10%)",
        "",
    ])

    # Find best points
    aggregates_df = pd.DataFrame([
        {'method': m, 'setting': s, **v}
        for m, settings in results['aggregated'].items()
        for s, v in settings.items()
    ])

    best = find_best_operating_points(aggregates_df)

    lines.extend([
        "| Method | Setting | Secure% | Other% |",
        "|--------|---------|---------|--------|",
    ])

    for _, row in best.iterrows():
        lines.append(
            f"| {row['method']} | {row['setting']} | "
            f"{row['expanded_secure_rate']*100:.1f}% | "
            f"{row['expanded_other_rate']*100:.1f}% |"
        )

    summary = "\n".join(lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(summary)
        print(f"Saved summary: {output_path}")

    return summary


# =============================================================================
# PART 3B ANALYSIS
# =============================================================================

def load_3B_results(results_path: Path = None) -> Dict:
    """Load Part 3B results from JSON."""
    if results_path is None:
        results_files = list(DATA_DIR.glob("results_3B_*.json"))
        if not results_files:
            raise FileNotFoundError("No 3B results found")
        results_path = sorted(results_files)[-1]

    with open(results_path) as f:
        return json.load(f)


def load_3B_logits(logits_path: Path = None) -> pd.DataFrame:
    """Load logit gap CSV."""
    if logits_path is None:
        logit_files = list(DATA_DIR.glob("results_3B_logits_*.csv"))
        if not logit_files:
            raise FileNotFoundError("No 3B logit results found")
        logits_path = sorted(logit_files)[-1]

    return pd.read_csv(logits_path)


def compute_gap_shift_statistics(logits_df: pd.DataFrame) -> Dict:
    """Compute detailed statistics for gap shifts."""
    # Pivot to wide format
    pivot = logits_df.pivot(index='prompt_id', columns='setting', values='gap')

    stats = {}

    for setting in ['S1', 'S2']:
        if setting not in pivot.columns or 'S0' not in pivot.columns:
            continue

        shift = pivot[setting] - pivot['S0']

        # Wilcoxon signed-rank test
        try:
            stat, p_value = scipy_stats.wilcoxon(shift)
        except:
            stat, p_value = np.nan, np.nan

        # Effect size (Cohen's d)
        d = shift.mean() / shift.std() if shift.std() > 0 else 0

        stats[setting] = {
            'mean_shift': float(shift.mean()),
            'std_shift': float(shift.std()),
            'median_shift': float(shift.median()),
            'positive_frac': float((shift > 0).mean()),
            'wilcoxon_stat': float(stat) if not np.isnan(stat) else None,
            'wilcoxon_p': float(p_value) if not np.isnan(p_value) else None,
            'cohens_d': float(d),
        }

    return stats


def generate_3B_summary(results: Dict, logits_df: pd.DataFrame = None, output_path: Path = None) -> str:
    """Generate markdown summary for Part 3B."""
    timestamp = results.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
    stats = results.get('results_stats', {})

    lines = [
        "# Experiment 3B Summary: Forced-Choice Logit Gap Validation",
        "",
        f"**Timestamp**: {timestamp}",
        f"**N forced-choice prompts**: {results['config']['n_forced_choice_prompts']}",
        "",
        "## Operating Points Used",
        "",
        f"- **S0**: Baseline (α=0)",
        f"- **S1**: {results['config']['operating_points']['S1']}",
        f"- **S2**: {results['config']['operating_points']['S2']}",
        "",
        "## Logit Gap Statistics",
        "",
        "| Setting | Mean Gap | Std | Median |",
        "|---------|----------|-----|--------|",
    ]

    for setting in ['S0', 'S1', 'S2']:
        s = stats.get(f'{setting}_gap', {})
        lines.append(
            f"| {setting} | {s.get('mean', 0):.2f} | {s.get('std', 0):.2f} | {s.get('median', 0):.2f} |"
        )

    lines.extend([
        "",
        "## Gap Shifts (vs S0 Baseline)",
        "",
        "| Setting | Mean Shift | % Positive | Cohen's d |",
        "|---------|------------|------------|-----------|",
    ])

    for setting in ['S1', 'S2']:
        shift = stats.get(f'{setting}_shift', {})
        lines.append(
            f"| {setting} | +{shift.get('mean', 0):.2f} | "
            f"{shift.get('positive_frac', 0)*100:.1f}% | "
            f"{compute_cohens_d(shift):.2f} |"
        )

    lines.extend([
        "",
        "## Free Generation Results",
        "",
        "| Setting | Secure% | Insecure% | Other% |",
        "|---------|---------|-----------|--------|",
    ])

    for setting in ['S0', 'S1', 'S2']:
        fg = stats.get(f'{setting}_freegen', {})
        lines.append(
            f"| {setting} | {fg.get('secure_rate', 0)*100:.1f}% | "
            f"{fg.get('insecure_rate', 0)*100:.1f}% | "
            f"{fg.get('other_rate', 0)*100:.1f}% |"
        )

    if stats.get('gap_secure_correlation') is not None:
        lines.extend([
            "",
            f"## Correlation: Gap vs Secure Label",
            "",
            f"**Pearson r = {stats['gap_secure_correlation']:.3f}**",
            "",
            "Positive correlation indicates higher logit gap → more likely secure output.",
        ])

    lines.extend([
        "",
        "## Interpretation",
        "",
        "The logit gap Δ = logit(safe) - logit(unsafe) measures the model's local preference "
        "for the safe API at the decision point. Positive shifts under steering indicate "
        "the steering vector successfully pushes the decision boundary toward safe APIs.",
    ])

    summary = "\n".join(lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(summary)
        print(f"Saved summary: {output_path}")

    return summary


def compute_cohens_d(shift_stats: Dict) -> float:
    """Compute Cohen's d from shift statistics."""
    mean = shift_stats.get('mean', 0)
    std = shift_stats.get('std', 1)
    return mean / std if std > 0 else 0


# =============================================================================
# MANIFEST GENERATION
# =============================================================================

def create_manifest_3A(
    results: Dict,
    fold_results: List[Dict],
    output_path: Path = None,
) -> Dict:
    """Create manifest for Part 3A with full configuration details."""
    manifest = {
        'timestamp': results.get('timestamp'),
        'experiment': '3A_precision_steering',
        'config': results.get('config', {}),
        'calibration_info': {},
        'top_k_selections': {},
    }

    # Collect calibration info from folds
    for fold in fold_results:
        fold_id = fold['fold_id']

        for method, calib in fold.get('calibration_info', {}).items():
            if method not in manifest['calibration_info']:
                manifest['calibration_info'][method] = {}
            manifest['calibration_info'][method][fold_id] = calib

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved manifest: {output_path}")

    return manifest


def create_manifest_3B(results: Dict, output_path: Path = None) -> Dict:
    """Create manifest for Part 3B."""
    manifest = {
        'timestamp': results.get('timestamp'),
        'experiment': '3B_forced_choice_logit_gap',
        'config': results.get('config', {}),
        'forced_choice_stats': results.get('forced_choice_stats', {}),
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved manifest: {output_path}")

    return manifest


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Analysis module loaded.")
    print("Available functions:")
    print("  - load_3A_results(), create_per_example_csv(), generate_3A_summary()")
    print("  - load_3B_results(), generate_3B_summary()")
    print("  - find_best_operating_points()")
