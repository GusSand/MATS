#!/usr/bin/env python3
"""
Statistical Tables: Bootstrap 95% CIs for all main results.

Computes per-fold secure rates and bootstraps CIs across LOBO folds.
Covers all completed model/CWE experiments.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

np.random.seed(42)

EXPERIMENTS_ROOT = Path(__file__).parent.parent


def load_fold_results(fold_dir, timestamp_suffix):
    """Load all fold result files matching a timestamp."""
    pattern = f"fold_pair_*_{timestamp_suffix}.json"
    fold_files = sorted(fold_dir.glob(pattern))
    folds = []
    for fp in fold_files:
        with open(fp) as f:
            folds.append(json.load(f))
    return folds


def get_fold_secure_rates(folds, alpha_key):
    """Extract per-fold strict secure rate at a given alpha."""
    rates = []
    for fold in folds:
        if 'summary' in fold and alpha_key in fold['summary']:
            rates.append(fold['summary'][alpha_key]['strict']['secure_rate'])
        elif alpha_key in fold.get('alpha_results', {}):
            items = fold['alpha_results'][alpha_key]
            n = len(items)
            secure = sum(1 for r in items if r['strict_label'] == 'secure')
            rates.append(secure / n if n > 0 else 0)
    return np.array(rates)


def bootstrap_ci(fold_rates, n_bootstrap=10000, ci=0.95):
    """Bootstrap CI for mean of fold-level rates."""
    n_folds = len(fold_rates)
    if n_folds == 0:
        return 0, 0, 0

    boot_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = np.random.choice(fold_rates, size=n_folds, replace=True)
        boot_means[i] = sample.mean()

    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, alpha * 100)
    hi = np.percentile(boot_means, (1 - alpha) * 100)
    return float(fold_rates.mean()), float(lo), float(hi)


def main():
    print("=" * 70)
    print("Statistical Tables: Bootstrap 95% CIs (10,000 resamples)")
    print("=" * 70)

    # Define all experiments
    experiments = [
        {
            'name': 'Llama-8B CWE-787',
            'fold_dir': EXPERIMENTS_ROOT / '01-12_llama8b_cwe787_lobo_steering/data/fold_results',
            'timestamp': '20260113_171820',
            'baseline_alpha': '0.0',
            'best_alpha': '3.5',
        },
        {
            'name': 'Mistral-7B CWE-787',
            'fold_dir': EXPERIMENTS_ROOT / '02-05_cross_model_cwe787_steering/experiment_4a_mistral7b/data/fold_results',
            'timestamp': '20260205_045755',
            'baseline_alpha': '0.0',
            'best_alpha': '3.5',
        },
        {
            'name': 'Llama-70B CWE-787',
            'fold_dir': EXPERIMENTS_ROOT / '02-05_cross_model_cwe787_steering/experiment_4b_llama70b/data/fold_results',
            'timestamp': '20260205_111622',
            'baseline_alpha': '0.0',
            'best_alpha': '4.0',
        },
        {
            'name': 'Llama-8B CWE-119',
            'fold_dir': EXPERIMENTS_ROOT / '02-05_cross_cwe_steering/experiment_cwe119_llama8b/data/fold_results',
            'timestamp': '20260205_173625',
            'baseline_alpha': '0.0',
            'best_alpha': '4.0',
        },
        {
            'name': 'Llama-8B CWE-134 (pilot, 2 folds)',
            'fold_dir': EXPERIMENTS_ROOT / '02-05_cross_cwe_steering/experiment_cwe134_llama8b/data/fold_results',
            'timestamp': '20260205_231906',
            'baseline_alpha': '0.0',
            'best_alpha': '1.5',
            'prefix': 'pilot_fold_pair',
        },
    ]

    all_results = []

    for exp in experiments:
        fold_dir = exp['fold_dir']
        ts = exp['timestamp']
        prefix = exp.get('prefix', 'fold_pair')

        pattern = f"{prefix}_*_{ts}.json"
        fold_files = sorted(fold_dir.glob(pattern))

        folds = []
        for fp in fold_files:
            with open(fp) as f:
                folds.append(json.load(f))

        n_folds = len(folds)
        if n_folds == 0:
            print(f"\n  WARNING: No fold files for {exp['name']} (pattern: {pattern})")
            continue

        # Baseline rates
        baseline_rates = get_fold_secure_rates(folds, exp['baseline_alpha'])
        bl_mean, bl_lo, bl_hi = bootstrap_ci(baseline_rates)

        # Best alpha rates
        best_rates = get_fold_secure_rates(folds, exp['best_alpha'])
        best_mean, best_lo, best_hi = bootstrap_ci(best_rates)

        # Improvement
        improvement_per_fold = best_rates - baseline_rates
        imp_mean, imp_lo, imp_hi = bootstrap_ci(improvement_per_fold)

        result = {
            'name': exp['name'],
            'n_folds': n_folds,
            'best_alpha': exp['best_alpha'],
            'baseline': {'mean': bl_mean, 'ci_lo': bl_lo, 'ci_hi': bl_hi},
            'steered': {'mean': best_mean, 'ci_lo': best_lo, 'ci_hi': best_hi},
            'improvement': {'mean': imp_mean, 'ci_lo': imp_lo, 'ci_hi': imp_hi},
            'fold_baseline_rates': baseline_rates.tolist(),
            'fold_steered_rates': best_rates.tolist(),
        }
        all_results.append(result)

        print(f"\n  {exp['name']} ({n_folds} folds, best alpha={exp['best_alpha']})")
        print(f"    Baseline:    {bl_mean*100:5.1f}%  [{bl_lo*100:.1f}%, {bl_hi*100:.1f}%]")
        print(f"    Steered:     {best_mean*100:5.1f}%  [{best_lo*100:.1f}%, {best_hi*100:.1f}%]")
        print(f"    Improvement: {imp_mean*100:+5.1f}pp [{imp_lo*100:+.1f}pp, {imp_hi*100:+.1f}pp]")
        print(f"    Per-fold steered: {[f'{r:.1%}' for r in best_rates]}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Experiment':<30} {'Folds':>5} {'Baseline':>10} {'Steered':>10} {'95% CI':>20}")
    print("-" * 77)

    for r in all_results:
        bl = f"{r['baseline']['mean']*100:.1f}%"
        st = f"{r['steered']['mean']*100:.1f}%"
        ci = f"[{r['steered']['ci_lo']*100:.1f}%, {r['steered']['ci_hi']*100:.1f}%]"
        print(f"{r['name']:<30} {r['n_folds']:>5} {bl:>10} {st:>10} {ci:>20}")

    # Full alpha sweep for each experiment
    print("\n" + "=" * 70)
    print("FULL ALPHA SWEEP WITH CIs")
    print("=" * 70)

    for exp in experiments:
        fold_dir = exp['fold_dir']
        ts = exp['timestamp']
        prefix = exp.get('prefix', 'fold_pair')

        pattern = f"{prefix}_*_{ts}.json"
        fold_files = sorted(fold_dir.glob(pattern))

        folds = []
        for fp in fold_files:
            with open(fp) as f:
                folds.append(json.load(f))

        if not folds:
            continue

        # Get all alpha values from first fold
        if 'summary' in folds[0]:
            alphas = sorted(folds[0]['summary'].keys(), key=float)
        else:
            alphas = sorted(folds[0].get('alpha_results', {}).keys(), key=float)

        print(f"\n  {exp['name']} ({len(folds)} folds)")
        print(f"  {'Alpha':>6} {'Mean':>8} {'95% CI':>20} {'Per-fold rates'}")
        print(f"  {'-'*60}")

        for alpha_key in alphas:
            rates = get_fold_secure_rates(folds, alpha_key)
            mean, lo, hi = bootstrap_ci(rates)
            rates_str = " ".join(f"{r:.0%}" for r in rates)
            print(f"  {float(alpha_key):>6.1f} {mean*100:>7.1f}% [{lo*100:>5.1f}%, {hi*100:>5.1f}%]  {rates_str}")

    # Save results
    output_dir = Path(__file__).parent / "cross_cwe_analysis" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_data = {
        'n_bootstrap': 10000,
        'ci_level': 0.95,
        'seed': 42,
        'results': all_results,
    }
    results_path = output_dir / "statistical_tables.json"
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
