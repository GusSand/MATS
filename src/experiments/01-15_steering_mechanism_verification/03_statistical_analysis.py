#!/usr/bin/env python3
"""
Steering Mechanism Verification - Statistical Analysis

Performs:
1. Effect size computation (Cohen's d)
2. Significance tests (t-tests, Mann-Whitney U)
3. Bootstrap confidence intervals
4. Hypothesis testing for mechanism verification
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_config import (
    RESULTS_DIR, LAYERS_TO_EXTRACT, STEERING_LAYER,
    BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_CI_LEVEL,
)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# =============================================================================
# STATISTICAL FUNCTIONS
# =============================================================================

def cohens_d(group1, group2):
    """
    Compute Cohen's d effect size.

    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large
    """
    n1, n2 = len(group1), len(group2)

    if n1 < 2 or n2 < 2:
        return 0.0

    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))

    if pooled_std < 1e-8:
        return 0.0

    return (np.mean(group2) - np.mean(group1)) / pooled_std


def interpret_cohens_d(d):
    """Interpret Cohen's d effect size."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence interval for the mean."""
    if len(data) < 2:
        m = np.mean(data) if len(data) > 0 else 0
        return m, m

    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))

    lower = np.percentile(means, (1-ci)/2 * 100)
    upper = np.percentile(means, (1+ci)/2 * 100)
    return float(lower), float(upper)


def compare_conditions(data_a, data_b, name):
    """
    Comprehensive statistical comparison between two conditions.

    Returns dict with means, effect sizes, and significance tests.
    """
    data_a = np.array(data_a)
    data_b = np.array(data_b)

    # Basic stats
    mean_a, std_a = np.mean(data_a), np.std(data_a)
    mean_b, std_b = np.mean(data_b), np.std(data_b)

    # Effect size
    d = cohens_d(data_a, data_b)

    # T-test (parametric)
    if len(data_a) >= 2 and len(data_b) >= 2:
        t_stat, p_ttest = stats.ttest_ind(data_a, data_b)
    else:
        t_stat, p_ttest = 0.0, 1.0

    # Mann-Whitney U (non-parametric)
    if len(data_a) >= 2 and len(data_b) >= 2:
        try:
            u_stat, p_mannwhitney = stats.mannwhitneyu(data_a, data_b, alternative='two-sided')
        except ValueError:
            u_stat, p_mannwhitney = 0.0, 1.0
    else:
        u_stat, p_mannwhitney = 0.0, 1.0

    # Bootstrap CIs
    ci_a = bootstrap_ci(data_a, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_CI_LEVEL)
    ci_b = bootstrap_ci(data_b, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_CI_LEVEL)

    return {
        "comparison": name,
        "group_a": {
            "mean": float(mean_a),
            "std": float(std_a),
            "ci_95": list(ci_a),
            "n": len(data_a)
        },
        "group_b": {
            "mean": float(mean_b),
            "std": float(std_b),
            "ci_95": list(ci_b),
            "n": len(data_b)
        },
        "effect_size": {
            "cohens_d": float(d),
            "interpretation": interpret_cohens_d(d)
        },
        "significance": {
            "t_test_statistic": float(t_stat),
            "t_test_p": float(p_ttest),
            "mann_whitney_u": float(u_stat),
            "mann_whitney_p": float(p_mannwhitney),
            "significant_at_05": p_ttest < 0.05,
            "significant_at_01": p_ttest < 0.01,
            "significant_at_001": p_ttest < 0.001
        }
    }


# =============================================================================
# MAIN
# =============================================================================

def find_latest_file(directory, pattern):
    """Find the most recent file matching pattern."""
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}")
    return files[-1]


def main():
    print("=" * 60)
    print("Steering Mechanism Verification - Statistical Analysis")
    print("=" * 60)

    # Find most recent metrics file
    try:
        latest_metrics = find_latest_file(RESULTS_DIR, "metrics_*.json")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run 02_compute_metrics.py first.")
        return

    print(f"\nLoading metrics from: {latest_metrics}")
    with open(latest_metrics, 'r') as f:
        metrics = json.load(f)

    results = {
        "probe_projection_comparisons": {},
        "steering_alignment_stats": {},
        "hypothesis_tests": [],
        "gap_closure": {},
        "sae_feature_comparisons": {},
    }

    # ==========================================================================
    # 1. Probe projection comparisons at each layer
    # ==========================================================================
    print("\n" + "-" * 40)
    print("1. Probe Projection Comparisons")
    print("-" * 40)

    for layer in LAYERS_TO_EXTRACT:
        layer_key = str(layer)

        proj_a = metrics["probe_projections"]["A"].get(layer_key, [])
        proj_b = metrics["probe_projections"]["B"].get(layer_key, [])
        proj_c = metrics["probe_projections"]["C"].get(layer_key, [])

        if proj_a and proj_b and proj_c:
            # A vs B: Does steering shift toward secure?
            comp_ab = compare_conditions(proj_a, proj_b, f"L{layer}: A(baseline) vs B(steered)")

            # B vs C: How close is steered to natural secure?
            comp_bc = compare_conditions(proj_b, proj_c, f"L{layer}: B(steered) vs C(natural)")

            # A vs C: Full gap between baseline and natural
            comp_ac = compare_conditions(proj_a, proj_c, f"L{layer}: A(baseline) vs C(natural)")

            results["probe_projection_comparisons"][layer_key] = {
                "A_vs_B": comp_ab,
                "B_vs_C": comp_bc,
                "A_vs_C": comp_ac
            }

            # Key metric: How much of the A->C gap does steering close?
            gap_ac = np.mean(proj_c) - np.mean(proj_a)
            shift_ab = np.mean(proj_b) - np.mean(proj_a)

            if abs(gap_ac) > 1e-8:
                gap_closure = shift_ab / gap_ac * 100
            else:
                gap_closure = 0.0

            results["gap_closure"][layer_key] = {
                "gap_A_to_C": float(gap_ac),
                "shift_A_to_B": float(shift_ab),
                "closure_percent": float(gap_closure)
            }

            print(f"\nLayer {layer}:")
            print(f"  A (baseline):    {np.mean(proj_a):.4f} +/- {np.std(proj_a):.4f}")
            print(f"  B (steered):     {np.mean(proj_b):.4f} +/- {np.std(proj_b):.4f}")
            print(f"  C (natural):     {np.mean(proj_c):.4f} +/- {np.std(proj_c):.4f}")
            print(f"  Gap closure:     {gap_closure:.1f}%")
            print(f"  A->B effect:     d={comp_ab['effect_size']['cohens_d']:.3f} ({comp_ab['effect_size']['interpretation']})")
            print(f"  A->B p-value:    {comp_ab['significance']['t_test_p']:.2e}")

    # ==========================================================================
    # 2. Steering alignment statistics
    # ==========================================================================
    print("\n" + "-" * 40)
    print("2. Steering Alignment Statistics")
    print("-" * 40)

    if metrics.get("steering_alignment"):
        alignments = [a["alignment"] for a in metrics["steering_alignment"]]
        ratios = [a["alignment_ratio"] for a in metrics["steering_alignment"]]
        parallel_mags = [a["parallel_magnitude"] for a in metrics["steering_alignment"]]
        orthogonal_mags = [a["orthogonal_magnitude"] for a in metrics["steering_alignment"]]

        results["steering_alignment_stats"] = {
            "alignment": {
                "mean": float(np.mean(alignments)),
                "std": float(np.std(alignments)),
                "ci_95": list(bootstrap_ci(alignments, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_CI_LEVEL))
            },
            "ratio": {
                "mean": float(np.mean(ratios)),
                "std": float(np.std(ratios)),
                "ci_95": list(bootstrap_ci(ratios, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_CI_LEVEL))
            },
            "parallel_magnitude": {
                "mean": float(np.mean(parallel_mags)),
                "std": float(np.std(parallel_mags))
            },
            "orthogonal_magnitude": {
                "mean": float(np.mean(orthogonal_mags)),
                "std": float(np.std(orthogonal_mags))
            }
        }

        print(f"Alignment:      {np.mean(alignments):.4f} +/- {np.std(alignments):.4f}")
        print(f"Ratio (||/|_|): {np.mean(ratios):.4f} +/- {np.std(ratios):.4f}")
        print(f"Parallel mag:   {np.mean(parallel_mags):.4f}")
        print(f"Orthogonal mag: {np.mean(orthogonal_mags):.4f}")

    # ==========================================================================
    # 3. Hypothesis tests
    # ==========================================================================
    print("\n" + "-" * 40)
    print("3. Hypothesis Tests")
    print("-" * 40)

    # H1: Steering increases probe projection at L31 (B > A)
    layer_key = str(STEERING_LAYER)
    proj_a = metrics["probe_projections"]["A"].get(layer_key, [])
    proj_b = metrics["probe_projections"]["B"].get(layer_key, [])

    if proj_a and proj_b:
        # One-sided t-test: B > A
        t_stat, p_two_sided = stats.ttest_ind(proj_a, proj_b)
        # For one-sided, divide by 2 if t_stat is in expected direction
        if np.mean(proj_b) > np.mean(proj_a):
            p_one_sided = p_two_sided / 2
        else:
            p_one_sided = 1 - p_two_sided / 2

        h1_result = {
            "hypothesis": f"H1: Steering increases probe projection at L{STEERING_LAYER} (B > A)",
            "test": "one-sided t-test",
            "t_statistic": float(t_stat),
            "p_value": float(p_one_sided),
            "mean_A": float(np.mean(proj_a)),
            "mean_B": float(np.mean(proj_b)),
            "supported": p_one_sided < 0.05 and np.mean(proj_b) > np.mean(proj_a)
        }
        results["hypothesis_tests"].append(h1_result)
        print(f"\nH1: Steering shifts toward secure direction (B > A at L{STEERING_LAYER})")
        print(f"  mean(A) = {np.mean(proj_a):.4f}, mean(B) = {np.mean(proj_b):.4f}")
        print(f"  t = {t_stat:.3f}, p = {p_one_sided:.2e}")
        print(f"  Supported: {h1_result['supported']}")

    # H2: Alignment is positive (steering moves in intended direction)
    if metrics.get("steering_alignment"):
        alignments = [a["alignment"] for a in metrics["steering_alignment"]]
        t_stat, p_two_sided = stats.ttest_1samp(alignments, 0)
        # One-sided test for alignment > 0
        if np.mean(alignments) > 0:
            p_one_sided = p_two_sided / 2
        else:
            p_one_sided = 1 - p_two_sided / 2

        h2_result = {
            "hypothesis": "H2: Steering alignment is positive (activations move toward steering direction)",
            "test": "one-sample t-test vs 0",
            "t_statistic": float(t_stat),
            "p_value": float(p_one_sided),
            "mean_alignment": float(np.mean(alignments)),
            "supported": p_one_sided < 0.05 and np.mean(alignments) > 0
        }
        results["hypothesis_tests"].append(h2_result)
        print(f"\nH2: Alignment > 0 (change is in steering direction)")
        print(f"  mean(alignment) = {np.mean(alignments):.4f}")
        print(f"  t = {t_stat:.3f}, p = {p_one_sided:.2e}")
        print(f"  Supported: {h2_result['supported']}")

    # H3: Gap closure > 30% at steering layer
    gap_info = results["gap_closure"].get(str(STEERING_LAYER), {})
    if gap_info:
        closure = gap_info.get("closure_percent", 0)
        h3_result = {
            "hypothesis": f"H3: Steering closes >30% of A->C gap at L{STEERING_LAYER}",
            "test": "direct comparison",
            "closure_percent": float(closure),
            "threshold": 30.0,
            "supported": closure > 30.0
        }
        results["hypothesis_tests"].append(h3_result)
        print(f"\nH3: Gap closure > 30% at L{STEERING_LAYER}")
        print(f"  closure = {closure:.1f}%")
        print(f"  Supported: {h3_result['supported']}")

    # ==========================================================================
    # 4. SAE feature comparisons (if available)
    # ==========================================================================
    if metrics.get("sae_features") and metrics["sae_features"]["A"].get("promoting"):
        print("\n" + "-" * 40)
        print("4. SAE Feature Analysis")
        print("-" * 40)

        # Aggregate promoting features across samples
        def aggregate_sae_features(condition, feature_type):
            """Aggregate feature activations for a condition."""
            features_list = metrics["sae_features"][condition][feature_type]
            aggregated = {}
            for sample_features in features_list:
                for layer, feats in sample_features.items():
                    if layer not in aggregated:
                        aggregated[layer] = {}
                    for fid, val in feats.items():
                        if val is not None:
                            if fid not in aggregated[layer]:
                                aggregated[layer][fid] = []
                            aggregated[layer][fid].append(val)
            return aggregated

        for feature_type in ["promoting", "suppressing"]:
            feats_a = aggregate_sae_features("A", feature_type)
            feats_b = aggregate_sae_features("B", feature_type)
            feats_c = aggregate_sae_features("C", feature_type)

            results["sae_feature_comparisons"][feature_type] = {}

            print(f"\n{feature_type.upper()} features:")
            for layer in feats_a:
                for fid in feats_a[layer]:
                    vals_a = feats_a[layer].get(fid, [])
                    vals_b = feats_b.get(layer, {}).get(fid, [])
                    vals_c = feats_c.get(layer, {}).get(fid, [])

                    if vals_a and vals_b:
                        key = f"L{layer}:{fid}"
                        comp = compare_conditions(vals_a, vals_b, f"{key} A vs B")
                        results["sae_feature_comparisons"][feature_type][key] = comp

                        direction = "increase" if np.mean(vals_b) > np.mean(vals_a) else "decrease"
                        print(f"  {key}: A={np.mean(vals_a):.3f} -> B={np.mean(vals_b):.3f} ({direction}), d={comp['effect_size']['cohens_d']:.2f}")

    # ==========================================================================
    # SUCCESS CRITERIA EVALUATION
    # ==========================================================================
    print("\n" + "=" * 60)
    print("SUCCESS CRITERIA EVALUATION")
    print("=" * 60)

    # Initialize results
    results["success_criteria"] = {
        "primary": {},
        "secondary": {},
        "tertiary": {},
        "overall_verdict": None
    }

    # -------------------------------------------------------------------------
    # PRIMARY (Must Have): Probe projection B > A at L31, p < 0.05, d > 0.5
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("PRIMARY CRITERION (Must Have)")
    print("-" * 40)
    print("Probe projection at L31: B > A with p < 0.05 AND Cohen's d > 0.5")

    layer_key = str(STEERING_LAYER)
    proj_a = metrics["probe_projections"]["A"].get(layer_key, [])
    proj_b = metrics["probe_projections"]["B"].get(layer_key, [])

    if proj_a and proj_b:
        comp_ab = results["probe_projection_comparisons"].get(layer_key, {}).get("A_vs_B", {})

        p_value = comp_ab.get("significance", {}).get("t_test_p", 1.0)
        cohens_d_value = comp_ab.get("effect_size", {}).get("cohens_d", 0.0)
        mean_a = np.mean(proj_a)
        mean_b = np.mean(proj_b)

        # Check all conditions
        direction_correct = mean_b > mean_a
        p_significant = p_value < 0.05
        effect_large_enough = abs(cohens_d_value) > 0.5

        primary_pass = direction_correct and p_significant and effect_large_enough

        results["success_criteria"]["primary"] = {
            "criterion": "Probe projection B > A at L31 with p < 0.05 and d > 0.5",
            "mean_A": float(mean_a),
            "mean_B": float(mean_b),
            "direction_correct": direction_correct,
            "p_value": float(p_value),
            "p_significant": p_significant,
            "cohens_d": float(cohens_d_value),
            "effect_large_enough": effect_large_enough,
            "PASS": primary_pass
        }

        print(f"\n  mean(A) = {mean_a:.4f}")
        print(f"  mean(B) = {mean_b:.4f}")
        print(f"  Direction B > A: {'YES' if direction_correct else 'NO'}")
        print(f"  p-value = {p_value:.2e} (threshold: < 0.05): {'PASS' if p_significant else 'FAIL'}")
        print(f"  Cohen's d = {cohens_d_value:.3f} (threshold: > 0.5): {'PASS' if effect_large_enough else 'FAIL'}")
        print(f"\n  >>> PRIMARY CRITERION: {'PASS' if primary_pass else 'FAIL'} <<<")
    else:
        primary_pass = False
        results["success_criteria"]["primary"]["PASS"] = False
        print("\n  ERROR: Missing probe projection data")

    # -------------------------------------------------------------------------
    # SECONDARY (Should Have): Gap closure >= 30% AND Alignment ratio > 1
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("SECONDARY CRITERIA (Should Have)")
    print("-" * 40)

    # Gap closure >= 30%
    print("\n1. Gap closure >= 30%")
    gap_info = results["gap_closure"].get(str(STEERING_LAYER), {})
    gap_closure = gap_info.get("closure_percent", 0.0)
    gap_pass = gap_closure >= 30.0

    results["success_criteria"]["secondary"]["gap_closure"] = {
        "criterion": "Gap closure >= 30%",
        "value": float(gap_closure),
        "threshold": 30.0,
        "PASS": gap_pass
    }

    print(f"   Gap A->C: {gap_info.get('gap_A_to_C', 0):.4f}")
    print(f"   Shift A->B: {gap_info.get('shift_A_to_B', 0):.4f}")
    print(f"   Closure: {gap_closure:.1f}% (threshold: >= 30%): {'PASS' if gap_pass else 'FAIL'}")

    # Alignment ratio > 1
    print("\n2. Steering alignment ratio > 1")
    alignment_stats = results.get("steering_alignment_stats", {})
    ratio_mean = alignment_stats.get("ratio", {}).get("mean", 0.0)
    ratio_pass = ratio_mean > 1.0

    results["success_criteria"]["secondary"]["alignment_ratio"] = {
        "criterion": "Steering alignment ratio > 1",
        "value": float(ratio_mean),
        "threshold": 1.0,
        "PASS": ratio_pass
    }

    print(f"   Mean ratio (parallel/orthogonal): {ratio_mean:.3f}")
    print(f"   Threshold: > 1.0: {'PASS' if ratio_pass else 'FAIL'}")

    secondary_pass = gap_pass and ratio_pass
    results["success_criteria"]["secondary"]["PASS"] = secondary_pass
    print(f"\n  >>> SECONDARY CRITERIA: {'PASS' if secondary_pass else 'FAIL'} <<<")

    # -------------------------------------------------------------------------
    # TERTIARY (Nice to Have): SAE features move in predicted direction
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("TERTIARY CRITERIA (Nice to Have)")
    print("-" * 40)
    print("SAE features move in predicted direction")

    sae_comparisons = results.get("sae_feature_comparisons", {})
    promoting_correct = 0
    promoting_total = 0
    suppressing_correct = 0
    suppressing_total = 0

    if sae_comparisons.get("promoting"):
        for key, comp in sae_comparisons["promoting"].items():
            promoting_total += 1
            # Security-promoting should increase A -> B
            if comp["group_b"]["mean"] > comp["group_a"]["mean"]:
                promoting_correct += 1

    if sae_comparisons.get("suppressing"):
        for key, comp in sae_comparisons["suppressing"].items():
            suppressing_total += 1
            # Security-suppressing should decrease A -> B
            if comp["group_b"]["mean"] < comp["group_a"]["mean"]:
                suppressing_correct += 1

    total_features = promoting_total + suppressing_total
    correct_features = promoting_correct + suppressing_correct

    if total_features > 0:
        tertiary_rate = correct_features / total_features
        tertiary_pass = tertiary_rate > 0.5  # Majority in correct direction
    else:
        tertiary_rate = 0.0
        tertiary_pass = None  # N/A - no SAE data

    results["success_criteria"]["tertiary"] = {
        "criterion": "SAE features move in predicted direction",
        "promoting_correct": promoting_correct,
        "promoting_total": promoting_total,
        "suppressing_correct": suppressing_correct,
        "suppressing_total": suppressing_total,
        "rate": float(tertiary_rate) if total_features > 0 else None,
        "PASS": tertiary_pass
    }

    if total_features > 0:
        print(f"\n   Promoting features (should increase A->B): {promoting_correct}/{promoting_total}")
        print(f"   Suppressing features (should decrease A->B): {suppressing_correct}/{suppressing_total}")
        print(f"   Overall correct: {correct_features}/{total_features} ({tertiary_rate:.1%})")
        print(f"\n  >>> TERTIARY CRITERION: {'PASS' if tertiary_pass else 'FAIL'} <<<")
    else:
        print("\n   No SAE feature data available (SAE analysis may have been skipped)")
        print("\n  >>> TERTIARY CRITERION: N/A <<<")

    # -------------------------------------------------------------------------
    # OVERALL VERDICT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("OVERALL VERDICT")
    print("=" * 60)

    if primary_pass:
        if secondary_pass:
            verdict = "STRONG POSITIVE - Mechanism verified"
            verdict_detail = "Primary AND secondary criteria met. Steering works through predicted mechanism."
        else:
            verdict = "POSITIVE - Core mechanism verified"
            verdict_detail = "Primary criterion met. Secondary criteria partially met - some caveats apply."
    else:
        verdict = "NEGATIVE - Mechanism NOT verified"
        verdict_detail = "Primary criterion failed. Steering may work through a different mechanism."

    results["success_criteria"]["overall_verdict"] = {
        "verdict": verdict,
        "detail": verdict_detail,
        "primary_pass": primary_pass,
        "secondary_pass": secondary_pass,
        "tertiary_pass": tertiary_pass
    }

    print(f"\n  {verdict}")
    print(f"  {verdict_detail}")
    print(f"\n  Primary (Must Have):   {'PASS' if primary_pass else 'FAIL'}")
    print(f"  Secondary (Should Have): {'PASS' if secondary_pass else 'FAIL'}")
    print(f"  Tertiary (Nice to Have): {'PASS' if tertiary_pass else 'N/A' if tertiary_pass is None else 'FAIL'}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"statistics_{timestamp}.json"

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print(f"\nStatistical analysis saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
