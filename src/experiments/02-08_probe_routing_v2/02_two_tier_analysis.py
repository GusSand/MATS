#!/usr/bin/env python3
"""
Experiment 8.5 — Part B: 2-Tier Deployment Architecture Analysis

Computes the 2-tier vs 3-way routing comparison using existing data
from Experiment 8 Phases 1-3 plus Part A probe results.

NO GPU needed — pure analysis of existing results.

Data sources:
- Phase 1 (baseline): neutral_baseline_results_*.json
- Phase 2 (per-CWE steering): neutral_steered_results_*.json
- Phase 3 (cross-CWE): neutral_cross_cwe_results_*.json
- Part A: 3way_probe_results_*.json (binary probe accuracy)
"""

import json
import glob
import numpy as np
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
ANALYSIS_DIR = SCRIPT_DIR / "analysis"

# Experiment 8 results
EXP8_DIR = Path("/home/paperspace/MATS/src/experiments/02-05_cross_cwe_steering/neutral_eval/results")


def load_latest(directory, pattern):
    """Load the most recent JSON file matching pattern."""
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}")
    path = files[-1]
    print(f"  Loading: {path.name}")
    with open(path) as f:
        return json.load(f)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Experiment 8.5 — Part B: 2-Tier Architecture Analysis")
    print("=" * 70)

    # ─── Load existing results ───────────────────────────────────────────
    print("\nLoading Experiment 8 results...")
    phase1 = load_latest(EXP8_DIR, "neutral_baseline_results_*.json")
    phase2 = load_latest(EXP8_DIR, "neutral_steered_results_*.json")
    phase3 = load_latest(EXP8_DIR, "neutral_cross_cwe_results_*.json")

    # Try loading Part A results
    part_a = None
    try:
        part_a = load_latest(RESULTS_DIR, "3way_probe_results_*.json")
        print("  Part A results loaded.")
    except FileNotFoundError:
        print("  Part A results not found — will compute without probe accuracy.")

    # ─── Extract key numbers ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("KEY DATA POINTS")
    print("=" * 70)

    # Phase 1: Baselines (no steering)
    baselines = phase3["neutral_baselines"]
    print(f"\nPhase 1 — Baselines (no steering):")
    for cwe, rate in baselines.items():
        print(f"  {cwe}: {rate*100:.1f}%")

    # Phase 2: Best per-CWE steering rates
    # Extract from Phase 2 summary
    best_alphas = phase3["best_alphas"]
    print(f"\nPhase 2 — Best alphas: {best_alphas}")

    # Phase 2 best rates — extract from the steered results
    steered_rates = {}
    if "per_cwe_summary" in phase2:
        for cwe_key, cwe_data in phase2["per_cwe_summary"].items():
            best = max(cwe_data["alpha_results"], key=lambda x: x["secure_rate"])
            steered_rates[cwe_key] = best["secure_rate"]
    else:
        # Use known values from research journal
        steered_rates = {
            "CWE-787": 1.000,
            "CWE-119": 0.814,
            "CWE-134": 1.000,
        }
    print(f"\nPhase 2 — Per-CWE steered rates (perfect routing):")
    for cwe, rate in steered_rates.items():
        print(f"  {cwe}: {rate*100:.1f}%")

    # Phase 3: Cross-CWE rates
    cross_cwe = phase3["results"]
    print(f"\nPhase 3 — Cross-CWE rates:")
    for steer_cwe, targets in cross_cwe.items():
        for prompt_cwe, data in targets.items():
            sr = data["secure_rate"]
            delta = data["delta_pp"]
            print(f"  {steer_cwe} vec → {prompt_cwe} prompts: "
                  f"{sr*100:.1f}% (Δ={delta:+.1f}pp)")

    # ═══════════════════════════════════════════════════════════════════════
    # ROUTING STRATEGY COMPARISON
    # ═══════════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print("ROUTING STRATEGY ANALYSIS")
    print(f"{'='*70}")

    # --- Strategy 1: No steering ---
    no_steer = {
        "CWE-787": baselines["CWE-787"],
        "CWE-119": baselines["CWE-119"],
        "CWE-134": baselines["CWE-134"],
    }

    # --- Strategy 2: Perfect 3-way routing ---
    perfect_3way = {
        "CWE-787": steered_rates["CWE-787"],
        "CWE-119": steered_rates["CWE-119"],
        "CWE-134": steered_rates["CWE-134"],
    }

    # --- Strategy 3: 2-Tier routing ---
    # Binary probe: format-string (CWE-134) vs buffer (CWE-787 + CWE-119)
    # If format-string → apply CWE-134 vector
    # If buffer → apply CWE-787 vector (even for CWE-119 prompts)
    two_tier = {
        "CWE-787": steered_rates["CWE-787"],  # Correctly routed to CWE-787
        "CWE-119": cross_cwe["CWE-787"]["CWE-119"]["secure_rate"],  # CWE-787 vec on CWE-119
        "CWE-134": steered_rates["CWE-134"],  # Correctly routed to CWE-134
    }

    # --- Strategy 4: Naive CWE-787 only ---
    # Apply CWE-787 vector to ALL prompts
    naive_787 = {
        "CWE-787": steered_rates["CWE-787"],
        "CWE-119": cross_cwe["CWE-787"]["CWE-119"]["secure_rate"],
        "CWE-134": cross_cwe["CWE-787"]["CWE-134"]["secure_rate"],
    }

    # --- Strategy 5: Naive CWE-119 only ---
    naive_119 = {
        "CWE-787": cross_cwe["CWE-119"]["CWE-787"]["secure_rate"],
        "CWE-119": steered_rates["CWE-119"],
        "CWE-134": cross_cwe["CWE-119"]["CWE-134"]["secure_rate"],
    }

    strategies = {
        "No steering": no_steer,
        "Perfect 3-way": perfect_3way,
        "2-Tier (binary probe)": two_tier,
        "Naive CWE-787 only": naive_787,
        "Naive CWE-119 only": naive_119,
    }

    # ─── Table 2: Security Impact ────────────────────────────────────────
    print(f"\n{'='*80}")
    print("TABLE 2: ROUTING STRATEGY — SECURITY IMPACT")
    print(f"{'='*80}")
    print(f"{'Strategy':<25} {'CWE-787':>10} {'CWE-119':>10} {'CWE-134':>10} {'Average':>10}")
    print("-" * 70)

    for name, rates in strategies.items():
        avg = np.mean(list(rates.values()))
        print(f"{name:<25} "
              f"{rates['CWE-787']*100:>9.1f}% "
              f"{rates['CWE-119']*100:>9.1f}% "
              f"{rates['CWE-134']*100:>9.1f}% "
              f"{avg*100:>9.1f}%")

    # ─── Cost analysis ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("2-TIER COST ANALYSIS")
    print(f"{'='*70}")

    cost_119 = steered_rates["CWE-119"] - two_tier["CWE-119"]
    cost_avg = np.mean(list(perfect_3way.values())) - np.mean(list(two_tier.values()))
    gain_vs_naive = np.mean(list(two_tier.values())) - np.mean(list(naive_787.values()))

    print(f"\n  CWE-119 cost (perfect→2-tier): {cost_119*100:+.1f}pp "
          f"({steered_rates['CWE-119']*100:.1f}% → {two_tier['CWE-119']*100:.1f}%)")
    print(f"  Average cost (perfect→2-tier):  {cost_avg*100:+.1f}pp "
          f"({np.mean(list(perfect_3way.values()))*100:.1f}% → {np.mean(list(two_tier.values()))*100:.1f}%)")
    print(f"  Average gain (naive→2-tier):    {gain_vs_naive*100:+.1f}pp "
          f"({np.mean(list(naive_787.values()))*100:.1f}% → {np.mean(list(two_tier.values()))*100:.1f}%)")

    # ─── CWE-119/CWE-787 confusion symmetry ─────────────────────────────
    print(f"\n{'='*70}")
    print("CWE-119/CWE-787 CONFUSION ANALYSIS")
    print(f"{'='*70}")

    r_787_on_119 = cross_cwe["CWE-787"]["CWE-119"]  # CWE-787 vec applied to CWE-119 prompts
    r_119_on_787 = cross_cwe["CWE-119"]["CWE-787"]  # CWE-119 vec applied to CWE-787 prompts

    print(f"\n  CWE-787 vector → CWE-119 prompts: {r_787_on_119['secure_rate']*100:.1f}% "
          f"(baseline: {r_787_on_119['baseline_rate']*100:.1f}%, Δ={r_787_on_119['delta_pp']:+.1f}pp)")
    print(f"  CWE-119 vector → CWE-787 prompts: {r_119_on_787['secure_rate']*100:.1f}% "
          f"(baseline: {r_119_on_787['baseline_rate']*100:.1f}%, Δ={r_119_on_787['delta_pp']:+.1f}pp)")

    # Per-prompt breakdown
    print(f"\n  Per-prompt: CWE-787 vec → CWE-119 prompts:")
    for pr in r_787_on_119["per_prompt_summary"]:
        print(f"    {pr['id']}: {pr['secure_rate']*100:.0f}% "
              f"({pr['n_secure']}/{pr['n_total']})")

    print(f"\n  Per-prompt: CWE-119 vec → CWE-787 prompts:")
    for pr in r_119_on_787["per_prompt_summary"]:
        print(f"    {pr['id']}: {pr['secure_rate']*100:.0f}% "
              f"({pr['n_secure']}/{pr['n_total']})")

    asymmetry = r_787_on_119["secure_rate"] - r_119_on_787["secure_rate"]
    print(f"\n  Asymmetry: CWE-787→CWE-119 is {abs(asymmetry)*100:.1f}pp "
          f"{'higher' if asymmetry > 0 else 'lower'} than CWE-119→CWE-787")

    if r_787_on_119["secure_rate"] > r_119_on_787["secure_rate"]:
        print("  → CWE-787 vector transfers better to CWE-119 than vice versa")
        print("  → Supports using CWE-787 as the 'buffer' vector in 2-tier")
    else:
        print("  → CWE-119 vector transfers better to CWE-787 than vice versa")
        print("  → Consider using CWE-119 as the 'buffer' vector instead")

    # ─── Binary probe accuracy from Part A ───────────────────────────────
    if part_a:
        print(f"\n{'='*70}")
        print("BINARY PROBE ROUTING ACCURACY (from Part A)")
        print(f"{'='*70}")

        for layer_key, layer_data in part_a["per_layer"].items():
            layer_num = layer_key.replace("layer_", "")
            br = layer_data["binary"]
            print(f"\n  Layer {layer_num}:")
            for method, label in [("neutral_loo", "Neutral LOO"),
                                  ("adv_trained", "Adv-trained"),
                                  ("mixed_trained", "Mixed")]:
                r = br[method]
                print(f"    {label}: {r['accuracy']*100:.1f}% "
                      f"(buffer={r.get('buffer_accuracy', 0)*100:.1f}%, "
                      f"format={r.get('format_accuracy', 0)*100:.1f}%)")

        # Compute realistic 2-tier rates accounting for probe errors
        print(f"\n  Realistic 2-tier (accounting for probe misrouting):")
        best_binary = part_a["per_layer"]["layer_31"]["binary"]
        for method, label in [("neutral_loo", "LOO probe"),
                              ("adv_trained", "Adv probe"),
                              ("mixed_trained", "Mixed probe")]:
            r = best_binary[method]
            # If probe accuracy is X%, then (1-X)% of the time wrong vector is applied
            # For CWE-134 prompts misclassified as buffer: CWE-787 vec applied
            # For buffer prompts misclassified as format: CWE-134 vec applied
            buf_acc = r.get("buffer_accuracy", 0)
            fmt_acc = r.get("format_accuracy", 0)

            # CWE-787: if correctly classified as buffer → CWE-787 vec (100%)
            #          if misclassified as format → CWE-134 vec (48.6%)
            r787 = buf_acc * steered_rates["CWE-787"] + \
                   (1 - buf_acc) * cross_cwe["CWE-134"]["CWE-787"]["secure_rate"]

            # CWE-119: if correctly classified as buffer → CWE-787 vec (64.3%)
            #          if misclassified as format → CWE-134 vec (69.3%)
            r119 = buf_acc * cross_cwe["CWE-787"]["CWE-119"]["secure_rate"] + \
                   (1 - buf_acc) * cross_cwe["CWE-134"]["CWE-119"]["secure_rate"]

            # CWE-134: if correctly classified as format → CWE-134 vec (100%)
            #          if misclassified as buffer → CWE-787 vec (92.1%)
            r134 = fmt_acc * steered_rates["CWE-134"] + \
                   (1 - fmt_acc) * cross_cwe["CWE-787"]["CWE-134"]["secure_rate"]

            avg = (r787 + r119 + r134) / 3
            print(f"    {label}: CWE-787={r787*100:.1f}%, CWE-119={r119*100:.1f}%, "
                  f"CWE-134={r134*100:.1f}%, avg={avg*100:.1f}%")

    # ─── Decision summary ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("DECISION SUMMARY")
    print(f"{'='*70}")

    avg_perfect = np.mean(list(perfect_3way.values()))
    avg_2tier = np.mean(list(two_tier.values()))
    avg_naive = np.mean(list(naive_787.values()))
    avg_nosteer = np.mean(list(no_steer.values()))

    print(f"\n  Perfect 3-way:  {avg_perfect*100:.1f}% avg secure")
    print(f"  2-Tier:         {avg_2tier*100:.1f}% avg secure (cost: {(avg_perfect-avg_2tier)*100:.1f}pp)")
    print(f"  Naive CWE-787:  {avg_naive*100:.1f}% avg secure")
    print(f"  No steering:    {avg_nosteer*100:.1f}% avg secure")

    if cost_119 * 100 < 10:
        print(f"\n  VERDICT: 2-tier cost is small (<10pp on CWE-119). "
              f"2-tier is the recommended deployment architecture.")
    elif cost_119 * 100 < 20:
        print(f"\n  VERDICT: 2-tier cost is moderate ({cost_119*100:.1f}pp on CWE-119). "
              f"2-tier is viable as minimum deployment; invest in 3-way routing.")
    else:
        print(f"\n  VERDICT: 2-tier cost is high ({cost_119*100:.1f}pp on CWE-119). "
              f"3-way routing is necessary for acceptable CWE-119 coverage.")

    # ─── Save results ────────────────────────────────────────────────────
    output = {
        "timestamp": timestamp,
        "experiment": "Experiment 8.5 Part B: 2-Tier Analysis",
        "strategies": {
            name: {cwe: float(rate) for cwe, rate in rates.items()}
            for name, rates in strategies.items()
        },
        "averages": {
            name: float(np.mean(list(rates.values())))
            for name, rates in strategies.items()
        },
        "cost_analysis": {
            "cwe119_cost_pp": float(cost_119 * 100),
            "avg_cost_pp": float(cost_avg * 100),
            "gain_vs_naive_pp": float(gain_vs_naive * 100),
        },
        "confusion_analysis": {
            "cwe787_vec_on_cwe119": float(r_787_on_119["secure_rate"]),
            "cwe119_vec_on_cwe787": float(r_119_on_787["secure_rate"]),
            "asymmetry_pp": float(asymmetry * 100),
        },
        "phase2_best_alphas": best_alphas,
        "phase2_steered_rates": {k: float(v) for k, v in steered_rates.items()},
        "phase1_baselines": {k: float(v) for k, v in baselines.items()},
    }

    results_path = RESULTS_DIR / f"2tier_simulation_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {results_path}")
    print("\nPart B complete.")


if __name__ == "__main__":
    main()
