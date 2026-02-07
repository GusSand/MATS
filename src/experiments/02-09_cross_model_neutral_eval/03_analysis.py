#!/usr/bin/env python3
"""
Experiment 9 — Phase 5: Cross-Model Analysis

Loads results from both models and builds:
  - Table 1: Neutral Prompt Security Rate
  - Table 2: Instruction Resistance Gap
  - Table 3: Cross-CWE Interference Check
  - Per-model summary tables
  - Cross-model comparison markdown

No GPU required.

Usage:
  python 03_analysis.py
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

# ─── Path Setup ─────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent

# ─── Llama-8B Reference Data (from Experiment 8) ───────────────────────────

LLAMA_DATA = {
    "model_key": "llama8b",
    "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "neutral_baseline": {
        "CWE-787": 0.471,
        "CWE-119": 0.650,
        "CWE-134": 1.000,
    },
    "neutral_steered": {
        "CWE-787": 1.000,
        "CWE-119": 0.814,
        "CWE-134": 1.000,
    },
    "adversarial_baseline": {
        "CWE-787": 0.000,
        "CWE-119": 0.000,
        "CWE-134": 0.667,
    },
    "adversarial_steered": {
        "CWE-787": 0.524,
        "CWE-119": 0.200,
        "CWE-134": 0.900,
    },
    "best_alphas": {
        "CWE-787": 3.5,
        "CWE-119": 4.0,
        "CWE-134": 1.5,
    },
}

MODEL_CONFIGS = {
    "mistral7b": {
        "display_name": "Mistral-7B",
        "data_dir": SCRIPT_DIR / "mistral7b" / "data",
        "analysis_dir": SCRIPT_DIR / "mistral7b" / "analysis",
        "adversarial_steered": {"CWE-787": 0.924},
        "adversarial_baseline": {"CWE-787": 0.267},
    },
    "qwen14b": {
        "display_name": "Qwen-14B",
        "data_dir": SCRIPT_DIR / "qwen14b" / "data",
        "analysis_dir": SCRIPT_DIR / "qwen14b" / "analysis",
        "adversarial_steered": {"CWE-787": 0.771},
        "adversarial_baseline": {"CWE-787": 0.029},
    },
}


# ─── Data Loading ───────────────────────────────────────────────────────────

def find_latest_json(directory, pattern):
    """Find the latest JSON file matching a pattern."""
    files = sorted(directory.glob(pattern))
    if not files:
        return None
    return files[-1]


def load_model_results(model_key, config):
    """Load all results for a model."""
    data_dir = config["data_dir"]
    results = {"model_key": model_key}

    # Phase 2: Neutral baseline
    baseline_file = find_latest_json(data_dir, "neutral_baseline_results_*.json")
    if baseline_file:
        with open(baseline_file) as f:
            bl = json.load(f)
        results["neutral_baseline"] = {
            cwe: bl["cwe_summary"][cwe]["secure_rate"]
            for cwe in ["CWE-787", "CWE-119", "CWE-134"]
        }
        results["baseline_details"] = bl["cwe_summary"]
        print(f"  Loaded baseline: {baseline_file.name}")
    else:
        print(f"  WARNING: No baseline results found in {data_dir}")
        return None

    # Phase 3: Neutral steered
    steered_file = find_latest_json(data_dir, "neutral_steered_results_*.json")
    if steered_file:
        with open(steered_file) as f:
            st = json.load(f)
        results["best_alphas"] = st["best_alphas"]
        results["best_rates"] = st["best_rates"]
        results["neutral_steered"] = st["best_rates"]
        results["alpha_grids"] = st.get("alpha_grids", {})
        results["steered_details"] = st["results"]
        print(f"  Loaded steered: {steered_file.name}")
    else:
        print(f"  WARNING: No steered results found in {data_dir}")
        return None

    # Phase 4: Cross-CWE check
    cross_file = find_latest_json(data_dir, "cross_cwe_sanity_check_*.json")
    if cross_file:
        with open(cross_file) as f:
            cr = json.load(f)
        results["cross_cwe"] = cr["results"]
        print(f"  Loaded cross-CWE: {cross_file.name}")
    else:
        print(f"  WARNING: No cross-CWE results found in {data_dir}")

    # Add existing adversarial data
    results["adversarial_steered"] = config["adversarial_steered"]
    results["adversarial_baseline"] = config["adversarial_baseline"]

    return results


# ─── Bootstrap CI ───────────────────────────────────────────────────────────

def bootstrap_ci(n_secure, n_total, n_bootstrap=10000, ci=0.95):
    """Compute bootstrap confidence interval for a proportion."""
    if n_total == 0:
        return 0, 0, 0
    p = n_secure / n_total
    rng = np.random.RandomState(42)
    samples = rng.binomial(n_total, p, size=n_bootstrap) / n_total
    alpha = (1 - ci) / 2
    lo = np.percentile(samples, alpha * 100)
    hi = np.percentile(samples, (1 - alpha) * 100)
    return p, lo, hi


# ─── Table Builders ─────────────────────────────────────────────────────────

def build_table1(all_data):
    """Table 1: Neutral Prompt Security Rate."""
    lines = []
    lines.append("Table 1: Neutral Prompt Security Rate (% secure generations)")
    lines.append("")
    lines.append("| Model          | Condition          | CWE-787 | CWE-119 | CWE-134 | Avg   |")
    lines.append("|----------------|-------------------|---------|---------|---------|-------|")

    for model_key, data in all_data.items():
        name = data["display_name"]
        bl = data.get("neutral_baseline", {})
        st = data.get("neutral_steered", {})

        bl_vals = [bl.get(c, 0) * 100 for c in ["CWE-787", "CWE-119", "CWE-134"]]
        bl_avg = sum(bl_vals) / 3
        lines.append(
            f"| {name:<14} | Neutral baseline  "
            f"| {bl_vals[0]:>5.1f}%  | {bl_vals[1]:>5.1f}%  | {bl_vals[2]:>5.1f}%  "
            f"| {bl_avg:>5.1f}%|"
        )

        st_vals = [st.get(c, 0) * 100 for c in ["CWE-787", "CWE-119", "CWE-134"]]
        st_avg = sum(st_vals) / 3
        lines.append(
            f"| {name:<14} | Neutral + steer   "
            f"| {st_vals[0]:>5.1f}%  | {st_vals[1]:>5.1f}%  | {st_vals[2]:>5.1f}%  "
            f"| {st_avg:>5.1f}%|"
        )

    return "\n".join(lines)


def build_table2(all_data):
    """Table 2: Instruction Resistance Gap."""
    lines = []
    lines.append("Table 2: Instruction Resistance Gap (neutral_steered - adversarial_steered)")
    lines.append("")
    lines.append("| Model          | CWE-787 | CWE-119 | CWE-134 | Avg Gap |")
    lines.append("|----------------|---------|---------|---------|---------|")

    for model_key, data in all_data.items():
        name = data["display_name"]
        st = data.get("neutral_steered", {})
        adv_st = data.get("adversarial_steered", {})

        gaps = []
        gap_strs = []
        for cwe in ["CWE-787", "CWE-119", "CWE-134"]:
            if cwe in st and cwe in adv_st:
                gap = (st[cwe] - adv_st[cwe]) * 100
                gaps.append(gap)
                gap_strs.append(f"{gap:>+5.1f}pp")
            else:
                gap_strs.append("   --  ")

        avg_gap = f"{sum(gaps)/len(gaps):>+5.1f}pp" if gaps else "   --  "
        lines.append(
            f"| {name:<14} | {gap_strs[0]} | {gap_strs[1]} | {gap_strs[2]} | {avg_gap} |"
        )

    return "\n".join(lines)


def build_table3(all_data):
    """Table 3: Cross-CWE Interference Check."""
    lines = []
    lines.append("Table 3: Cross-CWE Interference Check")
    lines.append("(Secure rate on OTHER CWEs' neutral prompts when steering is active)")
    lines.append("")
    lines.append("| Model      | Steering | CWE-787 (other) | CWE-119 (other) | CWE-134 (other) |")
    lines.append("|------------|----------|-----------------|-----------------|-----------------|")

    for model_key, data in all_data.items():
        if model_key == "llama8b":
            continue  # Llama doesn't have cross-CWE data
        name = data["display_name"]
        cross = data.get("cross_cwe", {})
        bl = data.get("neutral_baseline", {})

        for steer_cwe in ["CWE-787", "CWE-119", "CWE-134"]:
            cells = []
            for prompt_cwe in ["CWE-787", "CWE-119", "CWE-134"]:
                if prompt_cwe == steer_cwe:
                    cells.append("      N/A      ")
                elif steer_cwe in cross and prompt_cwe in cross[steer_cwe]:
                    r = cross[steer_cwe][prompt_cwe]
                    sr = r["secure_rate"] * 100
                    bl_r = bl.get(prompt_cwe, 0) * 100
                    delta = sr - bl_r
                    cells.append(f" {sr:>5.1f}% ({delta:+.0f}pp)")
                else:
                    cells.append("      --       ")

            lines.append(
                f"| {name:<10} | {steer_cwe} | {cells[0]} | {cells[1]} | {cells[2]} |"
            )

    return "\n".join(lines)


def build_per_alpha_detail(data, model_name):
    """Build per-alpha detail table for a model."""
    lines = []
    lines.append(f"Per-Alpha Detail: {model_name}")
    lines.append("")

    bl = data.get("neutral_baseline", {})
    details = data.get("steered_details", {})

    for cwe in ["CWE-787", "CWE-119", "CWE-134"]:
        bl_rate = bl.get(cwe, 0) * 100
        lines.append(f"  {cwe} (baseline: {bl_rate:.1f}%):")
        if cwe in details:
            best_alpha = data.get("best_alphas", {}).get(cwe, None)
            for alpha_key in sorted(details[cwe].keys(), key=float):
                r = details[cwe][alpha_key]
                sr = r["secure_rate"] * 100
                delta = sr - bl_rate
                marker = " <- best" if best_alpha and float(alpha_key) == best_alpha else ""
                lines.append(
                    f"    a={alpha_key}: {sr:>5.1f}% ({delta:+.1f}pp) "
                    f"[{r['total_secure']}/{r['total']} secure, "
                    f"{r['total_refusal']} refusals]{marker}"
                )
        lines.append("")

    return "\n".join(lines)


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("Experiment 9 — Phase 5: Cross-Model Analysis")
    print(f"Timestamp: {timestamp}")
    print("=" * 70)

    # Load results for each model
    all_data = {}

    # Llama-8B (reference data, hardcoded from experiment 8)
    all_data["llama8b"] = {
        "display_name": "Llama-8B",
        **LLAMA_DATA,
    }
    print("\nLlama-8B: Using reference data from Experiment 8")

    # Mistral-7B and Qwen-14B (from saved results)
    for model_key, config in MODEL_CONFIGS.items():
        print(f"\n{config['display_name']}:")
        results = load_model_results(model_key, config)
        if results:
            all_data[model_key] = {
                "display_name": config["display_name"],
                **results,
            }
        else:
            print(f"  SKIPPED: Missing data for {model_key}")

    if len(all_data) < 2:
        print("\nERROR: Need at least 2 models to build comparison. "
              "Run 02_neutral_eval.py first.")
        return

    # ─── Build Tables ─────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("CROSS-MODEL COMPARISON RESULTS")
    print(f"{'=' * 70}")

    table1 = build_table1(all_data)
    print(f"\n{table1}")

    table2 = build_table2(all_data)
    print(f"\n{table2}")

    table3 = build_table3(all_data)
    print(f"\n{table3}")

    # ─── Hypothesis Checks ────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("HYPOTHESIS EVALUATION")
    print(f"{'=' * 70}")

    # H1: Instruction resistance gap is architecture-dependent
    print("\nH1: Instruction resistance gap is architecture-dependent")
    for mk in ["llama8b", "mistral7b", "qwen14b"]:
        if mk not in all_data:
            continue
        d = all_data[mk]
        st = d.get("neutral_steered", {})
        adv = d.get("adversarial_steered", {})
        if "CWE-787" in st and "CWE-787" in adv:
            gap = (st["CWE-787"] - adv["CWE-787"]) * 100
            print(f"  {d['display_name']}: CWE-787 gap = {gap:+.1f}pp "
                  f"(neutral={st['CWE-787']*100:.1f}%, "
                  f"adversarial={adv['CWE-787']*100:.1f}%)")

    # H2: CWE-134 neutral baselines high across all models
    print("\nH2: CWE-134 neutral baselines high across all models")
    for mk in ["llama8b", "mistral7b", "qwen14b"]:
        if mk not in all_data:
            continue
        d = all_data[mk]
        bl = d.get("neutral_baseline", {})
        if "CWE-134" in bl:
            rate = bl["CWE-134"] * 100
            verdict = "CONFIRMED" if rate >= 90 else "PARTIAL" if rate >= 70 else "REJECTED"
            print(f"  {d['display_name']}: CWE-134 baseline = {rate:.1f}% [{verdict}]")

    # H3: CWE-119 remains hardest to steer
    print("\nH3: CWE-119 remains hardest to steer")
    for mk in ["llama8b", "mistral7b", "qwen14b"]:
        if mk not in all_data:
            continue
        d = all_data[mk]
        st = d.get("neutral_steered", {})
        if all(c in st for c in ["CWE-787", "CWE-119", "CWE-134"]):
            rates = {c: st[c] * 100 for c in ["CWE-787", "CWE-119", "CWE-134"]}
            worst = min(rates, key=rates.get)
            verdict = "CONFIRMED" if worst == "CWE-119" else f"REJECTED (worst={worst})"
            print(f"  {d['display_name']}: rates={rates} [{verdict}]")

    # H4: No cross-CWE interference
    print("\nH4: No cross-CWE interference")
    for mk in ["mistral7b", "qwen14b"]:
        if mk not in all_data or "cross_cwe" not in all_data[mk]:
            continue
        d = all_data[mk]
        cross = d["cross_cwe"]
        bl = d["neutral_baseline"]
        any_degrade = False
        for steer_cwe, steer_res in cross.items():
            for prompt_cwe, r in steer_res.items():
                delta = r["secure_rate"] * 100 - bl.get(prompt_cwe, 0) * 100
                if delta < -5:
                    any_degrade = True
                    print(f"  {d['display_name']}: {steer_cwe}->{prompt_cwe} "
                          f"DEGRADATION {delta:+.1f}pp")
        if not any_degrade:
            print(f"  {d['display_name']}: No interference detected [CONFIRMED]")

    # ─── Save Markdown Reports ────────────────────────────────────────────
    # Cross-model comparison
    comparison_md = []
    comparison_md.append("# Experiment 9: Cross-Model Neutral Evaluation Results")
    comparison_md.append(f"\nGenerated: {timestamp}")
    comparison_md.append(f"\nModels: Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2.5-14B-Instruct")
    comparison_md.append(f"\n## Results\n")
    comparison_md.append(table1)
    comparison_md.append(f"\n{table2}")
    comparison_md.append(f"\n{table3}")

    # Best alphas
    comparison_md.append("\n## Best Steering Alphas\n")
    comparison_md.append("| Model | CWE-787 | CWE-119 | CWE-134 |")
    comparison_md.append("|-------|---------|---------|---------|")
    for mk in ["llama8b", "mistral7b", "qwen14b"]:
        if mk not in all_data:
            continue
        d = all_data[mk]
        alphas = d.get("best_alphas", {})
        a787 = alphas.get("CWE-787", "--")
        a119 = alphas.get("CWE-119", "--")
        a134 = alphas.get("CWE-134", "--")
        comparison_md.append(f"| {d['display_name']} | {a787} | {a119} | {a134} |")

    # Hypothesis summary
    comparison_md.append("\n## Hypothesis Evaluation\n")
    comparison_md.append("- **H1** (Instruction resistance gap is architecture-dependent): See Table 2")
    comparison_md.append("- **H2** (CWE-134 baselines high across models): Check baseline column in Table 1")
    comparison_md.append("- **H3** (CWE-119 hardest to steer): Check steered rates in Table 1")
    comparison_md.append("- **H4** (No cross-CWE interference): See Table 3")

    comparison_path = SCRIPT_DIR / "cross_model_comparison.md"
    with open(comparison_path, 'w') as f:
        f.write("\n".join(comparison_md))
    print(f"\nSaved: {comparison_path}")

    # Per-model summary tables
    for mk in ["mistral7b", "qwen14b"]:
        if mk not in all_data:
            continue
        d = all_data[mk]
        config = MODEL_CONFIGS[mk]
        analysis_dir = config["analysis_dir"]
        analysis_dir.mkdir(parents=True, exist_ok=True)

        summary_lines = []
        summary_lines.append(f"# {d['display_name']} Neutral Evaluation Summary")
        summary_lines.append(f"\nGenerated: {timestamp}")
        summary_lines.append(f"\n## Neutral Baseline\n")

        bl = d.get("neutral_baseline", {})
        for cwe in ["CWE-787", "CWE-119", "CWE-134"]:
            rate = bl.get(cwe, 0) * 100
            summary_lines.append(f"- {cwe}: {rate:.1f}%")

        summary_lines.append(f"\n## Neutral + Steering (Best Alpha)\n")
        st = d.get("neutral_steered", {})
        alphas = d.get("best_alphas", {})
        for cwe in ["CWE-787", "CWE-119", "CWE-134"]:
            rate = st.get(cwe, 0) * 100
            alpha = alphas.get(cwe, "--")
            delta = rate - bl.get(cwe, 0) * 100
            summary_lines.append(f"- {cwe}: {rate:.1f}% (a={alpha}, {delta:+.1f}pp from baseline)")

        summary_lines.append(f"\n## Per-Alpha Detail\n")
        summary_lines.append("```")
        summary_lines.append(build_per_alpha_detail(d, d["display_name"]))
        summary_lines.append("```")

        if "cross_cwe" in d:
            summary_lines.append(f"\n## Cross-CWE Check\n")
            cross = d["cross_cwe"]
            for steer_cwe, steer_res in cross.items():
                for prompt_cwe, r in steer_res.items():
                    sr = r["secure_rate"] * 100
                    delta = r.get("delta_pp", 0)
                    summary_lines.append(
                        f"- {steer_cwe} -> {prompt_cwe}: {sr:.1f}% ({delta:+.1f}pp from baseline)"
                    )

        summary_path = analysis_dir / "summary_table.md"
        with open(summary_path, 'w') as f:
            f.write("\n".join(summary_lines))
        print(f"Saved: {summary_path}")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
