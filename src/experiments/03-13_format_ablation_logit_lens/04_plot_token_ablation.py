#!/usr/bin/env python3
"""
Plot token ablation results for Experiment 29b.

Produces a figure with overlay of unmodified adversarial vs ablated P(snprintf) curves,
plus the secure variant for reference.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def load_latest_results():
    results_dir = Path(__file__).parent / "results"
    files = sorted(results_dir.glob("token_ablation_logit_lens_*.json"))
    if not files:
        print("No token ablation results files found!")
        sys.exit(1)
    latest = files[-1]
    print(f"Loading: {latest}")
    with open(latest) as f:
        return json.load(f), latest


def extract_trajectories(results):
    n_layers = results["n_layers"]
    conditions = ["unmodified_adversarial", "zero_ablated", "mean_ablated",
                  "neutral_substituted", "no_comment_baseline", "secure_variant"]

    all_traj = {c: [] for c in conditions}

    for scenario_id, sdata in results["results"].items():
        for cond in conditions:
            if cond in sdata and "layer_data" in sdata[cond]:
                traj = [sdata[cond]["layer_data"][f"layer_{l}"]["p_snprintf"]
                        for l in range(n_layers)]
                all_traj[cond].append(traj)

    mean_traj = {}
    std_traj = {}
    for cond in conditions:
        if all_traj[cond]:
            arr = np.array(all_traj[cond])
            mean_traj[cond] = np.mean(arr, axis=0)
            std_traj[cond] = np.std(arr, axis=0)

    return mean_traj, std_traj, all_traj


def plot_main_figure(results, output_path):
    n_layers = results["n_layers"]
    layers = np.arange(n_layers)

    mean_traj, std_traj, _ = extract_trajectories(results)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel A: Full comparison ---
    ax = axes[0]
    plot_specs = {
        "unmodified_adversarial": {"color": "#d62728", "ls": "-", "lw": 2.5,
                                    "label": "Unmodified adversarial"},
        "mean_ablated": {"color": "#ff7f0e", "ls": "-", "lw": 2,
                          "label": "Format-ablated (mean)"},
        "zero_ablated": {"color": "#bcbd22", "ls": "--", "lw": 1.5,
                          "label": "Format-ablated (zero)"},
        "neutral_substituted": {"color": "#17becf", "ls": "--", "lw": 1.5,
                                 "label": "Neutral substitution"},
        "no_comment_baseline": {"color": "#2ca02c", "ls": "-", "lw": 2,
                                 "label": "No-comment baseline"},
        "secure_variant": {"color": "#1f77b4", "ls": "-", "lw": 2.5,
                            "label": "Secure variant"},
    }

    for cond, spec in plot_specs.items():
        if cond in mean_traj:
            ax.plot(layers, mean_traj[cond], color=spec["color"], ls=spec["ls"],
                    lw=spec["lw"], label=spec["label"])
            if cond in std_traj:
                ax.fill_between(layers, mean_traj[cond] - std_traj[cond],
                                mean_traj[cond] + std_traj[cond],
                                alpha=0.1, color=spec["color"])

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("P(snprintf)", fontsize=12)
    ax.set_title("(a) Format-Token Ablation: P(snprintf) Recovery", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0, n_layers - 1)
    ax.grid(True, alpha=0.3)

    # --- Panel B: Recovery bar chart at L31 ---
    ax2 = axes[1]
    summary = results.get("summary", {})
    l31_means = summary.get("l31_means", {})
    l31_stds = summary.get("l31_stds", {})

    bar_order = ["unmodified_adversarial", "zero_ablated", "mean_ablated",
                 "neutral_substituted", "no_comment_baseline", "secure_variant"]
    bar_labels = ["Unmod.\nAdvers.", "Zero\nAblated", "Mean\nAblated",
                  "Neutral\nSubst.", "No\nComment", "Secure\nVariant"]
    bar_colors = ["#d62728", "#bcbd22", "#ff7f0e", "#17becf", "#2ca02c", "#1f77b4"]

    vals = [l31_means.get(c, 0) for c in bar_order]
    errs = [l31_stds.get(c, 0) for c in bar_order]

    bars = ax2.bar(range(len(vals)), vals, yerr=errs, capsize=4,
                   color=bar_colors, edgecolor="black", linewidth=0.5, alpha=0.85)

    ax2.set_xticks(range(len(bar_labels)))
    ax2.set_xticklabels(bar_labels, fontsize=9)
    ax2.set_ylabel("P(snprintf) at L31", fontsize=12)
    ax2.set_title("(b) Layer 31 Recovery by Ablation Method", fontsize=13)
    ax2.grid(True, alpha=0.3, axis="y")

    # Add recovery percentages
    recovery = summary.get("recovery_pct", {})
    adv_val = l31_means.get("unmodified_adversarial", 0)
    for i, cond in enumerate(bar_order):
        if cond in recovery:
            pct = recovery[cond]
            ax2.annotate(f"{pct:.0f}%", xy=(i, vals[i]), xytext=(0, 5),
                         textcoords="offset points", ha="center", fontsize=8,
                         fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")
    plt.close()


def plot_per_scenario(results, output_dir):
    """Per-scenario subplots for the appendix."""
    n_layers = results["n_layers"]
    layers = np.arange(n_layers)
    scenarios = [s for s in results["results"].keys() if "metadata" not in s or True]
    # Filter to actual scenarios
    scenarios = [s for s in results["results"].keys()
                 if isinstance(results["results"][s], dict) and "unmodified_adversarial" in results["results"][s]]

    n = len(scenarios)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    colors = {
        "unmodified_adversarial": "#d62728",
        "mean_ablated": "#ff7f0e",
        "no_comment_baseline": "#2ca02c",
        "secure_variant": "#1f77b4",
    }

    for i, scenario_id in enumerate(scenarios):
        if i >= len(axes):
            break
        ax = axes[i]
        sdata = results["results"][scenario_id]

        for cond, color in colors.items():
            if cond in sdata and "layer_data" in sdata[cond]:
                traj = [sdata[cond]["layer_data"][f"layer_{l}"]["p_snprintf"]
                        for l in range(n_layers)]
                label = cond.replace("_", " ").title() if i == 0 else None
                ax.plot(layers, traj, color=color, lw=1.5, label=label)

        ax.set_title(scenario_id.replace("neutral_787_", "Scenario "), fontsize=10)
        ax.set_xlabel("Layer", fontsize=9)
        ax.set_ylabel("P(snprintf)", fontsize=9)
        ax.set_xlim(0, n_layers - 1)
        ax.grid(True, alpha=0.3)

    # Hide unused
    for j in range(len(scenarios), len(axes)):
        axes[j].set_visible(False)

    axes[0].legend(fontsize=8, loc="upper left")
    plt.suptitle("Token Ablation: P(snprintf) by Scenario", fontsize=14, y=1.02)
    plt.tight_layout()

    path = output_dir / "token_ablation_per_scenario.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Per-scenario figure saved to: {path}")
    plt.close()


if __name__ == "__main__":
    results, results_path = load_latest_results()
    output_dir = Path(__file__).parent / "results"

    main_fig = output_dir / "token_ablation_main.png"
    plot_main_figure(results, main_fig)
    plot_per_scenario(results, output_dir)

    summary = results.get("summary", {})
    print(f"\nCausal test confirmed: {summary.get('causation_confirmed', 'N/A')}")
    recovery = summary.get("recovery_pct", {})
    for method, pct in recovery.items():
        print(f"  {method}: {pct:.1f}% recovery")
