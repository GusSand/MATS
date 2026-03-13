#!/usr/bin/env python3
"""
Plot format ablation logit lens results.

Produces a figure with three P(snprintf) trajectories (adversarial, neutral, secure)
across layers. This is the "format ablation" figure for the paper.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import glob


def load_latest_results():
    """Load the most recent results file."""
    results_dir = Path(__file__).parent / "results"
    files = sorted(results_dir.glob("format_ablation_logit_lens_*.json"))
    if not files:
        print("No results files found!")
        sys.exit(1)
    latest = files[-1]
    print(f"Loading: {latest}")
    with open(latest) as f:
        return json.load(f), latest


def extract_trajectories(results):
    """Extract P(snprintf) trajectories for each condition, averaged across scenarios."""
    n_layers = results["n_layers"]
    conditions = ["adversarial", "neutral", "secure"]

    # Per-scenario trajectories
    all_trajectories = {c: [] for c in conditions}

    for scenario_id, scenario_data in results["results"].items():
        for condition in conditions:
            layer_data = scenario_data[condition]["layer_data"]
            trajectory = []
            for l in range(n_layers):
                p = layer_data[f"layer_{l}"]["p_snprintf"]
                trajectory.append(p)
            all_trajectories[condition].append(trajectory)

    # Average trajectories
    mean_trajectories = {}
    std_trajectories = {}
    for condition in conditions:
        arr = np.array(all_trajectories[condition])
        mean_trajectories[condition] = np.mean(arr, axis=0)
        std_trajectories[condition] = np.std(arr, axis=0)

    return mean_trajectories, std_trajectories, all_trajectories


def plot_main_figure(results, output_path):
    """Create the main format ablation figure for the paper."""
    n_layers = results["n_layers"]
    layers = np.arange(n_layers)

    mean_traj, std_traj, all_traj = extract_trajectories(results)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel A: P(snprintf) trajectories ---
    ax = axes[0]
    colors = {"adversarial": "#d62728", "neutral": "#2ca02c", "secure": "#1f77b4"}
    labels = {"adversarial": "Adversarial (format instruction)",
              "neutral": "Neutral (no instruction)",
              "secure": "Secure (security instruction)"}

    for condition in ["adversarial", "neutral", "secure"]:
        mean = mean_traj[condition]
        std = std_traj[condition]
        ax.plot(layers, mean, color=colors[condition], label=labels[condition], linewidth=2)
        ax.fill_between(layers, mean - std, mean + std, alpha=0.15, color=colors[condition])

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("P(snprintf)", fontsize=12)
    ax.set_title("(a) P(snprintf) Emergence by Prompt Condition", fontsize=13)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_xlim(0, n_layers - 1)
    ax.grid(True, alpha=0.3)

    # --- Panel B: Security preference signal (P(snprintf) - P(sprintf)) ---
    ax2 = axes[1]

    for condition in ["adversarial", "neutral", "secure"]:
        # Compute P(snprintf) - P(sprintf) per scenario, then average
        diff_trajectories = []
        for scenario_id, scenario_data in results["results"].items():
            layer_data = scenario_data[condition]["layer_data"]
            diff_traj = []
            for l in range(n_layers):
                p_sn = layer_data[f"layer_{l}"]["p_snprintf"]
                p_sp = layer_data[f"layer_{l}"]["p_sprintf"]
                diff_traj.append(p_sn - p_sp)
            diff_trajectories.append(diff_traj)

        arr = np.array(diff_trajectories)
        mean_diff = np.mean(arr, axis=0)
        std_diff = np.std(arr, axis=0)

        ax2.plot(layers, mean_diff, color=colors[condition], label=labels[condition], linewidth=2)
        ax2.fill_between(layers, mean_diff - std_diff, mean_diff + std_diff, alpha=0.15, color=colors[condition])

    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel("Layer", fontsize=12)
    ax2.set_ylabel("P(snprintf) - P(sprintf)", fontsize=12)
    ax2.set_title("(b) Security Preference Signal by Condition", fontsize=13)
    ax2.legend(fontsize=10, loc="upper left")
    ax2.set_xlim(0, n_layers - 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")
    plt.close()


def plot_per_scenario(results, output_dir):
    """Create per-scenario subplots for appendix."""
    n_layers = results["n_layers"]
    layers = np.arange(n_layers)
    scenarios = list(results["results"].keys())
    n_scenarios = len(scenarios)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()

    colors = {"adversarial": "#d62728", "neutral": "#2ca02c", "secure": "#1f77b4"}

    for i, scenario_id in enumerate(scenarios):
        if i >= 8:
            break
        ax = axes[i]
        scenario_data = results["results"][scenario_id]

        for condition in ["adversarial", "neutral", "secure"]:
            layer_data = scenario_data[condition]["layer_data"]
            traj = [layer_data[f"layer_{l}"]["p_snprintf"] for l in range(n_layers)]
            ax.plot(layers, traj, color=colors[condition], linewidth=1.5,
                    label=condition.capitalize() if i == 0 else None)

        ax.set_title(scenario_id.replace("neutral_787_", "Scenario "), fontsize=10)
        ax.set_xlabel("Layer", fontsize=9)
        ax.set_ylabel("P(snprintf)", fontsize=9)
        ax.set_xlim(0, n_layers - 1)
        ax.grid(True, alpha=0.3)

    # Hide unused subplot
    if n_scenarios < 8:
        for j in range(n_scenarios, 8):
            axes[j].set_visible(False)

    # Add legend to first subplot
    axes[0].legend(fontsize=8, loc="upper left")

    plt.suptitle("P(snprintf) Emergence by Scenario and Condition", fontsize=14, y=1.02)
    plt.tight_layout()

    per_scenario_path = output_dir / "format_ablation_per_scenario.png"
    plt.savefig(per_scenario_path, dpi=150, bbox_inches='tight')
    print(f"Per-scenario figure saved to: {per_scenario_path}")
    plt.close()


if __name__ == "__main__":
    results, results_path = load_latest_results()
    output_dir = Path(__file__).parent / "results"

    # Main figure
    main_fig_path = output_dir / "format_ablation_main.png"
    plot_main_figure(results, main_fig_path)

    # Per-scenario figure
    plot_per_scenario(results, output_dir)

    # Print summary
    summary = results.get("summary", {})
    l31 = summary.get("l31_means", {})
    print(f"\nL31 P(snprintf) means:")
    print(f"  Adversarial: {l31.get('adversarial', 'N/A'):.6f}")
    print(f"  Neutral:     {l31.get('neutral', 'N/A'):.6f}")
    print(f"  Secure:      {l31.get('secure', 'N/A'):.6f}")
    print(f"  Hypothesis confirmed: {summary.get('hypothesis_confirmed', 'N/A')}")
