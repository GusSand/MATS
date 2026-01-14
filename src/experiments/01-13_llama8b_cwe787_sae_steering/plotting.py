"""
Plotting Module for Experiment 3

Generates publication-quality figures for:
- Part 3A: Secure% vs Other% tradeoff curves
- Part 3B: Logit gap distributions and shifts
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional

from experiment_config import DATA_DIR, FIGURES_DIR


# Set publication style
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


# Color palette for methods
METHOD_COLORS = {
    'M1_mean_diff': '#1f77b4',       # Blue
    'M2a_sae_L31_1895': '#ff7f0e',   # Orange
    'M2b_sae_L30_10391': '#2ca02c',  # Green
    'M3a_sae_top5': '#d62728',       # Red
    'M3b_sae_top10': '#9467bd',      # Purple
}

METHOD_LABELS = {
    'M1_mean_diff': 'Mean-diff (L31)',
    'M2a_sae_L31_1895': 'SAE L31:1895',
    'M2b_sae_L30_10391': 'SAE L30:10391',
    'M3a_sae_top5': 'SAE Top-5',
    'M3b_sae_top10': 'SAE Top-10',
}

SETTING_COLORS = {
    'S0': '#808080',  # Gray
    'S1': '#1f77b4',  # Blue
    'S2': '#ff7f0e',  # Orange
}


def plot_tradeoff_curve(
    aggregates_df: pd.DataFrame,
    scoring: str = 'expanded',
    output_path: Path = None,
    title: str = None,
) -> plt.Figure:
    """
    Plot Secure% vs Other% tradeoff curve for Part 3A.

    Each method is a line with points for each α/σ setting.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    secure_col = f'{scoring}_secure_rate'
    other_col = f'{scoring}_other_rate'

    for method in sorted(aggregates_df['method'].unique()):
        method_data = aggregates_df[aggregates_df['method'] == method].copy()

        # Sort by other_rate for plotting
        method_data = method_data.sort_values(other_col)

        color = METHOD_COLORS.get(method, '#333333')
        label = METHOD_LABELS.get(method, method)

        # Plot line
        ax.plot(
            method_data[other_col] * 100,
            method_data[secure_col] * 100,
            marker='o',
            markersize=6,
            linewidth=2,
            color=color,
            label=label,
        )

        # Annotate key points
        for _, row in method_data.iterrows():
            ax.annotate(
                row['setting'],
                (row[other_col] * 100, row[secure_col] * 100),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
                alpha=0.7,
            )

    # Add 10% threshold line
    ax.axvline(x=10, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='10% Other threshold')

    ax.set_xlabel('Other Rate (%)')
    ax.set_ylabel('Secure Rate (%)')

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Secure vs Other Tradeoff ({scoring.upper()} Scoring)')

    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    ax.set_xlim(0, max(aggregates_df[other_col] * 100) * 1.1)
    ax.set_ylim(0, max(aggregates_df[secure_col] * 100) * 1.1)

    plt.tight_layout()

    if output_path:
        # Save both PDF and PNG
        fig.savefig(output_path.with_suffix('.pdf'))
        fig.savefig(output_path.with_suffix('.png'))
        print(f"Saved: {output_path.with_suffix('.pdf')}, {output_path.with_suffix('.png')}")

    return fig


def plot_method_comparison_bars(
    aggregates_df: pd.DataFrame,
    setting_filter: str = None,
    scoring: str = 'expanded',
    output_path: Path = None,
) -> plt.Figure:
    """
    Plot bar comparison of methods at a specific setting.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    secure_col = f'{scoring}_secure_rate'
    insecure_col = f'{scoring}_insecure_rate'
    other_col = f'{scoring}_other_rate'

    # Filter to specific setting if provided
    if setting_filter:
        df = aggregates_df[aggregates_df['setting'] == setting_filter].copy()
    else:
        # Use best setting per method (highest secure)
        df = aggregates_df.loc[aggregates_df.groupby('method')[secure_col].idxmax()]

    methods = df['method'].tolist()
    x = np.arange(len(methods))
    width = 0.25

    bars1 = ax.bar(x - width, df[secure_col] * 100, width, label='Secure', color='#2ecc71')
    bars2 = ax.bar(x, df[insecure_col] * 100, width, label='Insecure', color='#e74c3c')
    bars3 = ax.bar(x + width, df[other_col] * 100, width, label='Other', color='#95a5a6')

    ax.set_xlabel('Method')
    ax.set_ylabel('Rate (%)')
    ax.set_title(f'Method Comparison ({scoring.upper()} Scoring)')
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path.with_suffix('.pdf'))
        fig.savefig(output_path.with_suffix('.png'))
        print(f"Saved: {output_path}")

    return fig


def plot_logit_gap_distribution(
    logits_df: pd.DataFrame,
    output_path: Path = None,
) -> plt.Figure:
    """
    Plot logit gap distributions for Part 3B.

    Box plots showing Δlogit distribution for S0, S1, S2.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    settings = ['S0', 'S1', 'S2']
    data = [logits_df[logits_df['setting'] == s]['gap'].values for s in settings]

    bp = ax.boxplot(data, labels=settings, patch_artist=True)

    # Color the boxes
    for i, (patch, setting) in enumerate(zip(bp['boxes'], settings)):
        patch.set_facecolor(SETTING_COLORS.get(setting, '#808080'))
        patch.set_alpha(0.7)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('Setting')
    ax.set_ylabel('Logit Gap (Δ = logit(safe) - logit(unsafe))')
    ax.set_title('Logit Gap Distribution by Steering Setting')

    # Add mean markers
    for i, d in enumerate(data):
        ax.scatter(i + 1, np.mean(d), marker='D', color='white', edgecolors='black', s=50, zorder=5)

    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path.with_suffix('.pdf'))
        fig.savefig(output_path.with_suffix('.png'))
        print(f"Saved: {output_path}")

    return fig


def plot_logit_gap_shift(
    logits_df: pd.DataFrame,
    output_path: Path = None,
) -> plt.Figure:
    """
    Plot per-prompt logit gap shifts from S0 baseline.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pivot to compute shifts
    pivot = logits_df.pivot(index='prompt_id', columns='setting', values='gap')

    for i, (setting, ax) in enumerate(zip(['S1', 'S2'], axes)):
        if setting not in pivot.columns:
            continue

        shift = pivot[setting] - pivot['S0']

        # Histogram
        ax.hist(shift, bins=30, color=SETTING_COLORS[setting], alpha=0.7, edgecolor='black')

        # Add mean line
        mean_shift = shift.mean()
        ax.axvline(x=mean_shift, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_shift:.2f}')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

        ax.set_xlabel(f'Δlogit Shift ({setting} - S0)')
        ax.set_ylabel('Count')
        ax.set_title(f'{setting} Shift Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path.with_suffix('.pdf'))
        fig.savefig(output_path.with_suffix('.png'))
        print(f"Saved: {output_path}")

    return fig


def plot_gap_by_vuln_type(
    logits_df: pd.DataFrame,
    output_path: Path = None,
) -> plt.Figure:
    """
    Plot logit gap by vulnerability type (sprintf vs strcat).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    vuln_types = logits_df['vuln_type'].unique()

    for ax, vuln_type in zip(axes, vuln_types):
        vt_data = logits_df[logits_df['vuln_type'] == vuln_type]

        settings = ['S0', 'S1', 'S2']
        data = [vt_data[vt_data['setting'] == s]['gap'].values for s in settings]

        bp = ax.boxplot(data, labels=settings, patch_artist=True)

        for patch, setting in zip(bp['boxes'], settings):
            patch.set_facecolor(SETTING_COLORS.get(setting, '#808080'))
            patch.set_alpha(0.7)

        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Setting')
        ax.set_ylabel('Logit Gap')
        ax.set_title(f'{vuln_type} Prompts')
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Logit Gap by Vulnerability Type', y=1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path.with_suffix('.pdf'))
        fig.savefig(output_path.with_suffix('.png'))
        print(f"Saved: {output_path}")

    return fig


def plot_correlation_scatter(
    logits_df: pd.DataFrame,
    freegen_df: pd.DataFrame,
    output_path: Path = None,
) -> plt.Figure:
    """
    Plot correlation between logit gap and secure rate from free generation.
    """
    # Compute secure rate per prompt per setting
    secure_rates = freegen_df.groupby(['prompt_id', 'setting']).apply(
        lambda x: (x['expanded_label'] == 'secure').mean()
    ).reset_index(name='secure_rate')

    # Merge with logits
    merged = pd.merge(
        logits_df[['prompt_id', 'setting', 'gap']],
        secure_rates,
        on=['prompt_id', 'setting'],
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    for setting in ['S0', 'S1', 'S2']:
        setting_data = merged[merged['setting'] == setting]
        ax.scatter(
            setting_data['gap'],
            setting_data['secure_rate'] * 100,
            color=SETTING_COLORS[setting],
            alpha=0.6,
            label=setting,
            s=30,
        )

    # Fit overall trend line
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        merged['gap'], merged['secure_rate'] * 100
    )
    x_line = np.linspace(merged['gap'].min(), merged['gap'].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=2,
            label=f'Linear fit (r={r_value:.2f})')

    ax.set_xlabel('Logit Gap')
    ax.set_ylabel('Secure Rate (%)')
    ax.set_title('Logit Gap vs Secure Rate Correlation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path.with_suffix('.pdf'))
        fig.savefig(output_path.with_suffix('.png'))
        print(f"Saved: {output_path}")

    return fig


def generate_all_figures(
    results_3A_path: Path = None,
    results_3B_path: Path = None,
    logits_3B_path: Path = None,
    freegen_3B_path: Path = None,
    output_dir: Path = None,
):
    """
    Generate all figures for Experiment 3.
    """
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Part 3A figures
    if results_3A_path or list(DATA_DIR.glob("results_3A_*.json")):
        if results_3A_path is None:
            results_3A_path = sorted(DATA_DIR.glob("results_3A_*.json"))[-1]

        print(f"\nGenerating 3A figures from: {results_3A_path}")

        with open(results_3A_path) as f:
            results_3A = json.load(f)

        # Create aggregates DataFrame
        aggregates_df = pd.DataFrame([
            {'method': m, 'setting': s, **v}
            for m, settings in results_3A['aggregated'].items()
            for s, v in settings.items()
        ])

        # Figure 3: Tradeoff curves
        plot_tradeoff_curve(
            aggregates_df, scoring='strict',
            output_path=output_dir / 'fig3_tradeoff_strict'
        )
        plot_tradeoff_curve(
            aggregates_df, scoring='expanded',
            output_path=output_dir / 'fig3_tradeoff_expanded'
        )

        # Method comparison bars
        plot_method_comparison_bars(
            aggregates_df, scoring='expanded',
            output_path=output_dir / 'fig3_method_comparison'
        )

    # Part 3B figures
    if logits_3B_path or list(DATA_DIR.glob("results_3B_logits_*.csv")):
        if logits_3B_path is None:
            logits_3B_path = sorted(DATA_DIR.glob("results_3B_logits_*.csv"))[-1]

        print(f"\nGenerating 3B figures from: {logits_3B_path}")

        logits_df = pd.read_csv(logits_3B_path)

        # Figure 4: Logit gap distribution
        plot_logit_gap_distribution(
            logits_df,
            output_path=output_dir / 'fig4_logit_gap_boxplot'
        )

        # Logit gap shifts
        plot_logit_gap_shift(
            logits_df,
            output_path=output_dir / 'fig4_logit_gap_shift'
        )

        # By vulnerability type
        if 'vuln_type' in logits_df.columns:
            plot_gap_by_vuln_type(
                logits_df,
                output_path=output_dir / 'fig4_logit_gap_by_type'
            )

        # Correlation plot (requires freegen data)
        if freegen_3B_path or list(DATA_DIR.glob("results_3B_freegen_*.csv")):
            if freegen_3B_path is None:
                freegen_3B_path = sorted(DATA_DIR.glob("results_3B_freegen_*.csv"))[-1]

            freegen_df = pd.read_csv(freegen_3B_path)
            plot_correlation_scatter(
                logits_df, freegen_df,
                output_path=output_dir / 'fig4_gap_secure_correlation'
            )

    print(f"\nAll figures saved to: {output_dir}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Generating figures...")
    generate_all_figures()
