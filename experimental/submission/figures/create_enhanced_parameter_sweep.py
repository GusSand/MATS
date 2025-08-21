#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import json
import matplotlib as mpl

# Set large font sizes
mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 16

# Load data
with open('../parameter_sweep_results.json', 'r') as f:
    sweep_data = json.load(f)

with open('../fine_sweep_results.json', 'r') as f:
    fine_data = json.load(f)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

# Main sweep plot
ablation_values = [d['ablation_value'] for d in sweep_data]
bug_rates = [d['bug_rate'] for d in sweep_data]
coherence_scores = [d['avg_coherence'] for d in sweep_data]

# Plot bug rate
color1 = '#e74c3c'
ax1.plot(ablation_values, bug_rates, 'o-', color=color1, linewidth=3, markersize=10, label='Bug Rate')
ax1.set_xlabel('Ablation Value', fontsize=20, fontweight='bold')
ax1.set_ylabel('Bug Rate (%)', color=color1, fontsize=20, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color1, labelsize=18)
ax1.set_ylim(-5, 105)
ax1.grid(True, alpha=0.3)

# Plot coherence on secondary axis
ax1_twin = ax1.twinx()
color2 = '#3498db'
ax1_twin.plot(ablation_values, coherence_scores, 's--', color=color2, linewidth=3, markersize=10, label='Coherence')
ax1_twin.set_ylabel('Coherence Score (1-4)', color=color2, fontsize=20, fontweight='bold')
ax1_twin.tick_params(axis='y', labelcolor=color2, labelsize=18)
ax1_twin.set_ylim(0.5, 4.5)

# Add transition zone shading
ax1.axvspan(-4.5, -4.0, alpha=0.2, color='yellow', label='Transition Zone')

# Title
ax1.set_title('Ablation Parameter Sweep: Bug Rate vs Coherence\nLlama-3.1-8B-Instruct', 
              fontsize=26, fontweight='bold', pad=20)

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=18)

# Fine sweep plot (zoomed in)
fine_ablation = [d['ablation_value'] for d in fine_data]
fine_bug_rates = [d['bug_rate'] for d in fine_data]
fine_unclear_rates = [d['unclear_rate'] for d in fine_data]

# Create stacked area plot
ax2.fill_between(fine_ablation, 0, fine_bug_rates, color='#e74c3c', alpha=0.8, label='Bug (says 9.11)')
ax2.fill_between(fine_ablation, fine_bug_rates, 
                 [b + u for b, u in zip(fine_bug_rates, fine_unclear_rates)], 
                 color='#95a5a6', alpha=0.8, label='Incoherent')
ax2.fill_between(fine_ablation, 
                 [b + u for b, u in zip(fine_bug_rates, fine_unclear_rates)], 
                 100, color='#27ae60', alpha=0.8, label='Correct (would say 9.8)')

# Add critical transition line
critical_point = -4.325  # Approximate midpoint of transition
ax2.axvline(x=critical_point, color='black', linestyle='--', linewidth=3, alpha=0.7)
ax2.text(critical_point + 0.01, 50, 'Critical\nTransition', fontsize=16, fontweight='bold', 
         ha='left', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

ax2.set_xlabel('Ablation Value', fontsize=20, fontweight='bold')
ax2.set_ylabel('Response Distribution (%)', fontsize=20, fontweight='bold')
ax2.set_title('Fine-Grained Analysis: Transition Zone Detail', fontsize=24, fontweight='bold', pad=15)
ax2.set_xlim(-4.5, -4.0)
ax2.set_ylim(0, 100)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=18, loc='right')

# Add annotations
ax2.annotate('No "Sweet Spot"', xy=(-4.325, 50), xytext=(-4.45, 20),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=18, fontweight='bold', color='red',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7))

# Add example outputs as text
fig.text(0.5, 0.02, 
         'Example outputs: Ablation 0: "9.11 is bigger than 9.8" → Ablation -4.5: "The number 100 is not mentioned..."',
         ha='center', fontsize=14, style='italic', wrap=True)

plt.tight_layout()

# Save as PNG
plt.savefig('parameter_sweep_enhanced.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
print("Saved as parameter_sweep_enhanced.png")

# Save as PDF
plt.savefig('parameter_sweep_enhanced.pdf', format='pdf', bbox_inches='tight', pad_inches=0.3)
print("Saved as parameter_sweep_enhanced.pdf")

# Create a simplified single-plot version with extra large fonts
fig2, ax = plt.subplots(figsize=(16, 10))

# Increase font sizes even more
mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.titlesize'] = 32
mpl.rcParams['axes.labelsize'] = 28
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['legend.fontsize'] = 24

# Plot main sweep with thicker lines
ax.plot(ablation_values, bug_rates, 'o-', color='#e74c3c', linewidth=5, 
        markersize=16, label='Bug Rate (%)', markeredgewidth=2, markeredgecolor='darkred')

# Add coherence normalized to 0-100 scale
coherence_normalized = [(c/4) * 100 for c in coherence_scores]
ax.plot(ablation_values, coherence_normalized, 's--', color='#3498db', linewidth=5, 
        markersize=14, label='Coherence (normalized)', markeredgewidth=2, markeredgecolor='darkblue')

# Highlight transition zone
ax.axvspan(-4.5, -4.0, alpha=0.3, color='yellow')
ax.text(-4.25, 50, 'TRANSITION\nZONE', fontsize=28, fontweight='bold', 
        ha='center', va='center', rotation=0,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8))

# Labels and title
ax.set_xlabel('Ablation Strength', fontsize=32, fontweight='bold')
ax.set_ylabel('Percentage', fontsize=32, fontweight='bold')
ax.set_title('Ablation Parameter Sweep: No "Sweet Spot" Exists\nIrremediable Entanglement in Llama-3.1-8B-Instruct', 
             fontsize=36, fontweight='bold', pad=30)

# Grid and limits
ax.grid(True, alpha=0.4, linewidth=2)
ax.set_xlim(-5.5, 0.5)
ax.set_ylim(-5, 105)

# Legend
ax.legend(fontsize=28, loc='lower left', frameon=True, fancybox=True, shadow=True)

# Add key finding annotation
ax.annotate('Direct transition:\nBuggy → Incoherent\n(No fix possible)', 
            xy=(-4.3, 50), xytext=(-2.5, 30),
            arrowprops=dict(arrowstyle='->', color='red', lw=4),
            fontsize=26, fontweight='bold', color='red',
            bbox=dict(boxstyle="round,pad=0.7", facecolor="white", edgecolor='red', linewidth=3))

# Make tick marks more visible
ax.tick_params(axis='both', which='major', labelsize=24, width=3, length=12)

plt.tight_layout()

# Save simplified version
plt.savefig('parameter_sweep_simple_xlarge.png', dpi=300, bbox_inches='tight', pad_inches=0.4)
print("Saved as parameter_sweep_simple_xlarge.png")

plt.savefig('parameter_sweep_simple_xlarge.pdf', format='pdf', bbox_inches='tight', pad_inches=0.4)
print("Saved as parameter_sweep_simple_xlarge.pdf")

plt.close('all')