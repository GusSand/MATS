#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import json
import matplotlib as mpl

# Set large font sizes
mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.titlesize'] = 22
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 16

# Load data
with open('../parameter_sweep_results.json', 'r') as f:
    sweep_data = json.load(f)

with open('../fine_sweep_results.json', 'r') as f:
    fine_data = json.load(f)

# Load circuit info to get the neurons
with open('../identified_circuits.json', 'r') as f:
    circuits = json.load(f)

# Get hijacker neurons
hijackers = circuits["chat"]["hijacker_cluster"]
unique_hijackers = list(set(tuple(n) for n in hijackers))
neuron_list = ", ".join([f"L{layer}/N{neuron}" for layer, neuron in sorted(unique_hijackers)[:3]]) + f" + {len(unique_hijackers)-3} more"

# Create figure with two subplots - increase figure size and adjust spacing
fig = plt.figure(figsize=(14, 16))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.35)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

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

# Title with more padding
ax1.set_title(f'Ablation Parameter Sweep: {len(unique_hijackers)} Hijacker Neurons\nLlama-3.1-8B-Instruct - Chat Template Circuit\n{neuron_list}', 
              fontsize=22, fontweight='bold', pad=25, linespacing=1.5)

# Legend - move to avoid overlap
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=16, 
           bbox_to_anchor=(0.98, 0.5))

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
ax2.set_title('Fine-Grained Analysis: Transition Zone Detail', fontsize=22, fontweight='bold', pad=15)
ax2.set_xlim(-4.5, -4.0)
ax2.set_ylim(0, 100)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=18, loc='right')

# Add annotations
ax2.annotate('No "Sweet Spot"', xy=(-4.325, 50), xytext=(-4.45, 20),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=18, fontweight='bold', color='red',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7))

# Add example outputs as text with better positioning
fig.text(0.5, 0.01, 
         'Example outputs: Ablation 0: "9.11 is bigger than 9.8" → Ablation -4.5: "The number 100 is not mentioned..."',
         ha='center', fontsize=14, style='italic', wrap=True,
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.7))

plt.tight_layout(rect=[0, 0.025, 1, 1])  # Leave space for bottom text

# Save as PNG
plt.savefig('parameter_sweep_enhanced_final.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
print("Saved as parameter_sweep_enhanced_final.png")

# Save as PDF
plt.savefig('parameter_sweep_enhanced_final.pdf', format='pdf', bbox_inches='tight', pad_inches=0.3)
print("Saved as parameter_sweep_enhanced_final.pdf")

# Also create a cleaner version with neuron info in a separate box
fig2 = plt.figure(figsize=(14, 16))
gs2 = fig2.add_gridspec(3, 1, height_ratios=[0.15, 1, 1], hspace=0.25)
ax_info = fig2.add_subplot(gs2[0])
ax3 = fig2.add_subplot(gs2[1])
ax4 = fig2.add_subplot(gs2[2])

# Info box at top
ax_info.axis('off')
info_text = f"Ablating {len(unique_hijackers)} Hijacker Neurons from Chat Template Circuit:\n{neuron_list}"
ax_info.text(0.5, 0.5, info_text, transform=ax_info.transAxes, 
             ha='center', va='center', fontsize=18,
             bbox=dict(boxstyle="round,pad=0.7", facecolor="lightblue", alpha=0.7))

# Repeat the plots without subtitle in title
# Main sweep plot
ax3.plot(ablation_values, bug_rates, 'o-', color=color1, linewidth=3, markersize=10, label='Bug Rate')
ax3.set_xlabel('Ablation Value', fontsize=20, fontweight='bold')
ax3.set_ylabel('Bug Rate (%)', color=color1, fontsize=20, fontweight='bold')
ax3.tick_params(axis='y', labelcolor=color1, labelsize=18)
ax3.set_ylim(-5, 105)
ax3.grid(True, alpha=0.3)

# Plot coherence on secondary axis
ax3_twin = ax3.twinx()
ax3_twin.plot(ablation_values, coherence_scores, 's--', color=color2, linewidth=3, markersize=10, label='Coherence')
ax3_twin.set_ylabel('Coherence Score (1-4)', color=color2, fontsize=20, fontweight='bold')
ax3_twin.tick_params(axis='y', labelcolor=color2, labelsize=18)
ax3_twin.set_ylim(0.5, 4.5)

# Add transition zone shading
ax3.axvspan(-4.5, -4.0, alpha=0.2, color='yellow', label='Transition Zone')

# Cleaner title
ax3.set_title('Ablation Parameter Sweep: Bug Rate vs Coherence\nLlama-3.1-8B-Instruct', 
              fontsize=24, fontweight='bold', pad=20)

# Legend
lines3, labels3 = ax3.get_legend_handles_labels()
lines4, labels4 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines3 + lines4, labels3 + labels4, loc='center right', fontsize=16)

# Fine sweep plot
ax4.fill_between(fine_ablation, 0, fine_bug_rates, color='#e74c3c', alpha=0.8, label='Bug (says 9.11)')
ax4.fill_between(fine_ablation, fine_bug_rates, 
                 [b + u for b, u in zip(fine_bug_rates, fine_unclear_rates)], 
                 color='#95a5a6', alpha=0.8, label='Incoherent')
ax4.fill_between(fine_ablation, 
                 [b + u for b, u in zip(fine_bug_rates, fine_unclear_rates)], 
                 100, color='#27ae60', alpha=0.8, label='Correct (would say 9.8)')

ax4.axvline(x=critical_point, color='black', linestyle='--', linewidth=3, alpha=0.7)
ax4.text(critical_point + 0.01, 50, 'Critical\nTransition', fontsize=16, fontweight='bold', 
         ha='left', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

ax4.set_xlabel('Ablation Value', fontsize=20, fontweight='bold')
ax4.set_ylabel('Response Distribution (%)', fontsize=20, fontweight='bold')
ax4.set_title('Fine-Grained Analysis: Transition Zone Detail', fontsize=22, fontweight='bold', pad=15)
ax4.set_xlim(-4.5, -4.0)
ax4.set_ylim(0, 100)
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=18, loc='right')

ax4.annotate('No "Sweet Spot"', xy=(-4.325, 50), xytext=(-4.45, 20),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=18, fontweight='bold', color='red',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7))

# Add example outputs
fig2.text(0.5, 0.01, 
         'Example outputs: Ablation 0: "9.11 is bigger than 9.8" → Ablation -4.5: "The number 100 is not mentioned..."',
         ha='center', fontsize=14, style='italic', wrap=True,
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.7))

plt.tight_layout(rect=[0, 0.025, 1, 1])

# Save clean version
plt.savefig('parameter_sweep_clean.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
print("Saved as parameter_sweep_clean.png")

plt.savefig('parameter_sweep_clean.pdf', format='pdf', bbox_inches='tight', pad_inches=0.3)
print("Saved as parameter_sweep_clean.pdf")

plt.close('all')