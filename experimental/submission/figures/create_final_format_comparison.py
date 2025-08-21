#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Set font sizes globally for better readability
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 14

# Data
formats_all = ['Chat Template', 'Q&A Format', 'Simple Format']
error_rates_all = [100, 90, 0]
correct_rates_all = [0, 10, 100]

# Create figure with more space
fig, ax = plt.subplots(figsize=(14, 9))

x = np.arange(len(formats_all))
width = 0.3  # Narrower bars for more spacing

# Create bars
bars1 = ax.bar(x - width/2, error_rates_all, width, label='Incorrect (Bug)', color='#e74c3c', alpha=0.8)
bars2 = ax.bar(x + width/2, correct_rates_all, width, label='Correct', color='#27ae60', alpha=0.8)

# Add value labels with more spacing
for bar in bars1:
    height = bar.get_height()
    if height > 0:
        ax.annotate(f'{int(height)}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=18, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    if height > 0:
        ax.annotate(f'{int(height)}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=18, fontweight='bold')

# Customize chart
ax.set_xlabel('Prompt Format', fontsize=20, fontweight='bold')
ax.set_ylabel('Percentage of Responses', fontsize=20, fontweight='bold')
ax.set_title('Llama-3.1-8B-Instruct Decimal Bug Across All Formats\n"Which is bigger: 9.8 or 9.11?"', 
              fontsize=24, fontweight='bold', pad=30)
ax.set_xticks(x)
ax.set_xticklabels(formats_all, fontsize=18)
ax.legend(fontsize=18, loc='upper left')
ax.set_ylim(0, 125)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add dividing line between buggy and correct formats
ax.axvline(x=1.5, color='black', linestyle='--', linewidth=2.5, alpha=0.5)

plt.tight_layout()

# Save as PNG
plt.savefig('/home/paperspace/dev/MATS9/submission/figures/llama_format_comparison.png', 
            dpi=300, bbox_inches='tight', pad_inches=0.3)
print("Saved as figures/llama_format_comparison.png")

# Save as PDF for LaTeX
plt.savefig('/home/paperspace/dev/MATS9/submission/figures/llama_format_comparison.pdf', 
            format='pdf', bbox_inches='tight', pad_inches=0.3)
print("Saved as figures/llama_format_comparison.pdf")

plt.close()