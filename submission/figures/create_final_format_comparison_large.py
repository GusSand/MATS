#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Set much larger font sizes for document insertion
mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.titlesize'] = 32
mpl.rcParams['axes.labelsize'] = 28
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['legend.fontsize'] = 24

# Data
formats_all = ['Chat Template', 'Q&A Format', 'Simple Format']
error_rates_all = [100, 90, 0]
correct_rates_all = [0, 10, 100]

# Create figure with more space
fig, ax = plt.subplots(figsize=(16, 10))

x = np.arange(len(formats_all))
width = 0.3  # Narrower bars for more spacing

# Create bars
bars1 = ax.bar(x - width/2, error_rates_all, width, label='Incorrect (Bug)', color='#e74c3c', alpha=0.8)
bars2 = ax.bar(x + width/2, correct_rates_all, width, label='Correct', color='#27ae60', alpha=0.8)

# Add value labels with much larger font
for bar in bars1:
    height = bar.get_height()
    if height > 0:
        ax.annotate(f'{int(height)}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=28, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    if height > 0:
        ax.annotate(f'{int(height)}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=28, fontweight='bold')

# Customize chart with larger text
ax.set_xlabel('Prompt Format', fontsize=32, fontweight='bold')
ax.set_ylabel('Percentage of Responses', fontsize=32, fontweight='bold')
ax.set_title('Llama-3.1-8B-Instruct Decimal Bug Across All Formats\n"Which is bigger: 9.8 or 9.11?"', 
              fontsize=36, fontweight='bold', pad=35)
ax.set_xticks(x)
ax.set_xticklabels(formats_all, fontsize=28)
ax.legend(fontsize=28, loc='upper left')
ax.set_ylim(0, 125)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add dividing line between buggy and correct formats
ax.axvline(x=1.5, color='black', linestyle='--', linewidth=3, alpha=0.5)

# Make tick marks more visible
ax.tick_params(axis='both', which='major', labelsize=24, width=2, length=8)

plt.tight_layout()

# Save as PNG
plt.savefig('/home/paperspace/dev/MATS9/submission/figures/llama_format_comparison_large.png', 
            dpi=300, bbox_inches='tight', pad_inches=0.3)
print("Saved as figures/llama_format_comparison_large.png")

# Save as PDF for LaTeX
plt.savefig('/home/paperspace/dev/MATS9/submission/figures/llama_format_comparison_large.pdf', 
            format='pdf', bbox_inches='tight', pad_inches=0.3)
print("Saved as figures/llama_format_comparison_large.pdf")

plt.close()

# Also create an EXTRA large version
mpl.rcParams['font.size'] = 26
mpl.rcParams['axes.titlesize'] = 40
mpl.rcParams['axes.labelsize'] = 36
mpl.rcParams['xtick.labelsize'] = 32
mpl.rcParams['ytick.labelsize'] = 32
mpl.rcParams['legend.fontsize'] = 32

fig2, ax2 = plt.subplots(figsize=(18, 11))

bars3 = ax2.bar(x - width/2, error_rates_all, width, label='Incorrect (Bug)', color='#e74c3c', alpha=0.8, linewidth=2, edgecolor='darkred')
bars4 = ax2.bar(x + width/2, correct_rates_all, width, label='Correct', color='#27ae60', alpha=0.8, linewidth=2, edgecolor='darkgreen')

# Add value labels with even larger font
for bar in bars3:
    height = bar.get_height()
    if height > 0:
        ax2.annotate(f'{int(height)}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 12),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=36, fontweight='bold')

for bar in bars4:
    height = bar.get_height()
    if height > 0:
        ax2.annotate(f'{int(height)}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 12),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=36, fontweight='bold')

ax2.set_xlabel('Prompt Format', fontsize=40, fontweight='bold')
ax2.set_ylabel('Percentage of Responses', fontsize=40, fontweight='bold')
ax2.set_title('Llama-3.1-8B-Instruct Decimal Bug Across All Formats\n"Which is bigger: 9.8 or 9.11?"', 
              fontsize=44, fontweight='bold', pad=40)
ax2.set_xticks(x)
ax2.set_xticklabels(formats_all, fontsize=36)
ax2.legend(fontsize=36, loc='upper left')
ax2.set_ylim(0, 130)
ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.5)

ax2.axvline(x=1.5, color='black', linestyle='--', linewidth=4, alpha=0.5)

# Make tick marks even more visible
ax2.tick_params(axis='both', which='major', labelsize=32, width=3, length=10)

plt.tight_layout()

# Save extra large versions
plt.savefig('/home/paperspace/dev/MATS9/submission/figures/llama_format_comparison_xlarge.png', 
            dpi=300, bbox_inches='tight', pad_inches=0.4)
print("Saved as figures/llama_format_comparison_xlarge.png")

plt.savefig('/home/paperspace/dev/MATS9/submission/figures/llama_format_comparison_xlarge.pdf', 
            format='pdf', bbox_inches='tight', pad_inches=0.4)
print("Saved as figures/llama_format_comparison_xlarge.pdf")

plt.close()