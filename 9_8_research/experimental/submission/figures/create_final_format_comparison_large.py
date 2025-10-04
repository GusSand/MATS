#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from textwrap import wrap

# Set up matplotlib for publication quality (matching sample_viz.py)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.switch_backend('agg')  # Fixed: removed the extra .pyplot

# Set much larger font sizes for document insertion (matching sample_viz.py style)
mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.titlesize'] = 50  # Increased from 32
mpl.rcParams['axes.labelsize'] = 40  # Increased from 28
mpl.rcParams['xtick.labelsize'] = 35  # Increased from 24
mpl.rcParams['ytick.labelsize'] = 35  # Increased from 24
mpl.rcParams['legend.fontsize'] = 30  # Increased from 24

# Set seaborn style to match sample_viz.py
import seaborn as sns
sns.set_style('whitegrid')

# Chart Title
chart_title = "Which is bigger: 9.8 or 9.11?"

# Data
formats_all = ['Chat Template', 'Q&A Format', 'Simple Format']
error_rates_all = [100, 90, 0]
correct_rates_all = [0, 10, 100]

# Create figure with more space
fig, ax = plt.subplots(figsize=(16, 10))

x = np.arange(len(formats_all))
width = 0.2  # Reduced from 0.3 to make bars narrower and more spaced out

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
                    fontsize=35, fontweight='bold')  # Increased from 28

for bar in bars2:
    height = bar.get_height()
    if height > 0:
        ax.annotate(f'{int(height)}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=35, fontweight='bold')  # Increased from 28

# Customize chart with larger text
ax.set_xlabel('Prompt Format', fontsize=40, fontweight='bold')  # Increased from 32
ax.set_ylabel('% of Responses', fontsize=40, fontweight='bold')  # Increased from 32
ax.set_title(chart_title, 
              fontsize=50, fontweight='bold', pad=35)  # Increased from 36
ax.set_xticks(x)
ax.set_xticklabels(formats_all, fontsize=35)  # Increased from 28
ax.legend(fontsize=30, loc='center left', bbox_to_anchor=(1.02, 0.5))  # Move outside to the right
ax.set_ylim(0, 125)
ax.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.5)  # Changed to match sample_viz.py style

# Add dividing line between buggy and correct formats
ax.axvline(x=1.5, color='black', linestyle='--', linewidth=3, alpha=0.5)

# Make tick marks more visible
ax.tick_params(axis='both', which='major', labelsize=35, width=2, length=8)  # Increased from 24

# Remove spines to match sample_viz.py style
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)  # Added this line to match sample_viz.py

# FIXED: Add more space between x-axis labels and tick marks
ax.tick_params(axis='x', pad=20)  # Increased padding between x-axis labels and tick marks

plt.tight_layout()

# Save as PNG
file_path = '/home/paperspace/dev/MATS9/submission/figures/llama_format_comparison_large2.png'
plt.savefig(file_path, 
            dpi=300, bbox_inches='tight', pad_inches=0.3)
print(f"Saved as {file_path}")

# Save as PDF for LaTeX
file_path = '/home/paperspace/dev/MATS9/submission/figures/llama_format_comparison_large2.pdf'
plt.savefig(file_path, 
            format='pdf', bbox_inches='tight', pad_inches=0.3)
print(f"Saved as {file_path}")

plt.close()

# Also create an EXTRA large version
mpl.rcParams['font.size'] = 26
mpl.rcParams['axes.titlesize'] = 60  # Increased from 40
mpl.rcParams['axes.labelsize'] = 50  # Increased from 36
mpl.rcParams['xtick.labelsize'] = 45  # Increased from 32
mpl.rcParams['ytick.labelsize'] = 45  # Increased from 32
mpl.rcParams['legend.fontsize'] = 40  # Increased from 32

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
                    fontsize=45, fontweight='bold')  # Increased from 36

for bar in bars4:
    height = bar.get_height()
    if height > 0:
        ax2.annotate(f'{int(height)}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 12),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=45, fontweight='bold')  # Increased from 36

ax2.set_xlabel('Prompt Format', fontsize=50, fontweight='bold')  # Increased from 40
ax2.set_ylabel('Percentage of Responses', fontsize=50, fontweight='bold')  # Increased from 40
ax2.set_title(chart_title, 
              fontsize=60, fontweight='bold', pad=40)  # Increased from 44
ax2.set_xticks(x)
ax2.set_xticklabels(formats_all, fontsize=45)  # Increased from 36
ax2.legend(fontsize=40, loc='center left', bbox_to_anchor=(1.02, 0.5))  # Move outside to the right
ax2.set_ylim(0, 130)
ax2.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.5)  # Changed to match sample_viz.py style

ax2.axvline(x=1.5, color='black', linestyle='--', linewidth=4, alpha=0.5)

# Make tick marks even more visible
ax2.tick_params(axis='both', which='major', labelsize=45, width=3, length=10)  # Increased from 32

# Remove spines to match sample_viz.py style
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)  # Added this line to match sample_viz.py

# FIXED: Add more space between x-axis labels and tick marks
ax2.tick_params(axis='x', pad=25)  # Increased padding between x-axis labels and tick marks

plt.tight_layout()

# Save extra large versions
file_path = '/home/paperspace/dev/MATS9/submission/figures/llama_format_comparison_xlarge2.png'
plt.savefig(file_path, 
            dpi=300, bbox_inches='tight', pad_inches=0.4)
print(f"Saved as {file_path}")

file_path = '/home/paperspace/dev/MATS9/submission/figures/llama_format_comparison_xlarge2.pdf' 
plt.savefig(file_path, 
            format='pdf', bbox_inches='tight', pad_inches=0.4)
print(f"Saved as {file_path}")

plt.close()