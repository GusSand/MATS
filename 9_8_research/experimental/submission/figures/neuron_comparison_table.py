#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Create figure for the table
fig, ax = plt.subplots(figsize=(16, 12))
ax.axis('tight')
ax.axis('off')

# Define the table data
headers = ['Layer', 'Buggy Response\n(Chat Template - "9.11 is bigger")', 'Non-Buggy Response\n(Simple Format - "9.8 is bigger")']

# Organize data by layer
table_data = [
    ['7', 'Neuron 1978', '-'],
    ['11', '-', 'Neuron 11862'],
    ['13', 'Neuron 10352', 'Neuron 10352'],
    ['14', 'Neuron 13315 (2x)\nNeuron 2451\nNeuron 12639 ⚠️', 'Neuron 13315 (2x)\nNeuron 12639 ⚠️'],
    ['15', 'Neuron 3136 (2x)\nNeuron 5076\nNeuron 421', 'Neuron 3136 (2x)\nNeuron 5076 (2x)\nNeuron 421'],
    ['28', 'Neuron 10823\nNeuron 8818\nNeuron 5336', 'Neuron 11450\nNeuron 12900\nNeuron 10823'],
    ['29', 'Neuron 664\nNeuron 12248\nNeuron 1435', 'Neuron 12248\nNeuron 10726\nNeuron 2836'],
    ['30', 'Neuron 840\nNeuron 13679\nNeuron 7305', 'Neuron 840\nNeuron 14215 (multiple)'],
    ['31', 'Neuron 801\nNeuron 4581\nNeuron 13336 (12.0)\nNeuron 12004 (11.5)\nNeuron 9692\nNeuron 2398\nNeuron 12111', 'Neuron 801\nNeuron 4581\nNeuron 13336 (14.8) ↑\nNeuron 12004 (14.4) ↑\nNeuron 9692\nNeuron 2398\nNeuron 9692']
]

# Create the table
table = ax.table(cellText=table_data, colLabels=headers, 
                 cellLoc='left', loc='center',
                 colWidths=[0.08, 0.46, 0.46])

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Header styling
for i in range(len(headers)):
    cell = table[(0, i)]
    cell.set_text_props(weight='bold', color='white')
    cell.set_facecolor('#34495e')
    cell.set_height(0.08)

# Row styling
for i in range(1, len(table_data) + 1):
    # Layer column
    cell = table[(i, 0)]
    cell.set_facecolor('#ecf0f1')
    cell.set_text_props(weight='bold')
    
    # Buggy response column
    cell = table[(i, 1)]
    cell.set_facecolor('#ffe6e6')
    
    # Non-buggy response column
    cell = table[(i, 2)]
    cell.set_facecolor('#e6ffe6')
    
    # Highlight layers with entangled neurons
    if table_data[i-1][0] in ['13', '14', '15']:
        for j in range(3):
            cell = table[(i, j)]
            cell.set_linewidth(2)
            cell.set_edgecolor('#e74c3c')

# Add title
plt.title('Neuron Activation Comparison: Buggy vs Non-Buggy Responses\nLlama-3.1-8B-Instruct on "Which is bigger: 9.8 or 9.11?"', 
          fontsize=20, fontweight='bold', pad=20)

# Add legend/notes
fig.text(0.5, 0.08, '⚠️ = Entangled neuron (fires in both buggy and correct responses)\n' +
                    '↑ = Higher activation in correct response\n' +
                    'Red borders = Layers with shared neurons between buggy/correct responses\n' +
                    'Layers 2-15: Hijacker Circuit | Layers 28-31: Reasoning Circuit',
         ha='center', fontsize=12, style='italic', wrap=True,
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.5))

plt.tight_layout()

# Save as PNG
plt.savefig('/home/paperspace/dev/MATS9/submission/figures/neuron_comparison_table.png', 
            dpi=300, bbox_inches='tight', pad_inches=0.3)
print("Saved as figures/neuron_comparison_table.png")

# Save as PDF
plt.savefig('/home/paperspace/dev/MATS9/submission/figures/neuron_comparison_table.pdf', 
            format='pdf', bbox_inches='tight', pad_inches=0.3)
print("Saved as figures/neuron_comparison_table.pdf")

# Create a simplified markdown version
markdown_table = """
# Neuron Firing Comparison: Buggy vs Non-Buggy Responses

| Layer | Buggy Response (Chat Template) | Non-Buggy Response (Simple Format) |
|-------|-------------------------------|-----------------------------------|
| **7** | Neuron 1978 | - |
| **11** | - | Neuron 11862 |
| **13** | Neuron 10352 | Neuron 10352 ✓ |
| **14** | Neuron 13315 (2x), 2451, **12639** ⚠️ | Neuron 13315 (2x), **12639** ⚠️ |
| **15** | Neuron 3136 (2x), 5076, 421 | Neuron 3136 (2x), 5076 (2x), 421 |
| **28** | Neuron 10823, 8818, 5336 | Neuron 11450, 12900, 10823 |
| **29** | Neuron 664, 12248, 1435 | Neuron 12248, 10726, 2836 |
| **30** | Neuron 840, 13679, 7305 | Neuron 840, 14215 (multiple) |
| **31** | Neuron 801, 4581, 13336 (12.0), 12004 (11.5), 9692, 2398, 12111 | Neuron 801, 4581, 13336 (14.8) ↑, 12004 (14.4) ↑, 9692, 2398 |

**Key:**
- ⚠️ = Entangled neuron (fires in both buggy and correct responses)
- ✓ = Shared neuron
- ↑ = Higher activation in correct response
- Layers 2-15: Hijacker Circuit
- Layers 28-31: Reasoning Circuit

**Critical Finding:** Layer 14, Neuron 12639 is active in BOTH responses, demonstrating irremediable entanglement.
"""

with open('/home/paperspace/dev/MATS9/submission/figures/neuron_comparison_table.md', 'w') as f:
    f.write(markdown_table)
print("Saved markdown version as figures/neuron_comparison_table.md")

plt.close()