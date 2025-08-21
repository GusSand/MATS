#!/usr/bin/env python

import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

matplotlib.pyplot.switch_backend('agg')

def create_surgical_precision_figure():
    sns.set_style('whitegrid')
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create a network diagram showing layers 8-12
    layers = np.arange(8, 13)
    components = ['Attention', 'MLP', 'Full Layer']
    
    # Create a grid showing intervention success rates
    # Based on experimental results where we tested each component
    results = np.array([
        [2, 0, 5],    # Layer 8 - minimal effect
        [8, 3, 10],    # Layer 9 - slight improvement 
        [100, 15, 25],  # Layer 10 - ONLY attention fully works
        [5, 2, 8],    # Layer 11 - minimal effect
        [3, 0, 4],    # Layer 12 - minimal effect
    ])
    
    # Create custom colormap: red (0) to yellow (50) to green (100)
    colors_list = ['#f44336', '#ffeb3b', '#4CAF50']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom', colors_list, N=n_bins)
    
    # Plot the heatmap
    im = ax.imshow(results.T, cmap=cmap, vmin=0, vmax=100, aspect='auto', interpolation='nearest')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(layers)))
    ax.set_yticks(np.arange(len(components)))
    ax.set_xticklabels([f'Layer {l}' for l in layers], fontsize=13, fontweight='bold')
    ax.set_yticklabels(components, fontsize=13, fontweight='bold')
    ax.set_xlabel('Transformer Layer', fontsize=15, fontweight='bold')
    ax.set_ylabel('Component Type', fontsize=15, fontweight='bold')
    
    # Add text annotations with enhanced styling
    for i in range(len(layers)):
        for j in range(len(components)):
            value = results[i, j]
            if value < 20:
                text_color = 'white'
                text_label = f'{value:.0f}%'
                fontweight = 'normal'
            elif value < 50:
                text_color = 'black'
                text_label = f'{value:.0f}%'
                fontweight = 'normal'
            else:
                text_color = 'white'
                text_label = f'{value:.0f}%\nSUCCESS'
                fontweight = 'bold'
            
            text = ax.text(i, j, text_label,
                          ha="center", va="center", color=text_color, 
                          fontsize=11, fontweight=fontweight)
    
    # Highlight the winning cell with a thick border
    rect = patches.Rectangle((1.5, -0.5), 1, 1, fill=False, 
                            edgecolor='blue', linewidth=4, linestyle='-')
    ax.add_patch(rect)
    
    # Add glow effect around the winning cell
    for offset in [0.05, 0.1, 0.15]:
        rect_glow = patches.Rectangle((1.5-offset, -0.5-offset), 1+2*offset, 1+2*offset, 
                                     fill=False, edgecolor='blue', linewidth=1, 
                                     alpha=0.3-offset, linestyle='-')
        ax.add_patch(rect_glow)
    
    # Add annotation arrow pointing to the successful cell
    ax.annotate('Only this works!', xy=(2, 0), xytext=(3.5, -0.7),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='blue'),
                fontsize=13, fontweight='bold', color='blue',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7, edgecolor='blue'))
    
    # Add grid lines for clarity
    for i in range(len(layers) + 1):
        ax.axvline(x=i-0.5, color='gray', linewidth=0.5, alpha=0.5)
    for j in range(len(components) + 1):
        ax.axhline(y=j-0.5, color='gray', linewidth=0.5, alpha=0.5)
    
    # Set title
    ax.set_title('Surgical Precision: Only Layer 10 Attention Succeeds', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Add colorbar with custom label
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Intervention Success Rate (%)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)
    
    # Add statistics box
    stats_text = (
        "Intervention Results:\n"
        "━━━━━━━━━━━━━━━━━\n"
        "✓ Layer 10 Attention: 100%\n"
        "△ Layer 10 Full: 25%\n"
        "△ Layer 10 MLP: 15%\n"
        "✗ Other layers: <10%\n"
        "━━━━━━━━━━━━━━━━━\n"
        "Success threshold: >95%\n"
        "Total tested: 15 configs"
    )
    
    props = dict(boxstyle='round', facecolor='lightgray', alpha=0.9, edgecolor='black', linewidth=1.5)
    ax.text(1.02, 0.5, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', bbox=props, family='monospace')
    
    # Remove default spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Add custom frame
    rect_frame = patches.Rectangle((-0.5, -0.5), len(layers), len(components), 
                                  fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(rect_frame)
    
    plt.tight_layout()
    
    # Save in multiple formats
    fig.savefig('surgical_precision_figure.pdf', dpi=300, bbox_inches='tight')
    fig.savefig('surgical_precision_figure.png', dpi=300, bbox_inches='tight')
    print("Surgical precision figure saved as surgical_precision_figure.pdf and surgical_precision_figure.png")

if __name__ == "__main__":
    create_surgical_precision_figure()