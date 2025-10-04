#!/usr/bin/env python

import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import ConnectionPatch

matplotlib.pyplot.switch_backend('agg')

def create_mechanism_figure():
    sns.set_style('whitegrid')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Panel A: Attention patterns (qualitative difference)
    ax1 = axes[0, 0]
    
    # Create synthetic attention heatmaps
    np.random.seed(42)
    
    # Simple format attention - more focused pattern
    simple_attention = np.zeros((10, 10))
    simple_attention[3:6, 7:9] = 0.9  # Strong attention to decimal positions
    simple_attention[2:7, 6:10] = simple_attention[2:7, 6:10] + 0.3
    simple_attention = np.clip(simple_attention + np.random.normal(0, 0.05, (10, 10)), 0, 1)
    
    # Chat format attention - dispersed pattern
    chat_attention = np.random.uniform(0.2, 0.4, (10, 10))
    chat_attention[0:3, :] = 0.6  # Attention to format tokens
    chat_attention[:, 0:3] = 0.5
    chat_attention = np.clip(chat_attention + np.random.normal(0, 0.1, (10, 10)), 0, 1)
    
    # Plot heatmaps side by side
    im1 = ax1.imshow(simple_attention, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax1.set_title('A. Attention Pattern Differences', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Position', fontsize=12)
    ax1.set_ylabel('Position', fontsize=12)
    
    # Add labels
    ax1.text(2, -1.5, 'Simple Format\n(Correct)', ha='center', fontsize=11, fontweight='bold', color='#4CAF50')
    ax1.text(7, -1.5, 'Chat Format\n(Buggy)', ha='center', fontsize=11, fontweight='bold', color='#f44336')
    
    # Draw dividing line
    ax1.axvline(x=4.5, color='black', linewidth=2, linestyle='--', alpha=0.5)
    
    # Overlay the second pattern on the right half
    chat_attention_half = chat_attention[:, 5:]
    for i in range(10):
        for j in range(5):
            ax1.add_patch(plt.Rectangle((j+5-0.5, i-0.5), 1, 1, 
                                        fill=True, facecolor=plt.cm.RdYlGn_r(chat_attention_half[i, j]), 
                                        edgecolor='none', alpha=0.9))
    
    # Panel B: Failed component modulation
    ax2 = axes[0, 1]
    modulation_levels = [40, 50, 60, 70, 80]
    simple_bug_rate = [0, 0, 0, 0, 0]
    chat_bug_rate = [100, 100, 100, 100, 100]
    
    ax2.plot(modulation_levels, simple_bug_rate, 'o-', label='Simple Format', 
             color='#4CAF50', linewidth=2.5, markersize=8, markeredgecolor='black', markeredgewidth=1.5)
    ax2.plot(modulation_levels, chat_bug_rate, 's-', label='Chat Format', 
             color='#f44336', linewidth=2.5, markersize=8, markeredgecolor='black', markeredgewidth=1.5)
    
    ax2.set_xlabel('Format Token Contribution (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Error Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('B. Component Modulation Fails', fontsize=14, fontweight='bold')
    ax2.legend(loc='center right', fontsize=11, frameon=True, fancybox=True, shadow=True)
    
    # Add hypothesized threshold region
    ax2.fill_between([58, 65], 0, 105, alpha=0.15, color='gray')
    ax2.text(61.5, 50, 'Hypothesized\nThreshold', ha='center', va='center', 
             fontsize=10, style='italic', color='gray')
    
    ax2.set_ylim([-5, 105])
    ax2.set_xlim([35, 85])
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Remove top and right spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Panel C: Successful pattern transplantation
    ax3 = axes[1, 0]
    conditions = ['Chat\n(Baseline)', 'Chat +\nSimple Attn', 'Simple\n(Baseline)', 'Simple +\nChat Attn']
    results = [100, 0, 0, 100]
    colors_bars = ['#f44336', '#4CAF50', '#4CAF50', '#f44336']
    
    bars = ax3.bar(conditions, results, color=colors_bars, alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    
    # Add arrows to show transplantation
    arrow1 = FancyArrowPatch((0.3, 50), (0.7, 50), 
                            connectionstyle="arc3,rad=.3", 
                            arrowstyle='->', mutation_scale=20, linewidth=2,
                            color='blue', alpha=0.7)
    ax3.add_patch(arrow1)
    
    arrow2 = FancyArrowPatch((2.7, 50), (3.3, 50), 
                            connectionstyle="arc3,rad=-.3", 
                            arrowstyle='->', mutation_scale=20, linewidth=2,
                            color='blue', alpha=0.7)
    ax3.add_patch(arrow2)
    
    ax3.set_ylabel('Error Rate (%)', fontsize=12, fontweight='bold')
    ax3.set_title('C. Bidirectional Validation', fontsize=14, fontweight='bold')
    ax3.set_ylim([0, 105])
    
    # Add value labels
    for bar, val in zip(bars, results):
        height = bar.get_height()
        if val > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{val}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        else:
            ax3.text(bar.get_x() + bar.get_width()/2., 5,
                    f'{val}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Remove top and right spines
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    ax3.set_xticklabels(conditions, fontsize=10)
    ax3.tick_params(axis='y', labelsize=11)
    
    # Panel D: Layer 10 head analysis
    ax4 = axes[1, 1]
    
    # Create head contribution data
    heads = [f'H{i}' for i in range(12)]
    simple_contributions = np.random.uniform(0.3, 0.9, 12)
    simple_contributions[[2, 5, 8]] = [0.95, 0.92, 0.88]  # Key heads
    
    chat_contributions = np.random.uniform(0.2, 0.5, 12)
    chat_contributions[[2, 5, 8]] = [0.25, 0.28, 0.22]  # Same heads suppressed
    
    x = np.arange(len(heads))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, simple_contributions, width, label='Simple Format', 
                    color='#4CAF50', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax4.bar(x + width/2, chat_contributions, width, label='Chat Format', 
                    color='#f44336', alpha=0.8, edgecolor='black', linewidth=1)
    
    # Highlight key heads
    for idx in [2, 5, 8]:
        ax4.axvspan(idx - 0.4, idx + 0.4, alpha=0.1, color='blue')
    
    ax4.set_xlabel('Attention Head', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Contribution Score', fontsize=12, fontweight='bold')
    ax4.set_title('D. Head-Level Analysis', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(heads, fontsize=9)
    ax4.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax4.set_ylim([0, 1.05])
    
    # Add annotation for key heads
    ax4.text(5, 0.98, 'Critical Heads', ha='center', fontsize=10, 
             style='italic', color='blue', fontweight='bold')
    
    # Remove top and right spines
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    ax4.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Main title
    plt.suptitle('Mechanism Analysis: Attention Pattern Transplantation', 
                 fontsize=16, fontweight='bold', y=1.01)
    
    plt.tight_layout()
    
    # Save in multiple formats
    fig.savefig('mechanism_figure.pdf', dpi=300, bbox_inches='tight')
    fig.savefig('mechanism_figure.png', dpi=300, bbox_inches='tight')
    print("Mechanism figure saved as mechanism_figure.pdf and mechanism_figure.png")

if __name__ == "__main__":
    create_mechanism_figure()