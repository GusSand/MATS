#!/usr/bin/env python
"""
Create professional visualization of even/odd attention head discovery
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy import stats

# Set font rendering for publication quality
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['axes.titleweight'] = 'bold'
plt.switch_backend('agg')

# Set seaborn style
sns.set_style('whitegrid')

def create_even_odd_discovery_figure():
    # Create figure with 4 panels
    fig = plt.figure(figsize=(16, 12))
    
    # Panel A: Even vs Odd Heads Comparison
    ax1 = plt.subplot(2, 2, 1)
    
    # Data for even vs odd comparison
    head_types = ['Even Heads\n(0,2,4,...,30)', 'Odd Heads\n(1,3,5,...,31)']
    success_rates = [100, 0]
    colors = ['#2ecc71', '#e74c3c']  # Green for success, red for failure
    
    bars = ax1.bar(head_types, success_rates, color=colors, width=0.6, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{rate}%', ha='center', va='center', 
                fontsize=24, fontweight='bold', color='white')
    
    # Add n=100 annotation
    ax1.text(0.5, 0.95, 'n=100 trials', transform=ax1.transAxes,
            ha='center', va='top', fontsize=14, fontweight='bold')
    
    ax1.set_ylabel('Success Rate (%)', fontsize=18, fontweight='bold')
    ax1.set_title('A. Even vs Odd Head Specialization', fontsize=20, fontweight='bold', pad=20)
    ax1.set_ylim(0, 110)
    ax1.set_xticklabels(head_types, fontsize=16, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.3)
    
    # Panel B: Threshold Discovery
    ax2 = plt.subplot(2, 2, 2)
    
    num_heads = [16, 8, 4, 2, 1]
    even_success = [100, 100, 0, 0, 0]
    
    ax2.plot(num_heads, even_success, 'o-', color='#2ecc71', linewidth=3, 
             markersize=12, markeredgecolor='black', markeredgewidth=2)
    
    # Add threshold zone
    ax2.axvspan(4, 8, alpha=0.2, color='gold', label='Critical Threshold')
    ax2.axvline(x=6, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.text(6, 50, 'Threshold:\n8 heads required', ha='center', va='center',
            fontsize=14, fontweight='bold', bbox=dict(boxstyle='round', 
            facecolor='white', edgecolor='red', linewidth=2))
    
    ax2.set_xlabel('Number of Even Heads', fontsize=18, fontweight='bold')
    ax2.set_ylabel('Success Rate (%)', fontsize=18, fontweight='bold')
    ax2.set_title('B. Sharp Threshold at 8 Even Heads', fontsize=20, fontweight='bold', pad=20)
    ax2.set_xticks(num_heads)
    ax2.set_xticklabels(num_heads, fontsize=14, fontweight='bold')
    ax2.set_ylim(-5, 110)
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(loc='upper right', fontsize=12, frameon=True)
    
    # Panel C: Individual Head Redundancy
    ax3 = plt.subplot(2, 2, 3)
    
    # Testing removal of individual even heads
    removed_heads = ['None\n(All 16)', 'Remove\nHead 0', 'Remove\nHead 2', 
                     'Remove\nHead 4', 'Remove\nHead 6']
    redundancy_success = [100, 100, 100, 100, 100]
    
    bars = ax3.bar(range(len(removed_heads)), redundancy_success, 
                   color='#3498db', width=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, rate in zip(bars, redundancy_success):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{rate}%', ha='center', va='center',
                fontsize=18, fontweight='bold', color='white')
    
    ax3.set_xticks(range(len(removed_heads)))
    ax3.set_xticklabels(removed_heads, fontsize=12, fontweight='bold', rotation=0)
    ax3.set_ylabel('Success Rate (%)', fontsize=18, fontweight='bold')
    ax3.set_title('C. Perfect Redundancy Among Even Heads', fontsize=20, fontweight='bold', pad=20)
    ax3.set_ylim(0, 110)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(axis='y', alpha=0.3)
    
    # Panel D: Mixed Compositions  
    ax4 = plt.subplot(2, 2, 4)
    
    compositions = ['16 Even\n0 Odd', '8 Even\n8 Odd', '4 Even\n12 Odd', '0 Even\n16 Odd']
    mixed_success = [100, 0, 0, 0]
    colors_mixed = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
    
    bars = ax4.bar(range(len(compositions)), mixed_success, 
                   color=colors_mixed, width=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, rate in zip(bars, mixed_success):
        if rate > 0:
            ax4.text(bar.get_x() + bar.get_width()/2., rate/2,
                    f'{rate}%', ha='center', va='center',
                    fontsize=18, fontweight='bold', color='white')
        else:
            ax4.text(bar.get_x() + bar.get_width()/2., 5,
                    f'{rate}%', ha='center', va='center',
                    fontsize=18, fontweight='bold', color='black')
    
    ax4.set_xticks(range(len(compositions)))
    ax4.set_xticklabels(compositions, fontsize=12, fontweight='bold')
    ax4.set_ylabel('Success Rate (%)', fontsize=18, fontweight='bold')
    ax4.set_title('D. Odd Heads Incompatible with Task', fontsize=20, fontweight='bold', pad=20)
    ax4.set_ylim(0, 110)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.grid(axis='y', alpha=0.3)
    
    # Add super title
    fig.suptitle('Even/Odd Attention Head Specialization in Layer 10',
                 fontsize=24, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save as both PNG and PDF
    fig.savefig('/home/paperspace/dev/MATS9/working_scripts/even_odd_discovery.pdf', 
                dpi=300, bbox_inches='tight')
    fig.savefig('/home/paperspace/dev/MATS9/working_scripts/even_odd_discovery.png', 
                dpi=300, bbox_inches='tight')
    
    print("Created even_odd_discovery.pdf and even_odd_discovery.png")

if __name__ == "__main__":
    create_even_odd_discovery_figure()