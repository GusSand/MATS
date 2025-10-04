# visualize_results.py
"""
Generate figures quickly after experiments
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_phase_transition():
    df = pd.read_csv('results/phase_transition.csv')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by ablation value and calculate rates
    grouped = df.groupby('ablation_value').agg({
        'has_bug': 'mean',
        'is_coherent': 'mean'
    }).reset_index()
    
    ax.plot(grouped['ablation_value'], grouped['has_bug'] * 100, 
            'r-', linewidth=2, label='Bug Rate')
    ax.plot(grouped['ablation_value'], grouped['is_coherent'] * 100,
            'b-', linewidth=2, label='Coherence')
    
    ax.axvspan(-4.3, -4.5, alpha=0.3, color='yellow', label='Transition Zone')
    
    ax.set_xlabel('Ablation Strength', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Phase Transition in 9.11 Bug', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/phase_transition.png', dpi=300)
    print("✓ Phase transition plot saved")

def plot_causal_heatmap():
    effects = np.load('results/causal_effects.npy', allow_pickle=True).item()
    
    # Convert to matrix
    n_layers = 32
    n_positions = len(list(effects.values())[0])
    
    matrix = np.zeros((n_layers, n_positions))
    for layer, layer_effects in effects.items():
        matrix[layer] = layer_effects
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(matrix, cmap='RdBu', center=0, 
                xticklabels=10, yticklabels=2,
                cbar_kws={'label': 'Causal Effect'})
    
    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title('Causal Tracing: 9.11 Bug Information Flow', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/causal_tracing.png', dpi=300)
    print("✓ Causal tracing heatmap saved")

if __name__ == "__main__":
    plot_phase_transition()
    plot_causal_heatmap()
    print("\nAll figures generated!")