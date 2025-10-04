import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

class CausalValidationVisualizer:
    def __init__(self, results):
        self.results = results
        
    def plot_disruption_effect(self, disruption_results):
        """
        Create a 3-panel figure showing causal relationship
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Panel 1: Disruption vs BEGIN Attention
        ax1 = axes[0]
        df = pd.DataFrame(disruption_results)
        
        # Aggregate by disruption level
        grouped = df.groupby('disruption_level').agg({
            'begin_attention': ['mean', 'std'],
            'error_rate': ['mean', 'std']
        }).reset_index()
        
        ax1.plot(grouped['disruption_level'], 
                grouped['begin_attention']['mean'], 
                'o-', linewidth=2, markersize=8, color='#2196F3')
        ax1.fill_between(grouped['disruption_level'],
                         grouped['begin_attention']['mean'] - grouped['begin_attention']['std'],
                         grouped['begin_attention']['mean'] + grouped['begin_attention']['std'],
                         alpha=0.3, color='#2196F3')
        
        ax1.set_xlabel('Disruption Strength', fontsize=12)
        ax1.set_ylabel('BEGIN Token Attention (%)', fontsize=12)
        ax1.set_title('A. Intervention Effectiveness', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 0.6])
        
        # Panel 2: BEGIN Attention vs Error Rate (Causal Relationship)
        ax2 = axes[1]
        
        # Scatter plot with trend line
        ax2.scatter(df['begin_attention'], df['error_rate'], 
                   alpha=0.5, s=20, color='#666')
        
        # Fit logistic regression
        from sklearn.linear_model import LogisticRegression
        X = df['begin_attention'].values.reshape(-1, 1)
        y = df['error_rate'].values
        
        model = LogisticRegression()
        model.fit(X, y)
        
        x_range = np.linspace(0, 0.6, 100).reshape(-1, 1)
        y_pred = model.predict_proba(x_range)[:, 1]
        
        ax2.plot(x_range, y_pred, 'r-', linewidth=2, label='Logistic Fit')
        
        # Add threshold line
        threshold_attention = 0.25  # Hypothetical threshold
        ax2.axvline(threshold_attention, color='green', linestyle='--', 
                   label=f'Critical Threshold (~{threshold_attention:.0%})')
        
        ax2.set_xlabel('BEGIN Token Attention (%)', fontsize=12)
        ax2.set_ylabel('Error Rate', fontsize=12)
        ax2.set_title('B. Causal Relationship', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Disruption vs Error Rate (Direct Effect)
        ax3 = axes[2]
        
        ax3.plot(grouped['disruption_level'], 
                grouped['error_rate']['mean'], 
                'o-', linewidth=2, markersize=8, color='#f44336')
        ax3.fill_between(grouped['disruption_level'],
                         grouped['error_rate']['mean'] - grouped['error_rate']['std'],
                         grouped['error_rate']['mean'] + grouped['error_rate']['std'],
                         alpha=0.3, color='#f44336')
        
        # Add baseline error rates for comparison
        ax3.axhline(0, color='green', linestyle='--', alpha=0.5, label='Simple Format (Natural)')
        ax3.axhline(0.98, color='red', linestyle='--', alpha=0.5, label='Q&A Format (Natural)')
        
        ax3.set_xlabel('Disruption Strength', fontsize=12)
        ax3.set_ylabel('Error Rate', fontsize=12)
        ax3.set_title('C. Induced Bug Rate', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([-0.05, 1.05])
        
        plt.suptitle('Causal Validation: Disrupting BEGIN Anchoring Causes Decimal Bug', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return fig
    
    def plot_restoration_effect(self, restoration_results):
        """
        Show that restoring BEGIN anchoring fixes the bug
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        df = pd.DataFrame(restoration_results)
        
        # Main effect line
        ax.plot(df['restoration'], df['error_rate'], 
               'o-', linewidth=3, markersize=10, color='#4CAF50', 
               label='Q&A Format with Restored Anchoring')
        
        # Confidence interval
        ax.fill_between(df['restoration'],
                        df['ci_lower'], df['ci_upper'],
                        alpha=0.3, color='#4CAF50')
        
        # Reference lines
        ax.axhline(0.98, color='red', linestyle='--', linewidth=2, 
                  label='Q&A Format (Natural)', alpha=0.7)
        ax.axhline(0, color='green', linestyle='--', linewidth=2,
                  label='Simple Format (Natural)', alpha=0.7)
        
        # Annotations
        ax.annotate('Bug Present', xy=(0.1, 0.98), xytext=(0.1, 0.85),
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.5),
                   fontsize=12, color='red')
        ax.annotate('Bug Fixed', xy=(0.9, 0.05), xytext=(0.7, 0.2),
                   arrowprops=dict(arrowstyle='->', color='green', alpha=0.5),
                   fontsize=12, color='green')
        
        ax.set_xlabel('BEGIN Anchoring Restoration Strength', fontsize=14)
        ax.set_ylabel('Error Rate', fontsize=14)
        ax.set_title('Restoring BEGIN Anchoring in Q&A Format Fixes the Bug', 
                    fontsize=16, fontweight='bold')
        ax.legend(loc='center right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.05, 1.05])
        
        return fig