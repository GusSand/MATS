import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from attention.causal.disrupt import AttentionAnchoringExperiment
from attention.causal.viz import CausalValidationVisualizer
from attention.causal.stats import statistical_analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Initialize
experimenter = AttentionAnchoringExperiment()

# Run disruption experiment
print("Running disruption experiment...")
disruption_results = experimenter.run_causal_experiment(n_trials=10)  # Reduced for testing

# Run restoration experiment  
print("Running restoration experiment...")
restoration_results = experimenter.run_restoration_experiment(n_trials=10)  # Reduced for testing

# Statistical analysis
print("\nStatistical Analysis:")
stats_results = statistical_analysis(disruption_results)

# Generate visualizations
visualizer = CausalValidationVisualizer(disruption_results)
fig1 = visualizer.plot_disruption_effect(disruption_results)
fig2 = visualizer.plot_restoration_effect(restoration_results)

# Save figures
fig1.savefig('causal_validation_disruption.pdf', dpi=300, bbox_inches='tight')
fig2.savefig('causal_validation_restoration.pdf', dpi=300, bbox_inches='tight')

print("\nCausal validation complete!")

# Calculate summary statistics for the print statement
df = pd.DataFrame(disruption_results)
baseline_data = df[df['disruption_level'] == 0.0]
disrupted_data = df[df['disruption_level'] == 1.0]

if len(baseline_data) > 0 and len(disrupted_data) > 0:
    baseline = baseline_data['begin_attention'].mean()
    disrupted = disrupted_data['begin_attention'].mean()
    baseline_error = baseline_data['error_rate'].mean()
    disrupted_error = disrupted_data['error_rate'].mean()
    
    print(f"Key finding: Reducing BEGIN attention from {baseline:.1%} to {disrupted:.1%} ")
    print(f"increases error rate from {baseline_error:.1%} to {disrupted_error:.1%}")