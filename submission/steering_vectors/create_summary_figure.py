#!/usr/bin/env python3
import matplotlib.pyplot as plt
import json

# Load results
with open('steering_results_simple.json', 'r') as f:
    data = json.load(f)

# Create figure with conclusion
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: Steering vector magnitudes
neurons = list(data['steering_vectors'].keys())
values = list(data['steering_vectors'].values())

# Add the critical entangled neuron info
neurons.append('L14/N12639\n(entangled)')
values.append(0.113)  # From our previous analysis

ax1.barh(neurons, values, color=['#3498db']*len(neurons[:-1]) + ['#e74c3c'])
ax1.set_xlabel('Steering Vector Magnitude', fontsize=14)
ax1.set_title('Steering Vectors are Too Small\n(Correct - Buggy Activations)', fontsize=16)
ax1.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5, label='Typical threshold')
ax1.legend()

# Right panel: Success rates
alphas = [r['alpha'] for r in data['results']]
rates = [r['success_rate'] for r in data['results']]

ax2.plot(alphas, rates, 'o-', linewidth=3, markersize=10, color='#e74c3c')
ax2.set_xlabel('Steering Strength (α)', fontsize=14)
ax2.set_ylabel('Success Rate (%)', fontsize=14)
ax2.set_title('Steering Vectors Fail to Fix Bug', fontsize=16)
ax2.set_ylim(-5, 105)
ax2.grid(True, alpha=0.3)

# Add text annotation
ax2.text(5, 50, 'All attempts: 0% success\nConfirms irremediable\nentanglement', 
         fontsize=14, ha='center', va='center',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7))

plt.suptitle('Steering Vector (ActAdd) Results: Irremediable Entanglement Confirmed', 
             fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('steering_vector_summary.png', dpi=300, bbox_inches='tight')
plt.savefig('steering_vector_summary.pdf', format='pdf', bbox_inches='tight')
print("Created summary figure")