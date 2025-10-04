#!/usr/bin/env python
"""
Parameter sweep to find the optimal ablation value that:
1. Eliminates the bug (9.11 > 9.8)
2. Maintains coherent output
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

print("🔬 Ablation Parameter Sweep: Finding the Sweet Spot")
print("="*70)

# Load the identified circuits
with open("identified_circuits.json", "r") as f:
    circuits = json.load(f)

# Get ALL hijacker neurons from chat format
all_hijackers = circuits["chat"]["hijacker_cluster"]
unique_hijackers = list(set(tuple(n) for n in all_hijackers))

print(f"Targeting {len(unique_hijackers)} unique hijacker neurons:")
for layer, neuron in sorted(unique_hijackers):
    print(f"  L{layer}/N{neuron}")

# Load model
print("\n" + "-"*70)
print("Loading model...")
model_path = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# Global variables
NEURONS_TO_ABLATE = set()
ABLATION_VALUE = 0.0

def ablation_hook(module, input, output, layer_idx):
    """Hook that sets specific neurons to ablation value"""
    if hasattr(output, 'clone'):
        new_output = output.clone()
        
        for neuron_layer, neuron_idx in NEURONS_TO_ABLATE:
            if neuron_layer == layer_idx:
                new_output[:, :, neuron_idx] = ABLATION_VALUE
        
        return new_output
    return output

def score_coherence(text):
    """
    Score the coherence of the output on a 1-5 scale.
    5 = Perfect answer with correct reasoning
    4 = Clear answer (right or wrong) with some reasoning
    3 = Clear answer but minimal reasoning
    2 = Confused but attempting to answer
    1 = Complete nonsense/garbage
    """
    text_lower = text.lower()
    
    # Check for clear comparison statements
    has_comparison = any(phrase in text_lower for phrase in [
        "is bigger", "is larger", "is greater", 
        "bigger than", "larger than", "greater than"
    ])
    
    # Check if it mentions the numbers
    mentions_98 = "9.8" in text
    mentions_911 = "9.11" in text
    
    # Check for explanation
    has_explanation = any(word in text_lower for word in [
        "because", "since", "as", "explanation", "reason"
    ])
    
    # Check for nonsense patterns
    is_nonsense = any(pattern in text_lower for pattern in [
        "not provided", "not comparable", "answer is r2", 
        "30.5", "30.00", "they are not"
    ]) or len(text) < 10
    
    # Score based on criteria
    if is_nonsense:
        return 1
    elif has_comparison and mentions_98 and mentions_911:
        if has_explanation:
            return 5
        else:
            return 4
    elif has_comparison and (mentions_98 or mentions_911):
        return 3
    elif mentions_98 or mentions_911:
        return 2
    else:
        return 1

def run_sweep_test(prompt_messages, ablation_value, num_iterations=10):
    """Run test with specific ablation value."""
    global NEURONS_TO_ABLATE, ABLATION_VALUE
    
    # Set ablation parameters
    ABLATION_VALUE = ablation_value
    NEURONS_TO_ABLATE = set(unique_hijackers)
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize once
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Run iterations
    bug_count = 0
    coherence_scores = []
    example_outputs = []
    
    for i in range(num_iterations):
        # Register hooks
        hooks = []
        for idx, layer in enumerate(model.model.layers):
            hook = layer.mlp.act_fn.register_forward_hook(
                lambda module, input, output, idx=idx: ablation_hook(module, input, output, idx)
            )
            hooks.append(hook)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.2,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True
            )
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        
        # Decode
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract generated part
        if "assistant" in full_text:
            generated_text = full_text.split("assistant")[-1].strip()
        else:
            generated_text = full_text
        
        # Save first few examples
        if i < 3:
            example_outputs.append(generated_text)
        
        # Check for bug
        generated_lower = generated_text.lower()
        if "9.11" in generated_text and "is bigger" in generated_lower:
            pos_911 = generated_text.find("9.11")
            pos_bigger = generated_lower.find("is bigger")
            if pos_911 < pos_bigger and pos_bigger - pos_911 < 20:
                bug_count += 1
        
        # Score coherence
        coherence = score_coherence(generated_text)
        coherence_scores.append(coherence)
    
    bug_rate = (bug_count / num_iterations) * 100
    avg_coherence = sum(coherence_scores) / len(coherence_scores)
    
    return {
        "ablation_value": ablation_value,
        "bug_rate": bug_rate,
        "avg_coherence": avg_coherence,
        "coherence_scores": coherence_scores,
        "example_outputs": example_outputs
    }

# Test prompt
test_prompt = [{"role": "user", "content": "Which is bigger: 9.8 or 9.11?"}]

# Parameter sweep
ablation_values = [0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0]
results = []

print("\n" + "="*70)
print("PARAMETER SWEEP")
print("="*70)

for ablation_val in tqdm(ablation_values, desc="Testing ablation values"):
    print(f"\n\nTesting ablation value: {ablation_val}")
    result = run_sweep_test(test_prompt, ablation_val, num_iterations=10)
    results.append(result)
    
    print(f"  Bug rate: {result['bug_rate']:.1f}%")
    print(f"  Avg coherence: {result['avg_coherence']:.2f}/5")
    print(f"  Example: {result['example_outputs'][0][:60]}...")

# Find the sweet spot
print("\n" + "="*70)
print("RESULTS ANALYSIS")
print("="*70)

# Display results table
print("\n{:<12} {:<12} {:<15}".format("Ablation", "Bug Rate %", "Coherence /5"))
print("-" * 40)
for result in results:
    print("{:<12} {:<12.1f} {:<15.2f}".format(
        result['ablation_value'], 
        result['bug_rate'], 
        result['avg_coherence']
    ))

# Find optimal value
# Look for lowest bug rate with highest coherence
candidates = []
for result in results:
    if result['bug_rate'] < 20:  # Less than 20% bug rate
        candidates.append(result)

if candidates:
    # Sort by coherence score
    best = max(candidates, key=lambda x: x['avg_coherence'])
    print(f"\n🎯 SWEET SPOT FOUND!")
    print(f"   Ablation value: {best['ablation_value']}")
    print(f"   Bug rate: {best['bug_rate']:.1f}%")
    print(f"   Coherence: {best['avg_coherence']:.2f}/5")
    print(f"\n   Example outputs:")
    for i, ex in enumerate(best['example_outputs']):
        print(f"   {i+1}. {ex}")
else:
    print("\n⚠️ No sweet spot found - all low bug rates have poor coherence")

# Save results
with open("parameter_sweep_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Plot results
plt.figure(figsize=(10, 6))
ablation_vals = [r['ablation_value'] for r in results]
bug_rates = [r['bug_rate'] for r in results]
coherences = [r['avg_coherence'] for r in results]

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot bug rate
color = 'tab:red'
ax1.set_xlabel('Ablation Value')
ax1.set_ylabel('Bug Rate %', color=color)
ax1.plot(ablation_vals, bug_rates, 'o-', color=color, linewidth=2, markersize=8)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(-5, 105)

# Plot coherence on second y-axis
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Coherence Score (1-5)', color=color)
ax2.plot(ablation_vals, coherences, 's-', color=color, linewidth=2, markersize=8)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0.5, 5.5)

# Add grid and title
ax1.grid(True, alpha=0.3)
plt.title('Ablation Parameter Sweep: Bug Rate vs Coherence', fontsize=14, fontweight='bold')

# Highlight sweet spot if found
if candidates:
    sweet_spot = best['ablation_value']
    ax1.axvline(x=sweet_spot, color='green', linestyle='--', alpha=0.7, linewidth=2)
    ax1.text(sweet_spot + 0.1, 50, f'Sweet Spot\n({sweet_spot})', 
             color='green', fontweight='bold')

plt.tight_layout()
plt.savefig('ablation_sweep_plot.png', dpi=150)
print(f"\n📊 Plot saved to: ablation_sweep_plot.png")

print("\n✅ Parameter sweep complete!")
print(f"Results saved to: parameter_sweep_results.json")