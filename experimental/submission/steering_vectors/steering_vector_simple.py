#!/usr/bin/env python3
"""
Simplified Steering Vector Experiment - Cleaner Implementation
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("STEERING VECTOR EXPERIMENT - ActAdd for Decimal Bug")
print("="*70)

# Load model
print("\nLoading model...")
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# Load hijacker neurons
with open("../identified_circuits.json", "r") as f:
    circuits = json.load(f)
hijacker_neurons = [(n[0], n[1]) for n in circuits["chat"]["hijacker_cluster"]]
unique_neurons = list(set(hijacker_neurons))
print(f"Found {len(unique_neurons)} unique neurons to steer")

# Global storage
activations = {}
steering_vectors = {}
current_mode = None

def activation_hook(module, input, output):
    """Collect intermediate MLP activations"""
    if current_mode is None:
        return output
    
    x = input[0]
    # Get intermediate activations (before down_proj)
    intermediate = module.act_fn(module.gate_proj(x)) * module.up_proj(x)
    
    # Store mean activation for each neuron
    layer_idx = int(module.__module_name__.split('.')[2])
    for target_layer, target_neuron in unique_neurons:
        if target_layer == layer_idx and target_neuron < intermediate.shape[-1]:
            key = (target_layer, target_neuron)
            if key not in activations[current_mode]:
                activations[current_mode] = {}
            # Store mean activation across sequence
            activations[current_mode][key] = intermediate[0, :, target_neuron].mean().item()
    
    return output

# Register hooks
print("\nRegistering hooks...")
for i, layer in enumerate(model.model.layers):
    layer.mlp.__module_name__ = f'model.layers.{i}'
    layer.mlp.register_forward_hook(activation_hook)

# Collect activations for CORRECT behavior
print("\n[1] Collecting activations for CORRECT behavior (Simple Format)...")
current_mode = 'correct'
activations['correct'] = {}

prompt = "Which is bigger: 9.8 or 9.11?\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20, temperature=0.1, do_sample=True)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Response: {response[len(prompt):30]}...")

# Collect activations for BUGGY behavior
print("\n[2] Collecting activations for BUGGY behavior (Chat Template)...")
current_mode = 'buggy'
activations['buggy'] = {}

messages = [{"role": "user", "content": "Which is bigger: 9.8 or 9.11?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20, temperature=0.1, do_sample=True)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Response: {response[len(prompt):30]}...")

# Calculate steering vectors
print("\n[3] Calculating steering vectors...")
for neuron in unique_neurons:
    if neuron in activations['correct'] and neuron in activations['buggy']:
        correct_val = activations['correct'][neuron]
        buggy_val = activations['buggy'][neuron]
        steering_vectors[neuron] = correct_val - buggy_val
        if abs(steering_vectors[neuron]) > 0.05:
            print(f"  L{neuron[0]}/N{neuron[1]}: {steering_vectors[neuron]:.3f}")

# Test steering intervention
print("\n[4] Testing steering intervention...")
current_mode = None

def steering_hook(module, input, output):
    """Apply steering vectors"""
    x = input[0]
    intermediate = module.act_fn(module.gate_proj(x)) * module.up_proj(x)
    
    layer_idx = int(module.__module_name__.split('.')[2])
    modified = False
    
    for (target_layer, target_neuron), steer_val in steering_vectors.items():
        if target_layer == layer_idx and target_neuron < intermediate.shape[-1]:
            intermediate[:, :, target_neuron] += alpha * steer_val
            modified = True
    
    if modified:
        return module.down_proj(intermediate)
    return output

# Remove old hooks and register steering hooks
for layer in model.model.layers:
    layer.mlp._forward_hooks.clear()
    layer.mlp.register_forward_hook(steering_hook)

# Test different alpha values
results = []
for alpha in [0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
    print(f"\nAlpha = {alpha}")
    
    # Test 5 times
    successes = 0
    for _ in range(5):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=30, temperature=0.2, do_sample=True)
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Check if correct
        if '9.8' in response and ('bigger' in response.lower() or 'larger' in response.lower()):
            if '9.11' not in response or response.lower().index('9.8') < response.lower().index('9.11'):
                successes += 1
    
    success_rate = (successes / 5) * 100
    results.append({'alpha': alpha, 'success_rate': success_rate})
    print(f"  Success rate: {success_rate}%")
    
    if alpha == 2.0 and successes > 0:
        print(f"  Example: {response[:50]}...")

# Visualize results
print("\n[5] Creating visualization...")
alphas = [r['alpha'] for r in results]
rates = [r['success_rate'] for r in results]

plt.figure(figsize=(10, 6))
plt.plot(alphas, rates, 'o-', linewidth=3, markersize=10)
plt.xlabel('Steering Strength (α)', fontsize=14)
plt.ylabel('Success Rate (%)', fontsize=14)
plt.title('Steering Vector Effectiveness', fontsize=16)
plt.grid(True, alpha=0.3)
plt.ylim(-5, 105)
plt.savefig('steering_results_simple.png', dpi=300)

# Summary
best = max(results, key=lambda x: x['success_rate'])
print(f"\n[6] Best result: α={best['alpha']} with {best['success_rate']}% success")

# Save results
with open('steering_results_simple.json', 'w') as f:
    json.dump({
        'steering_vectors': {f"L{k[0]}/N{k[1]}": v for k, v in steering_vectors.items()},
        'results': results
    }, f, indent=2)

print("\n" + "="*70)
print("EXPERIMENT COMPLETE")
print("="*70)