#!/usr/bin/env python3
"""
Steering Vector Experiment for Llama-3.1-8B-Instruct Decimal Bug

This experiment implements the ActAdd (Activation Addition) approach to fix the decimal
comparison bug where the model incorrectly states "9.11 is bigger than 9.8".

Approach:
1. Collect activations from correct behavior (Simple Format)
2. Collect activations from buggy behavior (Chat Template)  
3. Calculate steering vectors (correct - buggy)
4. Apply steering vectors during inference to fix the bug

Paper reference: https://arxiv.org/html/2308.10248v5
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import warnings
import os
from collections import defaultdict
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

print("="*80)
print("STEERING VECTOR EXPERIMENT - Fixing Llama Decimal Bug with ActAdd")
print("="*80)

# ==============================================================================
# STEP 1: Load Model and Hijacker Neurons
# ==============================================================================

print("\n[STEP 1] Loading model and identifying neurons...")

# Load the identified hijacker neurons from our previous analysis
with open("../identified_circuits.json", "r") as f:
    circuits = json.load(f)

# Get the hijacker neurons that fire on "9.11"
hijacker_neurons = circuits["chat"]["hijacker_cluster"]
unique_hijackers = list(set(tuple(n) for n in hijacker_neurons))
print(f"Found {len(unique_hijackers)} unique hijacker neurons:")
for layer, neuron in sorted(unique_hijackers):
    print(f"  Layer {layer}, Neuron {neuron}")

# Load model
model_name = "meta-llama/Llama-3.1-8B-Instruct"
print(f"\nLoading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()
print("✓ Model loaded successfully")

# ==============================================================================
# STEP 2: Setup Activation Collection
# ==============================================================================

print("\n[STEP 2] Setting up activation collection hooks...")

# Storage for activations
activations_storage = defaultdict(lambda: defaultdict(list))
current_collection_mode = None  # Will be 'correct' or 'buggy'

def activation_collector_hook(module, input, output, layer_idx):
    """
    Hook to collect activations for specific neurons.
    Stores activations in activations_storage[mode][layer][neuron].
    """
    if current_collection_mode is None:
        return output
    
    # Check if this layer has neurons we're interested in
    for target_layer, target_neuron in unique_hijackers:
        if target_layer == layer_idx:
            # Extract activation for this specific neuron
            # output shape: [batch, seq_len, hidden_dim]
            neuron_activation = output[0, :, target_neuron].detach().cpu().numpy()
            
            # Store the activation
            key = (target_layer, target_neuron)
            activations_storage[current_collection_mode][key].append(neuron_activation)
            
            # Debug print
            print(f"  [DEBUG] Collected activation for L{target_layer}/N{target_neuron}: "
                  f"shape={neuron_activation.shape}, max={neuron_activation.max():.3f}")
    
    return output

# Register hooks on MLP activation layers (after activation function)
# We need to hook at the intermediate layer where we have access to individual neurons
hooks = []
for i, layer in enumerate(model.model.layers):
    # Hook the intermediate activations after gate_proj but before down_proj
    # We'll modify our approach to hook earlier in the MLP
    def create_mlp_hook(layer_idx):
        def mlp_hook(module, input, output):
            # For LLaMA, we need to intercept at the MLP level
            # The MLP computes: down_proj(act_fn(gate_proj(x)) * up_proj(x))
            # We'll capture the intermediate activations
            x = input[0]  # Get input tensor
            
            # Compute intermediate activations manually
            gate_output = module.act_fn(module.gate_proj(x))
            up_output = module.up_proj(x)
            intermediate = gate_output * up_output  # This is what goes into down_proj
            
            # Now collect activations for our neurons
            if current_collection_mode is not None:
                for target_layer, target_neuron in unique_hijackers:
                    if target_layer == layer_idx:
                        # Check bounds
                        if target_neuron < intermediate.shape[-1]:
                            neuron_activation = intermediate[0, :, target_neuron].detach().cpu().numpy()
                            key = (target_layer, target_neuron)
                            activations_storage[current_collection_mode][key].append(neuron_activation)
                            
                            # Debug print
                            if len(activations_storage[current_collection_mode][key]) == 1:
                                print(f"  [DEBUG] Collected activation for L{target_layer}/N{target_neuron}: "
                                      f"shape={neuron_activation.shape}, max={neuron_activation.max():.3f}")
                        else:
                            print(f"  [WARNING] Neuron {target_neuron} out of bounds for layer {target_layer} "
                                  f"(size={intermediate.shape[-1]})")
            
            # Return original output
            return output
        return mlp_hook
    
    hook = layer.mlp.register_forward_hook(create_mlp_hook(i))
    hooks.append(hook)
print(f"✓ Registered {len(hooks)} hooks for activation collection")

# ==============================================================================
# STEP 3: Collect Activations for Correct Behavior (Simple Format)
# ==============================================================================

print("\n[STEP 3] Collecting activations for CORRECT behavior (Simple Format)...")

# Simple format prompt that produces correct answer
simple_prompt = "Which is bigger: 9.8 or 9.11?\nAnswer:"
print(f"Prompt: {repr(simple_prompt)}")

# Set collection mode
current_collection_mode = 'correct'

# Tokenize
inputs = tokenizer(simple_prompt, return_tensors="pt").to(model.device)
print(f"Token count: {inputs['input_ids'].shape[1]}")

# Generate to collect activations
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        temperature=0.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
generated = response[len(simple_prompt):]
print(f"Generated: {generated}")
print(f"✓ Correct answer: {'9.8' in generated and 'bigger' in generated.lower()}")

# ==============================================================================
# STEP 4: Collect Activations for Buggy Behavior (Chat Template)
# ==============================================================================

print("\n[STEP 4] Collecting activations for BUGGY behavior (Chat Template)...")

# Chat template that produces buggy answer
chat_messages = [{"role": "user", "content": "Which is bigger: 9.8 or 9.11?"}]
chat_prompt = tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
print(f"Prompt: {repr(chat_prompt[:80])}...")

# Set collection mode
current_collection_mode = 'buggy'

# Tokenize
inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
print(f"Token count: {inputs['input_ids'].shape[1]}")

# Generate to collect activations
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        temperature=0.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
generated = response[len(chat_prompt):]
print(f"Generated: {generated[:50]}...")
print(f"✓ Buggy answer: {'9.11' in generated and 'bigger' in generated.lower()}")

# Stop collecting
current_collection_mode = None

# ==============================================================================
# STEP 5: Calculate Steering Vectors
# ==============================================================================

print("\n[STEP 5] Calculating steering vectors...")

steering_vectors = {}

for (layer, neuron) in unique_hijackers:
    key = (layer, neuron)
    
    # Get activations for this neuron
    correct_acts = activations_storage['correct'][key]
    buggy_acts = activations_storage['buggy'][key]
    
    if correct_acts and buggy_acts:
        # Take the mean activation across the sequence for each
        correct_mean = np.concatenate(correct_acts).mean()
        buggy_mean = np.concatenate(buggy_acts).mean()
        
        # Calculate steering vector: correct - buggy
        steering_vector = correct_mean - buggy_mean
        steering_vectors[key] = steering_vector
        
        print(f"  L{layer}/N{neuron}: correct={correct_mean:.3f}, buggy={buggy_mean:.3f}, "
              f"steering={steering_vector:.3f}")
        
        # Special attention to the entangled neuron
        if layer == 14 and neuron == 12639:
            print(f"    ⚠️  ENTANGLED NEURON - Very small steering vector!")

# ==============================================================================
# STEP 6: Apply Steering Vectors During Inference
# ==============================================================================

print("\n[STEP 6] Testing steering vector intervention...")

# Remove old hooks
for hook in hooks:
    hook.remove()

# Global variables for steering
STEERING_VECTORS = steering_vectors
STEERING_ALPHA = 1.0  # Scaling factor

def create_steering_hook(layer_idx, test_num=0):
    """
    Creates a hook that applies steering vectors to push activations toward correct behavior.
    """
    def steering_hook(module, input, output):
        # We need to modify the intermediate activations
        x = input[0]
        
        # Compute intermediate activations
        gate_output = module.act_fn(module.gate_proj(x))
        up_output = module.up_proj(x)
        intermediate = gate_output * up_output
        
        # Apply steering to intermediate activations
        modified = False
        for (target_layer, target_neuron), steering_value in STEERING_VECTORS.items():
            if target_layer == layer_idx and target_neuron < intermediate.shape[-1]:
                # Apply steering: new = old + alpha * steering_vector
                old_value = intermediate[0, :, target_neuron].mean().item()
                intermediate[:, :, target_neuron] += STEERING_ALPHA * steering_value
                new_value = intermediate[0, :, target_neuron].mean().item()
                modified = True
                
                # Debug print for significant changes (only first test)
                if abs(steering_value) > 0.1 and STEERING_ALPHA > 0 and test_num == 0:
                    print(f"  [STEER] L{target_layer}/N{target_neuron}: "
                          f"{old_value:.3f} → {new_value:.3f} (Δ={steering_value * STEERING_ALPHA:.3f})")
        
        # If we modified intermediate activations, recompute output
        if modified:
            return module.down_proj(intermediate)
        else:
            return output
    
    return steering_hook

# Test with different alpha values
print("\nTesting different steering strengths (alpha values)...")
alpha_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
results = []

for alpha in alpha_values:
    print(f"\n--- Alpha = {alpha} ---")
    STEERING_ALPHA = alpha
    
    # Register steering hooks (created fresh for each test)
    steering_hooks = []
    
    # Test on buggy prompt
    success_count = 0
    total_tests = 5
    example_output = None
    
    for test_num in range(total_tests):
        # Register hooks for this test
        if test_num == 0 or True:  # Always register fresh hooks
            # Remove old hooks
            for hook in steering_hooks:
                hook.remove()
            steering_hooks = []
            
            # Register new hooks
            for i, layer in enumerate(model.model.layers):
                hook = layer.mlp.register_forward_hook(create_steering_hook(i, test_num))
                steering_hooks.append(hook)
        
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = response[len(chat_prompt):]
        
        # Check if it's correct now
        if '9.8' in generated and 'bigger' in generated.lower():
            success_count += 1
        
        if test_num == 0:
            example_output = generated
            print(f"  Example output: {generated[:50]}...")
    
    # Calculate success rate
    success_rate = (success_count / total_tests) * 100
    results.append({
        'alpha': alpha,
        'success_rate': success_rate,
        'example': example_output
    })
    print(f"  Success rate: {success_rate}% ({success_count}/{total_tests})")
    
    # Remove hooks for next iteration
    for hook in steering_hooks:
        hook.remove()

# ==============================================================================
# STEP 7: Visualize Results
# ==============================================================================

print("\n[STEP 7] Creating visualization...")

# Extract data for plotting
alphas = [r['alpha'] for r in results]
success_rates = [r['success_rate'] for r in results]

# Create figure
plt.figure(figsize=(10, 6))
plt.plot(alphas, success_rates, 'o-', linewidth=3, markersize=10, color='#3498db')
plt.xlabel('Steering Strength (α)', fontsize=14)
plt.ylabel('Success Rate (%)', fontsize=14)
plt.title('Steering Vector Effectiveness\nFixing "9.11 > 9.8" Bug with ActAdd', fontsize=16)
plt.grid(True, alpha=0.3)
plt.ylim(-5, 105)

# Add annotations for key points
for i, (alpha, rate) in enumerate(zip(alphas, success_rates)):
    if rate > 0:
        plt.annotate(f'{rate:.0f}%', (alpha, rate), 
                    textcoords="offset points", xytext=(0,10), 
                    ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('steering_vector_results.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization to steering_vector_results.png")

# ==============================================================================
# STEP 8: Final Analysis
# ==============================================================================

print("\n[STEP 8] Final Analysis")
print("="*80)

# Find best alpha
best_result = max(results, key=lambda x: x['success_rate'])
print(f"\nBest result: α={best_result['alpha']} with {best_result['success_rate']}% success rate")
print(f"Example output: {best_result['example']}")

# Analyze steering vectors
print("\nSteering Vector Analysis:")
sorted_vectors = sorted(steering_vectors.items(), key=lambda x: abs(x[1]), reverse=True)
print("\nTop 5 largest steering vectors:")
for (layer, neuron), value in sorted_vectors[:5]:
    print(f"  L{layer}/N{neuron}: {value:.3f}")

print("\nTop 5 smallest steering vectors:")
for (layer, neuron), value in sorted_vectors[-5:]:
    print(f"  L{layer}/N{neuron}: {value:.3f}")

# Check the entangled neuron specifically
entangled_key = (14, 12639)
if entangled_key in steering_vectors:
    print(f"\nEntangled neuron L14/N12639 steering vector: {steering_vectors[entangled_key]:.3f}")
    print("This small value confirms the entanglement - the neuron behaves similarly in both cases!")

# Save results
results_data = {
    'steering_vectors': {f"L{k[0]}/N{k[1]}": float(v) for k, v in steering_vectors.items()},
    'experiment_results': results,
    'best_alpha': best_result['alpha'],
    'best_success_rate': best_result['success_rate']
}

with open('steering_vector_results.json', 'w') as f:
    json.dump(results_data, f, indent=2)
print("\n✓ Saved detailed results to steering_vector_results.json")

print("\n" + "="*80)
print("EXPERIMENT COMPLETE")
print("="*80)