#!/usr/bin/env python
"""
Record the top 5-10 neurons that activate most strongly on '9.11 is greater than 9.8' tokens
using chat template format (which has the bug).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import os
import numpy as np
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

print("🔍 Recording Top Neurons for '9.11 is greater than 9.8' Bug")
print("="*60)

# Load model
model_path = "meta-llama/Llama-3.1-8B-Instruct"
print(f"Loading {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# Storage for activations
layer_activations = defaultdict(list)
token_positions = []

def capture_activations_hook(module, input, output, layer_idx):
    """Hook to capture MLP activations"""
    if hasattr(output, 'detach'):
        # Store the activation values
        activation = output.detach().cpu().float()
        layer_activations[layer_idx].append(activation)

# Register hooks on all MLP layers
hooks = []
for idx, layer in enumerate(model.model.layers):
    # Hook the MLP activation (after activation function)
    hook = layer.mlp.act_fn.register_forward_hook(
        lambda module, input, output, idx=idx: capture_activations_hook(module, input, output, idx)
    )
    hooks.append(hook)

# Use chat template format (which has the bug)
messages = [{"role": "user", "content": "Which is bigger: 9.8 or 9.11?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print(f"\nPrompt: {repr(prompt[:80])}...")

# Tokenize to see the tokens
inputs = tokenizer(prompt, return_tensors="pt")
if torch.cuda.is_available():
    inputs = {k: v.cuda() for k, v in inputs.items()}

prompt_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
print(f"\nPrompt tokens ({len(prompt_tokens)}): {prompt_tokens}")

# Generate with the model
print("\nGenerating response...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.0,  # Deterministic
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True
    )

# Get the generated tokens
generated_ids = outputs.sequences[0][len(inputs['input_ids'][0]):]
generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids)
generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

print(f"\nGenerated text: {generated_text[:100]}...")
print(f"\nGenerated tokens ({len(generated_tokens)}): {generated_tokens}")

# Find the positions of '9.11', 'is', 'greater', 'than', '9.8' tokens
target_patterns = [
    ['9', '.', '11'],
    ['greater'],
    ['than'],
    ['9', '.', '8']
]

# Find where these patterns appear in the generated tokens
important_positions = []
for i, token in enumerate(generated_tokens):
    # Check for '9.11'
    if i + 2 < len(generated_tokens) and generated_tokens[i:i+3] == ['9', '.', '11']:
        important_positions.extend([len(prompt_tokens) + i, len(prompt_tokens) + i + 1, len(prompt_tokens) + i + 2])
    # Check for 'greater'
    elif 'greater' in token.lower():
        important_positions.append(len(prompt_tokens) + i)
    # Check for 'than'
    elif 'than' in token.lower():
        important_positions.append(len(prompt_tokens) + i)
    # Check for '9.8'
    elif i + 2 < len(generated_tokens) and generated_tokens[i:i+3] == ['9', '.', '8']:
        important_positions.extend([len(prompt_tokens) + i, len(prompt_tokens) + i + 1, len(prompt_tokens) + i + 2])

print(f"\nImportant token positions: {important_positions}")

# Analyze activations at these positions
print("\n" + "="*60)
print("TOP ACTIVATING NEURONS ON '9.11 is greater than 9.8' TOKENS:")
print("="*60)

# Collect all activations for the important positions
all_top_neurons = []

for layer_idx in sorted(layer_activations.keys()):
    activations = layer_activations[layer_idx]
    
    # Concatenate all forward passes for this layer
    layer_act = torch.cat(activations, dim=1)  # Shape: [batch, total_tokens, hidden_dim]
    
    for pos in important_positions:
        if pos < layer_act.shape[1]:
            # Get activations at this position
            pos_activations = layer_act[0, pos, :]  # Shape: [hidden_dim]
            
            # Find top 10 neurons
            top_values, top_indices = torch.topk(pos_activations, k=min(10, pos_activations.shape[0]))
            
            for rank, (neuron_idx, value) in enumerate(zip(top_indices, top_values)):
                all_top_neurons.append({
                    'layer': layer_idx,
                    'neuron': neuron_idx.item(),
                    'activation': value.item(),
                    'position': pos,
                    'token': generated_tokens[pos - len(prompt_tokens)] if pos >= len(prompt_tokens) else prompt_tokens[pos]
                })

# Sort by activation strength and get top 10 overall
all_top_neurons.sort(key=lambda x: x['activation'], reverse=True)
top_10_neurons = all_top_neurons[:10]

print("\nTop 10 neurons by activation strength:")
print(f"{'Rank':<6} {'Layer':<8} {'Neuron':<10} {'Activation':<12} {'Token':<15} {'Position'}")
print("-" * 70)

for rank, neuron_info in enumerate(top_10_neurons, 1):
    print(f"{rank:<6} {neuron_info['layer']:<8} {neuron_info['neuron']:<10} "
          f"{neuron_info['activation']:<12.4f} {neuron_info['token']:<15} {neuron_info['position']}")

# Also show top neurons per layer
print("\n" + "="*60)
print("TOP NEURONS PER LAYER:")
print("="*60)

# Group by layer
layer_grouped = defaultdict(list)
for neuron in all_top_neurons:
    layer_grouped[neuron['layer']].append(neuron)

# Show top 5 per layer for the most active layers
active_layers = sorted(layer_grouped.keys(), 
                      key=lambda l: max(n['activation'] for n in layer_grouped[l]), 
                      reverse=True)[:5]

for layer_idx in active_layers:
    print(f"\nLayer {layer_idx}:")
    layer_neurons = sorted(layer_grouped[layer_idx], key=lambda x: x['activation'], reverse=True)[:5]
    for neuron_info in layer_neurons:
        print(f"  Neuron {neuron_info['neuron']:<8} Activation: {neuron_info['activation']:<10.4f} "
              f"Token: {neuron_info['token']:<15} Pos: {neuron_info['position']}")

# Clean up hooks
for hook in hooks:
    hook.remove()

print("\n" + "="*60)
print("✅ Analysis complete!")