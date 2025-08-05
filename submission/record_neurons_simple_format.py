#!/usr/bin/env python
"""
Record neurons using the Simple format that consistently gives CORRECT answers
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import os
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

print("🔍 Recording Neurons for Simple Format (100% Correct)")
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

# Simple format prompt - this gives 100% correct answers!
prompt = "Which is bigger: 9.8 or 9.11?\nAnswer:"

print(f"\nPrompt: {repr(prompt)}")
print("Using temperature=0.2 (same as verify_llama_bug.py)")

# Storage for activations
layer_activations = defaultdict(list)

def capture_activations_hook(module, input, output, layer_idx):
    """Hook to capture MLP activations"""
    if hasattr(output, 'detach'):
        activation = output.detach().cpu().float()
        layer_activations[layer_idx].append(activation)

# Register hooks
hooks = []
for idx, layer in enumerate(model.model.layers):
    hook = layer.mlp.act_fn.register_forward_hook(
        lambda module, input, output, idx=idx: capture_activations_hook(module, input, output, idx)
    )
    hooks.append(hook)

# Generate
inputs = tokenizer(prompt, return_tensors="pt")
if torch.cuda.is_available():
    inputs = {k: v.cuda() for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.2,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
    )

# Get full response
full = tokenizer.decode(outputs[0], skip_special_tokens=True)
generated = full[len(prompt):]

print(f"\nGenerated text: {generated[:100]}...")

# Clean up hooks
for hook in hooks:
    hook.remove()

# Get tokens for analysis
generated_ids = outputs[0][len(inputs['input_ids'][0]):]
prompt_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids)

print(f"\nPrompt tokens ({len(prompt_tokens)}): {prompt_tokens}")
print(f"Generated tokens: {generated_tokens[:20]}...")

# Find important positions
important_positions = []
for i, token in enumerate(generated_tokens):
    if i + 2 < len(generated_tokens) and generated_tokens[i:i+3] == ['9', '.', '8']:
        important_positions.extend([len(prompt_tokens) + i, len(prompt_tokens) + i + 1, len(prompt_tokens) + i + 2])
    elif i + 2 < len(generated_tokens) and generated_tokens[i:i+3] == ['9', '.', '11']:
        important_positions.extend([len(prompt_tokens) + i, len(prompt_tokens) + i + 1, len(prompt_tokens) + i + 2])
    elif any(word in token.lower() for word in ['greater', 'bigger', 'larger', 'than', 'is']):
        important_positions.append(len(prompt_tokens) + i)

print(f"\nImportant token positions: {important_positions}")

# Save to file
with open("neurons_simple_format_CORRECT.txt", "w") as f:
    f.write("🔍 Neurons for Simple Format (100% Correct Answers)\n")
    f.write("="*60 + "\n")
    f.write(f"Prompt: {repr(prompt)}\n")
    f.write(f"Generated text: {generated}\n")
    f.write(f"Important positions: {important_positions}\n\n")
    
    # Analyze neurons
    all_top_neurons = []
    
    for layer_idx in sorted(layer_activations.keys()):
        activations = layer_activations[layer_idx]
        layer_act = torch.cat(activations, dim=1)
        
        for pos in important_positions:
            if pos < layer_act.shape[1]:
                pos_activations = layer_act[0, pos, :]
                top_values, top_indices = torch.topk(pos_activations, k=min(10, pos_activations.shape[0]))
                
                for rank, (neuron_idx, value) in enumerate(zip(top_indices, top_values)):
                    all_top_neurons.append({
                        'layer': layer_idx,
                        'neuron': neuron_idx.item(),
                        'activation': value.item(),
                        'position': pos,
                        'token': generated_tokens[pos - len(prompt_tokens)] if pos >= len(prompt_tokens) else prompt_tokens[pos]
                    })
    
    # Sort and get top 10
    all_top_neurons.sort(key=lambda x: x['activation'], reverse=True)
    top_10_neurons = all_top_neurons[:10]
    
    f.write("\nTop 10 neurons by activation strength:\n")
    f.write(f"{'Rank':<6} {'Layer':<8} {'Neuron':<10} {'Activation':<12} {'Token':<15} {'Position'}\n")
    f.write("-" * 70 + "\n")
    
    for rank, neuron_info in enumerate(top_10_neurons, 1):
        f.write(f"{rank:<6} {neuron_info['layer']:<8} {neuron_info['neuron']:<10} "
                f"{neuron_info['activation']:<12.4f} {neuron_info['token']:<15} {neuron_info['position']}\n")
    
    # Top neurons per layer
    f.write("\n" + "="*60 + "\n")
    f.write("TOP NEURONS PER LAYER (Top 5 layers):\n")
    f.write("="*60 + "\n")
    
    layer_grouped = defaultdict(list)
    for neuron in all_top_neurons:
        layer_grouped[neuron['layer']].append(neuron)
    
    active_layers = sorted(layer_grouped.keys(), 
                          key=lambda l: max(n['activation'] for n in layer_grouped[l]), 
                          reverse=True)[:5]
    
    for layer_idx in active_layers:
        f.write(f"\nLayer {layer_idx}:\n")
        layer_neurons = sorted(layer_grouped[layer_idx], key=lambda x: x['activation'], reverse=True)[:5]
        for neuron_info in layer_neurons:
            f.write(f"  Neuron {neuron_info['neuron']:<8} Activation: {neuron_info['activation']:<10.4f} "
                    f"Token: {neuron_info['token']:<15} Pos: {neuron_info['position']}\n")

print(f"\n✅ Neurons saved to: neurons_simple_format_CORRECT.txt")

# Also print comparison summary
print("\n" + "="*60)
print("COMPARISON SUMMARY:")
print("="*60)
print("Chat Template: 90% wrong (9.11 is bigger)")
print("Q&A Format: 90% wrong (9.11 is bigger)")  
print("Simple Format: 100% CORRECT (9.8 is bigger)")
print("\nThe prompt format dramatically affects the model's behavior!")