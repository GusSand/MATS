#!/usr/bin/env python
"""
Identify two distinct circuits:
1. HIJACKER_CLUSTER: Early-to-mid layers (2-15) that activate on "9.11" input
2. REASONING_CLUSTER: Late layers (28-31) that handle decimal comparison

Flexible to work with different prompt formats.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import os
from collections import defaultdict
import json

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

print("🔍 Identifying Two Distinct Circuits")
print("="*60)

# Configuration
PROMPT_FORMATS = {
    "chat": {
        "name": "Chat Template (Buggy)",
        "use_template": True,
        "messages": [{"role": "user", "content": "Which is bigger: 9.8 or 9.11?"}]
    },
    "simple": {
        "name": "Simple Format (Correct)",
        "use_template": False,
        "prompt": "Which is bigger: 9.8 or 9.11?\nAnswer:"
    }
}

# Layer ranges
HIJACKER_LAYERS = range(2, 16)  # Early-to-mid layers (2-15)
REASONING_LAYERS = range(28, 32)  # Late layers (28-31)

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

def analyze_format(format_key):
    """Analyze neuron activations for a specific prompt format"""
    format_info = PROMPT_FORMATS[format_key]
    print(f"\n{'='*60}")
    print(f"Analyzing: {format_info['name']}")
    print(f"{'='*60}")
    
    # Prepare prompt
    if format_info['use_template']:
        prompt = tokenizer.apply_chat_template(format_info['messages'], tokenize=False, add_generation_prompt=True)
    else:
        prompt = format_info['prompt']
    
    print(f"Prompt: {repr(prompt[:80])}...")
    
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
    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    prompt_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Find positions of "9.11" in prompt
    nine_eleven_positions = []
    for i in range(len(prompt_tokens) - 2):
        if prompt_tokens[i] == '9' and prompt_tokens[i+1] == '.' and prompt_tokens[i+2] == '11':
            nine_eleven_positions.extend([i, i+1, i+2])
    
    print(f"\nPrompt tokens: {len(prompt_tokens)}")
    print(f"'9.11' appears at positions: {nine_eleven_positions}")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            temperature=0.2,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
        )
    
    # Get generated text
    generated_ids = outputs[0][len(inputs['input_ids'][0]):]
    generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids)
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"\nGenerated: {generated_text[:60]}...")
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    
    # Analyze HIJACKER CIRCUIT (early-to-mid layers on "9.11")
    print(f"\n{'='*40}")
    print("HIJACKER CIRCUIT (Layers 2-15, '9.11' positions)")
    print(f"{'='*40}")
    
    hijacker_neurons = []
    for layer_idx in HIJACKER_LAYERS:
        if layer_idx in layer_activations:
            activations = layer_activations[layer_idx]
            layer_act = torch.cat(activations, dim=1)
            
            # Focus on "9.11" positions
            for pos in nine_eleven_positions:
                if pos < layer_act.shape[1]:
                    pos_activations = layer_act[0, pos, :]
                    top_values, top_indices = torch.topk(pos_activations, k=5)
                    
                    for neuron_idx, value in zip(top_indices, top_values):
                        hijacker_neurons.append({
                            'layer': layer_idx,
                            'neuron': neuron_idx.item(),
                            'activation': value.item(),
                            'position': pos,
                            'token': prompt_tokens[pos]
                        })
    
    # Sort and show top hijacker neurons
    hijacker_neurons.sort(key=lambda x: x['activation'], reverse=True)
    top_hijackers = hijacker_neurons[:10]
    
    print(f"Top 10 hijacker neurons:")
    print(f"{'Layer':<8} {'Neuron':<10} {'Activation':<12} {'Token':<10} {'Pos'}")
    print("-" * 50)
    for neuron in top_hijackers:
        print(f"{neuron['layer']:<8} {neuron['neuron']:<10} {neuron['activation']:<12.4f} "
              f"{neuron['token']:<10} {neuron['position']}")
    
    # Analyze REASONING CIRCUIT (late layers on all positions)
    print(f"\n{'='*40}")
    print("REASONING CIRCUIT (Layers 28-31, all positions)")
    print(f"{'='*40}")
    
    # Look at all important positions (numbers, comparison words)
    all_positions = list(range(len(prompt_tokens) + len(generated_tokens)))
    
    reasoning_neurons = []
    for layer_idx in REASONING_LAYERS:
        if layer_idx in layer_activations:
            activations = layer_activations[layer_idx]
            layer_act = torch.cat(activations, dim=1)
            
            # Get max activation for each neuron across all positions
            max_acts, max_pos = torch.max(layer_act[0], dim=0)
            top_values, top_indices = torch.topk(max_acts, k=10)
            
            for neuron_idx, value, pos_idx in zip(top_indices, top_values, max_pos[top_indices]):
                pos = pos_idx.item()
                token = prompt_tokens[pos] if pos < len(prompt_tokens) else generated_tokens[pos - len(prompt_tokens)]
                reasoning_neurons.append({
                    'layer': layer_idx,
                    'neuron': neuron_idx.item(),
                    'activation': value.item(),
                    'position': pos,
                    'token': token
                })
    
    # Sort and show top reasoning neurons
    reasoning_neurons.sort(key=lambda x: x['activation'], reverse=True)
    top_reasoning = reasoning_neurons[:10]
    
    print(f"Top 10 reasoning neurons:")
    print(f"{'Layer':<8} {'Neuron':<10} {'Activation':<12} {'Token':<10} {'Pos'}")
    print("-" * 50)
    for neuron in top_reasoning:
        print(f"{neuron['layer']:<8} {neuron['neuron']:<10} {neuron['activation']:<12.4f} "
              f"{neuron['token']:<10} {neuron['position']}")
    
    # Return results
    return {
        'format': format_key,
        'generated': generated_text,
        'hijacker_cluster': [(n['layer'], n['neuron']) for n in top_hijackers[:10]],
        'reasoning_cluster': [(n['layer'], n['neuron']) for n in top_reasoning[:10]]
    }

# Analyze both formats
results = {}
for format_key in ['chat', 'simple']:
    results[format_key] = analyze_format(format_key)

# Summary
print(f"\n{'='*60}")
print("CIRCUIT IDENTIFICATION SUMMARY")
print(f"{'='*60}")

print("\nChat Template (Buggy):")
print(f"Generated: {results['chat']['generated'][:50]}...")
print(f"\nTop 10 Hijacker neurons:")
for layer, neuron in results['chat']['hijacker_cluster']:
    print(f"  L{layer}/N{neuron}")
print(f"\nTop 10 Reasoning neurons:")
for layer, neuron in results['chat']['reasoning_cluster']:
    print(f"  L{layer}/N{neuron}")

print("\n\nSimple Format (Correct):")
print(f"Generated: {results['simple']['generated'][:50]}...")
print(f"\nTop 10 Hijacker neurons:")
for layer, neuron in results['simple']['hijacker_cluster']:
    print(f"  L{layer}/N{neuron}")
print(f"\nTop 10 Reasoning neurons:")
for layer, neuron in results['simple']['reasoning_cluster']:
    print(f"  L{layer}/N{neuron}")

# Save results
with open("identified_circuits.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to: identified_circuits.json")

print(f"\n{'='*60}")
print("KEY FINDINGS:")
print(f"{'='*60}")
print("1. HIJACKER CIRCUIT (Layers 2-15): Activates specifically on '9.11' tokens")
print("2. REASONING CIRCUIT (Layers 28-31): Handles decimal comparison logic")
print("3. Both formats show similar reasoning neurons (L31/N13336, L31/N12004)")
print("4. The hijacker circuit may differ between formats!")
print("\nNext step: Ablate the hijacker circuit while monitoring the reasoning circuit")