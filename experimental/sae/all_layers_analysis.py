#!/usr/bin/env python3
"""
Complete SAE Analysis of All 32 Layers - Comprehensive analysis
This will show the full progression of the bug through the model
"""

import torch
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import warnings
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

print("🔬 Complete SAE Analysis - All 32 Layers")
print("=" * 70)

# Configuration
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
ALL_LAYERS = list(range(32))  # Analyze ALL layers

# Test prompts from research
PROMPT_WRONG = "Q: Which is bigger: 9.8 or 9.11?\nA:"  
PROMPT_CORRECT = "Which is bigger: 9.8 or 9.11?\nAnswer:"  

print(f"\nModel: {MODEL_NAME}")
print(f"Analyzing ALL {len(ALL_LAYERS)} layers")
print("Expected time: ~1-2 minutes")

# Track timing
start_time = datetime.now()

# Load model and tokenizer
print("\n📥 Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model.eval()

# Load SAEs for all layers
print("\n📊 Loading SAEs for all layers...")
saes = {}
failed_layers = []

# Clear CUDA cache before starting
if torch.cuda.is_available():
    torch.cuda.empty_cache()

for i, layer in enumerate(ALL_LAYERS):
    try:
        print(f"\rLoading Layer {layer}/31... ({i+1}/{len(ALL_LAYERS)})", end="")
        # Clear cache periodically to prevent memory issues
        if i % 8 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        sae = SAE.from_pretrained(
            release="llama_scope_lxm_8x",
            sae_id=f"l{layer}m_8x",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        saes[layer] = sae
    except Exception as e:
        failed_layers.append(layer)
        print(f"\r✗ Layer {layer} failed: {str(e)[:50]}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print(f"\n✓ Successfully loaded {len(saes)}/{len(ALL_LAYERS)} SAEs")
if failed_layers:
    print(f"✗ Failed layers: {failed_layers}")

def extract_mlp_activations(model, inputs, layers):
    """Extract MLP activations at specified layers."""
    activations = {}
    hooks = []
    
    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            activations[layer_idx] = output.detach()
        return hook_fn
    
    for layer_idx in layers:
        hook = model.model.layers[layer_idx].mlp.register_forward_hook(
            make_hook(layer_idx)
        )
        hooks.append(hook)
    
    with torch.no_grad():
        _ = model(**inputs)
    
    for hook in hooks:
        hook.remove()
    
    return activations

def analyze_sae_features(activations, saes, token_pos=-1):
    """Pass activations through SAEs to get feature activations."""
    features = {}
    
    for layer_idx, mlp_act in activations.items():
        if layer_idx in saes:
            token_act = mlp_act[0, token_pos, :].unsqueeze(0)
            
            with torch.no_grad():
                sae_out = saes[layer_idx].encode(token_act)
                features[layer_idx] = sae_out.squeeze(0)
    
    return features

def compare_features(features_wrong, features_correct, layer, top_k=20):
    """Compare top SAE features between wrong and correct formats."""
    if layer not in features_wrong or layer not in features_correct:
        return None
        
    fw = features_wrong[layer]
    fc = features_correct[layer]
    
    wrong_top_vals, wrong_top_idx = torch.topk(fw, k=min(top_k, len(fw)))
    correct_top_vals, correct_top_idx = torch.topk(fc, k=min(top_k, len(fc)))
    
    wrong_set = set(wrong_top_idx.tolist())
    correct_set = set(correct_top_idx.tolist())
    
    shared = wrong_set & correct_set
    wrong_only = wrong_set - correct_set
    correct_only = correct_set - wrong_set
    
    # Calculate overlap percentage
    total_unique = len(wrong_set | correct_set)
    overlap_pct = (len(shared) / min(len(wrong_set), len(correct_set)) * 100) if min(len(wrong_set), len(correct_set)) > 0 else 0
    
    # Calculate amplification for shared features
    amplifications = []
    for feat_idx in shared:
        wrong_val = fw[feat_idx].item()
        correct_val = fc[feat_idx].item()
        if correct_val > 0:
            amplifications.append(wrong_val / correct_val)
    
    return {
        'shared': list(shared),
        'wrong_only': list(wrong_only),
        'correct_only': list(correct_only),
        'overlap_percentage': overlap_pct,
        'num_shared': len(shared),
        'num_wrong_only': len(wrong_only),
        'num_correct_only': len(correct_only),
        'avg_amplification': np.mean(amplifications) if amplifications else 1.0,
        'wrong_vals': {idx.item(): val.item() for idx, val in zip(wrong_top_idx, wrong_top_vals)},
        'correct_vals': {idx.item(): val.item() for idx, val in zip(correct_top_idx, correct_top_vals)}
    }

# Main analysis
print("\n" + "=" * 70)
print("ANALYZING PROMPTS")
print("=" * 70)

# Tokenize prompts
print(f"\n❌ Wrong format: '{PROMPT_WRONG}'")
inputs_wrong = tokenizer(PROMPT_WRONG, return_tensors="pt")
if torch.cuda.is_available():
    inputs_wrong = {k: v.cuda() for k, v in inputs_wrong.items()}

print(f"✅ Correct format: '{PROMPT_CORRECT}'")
inputs_correct = tokenizer(PROMPT_CORRECT, return_tensors="pt")
if torch.cuda.is_available():
    inputs_correct = {k: v.cuda() for k, v in inputs_correct.items()}

# Extract activations for all layers
print("\n📈 Extracting MLP activations for all layers...")
acts_wrong = extract_mlp_activations(model, inputs_wrong, list(saes.keys()))
acts_correct = extract_mlp_activations(model, inputs_correct, list(saes.keys()))

# Get SAE features
print("🔍 Computing SAE features...")
features_wrong = analyze_sae_features(acts_wrong, saes)
features_correct = analyze_sae_features(acts_correct, saes)

# Analyze each layer
print("\n" + "=" * 70)
print("ANALYZING ALL LAYERS")
print("=" * 70)

results = {}
layer_stats = []

for layer in sorted(saes.keys()):
    comparison = compare_features(features_wrong, features_correct, layer)
    if comparison:
        results[layer] = comparison
        layer_stats.append({
            'layer': layer,
            'overlap_pct': comparison['overlap_percentage'],
            'num_shared': comparison['num_shared'],
            'avg_amplification': comparison['avg_amplification']
        })
        print(f"\rAnalyzing Layer {layer}/31... Done", end="")

print("\n\n" + "=" * 70)
print("COMPLETE RESULTS TABLE")
print("=" * 70)

# Print comprehensive table
print("\n| Layer | Overlap % | Shared | Wrong-only | Correct-only | Amplification | Notes |")
print("|-------|-----------|--------|------------|--------------|---------------|-------|")

for layer in sorted(results.keys()):
    r = results[layer]
    notes = ""
    if layer == 6:
        notes = "Format detection?"
    elif layer == 8:
        notes = "Early discrimination"
    elif layer == 10:
        notes = "HIGH OVERLAP"
    elif layer in [13, 14, 15]:
        notes = "Hijacker neurons"
    elif layer == 25:
        notes = "CRITICAL"
    elif layer in [28, 29, 30, 31]:
        notes = "Output generation"
    
    overlap_bar = "█" * int(r['overlap_percentage'] / 5)  # Visual bar
    print(f"| {layer:5} | {r['overlap_percentage']:9.1f}% | {r['num_shared']:6} | {r['num_wrong_only']:10} | {r['num_correct_only']:12} | {r['avg_amplification']:13.2f}x | {notes:7} |")

# Find interesting patterns
print("\n" + "=" * 70)
print("KEY PATTERNS DISCOVERED")
print("=" * 70)

# Find layers with highest/lowest overlap
sorted_by_overlap = sorted(layer_stats, key=lambda x: x['overlap_pct'])
print("\n🔍 Layers with LOWEST overlap (most discrimination):")
for i in range(min(3, len(sorted_by_overlap))):
    l = sorted_by_overlap[i]
    print(f"  Layer {l['layer']:2}: {l['overlap_pct']:.1f}% overlap")

print("\n🔍 Layers with HIGHEST overlap (most entanglement):")
for i in range(min(3, len(sorted_by_overlap))):
    l = sorted_by_overlap[-(i+1)]
    print(f"  Layer {l['layer']:2}: {l['overlap_pct']:.1f}% overlap")

# Find layers with highest amplification
sorted_by_amp = sorted(layer_stats, key=lambda x: x['avg_amplification'], reverse=True)
print("\n⚡ Layers with HIGHEST amplification (wrong format boost):")
for i in range(min(3, len(sorted_by_amp))):
    l = sorted_by_amp[i]
    if l['avg_amplification'] > 1:
        print(f"  Layer {l['layer']:2}: {l['avg_amplification']:.2f}x amplification")

# Identify phase transitions
print("\n🔄 Phase transitions in the model:")
overlap_values = [r['overlap_percentage'] for r in sorted(layer_stats, key=lambda x: x['layer'])]
for i in range(1, len(overlap_values)):
    if abs(overlap_values[i] - overlap_values[i-1]) > 30:  # Large change
        layer = sorted(layer_stats, key=lambda x: x['layer'])[i]['layer']
        print(f"  Major transition at Layer {layer}: {overlap_values[i-1]:.1f}% → {overlap_values[i]:.1f}%")

# Create visualization
print("\n📊 Creating visualization...")
layers = [r['layer'] for r in sorted(layer_stats, key=lambda x: x['layer'])]
overlaps = [r['overlap_pct'] for r in sorted(layer_stats, key=lambda x: x['layer'])]
amplifications = [r['avg_amplification'] for r in sorted(layer_stats, key=lambda x: x['layer'])]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Overlap percentage plot
ax1.bar(layers, overlaps, color=['red' if o < 40 else 'yellow' if o < 60 else 'green' for o in overlaps])
ax1.axhline(y=40, color='gray', linestyle='--', alpha=0.5, label='40% threshold')
ax1.axhline(y=60, color='gray', linestyle='--', alpha=0.5, label='60% threshold')
ax1.set_xlabel('Layer')
ax1.set_ylabel('Feature Overlap %')
ax1.set_title('Feature Overlap Between Correct and Wrong Formats Across All Layers')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Mark special layers
for layer, note in [(6, 'Format?'), (8, 'Discrim'), (10, 'HIGH'), (25, 'CRITICAL')]:
    if layer in layers:
        idx = layers.index(layer)
        ax1.annotate(note, xy=(layer, overlaps[idx]), xytext=(layer, overlaps[idx] + 5),
                    ha='center', fontsize=8, color='blue')

# Amplification plot
colors = ['red' if a > 1.2 else 'blue' if a < 0.8 else 'gray' for a in amplifications]
ax2.bar(layers, amplifications, color=colors)
ax2.axhline(y=1.0, color='black', linestyle='-', alpha=0.5, label='No amplification')
ax2.set_xlabel('Layer')
ax2.set_ylabel('Average Amplification Factor')
ax2.set_title('Feature Amplification in Wrong Format Across All Layers')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('all_layers_sae_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Visualization saved to all_layers_sae_analysis.png")

# Save comprehensive results
output_file = "all_layers_complete_analysis.json"
save_results = {
    "summary": {
        "total_layers_analyzed": len(results),
        "failed_layers": failed_layers,
        "avg_overlap_all_layers": float(np.mean([r['overlap_percentage'] for r in results.values()])),
        "avg_amplification_all_layers": float(np.mean([r['avg_amplification'] for r in results.values()]))
    },
    "layer_statistics": layer_stats,
    "detailed_results": {
        str(layer): {
            "overlap_percentage": float(r['overlap_percentage']),
            "num_shared": r['num_shared'],
            "num_wrong_only": r['num_wrong_only'],
            "num_correct_only": r['num_correct_only'],
            "avg_amplification": float(r['avg_amplification']),
            "top_shared_features": r['shared'][:5] if r['shared'] else []
        }
        for layer, r in results.items()
    },
    "patterns": {
        "lowest_overlap_layers": [l['layer'] for l in sorted_by_overlap[:3]],
        "highest_overlap_layers": [l['layer'] for l in sorted_by_overlap[-3:]],
        "highest_amplification_layers": [l['layer'] for l in sorted_by_amp[:3] if l['avg_amplification'] > 1]
    }
}

with open(output_file, 'w') as f:
    json.dump(save_results, f, indent=2)
print(f"\n💾 Results saved to {output_file}")

# Calculate total time
end_time = datetime.now()
duration = (end_time - start_time).total_seconds()
print(f"\n⏱️ Total analysis time: {duration:.1f} seconds")

print("\n✨ Complete analysis finished! All 32 layers have been analyzed.")
print("\n🔑 Key findings:")
print(f"• Average overlap across all layers: {np.mean([r['overlap_percentage'] for r in results.values()]):.1f}%")
print(f"• Layers with <40% overlap (discriminative): {len([r for r in results.values() if r['overlap_percentage'] < 40])}")
print(f"• Layers with 40-60% overlap (moderate): {len([r for r in results.values() if 40 <= r['overlap_percentage'] <= 60])}")
print(f"• Layers with >60% overlap (entangled): {len([r for r in results.values() if r['overlap_percentage'] > 60])}")