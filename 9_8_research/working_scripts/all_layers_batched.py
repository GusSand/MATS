#!/usr/bin/env python3
"""
Complete SAE Analysis of All 32 Layers - Batched version to avoid memory issues
Processes layers in groups to manage GPU memory
"""

import torch
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import warnings
import os
import gc
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For debugging

print("🔬 Complete SAE Analysis - All 32 Layers (Batched)")
print("=" * 70)

# Configuration
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
ALL_LAYERS = list(range(32))
BATCH_SIZE = 4  # Process 4 layers at a time

# Test prompts
PROMPT_WRONG = "Q: Which is bigger: 9.8 or 9.11?\nA:"  
PROMPT_CORRECT = "Which is bigger: 9.8 or 9.11?\nAnswer:"  

print(f"\nModel: {MODEL_NAME}")
print(f"Analyzing ALL {len(ALL_LAYERS)} layers in batches of {BATCH_SIZE}")
print("Expected time: ~2-3 minutes")

start_time = datetime.now()

# Load model and tokenizer once
print("\n📥 Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model.eval()

# Tokenize prompts once
print("\n📝 Preparing prompts...")
inputs_wrong = tokenizer(PROMPT_WRONG, return_tensors="pt")
inputs_correct = tokenizer(PROMPT_CORRECT, return_tensors="pt")
if torch.cuda.is_available():
    inputs_wrong = {k: v.cuda() for k, v in inputs_wrong.items()}
    inputs_correct = {k: v.cuda() for k, v in inputs_correct.items()}

def extract_mlp_activation(model, inputs, layer_idx):
    """Extract MLP activation for a single layer."""
    activation = None
    
    def hook_fn(module, input, output):
        nonlocal activation
        activation = output.detach().cpu()  # Move to CPU immediately
    
    hook = model.model.layers[layer_idx].mlp.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        _ = model(**inputs)
    
    hook.remove()
    return activation

def analyze_layer(layer_idx, token_pos=-1):
    """Analyze a single layer and return results."""
    try:
        # Load SAE for this layer
        sae = SAE.from_pretrained(
            release="llama_scope_lxm_8x",
            sae_id=f"l{layer_idx}m_8x",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Get activations
        act_wrong = extract_mlp_activation(model, inputs_wrong, layer_idx)
        act_correct = extract_mlp_activation(model, inputs_correct, layer_idx)
        
        # Extract features
        with torch.no_grad():
            # Process on GPU if available
            if torch.cuda.is_available():
                token_act_wrong = act_wrong[0, token_pos, :].unsqueeze(0).cuda()
                token_act_correct = act_correct[0, token_pos, :].unsqueeze(0).cuda()
            else:
                token_act_wrong = act_wrong[0, token_pos, :].unsqueeze(0)
                token_act_correct = act_correct[0, token_pos, :].unsqueeze(0)
            
            feat_wrong = sae.encode(token_act_wrong).squeeze(0)
            feat_correct = sae.encode(token_act_correct).squeeze(0)
        
        # Compare features
        top_k = 20
        wrong_top_vals, wrong_top_idx = torch.topk(feat_wrong, k=min(top_k, len(feat_wrong)))
        correct_top_vals, correct_top_idx = torch.topk(feat_correct, k=min(top_k, len(feat_correct)))
        
        wrong_set = set(wrong_top_idx.cpu().tolist())
        correct_set = set(correct_top_idx.cpu().tolist())
        
        shared = wrong_set & correct_set
        wrong_only = wrong_set - correct_set
        correct_only = correct_set - wrong_set
        
        overlap_pct = (len(shared) / min(len(wrong_set), len(correct_set)) * 100) if min(len(wrong_set), len(correct_set)) > 0 else 0
        
        # Calculate amplification
        amplifications = []
        for feat_idx in shared:
            wrong_val = feat_wrong[feat_idx].cpu().item()
            correct_val = feat_correct[feat_idx].cpu().item()
            if correct_val > 0:
                amplifications.append(wrong_val / correct_val)
        
        result = {
            'layer': layer_idx,
            'overlap_percentage': overlap_pct,
            'num_shared': len(shared),
            'num_wrong_only': len(wrong_only),
            'num_correct_only': len(correct_only),
            'avg_amplification': np.mean(amplifications) if amplifications else 1.0
        }
        
        # Clean up
        del sae, act_wrong, act_correct, feat_wrong, feat_correct
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return result
        
    except Exception as e:
        print(f"\n✗ Layer {layer_idx} failed: {str(e)[:50]}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None

# Process layers in batches
print("\n📊 Analyzing layers...")
all_results = []
failed_layers = []

for batch_start in range(0, len(ALL_LAYERS), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(ALL_LAYERS))
    batch_layers = ALL_LAYERS[batch_start:batch_end]
    
    print(f"\n🔄 Processing batch: Layers {batch_start}-{batch_end-1}")
    
    for layer_idx in batch_layers:
        print(f"  Analyzing Layer {layer_idx}...", end=" ")
        result = analyze_layer(layer_idx)
        
        if result:
            all_results.append(result)
            print(f"✓ (Overlap: {result['overlap_percentage']:.1f}%)")
        else:
            failed_layers.append(layer_idx)
            print("✗")
    
    # Clear memory after each batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# Sort results by layer
all_results.sort(key=lambda x: x['layer'])

print("\n" + "=" * 70)
print("COMPLETE RESULTS TABLE")
print("=" * 70)

print("\n| Layer | Overlap % | Shared | Wrong | Correct | Amp   | Visual              | Notes |")
print("|-------|-----------|--------|-------|---------|-------|---------------------|-------|")

for r in all_results:
    layer = r['layer']
    notes = ""
    if layer == 6:
        notes = "Format?"
    elif layer == 8:
        notes = "Early disc"
    elif layer == 10:
        notes = "HIGH"
    elif layer in [13, 14, 15]:
        notes = "Hijacker"
    elif layer == 25:
        notes = "CRITICAL"
    elif layer >= 28:
        notes = "Output"
    
    # Visual bar for overlap
    bar_length = int(r['overlap_percentage'] / 5)
    visual_bar = "█" * bar_length + "░" * (20 - bar_length)
    
    print(f"| {layer:5} | {r['overlap_percentage']:9.1f}% | {r['num_shared']:6} | {r['num_wrong_only']:5} | {r['num_correct_only']:7} | {r['avg_amplification']:5.2f}x | {visual_bar} | {notes:7} |")

print("\n" + "=" * 70)
print("KEY PATTERNS DISCOVERED")
print("=" * 70)

# Analysis
sorted_by_overlap = sorted(all_results, key=lambda x: x['overlap_percentage'])
sorted_by_amp = sorted(all_results, key=lambda x: x['avg_amplification'], reverse=True)

print("\n🔍 Layers with LOWEST overlap (most discrimination):")
for i in range(min(5, len(sorted_by_overlap))):
    l = sorted_by_overlap[i]
    print(f"  Layer {l['layer']:2}: {l['overlap_percentage']:5.1f}% overlap")

print("\n🔍 Layers with HIGHEST overlap (most entanglement):")
for i in range(min(5, len(sorted_by_overlap))):
    l = sorted_by_overlap[-(i+1)]
    print(f"  Layer {l['layer']:2}: {l['overlap_percentage']:5.1f}% overlap")

print("\n⚡ Layers with HIGHEST amplification:")
for i in range(min(5, len(sorted_by_amp))):
    l = sorted_by_amp[i]
    if l['avg_amplification'] > 1:
        print(f"  Layer {l['layer']:2}: {l['avg_amplification']:.2f}x amplification")

# Identify transitions
print("\n🔄 Major transitions (>30% change):")
for i in range(1, len(all_results)):
    prev = all_results[i-1]['overlap_percentage']
    curr = all_results[i]['overlap_percentage']
    if abs(curr - prev) > 30:
        print(f"  Layer {all_results[i-1]['layer']}→{all_results[i]['layer']}: {prev:.1f}% → {curr:.1f}% ({curr-prev:+.1f}%)")

# Statistics
overlaps = [r['overlap_percentage'] for r in all_results]
amplifications = [r['avg_amplification'] for r in all_results]

print("\n📈 Overall Statistics:")
print(f"  Average overlap: {np.mean(overlaps):.1f}% (std: {np.std(overlaps):.1f}%)")
print(f"  Average amplification: {np.mean(amplifications):.2f}x")
print(f"  Layers analyzed: {len(all_results)}/{len(ALL_LAYERS)}")
if failed_layers:
    print(f"  Failed layers: {failed_layers}")

# Categorize layers
low_overlap = len([r for r in all_results if r['overlap_percentage'] < 40])
medium_overlap = len([r for r in all_results if 40 <= r['overlap_percentage'] <= 60])
high_overlap = len([r for r in all_results if r['overlap_percentage'] > 60])

print(f"\n📊 Layer Distribution:")
print(f"  Low overlap (<40%): {low_overlap} layers - Discriminative")
print(f"  Medium overlap (40-60%): {medium_overlap} layers - Moderate entanglement")
print(f"  High overlap (>60%): {high_overlap} layers - High entanglement")

# Create visualization
if len(all_results) > 0:
    print("\n📊 Creating visualization...")
    
    layers = [r['layer'] for r in all_results]
    overlaps_plot = [r['overlap_percentage'] for r in all_results]
    amps_plot = [r['avg_amplification'] for r in all_results]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Overlap plot
    colors1 = ['red' if o < 40 else 'yellow' if o < 60 else 'green' for o in overlaps_plot]
    ax1.bar(layers, overlaps_plot, color=colors1, edgecolor='black', linewidth=0.5)
    ax1.axhline(y=40, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(y=60, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Feature Overlap %')
    ax1.set_title('Feature Overlap Between Correct and Wrong Formats Across All Layers')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(layers)
    
    # Amplification plot
    colors2 = ['red' if a > 1.5 else 'blue' if a < 0.7 else 'gray' for a in amps_plot]
    ax2.bar(layers, amps_plot, color=colors2, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=1.0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Amplification Factor')
    ax2.set_title('Feature Amplification in Wrong Format')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(layers)
    
    plt.tight_layout()
    plt.savefig('all_layers_complete_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved to all_layers_complete_analysis.png")

# Save results
output = {
    "metadata": {
        "total_layers": len(ALL_LAYERS),
        "layers_analyzed": len(all_results),
        "failed_layers": failed_layers,
        "analysis_time": (datetime.now() - start_time).total_seconds()
    },
    "statistics": {
        "avg_overlap": float(np.mean(overlaps)),
        "std_overlap": float(np.std(overlaps)),
        "avg_amplification": float(np.mean(amplifications)),
        "distribution": {
            "low_overlap": low_overlap,
            "medium_overlap": medium_overlap,
            "high_overlap": high_overlap
        }
    },
    "layer_results": all_results,
    "key_findings": {
        "lowest_overlap_layers": [l['layer'] for l in sorted_by_overlap[:5]],
        "highest_overlap_layers": [l['layer'] for l in sorted_by_overlap[-5:]],
        "highest_amplification_layers": [l['layer'] for l in sorted_by_amp[:5] if l['avg_amplification'] > 1]
    }
}

with open('all_32_layers_analysis.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n💾 Results saved to all_32_layers_analysis.json")
print(f"⏱️ Total time: {(datetime.now() - start_time).total_seconds():.1f} seconds")
print("\n✨ Analysis complete!")