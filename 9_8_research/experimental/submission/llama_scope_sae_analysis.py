#!/usr/bin/env python
"""
SAE Feature Analysis using Llama-Scope SAEs for layers 13-15
These are exactly the layers where we found hijacker neurons!
"""

import torch
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import os
import json
import numpy as np
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

print("🔬 SAE Feature Analysis with Llama-Scope")
print("=" * 70)
print("Analyzing layers 13-15 where hijacker neurons were found!")
print("=" * 70)

# Configuration
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
LAYERS_TO_ANALYZE = [13, 14, 15]
SAE_EXPANSION = "32x"  # Use 32x for more features

# Load the base model and tokenizer
print(f"\nLoading {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model.eval()

# Load SAEs for each layer
saes = {}
for layer_idx in LAYERS_TO_ANALYZE:
    print(f"\nLoading SAE for layer {layer_idx}...")
    try:
        # Try Llama-Scope format
        sae = SAE.from_pretrained(
            release="fnlp/Llama-Scope",
            sae_id=f"L{layer_idx}R-{SAE_EXPANSION}",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        saes[layer_idx] = sae
        print(f"✓ Loaded SAE for layer {layer_idx}")
        print(f"  Features: {sae.cfg.d_sae if hasattr(sae.cfg, 'd_sae') else 'unknown'}")
    except Exception as e:
        print(f"✗ Failed to load SAE for layer {layer_idx}: {str(e)[:100]}")

if not saes:
    print("\n❌ No SAEs loaded successfully. Exiting.")
    exit()

# Load our identified hijacker neurons for reference
with open("identified_circuits.json", "r") as f:
    circuits = json.load(f)

hijacker_info = defaultdict(set)
for layer, neuron in circuits["chat"]["hijacker_cluster"]:
    if layer in LAYERS_TO_ANALYZE:
        hijacker_info[layer].add(neuron)

print("\nHijacker neurons to watch:")
for layer in sorted(hijacker_info.keys()):
    print(f"  Layer {layer}: {sorted(hijacker_info[layer])}")

# Define prompts
PROMPT_BAD = [{"role": "user", "content": "Which is bigger: 9.8 or 9.11?"}]
PROMPT_GOOD = "Which is bigger: 9.8 or 9.11?\nAnswer:"

def get_sae_features_all_layers(prompt, saes, model, tokenizer):
    """Extract SAE features for all layers."""
    # Prepare prompt
    if isinstance(prompt, list):
        prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    else:
        prompt_text = prompt
    
    print(f"\nAnalyzing: '{prompt_text[:50]}...'")
    
    # Tokenize
    inputs = tokenizer(prompt_text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Storage for activations per layer
    layer_features = {}
    
    # Register hooks for all layers
    hooks = []
    mlp_acts = {}
    
    def make_hook_fn(layer_idx):
        def hook_fn(module, input, output):
            mlp_acts[layer_idx] = output.detach()
        return hook_fn
    
    for layer_idx in saes.keys():
        hook = model.model.layers[layer_idx].mlp.register_forward_hook(
            make_hook_fn(layer_idx)
        )
        hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        _ = model(**inputs)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Process each layer's activations through its SAE
    for layer_idx, sae in saes.items():
        if layer_idx in mlp_acts:
            # Get last token activations
            last_token_mlp = mlp_acts[layer_idx][0, -1, :].unsqueeze(0)
            
            # Pass through SAE encoder
            with torch.no_grad():
                features = sae.encode(last_token_mlp.to(sae.device))
            
            features = features.squeeze(0)
            layer_features[layer_idx] = features
            
            # Stats
            active = (features > 0.1).sum().item()
            max_act = features.max().item()
            print(f"  Layer {layer_idx}: {active} active features, max={max_act:.3f}")
    
    return layer_features, prompt_text

def analyze_top_features(features, top_k=30):
    """Extract top-k features from activation tensor."""
    nonzero_mask = features > 0.01
    if nonzero_mask.sum() == 0:
        return []
    
    active_features = features[nonzero_mask]
    active_indices = torch.where(nonzero_mask)[0]
    
    sorted_acts, sorted_idx = torch.sort(active_features, descending=True)
    
    results = []
    for i in range(min(top_k, len(sorted_acts))):
        feature_idx = active_indices[sorted_idx[i]].item()
        activation = sorted_acts[i].item()
        results.append({
            'feature_idx': feature_idx,
            'activation': activation
        })
    
    return results

# Analyze bad state (chat format)
print("\n" + "="*70)
print("ANALYZING BAD STATE (Chat Format - Shows Bug)")
print("="*70)

bad_features, bad_prompt = get_sae_features_all_layers(PROMPT_BAD, saes, model, tokenizer)

# Analyze good state (simple format)
print("\n" + "="*70)
print("ANALYZING GOOD STATE (Simple Format - Correct)")
print("="*70)

good_features, good_prompt = get_sae_features_all_layers(PROMPT_GOOD, saes, model, tokenizer)

# Layer-by-layer comparison
print("\n" + "="*70)
print("LAYER-BY-LAYER ENTANGLEMENT ANALYSIS")
print("="*70)

all_results = {}

for layer_idx in sorted(saes.keys()):
    print(f"\n--- Layer {layer_idx} ---")
    
    if layer_idx not in bad_features or layer_idx not in good_features:
        print("Skipping - missing features")
        continue
    
    bad_top = analyze_top_features(bad_features[layer_idx])
    good_top = analyze_top_features(good_features[layer_idx])
    
    # Compare
    bad_set = {f['feature_idx'] for f in bad_top}
    good_set = {f['feature_idx'] for f in good_top}
    
    shared = bad_set.intersection(good_set)
    bad_only = bad_set - good_set
    good_only = good_set - bad_set
    
    print(f"Shared SAE features: {len(shared)}/{len(bad_set)}")
    
    if shared:
        # Calculate amplifications
        bad_dict = {f['feature_idx']: f['activation'] for f in bad_top}
        good_dict = {f['feature_idx']: f['activation'] for f in good_top}
        
        amplifications = []
        for feat_idx in shared:
            bad_act = bad_dict[feat_idx]
            good_act = good_dict[feat_idx]
            if good_act > 0:
                ratio = bad_act / good_act
                amplifications.append({
                    'feature': feat_idx,
                    'bad_activation': bad_act,
                    'good_activation': good_act,
                    'amplification': ratio
                })
        
        # Sort by amplification
        amplifications.sort(key=lambda x: x['amplification'], reverse=True)
        
        print(f"\nTop 5 amplified features:")
        for amp in amplifications[:5]:
            print(f"  Feature {amp['feature']:5d}: {amp['amplification']:4.2f}x "
                  f"(bad: {amp['bad_activation']:.3f}, good: {amp['good_activation']:.3f})")
        
        # Store results
        all_results[layer_idx] = {
            'n_shared': len(shared),
            'shared_features': list(shared),
            'amplifications': amplifications[:10],
            'bad_only': list(bad_only)[:10],
            'good_only': list(good_only)[:10]
        }

# Final summary
print("\n" + "="*70)
print("IRREMEDIABLE ENTANGLEMENT HYPOTHESIS")
print("="*70)

total_shared = sum(r['n_shared'] for r in all_results.values())
highly_amplified = []

for layer_idx, results in all_results.items():
    for amp in results['amplifications']:
        if amp['amplification'] > 2.0:
            highly_amplified.append((layer_idx, amp))

if total_shared > 20:
    print("✅ STRONG EVIDENCE FOR ENTANGLEMENT:")
    print(f"\n   - {total_shared} total shared SAE features across layers")
    print(f"   - {len(highly_amplified)} features show >2x amplification")
    
    if highly_amplified:
        print("\n   Most amplified features:")
        for layer, amp in highly_amplified[:5]:
            print(f"     Layer {layer}, Feature {amp['feature']}: "
                  f"{amp['amplification']:.1f}x amplification")
    
    print("\n   CONCLUSION: The bug uses the SAME learned features")
    print("   as correct decimal processing, but amplifies them!")
    print("   This is true 'irremediable entanglement' - you cannot")
    print("   separate the bug from normal function at the feature level.")
else:
    print(f"⚠️  Limited entanglement: only {total_shared} shared features")

# Save results
results = {
    'sae_source': 'Llama-Scope',
    'layers_analyzed': LAYERS_TO_ANALYZE,
    'layer_results': all_results,
    'total_shared_features': total_shared,
    'highly_amplified_count': len(highly_amplified)
}

with open("llama_scope_sae_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to: llama_scope_sae_results.json")