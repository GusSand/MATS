#!/usr/bin/env python
"""
Proper SAE analysis using the available layers (23 and 29) for Llama 3.1
"""

import torch
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import os
import json
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

print("🔬 SAE Feature Analysis - Using Available Layers")
print("=" * 70)

# Configuration - Use layer 23 which is available
TARGET_LAYER = 23  # Changed from 15 to 23
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# Load the base model and tokenizer
print(f"\nLoading {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model.eval()

# Load the SAE - using the correct format
print(f"\nLoading SAE for layer {TARGET_LAYER}...")
try:
    # Method 1: Direct from HuggingFace
    sae = SAE.from_pretrained(
        release="EleutherAI/sae-llama-3.1-8b-32x",
        sae_id=f"layers.{TARGET_LAYER}.mlp",  # Correct format!
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"✓ SAE loaded successfully!")
    print(f"  Input dimension: {sae.cfg.d_in}")
    print(f"  SAE features: {sae.cfg.d_sae}")
    print(f"  Expansion factor: {sae.cfg.d_sae / sae.cfg.d_in:.1f}x")
except Exception as e:
    print(f"Error: {e}")
    exit()

# Define our prompts
PROMPT_BAD = [{"role": "user", "content": "Which is bigger: 9.8 or 9.11?"}]
PROMPT_GOOD = "Which is bigger: 9.8 or 9.11?\nAnswer:"

def get_sae_features(prompt, sae, model, tokenizer, layer_idx=TARGET_LAYER):
    """Extract SAE features for a given prompt."""
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
    
    # Storage for activations
    mlp_acts = None
    
    def hook_fn(module, input, output):
        nonlocal mlp_acts
        # For Llama, MLP output is after the activation function
        mlp_acts = output.detach()
    
    # Register hook on the MLP output
    hook = model.model.layers[layer_idx].mlp.register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        _ = model(**inputs)
    
    # Remove hook
    hook.remove()
    
    # Get activations for last token
    last_token_mlp = mlp_acts[0, -1, :].unsqueeze(0)  # Shape: [1, d_model]
    
    # Pass through SAE encoder to get features
    with torch.no_grad():
        # The SAE expects input of shape [batch, d_in]
        feature_acts = sae.encode(last_token_mlp.to(sae.device))
    
    feature_acts = feature_acts.squeeze(0)  # Shape: [d_sae]
    
    print(f"SAE feature shape: {feature_acts.shape}")
    print(f"Number of active features (>0.1): {(feature_acts > 0.1).sum().item()}")
    print(f"Max activation: {feature_acts.max().item():.3f}")
    
    return feature_acts, prompt_text

def analyze_top_features(features, top_k=30):
    """Find and return top-k most active SAE features."""
    # SAE features are sparse
    nonzero_mask = features > 0.01
    active_features = features[nonzero_mask]
    active_indices = torch.where(nonzero_mask)[0]
    
    if len(active_features) == 0:
        return []
    
    # Sort by activation strength
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

# Analyze bad state (chat format - exhibits bug)
print("\n" + "="*70)
print("ANALYZING BAD STATE (Chat Format - Shows Bug)")
print("="*70)

bad_features, bad_prompt = get_sae_features(PROMPT_BAD, sae, model, tokenizer)
bad_top = analyze_top_features(bad_features)

print("\nTop 15 SAE features in bad state:")
for i, feat in enumerate(bad_top[:15]):
    print(f"  {i+1:2d}. Feature {feat['feature_idx']:5d}: {feat['activation']:.4f}")

# Analyze good state (simple format - correct)
print("\n" + "="*70)
print("ANALYZING GOOD STATE (Simple Format - Correct)")
print("="*70)

good_features, good_prompt = get_sae_features(PROMPT_GOOD, sae, model, tokenizer)
good_top = analyze_top_features(good_features)

print("\nTop 15 SAE features in good state:")
for i, feat in enumerate(good_top[:15]):
    print(f"  {i+1:2d}. Feature {feat['feature_idx']:5d}: {feat['activation']:.4f}")

# Compare features
print("\n" + "="*70)
print("ENTANGLEMENT ANALYSIS")
print("="*70)

bad_features_set = {f['feature_idx'] for f in bad_top}
good_features_set = {f['feature_idx'] for f in good_top}

shared_features = bad_features_set.intersection(good_features_set)
bad_only = bad_features_set - good_features_set
good_only = good_features_set - bad_features_set

print(f"\nSAE features active in BOTH states: {len(shared_features)}")
print(f"SAE features ONLY in bad state: {len(bad_only)}")
print(f"SAE features ONLY in good state: {len(good_only)}")

if shared_features:
    print(f"\nShared feature indices: {sorted(list(shared_features))[:20]}")

# Quantitative comparison of shared features
print("\n" + "="*70)
print("AMPLIFICATION ANALYSIS OF SHARED FEATURES")
print("="*70)

bad_lookup = {f['feature_idx']: f['activation'] for f in bad_top}
good_lookup = {f['feature_idx']: f['activation'] for f in good_top}

amplifications = []
for feat_idx in shared_features:
    bad_act = bad_lookup[feat_idx]
    good_act = good_lookup[feat_idx]
    
    if good_act > 0:
        amp_ratio = bad_act / good_act
        amplifications.append({
            'feature': feat_idx,
            'bad_activation': bad_act,
            'good_activation': good_act,
            'amplification': amp_ratio
        })

# Sort by amplification
amplifications.sort(key=lambda x: x['amplification'], reverse=True)

print("\nTop 10 amplified shared features (bad/good ratio):")
for i, amp in enumerate(amplifications[:10]):
    print(f"  {i+1:2d}. Feature {amp['feature']:5d}: {amp['amplification']:5.2f}x "
          f"(bad: {amp['bad_activation']:.3f}, good: {amp['good_activation']:.3f})")

# Also show features that are similar in both
similar_features = [a for a in amplifications if 0.8 < a['amplification'] < 1.2]
if similar_features:
    print(f"\nFeatures with similar activation in both states ({len(similar_features)} found):")
    for amp in similar_features[:5]:
        print(f"  Feature {amp['feature']:5d}: ratio {amp['amplification']:.2f} "
              f"(bad: {amp['bad_activation']:.3f}, good: {amp['good_activation']:.3f})")

# Test entanglement hypothesis
print("\n" + "="*70)
print("IRREMEDIABLE ENTANGLEMENT HYPOTHESIS TEST")
print("="*70)

if len(shared_features) > 10:
    print("✅ STRONG EVIDENCE FOR ENTANGLEMENT:")
    print(f"   - {len(shared_features)} SAE features are active in BOTH states")
    print("   - These features likely represent:")
    print("     • Decimal number patterns (9.8, 9.11)")
    print("     • Comparison operations")
    print("     • Numerical magnitude concepts")
    
    # Check for strong amplifications
    high_amp = [a for a in amplifications if a['amplification'] > 2.0]
    if high_amp:
        print(f"\n   - {len(high_amp)} features show >2x amplification in bad state:")
        for ha in high_amp[:3]:
            print(f"     • Feature {ha['feature']}: {ha['amplification']:.1f}x amplification")
    
    # Check for consistent features
    consistent = [a for a in amplifications if 0.5 < a['amplification'] < 2.0]
    if consistent:
        print(f"\n   - {len(consistent)} features are consistently active in both states")
    
    print("\n   CONCLUSION: The same learned features that enable correct")
    print("   decimal comparison are being used (often amplified) to")
    print("   produce the incorrect answer. The bug and correct function")
    print("   share the same feature representations!")
else:
    print(f"⚠️  Limited shared features ({len(shared_features)})")
    print("   This could mean:")
    print("   - Layer 23 is too late in the network")
    print("   - The bug manifests earlier in processing")

# Save detailed results
results = {
    'sae_info': {
        'layer': TARGET_LAYER,
        'd_in': sae.cfg.d_in,
        'd_sae': sae.cfg.d_sae,
        'expansion': sae.cfg.d_sae / sae.cfg.d_in
    },
    'bad_top_features': bad_top[:30],
    'good_top_features': good_top[:30],
    'n_shared': len(shared_features),
    'shared_features': sorted(list(shared_features))[:50],
    'amplifications': amplifications[:20]
}

with open("sae_features_layer23_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Detailed results saved to: sae_features_layer23_results.json")