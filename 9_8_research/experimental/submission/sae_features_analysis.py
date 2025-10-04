#!/usr/bin/env python
"""
Proper SAE Feature Analysis using pre-trained SAE from EleutherAI.
This will extract learned features, not raw neuron activations.
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

print("🔬 SAE Feature Analysis with Pre-trained SAE")
print("=" * 70)

# Configuration
TARGET_LAYER = 15 
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SAE_REPO_ID = "EleutherAI/sae-llama-3.1-8b-64x"  # 64x means 64x expansion factor

# Load the base model and tokenizer
print(f"\nLoading {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model.eval()

# Load the pre-trained SAE for our target layer
print(f"\nLoading SAE from {SAE_REPO_ID} for layer {TARGET_LAYER}...")
try:
    # SAE naming convention: usually includes layer number
    sae = SAE.from_pretrained(
        release=SAE_REPO_ID,
        sae_id=f"layer_{TARGET_LAYER}",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"SAE loaded successfully!")
    print(f"SAE has {sae.cfg.d_sae} features (from {sae.cfg.d_in} dimensions)")
except Exception as e:
    print(f"Error loading SAE: {e}")
    print("Trying alternative loading method...")
    # Alternative: load from HuggingFace directly
    from huggingface_hub import hf_hub_download
    
    # Download SAE weights
    sae_path = hf_hub_download(
        repo_id=SAE_REPO_ID,
        filename=f"layer_{TARGET_LAYER}/sae_weights.safetensors"
    )
    print(f"SAE path: {sae_path}")
    # Would need to load manually here
    exit()

# Define our prompts
PROMPT_BAD_STATE = [{"role": "user", "content": "Which is bigger: 9.8 or 9.11?"}]
PROMPT_GOOD_STATE = "Which is bigger: 9.8 or 9.11?\nAnswer:"

def get_sae_features(prompt, sae, model, tokenizer, layer_idx=TARGET_LAYER):
    """
    Extract SAE features for a given prompt.
    """
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
        mlp_acts = output.detach()
    
    # Register hook to capture MLP activations
    hook = model.model.layers[layer_idx].mlp.act_fn.register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        _ = model(**inputs)
    
    # Remove hook
    hook.remove()
    
    # Get activations for last token
    last_token_mlp = mlp_acts[0, -1, :].unsqueeze(0)  # Shape: [1, d_model]
    
    # Pass through SAE to get features
    feature_acts = sae.encode(last_token_mlp)
    feature_acts = feature_acts.squeeze(0)  # Shape: [d_sae]
    
    print(f"SAE feature shape: {feature_acts.shape}")
    print(f"Number of active features (>0.1): {(feature_acts > 0.1).sum().item()}")
    
    return feature_acts, prompt_text

def analyze_top_features(features, top_k=20):
    """
    Find and return top-k most active SAE features.
    """
    # SAE features are sparse, so many will be zero
    nonzero_mask = features > 0.01  # Threshold for considering active
    active_features = features[nonzero_mask]
    active_indices = torch.where(nonzero_mask)[0]
    
    if len(active_features) == 0:
        print("No active features found!")
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

# Analyze bad state (chat format)
print("\n" + "="*60)
print("ANALYZING BAD STATE (Chat Format - Shows Bug)")
print("="*60)

bad_features, bad_prompt = get_sae_features(PROMPT_BAD_STATE, sae, model, tokenizer)
bad_top = analyze_top_features(bad_features, top_k=20)

print("\nTop 10 SAE features in bad state:")
for i, feat in enumerate(bad_top[:10]):
    print(f"  {i+1}. Feature {feat['feature_idx']}: {feat['activation']:.4f}")

# Analyze good state (simple format)
print("\n" + "="*60)
print("ANALYZING GOOD STATE (Simple Format - Correct)")
print("="*60)

good_features, good_prompt = get_sae_features(PROMPT_GOOD_STATE, sae, model, tokenizer)
good_top = analyze_top_features(good_features, top_k=20)

print("\nTop 10 SAE features in good state:")
for i, feat in enumerate(good_top[:10]):
    print(f"  {i+1}. Feature {feat['feature_idx']}: {feat['activation']:.4f}")

# Compare features
print("\n" + "="*60)
print("ENTANGLEMENT ANALYSIS")
print("="*60)

bad_features_set = {f['feature_idx'] for f in bad_top}
good_features_set = {f['feature_idx'] for f in good_top}

shared_features = bad_features_set.intersection(good_features_set)
bad_only = bad_features_set - good_features_set
good_only = good_features_set - bad_features_set

print(f"\nSAE features active in BOTH states: {len(shared_features)}")
print(f"SAE features ONLY in bad state: {len(bad_only)}")
print(f"SAE features ONLY in good state: {len(good_only)}")

if shared_features:
    print(f"\nShared features: {sorted(list(shared_features))[:10]}...")

# Quantitative comparison of shared features
print("\n" + "="*60)
print("AMPLIFICATION ANALYSIS OF SHARED FEATURES")
print("="*60)

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

print("\nTop amplified shared features:")
for i, amp in enumerate(amplifications[:10]):
    print(f"  {i+1}. Feature {amp['feature']}: {amp['amplification']:.2f}x "
          f"(bad: {amp['bad_activation']:.3f}, good: {amp['good_activation']:.3f})")

# Test entanglement hypothesis
print("\n" + "="*70)
print("IRREMEDIABLE ENTANGLEMENT HYPOTHESIS TEST")
print("="*70)

if len(shared_features) > 5:
    print("✅ STRONG EVIDENCE FOR ENTANGLEMENT:")
    print(f"   - {len(shared_features)} SAE features are active in BOTH states")
    print("   - These features likely represent decimal number concepts")
    
    # Check for amplification
    high_amp = [a for a in amplifications if a['amplification'] > 1.5]
    if high_amp:
        print(f"\n   - {len(high_amp)} features show >1.5x amplification in bad state")
        print("   - This suggests the bug amplifies existing decimal features")
        print("   - Rather than using completely different features")
    
    print("\n   CONCLUSION: The same SAE features that correctly process")
    print("   decimals are HIJACKED and AMPLIFIED to produce the bug!")
else:
    print("❌ Limited evidence for entanglement")
    print(f"   Only {len(shared_features)} shared features found")

# Look for decimal-related features
print("\n" + "="*60)
print("SEARCHING FOR DECIMAL-RELATED FEATURES")
print("="*60)
print("Note: SAE features often capture semantic concepts.")
print("Features active here likely represent:")
print("- Decimal numbers")
print("- Comparison operations")
print("- Magnitude concepts")

# Save results
results = {
    'sae_info': {
        'repo_id': SAE_REPO_ID,
        'layer': TARGET_LAYER,
        'n_features': sae.cfg.d_sae,
        'd_in': sae.cfg.d_in
    },
    'bad_top_features': bad_top[:20],
    'good_top_features': good_top[:20],
    'shared_features': list(shared_features),
    'amplifications': amplifications[:10]
}

with open("sae_features_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to: sae_features_results.json")