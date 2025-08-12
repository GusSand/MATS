#!/usr/bin/env python3
"""
Load Llama-Scope SAEs using the correct format from SAELens documentation
"""

import torch
from sae_lens import SAE
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

print("Loading Llama-Scope SAEs - Correct Format")
print("=" * 70)

# Based on the SAE table, we have these Llama-Scope options:
# - llama_scope_lxr_8x (8x expansion, residual stream)
# - llama_scope_lxr_32x (32x expansion, residual stream)
# - llama_scope_lxm_8x (8x expansion, MLP)
# - llama_scope_lxm_32x (32x expansion, MLP)
# - llama_scope_lxa_8x (8x expansion, attention)
# - llama_scope_lxa_32x (32x expansion, attention)

# Your research focused on MLP neurons, so let's use lxm (MLP) SAEs
# Critical layers from your research:
target_layers = [7, 13, 14, 15, 25, 28, 29, 30, 31]

print("\nTarget layers based on your research:")
print("- Layers 7-15: Early-mid processing (hijacker neurons)")
print("- Layer 25: Critical divergence point")
print("- Layers 28-31: Late processing")

saes = {}
successful = []
failed = []

# Try loading with the correct release name
for layer in target_layers:
    print(f"\nLayer {layer}:")
    
    # Try different SAE IDs that might work
    possible_ids = [
        f"blocks.{layer}.hook_mlp_out",  # MLP output hook
        f"layer_{layer}_mlp",             # Alternative naming
        f"L{layer}M",                     # Llama-scope specific
        f"{layer}",                       # Just the layer number
    ]
    
    loaded = False
    for sae_id in possible_ids:
        try:
            print(f"  Trying SAE ID: {sae_id}")
            
            # Try with 8x expansion first (faster, smaller)
            sae = SAE.from_pretrained(
                release="llama_scope_lxm_8x",  # MLP, 8x expansion
                sae_id=sae_id,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            saes[layer] = sae
            successful.append(layer)
            
            # Get info if available
            if hasattr(sae, 'cfg'):
                d_in = getattr(sae.cfg, 'd_in', 'unknown')
                d_sae = getattr(sae.cfg, 'd_sae', 'unknown')
                print(f"  ✓ Loaded! Input: {d_in}, Features: {d_sae}")
            else:
                print(f"  ✓ Loaded!")
            
            loaded = True
            break
            
        except Exception as e:
            continue
    
    if not loaded:
        # Try the residual stream version as fallback
        try:
            print(f"  Trying residual stream SAE...")
            sae = SAE.from_pretrained(
                release="llama_scope_lxr_8x",  # Residual stream, 8x
                sae_id=f"blocks.{layer}.hook_resid_post",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            saes[layer] = sae
            successful.append(layer)
            print(f"  ✓ Loaded residual stream SAE!")
            loaded = True
        except:
            pass
    
    if not loaded:
        failed.append(layer)
        print(f"  ✗ Could not load SAE for layer {layer}")

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

if successful:
    print(f"✅ Successfully loaded SAEs for layers: {successful}")
    print(f"Total: {len(successful)}/{len(target_layers)} layers")
    
    # Save the loaded SAEs info
    sae_info = {
        'loaded_layers': successful,
        'failed_layers': failed,
        'sae_type': 'llama_scope_lxm_8x'
    }
    
    import json
    with open('loaded_saes.json', 'w') as f:
        json.dump(sae_info, f, indent=2)
    
    print("\nSAE info saved to loaded_saes.json")
    print("Ready for circuit analysis!")
    
else:
    print(f"❌ Could not load any SAEs")
    print("\nThis might mean:")
    print("1. The SAE IDs are different than expected")
    print("2. Network/authentication issues")
    print("3. The SAEs need special access permissions")
    
print("\nIf this doesn't work, we can:")
print("1. Check the exact SAE IDs in the Hugging Face repo")
print("2. Use the llama-3-8b-it-res-jh release instead")
print("3. Train custom lightweight SAEs (30-60 min per layer)")