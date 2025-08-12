#!/usr/bin/env python3
"""
Load Llama-Scope SAEs with the CORRECT format
Based on the error messages, the format is: l{layer}m_8x for MLP, l{layer}r_8x for residual
"""

import torch
from sae_lens import SAE
import warnings
import os
import json

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

print("🎯 Loading Llama-Scope SAEs - CORRECT FORMAT")
print("=" * 70)

# Your critical layers from the research
target_layers = [7, 13, 14, 15, 25, 28, 29, 30, 31]

print("\nTarget layers based on your decimal bug research:")
print("- Layers 7-15: Early-mid processing (hijacker neurons)")
print("- Layer 25: Critical divergence point (from logit lens)")
print("- Layers 28-31: Late processing")

# Store loaded SAEs
mlp_saes = {}
residual_saes = {}

print("\n" + "=" * 70)
print("Loading MLP SAEs (for neuron analysis)")
print("=" * 70)

for layer in target_layers:
    sae_id = f"l{layer}m_8x"  # Correct format: l{layer}m_8x
    print(f"\nLayer {layer} MLP (ID: {sae_id}):")
    
    try:
        sae = SAE.from_pretrained(
            release="llama_scope_lxm_8x",
            sae_id=sae_id,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        mlp_saes[layer] = sae
        
        # Get info
        if hasattr(sae, 'cfg'):
            d_in = getattr(sae.cfg, 'd_in', 'unknown')
            d_sae = getattr(sae.cfg, 'd_sae', 'unknown')
            print(f"  ✓ Loaded! Input dim: {d_in}, SAE features: {d_sae}")
        else:
            print(f"  ✓ Loaded successfully!")
            
    except Exception as e:
        print(f"  ✗ Failed: {str(e)[:100]}")

print("\n" + "=" * 70)
print("Loading Residual Stream SAEs (for information flow)")
print("=" * 70)

# Also load residual stream SAEs for layer 25 (critical point)
critical_layers = [25, 24, 26]  # Focus on the critical divergence area

for layer in critical_layers:
    sae_id = f"l{layer}r_8x"  # Correct format: l{layer}r_8x
    print(f"\nLayer {layer} Residual (ID: {sae_id}):")
    
    try:
        sae = SAE.from_pretrained(
            release="llama_scope_lxr_8x",
            sae_id=sae_id,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        residual_saes[layer] = sae
        
        if hasattr(sae, 'cfg'):
            d_in = getattr(sae.cfg, 'd_in', 'unknown')
            d_sae = getattr(sae.cfg, 'd_sae', 'unknown')
            print(f"  ✓ Loaded! Input dim: {d_in}, SAE features: {d_sae}")
        else:
            print(f"  ✓ Loaded successfully!")
            
    except Exception as e:
        print(f"  ✗ Failed: {str(e)[:100]}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\n✅ MLP SAEs loaded: {len(mlp_saes)}/{len(target_layers)}")
if mlp_saes:
    print(f"   Layers: {sorted(mlp_saes.keys())}")

print(f"\n✅ Residual SAEs loaded: {len(residual_saes)}/{len(critical_layers)}")
if residual_saes:
    print(f"   Layers: {sorted(residual_saes.keys())}")

# Save configuration
config = {
    'mlp_layers': sorted(mlp_saes.keys()),
    'residual_layers': sorted(residual_saes.keys()),
    'total_saes': len(mlp_saes) + len(residual_saes),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

with open('sae_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"\n📁 Configuration saved to: sae_config.json")

if mlp_saes or residual_saes:
    print("\n🚀 SAEs loaded successfully! Ready for analysis.")
    print("\nNext steps:")
    print("1. Load your model and test prompts")
    print("2. Extract activations at these layers")
    print("3. Pass activations through SAEs to get features")
    print("4. Analyze which features correlate with the bug")
else:
    print("\n❌ No SAEs loaded. Check your internet connection or authentication.")