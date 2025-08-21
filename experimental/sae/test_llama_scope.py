#!/usr/bin/env python3
"""
Test loading Llama-Scope SAEs for Llama 3.1 8B
"""

import torch
from sae_lens import SAE
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

print("Testing Llama-Scope SAE Loading")
print("=" * 70)

# Test layers from your research
test_layers = [7, 13, 14, 15, 25, 28, 29, 30, 31]

print("\nAttempting to load SAEs for key layers...")
print("Layers 7-15: Early-mid processing")
print("Layer 25: Critical divergence point (from your research)")
print("Layers 28-31: Late processing")

successful_loads = []
failed_loads = []

for layer in test_layers:
    print(f"\nLayer {layer}:")
    try:
        # Try the 8x expansion first (smaller, faster)
        sae = SAE.from_pretrained(
            release="fnlp/Llama-Scope-8x",
            sae_id=f"L{layer}R",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Check SAE properties
        if hasattr(sae, 'cfg'):
            d_in = sae.cfg.d_in if hasattr(sae.cfg, 'd_in') else 'unknown'
            d_sae = sae.cfg.d_sae if hasattr(sae.cfg, 'd_sae') else 'unknown'
            print(f"  ✓ Loaded successfully!")
            print(f"    Input dim: {d_in}")
            print(f"    SAE features: {d_sae}")
        else:
            print(f"  ✓ Loaded (config details unavailable)")
        
        successful_loads.append(layer)
        
    except Exception as e:
        print(f"  ✗ Failed: {str(e)[:100]}")
        failed_loads.append(layer)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Successfully loaded: {successful_loads}")
print(f"Failed to load: {failed_loads}")

if successful_loads:
    print(f"\n✅ {len(successful_loads)}/{len(test_layers)} SAEs loaded successfully")
    print("You can proceed with SAE analysis for these layers!")
else:
    print("\n❌ No SAEs loaded successfully")
    print("This might be due to:")
    print("  1. Network/download issues")
    print("  2. SAE naming convention changes")
    print("  3. Missing model files on Hugging Face")