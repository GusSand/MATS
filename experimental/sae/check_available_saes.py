#!/usr/bin/env python3
"""
Check available SAEs from Llama-Scope
"""

from sae_lens import SAE
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

print("Checking Available Llama-Scope SAEs")
print("=" * 70)

# Try different naming conventions
test_configs = [
    # Format 1: Layer-specific with expansion
    ("fnlp/Llama-Scope-8x", "L7R-8x"),
    ("fnlp/Llama-Scope-8x", "layer_7/width_131k"),
    ("fnlp/Llama-Scope", "L7R-8x"),
    
    # Format 2: Direct model names
    ("fnlp/Llama-3.1-8B-Instruct-SAE", "L7"),
    ("fnlp/llama-3.1-8b-sae", "layer_7"),
    
    # Format 3: Without expansion suffix
    ("fnlp/Llama-Scope-8x", "L7R"),
    ("fnlp/Llama-Scope-8x", "L7"),
]

print("\nTrying different SAE naming conventions...")

for release, sae_id in test_configs:
    try:
        print(f"\nTrying: release='{release}', sae_id='{sae_id}'")
        sae = SAE.from_pretrained(
            release=release,
            sae_id=sae_id,
            device="cpu"
        )
        print(f"  ✓ SUCCESS! This format works!")
        
        # Try to get more info
        if hasattr(sae, 'cfg'):
            print(f"    Config available: {sae.cfg}")
        break
        
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg:
            print(f"  ✗ 404 Not Found")
        elif "repository" in error_msg.lower():
            print(f"  ✗ Repository issue: {error_msg[:50]}")
        else:
            print(f"  ✗ Error: {error_msg[:100]}")

print("\n" + "=" * 70)
print("If all attempts failed, we may need to:")
print("1. Use the SAE from a different source")
print("2. Train our own SAE on Llama 3.1")
print("3. Use alternative interpretability methods")