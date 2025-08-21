#!/usr/bin/env python
"""
List the pretrained SAEs directory
"""

from sae_lens.loading import pretrained_saes_directory
import json

print("Pretrained SAEs Directory:")
print("="*70)

# Pretty print the directory
print(json.dumps(pretrained_saes_directory, indent=2))

# Look for Llama models
print("\n" + "="*70)
print("Llama-related SAEs:")
for key, value in pretrained_saes_directory.items():
    if "llama" in key.lower():
        print(f"\n{key}:")
        print(json.dumps(value, indent=2))