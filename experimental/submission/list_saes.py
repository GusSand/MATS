#!/usr/bin/env python
"""
List available pre-trained SAEs in sae-lens
"""

from sae_lens import pretrained_saes_directory

print("Available pre-trained SAEs:")
print("="*60)

for release_name, info in pretrained_saes_directory.items():
    print(f"\nRelease: {release_name}")
    if isinstance(info, dict):
        for k, v in info.items():
            print(f"  {k}: {v}")
    else:
        print(f"  Info: {info}")

print("\n" + "="*60)
print("Note: Use the release name when loading SAEs")
print("Example: SAE.from_pretrained(release='gemma-2b-res-jb', sae_id='layer_8')")