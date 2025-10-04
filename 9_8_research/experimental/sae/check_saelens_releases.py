#!/usr/bin/env python3
"""
Check what SAE releases are actually available in SAELens
"""

import warnings
import os
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

print("Checking SAELens Available Releases")
print("=" * 70)

# Try to access the pretrained SAEs registry
try:
    from sae_lens.pretrained_saes import PRETRAINED_SAES_REGISTRY
    
    print(f"\nFound {len(PRETRAINED_SAES_REGISTRY)} registered SAE releases:")
    
    # Look for Llama-related entries
    llama_releases = []
    for release_name in PRETRAINED_SAES_REGISTRY:
        if 'llama' in release_name.lower():
            llama_releases.append(release_name)
            print(f"  - {release_name}")
    
    if not llama_releases:
        print("\nNo Llama-specific releases found. Showing all releases:")
        for i, release in enumerate(list(PRETRAINED_SAES_REGISTRY.keys())[:20]):
            print(f"  {i+1}. {release}")
            
except ImportError:
    print("Could not import PRETRAINED_SAES_REGISTRY")
    
    # Try alternative import
    try:
        import sae_lens
        
        # Check what's available in the module
        print("\nChecking sae_lens module contents...")
        
        # Look for any registry or config
        registry_attrs = [attr for attr in dir(sae_lens) if 'registry' in attr.lower() or 'pretrained' in attr.lower()]
        print(f"Registry-related attributes: {registry_attrs}")
        
        # Try to find SAE class methods
        from sae_lens import SAE
        
        print("\nSAE class methods for loading:")
        load_methods = [m for m in dir(SAE) if 'load' in m.lower() or 'from' in m.lower()]
        for method in load_methods:
            print(f"  - {method}")
            
    except Exception as e:
        print(f"Error exploring module: {e}")

# Check if there's a config file or constant defining available models
try:
    from sae_lens.pretrained_saes import NAMED_PRETRAINED_SAE_LOADERS
    
    print("\n\nNamed SAE Loaders available:")
    for name in NAMED_PRETRAINED_SAE_LOADERS:
        print(f"  - {name}")
        
except ImportError:
    pass

# Try one more approach - check for Llama in a different format
print("\n" + "=" * 70)
print("Trying direct load with different release names...")

from sae_lens import SAE

test_releases = [
    "llama-3-8b-it-res-jh",  # From the SAE table
    "llama_scope_lxm_8x",
    "llama_scope_lxr_8x",
    "llama-scope-8x",
    "Llama-Scope-8x",
]

for release in test_releases:
    try:
        print(f"\nTrying release: '{release}'")
        # Just try to trigger the error to see available SAE IDs
        sae = SAE.from_pretrained(
            release=release,
            sae_id="dummy",  # This will fail but might show us valid IDs
            device="cpu"
        )
    except Exception as e:
        error_str = str(e)
        if "available" in error_str.lower() or "valid" in error_str.lower():
            print(f"  Error message (might contain valid IDs): {error_str[:200]}")
        else:
            print(f"  Failed: {error_str[:100]}")