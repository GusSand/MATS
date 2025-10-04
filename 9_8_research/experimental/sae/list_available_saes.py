#!/usr/bin/env python3
"""
List all available pre-trained SAEs in SAELens
"""

from sae_lens import pretrained_saes_directory
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

print("Available Pre-trained SAEs in SAELens")
print("=" * 70)

try:
    # Get the directory of available SAEs
    print("\nChecking pretrained SAEs directory...")
    
    # This should list all available SAEs
    if hasattr(pretrained_saes_directory, 'get_pretrained_saes_directory'):
        directory = pretrained_saes_directory.get_pretrained_saes_directory()
        print(f"Found {len(directory)} SAE entries")
        
        # Filter for Llama-related SAEs
        llama_saes = [entry for entry in directory if 'llama' in entry.lower() or 'meta' in entry.lower()]
        
        if llama_saes:
            print(f"\nLlama-related SAEs:")
            for sae in llama_saes[:10]:  # Show first 10
                print(f"  - {sae}")
        else:
            print("\nNo Llama-specific SAEs found")
            
        # Show a sample of all available
        print(f"\nSample of all available SAEs:")
        for i, entry in enumerate(list(directory)[:10]):
            print(f"  {i+1}. {entry}")
            
    else:
        # Try alternative method
        print("Trying alternative listing method...")
        import sae_lens
        
        # Check for any available listing methods
        available_methods = [attr for attr in dir(sae_lens) if 'list' in attr.lower() or 'directory' in attr.lower()]
        print(f"Available directory methods: {available_methods}")
        
except Exception as e:
    print(f"Error accessing SAE directory: {e}")
    
    # Try to import and check manually
    print("\nTrying manual check...")
    try:
        from sae_lens.pretrained_saes import PRETRAINED_SAES
        print(f"Found PRETRAINED_SAES with {len(PRETRAINED_SAES)} entries")
        
        # Show Llama entries
        for key in PRETRAINED_SAES:
            if 'llama' in key.lower():
                print(f"  - {key}: {PRETRAINED_SAES[key]}")
                
    except ImportError:
        print("Could not import PRETRAINED_SAES")

print("\n" + "=" * 70)
print("Note: Llama-Scope SAEs might need to be loaded differently")
print("or might not be integrated into SAELens yet.")