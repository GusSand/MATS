#!/usr/bin/env python3
"""
Load Llama-Scope SAEs directly from Hugging Face
Following the correct format from the documentation
"""

import torch
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

print("Loading Llama-Scope SAEs for Circuit Analysis")
print("=" * 70)

# Based on your research findings:
# - Layer 25: Critical divergence point
# - Layers 7-15: Early-mid processing where hijacking occurs
# - Layers 28-31: Late processing

print("\nAttempting to load SAEs with corrected format...")

# Try the format from the Python code you provided
test_configs = [
    # Your provided format
    ("fnlp/Llama-Scope-8x", "L7R-8x"),
    ("fnlp/Llama-Scope-8x", "L15R-8x"),
    ("fnlp/Llama-Scope-8x", "L25R-8x"),
    ("fnlp/Llama-Scope-8x", "L28R-8x"),
    ("fnlp/Llama-Scope-8x", "L31R-8x"),
]

saes = {}
for release, sae_id in test_configs:
    layer_num = int(sae_id.split('R')[0][1:])  # Extract layer number
    print(f"\nLayer {layer_num}:")
    try:
        sae = SAE.from_pretrained(
            release,
            sae_id=sae_id,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        saes[layer_num] = sae
        print(f"  ✓ Loaded successfully!")
        
    except Exception as e:
        # If the standard format doesn't work, try without the expansion suffix
        try:
            alt_id = sae_id.replace("-8x", "")
            print(f"  Trying alternative ID: {alt_id}")
            sae = SAE.from_pretrained(
                release.replace("-8x", ""),
                sae_id=alt_id,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            saes[layer_num] = sae
            print(f"  ✓ Loaded with alternative format!")
        except:
            print(f"  ✗ Failed to load")

if not saes:
    print("\n" + "=" * 70)
    print("Standard loading failed. Trying manual download approach...")
    
    # Alternative: Try to load from Hugging Face model hub directly
    from huggingface_hub import snapshot_download
    import json
    
    try:
        # Download the model files
        cache_dir = "./sae_cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        print(f"\nDownloading Llama-Scope models to {cache_dir}...")
        
        # Try downloading the repository
        local_dir = snapshot_download(
            repo_id="fnlp/Llama-Scope-8x",
            cache_dir=cache_dir,
            local_dir=cache_dir,
            ignore_patterns=["*.md", "*.txt"]
        )
        
        print(f"Downloaded to: {local_dir}")
        
        # List what was downloaded
        import glob
        files = glob.glob(f"{cache_dir}/**/*", recursive=True)
        print(f"\nFound {len(files)} files")
        
        # Look for SAE files
        sae_files = [f for f in files if 'sae' in f.lower() or 'layer' in f.lower()]
        if sae_files:
            print("\nPotential SAE files:")
            for f in sae_files[:10]:
                print(f"  - {f}")
        
    except Exception as e:
        print(f"Download failed: {e}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if saes:
    print(f"✅ Successfully loaded {len(saes)} SAEs")
    print(f"Layers available: {sorted(saes.keys())}")
    print("\nYou can now proceed with SAE-based circuit analysis!")
else:
    print("❌ Could not load SAEs through standard methods")
    print("\nAlternative approaches to try:")
    print("1. Use SAEs from a different repository")
    print("2. Train custom SAEs on your specific layers")
    print("3. Use direct activation analysis (as in your previous work)")