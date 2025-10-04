#!/usr/bin/env python
"""
Check what SAE models are available and how to load them
"""

import sae_lens
from sae_lens import SAE
import requests
import json

print("SAE-Lens version:", sae_lens.__version__)
print("="*70)

# Method 1: Check if there's a registry
print("\n1. Checking for SAE registry...")
try:
    # Try to import various possible registry names
    from sae_lens import pretrained_saes
    print("Found pretrained_saes")
except:
    print("No pretrained_saes module")

try:
    from sae_lens import PRETRAINED_SAES
    print("Found PRETRAINED_SAES")
except:
    print("No PRETRAINED_SAES")

# Method 2: Check common SAE repositories on HuggingFace
print("\n2. Checking HuggingFace for SAE models...")
hf_repos = [
    "EleutherAI/sae-llama-3-8b-32x",
    "EleutherAI/sae-llama-3-8b-64x", 
    "EleutherAI/sae-llama-3.1-8b-32x",
    "EleutherAI/sae-llama-3.1-8b-64x",
    "jbloom/GPT2-Small-SAEs",
    "jbloom/Gemma-2b-IT-SAEs",
    "apollo-research/gemma-2b-res-jb",
]

for repo in hf_repos:
    try:
        url = f"https://huggingface.co/api/models/{repo}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✓ Found: {repo}")
            # Try to get files
            files_url = f"https://huggingface.co/api/models/{repo}/tree/main"
            files_response = requests.get(files_url, timeout=5)
            if files_response.status_code == 200:
                files = files_response.json()
                print(f"  Files: {[f['path'] for f in files[:5]]}...")
        else:
            print(f"✗ Not found: {repo}")
    except Exception as e:
        print(f"✗ Error checking {repo}: {str(e)}")

# Method 3: Try loading with different approaches
print("\n3. Testing SAE loading methods...")

# Test 1: Try gemma SAE (often used in examples)
print("\nTrying to load Gemma-2b SAE...")
try:
    sae = SAE.from_pretrained(
        release="gemma-2b-res-jb",
        sae_id="blocks.12.hook_resid_post",
        device="cpu"
    )
    print("✓ Successfully loaded Gemma SAE!")
    print(f"  SAE features: {sae.cfg.d_sae}")
except Exception as e:
    print(f"✗ Failed: {str(e)}")

# Test 2: Try different naming conventions
print("\nTrying different SAE ID formats for Llama...")
sae_id_formats = [
    "layer_15",
    "layers.15",
    "blocks.15.hook_mlp_out",
    "model.layers.15.mlp",
    "transformer.h.15.mlp",
]

for sae_id in sae_id_formats:
    try:
        print(f"\nTrying: release='EleutherAI/sae-llama-3.1-8b-32x', sae_id='{sae_id}'")
        sae = SAE.from_pretrained(
            release="EleutherAI/sae-llama-3.1-8b-32x",
            sae_id=sae_id,
            device="cpu"
        )
        print(f"✓ Success with {sae_id}!")
        break
    except Exception as e:
        print(f"✗ Failed: {str(e)[:100]}...")

# Method 4: Check sae-lens source for available models
print("\n4. Checking sae-lens source for model registry...")
try:
    import inspect
    import sae_lens.loading
    print("Found loading module:", sae_lens.loading.__file__)
    
    # Look for pretrained model configurations
    for name, obj in inspect.getmembers(sae_lens.loading):
        if "pretrained" in name.lower() or "registry" in name.lower():
            print(f"Found: {name}")
except Exception as e:
    print(f"Error: {e}")