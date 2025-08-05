#!/usr/bin/env python
"""
Check Llama-Scope SAEs and load them for our target layers
"""

import requests
import json
from huggingface_hub import list_repo_files

print("🔍 Checking Llama-Scope SAEs")
print("="*70)

repo_id = "fnlp/Llama-Scope"

# List all files in the repository
print(f"\nListing files in {repo_id}...")
try:
    files = list_repo_files(repo_id)
    
    # Filter for Llama 3.1 8B files
    llama_8b_files = [f for f in files if "Llama-3.1-8B" in f or "llama-3.1-8b" in f]
    
    if llama_8b_files:
        print(f"\nFound {len(llama_8b_files)} files for Llama 3.1 8B:")
        # Group by layer
        layers = {}
        for f in llama_8b_files[:50]:  # First 50 files
            parts = f.split('/')
            if len(parts) > 1 and 'layer' in parts[1]:
                layer = parts[1]
                if layer not in layers:
                    layers[layer] = []
                layers[layer].append(f)
        
        print(f"\nAvailable layers:")
        for layer in sorted(layers.keys()):
            print(f"  {layer}: {len(layers[layer])} files")
    else:
        # Try different patterns
        print("\nTrying alternative patterns...")
        for pattern in ["Llama", "llama", "8B", "8b"]:
            matching = [f for f in files if pattern in f]
            if matching:
                print(f"\nFiles containing '{pattern}': {len(matching)}")
                print("Examples:")
                for f in matching[:10]:
                    print(f"  {f}")
                break
                
except Exception as e:
    print(f"Error: {e}")

# Check the repository structure more broadly
print("\n" + "="*70)
print("Checking repository structure...")

try:
    # Get top-level directories
    all_files = list_repo_files(repo_id)
    top_dirs = set()
    for f in all_files:
        parts = f.split('/')
        if len(parts) > 0:
            top_dirs.add(parts[0])
    
    print("\nTop-level directories:")
    for d in sorted(top_dirs)[:20]:
        print(f"  {d}")
        
    # Look for anything related to our model
    print("\nSearching for 8B model SAEs...")
    relevant_dirs = [d for d in top_dirs if '8' in d or 'llama' in d.lower()]
    if relevant_dirs:
        print(f"Potentially relevant directories:")
        for d in relevant_dirs:
            print(f"  {d}")
            # Check what's inside
            sub_files = [f for f in all_files if f.startswith(d + '/')]
            if sub_files:
                print(f"    Example files: {sub_files[0]}")
                
except Exception as e:
    print(f"Error exploring repository: {e}")