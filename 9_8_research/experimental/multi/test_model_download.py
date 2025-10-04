#!/usr/bin/env python3
"""
Test script to verify we can download and load the Llama model
"""

import torch
from transformer_lens import HookedTransformer
import os
import sys
from datetime import datetime

print("="*60)
print("TESTING MODEL DOWNLOAD AND PERMISSIONS")
print(f"Start time: {datetime.now()}")
print("="*60)

# Check GPU availability
print("\n1. Checking GPU availability...")
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"✓ Found {gpu_count} GPU(s)")
    for i in range(gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        mem_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  Memory: {mem_gb:.1f} GB")
else:
    print("✗ No GPUs available!")
    sys.exit(1)

# Check Hugging Face cache directory
print("\n2. Checking HuggingFace cache permissions...")
hf_cache = os.path.expanduser("~/.cache/huggingface")
if os.path.exists(hf_cache):
    print(f"✓ HF cache exists: {hf_cache}")
    # Check write permissions
    test_file = os.path.join(hf_cache, "test_write.tmp")
    try:
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print("✓ Write permissions OK")
    except Exception as e:
        print(f"✗ No write permissions: {e}")
else:
    print(f"Creating HF cache directory: {hf_cache}")
    try:
        os.makedirs(hf_cache, exist_ok=True)
        print("✓ Cache directory created")
    except Exception as e:
        print(f"✗ Failed to create cache: {e}")
        sys.exit(1)

# Try to load the model
print("\n3. Attempting to load Llama-3.1-8B-Instruct...")
print("This may take 10-20 minutes if downloading for the first time...")

try:
    model = HookedTransformer.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        device="cuda:0",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    print("✓ Model loaded successfully!")
    
    # Test a simple generation
    print("\n4. Testing model generation...")
    test_prompt = "The capital of France is"
    output = model.generate(test_prompt, max_new_tokens=5, temperature=0)
    result = model.to_string(output[0])
    print(f"Test output: {result}")
    print("✓ Generation successful!")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED - Ready to run experiments!")
    print("="*60)
    
except Exception as e:
    print(f"\n✗ Failed to load model: {e}")
    print("\nPossible issues:")
    print("1. Need to authenticate with Hugging Face (use `huggingface-cli login`)")
    print("2. Need to accept Llama license agreement at:")
    print("   https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct")
    print("3. Insufficient disk space (need ~16GB)")
    print("4. Network connectivity issues")
    sys.exit(1)