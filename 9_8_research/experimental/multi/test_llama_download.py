#!/usr/bin/env python3
"""
Test Llama model download using transformers directly
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
from datetime import datetime

print("="*60)
print("TESTING LLAMA MODEL DOWNLOAD")
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

# Check HF token
print("\n2. Checking Hugging Face authentication...")
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if hf_token:
    print(f"✓ Found HF token (length: {len(hf_token)})")
else:
    print("⚠️  No HF token found. You may need to:")
    print("   1. Run: huggingface-cli login")
    print("   2. Or set HF_TOKEN environment variable")

# Try to download tokenizer first (smaller)
print("\n3. Testing tokenizer download...")
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("✓ Tokenizer downloaded successfully!")
except Exception as e:
    print(f"✗ Failed to download tokenizer: {e}")
    print("\nYou likely need to:")
    print("1. Authenticate with: huggingface-cli login")
    print("2. Accept the license at: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct")
    sys.exit(1)

# Now try the model
print("\n4. Testing model download (this may take 10-20 minutes)...")
print("   Model size: ~16GB")

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    print("✓ Model downloaded and loaded successfully!")
    
    # Test generation
    print("\n5. Testing generation...")
    inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5, temperature=0.01)
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   Output: {result}")
    print("✓ Generation successful!")
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("  Model is ready for experiments")
    print("="*60)
    
except Exception as e:
    print(f"✗ Failed to download/load model: {e}")
    print("\nPossible issues:")
    print("1. Not authenticated (run: huggingface-cli login)")
    print("2. License not accepted")
    print("3. Insufficient disk space")
    print("4. Network issues")
    sys.exit(1)