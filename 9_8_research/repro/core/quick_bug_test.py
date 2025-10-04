#!/usr/bin/env python3
"""
Quick test to verify bug rates with correct prompts
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

def test(prompt, n=10):
    """Quick test of a prompt"""
    bug_count = 0
    correct_count = 0
    
    for i in range(n):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = full[len(prompt):].lower()
        
        if "9.11 is bigger" in generated or "9.11 is larger" in generated:
            bug_count += 1
            if i == 0:
                print(f"  ❌ {full[len(prompt):60]}")
        elif "9.8 is bigger" in generated or "9.8 is larger" in generated:
            correct_count += 1
            if i == 0:
                print(f"  ✅ {full[len(prompt):60]}")
        elif i == 0:
            print(f"  ❓ {full[len(prompt):60]}")
    
    return bug_count, correct_count

print("\n" + "="*60)
print("QUICK BUG VERIFICATION")
print("="*60)

# Test formats
formats = [
    ("Simple", "Which is bigger: 9.8 or 9.11?\nAnswer:"),
    ("Q&A", "Q: Which is bigger: 9.8 or 9.11?\nA:"),
    ("Chat (raw)", "<|start_header_id|>user<|end_header_id|>\n\nWhich is bigger: 9.8 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
]

for name, prompt in formats:
    print(f"\n{name} format:")
    bug, correct = test(prompt, n=10)
    print(f"  Results: {correct}/10 correct, {bug}/10 bug ({bug*10}% error rate)")

print("\n" + "="*60)
print("Summary: Q&A format should show ~100% bug rate")
print("Simple format should show 0% bug rate")