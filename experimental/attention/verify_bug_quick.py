#!/usr/bin/env python
"""
Quick verification of bug behavior with exact conditions
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

print("🔍 Quick Bug Verification")
print("="*60)

# Load model - EXACT same as verify_llama_bug.py
model_path = "meta-llama/Llama-3.1-8B-Instruct"
print(f"Loading {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

def generate(prompt, max_new_tokens=50, temperature=0.0):
    """EXACT same generation as verify_llama_bug.py"""
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=temperature > 0,  # False when temp=0
        )
    
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full

# Test the EXACT prompts
print("\n1. Chat template (should show bug):")
messages = [{"role": "user", "content": "Which is bigger: 9.8 or 9.11?"}]
chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"Prompt preview: {repr(chat_prompt[:100])}...")

# Test 10 times
for i in range(10):
    response = generate(chat_prompt, temperature=0.0)
    generated = response[len(chat_prompt):]
    
    # EXACT same checking logic
    generated_lower = generated.lower()
    says_9_8_bigger = (
        ("9.8" in generated and any(w in generated_lower for w in ["bigger", "larger", "greater"])) or
        ("bigger than 9.11" in generated_lower) or
        ("larger than 9.11" in generated_lower) or
        ("greater than 9.11" in generated_lower)
    )
    says_9_11_bigger = (
        ("9.11" in generated and any(w in generated_lower for w in ["bigger", "larger", "greater"])) or
        ("bigger than 9.8" in generated_lower) or
        ("larger than 9.8" in generated_lower) or
        ("greater than 9.8" in generated_lower)
    )
    
    # Disambiguation if both detected
    if says_9_8_bigger and says_9_11_bigger:
        words = generated_lower.split()
        for j, word in enumerate(words):
            if word in ["9.8", "9.11"] and j + 1 < len(words):
                if words[j + 1] in ["is", "are"] and j + 2 < len(words):
                    if words[j + 2] in ["bigger", "larger", "greater"]:
                        if word == "9.8":
                            says_9_8_bigger = True
                            says_9_11_bigger = False
                        else:
                            says_9_8_bigger = False
                            says_9_11_bigger = True
                        break
    
    if says_9_8_bigger and not says_9_11_bigger:
        symbol = "✅ CORRECT"
    elif says_9_11_bigger and not says_9_8_bigger:
        symbol = "❌ BUG"
    else:
        symbol = "❓ UNCLEAR"
    
    print(f"  {i+1}: {symbol} - {generated[:60].strip()}")

print("\n2. Simple format (should be correct):")
simple_prompt = "Which is bigger: 9.8 or 9.11?\nAnswer:"
print(f"Prompt: {repr(simple_prompt)}")

for i in range(10):
    response = generate(simple_prompt, temperature=0.0)
    generated = response[len(simple_prompt):]
    
    # Same checking logic
    generated_lower = generated.lower()
    says_9_8_bigger = (
        ("9.8" in generated and any(w in generated_lower for w in ["bigger", "larger", "greater"])) or
        ("bigger than 9.11" in generated_lower) or
        ("larger than 9.11" in generated_lower) or
        ("greater than 9.11" in generated_lower)
    )
    says_9_11_bigger = (
        ("9.11" in generated and any(w in generated_lower for w in ["bigger", "larger", "greater"])) or
        ("bigger than 9.8" in generated_lower) or
        ("larger than 9.8" in generated_lower) or
        ("greater than 9.8" in generated_lower)
    )
    
    if says_9_8_bigger and says_9_11_bigger:
        words = generated_lower.split()
        for j, word in enumerate(words):
            if word in ["9.8", "9.11"] and j + 1 < len(words):
                if words[j + 1] in ["is", "are"] and j + 2 < len(words):
                    if words[j + 2] in ["bigger", "larger", "greater"]:
                        if word == "9.8":
                            says_9_8_bigger = True
                            says_9_11_bigger = False
                        else:
                            says_9_8_bigger = False
                            says_9_11_bigger = True
                        break
    
    if says_9_8_bigger and not says_9_11_bigger:
        symbol = "✅ CORRECT"
    elif says_9_11_bigger and not says_9_8_bigger:
        symbol = "❌ BUG"
    else:
        symbol = "❓ UNCLEAR"
    
    print(f"  {i+1}: {symbol} - {generated[:60].strip()}")

print("\n" + "="*60)
print("Note: Temperature=0.0 should give deterministic results")
print("All instances of same prompt should give same answer")