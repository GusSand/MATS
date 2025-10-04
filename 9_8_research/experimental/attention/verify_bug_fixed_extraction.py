#!/usr/bin/env python
"""
Verify bug with FIXED text extraction
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

print("🔍 Bug Verification with Fixed Extraction")
print("="*60)

# Load model
model_path = "meta-llama/Llama-3.1-8B-Instruct"
print(f"Loading {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

def generate_and_extract(prompt, max_new_tokens=50, temperature=0.0):
    """Generate and properly extract the response"""
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=temperature > 0,
        )
    
    # Decode WITHOUT skipping special tokens first
    full_with_special = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Find where the assistant's response actually starts
    # Look for the last "assistant<|end_header_id|>" pattern
    assistant_marker = "assistant<|end_header_id|>"
    if assistant_marker in full_with_special:
        # Get everything after the assistant marker
        parts = full_with_special.split(assistant_marker)
        generated = parts[-1]  # Get the last part (the actual response)
        
        # Clean up any remaining special tokens
        generated = generated.replace("<|eot_id|>", "")
        generated = generated.replace("<|end_of_text|>", "")
        generated = generated.strip()
        
        return generated
    else:
        # Fallback to simple extraction
        full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full[len(prompt):]

# Test formats
print("\n1. Chat template (should show bug 100% of the time):")
messages = [{"role": "user", "content": "Which is bigger: 9.8 or 9.11?"}]
chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

correct_count = 0
bug_count = 0

for i in range(10):
    generated = generate_and_extract(chat_prompt, temperature=0.0)
    
    # Check the answer
    generated_lower = generated.lower()
    
    if "9.8" in generated and ("bigger" in generated_lower or "larger" in generated_lower or "greater" in generated_lower):
        if "9.11" not in generated or generated.index("9.8") < generated.index("9.11"):
            symbol = "✅ CORRECT"
            correct_count += 1
        else:
            symbol = "❌ BUG"
            bug_count += 1
    elif "9.11" in generated and ("bigger" in generated_lower or "larger" in generated_lower or "greater" in generated_lower):
        symbol = "❌ BUG"
        bug_count += 1
    else:
        symbol = "❓ UNCLEAR"
    
    print(f"  {i+1}: {symbol} - {generated[:60]}")

print(f"\nChat template results: {correct_count} correct, {bug_count} bug ({bug_count*10}% error rate)")

print("\n2. Simple format (should be 100% correct):")
simple_prompt = "Which is bigger: 9.8 or 9.11?\nAnswer:"

correct_count = 0
bug_count = 0

for i in range(10):
    generated = generate_and_extract(simple_prompt, temperature=0.0)
    
    generated_lower = generated.lower()
    
    if "9.8" in generated and ("bigger" in generated_lower or "larger" in generated_lower or "greater" in generated_lower):
        if "9.11" not in generated or generated.index("9.8") < generated.index("9.11"):
            symbol = "✅ CORRECT"
            correct_count += 1
        else:
            symbol = "❌ BUG"
            bug_count += 1
    elif "9.11" in generated and ("bigger" in generated_lower or "larger" in generated_lower or "greater" in generated_lower):
        symbol = "❌ BUG"
        bug_count += 1
    else:
        symbol = "❓ UNCLEAR"
    
    print(f"  {i+1}: {symbol} - {generated[:60]}")

print(f"\nSimple format results: {correct_count} correct, {bug_count} bug ({bug_count*10}% error rate)")

print("\n3. Q&A format (should show bug):")
qa_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"

correct_count = 0
bug_count = 0

for i in range(10):
    generated = generate_and_extract(qa_prompt, temperature=0.0)
    
    generated_lower = generated.lower()
    
    if "9.8" in generated and ("bigger" in generated_lower or "larger" in generated_lower or "greater" in generated_lower):
        if "9.11" not in generated or generated.index("9.8") < generated.index("9.11"):
            symbol = "✅ CORRECT"
            correct_count += 1
        else:
            symbol = "❌ BUG"
            bug_count += 1
    elif "9.11" in generated and ("bigger" in generated_lower or "larger" in generated_lower or "greater" in generated_lower):
        symbol = "❌ BUG"
        bug_count += 1
    else:
        symbol = "❓ UNCLEAR"
    
    print(f"  {i+1}: {symbol} - {generated[:60]}")

print(f"\nQ&A format results: {correct_count} correct, {bug_count} bug ({bug_count*10}% error rate)")

print("\n" + "="*60)
print("Summary:")
print("Temperature=0.0 gives deterministic results")
print("The extraction issue was causing false 'UNCLEAR' results")