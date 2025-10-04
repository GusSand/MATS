#!/usr/bin/env python
"""
Debug what chat template is actually generating
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

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

def generate(prompt, max_new_tokens=50, temperature=0.0):
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
    
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full

# Test chat template
messages = [{"role": "user", "content": "Which is bigger: 9.8 or 9.11?"}]
chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("Full chat prompt:")
print(repr(chat_prompt))
print("\n" + "="*60 + "\n")

# Generate with skip_special_tokens=True (default)
response1 = generate(chat_prompt, temperature=0.0)
print("Response with skip_special_tokens=True:")
print(f"Full response: {repr(response1)}")
print(f"Prompt length: {len(chat_prompt)}")
print(f"Response length: {len(response1)}")
print(f"Generated part: {repr(response1[len(chat_prompt):])}")

print("\n" + "="*60 + "\n")

# Generate with skip_special_tokens=False to see everything
inputs = tokenizer(chat_prompt, return_tensors="pt")
if torch.cuda.is_available():
    inputs = {k: v.cuda() for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )

response2 = tokenizer.decode(outputs[0], skip_special_tokens=False)
print("Response with skip_special_tokens=False:")
print(f"Full response: {repr(response2)}")
print(f"Generated part: {repr(response2[len(chat_prompt):])}")

# Also try without system prompt
messages_no_system = [{"role": "user", "content": "Which is bigger: 9.8 or 9.11?"}]
chat_prompt_no_system = tokenizer.apply_chat_template(
    messages_no_system, 
    tokenize=False, 
    add_generation_prompt=True,
    # Try to skip system message
    add_special_tokens=False
)

print("\n" + "="*60 + "\n")
print("Chat prompt (attempting no system):")
print(repr(chat_prompt_no_system))

response3 = generate(chat_prompt_no_system, temperature=0.0)
print(f"\nGenerated: {repr(response3[len(chat_prompt_no_system):])}")