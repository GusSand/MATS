#!/usr/bin/env python
"""Test with proper chat template usage"""

import os
import sys
sys.path.extend([
    '/home/paperspace/dev/MATS9',
    '/home/paperspace/dev/MATS9/observatory/lib/neurondb',
    '/home/paperspace/dev/MATS9/observatory/lib/util',
])

from dotenv import load_dotenv
load_dotenv('/home/paperspace/dev/MATS9/.env')

import warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

from transformers import AutoTokenizer
import torch

print("🧪 Testing raw model behavior")
print("="*60)

# Load tokenizer and check chat template
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

# Create a proper chat
messages = [
    {"role": "user", "content": "Which is bigger: 9.8 or 9.11?"}
]

# Apply chat template
chat_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print("Chat template output:")
print(repr(chat_prompt))
print()

# Now test with raw transformers to see baseline behavior
from transformers import AutoModelForCausalLM

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Test multiple times
print("\nTesting baseline model behavior (no ablation):")
print("-" * 40)

for i in range(5):
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"Run {i+1}: {response.strip()}")

print("\n💡 This shows the baseline model behavior without any interventions.")