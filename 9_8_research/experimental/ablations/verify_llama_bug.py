#!/usr/bin/env python
"""
Verify Llama-3.1-8B-Instruct bug with exact format from our successful ablation
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

print("🔍 Verifying Llama-3.1-8B-Instruct Decimal Bug")
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

def generate(prompt, max_new_tokens=50, temperature=0.2):
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

# Test different prompt formats
test_prompts = [
    # The EXACT format from our ablation work
    {
        "name": "Ablation format (raw)",
        "prompt": "<|start_header_id|>user<|end_header_id|>\n\nWhich is bigger: 9.8 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    },
    # Using tokenizer's chat template
    {
        "name": "Chat template",
        "prompt": None,  # Will use chat template
        "messages": [{"role": "user", "content": "Which is bigger: 9.8 or 9.11?"}]
    },
    # Simple format
    {
        "name": "Simple format", 
        "prompt": "Which is bigger: 9.8 or 9.11?\nAnswer:"
    },
    # Q&A format
    {
        "name": "Q&A format",
        "prompt": "Q: Which is bigger: 9.8 or 9.11?\nA:"
    }
]

print("\nTesting different prompt formats:\n")

for test in test_prompts:
    print(f"📝 {test['name']}:")
    
    if test['prompt'] is None:
        # Use chat template
        prompt = tokenizer.apply_chat_template(test['messages'], tokenize=False, add_generation_prompt=True)
    else:
        prompt = test['prompt']
    
    print(f"Prompt: {repr(prompt[:80])}...")
    
    correct = 0
    incorrect = 0
    
    # Test 10 times
    for i in range(10):
        response = generate(prompt, temperature=0.2)
        
        # Extract just the generated part
        generated = response[len(prompt):]
        
        # Check answer
        if "9.8" in generated and any(w in generated.lower() for w in ["bigger", "larger", "greater"]):
            correct += 1
            symbol = "✓"
        elif "9.11" in generated and any(w in generated.lower() for w in ["bigger", "larger", "greater"]):
            incorrect += 1
            symbol = "✗"
        else:
            symbol = "?"
        
        if i < 3:
            print(f"  {symbol} {generated[:60].strip()}...")
    
    print(f"Results: {correct}/10 correct, {incorrect}/10 wrong ({incorrect*10}% error rate)\n")

print("="*60)
print("💡 Summary:")
print("The bug manifestation depends heavily on the exact prompt format!")
print("This explains the contradictory results across different tests.")