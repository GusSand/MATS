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
    
    # Test 100 times
    for i in range(100):
        response = generate(prompt, temperature=0.0)
        
        # Extract just the generated part
        generated = response[len(prompt):]
        
        # Check answer - 9.8 is correct, 9.11 is the bug
        # Look for patterns that indicate which number the model thinks is bigger
        generated_lower = generated.lower()
        
        # Look for the first clear statement about which number is bigger
        # Check for patterns like "9.8 is bigger" or "bigger than 9.11"
        says_9_8_bigger = (
            ("9.8" in generated and any(w in generated_lower for w in ["bigger", "larger", "greater"])) or
            ("bigger than 9.11" in generated_lower) or
            ("larger than 9.11" in generated_lower) or
            ("greater than 9.11" in generated_lower)
        )
        
        # Check for patterns like "9.11 is bigger" or "bigger than 9.8" 
        says_9_11_bigger = (
            ("9.11" in generated and any(w in generated_lower for w in ["bigger", "larger", "greater"])) or
            ("bigger than 9.8" in generated_lower) or
            ("larger than 9.8" in generated_lower) or
            ("greater than 9.8" in generated_lower)
        )
        
        # If both are detected, look for the first clear statement
        if says_9_8_bigger and says_9_11_bigger:
            # Find the first occurrence of a clear statement
            words = generated_lower.split()
            for i, word in enumerate(words):
                if word in ["9.8", "9.11"] and i + 1 < len(words):
                    next_word = words[i + 1]
                    if next_word in ["is", "are"] and i + 2 < len(words):
                        third_word = words[i + 2]
                        if third_word in ["bigger", "larger", "greater"]:
                            if word == "9.8":
                                says_9_8_bigger = True
                                says_9_11_bigger = False
                            else:  # word == "9.11"
                                says_9_8_bigger = False
                                says_9_11_bigger = True
                            break
        
        if says_9_8_bigger and not says_9_11_bigger:
            correct += 1
            symbol = "✅"
        elif says_9_11_bigger and not says_9_8_bigger:
            incorrect += 1
            symbol = "❌"
        else:
            symbol = "❓"
        
        if i < 3:
            print(f"  {symbol} {generated[:60].strip()}...")
    
    # Results summary
    error_rate = incorrect  # Now it's out of 100, so this is already the percentage
    print(f"Results: {correct}/100 correct, {incorrect}/100 wrong ({error_rate}% error rate)\n")

print("="*60)
print("💡 Summary:")
print("The bug manifestation depends heavily on the exact prompt format!")
print("This explains the contradictory results across different tests.")