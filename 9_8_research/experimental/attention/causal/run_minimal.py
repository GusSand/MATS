"""
Minimal test to verify the decimal comparison bug with proper detection
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For better error reporting

# Clear cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="cuda:0",
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
model.eval()

# Test formats
test_cases = [
    ("Simple", "Which is bigger: 9.8 or 9.11?\nAnswer:"),
    ("Q&A", "Q: Which is bigger: 9.8 or 9.11?\nA:"),
]

print("\nTesting formats:")
print("="*50)

for format_name, prompt in test_cases:
    print(f"\n{format_name} format:")
    print(f"Prompt: {repr(prompt)}")
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():  # Use mixed precision
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                temperature=None,  # Explicitly set to None
                top_p=None,  # Explicitly set to None
                pad_token_id=tokenizer.pad_token_id
            )
    
    # Decode
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = full_response[len(prompt):].strip()
    
    print(f"Response: {generated}")
    
    # Check for bug
    generated_lower = generated.lower()
    
    # Simple detection
    says_9_8_bigger = "9.8 is" in generated_lower and any(w in generated_lower for w in ["bigger", "larger", "greater"])
    says_9_11_bigger = "9.11 is" in generated_lower and any(w in generated_lower for w in ["bigger", "larger", "greater"])
    
    if says_9_8_bigger and not says_9_11_bigger:
        print("✅ CORRECT: Says 9.8 is bigger")
    elif says_9_11_bigger and not says_9_8_bigger:
        print("❌ BUG: Says 9.11 is bigger")
    else:
        print("❓ Unclear response")
    
    # Clear cache after each generation
    torch.cuda.empty_cache()

print("\n" + "="*50)
print("Done!")