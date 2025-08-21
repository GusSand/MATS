#!/usr/bin/env python3
"""
Minimal test to verify setup works
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

def minimal_test():
    """Run minimal test to verify everything works"""
    
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()
    
    print("Model loaded!")
    
    # Test basic generation
    print("\n1. Testing basic bug reproduction...")
    
    simple_prompt = "Which is bigger: 9.8 or 9.11?\nAnswer:"
    qa_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"
    
    for prompt_name, prompt in [("Simple", simple_prompt), ("Q&A", qa_prompt)]:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated[len(prompt):][:50]
        
        print(f"{prompt_name} format: {answer}")
        
        if "9.8 is bigger" in answer.lower() or "9.8 is larger" in answer.lower():
            print(f"  → Correct (9.8 > 9.11)")
        elif "9.11 is bigger" in answer.lower() or "9.11 is larger" in answer.lower():
            print(f"  → Bug (says 9.11 > 9.8)")
        else:
            print(f"  → Unclear")
    
    # Test attention patching
    print("\n2. Testing Layer 10 attention patching...")
    
    saved_activation = None
    
    def save_hook(module, input, output):
        nonlocal saved_activation
        if isinstance(output, tuple):
            saved_activation = output[0].detach().clone()
        else:
            saved_activation = output.detach().clone()
        return output
    
    def patch_hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        
        new_hidden = hidden.clone()
        min_len = min(hidden.shape[1], saved_activation.shape[1])
        new_hidden[:, :min_len, :] = saved_activation[:, :min_len, :]
        
        if isinstance(output, tuple):
            return (new_hidden,) + output[1:]
        return new_hidden
    
    # Save correct activation
    layer_10 = model.model.layers[10].self_attn
    hook = layer_10.register_forward_hook(save_hook)
    
    inputs = tokenizer(simple_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model(**inputs)
    
    hook.remove()
    print("  Saved correct format activation")
    
    # Patch during buggy generation
    hook = layer_10.register_forward_hook(patch_hook)
    
    inputs = tokenizer(qa_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    
    hook.remove()
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated[len(qa_prompt):][:50]
    
    print(f"  Q&A with patching: {answer}")
    
    if "9.8 is bigger" in answer.lower() or "9.8 is larger" in answer.lower():
        print(f"  → ✅ SUCCESS! Bug fixed by patching")
    elif "9.11 is bigger" in answer.lower() or "9.11 is larger" in answer.lower():
        print(f"  → ❌ FAILED - Still shows bug")
    else:
        print(f"  → ❓ Unclear result")
    
    print("\n✅ Minimal test complete!")
    print("\nIf both tests passed:")
    print("1. Simple format should be CORRECT")
    print("2. Q&A format should show BUG")
    print("3. Q&A with patching should be CORRECT")

if __name__ == "__main__":
    minimal_test()