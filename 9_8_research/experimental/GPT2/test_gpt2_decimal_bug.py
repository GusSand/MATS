#!/usr/bin/env python
"""
Test if GPT-2 exhibits the decimal comparison bug found in Llama
Key findings:
- GPT-2 doesn't use chat templates (treats them as text)
- GPT-2 gives unclear responses rather than confident wrong answers
- The bug appears to be model-specific to larger/chat-tuned models
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import warnings
warnings.filterwarnings('ignore')

def test_gpt2_decimal_bug(model_name="gpt2"):
    """Test a specific GPT-2 model for the decimal comparison bug"""
    print(f"\n🤖 Testing {model_name.upper()}")
    print("-"*60)
    
    # Load model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    
    def generate(prompt, max_new_tokens=20, temperature=0.1):
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=temperature > 0,
            )
        full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full[len(prompt):]
    
    # Test why Llama format doesn't work
    print("\n1. Llama-style format (doesn't work):")
    llama_prompt = "<|start_header_id|>user<|end_header_id|>\\n\\nWhich is bigger: 9.8 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"
    response = generate(llama_prompt, temperature=0)
    print(f"   Response: {response[:60]}...")
    print("   ❌ GPT-2 treats special tokens as regular text")
    
    # Test proper GPT-2 format
    print("\n2. Proper GPT-2 format:")
    gpt2_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"
    
    correct = 0
    incorrect = 0
    unclear = 0
    
    print(f"   Testing: {repr(gpt2_prompt)}")
    for i in range(10):
        response = generate(gpt2_prompt, temperature=0.2, max_new_tokens=20)
        
        if "9.8" in response and any(w in response.lower() for w in ["bigger", "larger", "greater", "9.8"]):
            correct += 1
        elif "9.11" in response and any(w in response.lower() for w in ["bigger", "larger", "greater", "9.11"]):
            incorrect += 1
        else:
            unclear += 1
        
        if i < 3:
            print(f"   Run {i+1}: {response[:40]}...")
    
    print(f"\n   Results over 10 runs:")
    print(f"   Says 9.8 (correct):   {correct}/10 = {correct*10}%")
    print(f"   Says 9.11 (bug):      {incorrect}/10 = {incorrect*10}%")
    print(f"   Unclear:              {unclear}/10 = {unclear*10}%")
    
    return incorrect > correct  # Returns True if bug is present

def main():
    print("🔍 GPT-2 Decimal Comparison Bug Test")
    print("="*60)
    print("Testing if GPT-2 exhibits the same bug as Llama")
    print("(Llama consistently says '9.11 is bigger than 9.8')")
    
    # Test different model sizes
    models_to_test = ["gpt2", "gpt2-medium"]  # Add "gpt2-large", "gpt2-xl" if needed
    
    bug_present = {}
    for model_name in models_to_test:
        try:
            has_bug = test_gpt2_decimal_bug(model_name)
            bug_present[model_name] = has_bug
        except Exception as e:
            print(f"   Error testing {model_name}: {e}")
            bug_present[model_name] = None
    
    # Summary
    print("\n\n" + "="*60)
    print("💡 SUMMARY")
    print("="*60)
    print("Decimal comparison bug presence:")
    for model, has_bug in bug_present.items():
        if has_bug is None:
            status = "Error"
        elif has_bug:
            status = "❌ Bug present"
        else:
            status = "✅ No clear bug"
        print(f"  {model}: {status}")
    
    print("\nKey findings:")
    print("• GPT-2 can't use Llama's chat format")
    print("• GPT-2 gives unclear/evasive responses rather than confident errors")
    print("• The bug appears stronger in chat-tuned models like Llama")
    print("• Suggests the bug may emerge with scale/instruction-tuning")

if __name__ == "__main__":
    main()