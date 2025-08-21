#!/usr/bin/env python3
"""
Test empty response bug in chat format with constraints
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("TESTING EMPTY RESPONSE BUG IN CHAT FORMAT")
print("="*70)

# Load model
model_name = "meta-llama/Llama-3.1-8B-Instruct"
print(f"\nLoading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

def test_empty_response_bug():
    """Test if model produces empty responses in chat format with constraints"""
    
    test_cases = [
        ("What is water?", "One word answer only."),
        ("What color is the sky?", "Answer with just one word."),
        ("Define democracy", "Maximum 10 words."),
        ("Explain gravity", "Use exactly three words."),
        ("What is 2+2?", "Answer with just the number."),
        ("Name a fruit", "One word only."),
        ("What is fire?", "Single word answer."),
        ("Define love", "Three words maximum."),
    ]
    
    empty_count = 0
    total_tests = len(test_cases)
    
    print("\nTesting empty response pattern:")
    print("-" * 50)
    
    for question, constraint in test_cases:
        # Chat format with constraint
        messages = [{"role": "user", "content": f"{question} {constraint}"}]
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Simple format with constraint
        simple_prompt = f"{question} {constraint}\nAnswer:"
        
        results = {}
        
        # Generate with chat format
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        chat_generated = response[len(chat_prompt):].strip()
        
        # Generate with simple format
        inputs = tokenizer(simple_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        simple_generated = response[len(simple_prompt):].strip()
        
        # Check for empty response bug
        chat_empty = len(chat_generated) == 0
        simple_empty = len(simple_generated) == 0
        
        if chat_empty and not simple_empty:
            empty_count += 1
            print(f"✓ BUG FOUND: {question[:30]}...")
            print(f"  Chat (empty): '{chat_generated}'")
            print(f"  Simple: '{simple_generated[:50]}...'")
        else:
            print(f"✗ No bug: {question[:30]}...")
            if chat_generated:
                print(f"  Chat: '{chat_generated[:50]}...'")
    
    bug_rate = (empty_count / total_tests) * 100
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Empty response bug rate: {bug_rate:.0f}% ({empty_count}/{total_tests})")
    
    if bug_rate > 50:
        print("\n🎯 FOUND CONSISTENT BUG: Empty responses in chat format with constraints!")
        print("\nBug characteristics:")
        print("• Chat format with strict constraints → empty response")
        print("• Simple format with same constraints → proper response")
        print("• This is likely due to over-cautious safety filtering")
        print("\nThis bug is a good candidate for sparse editing because:")
        print("1. Clear format-specific behavior difference")
        print("2. Not a deep reasoning issue")
        print("3. Likely controlled by specific safety/formatting neurons")
        print("4. Can be validated easily (response vs no response)")
        
        return True
    else:
        print("\nNo consistent empty response bug found.")
        return False

# Run the test
if test_empty_response_bug():
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print("Use sparse editing to target neurons that cause empty responses in chat format.")
    print("Goal: Allow chat format to generate responses like simple format does.")
    
    # Test which layers might be involved
    print("\n" + "="*70)
    print("HYPOTHESIS")
    print("="*70)
    print("This bug likely involves:")
    print("• Early layers (0-5): Format detection")
    print("• Middle layers (10-20): Safety/constraint processing")
    print("• Late layers (25-31): Response generation gating")
    print("\nSuggested approach:")
    print("1. Collect activations for both formats")
    print("2. Find neurons with high differential activation")
    print("3. Use sparse editing to adjust chat format behavior")