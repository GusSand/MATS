#!/usr/bin/env python3
"""
Search for simpler bugs that might be fixable with sparse editing
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("SEARCHING FOR FIXABLE BUGS IN LLAMA-3.1-8B-INSTRUCT")
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

def test_bug(description, prompts_and_expected):
    """Test if a bug exists consistently"""
    print(f"\n--- Testing: {description} ---")
    
    bug_count = 0
    total = len(prompts_and_expected)
    
    for prompt_type, prompt, expected_contains, bug_contains in prompts_and_expected:
        # Test with chat template
        if prompt_type == "chat":
            messages = [{"role": "user", "content": prompt}]
            full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            full_prompt = prompt
        
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = response[len(full_prompt):].lower()
        
        # Check for bug
        has_expected = any(exp in generated for exp in expected_contains)
        has_bug = any(bug in generated for bug in bug_contains)
        
        if has_bug and not has_expected:
            bug_count += 1
            print(f"  ❌ BUG in {prompt_type}: {generated[:50]}")
        elif has_expected:
            print(f"  ✓ Correct in {prompt_type}: {generated[:50]}")
        else:
            print(f"  ? Unclear in {prompt_type}: {generated[:50]}")
    
    bug_rate = bug_count / total * 100
    print(f"  Bug rate: {bug_rate:.0f}% ({bug_count}/{total})")
    
    return bug_rate > 50  # Consider it a consistent bug if >50% failure

# Test various potential bugs
bugs_to_test = []

# 1. Capitalization bug
bugs_to_test.append(("Capitalization of countries", [
    ("chat", "What is the capital of France?", ["paris"], ["france", "country"]),
    ("chat", "What is the capital of Japan?", ["tokyo"], ["japan", "country"]),
    ("chat", "What is the capital of Germany?", ["berlin"], ["germany", "country"]),
    ("simple", "Q: What is the capital of France?\nA:", ["paris"], ["france", "country"]),
]))

# 2. Basic arithmetic bugs
bugs_to_test.append(("Simple addition", [
    ("chat", "What is 2 + 2?", ["4", "four"], ["5", "6", "3"]),
    ("chat", "What is 5 + 3?", ["8", "eight"], ["7", "9", "6"]),
    ("simple", "Q: What is 2 + 2?\nA:", ["4", "four"], ["5", "6", "3"]),
]))

# 3. Color identification
bugs_to_test.append(("Color of objects", [
    ("chat", "What color is the sky?", ["blue"], ["red", "green", "yellow"]),
    ("chat", "What color is grass?", ["green"], ["blue", "red", "yellow"]),
    ("chat", "What color is a banana?", ["yellow"], ["blue", "red", "green"]),
    ("simple", "Q: What color is the sky?\nA:", ["blue"], ["red", "green", "yellow"]),
]))

# 4. Plural/singular confusion
bugs_to_test.append(("Plural forms", [
    ("chat", "What is the plural of 'mouse'?", ["mice"], ["mouses", "mouse"]),
    ("chat", "What is the plural of 'child'?", ["children"], ["childs", "child"]),
    ("chat", "What is the plural of 'goose'?", ["geese"], ["gooses", "goose"]),
    ("simple", "Q: What is the plural of 'mouse'?\nA:", ["mice"], ["mouses", "mouse"]),
]))

# 5. Days of the week
bugs_to_test.append(("Day after", [
    ("chat", "What day comes after Monday?", ["tuesday"], ["wednesday", "sunday"]),
    ("chat", "What day comes after Friday?", ["saturday"], ["sunday", "thursday"]),
    ("simple", "Q: What day comes after Monday?\nA:", ["tuesday"], ["wednesday", "sunday"]),
]))

# 6. Opposite words
bugs_to_test.append(("Opposites", [
    ("chat", "What is the opposite of 'hot'?", ["cold"], ["warm", "heat"]),
    ("chat", "What is the opposite of 'big'?", ["small", "little"], ["large", "huge"]),
    ("chat", "What is the opposite of 'up'?", ["down"], ["above", "over"]),
    ("simple", "Q: What is the opposite of 'hot'?\nA:", ["cold"], ["warm", "heat"]),
]))

# 7. Simple factual errors
bugs_to_test.append(("Number of states", [
    ("chat", "How many states are in the USA?", ["50", "fifty"], ["51", "49", "52"]),
    ("chat", "How many continents are there?", ["7", "seven"], ["6", "8", "5"]),
    ("simple", "Q: How many states are in the USA?\nA:", ["50", "fifty"], ["51", "49", "52"]),
]))

# 8. Gender pronouns
bugs_to_test.append(("Pronoun agreement", [
    ("chat", "Mary went to the store. What is her pronoun?", ["she", "her"], ["he", "his", "him"]),
    ("chat", "John went to the store. What is his pronoun?", ["he", "his", "him"], ["she", "her"]),
    ("simple", "Mary went to the store. What is her pronoun?\nAnswer:", ["she", "her"], ["he", "his", "him"]),
]))

# Test each potential bug
found_bugs = []
for bug_name, test_cases in bugs_to_test:
    if test_bug(bug_name, test_cases):
        found_bugs.append(bug_name)
        print(f"  → FOUND CONSISTENT BUG: {bug_name}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
if found_bugs:
    print(f"Found {len(found_bugs)} potential bugs to test:")
    for bug in found_bugs:
        print(f"  • {bug}")
else:
    print("No consistent bugs found in basic tests.")
    print("\nTrying format-specific bugs...")
    
    # Test format-specific differences
    print("\n--- Testing format-specific behaviors ---")
    
    # Test if there's a difference in verbosity
    test_prompts = [
        "Explain what water is in one word",
        "Name the color of snow in one word",
        "What is 1+1? Answer in one word",
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        
        # Chat template
        messages = [{"role": "user", "content": prompt}]
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=30, temperature=0.1, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        chat_response = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(chat_prompt):]
        
        # Simple format
        simple_prompt = f"{prompt}\nAnswer:"
        inputs = tokenizer(simple_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=30, temperature=0.1, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        simple_response = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(simple_prompt):]
        
        chat_words = len(chat_response.split())
        simple_words = len(simple_response.split())
        
        print(f"  Chat ({chat_words} words): {chat_response[:50]}")
        print(f"  Simple ({simple_words} words): {simple_response[:50]}")
        
        if chat_words > simple_words * 2:
            print(f"  → VERBOSITY BUG: Chat is {chat_words/simple_words:.1f}x more verbose!")

print("\n" + "="*70)
print("SEARCH COMPLETE")
print("="*70)