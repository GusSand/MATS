#!/usr/bin/env python3
"""
Test different prompt formats for the counting bug
"""
import torch
from transformer_lens import HookedTransformer
import json

def test_all_formats():
    """Test the counting bug with different prompt formats"""
    print("="*60)
    print("TESTING DIFFERENT PROMPT FORMATS FOR COUNTING BUG")
    print("="*60)
    
    # Load config
    with open('experiment_config.json', 'r') as f:
        config = json.load(f)
    
    prompt = "How many times does the letter 'r' appear in 'strawberry'?"
    correct_answer = "3"
    
    # Different prompt formats to test
    formats = {
        "chat_format": f"<|start_header_id|>user<|end_header_id|>\n{prompt}\n<|start_header_id|>assistant<|end_header_id|>",
        "qa_format": f"Q: {prompt}\nA:",
        "plain": prompt,
        "instruction": f"Please answer the following question:\n{prompt}\nAnswer:",
        "system_user": f"<|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant.<|start_header_id|>user<|end_header_id|>\n{prompt}\n<|start_header_id|>assistant<|end_header_id|>"
    }
    
    # Load model
    print("\nLoading model...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    model = HookedTransformer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        device=device,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True
    )
    
    # Enable attention result hooks
    model.cfg.use_attn_result = True
    model.setup()
    
    print(f"Model loaded on {device}")
    
    # Test each format
    results = {}
    for format_name, formatted_prompt in formats.items():
        print("\n" + "="*60)
        print(f"Testing: {format_name}")
        print("="*60)
        print(f"Prompt:\n{formatted_prompt}\n")
        print("-"*40)
        
        # Generate response
        output = model.generate(formatted_prompt, max_new_tokens=30, temperature=0)
        
        # Extract just the new generated text (after the prompt)
        if formatted_prompt in output:
            response = output[len(formatted_prompt):].strip()
        else:
            response = output.strip()
        
        print(f"Full output:\n{output}\n")
        print(f"Extracted response:\n{response}\n")
        
        # Check if it contains the correct answer
        has_correct = "3" in response
        has_three_word = "three" in response.lower()
        
        # Look for common wrong answers
        has_two = "2" in response or "two" in response.lower()
        
        results[format_name] = {
            "response": response,
            "has_correct_number": has_correct,
            "has_three_word": has_three_word,
            "has_wrong_answer": has_two,
            "length": len(response.split())
        }
        
        print(f"Contains '3': {has_correct}")
        print(f"Contains 'three': {has_three_word}")
        print(f"Contains wrong answer (2/two): {has_two}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY OF ALL FORMATS")
    print("="*60)
    
    for format_name, result in results.items():
        print(f"\n{format_name}:")
        print(f"  - Correct: {result['has_correct_number'] or result['has_three_word']}")
        print(f"  - Wrong: {result['has_wrong_answer']}")
        print(f"  - Response length: {result['length']} words")
        print(f"  - First 50 chars: {result['response'][:50]}...")
    
    # Test which format is most likely to produce the bug
    print("\n" + "="*60)
    print("FORMATS THAT SHOW THE BUG:")
    bug_formats = [name for name, res in results.items() 
                   if res['has_wrong_answer'] or (not res['has_correct_number'] and not res['has_three_word'])]
    if bug_formats:
        print(f"Formats with bug: {', '.join(bug_formats)}")
    else:
        print("None of the formats showed the counting bug!")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return results

if __name__ == "__main__":
    results = test_all_formats()