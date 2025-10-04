#!/usr/bin/env python
"""
Final test to demonstrate the true decimal bug pattern
Tests specific number pairs that consistently show the bug
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

def test_decimal_bug(model_name, model_path):
    """Test specific decimal comparisons that show the bug"""
    
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"Model: {model_path}")
    print(f"{'='*60}")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "</s>"
    
    # Test pairs that typically show the bug
    test_pairs = [
        # (num1, num2, correct_answer)
        ("9.8", "9.11", "9.8"),      # Classic case
        ("3.9", "3.11", "3.9"),      # Often fails
        ("5.6", "5.14", "5.6"),      # Often fails
        ("10.8", "10.11", "10.8"),   # Version-like
        ("2.7", "2.10", "2.7"),      # Usually works
        ("1.8", "1.12", "1.8"),      # Usually works
    ]
    
    # Use the Q&A format that works best
    results = {}
    
    for num1, num2, correct in test_pairs:
        prompt = f"Q: Which is bigger: {num1} or {num2}?\nA:"
        
        correct_count = 0
        responses = []
        
        print(f"\nTesting: {num1} vs {num2} (correct: {correct})")
        
        for i in range(5):
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.2,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=True,
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = response[len(prompt):].strip()
            
            # Check if correct answer appears
            if correct in generated and any(w in generated.lower() for w in ["bigger", "larger", "greater", correct]):
                correct_count += 1
                result = "✓"
            else:
                result = "✗"
            
            responses.append((result, generated[:50]))
            
            if i == 0:  # Show first response
                print(f"  Sample: {result} {generated[:60]}...")
        
        accuracy = (correct_count / 5) * 100
        results[f"{num1}_vs_{num2}"] = {
            "accuracy": accuracy,
            "responses": responses
        }
        
        print(f"  Accuracy: {correct_count}/5 = {accuracy}%")
    
    # Summary
    print(f"\n{'='*40}")
    print("SUMMARY")
    print(f"{'='*40}")
    
    total_accuracy = sum(r["accuracy"] for r in results.values()) / len(results)
    print(f"Overall accuracy: {total_accuracy:.1f}%")
    
    print("\nPer-comparison accuracy:")
    for comparison, data in results.items():
        print(f"  {comparison}: {data['accuracy']:.0f}%")
    
    # Identify bug pattern
    bug_comparisons = [k for k, v in results.items() if v["accuracy"] < 50]
    if bug_comparisons:
        print(f"\nBug detected in: {', '.join(bug_comparisons)}")
    else:
        print("\nNo clear bug pattern detected")
    
    # Clean up
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results

def main():
    print("="*80)
    print("FINAL DECIMAL BUG TEST")
    print("Testing specific number pairs that reveal the bug pattern")
    print("="*80)
    
    # Test key models
    models = [
        ("Gemma-2-2B Base", "google/gemma-2-2b"),
        ("Gemma-2-2B-IT", "google/gemma-2-2b-it"),
        ("Pythia-160M", "EleutherAI/pythia-160m"),  # Known to have bug
    ]
    
    all_results = {}
    
    for model_name, model_path in models:
        results = test_decimal_bug(model_name, model_path)
        all_results[model_name] = results
    
    # Final comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    
    print("\nBug presence by model:")
    for model_name, results in all_results.items():
        bug_count = sum(1 for r in results.values() if r["accuracy"] < 50)
        print(f"\n{model_name}:")
        print(f"  Bug in {bug_count}/{len(results)} comparisons")
        
        if bug_count > 0:
            bug_pairs = [k.replace("_vs_", " vs ") for k, v in results.items() if v["accuracy"] < 50]
            print(f"  Failed on: {', '.join(bug_pairs)}")

if __name__ == "__main__":
    main()