#!/usr/bin/env python3
"""
Extract CORRECT bug rates using the exact prompts that trigger the bug
Based on working_scripts/verify_llama_bug.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

class CorrectBugRateTester:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        print("Loading model for correct bug rate testing...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.model.eval()
        print("Model loaded successfully!")
    
    def test_single_prompt(self, prompt, temperature=0.0):
        """Test a single prompt and check for bug"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the generated part
        generated = full_response[len(prompt):]
        generated_lower = generated.lower()
        
        # Check for bug (saying 9.11 is bigger than 9.8)
        says_9_8_bigger = (
            ("9.8" in generated and any(w in generated_lower for w in ["bigger", "larger", "greater"])) or
            ("bigger than 9.11" in generated_lower) or
            ("larger than 9.11" in generated_lower) or
            ("greater than 9.11" in generated_lower)
        )
        
        says_9_11_bigger = (
            ("9.11" in generated and any(w in generated_lower for w in ["bigger", "larger", "greater"])) or
            ("bigger than 9.8" in generated_lower) or
            ("larger than 9.8" in generated_lower) or
            ("greater than 9.8" in generated_lower)
        )
        
        # If both detected, check first occurrence
        if says_9_8_bigger and says_9_11_bigger:
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
                            else:
                                says_9_8_bigger = False
                                says_9_11_bigger = True
                            break
        
        return {
            'response': generated[:100],
            'has_bug': says_9_11_bigger and not says_9_8_bigger,
            'is_correct': says_9_8_bigger and not says_9_11_bigger,
            'unclear': not (says_9_8_bigger or says_9_11_bigger)
        }
    
    def test_format(self, format_name, prompt, n_samples=100):
        """Test a specific format with multiple samples"""
        print(f"\nTesting {format_name} format ({n_samples} samples)...")
        print(f"Prompt: {repr(prompt[:80])}...")
        
        results = []
        bug_count = 0
        correct_count = 0
        unclear_count = 0
        
        for i in range(n_samples):
            result = self.test_single_prompt(prompt, temperature=0.0)
            results.append(result)
            
            if result['has_bug']:
                bug_count += 1
                symbol = "❌"
            elif result['is_correct']:
                correct_count += 1
                symbol = "✅"
            else:
                unclear_count += 1
                symbol = "❓"
            
            # Print first 3 examples
            if i < 3:
                print(f"  {symbol} {result['response'][:60].strip()}...")
        
        bug_rate = (bug_count / n_samples) * 100
        correct_rate = (correct_count / n_samples) * 100
        unclear_rate = (unclear_count / n_samples) * 100
        
        print(f"Results: {correct_count}/{n_samples} correct, {bug_count}/{n_samples} wrong")
        print(f"Rates: {correct_rate:.1f}% correct, {bug_rate:.1f}% bug, {unclear_rate:.1f}% unclear")
        
        return {
            'format': format_name,
            'prompt': prompt[:100] + '...' if len(prompt) > 100 else prompt,
            'n_samples': n_samples,
            'bug_count': bug_count,
            'correct_count': correct_count,
            'unclear_count': unclear_count,
            'bug_rate': bug_rate,
            'correct_rate': correct_rate,
            'unclear_rate': unclear_rate,
            'sample_responses': results[:5]
        }
    
    def run_full_experiment(self):
        """Test all prompt formats with the EXACT formats that trigger the bug"""
        
        # Define the EXACT formats from working_scripts
        formats = [
            {
                'name': 'Chat Template (with system)',
                'prompt': self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": "Which is bigger: 9.8 or 9.11?"}],
                    tokenize=False,
                    add_generation_prompt=True
                )
            },
            {
                'name': 'Chat Template (raw headers)',
                'prompt': "<|start_header_id|>user<|end_header_id|>\n\nWhich is bigger: 9.8 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            },
            {
                'name': 'Q&A Format',
                'prompt': "Q: Which is bigger: 9.8 or 9.11?\nA:"
            },
            {
                'name': 'Simple Format',
                'prompt': "Which is bigger: 9.8 or 9.11?\nAnswer:"
            }
        ]
        
        all_results = []
        
        for format_config in formats:
            result = self.test_format(
                format_config['name'],
                format_config['prompt'],
                n_samples=100
            )
            all_results.append(result)
        
        # Test generalization
        generalization = self.test_generalization()
        
        return {
            'format_results': all_results,
            'generalization': generalization,
            'metadata': {
                'model': "meta-llama/Llama-3.1-8B-Instruct",
                'temperature': 0.0,
                'max_new_tokens': 50
            }
        }
    
    def test_generalization(self):
        """Test with Q&A format (buggy) on different decimal pairs"""
        print("\nTesting generalization with Q&A format...")
        
        test_pairs = [
            ("9.8", "9.11"),
            ("8.7", "8.12"),
            ("10.9", "10.11"),
            ("7.85", "7.9"),
            ("3.4", "3.25")
        ]
        
        results = []
        
        for num1, num2 in test_pairs:
            prompt = f"Q: Which is bigger: {num1} or {num2}?\nA:"
            
            bug_count = 0
            correct_count = 0
            n_trials = 20
            
            for _ in range(n_trials):
                result = self.test_single_prompt(prompt)
                if result['has_bug']:
                    bug_count += 1
                elif result['is_correct']:
                    correct_count += 1
            
            bug_rate = (bug_count / n_trials) * 100
            correct_rate = (correct_count / n_trials) * 100
            
            # For intervention success, we'd use Simple format which should be 100% correct
            # But here we're testing if the bug generalizes with Q&A format
            results.append({
                'pair': f"{num1} vs {num2}",
                'bug_rate': bug_rate,
                'correct_rate': correct_rate
            })
            
            print(f"  {num1} vs {num2}: {bug_rate:.1f}% bug rate")
        
        return results

def main():
    tester = CorrectBugRateTester()
    
    print("="*60)
    print("CORRECT BUG RATE ANALYSIS")
    print("Using exact prompts that trigger the bug")
    print("="*60)
    
    results = tester.run_full_experiment()
    
    # Save results
    with open('bug_rates_data_correct.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    print("\nFormat Bug Rates:")
    print("-"*40)
    for format_result in results['format_results']:
        print(f"{format_result['format']:30s}: {format_result['bug_rate']:6.1f}% bug rate")
        print(f"{'':30s}  {format_result['correct_rate']:6.1f}% correct rate")
    
    print("\nGeneralization (Q&A format):")
    print("-"*40)
    for gen_result in results['generalization']:
        print(f"{gen_result['pair']:15s}: {gen_result['bug_rate']:6.1f}% bug rate")
    
    print("\nData saved to bug_rates_data_correct.json")
    
    # Key insights
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    
    qa_result = next((r for r in results['format_results'] if 'Q&A' in r['format']), None)
    simple_result = next((r for r in results['format_results'] if 'Simple' in r['format']), None)
    
    if qa_result and simple_result:
        print(f"✅ Simple format: {simple_result['correct_rate']:.1f}% correct (no bug!)")
        print(f"❌ Q&A format: {qa_result['bug_rate']:.1f}% bug rate")
        print(f"📊 Difference: {qa_result['bug_rate'] - simple_result['bug_rate']:.1f}% points")
    
    return results

if __name__ == "__main__":
    results = main()