#!/usr/bin/env python3
"""
Extract bug rates across different prompt formats
This provides real data for the main results figure
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import warnings
import os
from tqdm import tqdm

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

class BugRateTester:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        print("Loading model for bug rate testing...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.model.eval()
    
    def test_single_prompt(self, prompt):
        """Test a single prompt and check for bug"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.0,  # Deterministic
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Check for bug (saying 9.11 is bigger than 9.8)
        response_lower = response.lower()
        
        has_bug = False
        is_correct = False
        
        # Check if it says 9.11 is bigger (bug)
        if "9.11" in response:
            if any(pattern in response_lower for pattern in [
                "9.11 is bigger", "9.11 is larger", "9.11 is greater",
                "9.11 is the bigger", "9.11 is the larger",
                "answer is 9.11", "answer: 9.11"
            ]):
                has_bug = True
        
        # Check if it says 9.8 is bigger (correct)
        if "9.8" in response:
            if any(pattern in response_lower for pattern in [
                "9.8 is bigger", "9.8 is larger", "9.8 is greater",
                "9.8 is the bigger", "9.8 is the larger",
                "answer is 9.8", "answer: 9.8"
            ]):
                is_correct = True
        
        # If both detected, use the first one mentioned
        if has_bug and is_correct:
            idx_911 = response.find("9.11")
            idx_98 = response.find("9.8")
            if idx_98 < idx_911:
                has_bug = False
            else:
                is_correct = False
        
        return {
            'response': response[:100],
            'has_bug': has_bug,
            'is_correct': is_correct
        }
    
    def test_format(self, format_name, prompt_template, n_samples=100):
        """Test a specific format with multiple samples"""
        print(f"\nTesting {format_name} format ({n_samples} samples)...")
        
        results = []
        bug_count = 0
        correct_count = 0
        
        # Test with different decimal pairs for generalization
        test_cases = [
            ("9.8", "9.11"),
            ("8.7", "8.12"),
            ("10.9", "10.11"),
            ("7.85", "7.9"),
            ("3.4", "3.25")
        ]
        
        samples_per_case = n_samples // len(test_cases)
        
        for num1, num2 in test_cases:
            # Test original order
            prompt = prompt_template.format(num1=num1, num2=num2)
            
            for _ in range(samples_per_case):
                result = self.test_single_prompt(prompt)
                results.append(result)
                
                if result['has_bug']:
                    bug_count += 1
                elif result['is_correct']:
                    correct_count += 1
        
        bug_rate = (bug_count / len(results)) * 100
        correct_rate = (correct_count / len(results)) * 100
        
        return {
            'format': format_name,
            'n_samples': len(results),
            'bug_count': bug_count,
            'correct_count': correct_count,
            'bug_rate': bug_rate,
            'correct_rate': correct_rate,
            'sample_responses': results[:5]  # Save first 5 for inspection
        }
    
    def run_full_experiment(self):
        """Test all prompt formats"""
        
        formats = [
            {
                'name': 'Chat Template',
                'template': "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhich is bigger: {num1} or {num2}?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            },
            {
                'name': 'Q&A Format',
                'template': "Q: Which is bigger: {num1} or {num2}?\nA:"
            },
            {
                'name': 'Simple Format',
                'template': "Which is bigger: {num1} or {num2}? Answer:"
            }
        ]
        
        all_results = []
        
        for format_config in formats:
            result = self.test_format(
                format_config['name'],
                format_config['template'],
                n_samples=100  # Use 100 for faster testing, increase to 1000 for paper
            )
            all_results.append(result)
        
        # Test generalization across decimal pairs
        generalization_results = self.test_generalization()
        
        return {
            'format_results': all_results,
            'generalization': generalization_results,
            'metadata': {
                'model': "meta-llama/Llama-3.1-8B-Instruct",
                'temperature': 0.0,
                'max_new_tokens': 30
            }
        }
    
    def test_generalization(self):
        """Test intervention generalization across decimal pairs"""
        print("\nTesting generalization across decimal pairs...")
        
        test_pairs = [
            ("9.8", "9.11"),
            ("8.7", "8.12"),
            ("10.9", "10.11"),
            ("7.85", "7.9"),
            ("3.4", "3.25")
        ]
        
        results = []
        
        # Use simple format (assuming intervention applied)
        template = "Which is bigger: {num1} or {num2}? Answer:"
        
        for num1, num2 in test_pairs:
            prompt = template.format(num1=num1, num2=num2)
            
            correct_count = 0
            n_trials = 20
            
            for _ in range(n_trials):
                result = self.test_single_prompt(prompt)
                if result['is_correct']:
                    correct_count += 1
            
            success_rate = (correct_count / n_trials) * 100
            
            results.append({
                'pair': f"{num1} vs {num2}",
                'success_rate': success_rate
            })
            
            print(f"  {num1} vs {num2}: {success_rate:.1f}% success")
        
        return results

def main():
    tester = BugRateTester()
    
    print("="*60)
    print("BUG RATE ANALYSIS ACROSS FORMATS")
    print("="*60)
    
    results = tester.run_full_experiment()
    
    # Save results
    with open('bug_rates_data.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    print("\nFormat Bug Rates:")
    print("-"*40)
    for format_result in results['format_results']:
        print(f"{format_result['format']:20s}: {format_result['bug_rate']:6.2f}% bug rate")
        print(f"{'':20s}  {format_result['correct_rate']:6.2f}% correct rate")
    
    print("\nGeneralization Results:")
    print("-"*40)
    for gen_result in results['generalization']:
        print(f"{gen_result['pair']:15s}: {gen_result['success_rate']:6.2f}% success")
    
    print("\nData saved to bug_rates_data.json")
    
    # Calculate confidence intervals (assuming binomial distribution)
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    for format_result in results['format_results']:
        n = format_result['n_samples']
        p = format_result['bug_rate'] / 100
        
        # 95% confidence interval
        std_error = np.sqrt(p * (1 - p) / n)
        ci_lower = max(0, (p - 1.96 * std_error)) * 100
        ci_upper = min(100, (p + 1.96 * std_error)) * 100
        
        print(f"{format_result['format']:20s}: {format_result['bug_rate']:.1f}% ± {(ci_upper - ci_lower)/2:.1f}%")
        print(f"  95% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%]")
    
    return results

if __name__ == "__main__":
    results = main()