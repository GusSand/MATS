#!/usr/bin/env python3
"""
Comprehensive Format Sensitivity Analysis
=========================================

Test 25-30 decimal comparisons across multiple formats for both Pythia and Llama
to get robust statistics on format sensitivity effects.

Formats tested:
1. Q&A format (our original)
2. Transluce exact format (minimal chat)
3. Official chat template (full)
4. Simple format (no Q&A structure)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from datetime import datetime
import random

class ComprehensiveFormatTester:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.models = {}
        self.tokenizers = {}

        # Load both models
        model_names = [
            "EleutherAI/pythia-160m",
            "meta-llama/Meta-Llama-3.1-8B-Instruct"
        ]

        for model_name in model_names:
            print(f"Loading {model_name}...")

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if "llama" in model_name.lower() else torch.float32,
                device_map=self.device
            )
            model.eval()

            short_name = "pythia" if "pythia" in model_name.lower() else "llama"
            self.models[short_name] = model
            self.tokenizers[short_name] = tokenizer

    def generate_test_cases(self, n_cases: int = 25) -> list:
        """Generate systematic decimal comparison test cases"""
        test_cases = []

        # Ensure we get exactly n_cases
        random.seed(42)  # For reproducibility

        # Type 1: X.Y vs X.Z where Y < Z but X.Y > X.Z (the classic bug)
        for X in [1, 2, 5, 7, 9, 12]:
            for Y in [6, 7, 8, 9]:
                for Z in [10, 11, 12, 13]:
                    if Y < Z:  # Ensure this triggers the bug
                        test_cases.append((f"{X}.{Y}", f"{X}.{Z}"))

        # Type 2: Reverse order
        for X in [3, 6, 8]:
            for Y in [7, 8, 9]:
                for Z in [10, 11]:
                    test_cases.append((f"{X}.{Z}", f"{X}.{Y}"))

        # Type 3: Different integer parts
        additional_cases = [
            ("4.8", "5.1"), ("7.9", "8.2"), ("10.7", "11.1"),
            ("15.8", "16.2"), ("20.9", "21.1"), ("3.95", "4.1"),
            ("6.85", "7.1"), ("9.75", "10.2")
        ]
        test_cases.extend(additional_cases)

        # Sample exactly n_cases
        if len(test_cases) > n_cases:
            test_cases = random.sample(test_cases, n_cases)
        elif len(test_cases) < n_cases:
            # Add more cases if needed
            while len(test_cases) < n_cases:
                X = random.randint(1, 20)
                Y = random.randint(1, 9)
                Z = random.randint(10, 19)
                test_cases.append((f"{X}.{Y}", f"{X}.{Z}"))

        return test_cases[:n_cases]

    def format_prompt(self, num1: str, num2: str, format_type: str) -> str:
        """Generate prompt in specified format"""
        if format_type == "qa_format":
            return f"Q: Which is bigger: {num1} or {num2}?\nA:"

        elif format_type == "transluce_format":
            return f"<|start_header_id|>user<|end_header_id|>Which is bigger, {num1} or {num2}?<|eot_id|>"

        elif format_type == "chat_template":
            # This will be handled specially using apply_chat_template
            return f"Which is bigger, {num1} or {num2}?"

        elif format_type == "simple_format":
            return f"Which is bigger: {num1} or {num2}?"

        else:
            raise ValueError(f"Unknown format type: {format_type}")

    def test_single_case(self, model_name: str, num1: str, num2: str, format_type: str) -> dict:
        """Test a single decimal comparison case"""

        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]

        # Handle chat template specially
        if format_type == "chat_template" and model_name == "llama":
            messages = [{"role": "user", "content": self.format_prompt(num1, num2, format_type)}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = self.format_prompt(num1, num2, format_type)

        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Determine correctness
        correct_answer = num1 if float(num1) > float(num2) else num2

        def extract_answer(text):
            """Extract which number the model thinks is bigger"""
            text_lower = text.lower()

            # Direct number mentions
            if num1 in text and num2 not in text:
                return num1
            elif num2 in text and num1 not in text:
                return num2

            # Comparative statements
            elif f"{num1} is bigger" in text_lower or f"{num1} >" in text:
                return num1
            elif f"{num2} is bigger" in text_lower or f"{num2} >" in text:
                return num2

            # Positional indicators
            elif "first" in text_lower:
                return num1
            elif "second" in text_lower:
                return num2

            else:
                return "unclear"

        model_answer = extract_answer(response)
        is_correct = model_answer == correct_answer

        return {
            'num1': num1,
            'num2': num2,
            'correct_answer': correct_answer,
            'model_answer': model_answer,
            'is_correct': is_correct,
            'response': response.strip()[:100],
            'prompt_used': prompt[:100] + "..." if len(prompt) > 100 else prompt
        }

    def run_comprehensive_test(self, n_cases: int = 25):
        """Run comprehensive format sensitivity test"""

        print(f"\n🧪 COMPREHENSIVE FORMAT SENSITIVITY ANALYSIS")
        print("="*70)
        print(f"Testing {n_cases} decimal comparisons across multiple formats")
        print("Models: Pythia-160M, Llama-3.1-8B-Instruct")
        print("Formats: Q&A, Transluce, Chat Template, Simple\n")

        # Generate test cases
        test_cases = self.generate_test_cases(n_cases)
        print(f"Generated {len(test_cases)} test cases")

        # Define formats to test
        formats = {
            "qa_format": "Q: Which is bigger: X or Y?\\nA:",
            "transluce_format": "<|start_header_id|>user<|end_header_id|>Which is bigger, X or Y?<|eot_id|>",
            "chat_template": "Official chat template (Llama only)",
            "simple_format": "Which is bigger: X or Y?"
        }

        all_results = {
            'timestamp': datetime.now().isoformat(),
            'test_description': f'Comprehensive format sensitivity test with {n_cases} cases',
            'test_cases': test_cases,
            'formats_tested': formats,
            'results': {}
        }

        # Test each model with each format
        for model_name in ["pythia", "llama"]:
            print(f"\n{'='*20} TESTING {model_name.upper()} {'='*20}")
            all_results['results'][model_name] = {}

            for format_name, format_desc in formats.items():
                # Skip chat template for Pythia
                if format_name == "chat_template" and model_name == "pythia":
                    continue

                print(f"\n📋 Format: {format_name}")
                print(f"Description: {format_desc}")

                format_results = []
                correct_count = 0

                for i, (num1, num2) in enumerate(test_cases):
                    if i % 5 == 0:
                        print(f"  Progress: {i+1}/{len(test_cases)}")

                    try:
                        result = self.test_single_case(model_name, num1, num2, format_name)
                        format_results.append(result)

                        if result['is_correct']:
                            correct_count += 1

                    except Exception as e:
                        print(f"  Error on case {i+1}: {e}")
                        format_results.append({
                            'num1': num1, 'num2': num2, 'error': str(e)
                        })

                accuracy = correct_count / len(test_cases)

                all_results['results'][model_name][format_name] = {
                    'accuracy': accuracy,
                    'correct_count': correct_count,
                    'total_cases': len(test_cases),
                    'detailed_results': format_results
                }

                print(f"  📊 Accuracy: {accuracy:.1%} ({correct_count}/{len(test_cases)})")

        # Summary analysis
        print(f"\n{'='*20} COMPREHENSIVE SUMMARY {'='*20}")

        # Create comparison table
        print(f"\n📊 ACCURACY COMPARISON TABLE")
        print(f"{'Format':<20} {'Pythia':<10} {'Llama':<10} {'Difference':<12}")
        print("-" * 52)

        for format_name in ["qa_format", "transluce_format", "simple_format"]:
            pythia_acc = all_results['results']['pythia'][format_name]['accuracy']
            llama_acc = all_results['results']['llama'][format_name]['accuracy']
            diff = llama_acc - pythia_acc

            print(f"{format_name:<20} {pythia_acc:<10.1%} {llama_acc:<10.1%} {diff:+.1%}")

        # Chat template for Llama only
        if 'chat_template' in all_results['results']['llama']:
            llama_chat_acc = all_results['results']['llama']['chat_template']['accuracy']
            print(f"{'chat_template':<20} {'N/A':<10} {llama_chat_acc:<10.1%} {'N/A':<12}")

        # Key findings
        print(f"\n🎯 KEY FINDINGS:")

        # Best format for each model
        pythia_best = max(all_results['results']['pythia'].items(),
                         key=lambda x: x[1]['accuracy'])
        llama_best = max(all_results['results']['llama'].items(),
                        key=lambda x: x[1]['accuracy'])

        print(f"  • Best Pythia format: {pythia_best[0]} ({pythia_best[1]['accuracy']:.1%})")
        print(f"  • Best Llama format: {llama_best[0]} ({llama_best[1]['accuracy']:.1%})")

        # Format sensitivity analysis
        pythia_range = max(all_results['results']['pythia'][f]['accuracy'] for f in all_results['results']['pythia']) - \
                      min(all_results['results']['pythia'][f]['accuracy'] for f in all_results['results']['pythia'])
        llama_range = max(all_results['results']['llama'][f]['accuracy'] for f in all_results['results']['llama']) - \
                     min(all_results['results']['llama'][f]['accuracy'] for f in all_results['results']['llama'])

        print(f"  • Pythia format sensitivity: {pythia_range:.1%} range")
        print(f"  • Llama format sensitivity: {llama_range:.1%} range")

        # Statistical significance
        if llama_best[1]['accuracy'] > pythia_best[1]['accuracy'] + 0.1:
            print(f"  • Llama shows substantially better performance (+{llama_best[1]['accuracy'] - pythia_best[1]['accuracy']:.1%})")
        elif pythia_best[1]['accuracy'] > llama_best[1]['accuracy'] + 0.1:
            print(f"  • Pythia shows substantially better performance (+{pythia_best[1]['accuracy'] - llama_best[1]['accuracy']:.1%})")
        else:
            print(f"  • Similar performance across models")

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"comprehensive_format_sensitivity_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"\n📁 Results saved to: {filename}")
        return all_results

if __name__ == "__main__":
    tester = ComprehensiveFormatTester()
    results = tester.run_comprehensive_test(n_cases=25)

    print("\n" + "="*70)
    print("COMPREHENSIVE FORMAT SENSITIVITY ANALYSIS COMPLETE")
    print("="*70)