#!/usr/bin/env python3
"""
Test Transluce's EXACT Format
============================

Test the precise format Transluce uses:
<|start_header_id|>user<|end_header_id|>Which is bigger, 9.8 or 9.11?<|eot_id|>

Without system message or assistant prompt.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from datetime import datetime

class TransluceExactFormatTester:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

        print(f"Loading {self.model_name} for EXACT Transluce format testing...")
        print("Testing their precise format: <|start_header_id|>user<|end_header_id|>MESSAGE<|eot_id|>")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        self.model.eval()

    def test_exact_transluce_format(self, num1: str, num2: str) -> dict:
        """Test Transluce's exact format"""

        # Transluce's EXACT format (no system message, no assistant prompt)
        prompt = f"<|start_header_id|>user<|end_header_id|>Which is bigger, {num1} or {num2}?<|eot_id|>"

        print(f"\n🧪 Testing: {num1} vs {num2}")
        print(f"Exact prompt: {prompt}")

        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Determine correctness
        correct_answer = num1 if float(num1) > float(num2) else num2

        def extract_answer(text):
            """Extract which number the model thinks is bigger"""
            text_lower = text.lower()

            # Check for explicit number mentions
            if num1 in text and num2 not in text:
                return num1
            elif num2 in text and num1 not in text:
                return num2

            # Check for positional indicators
            elif "first" in text_lower or "earlier" in text_lower:
                return num1
            elif "second" in text_lower or "later" in text_lower:
                return num2

            # Check for comparative language
            elif f"{num1} is" in text and "bigger" in text_lower:
                return num1
            elif f"{num2} is" in text and "bigger" in text_lower:
                return num2
            elif f"{num1} >" in text:
                return num1
            elif f"{num2} >" in text:
                return num2
            else:
                return "unclear"

        model_answer = extract_answer(response)
        is_correct = model_answer == correct_answer

        result = {
            'prompt': prompt,
            'response': response.strip(),
            'model_answer': model_answer,
            'correct_answer': correct_answer,
            'is_correct': is_correct,
            'response_length': len(response.strip())
        }

        status = "✅ CORRECT" if is_correct else "❌ WRONG"
        print(f"  {status}: Model says '{model_answer}' (expected '{correct_answer}')")
        print(f"  Response: {response.strip()[:100]}...")

        return result

    def run_transluce_exact_test(self):
        """Run test with Transluce's exact format"""
        print("\n🎯 TRANSLUCE EXACT FORMAT TESTING")
        print("="*60)
        print("Testing the precise format they claim gives 55% accuracy...\n")

        # Test the same cases as Transluce would test
        test_cases = [
            # The classic case
            ("9.8", "9.11"),
            ("9.11", "9.8"),

            # Variations to test if it's real capability
            ("9.9", "9.11"),
            ("8.8", "8.11"),
            ("7.8", "7.11"),
            ("6.8", "6.11"),

            # Different decimal lengths
            ("1.6", "1.10"),
            ("2.7", "2.11"),
            ("5.9", "5.12"),
            ("12.8", "12.11"),

            # Reverse order cases
            ("1.11", "1.7"),
            ("5.12", "5.8"),
        ]

        all_results = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'test_description': 'Transluce exact format test',
            'format_used': '<|start_header_id|>user<|end_header_id|>MESSAGE<|eot_id|>',
            'test_cases': {}
        }

        correct_count = 0
        total_cases = len(test_cases)

        for i, (num1, num2) in enumerate(test_cases):
            print(f"\n{'='*15} CASE {i+1}/{total_cases} {'='*15}")

            result = self.test_exact_transluce_format(num1, num2)
            all_results['test_cases'][f'{num1}_vs_{num2}'] = result

            if result['is_correct']:
                correct_count += 1

        # Calculate accuracy
        accuracy = correct_count / total_cases

        print(f"\n{'='*20} FINAL RESULTS {'='*20}")
        print(f"📊 ACCURACY: {accuracy:.1%} ({correct_count}/{total_cases})")

        print(f"\n🎯 COMPARISON TO TRANSLUCE CLAIM:")
        if accuracy >= 0.5:
            print(f"   ✅ MATCHES TRANSLUCE! {accuracy:.1%} ≈ 55%")
            print("   📈 This format explains the discrepancy!")
        elif accuracy >= 0.3:
            print(f"   📈 CLOSER TO TRANSLUCE: {accuracy:.1%} vs 55%")
            print("   🤔 Partially explains discrepancy")
        else:
            print(f"   ❌ STILL DOESN'T MATCH: {accuracy:.1%} vs 55%")
            print("   🔍 Discrepancy remains unexplained")

        all_results['summary'] = {
            'total_cases': total_cases,
            'correct_count': correct_count,
            'accuracy': accuracy,
            'transluce_claim': 0.55,
            'matches_transluce': accuracy >= 0.4
        }

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"transluce_exact_format_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"\n📁 Results saved to: {filename}")
        return all_results

if __name__ == "__main__":
    tester = TransluceExactFormatTester()
    results = tester.run_transluce_exact_test()

    print("\n" + "="*60)
    print("TRANSLUCE EXACT FORMAT TEST COMPLETE")
    print("="*60)