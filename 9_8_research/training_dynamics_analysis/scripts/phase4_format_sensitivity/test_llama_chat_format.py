#!/usr/bin/env python3
"""
Test Llama with Transluce's Exact Chat Format
===========================================

Test the exact format used by Transluce to see if this explains the 55% vs 5% discrepancy:
<|start_header_id|>user<|end_header_id|>Which is bigger, 9.8 or 9.11?<|eot_id|>
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from datetime import datetime

class LlamaChatFormatTester:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Use instruct for chat format

        print(f"Loading {self.model_name} for chat format testing...")
        print("Testing Transluce's exact format vs our Q&A format...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        self.model.eval()

    def test_both_formats(self, question: str, num1: str, num2: str) -> dict:
        """Test both our Q&A format and Transluce's chat format"""

        # Our format
        our_prompt = f"Q: Which is bigger: {num1} or {num2}?\nA:"

        # Transluce format (exact)
        transluce_prompt = f"<|start_header_id|>user<|end_header_id|>Which is bigger, {num1} or {num2}?<|eot_id|>"

        # Also try with proper chat template
        chat_messages = [{"role": "user", "content": f"Which is bigger, {num1} or {num2}?"}]
        chat_template_prompt = self.tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)

        results = {}

        for format_name, prompt in [
            ("our_format", our_prompt),
            ("transluce_format", transluce_prompt),
            ("chat_template", chat_template_prompt)
        ]:
            print(f"\n🧪 Testing {format_name}:")
            print(f"Prompt: {prompt[:100]}...")

            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

            # Determine correctness
            correct_answer = num1 if float(num1) > float(num2) else num2

            def extract_answer(text):
                """Extract which number the model thinks is bigger"""
                text_lower = text.lower()
                if num1 in text and num2 not in text:
                    return num1
                elif num2 in text and num1 not in text:
                    return num2
                elif "first" in text_lower:
                    return num1
                elif "second" in text_lower:
                    return num2
                else:
                    return "unclear"

            model_answer = extract_answer(response)
            is_correct = model_answer == correct_answer

            results[format_name] = {
                'prompt': prompt,
                'response': response.strip(),
                'model_answer': model_answer,
                'correct_answer': correct_answer,
                'is_correct': is_correct
            }

            status = "✅ CORRECT" if is_correct else "❌ WRONG"
            print(f"  {status}: {model_answer} (expected {correct_answer})")
            print(f"  Response: {response.strip()[:60]}...")

        return results

    def run_format_comparison(self):
        """Run comprehensive format comparison"""
        print("\n🔍 LLAMA CHAT FORMAT vs Q&A FORMAT TESTING")
        print("="*70)
        print("Testing if format explains Transluce discrepancy...\n")

        # Test key cases
        test_cases = [
            # The classic case
            ("9.8", "9.11"),
            ("9.11", "9.8"),  # Order reversed

            # Other cases that should work if it's real capability
            ("9.9", "9.11"),
            ("8.8", "8.11"),
            ("7.8", "7.11"),

            # Systematic cases
            ("1.6", "1.10"),
            ("5.7", "5.11"),
            ("12.9", "12.10"),
        ]

        all_results = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'test_description': 'Format comparison: Our Q&A vs Transluce chat format',
            'test_cases': {}
        }

        format_scores = {'our_format': 0, 'transluce_format': 0, 'chat_template': 0}
        total_cases = len(test_cases)

        for i, (num1, num2) in enumerate(test_cases):
            print(f"\n{'='*20} CASE {i+1}/{total_cases}: {num1} vs {num2} {'='*20}")

            results = self.test_both_formats(f"Which is bigger, {num1} or {num2}?", num1, num2)
            all_results['test_cases'][f'{num1}_vs_{num2}'] = results

            # Track scores
            for format_name in format_scores:
                if results[format_name]['is_correct']:
                    format_scores[format_name] += 1

        # Summary
        print(f"\n{'='*20} FORMAT COMPARISON SUMMARY {'='*20}")

        for format_name, score in format_scores.items():
            accuracy = score / total_cases
            print(f"📊 {format_name.upper()}: {accuracy:.1%} ({score}/{total_cases})")

        all_results['summary'] = {
            'format_scores': format_scores,
            'total_cases': total_cases,
            'format_accuracies': {k: v/total_cases for k, v in format_scores.items()}
        }

        # Check if Transluce format explains the discrepancy
        transluce_acc = format_scores['transluce_format'] / total_cases
        our_acc = format_scores['our_format'] / total_cases

        print(f"\n🎯 KEY FINDING:")
        if transluce_acc > 0.4:
            print(f"   ✅ Transluce format achieves {transluce_acc:.1%} - THIS EXPLAINS THE DISCREPANCY!")
        elif transluce_acc > our_acc + 0.2:
            print(f"   📈 Transluce format better ({transluce_acc:.1%} vs {our_acc:.1%}) - partially explains discrepancy")
        else:
            print(f"   ❌ Format difference doesn't explain discrepancy ({transluce_acc:.1%} vs {our_acc:.1%})")

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"llama_format_comparison_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"\n📁 Results saved to: {filename}")
        return all_results

if __name__ == "__main__":
    tester = LlamaChatFormatTester()
    results = tester.run_format_comparison()

    print("\n" + "="*70)
    print("FORMAT COMPARISON COMPLETE")
    print("="*70)
    print(f"This test checks if prompt format explains the Transluce discrepancy")