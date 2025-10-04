#!/usr/bin/env python3
"""
Comprehensive Llama Testing - Compare to Pythia Findings
=======================================================

This script tests Llama-3.1-8B with the same comprehensive methodology we used for Pythia
to understand the differences in decimal comparison capabilities and patterns.

Key Questions:
1. Does Llama show the same ultra-specific memorization as Pythia?
2. Does Llama have the 55% baseline accuracy that Transluce found?
3. Does the even/odd head pattern generalize better in Llama?
4. How does the biblical interference manifest in Llama vs Pythia?

Test Categories (same as Pythia tests):
1. Pattern Specificity - test 9.8 vs 9.11 variations
2. Generalization - test other decimal pairs
3. Phrase Sensitivity - test prompt variations
4. Systematic Baseline - test X.Y vs X.Z like Transluce
5. Biblical Interference - test biblical context effects
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from datetime import datetime
import random

class LlamaComprehensiveTester:
    def __init__(self, device: str = "cuda"):
        self.device = device
        # Test base model vs instruct model to match Transluce study
        self.model_name = "meta-llama/Meta-Llama-3.1-8B"  # Base model, not instruct

        print(f"Loading {self.model_name} for comprehensive testing...")
        print("This will compare directly to our Pythia findings...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        self.model.eval()

        # Use same test suites as Pythia for direct comparison
        self.test_suites = self._create_test_suites()

    def _create_test_suites(self):
        return {
            'pattern_specificity': {
                'description': 'Test specificity of 9.8 vs 9.11 pattern (compare to Pythia ultra-specificity)',
                'cases': [
                    # Exact case that works in Pythia
                    "Q: Which is bigger: 9.8 or 9.11?\nA:",

                    # Order dependency (fails in Pythia)
                    "Q: Which is bigger: 9.11 or 9.8?\nA:",

                    # Number variations (fail in Pythia)
                    "Q: Which is bigger: 9.9 or 9.11?\nA:",
                    "Q: Which is bigger: 9.7 or 9.11?\nA:",
                    "Q: Which is bigger: 8.8 or 8.11?\nA:",
                    "Q: Which is bigger: 5.8 or 5.11?\nA:",

                    # 9.8 vs other endings (work in Pythia)
                    "Q: Which is bigger: 9.8 or 9.12?\nA:",
                    "Q: Which is bigger: 9.8 or 9.10?\nA:",
                ]
            },

            'phrase_sensitivity': {
                'description': 'Test sensitivity to phrase changes (Pythia is extremely sensitive)',
                'cases': [
                    # Original that works in Pythia
                    "Q: Which is bigger: 9.8 or 9.11?\nA:",

                    # Punctuation changes (break Pythia)
                    "Q: Which is bigger: 9.8 or 9.11? A:",
                    "Q: Which is bigger: 9.8 or 9.11?\n\nA:",
                    "Q Which is bigger: 9.8 or 9.11?\nA:",

                    # Word changes (break Pythia)
                    "Q: Which is larger: 9.8 or 9.11?\nA:",
                    "Q: Which is greater: 9.8 or 9.11?\nA:",
                    "Q: What is bigger: 9.8 or 9.11?\nA:",

                    # Format changes (break Pythia)
                    "Question: Which is bigger: 9.8 or 9.11?\nAnswer:",
                    "Which is bigger: 9.8 or 9.11?",

                    # Case changes (break Pythia)
                    "q: which is bigger: 9.8 or 9.11?\na:",
                ]
            },

            'systematic_baseline': {
                'description': 'Test systematic decimal comparison (Transluce found 55% for Llama, we found 0% for Pythia)',
                'cases': []  # Will be generated
            },

            'generalization_testing': {
                'description': 'Test generalization to other decimal pairs (0% success in Pythia)',
                'cases': [
                    # Biblical-style ratios
                    "Q: Which is bigger: 3.16 or 3.17?\nA:",
                    "Q: Which is bigger: 1.1 or 1.23?\nA:",
                    "Q: Which is bigger: 2.20 or 2.21?\nA:",

                    # Similar structure to 9.8 vs 9.11
                    "Q: Which is bigger: 7.8 or 7.11?\nA:",
                    "Q: Which is bigger: 6.9 or 6.12?\nA:",
                    "Q: Which is bigger: 4.7 or 4.14?\nA:",

                    # Different decimal lengths
                    "Q: Which is bigger: 7.85 or 7.9?\nA:",
                    "Q: Which is bigger: 8.95 or 8.9?\nA:",

                    # Multi-digit integers
                    "Q: Which is bigger: 10.9 or 10.11?\nA:",
                    "Q: Which is bigger: 12.7 or 12.13?\nA:",
                ]
            },

            'biblical_context': {
                'description': 'Test biblical interference effects (moderate evidence in Pythia)',
                'cases': [
                    # Biblical context
                    "Q: Which verse number is bigger: 9.8 or 9.11?\nA:",
                    "Q: In the Bible, which is bigger: 9:8 or 9:11?\nA:",
                    "Q: Which Bible verse comes later: 9.8 or 9.11?\nA:",

                    # Sequential interpretation
                    "Q: Which comes first in sequence: 9.8 or 9.11?\nA:",
                    "Q: Which timestamp is earlier: 9:8 or 9:11?\nA:",

                    # Mathematical context (control)
                    "Q: Which decimal number is bigger: 9.8 or 9.11?\nA:",
                    "Q: Mathematically, which is bigger: 9.8 or 9.11?\nA:",
                ]
            }
        }

    def _generate_systematic_cases(self):
        """Generate systematic X.Y vs X.Z like Transluce study"""
        cases = []
        # Smaller sample than Transluce's 1280 for practical testing
        for X in [1, 5, 9, 12, 15]:  # Sample of integers
            for Y in [6, 7, 8, 9]:    # First decimal
                for Z in [10, 11, 12, 13]:  # Second decimal
                    if Y != Z:
                        # Both orderings
                        cases.append(f"Q: Which is bigger: {X}.{Y} or {X}.{Z}?\nA:")
                        cases.append(f"Q: Which is bigger: {X}.{Z} or {X}.{Y}?\nA:")

        # Sample 40 cases for practical testing
        return random.sample(cases, min(40, len(cases)))

    def test_decimal_comparison(self, prompt: str) -> dict:
        """Test a decimal comparison and determine correctness"""

        # Extract numbers from prompt
        import re
        numbers = re.findall(r'\d+\.\d+', prompt)
        if len(numbers) != 2:
            return {'prompt': prompt, 'error': 'Could not extract two numbers'}

        num1, num2 = numbers[0], numbers[1]
        correct_answer = num1 if float(num1) > float(num2) else num2

        # Test with standard generation
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Check if response contains correct answer
        response_lower = response.lower()

        def extract_answer(text):
            """Extract which number the model thinks is bigger"""
            if num1 in text and num2 not in text:
                return num1
            elif num2 in text and num1 not in text:
                return num2
            elif "first" in text or "earlier" in text:
                return num1  # First mentioned
            elif "second" in text or "later" in text:
                return num2  # Second mentioned
            else:
                return "unclear"

        model_answer = extract_answer(response_lower)
        is_correct = model_answer == correct_answer

        return {
            'prompt': prompt,
            'numbers': [num1, num2],
            'correct_answer': correct_answer,
            'model_answer': model_answer,
            'is_correct': is_correct,
            'response': response.strip()[:60],
            'full_response': response.strip()
        }

    def run_comprehensive_llama_analysis(self):
        """Run comprehensive Llama testing to compare with Pythia findings"""
        print("\n🦙 COMPREHENSIVE LLAMA-3.1-8B TESTING")
        print("="*60)
        print("Comparing to Pythia-160M findings with identical methodology\n")

        # Generate systematic cases
        self.test_suites['systematic_baseline']['cases'] = self._generate_systematic_cases()

        all_results = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'comparison_target': 'Pythia-160M findings',
            'test_categories': {}
        }

        category_summaries = {}

        for category_name, category_data in self.test_suites.items():
            print(f"\n{'='*20} {category_name.upper().replace('_', ' ')} {'='*20}")
            print(f"Testing: {category_data['description']}")
            print("-" * 60)

            category_results = []
            correct_cases = 0

            for i, prompt in enumerate(category_data['cases']):
                print(f"\n📋 Case {i+1}/{len(category_data['cases'])}")
                print(f"Prompt: {prompt[:60]}...")

                result = self.test_decimal_comparison(prompt)
                category_results.append(result)

                if 'error' in result:
                    print(f"  ❌ Error: {result['error']}")
                    continue

                if result['is_correct']:
                    correct_cases += 1
                    print(f"  ✅ CORRECT: {result['model_answer']} (expected {result['correct_answer']})")
                else:
                    print(f"  ❌ WRONG: {result['model_answer']} (expected {result['correct_answer']})")

                print(f"  Response: {result['response']}...")

            # Category analysis
            total_valid = len([r for r in category_results if 'error' not in r])
            accuracy = correct_cases / total_valid if total_valid > 0 else 0

            category_summary = {
                'description': category_data['description'],
                'total_cases': len(category_data['cases']),
                'valid_cases': total_valid,
                'correct_cases': correct_cases,
                'accuracy': accuracy,
                'comparison_to_pythia': self._compare_to_pythia(category_name, accuracy, correct_cases, total_valid)
            }

            category_summaries[category_name] = category_summary
            all_results['test_categories'][category_name] = {
                'summary': category_summary,
                'detailed_results': category_results
            }

            print(f"\n📊 {category_name.upper()} SUMMARY:")
            print(f"  Accuracy: {accuracy:.1%} ({correct_cases}/{total_valid})")

        # Overall comparison analysis
        print(f"\n{'='*20} LLAMA vs PYTHIA COMPARISON {'='*20}")

        overall_comparison = self._generate_overall_comparison(category_summaries)
        all_results['llama_vs_pythia_analysis'] = overall_comparison

        print(f"\n🎯 KEY FINDINGS:")
        for finding in overall_comparison['key_findings']:
            print(f"  • {finding}")

        print(f"\n📈 ACCURACY COMPARISON:")
        for category, summary in category_summaries.items():
            pythia_note = summary['comparison_to_pythia']['note']
            print(f"  {category}: {summary['accuracy']:.1%} - {pythia_note}")

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"llama_comprehensive_analysis_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"\n📁 Results saved to: {filename}")
        return all_results

    def _compare_to_pythia(self, category_name: str, accuracy: float, correct: int, total: int) -> dict:
        """Compare results to Pythia findings"""

        pythia_results = {
            'pattern_specificity': {'accuracy': 0.375, 'note': 'Only 3/8 cases worked in Pythia'},
            'phrase_sensitivity': {'accuracy': 0.11, 'note': 'Extremely sensitive - 1/9 cases worked'},
            'systematic_baseline': {'accuracy': 0.0, 'note': 'Complete failure - 0% accuracy'},
            'generalization_testing': {'accuracy': 0.0, 'note': 'No generalization - 0/10 cases'},
            'biblical_context': {'accuracy': 0.33, 'note': 'Some context sensitivity - 3/9 cases'}
        }

        pythia_acc = pythia_results.get(category_name, {}).get('accuracy', 0)
        pythia_note = pythia_results.get(category_name, {}).get('note', 'No comparison data')

        if accuracy > pythia_acc + 0.2:
            comparison = "Much better than Pythia"
        elif accuracy > pythia_acc + 0.1:
            comparison = "Better than Pythia"
        elif accuracy > pythia_acc - 0.1:
            comparison = "Similar to Pythia"
        else:
            comparison = "Worse than Pythia"

        return {
            'pythia_accuracy': pythia_acc,
            'llama_accuracy': accuracy,
            'comparison': comparison,
            'note': pythia_note
        }

    def _generate_overall_comparison(self, summaries: dict) -> dict:
        """Generate overall comparison between Llama and Pythia"""

        key_findings = []

        # Systematic baseline comparison
        systematic_acc = summaries.get('systematic_baseline', {}).get('accuracy', 0)
        if systematic_acc > 0.4:
            key_findings.append(f"Llama has {systematic_acc:.1%} systematic accuracy vs Pythia's 0% - major difference!")
        elif systematic_acc > 0.1:
            key_findings.append(f"Llama has {systematic_acc:.1%} systematic accuracy - better than Pythia but still poor")
        else:
            key_findings.append(f"Llama has {systematic_acc:.1%} systematic accuracy - similar failure to Pythia")

        # Pattern specificity
        pattern_acc = summaries.get('pattern_specificity', {}).get('accuracy', 0)
        if pattern_acc > 0.7:
            key_findings.append("Llama shows broader decimal comparison capability than Pythia's ultra-specific pattern")
        elif pattern_acc > 0.4:
            key_findings.append("Llama shows moderate generalization beyond Pythia's narrow pattern")
        else:
            key_findings.append("Llama shows similar specificity to Pythia's narrow pattern")

        # Phrase sensitivity
        phrase_acc = summaries.get('phrase_sensitivity', {}).get('accuracy', 0)
        if phrase_acc > 0.5:
            key_findings.append("Llama is much more robust to phrase changes than Pythia")
        elif phrase_acc > 0.2:
            key_findings.append("Llama shows some robustness to phrase changes unlike Pythia")
        else:
            key_findings.append("Llama shows similar phrase sensitivity to Pythia")

        # Overall assessment
        avg_accuracy = sum(s.get('accuracy', 0) for s in summaries.values()) / len(summaries)

        if avg_accuracy > 0.5:
            overall_assessment = "Llama shows significantly better decimal comparison capabilities than Pythia"
        elif avg_accuracy > 0.2:
            overall_assessment = "Llama shows moderately better capabilities than Pythia but still has major deficits"
        else:
            overall_assessment = "Llama shows similar systematic decimal comparison failures to Pythia"

        return {
            'key_findings': key_findings,
            'overall_assessment': overall_assessment,
            'average_accuracy': avg_accuracy,
            'pythia_average': 0.088,  # Approximate from our testing
            'improvement_over_pythia': avg_accuracy - 0.088
        }

if __name__ == "__main__":
    tester = LlamaComprehensiveTester()
    results = tester.run_comprehensive_llama_analysis()

    print("\n" + "="*60)
    print("COMPREHENSIVE LLAMA vs PYTHIA COMPARISON COMPLETE")
    print("="*60)

    print(f"\nThis analysis directly compares Llama-3.1-8B capabilities")
    print(f"to our Pythia-160M findings using identical test methodology.")