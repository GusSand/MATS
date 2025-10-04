#!/usr/bin/env python3
"""
Biblical Interference Hypothesis Testing
=======================================

Based on Transluce's finding that Bible verse neurons interfere with decimal comparison,
we test whether Pythia's "even/odd specialization" is actually a memorized patch for
biblical interference affecting the 9.8 vs 9.11 comparison.

Test Categories:
1. Biblical Context Tests - Does adding biblical context affect the pattern?
2. Other Biblical Ratios - Do other X:Y vs X:Z patterns show similar behavior?
3. Systematic Baseline - Does Pythia have the same ~55% deficit on broader comparisons?
4. Sequential Interpretation - Does Pythia think 9:8 comes before 9:11?
5. Bible Verse Explicit - Direct biblical verse comparison tests
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from datetime import datetime
import random

class BiblicalInterferenceTester:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model_name = "EleutherAI/pythia-160m"

        print(f"Loading {self.model_name} for biblical interference testing...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        self.model.eval()

        # Test suites based on biblical interference hypothesis
        self.test_suites = {
            'biblical_context': {
                'description': 'Test if adding biblical context affects the 9.8 vs 9.11 pattern',
                'cases': [
                    # Original working case
                    "Q: Which is bigger: 9.8 or 9.11?\nA:",

                    # Biblical context variations
                    "Q: Which verse number is bigger: 9.8 or 9.11?\nA:",
                    "Q: Comparing chapter 9 verses 8 and 11, which is bigger?\nA:",
                    "Q: In the Bible, which is bigger: 9:8 or 9:11?\nA:",
                    "Q: Which Bible verse comes later: 9.8 or 9.11?\nA:",
                    "Q: John 9:8 vs John 9:11 - which verse number is bigger?\nA:",

                    # Mathematical context (control)
                    "Q: Which decimal number is bigger: 9.8 or 9.11?\nA:",
                    "Q: Mathematically, which is bigger: 9.8 or 9.11?\nA:",
                    "Q: As numbers, which is bigger: 9.8 or 9.11?\nA:",
                ]
            },

            'other_biblical_ratios': {
                'description': 'Test other common biblical ratios for similar patterns',
                'cases': [
                    # Common biblical references
                    "Q: Which is bigger: 3.16 or 3.17?\nA:",  # John 3:16
                    "Q: Which is bigger: 1.1 or 1.23?\nA:",   # Psalm references
                    "Q: Which is bigger: 2.20 or 2.21?\nA:",  # Common chapter:verse
                    "Q: Which is bigger: 4.4 or 4.12?\nA:",   # Mixed biblical
                    "Q: Which is bigger: 5.7 or 5.14?\nA:",   # Mixed biblical
                    "Q: Which is bigger: 7.11 or 7.14?\nA:",  # Reverse our pattern

                    # Biblical but different format
                    "Q: Which is bigger: 12.5 or 12.12?\nA:", # Higher chapters
                    "Q: Which is bigger: 23.1 or 23.15?\nA:", # Psalm range
                ]
            },

            'systematic_baseline': {
                'description': 'Test systematic decimal comparison (like Transluce\'s 1280 cases)',
                'cases': []  # Will be generated programmatically
            },

            'sequential_interpretation': {
                'description': 'Test if Pythia interprets numbers as sequential (biblical) rather than numerical',
                'cases': [
                    # Sequential vs numerical interpretation
                    "Q: Which comes first in sequence: 9.8 or 9.11?\nA:",
                    "Q: Which comes later in sequence: 9.8 or 9.11?\nA:",
                    "Q: In chronological order, which comes first: 9.8 or 9.11?\nA:",

                    # Time/order context
                    "Q: Which timestamp is earlier: 9:8 or 9:11?\nA:",
                    "Q: Which time comes first: 9.8 or 9.11?\nA:",

                    # Explicit biblical ordering
                    "Q: Which Bible verse comes first: 9:8 or 9:11?\nA:",
                    "Q: In a book, which reference comes first: 9.8 or 9.11?\nA:",
                ]
            },

            'explicit_bible_verses': {
                'description': 'Direct biblical verse comparisons to test interference',
                'cases': [
                    # Explicit biblical format
                    "Q: Which verse comes later: 9:8 or 9:11?\nA:",
                    "Q: Which chapter:verse is higher: 9:8 or 9:11?\nA:",
                    "Q: In John 9, which verse is bigger: verse 8 or verse 11?\nA:",

                    # Mixed biblical references
                    "Q: Which verse number is larger: Psalm 23:1 vs Psalm 23:6?\nA:",
                    "Q: Which comes later: Matthew 5:3 or Matthew 5:12?\nA:",
                ]
            }
        }

        # Generate systematic baseline cases (like Transluce study)
        self._generate_systematic_cases()

    def _generate_systematic_cases(self):
        """Generate systematic X.Y vs X.Z comparisons like Transluce study"""
        cases = []

        # Smaller scale version of Transluce's 1280 cases
        # X from 1-10, Y from 6-9, Z from 10-13
        for X in [1, 5, 9, 15, 20]:  # Sample of X values
            for Y in [6, 7, 8, 9]:   # Y range
                for Z in [10, 11, 12, 13]:  # Z range
                    if Y != Z:  # Avoid identical comparisons
                        # Both orderings
                        cases.append(f"Q: Which is bigger: {X}.{Y} or {X}.{Z}?\nA:")
                        cases.append(f"Q: Which is bigger: {X}.{Z} or {X}.{Y}?\nA:")

        self.test_suites['systematic_baseline']['cases'] = cases[:50]  # Limit for practical testing

    def test_case_with_patching(self, prompt: str) -> dict:
        """Test a case with even/odd head patching like our original method"""
        clean_prompt = "Which is bigger: 9.8 or 9.11?"

        def check_correct_response(output: str) -> str:
            """Determine what the model thinks is bigger"""
            output_lower = output.lower()

            # Extract numbers mentioned
            if "9.8" in output_lower and "9.11" not in output_lower:
                return "9.8"
            elif "9.11" in output_lower and "9.8" not in output_lower:
                return "9.11"
            elif "first" in output_lower or "earlier" in output_lower or "before" in output_lower:
                return "first_mentioned"
            elif "later" in output_lower or "second" in output_lower or "after" in output_lower:
                return "second_mentioned"
            else:
                return "unclear"

        # Get clean activation
        attention_module = self.model.gpt_neox.layers[6].attention
        saved_activation = None

        def save_hook(module, input, output):
            nonlocal saved_activation
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            saved_activation = hidden_states.detach().cpu()

        clean_inputs = self.tokenizer(clean_prompt, return_tensors="pt").to(self.device)
        hook = attention_module.register_forward_hook(save_hook)
        with torch.no_grad():
            self.model(**clean_inputs)
        hook.remove()

        # Test baseline
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
        baseline_response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        baseline_answer = check_correct_response(baseline_response)

        # Test even heads
        even_heads = [0, 2, 4, 6, 8, 10]

        def patch_hook_even(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            batch_size, seq_len, hidden_size = hidden_states.shape
            head_dim = hidden_size // 12
            hidden_reshaped = hidden_states.view(batch_size, seq_len, 12, head_dim)
            saved_reshaped = saved_activation.to(hidden_states.device).view(batch_size, -1, 12, head_dim)
            new_hidden = hidden_reshaped.clone()
            min_seq_len = min(seq_len, saved_reshaped.shape[1])

            for head_idx in even_heads:
                new_hidden[:, :min_seq_len, head_idx, :] = saved_reshaped[:, :min_seq_len, head_idx, :]

            new_hidden = new_hidden.view(batch_size, seq_len, hidden_size)
            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden

        hook = attention_module.register_forward_hook(patch_hook_even)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
        even_response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        hook.remove()
        even_answer = check_correct_response(even_response)

        # Test odd heads
        odd_heads = [1, 3, 5, 7, 9, 11]

        def patch_hook_odd(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            batch_size, seq_len, hidden_size = hidden_states.shape
            head_dim = hidden_size // 12
            hidden_reshaped = hidden_states.view(batch_size, seq_len, 12, head_dim)
            saved_reshaped = saved_activation.to(hidden_states.device).view(batch_size, -1, 12, head_dim)
            new_hidden = hidden_reshaped.clone()
            min_seq_len = min(seq_len, saved_reshaped.shape[1])

            for head_idx in odd_heads:
                new_hidden[:, :min_seq_len, head_idx, :] = saved_reshaped[:, :min_seq_len, head_idx, :]

            new_hidden = new_hidden.view(batch_size, seq_len, hidden_size)
            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden

        hook = attention_module.register_forward_hook(patch_hook_odd)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
        odd_response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        hook.remove()
        odd_answer = check_correct_response(odd_response)

        # Calculate specialization
        baseline_correct = baseline_answer == "9.8"
        even_correct = even_answer == "9.8"
        odd_correct = odd_answer == "9.8"
        specialization = (1 if even_correct else 0) - (1 if odd_correct else 0)

        return {
            'prompt': prompt,
            'baseline_answer': baseline_answer,
            'even_answer': even_answer,
            'odd_answer': odd_answer,
            'specialization': specialization,
            'baseline_response': baseline_response.strip()[:50],
            'even_response': even_response.strip()[:50],
            'odd_response': odd_response.strip()[:50],
            'baseline_correct': baseline_correct,
            'even_correct': even_correct,
            'odd_correct': odd_correct
        }

    def run_biblical_interference_analysis(self):
        """Run comprehensive biblical interference hypothesis testing"""
        print("\n📖 BIBLICAL INTERFERENCE HYPOTHESIS TESTING")
        print("="*60)
        print("Testing whether Pythia's pattern is a memorized patch for biblical interference\n")

        all_results = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'hypothesis': 'biblical_interference',
            'test_categories': {}
        }

        category_summaries = {}

        for category_name, category_data in self.test_suites.items():
            print(f"\n{'='*20} {category_name.upper().replace('_', ' ')} {'='*20}")
            print(f"Testing: {category_data['description']}")
            print("-" * 60)

            category_results = []
            specialization_cases = 0
            correct_baseline_cases = 0

            for i, prompt in enumerate(category_data['cases']):
                print(f"\n📋 Case {i+1}/{len(category_data['cases'])}")
                print(f"Prompt: {prompt[:60]}...")

                result = self.test_case_with_patching(prompt)
                category_results.append(result)

                spec = result['specialization']
                baseline_correct = result['baseline_correct']

                if baseline_correct:
                    correct_baseline_cases += 1

                if spec > 0:
                    specialization_cases += 1
                    print(f"  ✅ EVEN SPECIALIZATION (strength: {spec:+.0f})")
                elif spec < 0:
                    print(f"  🔴 ODD SPECIALIZATION (strength: {spec:+.0f})")
                else:
                    print(f"  ⚪ NO SPECIALIZATION")

                print(f"  Baseline: {result['baseline_answer']} | Even: {result['even_answer']} | Odd: {result['odd_answer']}")

            # Category analysis
            specialization_rate = specialization_cases / len(category_data['cases'])
            baseline_accuracy = correct_baseline_cases / len(category_data['cases'])

            category_summary = {
                'description': category_data['description'],
                'total_cases': len(category_data['cases']),
                'specialization_cases': specialization_cases,
                'specialization_rate': specialization_rate,
                'baseline_accuracy': baseline_accuracy,
                'biblical_evidence': self._analyze_biblical_evidence(category_name, category_results)
            }

            category_summaries[category_name] = category_summary
            all_results['test_categories'][category_name] = {
                'summary': category_summary,
                'detailed_results': category_results
            }

            print(f"\n📊 {category_name.upper()} SUMMARY:")
            print(f"  Specialization rate: {specialization_rate:.1%}")
            print(f"  Baseline accuracy: {baseline_accuracy:.1%}")

        # Overall biblical interference analysis
        print(f"\n{'='*20} BIBLICAL INTERFERENCE ANALYSIS {'='*20}")

        biblical_evidence = self._calculate_biblical_evidence(category_summaries)
        all_results['biblical_analysis'] = biblical_evidence

        print(f"\n🎯 BIBLICAL INTERFERENCE HYPOTHESIS VERDICT:")
        if biblical_evidence['overall_score'] > 0.7:
            print("✅ STRONG EVIDENCE FOR BIBLICAL INTERFERENCE")
        elif biblical_evidence['overall_score'] > 0.3:
            print("⚡ MODERATE EVIDENCE FOR BIBLICAL INTERFERENCE")
        else:
            print("❌ WEAK EVIDENCE FOR BIBLICAL INTERFERENCE")

        print(f"\nDetailed Evidence:")
        for evidence_type, score in biblical_evidence['evidence_scores'].items():
            print(f"  {evidence_type}: {score:.2f}")

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"biblical_interference_analysis_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"\n📁 Results saved to: {filename}")
        return all_results

    def _analyze_biblical_evidence(self, category_name: str, results: list) -> dict:
        """Analyze evidence for biblical interference in a category"""
        if category_name == 'biblical_context':
            # If biblical context changes the pattern, suggests biblical interference
            original_spec = results[0]['specialization'] if results else 0
            biblical_specs = [r['specialization'] for r in results[1:6] if 'verse' in r['prompt'] or 'Bible' in r['prompt']]
            avg_biblical_spec = sum(biblical_specs) / len(biblical_specs) if biblical_specs else 0

            context_sensitivity = abs(original_spec - avg_biblical_spec)
            return {
                'type': 'context_sensitivity',
                'biblical_evidence': context_sensitivity,
                'original_specialization': original_spec,
                'biblical_specialization': avg_biblical_spec
            }

        elif category_name == 'other_biblical_ratios':
            # If other biblical ratios show similar patterns, suggests biblical interference
            specialization_rate = sum(1 for r in results if r['specialization'] > 0) / len(results)
            return {
                'type': 'pattern_generalization',
                'biblical_evidence': specialization_rate,
                'specialization_rate': specialization_rate
            }

        elif category_name == 'systematic_baseline':
            # If systematic accuracy is low (~55% like Transluce), suggests broader deficit
            accuracy = sum(1 for r in results if r['baseline_correct']) / len(results)
            return {
                'type': 'systematic_deficit',
                'biblical_evidence': 1.0 - accuracy,  # Lower accuracy = more evidence for deficit
                'baseline_accuracy': accuracy
            }

        elif category_name == 'sequential_interpretation':
            # If model interprets as sequential rather than numerical, suggests biblical interference
            sequential_responses = sum(1 for r in results if 'first' in r['baseline_response'] or 'sequence' in r['baseline_response'])
            sequential_rate = sequential_responses / len(results) if results else 0
            return {
                'type': 'sequential_interpretation',
                'biblical_evidence': sequential_rate,
                'sequential_response_rate': sequential_rate
            }

        else:
            specialization_rate = sum(1 for r in results if r['specialization'] > 0) / len(results)
            return {
                'type': 'general',
                'biblical_evidence': specialization_rate,
                'specialization_rate': specialization_rate
            }

    def _calculate_biblical_evidence(self, summaries: dict) -> dict:
        """Calculate overall biblical interference evidence score"""
        evidence_scores = {}

        for category, summary in summaries.items():
            evidence = summary['biblical_evidence']
            evidence_scores[category] = evidence['biblical_evidence']

        # Weight different types of evidence for biblical interference
        weights = {
            'biblical_context': 0.3,        # Context sensitivity most important
            'systematic_baseline': 0.3,     # Systematic deficit like Transluce
            'sequential_interpretation': 0.2, # Sequential vs numerical interpretation
            'other_biblical_ratios': 0.15,  # Pattern generalization
            'explicit_bible_verses': 0.05   # Direct biblical testing
        }

        weighted_score = sum(evidence_scores.get(cat, 0) * weight
                           for cat, weight in weights.items())

        return {
            'overall_score': weighted_score,
            'evidence_scores': evidence_scores,
            'interpretation': {
                'high_score': 'Strong evidence for biblical interference hypothesis',
                'medium_score': 'Moderate evidence - some biblical effects present',
                'low_score': 'Weak evidence - pattern likely pure memorization'
            }
        }

if __name__ == "__main__":
    tester = BiblicalInterferenceTester()
    results = tester.run_biblical_interference_analysis()

    print("\n" + "="*60)
    print("BIBLICAL INTERFERENCE HYPOTHESIS TESTING COMPLETE")
    print("="*60)

    print(f"\nThis analysis tests whether Pythia's 9.8 vs 9.11 pattern")
    print(f"represents a memorized patch for biblical verse interference.")