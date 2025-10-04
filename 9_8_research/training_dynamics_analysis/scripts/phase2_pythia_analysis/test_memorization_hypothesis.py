#!/usr/bin/env python3
"""
Test Memorization vs Reasoning Hypothesis
========================================

This experiment tests whether the 9.8 vs 9.11 pattern represents:
1. Training data memorization (specific phrase learned)
2. Emergent reasoning capability (generalizable logic)

Test Categories:
1. Exact Phrase Variations - test robustness to small prompt changes
2. Semantic Equivalents - test with different phrasings of same question
3. Context Variations - test with different surrounding context
4. Tokenization Boundaries - test how tokenization affects the pattern
5. Training Data Probes - test for exact phrase memorization signatures
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from datetime import datetime

class MemorizationTester:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model_name = "EleutherAI/pythia-160m"

        print(f"Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        self.model.eval()

        # Test suite for memorization vs reasoning
        self.test_suite = {
            'exact_phrase': {
                'description': 'Exact phrase variations - test sensitivity to minor changes',
                'cases': [
                    # Exact original
                    "Q: Which is bigger: 9.8 or 9.11?\nA:",

                    # Punctuation variations
                    "Q: Which is bigger: 9.8 or 9.11? A:",
                    "Q: Which is bigger: 9.8 or 9.11?\n\nA:",
                    "Q Which is bigger: 9.8 or 9.11?\nA:",

                    # Spacing variations
                    "Q:Which is bigger: 9.8 or 9.11?\nA:",
                    "Q: Which is bigger:9.8 or 9.11?\nA:",
                    "Q: Which is bigger: 9.8or 9.11?\nA:",

                    # Case variations
                    "q: which is bigger: 9.8 or 9.11?\na:",
                    "Q: WHICH IS BIGGER: 9.8 OR 9.11?\nA:",
                ]
            },

            'semantic_equivalents': {
                'description': 'Semantic equivalents - same meaning, different words',
                'cases': [
                    # Different comparison words
                    "Q: Which is larger: 9.8 or 9.11?\nA:",
                    "Q: Which is greater: 9.8 or 9.11?\nA:",
                    "Q: Which is higher: 9.8 or 9.11?\nA:",
                    "Q: What is bigger: 9.8 or 9.11?\nA:",

                    # Different sentence structures
                    "Q: Between 9.8 and 9.11, which is bigger?\nA:",
                    "Q: Of 9.8 and 9.11, which is the bigger number?\nA:",
                    "Q: Compare 9.8 and 9.11. Which is bigger?\nA:",
                    "Q: Is 9.8 or 9.11 bigger?\nA:",

                    # Mathematical phrasing
                    "Q: 9.8 vs 9.11 - which is larger?\nA:",
                    "Q: Which number is greater: 9.8 or 9.11?\nA:",
                ]
            },

            'context_variations': {
                'description': 'Context variations - different surrounding text',
                'cases': [
                    # Different prefixes
                    "Question: Which is bigger: 9.8 or 9.11?\nAnswer:",
                    "Problem: Which is bigger: 9.8 or 9.11?\nSolution:",
                    "Math question: Which is bigger: 9.8 or 9.11?\nResponse:",

                    # No prefix
                    "Which is bigger: 9.8 or 9.11?",

                    # Different conversation context
                    "Student: Which is bigger: 9.8 or 9.11?\nTeacher:",
                    "User: Which is bigger: 9.8 or 9.11?\nBot:",

                    # Mid-conversation
                    "We're comparing decimals. Which is bigger: 9.8 or 9.11?\nThe answer is:",
                ]
            },

            'tokenization_probes': {
                'description': 'Tokenization boundary effects - test token-level memorization',
                'cases': [
                    # Extra spaces (changes tokenization)
                    "Q: Which is bigger: 9 . 8 or 9 . 11?\nA:",
                    "Q: Which is bigger: 9.8  or  9.11?\nA:",

                    # Different number formatting
                    "Q: Which is bigger: 09.8 or 09.11?\nA:",
                    "Q: Which is bigger: 9.80 or 9.11?\nA:",
                    "Q: Which is bigger: 9.8 or 9.11000?\nA:",
                ]
            },

            'training_data_signatures': {
                'description': 'Training data memorization signatures - test exact phrase recall',
                'cases': [
                    # Test if model continues exact training patterns
                    "Q: Which is bigger: 9.8 or 9.11?\nA: 9.8 is bigger because",
                    "Q: Which is bigger: 9.8 or 9.11?\nA: The answer is 9.8",
                    "Q: Which is bigger: 9.8 or 9.11?\nA: Looking at the numbers",

                    # Test for specific response patterns
                    "Q: Which is bigger: 9.8 or 9.11?\nA: When comparing decimals",
                    "Q: Which is bigger: 9.8 or 9.11?\nA: To compare these numbers",
                ]
            }
        }

    def test_prompt_specialization(self, prompt: str) -> dict:
        """Test even/odd specialization for a specific prompt"""
        clean_prompt = "Which is bigger: 9.8 or 9.11?"

        def check_correct(output: str) -> bool:
            output_lower = output.lower()
            return "9.8" in output_lower and "9.11" not in output_lower

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
        baseline_correct = check_correct(baseline_response)

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
        even_correct = check_correct(even_response)

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
        odd_correct = check_correct(odd_response)

        specialization = (1 if even_correct else 0) - (1 if odd_correct else 0)

        return {
            'prompt': prompt,
            'baseline_correct': baseline_correct,
            'even_correct': even_correct,
            'odd_correct': odd_correct,
            'specialization': specialization,
            'baseline_response': baseline_response.strip()[:50],
            'even_response': even_response.strip()[:50],
            'odd_response': odd_response.strip()[:50],
            'tokens': self.tokenizer.tokenize(prompt)
        }

    def run_memorization_analysis(self):
        """Run comprehensive memorization vs reasoning analysis"""
        print("\n🧠 MEMORIZATION vs REASONING ANALYSIS")
        print("="*60)
        print("Testing whether 9.8 vs 9.11 pattern is memorized or reasoned\n")

        all_results = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'categories': {}
        }

        category_summaries = {}

        for category_name, category_data in self.test_suite.items():
            print(f"\n{'='*20} {category_name.upper().replace('_', ' ')} {'='*20}")
            print(f"Testing: {category_data['description']}")
            print("-" * 60)

            category_results = []
            working_cases = 0

            for i, prompt in enumerate(category_data['cases']):
                print(f"\n📝 Case {i+1}/{len(category_data['cases'])}")
                print(f"Prompt: {prompt[:50]}...")

                result = self.test_prompt_specialization(prompt)
                category_results.append(result)

                spec = result['specialization']
                if spec > 0:
                    working_cases += 1
                    print(f"  ✅ WORKS (specialization: {spec:+.0f})")
                elif spec < 0:
                    print(f"  🔴 ODD PREFERENCE (specialization: {spec:+.0f})")
                else:
                    print(f"  ❌ FAILS (no specialization)")

                print(f"  Tokens: {len(result['tokens'])} - {result['tokens'][:5]}...")

            # Category analysis
            success_rate = working_cases / len(category_data['cases'])

            category_summary = {
                'description': category_data['description'],
                'total_cases': len(category_data['cases']),
                'working_cases': working_cases,
                'success_rate': success_rate,
                'memorization_evidence': self._analyze_memorization_evidence(category_name, category_results)
            }

            category_summaries[category_name] = category_summary
            all_results['categories'][category_name] = {
                'summary': category_summary,
                'detailed_results': category_results
            }

            print(f"\n📊 {category_name.upper()} SUMMARY:")
            print(f"  Working cases: {working_cases}/{len(category_data['cases'])} ({success_rate:.1%})")

        # Overall memorization analysis
        print(f"\n{'='*20} MEMORIZATION ANALYSIS {'='*20}")

        memorization_score = self._calculate_memorization_score(category_summaries)
        all_results['memorization_analysis'] = memorization_score

        print(f"\n🎯 MEMORIZATION vs REASONING VERDICT:")
        if memorization_score['overall_score'] > 0.7:
            print("✅ STRONG EVIDENCE FOR REASONING - pattern generalizes well")
        elif memorization_score['overall_score'] > 0.3:
            print("⚡ MIXED EVIDENCE - some generalization, some memorization")
        else:
            print("🚨 STRONG EVIDENCE FOR MEMORIZATION - pattern highly specific")

        print(f"\nDetailed Evidence:")
        for evidence_type, score in memorization_score['evidence_scores'].items():
            print(f"  {evidence_type}: {score:.2f}")

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"memorization_analysis_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"\n📁 Results saved to: {filename}")
        return all_results

    def _analyze_memorization_evidence(self, category_name: str, results: list) -> dict:
        """Analyze evidence for memorization in a category"""
        if category_name == 'exact_phrase':
            # High sensitivity to exact phrase changes suggests memorization
            success_rate = sum(1 for r in results if r['specialization'] > 0) / len(results)
            return {
                'type': 'phrase_sensitivity',
                'memorization_evidence': 1.0 - success_rate,  # More failures = more memorization
                'reasoning_evidence': success_rate
            }

        elif category_name == 'semantic_equivalents':
            # Good performance on semantic equivalents suggests reasoning
            success_rate = sum(1 for r in results if r['specialization'] > 0) / len(results)
            return {
                'type': 'semantic_generalization',
                'memorization_evidence': 1.0 - success_rate,
                'reasoning_evidence': success_rate
            }

        elif category_name == 'tokenization_probes':
            # Sensitivity to tokenization suggests token-level memorization
            success_rate = sum(1 for r in results if r['specialization'] > 0) / len(results)
            return {
                'type': 'tokenization_sensitivity',
                'memorization_evidence': 1.0 - success_rate,
                'reasoning_evidence': success_rate
            }

        else:
            success_rate = sum(1 for r in results if r['specialization'] > 0) / len(results)
            return {
                'type': 'general',
                'memorization_evidence': 1.0 - success_rate,
                'reasoning_evidence': success_rate
            }

    def _calculate_memorization_score(self, summaries: dict) -> dict:
        """Calculate overall memorization vs reasoning score"""
        evidence_scores = {}

        for category, summary in summaries.items():
            evidence = summary['memorization_evidence']
            evidence_scores[category] = evidence['reasoning_evidence']

        # Weight different types of evidence
        weights = {
            'exact_phrase': 0.3,      # Phrase sensitivity
            'semantic_equivalents': 0.4,  # Most important for reasoning
            'context_variations': 0.2,
            'tokenization_probes': 0.1,
            'training_data_signatures': 0.1
        }

        weighted_score = sum(evidence_scores.get(cat, 0) * weight
                           for cat, weight in weights.items())

        return {
            'overall_score': weighted_score,
            'evidence_scores': evidence_scores,
            'interpretation': {
                'high_score': 'Pattern generalizes - likely emergent reasoning',
                'medium_score': 'Mixed evidence - partial memorization and reasoning',
                'low_score': 'Highly specific - likely training data memorization'
            }
        }

if __name__ == "__main__":
    tester = MemorizationTester()
    results = tester.run_memorization_analysis()

    print("\n" + "="*60)
    print("MEMORIZATION vs REASONING ANALYSIS COMPLETE")
    print("="*60)

    print(f"\nThis analysis provides evidence for whether the 9.8 vs 9.11")
    print(f"pattern represents memorization or emergent reasoning capability.")