#!/usr/bin/env python3
"""
Comprehensive Decimal Testing Suite - Even/Odd Head Specialization Generalization
==============================================================================

This experiment tests whether the even/odd head specialization pattern generalizes
across different types of numerical comparison bugs beyond the original 9.8 vs 9.11.

Test Categories:
1. Type 1 (Classic Bug): Single digit vs double digit decimals - should show even specialization
2. Type 2 (Control): Same integer, different decimals - should work correctly without specialization
3. Type 3 (Mixed Length): Different decimal lengths - tests robustness
4. Type 4 (Tokenization): Multi-digit integers with decimals - tests tokenization effects

This will determine:
- Scope of even/odd specialization
- Generalization vs task-specific patterns
- Robustness across numerical formats
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
from contextlib import contextmanager
import json
from datetime import datetime
import time

class ComprehensiveDecimalTester:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model_name = "EleutherAI/pythia-160m"  # Final model with known specialization
        self.n_heads = 12
        self.target_layer = 6

        # Comprehensive test suite
        self.test_suite = {
            'type_1_bugs': {
                'description': 'Single digit vs double digit decimals (expected bug)',
                'cases': [
                    ('9.8', '9.11'),
                    ('8.9', '8.10'),
                    ('7.8', '7.11'),
                    ('6.9', '6.12'),
                    ('5.8', '5.13'),
                    ('4.7', '4.14'),
                    ('3.6', '3.15')
                ]
            },
            'type_2_correct': {
                'description': 'Same integer, different decimals (should work correctly)',
                'cases': [
                    ('8.7', '8.12'),
                    ('7.6', '7.13'),
                    ('6.5', '6.14'),
                    ('5.4', '5.15'),
                    ('4.3', '4.16'),
                    ('3.2', '3.17')
                ]
            },
            'type_3_mixed': {
                'description': 'Mixed decimal lengths (test robustness)',
                'cases': [
                    ('7.85', '7.9'),
                    ('8.95', '8.9'),
                    ('6.75', '6.8'),
                    ('5.65', '5.7'),
                    ('4.55', '4.6'),
                    ('3.45', '3.5')
                ]
            },
            'type_4_tokenization': {
                'description': 'Multi-digit integers (tokenization effects)',
                'cases': [
                    ('10.9', '10.11'),
                    ('20.8', '20.11'),
                    ('100.9', '100.11'),
                    ('12.7', '12.13'),
                    ('15.6', '15.14'),
                    ('25.5', '25.12')
                ]
            }
        }

        self.saved_activations = {}
        self.hooks = []

    @contextmanager
    def temporary_hooks(self):
        """Context manager for automatic hook cleanup"""
        try:
            yield
        finally:
            for hook in self.hooks:
                hook.remove()
            self.hooks.clear()

    def get_attention_module(self, model, layer_idx: int):
        """Get attention module for Pythia architecture"""
        return model.gpt_neox.layers[layer_idx].attention

    def selective_patch_hook(self, saved_activation: torch.Tensor, head_indices: List[int]):
        """Create hook for selective head patching - Pythia architecture"""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            batch_size, seq_len, hidden_size = hidden_states.shape
            head_dim = hidden_size // self.n_heads

            # Reshape for head-wise operations
            hidden_states_reshaped = hidden_states.view(batch_size, seq_len, self.n_heads, head_dim)
            saved_reshaped = saved_activation.to(hidden_states.device).view(batch_size, -1, self.n_heads, head_dim)

            new_hidden = hidden_states_reshaped.clone()
            min_seq_len = min(seq_len, saved_reshaped.shape[1])

            # Patch specified heads
            for head_idx in head_indices:
                if head_idx < self.n_heads:
                    new_hidden[:, :min_seq_len, head_idx, :] = saved_reshaped[:, :min_seq_len, head_idx, :]

            # Reshape back
            new_hidden = new_hidden.view(batch_size, seq_len, hidden_size)

            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden
        return hook_fn

    def test_number_pair(self, num1: str, num2: str, trials: int = 15) -> Dict:
        """Test a specific number pair with even/odd head patching"""

        # Load model fresh for each test
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        model.eval()

        # Determine correct answer
        correct_num = max(float(num1), float(num2))
        correct_answer = num1 if float(num1) == correct_num else num2

        # Create prompts
        clean_prompt = f"Which is bigger: {num1} or {num2}?"
        buggy_prompt = f"Q: Which is bigger: {num1} or {num2}?\nA:"

        # Head groups
        even_heads = list(range(0, self.n_heads, 2))  # [0, 2, 4, 6, 8, 10]
        odd_heads = list(range(1, self.n_heads, 2))   # [1, 3, 5, 7, 9, 11]

        def check_correct_answer(output: str) -> bool:
            """Check if output contains correct numerical answer"""
            output_lower = output.lower()
            # Look for the correct number in the response
            return correct_answer in output and (num1 if correct_answer != num1 else num2) not in output

        results = {
            'number_pair': f"{num1} vs {num2}",
            'correct_answer': correct_answer,
            'baseline_success_rate': 0.0,
            'even_success_rate': 0.0,
            'odd_success_rate': 0.0,
            'specialization_strength': 0.0,
            'sample_responses': {}
        }

        try:
            with self.temporary_hooks():
                # Test baseline (no patching)
                baseline_success = 0
                baseline_samples = []

                for trial in range(trials):
                    inputs = tokenizer(buggy_prompt, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=20,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id
                        )
                    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                    baseline_samples.append(response.strip()[:50])
                    if check_correct_answer(response):
                        baseline_success += 1

                results['baseline_success_rate'] = baseline_success / trials
                results['sample_responses']['baseline'] = baseline_samples[0] if baseline_samples else ""

                # Get clean activation for patching
                clean_inputs = tokenizer(clean_prompt, return_tensors="pt").to(self.device)
                attention_module = self.get_attention_module(model, self.target_layer)

                # Save clean activation
                def save_hook(module, input, output):
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                    else:
                        hidden_states = output
                    self.saved_activations['clean'] = hidden_states.detach().cpu()

                save_handle = attention_module.register_forward_hook(save_hook)
                with torch.no_grad():
                    model(**clean_inputs)
                save_handle.remove()

                # Test head groups
                for head_type, head_list in [('even', even_heads), ('odd', odd_heads)]:
                    success_count = 0
                    samples = []

                    # Install patching hook
                    if 'clean' in self.saved_activations:
                        patch_hook = self.selective_patch_hook(self.saved_activations['clean'], head_list)
                        hook_handle = attention_module.register_forward_hook(patch_hook)
                        self.hooks.append(hook_handle)

                        for trial in range(trials):
                            try:
                                inputs = tokenizer(buggy_prompt, return_tensors="pt").to(self.device)
                                with torch.no_grad():
                                    outputs = model.generate(
                                        **inputs,
                                        max_new_tokens=20,
                                        do_sample=False,
                                        pad_token_id=tokenizer.pad_token_id
                                    )
                                response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                                samples.append(response.strip()[:50])
                                if check_correct_answer(response):
                                    success_count += 1
                            except Exception:
                                continue

                        # Remove hook for this head type
                        hook_handle.remove()
                        self.hooks = [h for h in self.hooks if h != hook_handle]

                    success_rate = success_count / trials
                    results[f'{head_type}_success_rate'] = success_rate
                    results['sample_responses'][head_type] = samples[0] if samples else ""

                # Calculate specialization strength
                results['specialization_strength'] = results['even_success_rate'] - results['odd_success_rate']

        except Exception as e:
            results['error'] = str(e)
            results['specialization_strength'] = 0.0

        finally:
            # Cleanup
            del model
            del tokenizer
            torch.cuda.empty_cache()

        return results

    def run_comprehensive_test(self) -> Dict:
        """Run comprehensive testing across all categories"""
        print("\n🔢 COMPREHENSIVE DECIMAL TESTING SUITE")
        print("=" * 60)
        print("Testing even/odd head specialization generalization")
        print("across different numerical comparison scenarios\n")

        start_time = time.time()
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'test_categories': {},
            'summary': {}
        }

        category_summaries = {}

        for category_name, category_data in self.test_suite.items():
            print(f"\n{'='*20} {category_name.upper().replace('_', ' ')} {'='*20}")
            print(f"Testing: {category_data['description']}")
            print("-" * 60)

            category_results = []
            total_cases = len(category_data['cases'])

            for i, (num1, num2) in enumerate(category_data['cases']):
                print(f"\n📊 Case {i+1}/{total_cases}: {num1} vs {num2}")

                result = self.test_number_pair(num1, num2)
                category_results.append(result)

                # Show immediate results
                strength = result.get('specialization_strength', 0.0)
                even_rate = result.get('even_success_rate', 0.0)
                odd_rate = result.get('odd_success_rate', 0.0)

                print(f"  Even: {even_rate:.1%} | Odd: {odd_rate:.1%} | Strength: {strength:+.2f}")

                if 'error' in result:
                    print(f"  ❌ Error: {result['error']}")
                elif abs(strength) > 0.5:
                    print(f"  ✅ Strong specialization detected")
                elif abs(strength) > 0.3:
                    print(f"  ⚡ Moderate specialization")
                else:
                    print(f"  ⚪ No specialization")

            # Category summary
            strengths = [r.get('specialization_strength', 0.0) for r in category_results if 'error' not in r]
            avg_strength = np.mean(strengths) if strengths else 0.0
            strong_cases = sum(1 for s in strengths if abs(s) > 0.5)

            category_summary = {
                'description': category_data['description'],
                'total_cases': total_cases,
                'successful_tests': len(strengths),
                'average_specialization_strength': avg_strength,
                'strong_specialization_cases': strong_cases,
                'strong_specialization_rate': strong_cases / len(strengths) if strengths else 0.0
            }

            category_summaries[category_name] = category_summary
            all_results['test_categories'][category_name] = {
                'summary': category_summary,
                'detailed_results': category_results
            }

            print(f"\n📈 {category_name.upper()} SUMMARY:")
            print(f"  Average specialization: {avg_strength:+.2f}")
            print(f"  Strong cases: {strong_cases}/{len(strengths)} ({strong_cases/len(strengths)*100:.1f}%)")

        # Overall analysis
        print(f"\n{'='*20} COMPREHENSIVE ANALYSIS {'='*20}")

        total_time = time.time() - start_time

        # Cross-category comparison
        for cat_name, summary in category_summaries.items():
            print(f"\n{cat_name.replace('_', ' ').title()}:")
            print(f"  Strong specialization rate: {summary['strong_specialization_rate']:.1%}")
            print(f"  Average strength: {summary['average_specialization_strength']:+.2f}")

        all_results['summary'] = {
            'total_test_time_seconds': total_time,
            'category_summaries': category_summaries,
            'key_findings': self._generate_key_findings(category_summaries)
        }

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"comprehensive_decimal_testing_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"\n📁 Results saved to: {filename}")
        print(f"⏱️  Total analysis time: {total_time:.1f} seconds")

        return all_results

    def _generate_key_findings(self, summaries: Dict) -> List[str]:
        """Generate key findings from category summaries"""
        findings = []

        # Check if Type 1 bugs show strong specialization
        type1_rate = summaries.get('type_1_bugs', {}).get('strong_specialization_rate', 0)
        if type1_rate > 0.7:
            findings.append("Type 1 bugs (single vs double digit) consistently show even head specialization")
        elif type1_rate > 0.3:
            findings.append("Type 1 bugs show moderate even head specialization pattern")
        else:
            findings.append("Type 1 bugs do not consistently trigger specialization")

        # Check if controls work as expected
        type2_rate = summaries.get('type_2_correct', {}).get('strong_specialization_rate', 0)
        if type2_rate < 0.3:
            findings.append("Control cases (Type 2) correctly show minimal specialization")
        else:
            findings.append("Control cases unexpectedly show specialization - may indicate broader pattern")

        # Check robustness across formats
        type3_rate = summaries.get('type_3_mixed', {}).get('strong_specialization_rate', 0)
        type4_rate = summaries.get('type_4_tokenization', {}).get('strong_specialization_rate', 0)

        if type3_rate > 0.5 and type4_rate > 0.5:
            findings.append("Specialization pattern is robust across different decimal formats")
        elif type3_rate > 0.5 or type4_rate > 0.5:
            findings.append("Specialization shows some robustness but format-dependent effects exist")
        else:
            findings.append("Specialization pattern may be specific to certain numerical formats")

        return findings

if __name__ == "__main__":
    tester = ComprehensiveDecimalTester()
    results = tester.run_comprehensive_test()

    print("\n" + "="*60)
    print("COMPREHENSIVE DECIMAL TESTING COMPLETE")
    print("="*60)

    print("\nKey Findings:")
    for finding in results['summary']['key_findings']:
        print(f"• {finding}")

    print(f"\nThis analysis provides insight into the scope and generalization")
    print(f"of even/odd attention head specialization patterns.")