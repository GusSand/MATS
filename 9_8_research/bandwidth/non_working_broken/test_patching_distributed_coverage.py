#!/usr/bin/env python3
"""
Patching Experiment: Test Distributed Coverage Hypothesis
========================================================

This script implements actual model patching to test whether distributed coverage
across attention heads predicts success in fixing the 9.8 vs 9.11 numerical bug.

We'll test the key patterns identified in our hypothesis testing:
1. Distributed Coverage: [0, 4, 8, 12, 16, 20, 24, 28] - Should work
2. Clustered Even: [0, 2, 4, 6, 8, 10, 12, 14] - Should fail if coverage matters
3. Minimal Distributed: [0, 8, 16, 24] - Test minimum coverage (4 heads)
4. Dense Coverage: [0, 1, 2, 3, 4, 5, 6, 7] - Should fail if spacing matters
5. Head Count Tests: 4, 6, 8, 10, 12 heads with optimal spacing
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from tqdm import tqdm

class PatchingExperiment:
    def __init__(self):
        self.model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.layer = 10
        self.n_heads = 32

        # Load model and tokenizer
        print("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Test prompts for numerical comparison
        self.test_prompts = [
            "Which is larger: 9.8 or 9.11?",
            "Compare 9.8 and 9.11. Which is bigger?",
            "Is 9.8 > 9.11? Answer yes or no:",
            "Which number is bigger: 9.8 or 9.11?",
            "Between 9.8 and 9.11, which is the larger number?",
            "9.8 vs 9.11 - which is greater?",
            "Choose the larger: 9.8 or 9.11",
            "Which is more: 9.8 or 9.11?",
        ]

        # Expected correct answer indicators
        self.correct_indicators = ["9.8", "9.8", "yes", "9.8", "9.8", "9.8", "9.8", "9.8"]

    def get_clean_activations(self, prompt):
        """Get clean activations from the model without any intervention"""

        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)

        # Store activations
        activations = {}

        def hook_fn(module, input, output):
            # output is a tuple: (hidden_states, past_key_value)
            hidden_states = output[0]  # [batch, seq_len, hidden_size]
            activations['hidden_states'] = hidden_states.detach()

            # For attention, we need the attention weights
            # Llama attention returns (hidden_states, None, present_key_value)
            if hasattr(module, 'self_attn'):
                # This is the attention layer
                pass

        # Register hook on the target layer
        target_layer = self.model.model.layers[self.layer]
        handle = target_layer.register_forward_hook(hook_fn)

        try:
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits
        finally:
            handle.remove()

        return {
            'logits': logits,
            'activations': activations,
            'input_ids': input_ids
        }

    def patch_attention_heads(self, prompt, head_indices, clean_activations=None):
        """Patch specific attention heads and measure the effect"""

        if clean_activations is None:
            clean_activations = self.get_clean_activations(prompt)

        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)

        # Store patched activations
        patched_logits = None

        def attention_patch_hook(module, input, output):
            nonlocal patched_logits

            # Get the attention output (first element of output tuple)
            hidden_states = output[0]  # [batch, seq_len, hidden_size]

            # For attention patching, we need to zero out specific heads
            # In Llama, attention heads are organized in the hidden dimension
            batch_size, seq_len, hidden_size = hidden_states.shape
            head_dim = hidden_size // self.n_heads

            # Reshape to separate heads: [batch, seq_len, n_heads, head_dim]
            hidden_states_heads = hidden_states.view(batch_size, seq_len, self.n_heads, head_dim)

            # Zero out heads NOT in our selection (patch by keeping only selected heads)
            mask = torch.zeros(self.n_heads, device=hidden_states.device, dtype=hidden_states.dtype)
            mask[head_indices] = 1.0

            # Apply mask: [n_heads] -> [1, 1, n_heads, 1]
            mask = mask.view(1, 1, self.n_heads, 1)
            hidden_states_heads = hidden_states_heads * mask

            # Reshape back to original format
            hidden_states_patched = hidden_states_heads.view(batch_size, seq_len, hidden_size)

            # Return modified output
            return (hidden_states_patched,) + output[1:]

        # Register hook on the target attention layer
        target_layer = self.model.model.layers[self.layer].self_attn
        handle = target_layer.register_forward_hook(attention_patch_hook)

        try:
            with torch.no_grad():
                outputs = self.model(input_ids)
                patched_logits = outputs.logits
        finally:
            handle.remove()

        return patched_logits

    def test_pattern_performance(self, head_pattern, pattern_name):
        """Test a specific head pattern on all test prompts"""

        print(f"\nTesting pattern: {pattern_name}")
        print(f"Heads: {head_pattern}")

        results = {
            'pattern_name': pattern_name,
            'heads': head_pattern,
            'n_heads': len(head_pattern),
            'prompt_results': [],
            'success_rate': 0.0
        }

        correct_count = 0

        for i, (prompt, correct_answer) in enumerate(zip(self.test_prompts, self.correct_indicators)):
            print(f"  Prompt {i+1}: {prompt[:50]}...")

            # Get clean response
            clean_acts = self.get_clean_activations(prompt)

            # Get patched response
            patched_logits = self.patch_attention_heads(prompt, head_pattern, clean_acts)

            # Generate response with patched model
            with torch.no_grad():
                # Sample from the patched logits
                input_ids = clean_acts['input_ids']
                next_token_logits = patched_logits[0, -1, :]  # Last token logits

                # Get top tokens
                top_k = 10
                top_tokens = torch.topk(next_token_logits, top_k)
                top_token_ids = top_tokens.indices
                top_probs = torch.softmax(top_tokens.values, dim=0)

                # Decode top tokens
                top_words = [self.tokenizer.decode([tid.item()]).strip() for tid in top_token_ids]

                # Generate a short continuation
                generated = self.model.generate(
                    input_ids,
                    max_new_tokens=20,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=self.tokenizer.eos_token_id
                )

                response = self.tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)

            # Check if response contains correct answer
            is_correct = correct_answer.lower() in response.lower()
            if is_correct:
                correct_count += 1

            prompt_result = {
                'prompt': prompt,
                'expected': correct_answer,
                'response': response,
                'top_words': top_words[:5],
                'top_probs': top_probs[:5].tolist(),
                'is_correct': is_correct
            }

            results['prompt_results'].append(prompt_result)

            print(f"    Expected: {correct_answer}")
            print(f"    Response: {response[:100]}")
            print(f"    Correct: {is_correct}")

        results['success_rate'] = correct_count / len(self.test_prompts)
        print(f"  Success rate: {results['success_rate']:.2%} ({correct_count}/{len(self.test_prompts)})")

        return results

    def run_comprehensive_test(self):
        """Run the comprehensive distributed coverage test"""

        print("=" * 60)
        print("DISTRIBUTED COVERAGE PATCHING EXPERIMENT")
        print("=" * 60)
        print(f"Model: {self.model_name}")
        print(f"Layer: {self.layer}")
        print(f"Number of test prompts: {len(self.test_prompts)}")
        print()

        # Define test patterns based on our hypotheses
        test_patterns = {
            # Core hypothesis tests
            'distributed_coverage': [0, 4, 8, 12, 16, 20, 24, 28],
            'clustered_even': [0, 2, 4, 6, 8, 10, 12, 14],
            'minimal_distributed': [0, 8, 16, 24],
            'dense_coverage': [0, 1, 2, 3, 4, 5, 6, 7],

            # Head count ablation
            'heads_4_optimal': [0, 8, 16, 24],
            'heads_6_optimal': [0, 5, 11, 16, 21, 27],
            'heads_8_optimal': [0, 4, 8, 12, 16, 20, 24, 28],
            'heads_10_optimal': [0, 3, 6, 9, 12, 15, 18, 21, 24, 27],
            'heads_12_optimal': [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 31],

            # Even vs distributed test
            'even_poor_spacing': [0, 2, 4, 6, 8, 10, 12, 14],
            'mixed_good_spacing': [0, 4, 8, 12, 16, 20, 24, 28],

            # Spacing tests
            'irregular_spacing': [0, 1, 5, 12, 14, 18, 26, 30],
            'regular_spacing_alt': [1, 5, 9, 13, 17, 21, 25, 29],

            # Known successful patterns from previous experiments
            'known_working_1': [2, 4, 6, 8, 10, 12, 14, 16],
            'known_working_2': [16, 18, 20, 22, 24, 26, 28, 30],
        }

        # Store all results
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'layer': self.layer,
            'test_patterns': test_patterns,
            'pattern_results': {},
            'summary': {}
        }

        # Test each pattern
        for pattern_name, head_indices in test_patterns.items():
            try:
                pattern_results = self.test_pattern_performance(head_indices, pattern_name)
                all_results['pattern_results'][pattern_name] = pattern_results
            except Exception as e:
                print(f"Error testing pattern {pattern_name}: {e}")
                all_results['pattern_results'][pattern_name] = {
                    'error': str(e),
                    'success_rate': 0.0
                }

        # Analyze results
        all_results['summary'] = self.analyze_results(all_results['pattern_results'])

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"patching_distributed_coverage_results_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\nResults saved to: {output_file}")

        # Create visualization
        self.visualize_results(all_results)

        return all_results

    def analyze_results(self, pattern_results):
        """Analyze the pattern results to test hypotheses"""

        summary = {
            'hypothesis_tests': {},
            'success_rates_by_pattern': {},
            'head_count_analysis': {},
            'spacing_analysis': {}
        }

        # Extract success rates
        for pattern_name, results in pattern_results.items():
            if 'success_rate' in results:
                summary['success_rates_by_pattern'][pattern_name] = results['success_rate']

        # Test specific hypotheses
        rates = summary['success_rates_by_pattern']

        # Hypothesis 1: Distributed vs Clustered
        if 'distributed_coverage' in rates and 'clustered_even' in rates:
            summary['hypothesis_tests']['distributed_vs_clustered'] = {
                'distributed_rate': rates['distributed_coverage'],
                'clustered_rate': rates['clustered_even'],
                'hypothesis_supported': rates['distributed_coverage'] > rates['clustered_even']
            }

        # Hypothesis 2: Head count effects
        head_count_patterns = {}
        for pattern_name in rates:
            if 'heads_' in pattern_name and '_optimal' in pattern_name:
                n_heads = int(pattern_name.split('_')[1])
                head_count_patterns[n_heads] = rates[pattern_name]

        summary['head_count_analysis'] = head_count_patterns

        # Hypothesis 3: Spacing effects
        if 'regular_spacing_alt' in rates and 'irregular_spacing' in rates:
            summary['spacing_analysis'] = {
                'regular_spacing': rates['regular_spacing_alt'],
                'irregular_spacing': rates['irregular_spacing'],
                'spacing_matters': rates['regular_spacing_alt'] > rates['irregular_spacing']
            }

        # Hypothesis 4: Dense vs Distributed
        if 'dense_coverage' in rates and 'distributed_coverage' in rates:
            summary['hypothesis_tests']['dense_vs_distributed'] = {
                'dense_rate': rates['dense_coverage'],
                'distributed_rate': rates['distributed_coverage'],
                'hypothesis_supported': rates['distributed_coverage'] > rates['dense_coverage']
            }

        return summary

    def visualize_results(self, all_results):
        """Create visualization of experimental results"""

        pattern_results = all_results['pattern_results']
        success_rates = {name: res.get('success_rate', 0) for name, res in pattern_results.items()}

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Overall success rates
        ax1 = axes[0, 0]
        patterns = list(success_rates.keys())
        rates = list(success_rates.values())
        colors = ['green' if rate > 0.5 else 'red' for rate in rates]

        bars = ax1.bar(range(len(patterns)), rates, color=colors, alpha=0.7)
        ax1.set_xlabel('Pattern')
        ax1.set_ylabel('Success Rate')
        ax1.set_title('Success Rate by Pattern', fontweight='bold')
        ax1.set_xticks(range(len(patterns)))
        ax1.set_xticklabels(patterns, rotation=45, ha='right')

        # Add value labels
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.2f}', ha='center', va='bottom')

        # 2. Head count analysis
        ax2 = axes[0, 1]
        head_counts = []
        head_rates = []

        for pattern_name, rate in success_rates.items():
            if 'heads_' in pattern_name and '_optimal' in pattern_name:
                n_heads = int(pattern_name.split('_')[1])
                head_counts.append(n_heads)
                head_rates.append(rate)

        if head_counts:
            ax2.plot(head_counts, head_rates, 'bo-', linewidth=2, markersize=8)
            ax2.axvline(x=8, color='red', linestyle='--', alpha=0.7, label='Hypothesized optimal')
            ax2.set_xlabel('Number of Heads')
            ax2.set_ylabel('Success Rate')
            ax2.set_title('Success Rate vs Head Count', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()

        # 3. Hypothesis comparison
        ax3 = axes[1, 0]
        comparisons = [
            ('Distributed', 'distributed_coverage'),
            ('Clustered', 'clustered_even'),
            ('Dense', 'dense_coverage'),
            ('Minimal (4)', 'minimal_distributed')
        ]

        comp_names = []
        comp_rates = []
        for name, pattern in comparisons:
            if pattern in success_rates:
                comp_names.append(name)
                comp_rates.append(success_rates[pattern])

        colors = ['green', 'orange', 'red', 'blue'][:len(comp_names)]
        bars = ax3.bar(comp_names, comp_rates, color=colors, alpha=0.7)
        ax3.set_ylabel('Success Rate')
        ax3.set_title('Key Hypothesis Comparison', fontweight='bold')

        for bar, rate in zip(bars, comp_rates):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.2f}', ha='center', va='bottom')

        # 4. Pattern visualization
        ax4 = axes[1, 1]
        key_patterns = ['distributed_coverage', 'clustered_even', 'dense_coverage', 'minimal_distributed']

        for i, pattern_name in enumerate(key_patterns):
            if pattern_name in all_results['test_patterns']:
                heads = all_results['test_patterns'][pattern_name]
                rate = success_rates.get(pattern_name, 0)

                # Plot head positions
                ax4.scatter(heads, [i] * len(heads),
                           c=plt.cm.RdYlGn(rate), s=100, alpha=0.8, label=f'{pattern_name} ({rate:.2f})')

                # Plot all positions lightly
                ax4.scatter(range(32), [i] * 32, c='lightgray', s=20, alpha=0.3)

        ax4.set_xlabel('Head Index')
        ax4.set_ylabel('Pattern')
        ax4.set_title('Head Selection Patterns', fontweight='bold')
        ax4.set_yticks(range(len(key_patterns)))
        ax4.set_yticklabels(key_patterns)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"figures/patching_experiment_results_{timestamp}.png"
        os.makedirs("figures", exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_file}")

        plt.show()

def main():
    """Run the patching experiment"""

    experiment = PatchingExperiment()
    results = experiment.run_comprehensive_test()

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE - KEY FINDINGS:")
    print("=" * 60)

    # Print summary of key hypothesis tests
    summary = results['summary']

    if 'hypothesis_tests' in summary:
        for test_name, test_results in summary['hypothesis_tests'].items():
            print(f"\n{test_name.upper()}:")
            for key, value in test_results.items():
                print(f"  {key}: {value}")

    if 'head_count_analysis' in summary:
        print(f"\nHEAD COUNT ANALYSIS:")
        for count, rate in sorted(summary['head_count_analysis'].items()):
            print(f"  {count} heads: {rate:.2%} success")

    print(f"\nFull results saved to JSON file for detailed analysis.")

if __name__ == "__main__":
    main()