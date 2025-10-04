#!/usr/bin/env python3
"""
Test Distributed Coverage Hypothesis - PROPER ACTIVATION PATCHING
================================================================

This script uses the correct activation patching methodology from ../working_scripts/
to test whether distributed coverage across attention heads predicts success
in fixing the 9.8 vs 9.11 numerical bug.

Based on the methodology from validate_even_heads.py and verify_llama_bug.py
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
from contextlib import contextmanager
import json
from datetime import datetime
import time
import matplotlib.pyplot as plt
import os

class ProperPatchingExperiment:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        self.model.eval()

        self.n_heads = 32
        self.layer_idx = 10
        self.saved_activations = {}
        self.hooks = []

    def get_attention_module(self, layer_idx: int):
        return self.model.model.layers[layer_idx].self_attn

    def save_activation_hook(self, key: str):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self.saved_activations[key] = hidden_states.detach().cpu()
        return hook_fn

    def selective_patch_hook(self, saved_activation: torch.Tensor, head_indices: List[int]):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            batch_size, seq_len, hidden_size = hidden_states.shape
            head_dim = hidden_size // self.n_heads

            hidden_states_reshaped = hidden_states.view(batch_size, seq_len, self.n_heads, head_dim)
            saved_reshaped = saved_activation.to(hidden_states.device).view(batch_size, -1, self.n_heads, head_dim)

            new_hidden = hidden_states_reshaped.clone()
            min_seq_len = min(seq_len, saved_reshaped.shape[1])

            for head_idx in head_indices:
                new_hidden[:, :min_seq_len, head_idx, :] = saved_reshaped[:, :min_seq_len, head_idx, :]

            new_hidden = new_hidden.view(batch_size, seq_len, hidden_size)

            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden
        return hook_fn

    @contextmanager
    def save_activation_context(self, prompt: str):
        try:
            module = self.get_attention_module(self.layer_idx)
            key = f"layer_{self.layer_idx}_attention"

            hook = module.register_forward_hook(self.save_activation_hook(key))
            self.hooks.append(hook)

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                _ = self.model(**inputs)

            yield self.saved_activations

        finally:
            for hook in self.hooks:
                hook.remove()
            self.hooks.clear()

    @contextmanager
    def patch_activation_context(self, saved_activation: torch.Tensor, head_indices: List[int]):
        try:
            module = self.get_attention_module(self.layer_idx)
            hook = module.register_forward_hook(
                self.selective_patch_hook(saved_activation, head_indices)
            )
            self.hooks.append(hook)
            yield
        finally:
            for hook in self.hooks:
                hook.remove()
            self.hooks.clear()

    def generate(self, prompt: str, max_new_tokens: int = 20) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )

        generated = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        return generated

    def check_bug_fixed(self, output: str) -> bool:
        """Check if the model correctly identifies 9.8 as bigger than 9.11"""
        output_lower = output.lower()

        # Patterns indicating 9.8 is correctly identified as bigger
        correct_patterns = [
            "9.8 is bigger", "9.8 is larger", "9.8 is greater",
            "9.8 is more", "9.8 is the bigger", "9.8 is the larger"
        ]

        # Patterns indicating the bug (9.11 incorrectly identified as bigger)
        bug_patterns = [
            "9.11 is bigger", "9.11 is larger", "9.11 is greater",
            "9.11 is more", "9.11 is the bigger", "9.11 is the larger"
        ]

        has_correct = any(pattern in output_lower for pattern in correct_patterns)
        has_bug = any(pattern in output_lower for pattern in bug_patterns)

        # Also check for simple "9.8" as first answer
        if not has_correct and not has_bug:
            # Look for clean 9.8 answer
            lines = output.strip().split('\n')
            if lines and '9.8' in lines[0] and '9.11' not in lines[0]:
                has_correct = True

        return has_correct and not has_bug

    def calculate_coverage_metrics(self, heads: List[int]) -> Dict:
        """Calculate coverage metrics for a head selection"""
        heads = sorted(heads)
        n_heads = len(heads)

        # Coverage efficiency (proportion of space covered)
        coverage_efficiency = n_heads / 32

        # Spatial distribution
        gaps = [heads[i+1] - heads[i] for i in range(len(heads)-1)]
        mean_gap = np.mean(gaps) if gaps else 0
        gap_variance = np.var(gaps) if gaps else 0
        gap_regularity = 1 / (1 + gap_variance) if gap_variance > 0 else 1

        # Coverage uniformity
        expected_gap = 32 / n_heads if n_heads > 1 else 32
        gap_deviation = np.mean([abs(gap - expected_gap) for gap in gaps]) if gaps else 0

        # Span coverage
        span = heads[-1] - heads[0] if len(heads) > 1 else 0
        span_efficiency = span / 31 if span > 0 else 0  # 31 is max possible span

        return {
            'coverage_efficiency': coverage_efficiency,
            'mean_gap': mean_gap,
            'gap_variance': gap_variance,
            'gap_regularity': gap_regularity,
            'gap_deviation': gap_deviation,
            'span': span,
            'span_efficiency': span_efficiency
        }

    def test_head_subset(self, head_indices: List[int], n_trials: int = 50, name: str = "") -> Dict:
        """Test a specific subset of heads"""
        success_count = 0

        # Use exact prompts from the working methodology
        correct_prompt = "Which is bigger: 9.8 or 9.11?\nAnswer:"
        buggy_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"

        # Save activation once from correct context
        with self.save_activation_context(correct_prompt) as saved:
            correct_activation = saved[f"layer_{self.layer_idx}_attention"]

        print(f"Testing {name} ({len(head_indices)} heads)...")
        start_time = time.time()

        sample_outputs = []

        for trial in range(n_trials):
            with self.patch_activation_context(correct_activation, head_indices):
                output = self.generate(buggy_prompt, max_new_tokens=20)

            if self.check_bug_fixed(output):
                success_count += 1

            # Store first few outputs for inspection
            if trial < 3:
                sample_outputs.append(output.strip())

            # Progress indicator
            if (trial + 1) % 10 == 0:
                progress = (trial + 1) / n_trials
                bar_length = 20
                filled = int(bar_length * progress)
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"\r  [{bar}] {trial+1}/{n_trials} - {success_count} successes", end='')

        elapsed = time.time() - start_time
        success_rate = success_count / n_trials

        # Calculate 95% CI
        se = np.sqrt(success_rate * (1 - success_rate) / n_trials)
        ci_lower = max(0, success_rate - 1.96 * se)
        ci_upper = min(1, success_rate + 1.96 * se)

        print(f"\n  Result: {success_rate:.1%} [{ci_lower:.1%}, {ci_upper:.1%}] - {elapsed:.1f}s")

        # Show sample outputs
        print(f"  Samples: {sample_outputs[:2]}")

        # Calculate coverage metrics
        metrics = self.calculate_coverage_metrics(head_indices)

        return {
            'success_rate': success_rate,
            'success_count': success_count,
            'n_trials': n_trials,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'time_seconds': elapsed,
            'head_indices': head_indices,
            'coverage_metrics': metrics,
            'sample_outputs': sample_outputs
        }

    def run_comprehensive_test(self, n_trials: int = 50) -> Dict:
        """Run comprehensive distributed coverage test"""

        print("=" * 60)
        print("DISTRIBUTED COVERAGE HYPOTHESIS - PROPER PATCHING")
        print("=" * 60)
        print(f"Model: {self.model_name}")
        print(f"Layer: {self.layer_idx}")
        print(f"Trials per pattern: {n_trials}")
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
            'heads_12_optimal': [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 31],

            # Spacing tests
            'irregular_spacing': [0, 1, 5, 12, 14, 18, 26, 30],
            'regular_spacing_alt': [1, 5, 9, 13, 17, 21, 25, 29],

            # Known patterns from previous experiments
            'known_working_even_1': [2, 4, 6, 8, 10, 12, 14, 16],
            'known_working_even_2': [16, 18, 20, 22, 24, 26, 28, 30],

            # Controls
            'random_8_heads': [1, 3, 7, 11, 15, 19, 23, 29],
            'all_even_heads': list(range(0, 32, 2)),
            'all_odd_heads': list(range(1, 32, 2)),
        }

        # Store all results
        results = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'layer': self.layer_idx,
            'n_trials': n_trials,
            'test_patterns': test_patterns,
            'pattern_results': {},
            'analysis': {}
        }

        # Test each pattern
        for pattern_name, head_indices in test_patterns.items():
            try:
                pattern_results = self.test_head_subset(head_indices, n_trials, pattern_name)
                results['pattern_results'][pattern_name] = pattern_results
                print()
            except Exception as e:
                print(f"Error testing pattern {pattern_name}: {e}")
                results['pattern_results'][pattern_name] = {
                    'error': str(e),
                    'success_rate': 0.0
                }

        # Analyze results
        results['analysis'] = self.analyze_results(results['pattern_results'])

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"proper_patching_distributed_coverage_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {output_file}")

        # Create visualization
        self.visualize_results(results)

        return results

    def analyze_results(self, pattern_results: Dict) -> Dict:
        """Analyze the pattern results to test hypotheses"""

        analysis = {
            'hypothesis_tests': {},
            'success_rates_by_pattern': {},
            'head_count_analysis': {},
            'coverage_correlation': {}
        }

        # Extract success rates and metrics
        for pattern_name, results in pattern_results.items():
            if 'success_rate' in results:
                analysis['success_rates_by_pattern'][pattern_name] = results['success_rate']

        rates = analysis['success_rates_by_pattern']

        # Hypothesis 1: Distributed vs Clustered
        if 'distributed_coverage' in rates and 'clustered_even' in rates:
            analysis['hypothesis_tests']['distributed_vs_clustered'] = {
                'distributed_rate': rates['distributed_coverage'],
                'clustered_rate': rates['clustered_even'],
                'difference': rates['distributed_coverage'] - rates['clustered_even'],
                'hypothesis_supported': rates['distributed_coverage'] > rates['clustered_even']
            }

        # Hypothesis 2: Head count effects
        head_count_patterns = {}
        for pattern_name in rates:
            if 'heads_' in pattern_name and '_optimal' in pattern_name:
                try:
                    n_heads = int(pattern_name.split('_')[1])
                    head_count_patterns[n_heads] = rates[pattern_name]
                except:
                    pass

        analysis['head_count_analysis'] = head_count_patterns

        # Hypothesis 3: Even vs Odd
        if 'all_even_heads' in rates and 'all_odd_heads' in rates:
            analysis['hypothesis_tests']['even_vs_odd'] = {
                'even_rate': rates['all_even_heads'],
                'odd_rate': rates['all_odd_heads'],
                'difference': rates['all_even_heads'] - rates['all_odd_heads'],
                'even_advantage': rates['all_even_heads'] > rates['all_odd_heads']
            }

        # Hypothesis 4: Coverage metrics correlation
        coverage_data = []
        for pattern_name, results in pattern_results.items():
            if 'coverage_metrics' in results and 'success_rate' in results:
                metrics = results['coverage_metrics']
                coverage_data.append({
                    'pattern': pattern_name,
                    'success_rate': results['success_rate'],
                    'gap_regularity': metrics['gap_regularity'],
                    'span_efficiency': metrics['span_efficiency'],
                    'coverage_efficiency': metrics['coverage_efficiency']
                })

        if len(coverage_data) > 3:
            # Calculate correlations
            success_rates = [d['success_rate'] for d in coverage_data]
            gap_regularities = [d['gap_regularity'] for d in coverage_data]
            span_efficiencies = [d['span_efficiency'] for d in coverage_data]

            gap_corr = np.corrcoef(success_rates, gap_regularities)[0, 1]
            span_corr = np.corrcoef(success_rates, span_efficiencies)[0, 1]

            analysis['coverage_correlation'] = {
                'gap_regularity_correlation': gap_corr,
                'span_efficiency_correlation': span_corr,
                'coverage_data': coverage_data
            }

        return analysis

    def visualize_results(self, results: Dict):
        """Create visualization of experimental results"""

        pattern_results = results['pattern_results']
        success_rates = {name: res.get('success_rate', 0) for name, res in pattern_results.items()}

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Overall success rates
        ax1 = axes[0, 0]
        patterns = list(success_rates.keys())
        rates = list(success_rates.values())

        # Color by performance
        colors = ['green' if rate > 0.8 else 'orange' if rate > 0.5 else 'red' for rate in rates]

        bars = ax1.bar(range(len(patterns)), rates, color=colors, alpha=0.7)
        ax1.set_xlabel('Pattern')
        ax1.set_ylabel('Success Rate')
        ax1.set_title('Success Rate by Pattern', fontweight='bold')
        ax1.set_xticks(range(len(patterns)))
        ax1.set_xticklabels(patterns, rotation=45, ha='right')
        ax1.set_ylim(0, 1)

        # Add value labels
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.2f}', ha='center', va='bottom', fontsize=8)

        # 2. Head count analysis
        ax2 = axes[0, 1]
        if 'head_count_analysis' in results['analysis']:
            head_counts = list(results['analysis']['head_count_analysis'].keys())
            head_rates = list(results['analysis']['head_count_analysis'].values())

            if head_counts:
                ax2.plot(head_counts, head_rates, 'bo-', linewidth=2, markersize=8)
                ax2.set_xlabel('Number of Heads')
                ax2.set_ylabel('Success Rate')
                ax2.set_title('Success Rate vs Head Count', fontweight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(0, 1)

        # 3. Coverage correlation
        ax3 = axes[1, 0]
        if 'coverage_correlation' in results['analysis'] and 'coverage_data' in results['analysis']['coverage_correlation']:
            data = results['analysis']['coverage_correlation']['coverage_data']
            gap_regs = [d['gap_regularity'] for d in data]
            success_rates_corr = [d['success_rate'] for d in data]

            ax3.scatter(gap_regs, success_rates_corr, alpha=0.7, s=100)
            ax3.set_xlabel('Gap Regularity')
            ax3.set_ylabel('Success Rate')
            ax3.set_title('Success vs Gap Regularity', fontweight='bold')
            ax3.grid(True, alpha=0.3)

            # Add correlation coefficient
            corr = results['analysis']['coverage_correlation']['gap_regularity_correlation']
            ax3.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax3.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        # 4. Key hypothesis comparison
        ax4 = axes[1, 1]
        key_comparisons = []
        key_rates = []
        key_names = []

        if 'distributed_coverage' in success_rates:
            key_names.append('Distributed')
            key_rates.append(success_rates['distributed_coverage'])
        if 'clustered_even' in success_rates:
            key_names.append('Clustered')
            key_rates.append(success_rates['clustered_even'])
        if 'all_even_heads' in success_rates:
            key_names.append('All Even')
            key_rates.append(success_rates['all_even_heads'])
        if 'all_odd_heads' in success_rates:
            key_names.append('All Odd')
            key_rates.append(success_rates['all_odd_heads'])

        if key_names:
            colors = ['green', 'orange', 'blue', 'red'][:len(key_names)]
            bars = ax4.bar(key_names, key_rates, color=colors, alpha=0.7)
            ax4.set_ylabel('Success Rate')
            ax4.set_title('Key Hypothesis Comparison', fontweight='bold')
            ax4.set_ylim(0, 1)

            for bar, rate in zip(bars, key_rates):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{rate:.2f}', ha='center', va='bottom')

        plt.tight_layout()

        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"figures/proper_patching_results_{timestamp}.png"
        os.makedirs("figures", exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_file}")

        plt.show()

def main():
    """Run the proper patching experiment"""

    start_time = time.time()

    experiment = ProperPatchingExperiment(device="cuda")
    results = experiment.run_comprehensive_test(n_trials=50)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE - KEY FINDINGS:")
    print("=" * 60)

    # Print summary of key hypothesis tests
    analysis = results['analysis']

    if 'hypothesis_tests' in analysis:
        for test_name, test_results in analysis['hypothesis_tests'].items():
            print(f"\n{test_name.upper()}:")
            for key, value in test_results.items():
                print(f"  {key}: {value}")

    if 'head_count_analysis' in analysis:
        print(f"\nHEAD COUNT ANALYSIS:")
        for count, rate in sorted(analysis['head_count_analysis'].items()):
            print(f"  {count} heads: {rate:.1%} success")

    total_time = time.time() - start_time
    print(f"\nExperiment completed in {total_time:.1f} seconds")

if __name__ == "__main__":
    main()