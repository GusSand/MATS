#!/usr/bin/env python3
"""
Investigation: Why Specific 8 Even Heads Are Successful
=======================================================

This experiment tests the critical question: Are ANY 8 even heads sufficient,
or do specific combinations matter? We'll test:

1. Random 8 even head combinations
2. Specific successful combinations from previous research
3. What makes successful combinations different from unsuccessful ones

Based on previous research findings that "ANY 8 even heads achieve 100% success",
but we need to verify this and understand WHY.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager
import json
from datetime import datetime
import warnings
import os
import random
from itertools import combinations
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Model configuration
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LAYER_OF_INTEREST = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SpecificHeadInvestigator:
    """Investigate why specific 8 even heads are successful in patching"""

    def __init__(self, model_name: str = MODEL_NAME, device: str = DEVICE):
        self.device = device
        self.model_name = model_name

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()

        self.n_heads = 32
        self.layer_idx = LAYER_OF_INTEREST
        self.saved_activations = {}
        self.hooks = []

        # Define even heads for testing
        self.all_even_heads = [i for i in range(32) if i % 2 == 0]  # [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]

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
        output_lower = output.lower()

        correct_patterns = ["9.8 is bigger", "9.8 is larger", "9.8 is greater", "9.8"]
        bug_patterns = ["9.11 is bigger", "9.11 is larger", "9.11 is greater"]

        has_correct = any(pattern in output_lower for pattern in correct_patterns)
        has_bug = any(pattern in output_lower for pattern in bug_patterns)

        return has_correct and not has_bug

    def test_head_combination(self, head_indices: List[int], n_trials: int = 20) -> Dict:
        """Test a specific combination of heads"""
        success_count = 0
        outputs = []

        # Save activation from correct format
        correct_prompt = "Which is bigger: 9.8 or 9.11?\nAnswer:"
        with self.save_activation_context(correct_prompt) as saved:
            correct_activation = saved[f"layer_{self.layer_idx}_attention"]

        buggy_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"

        for trial in range(n_trials):
            with self.patch_activation_context(correct_activation, head_indices):
                output = self.generate(buggy_prompt, max_new_tokens=20)

            if self.check_bug_fixed(output):
                success_count += 1
            outputs.append(output.strip())

        success_rate = success_count / n_trials

        return {
            'head_indices': head_indices,
            'n_heads': len(head_indices),
            'success_rate': success_rate,
            'success_count': success_count,
            'n_trials': n_trials,
            'outputs': outputs,
            'head_pattern': self.analyze_head_pattern(head_indices)
        }

    def analyze_head_pattern(self, head_indices: List[int]) -> Dict:
        """Analyze the pattern of head indices"""
        if not head_indices:
            return {'type': 'empty'}

        # Check if all are even
        all_even = all(h % 2 == 0 for h in head_indices)
        all_odd = all(h % 2 == 1 for h in head_indices)

        # Check spatial patterns
        is_consecutive = all(head_indices[i] + 2 == head_indices[i+1] for i in range(len(head_indices)-1))
        is_first_half = all(h < 16 for h in head_indices)
        is_second_half = all(h >= 16 for h in head_indices)

        # Check distribution
        first_half_count = sum(1 for h in head_indices if h < 16)
        second_half_count = len(head_indices) - first_half_count

        return {
            'type': 'even' if all_even else ('odd' if all_odd else 'mixed'),
            'all_even': all_even,
            'all_odd': all_odd,
            'consecutive': is_consecutive,
            'first_half_only': is_first_half,
            'second_half_only': is_second_half,
            'first_half_count': first_half_count,
            'second_half_count': second_half_count,
            'balance_ratio': first_half_count / len(head_indices),
            'min_head': min(head_indices),
            'max_head': max(head_indices),
            'span': max(head_indices) - min(head_indices)
        }

    def test_systematic_combinations(self, n_random_tests: int = 50) -> pd.DataFrame:
        """Test systematic combinations to understand what makes 8 heads successful"""
        results = []

        print("Testing systematic head combinations...")

        # 1. Test known successful patterns from previous research
        print("\n1. TESTING KNOWN SUCCESSFUL PATTERNS")

        # First 8 even heads (known to work)
        first_8_even = [0, 2, 4, 6, 8, 10, 12, 14]
        result = self.test_head_combination(first_8_even, n_trials=30)
        result['combination_type'] = 'first_8_even'
        results.append(result)
        print(f"   First 8 even: {result['success_rate']:.1%}")

        # Last 8 even heads (known to work)
        last_8_even = [16, 18, 20, 22, 24, 26, 28, 30]
        result = self.test_head_combination(last_8_even, n_trials=30)
        result['combination_type'] = 'last_8_even'
        results.append(result)
        print(f"   Last 8 even: {result['success_rate']:.1%}")

        # Every other even head (known to work)
        every_other_even = [0, 4, 8, 12, 16, 20, 24, 28]
        result = self.test_head_combination(every_other_even, n_trials=30)
        result['combination_type'] = 'every_other_even'
        results.append(result)
        print(f"   Every other even: {result['success_rate']:.1%}")

        # 2. Test random 8 even head combinations
        print(f"\n2. TESTING {n_random_tests} RANDOM 8 EVEN HEAD COMBINATIONS")

        for i in range(n_random_tests):
            random_8_even = sorted(random.sample(self.all_even_heads, 8))
            result = self.test_head_combination(random_8_even, n_trials=10)  # Fewer trials for speed
            result['combination_type'] = 'random_8_even'
            result['random_seed'] = i
            results.append(result)

            if (i + 1) % 10 == 0:
                print(f"   Completed {i+1}/{n_random_tests} random combinations...")

        # 3. Test known failing patterns
        print(f"\n3. TESTING KNOWN FAILING PATTERNS")

        # 4 even heads (known to fail)
        first_4_even = [0, 2, 4, 6]
        result = self.test_head_combination(first_4_even, n_trials=20)
        result['combination_type'] = 'first_4_even'
        results.append(result)
        print(f"   First 4 even: {result['success_rate']:.1%}")

        # 8 odd heads (known to fail)
        first_8_odd = [1, 3, 5, 7, 9, 11, 13, 15]
        result = self.test_head_combination(first_8_odd, n_trials=20)
        result['combination_type'] = 'first_8_odd'
        results.append(result)
        print(f"   First 8 odd: {result['success_rate']:.1%}")

        # Mixed 4 even + 4 odd (known to fail)
        mixed_4_4 = [0, 2, 4, 6, 1, 3, 5, 7]
        result = self.test_head_combination(mixed_4_4, n_trials=20)
        result['combination_type'] = 'mixed_4even_4odd'
        results.append(result)
        print(f"   Mixed 4+4: {result['success_rate']:.1%}")

        # 4. Test edge cases
        print(f"\n4. TESTING EDGE CASES")

        # 12 even heads (more than needed)
        twelve_even = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
        result = self.test_head_combination(twelve_even, n_trials=20)
        result['combination_type'] = 'twelve_even'
        results.append(result)
        print(f"   12 even heads: {result['success_rate']:.1%}")

        # 6 even heads (fewer than threshold)
        six_even = [0, 2, 4, 6, 8, 10]
        result = self.test_head_combination(six_even, n_trials=20)
        result['combination_type'] = 'six_even'
        results.append(result)
        print(f"   6 even heads: {result['success_rate']:.1%}")

        return pd.DataFrame(results)

    def analyze_results(self, df: pd.DataFrame) -> Dict:
        """Analyze the results to understand patterns"""
        analysis = {}

        # Group by combination type
        by_type = df.groupby('combination_type')['success_rate'].agg(['mean', 'std', 'count'])
        analysis['by_combination_type'] = by_type.to_dict()

        # Analyze random 8 even combinations specifically
        random_8_even = df[df['combination_type'] == 'random_8_even']
        if not random_8_even.empty:
            analysis['random_8_even'] = {
                'mean_success': random_8_even['success_rate'].mean(),
                'std_success': random_8_even['success_rate'].std(),
                'min_success': random_8_even['success_rate'].min(),
                'max_success': random_8_even['success_rate'].max(),
                'perfect_success_count': (random_8_even['success_rate'] == 1.0).sum(),
                'zero_success_count': (random_8_even['success_rate'] == 0.0).sum(),
                'total_tested': len(random_8_even)
            }

        # Test the claim "ANY 8 even heads work"
        any_8_even_claim = True
        if not random_8_even.empty:
            any_8_even_claim = random_8_even['success_rate'].min() > 0.8  # Allow some variance

        analysis['any_8_even_claim_supported'] = any_8_even_claim

        return analysis

    def visualize_results(self, df: pd.DataFrame, analysis: Dict):
        """Create comprehensive visualization of results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Plot 1: Success rate by combination type
        ax1 = axes[0, 0]
        type_summary = df.groupby('combination_type')['success_rate'].agg(['mean', 'std'])
        type_summary.plot(kind='bar', y='mean', yerr='std', ax=ax1, capsize=5)
        ax1.set_title('Success Rate by Combination Type')
        ax1.set_ylabel('Success Rate')
        ax1.set_xlabel('Combination Type')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Random 8 even heads distribution
        ax2 = axes[0, 1]
        random_8_even = df[df['combination_type'] == 'random_8_even']
        if not random_8_even.empty:
            ax2.hist(random_8_even['success_rate'], bins=20, alpha=0.7, edgecolor='black')
            ax2.axvline(random_8_even['success_rate'].mean(), color='red', linestyle='--',
                       label=f'Mean: {random_8_even["success_rate"].mean():.2f}')
            ax2.set_title('Distribution of Random 8 Even Head Success Rates')
            ax2.set_xlabel('Success Rate')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # Plot 3: Success rate vs number of heads
        ax3 = axes[0, 2]
        df_with_n_heads = df.copy()
        df_with_n_heads['n_heads'] = df_with_n_heads['head_indices'].apply(len)
        head_count_summary = df_with_n_heads.groupby('n_heads')['success_rate'].agg(['mean', 'std'])
        head_count_summary.plot(kind='bar', y='mean', yerr='std', ax=ax3, capsize=5)
        ax3.set_title('Success Rate vs Number of Heads')
        ax3.set_ylabel('Success Rate')
        ax3.set_xlabel('Number of Heads')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Head type analysis
        ax4 = axes[1, 0]
        type_data = []
        for _, row in df.iterrows():
            pattern = row['head_pattern']
            type_data.append({
                'combination': row['combination_type'],
                'success_rate': row['success_rate'],
                'all_even': pattern['all_even'],
                'head_type': 'Even' if pattern['all_even'] else ('Odd' if pattern['all_odd'] else 'Mixed')
            })

        type_df = pd.DataFrame(type_data)
        type_by_head = type_df.groupby('head_type')['success_rate'].agg(['mean', 'std'])
        type_by_head.plot(kind='bar', y='mean', yerr='std', ax=ax4, capsize=5)
        ax4.set_title('Success Rate by Head Type')
        ax4.set_ylabel('Success Rate')
        ax4.set_xlabel('Head Type')
        ax4.grid(True, alpha=0.3)

        # Plot 5: Spatial distribution analysis
        ax5 = axes[1, 1]
        spatial_data = []
        for _, row in df.iterrows():
            if row['combination_type'] == 'random_8_even':
                pattern = row['head_pattern']
                spatial_data.append({
                    'balance_ratio': pattern['balance_ratio'],
                    'success_rate': row['success_rate'],
                    'span': pattern['span']
                })

        if spatial_data:
            spatial_df = pd.DataFrame(spatial_data)
            scatter = ax5.scatter(spatial_df['balance_ratio'], spatial_df['success_rate'],
                                c=spatial_df['span'], cmap='viridis', alpha=0.7)
            ax5.set_xlabel('First Half Balance Ratio')
            ax5.set_ylabel('Success Rate')
            ax5.set_title('Spatial Distribution vs Success')
            plt.colorbar(scatter, ax=ax5, label='Head Span')
            ax5.grid(True, alpha=0.3)

        # Plot 6: Summary statistics
        ax6 = axes[1, 2]
        ax6.axis('off')

        summary_text = "INVESTIGATION SUMMARY\n" + "="*30 + "\n\n"

        if 'random_8_even' in analysis:
            random_stats = analysis['random_8_even']
            summary_text += f"RANDOM 8 EVEN HEADS:\n"
            summary_text += f"  Tested: {random_stats['total_tested']}\n"
            summary_text += f"  Mean success: {random_stats['mean_success']:.2%}\n"
            summary_text += f"  Min success: {random_stats['min_success']:.2%}\n"
            summary_text += f"  Max success: {random_stats['max_success']:.2%}\n"
            summary_text += f"  Perfect (100%): {random_stats['perfect_success_count']}\n"
            summary_text += f"  Failed (0%): {random_stats['zero_success_count']}\n\n"

        summary_text += f"CLAIM VALIDATION:\n"
        claim_supported = analysis.get('any_8_even_claim_supported', False)
        summary_text += f"'ANY 8 even heads work': {'✅' if claim_supported else '❌'}\n\n"

        # Add top performing combination types
        summary_text += f"TOP PERFORMERS:\n"
        top_types = df.groupby('combination_type')['success_rate'].mean().sort_values(ascending=False).head(3)
        for combo_type, success_rate in top_types.items():
            summary_text += f"  {combo_type}: {success_rate:.1%}\n"

        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()
        plt.savefig('/home/paperspace/dev/MATS9/bandwidth/figures/specific_heads_investigation.png',
                   dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Run the investigation into specific even head requirements"""
    print("="*70)
    print("INVESTIGATION: Why Specific 8 Even Heads Are Successful")
    print("="*70)
    print("Testing the claim: 'ANY 8 even heads achieve 100% success'")
    print("And investigating what makes successful combinations work")
    print()

    # Initialize investigator
    investigator = SpecificHeadInvestigator()

    # Run systematic testing
    print("Running systematic head combination testing...")
    df_results = investigator.test_systematic_combinations(n_random_tests=30)

    # Analyze results
    print("\nAnalyzing results...")
    analysis = investigator.analyze_results(df_results)

    # Create visualization
    print("\nCreating visualization...")
    investigator.visualize_results(df_results, analysis)

    # Print detailed results
    print("\n" + "="*70)
    print("DETAILED INVESTIGATION RESULTS")
    print("="*70)

    print("\nSUCCESS RATES BY COMBINATION TYPE:")
    print("-" * 50)
    type_summary = df_results.groupby('combination_type')['success_rate'].agg(['mean', 'std', 'count'])
    print(type_summary)

    if 'random_8_even' in analysis:
        print(f"\nRANDOM 8 EVEN HEAD ANALYSIS:")
        print("-" * 50)
        random_stats = analysis['random_8_even']
        for key, value in random_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")

    # Test the critical claim
    print(f"\n" + "="*70)
    print("CRITICAL CLAIM VALIDATION")
    print("="*70)

    claim_supported = analysis.get('any_8_even_claim_supported', False)
    print(f"Claim: 'ANY 8 even heads achieve 100% success'")
    print(f"Status: {'✅ SUPPORTED' if claim_supported else '❌ REJECTED'}")

    if not claim_supported:
        random_8_even = df_results[df_results['combination_type'] == 'random_8_even']
        if not random_8_even.empty:
            failed_combinations = random_8_even[random_8_even['success_rate'] < 0.8]
            print(f"\nFound {len(failed_combinations)} combinations with <80% success rate:")
            for _, row in failed_combinations.iterrows():
                print(f"  Heads {row['head_indices']}: {row['success_rate']:.1%} success")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'/home/paperspace/dev/MATS9/bandwidth/specific_heads_investigation_{timestamp}.json'

    # Convert to JSON-serializable format
    results_dict = {
        'investigation_summary': analysis,
        'detailed_results': df_results.to_dict('records'),
        'claim_validation': {
            'claim': 'ANY 8 even heads achieve 100% success',
            'supported': claim_supported,
            'evidence': analysis.get('random_8_even', {})
        }
    }

    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)

    print(f"\n✅ Investigation complete!")
    print(f"✅ Results saved to {results_file}")
    print(f"✅ Visualization saved to /home/paperspace/dev/MATS9/bandwidth/figures/specific_heads_investigation.png")

    return df_results, analysis


if __name__ == "__main__":
    main()