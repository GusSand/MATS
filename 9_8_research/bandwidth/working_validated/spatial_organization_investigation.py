#!/usr/bin/env python3
"""
Comprehensive Investigation: Spatial Organization Hypothesis
===========================================================

This experiment tests all three aspects:
1. Test spatial organization hypothesis with structured patterns
2. Analyze computational properties requiring spatial organization
3. Cross-validate failed combinations for consistency

Based on our discovery that the "ANY 8 even heads" claim was overgeneralized,
we now test whether spatial organization is the key requirement.
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
import scipy.stats as stats

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Model configuration
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LAYER_OF_INTEREST = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SpatialOrganizationInvestigator:
    """Comprehensive investigation of spatial organization hypothesis"""

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

        # Define all even heads
        self.all_even_heads = [i for i in range(32) if i % 2 == 0]

        # Store our known failed combinations from previous investigation
        self.known_failed_combinations = [
            [0, 4, 12, 14, 16, 18, 26, 30],
            [2, 4, 6, 8, 12, 14, 18, 30],
            [0, 6, 10, 14, 18, 22, 24, 30],
            [0, 2, 4, 12, 14, 22, 26, 30],
            [2, 4, 6, 8, 12, 14, 18, 26],
            [6, 12, 14, 16, 18, 20, 22, 26],
            [0, 2, 4, 14, 16, 18, 24, 28],
            [2, 4, 6, 14, 22, 24, 28, 30],
            [0, 2, 4, 10, 14, 16, 22, 28],
            [0, 2, 14, 18, 22, 24, 26, 30],
            [0, 4, 6, 10, 14, 16, 20, 22]
        ]

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
            'spatial_metrics': self.analyze_spatial_organization(head_indices)
        }

    def analyze_spatial_organization(self, head_indices: List[int]) -> Dict:
        """Analyze spatial organization metrics of head indices"""
        if not head_indices:
            return {'type': 'empty'}

        head_indices = sorted(head_indices)

        # Basic metrics
        span = max(head_indices) - min(head_indices)
        gaps = [head_indices[i+1] - head_indices[i] for i in range(len(head_indices)-1)]

        # Spatial distribution metrics
        mean_gap = np.mean(gaps) if gaps else 0
        std_gap = np.std(gaps) if gaps else 0
        max_gap = max(gaps) if gaps else 0
        min_gap = min(gaps) if gaps else 0

        # Regularity metrics
        gap_regularity = 1 / (1 + std_gap) if std_gap > 0 else 1  # Higher = more regular
        coverage_efficiency = len(head_indices) / (span + 2) if span > 0 else 1  # Higher = better coverage

        # Pattern detection
        is_consecutive = all(gap == 2 for gap in gaps) if gaps else True
        is_uniform_spacing = len(set(gaps)) <= 1 if gaps else True

        # Clustering analysis
        clusters = self.detect_clusters(head_indices)
        n_clusters = len(clusters)
        cluster_sizes = [len(cluster) for cluster in clusters]

        # Balance metrics
        first_half = [h for h in head_indices if h < 16]
        second_half = [h for h in head_indices if h >= 16]
        balance_ratio = len(first_half) / len(head_indices) if head_indices else 0.5

        return {
            'span': span,
            'mean_gap': mean_gap,
            'std_gap': std_gap,
            'max_gap': max_gap,
            'min_gap': min_gap,
            'gap_regularity': gap_regularity,
            'coverage_efficiency': coverage_efficiency,
            'is_consecutive': is_consecutive,
            'is_uniform_spacing': is_uniform_spacing,
            'n_clusters': n_clusters,
            'cluster_sizes': cluster_sizes,
            'largest_cluster': max(cluster_sizes) if cluster_sizes else 0,
            'balance_ratio': balance_ratio,
            'gaps': gaps
        }

    def detect_clusters(self, head_indices: List[int], max_gap: int = 4) -> List[List[int]]:
        """Detect clusters of heads based on gap threshold"""
        if not head_indices:
            return []

        clusters = []
        current_cluster = [head_indices[0]]

        for i in range(1, len(head_indices)):
            if head_indices[i] - head_indices[i-1] <= max_gap:
                current_cluster.append(head_indices[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [head_indices[i]]

        clusters.append(current_cluster)
        return clusters

    def generate_spatial_patterns(self) -> Dict[str, List[int]]:
        """Generate various spatial patterns for testing"""
        patterns = {}

        # Known successful patterns
        patterns['first_8_even'] = [0, 2, 4, 6, 8, 10, 12, 14]
        patterns['last_8_even'] = [16, 18, 20, 22, 24, 26, 28, 30]
        patterns['every_other_even'] = [0, 4, 8, 12, 16, 20, 24, 28]

        # Test new consecutive patterns
        patterns['middle_8_even'] = [8, 10, 12, 14, 16, 18, 20, 22]
        patterns['offset_consecutive_1'] = [2, 4, 6, 8, 10, 12, 14, 16]
        patterns['offset_consecutive_2'] = [4, 6, 8, 10, 12, 14, 16, 18]

        # Test uniform spacing patterns
        patterns['every_3rd_even'] = [0, 6, 12, 18, 2, 8, 14, 20]  # Spacing of 6, then reorder
        patterns['every_3rd_even'] = sorted(patterns['every_3rd_even'])
        patterns['uniform_spacing_4'] = [0, 4, 8, 12, 16, 20, 24, 28]  # Same as every_other
        patterns['uniform_spacing_6'] = [0, 6, 12, 18, 24, 30, 2, 8]  # Spacing of 6
        patterns['uniform_spacing_6'] = sorted(patterns['uniform_spacing_6'])

        # Test balanced patterns
        patterns['balanced_quarters'] = [0, 2, 8, 10, 16, 18, 24, 26]  # 2 from each quarter
        patterns['balanced_halves'] = [0, 2, 4, 6, 16, 18, 20, 22]    # 4 from each half
        patterns['balanced_extremes'] = [0, 2, 4, 30, 16, 18, 20, 28] # Mixed extremes

        # Test irregular patterns (likely to fail)
        patterns['irregular_clustered'] = [0, 2, 4, 14, 16, 26, 28, 30]  # Heavy clustering
        patterns['irregular_gaps'] = [0, 8, 10, 14, 18, 22, 26, 30]     # Irregular gaps
        patterns['heavily_skewed'] = [0, 2, 4, 6, 8, 10, 24, 26]       # Skewed to beginning

        return patterns

    def part_1_test_spatial_hypothesis(self) -> pd.DataFrame:
        """Part 1: Test spatial organization hypothesis with structured patterns"""
        print("="*70)
        print("PART 1: TESTING SPATIAL ORGANIZATION HYPOTHESIS")
        print("="*70)

        patterns = self.generate_spatial_patterns()
        results = []

        for pattern_name, head_indices in patterns.items():
            print(f"\nTesting {pattern_name}: {head_indices}")
            result = self.test_head_combination(head_indices, n_trials=25)
            result['pattern_name'] = pattern_name
            result['pattern_type'] = self.classify_pattern_type(pattern_name)
            results.append(result)

            success_rate = result['success_rate']
            spatial_metrics = result['spatial_metrics']
            print(f"  Success: {success_rate:.1%}, Gap regularity: {spatial_metrics['gap_regularity']:.3f}, "
                  f"Coverage: {spatial_metrics['coverage_efficiency']:.3f}")

        return pd.DataFrame(results)

    def part_2_analyze_computational_properties(self, df_patterns: pd.DataFrame) -> Dict:
        """Part 2: Analyze what computational properties require spatial organization"""
        print("\n" + "="*70)
        print("PART 2: ANALYZING COMPUTATIONAL PROPERTIES")
        print("="*70)

        analysis = {}

        # Correlation analysis between spatial metrics and success
        spatial_metrics = ['gap_regularity', 'coverage_efficiency', 'span', 'mean_gap',
                          'std_gap', 'n_clusters', 'largest_cluster', 'balance_ratio']

        correlations = {}
        for metric in spatial_metrics:
            metric_values = [result['spatial_metrics'][metric] for result in df_patterns.to_dict('records')]
            success_rates = df_patterns['success_rate'].values

            if len(set(metric_values)) > 1:  # Only if there's variation
                corr, p_value = stats.pearsonr(metric_values, success_rates)
                correlations[metric] = {'correlation': corr, 'p_value': p_value}
                print(f"{metric:20s}: r={corr:+.3f}, p={p_value:.3f} {'*' if p_value < 0.05 else ''}")

        analysis['spatial_correlations'] = correlations

        # Find the most predictive spatial features
        successful_patterns = df_patterns[df_patterns['success_rate'] > 0.8]
        failed_patterns = df_patterns[df_patterns['success_rate'] < 0.2]

        print(f"\nSUCCESSFUL PATTERNS ({len(successful_patterns)}):")
        for _, row in successful_patterns.iterrows():
            metrics = row['spatial_metrics']
            print(f"  {row['pattern_name']:20s}: regularity={metrics['gap_regularity']:.3f}, "
                  f"coverage={metrics['coverage_efficiency']:.3f}, clusters={metrics['n_clusters']}")

        print(f"\nFAILED PATTERNS ({len(failed_patterns)}):")
        for _, row in failed_patterns.iterrows():
            metrics = row['spatial_metrics']
            print(f"  {row['pattern_name']:20s}: regularity={metrics['gap_regularity']:.3f}, "
                  f"coverage={metrics['coverage_efficiency']:.3f}, clusters={metrics['n_clusters']}")

        # Statistical comparison
        if len(successful_patterns) > 0 and len(failed_patterns) > 0:
            for metric in spatial_metrics:
                successful_values = [row['spatial_metrics'][metric] for _, row in successful_patterns.iterrows()]
                failed_values = [row['spatial_metrics'][metric] for _, row in failed_patterns.iterrows()]

                if len(successful_values) > 1 and len(failed_values) > 1:
                    t_stat, p_value = stats.ttest_ind(successful_values, failed_values)
                    print(f"{metric}: successful_mean={np.mean(successful_values):.3f}, "
                          f"failed_mean={np.mean(failed_values):.3f}, p={p_value:.3f}")

        analysis['successful_characteristics'] = {
            metric: [row['spatial_metrics'][metric] for _, row in successful_patterns.iterrows()]
            for metric in spatial_metrics
        } if len(successful_patterns) > 0 else {}

        analysis['failed_characteristics'] = {
            metric: [row['spatial_metrics'][metric] for _, row in failed_patterns.iterrows()]
            for metric in spatial_metrics
        } if len(failed_patterns) > 0 else {}

        return analysis

    def part_3_cross_validate_failed_combinations(self) -> pd.DataFrame:
        """Part 3: Cross-validate failed combinations for consistency"""
        print("\n" + "="*70)
        print("PART 3: CROSS-VALIDATING FAILED COMBINATIONS")
        print("="*70)

        results = []

        print(f"Re-testing {len(self.known_failed_combinations)} known failed combinations...")

        for i, head_combo in enumerate(self.known_failed_combinations):
            print(f"\nTesting failed combo {i+1}: {head_combo}")
            result = self.test_head_combination(head_combo, n_trials=30)  # More trials for reliability
            result['combo_index'] = i
            result['validation_type'] = 'failed_retest'
            results.append(result)

            print(f"  Success rate: {result['success_rate']:.1%} (expected: ~0%)")

            # Analyze why this combination fails
            spatial_metrics = result['spatial_metrics']
            print(f"  Spatial analysis: regularity={spatial_metrics['gap_regularity']:.3f}, "
                  f"clusters={spatial_metrics['n_clusters']}, span={spatial_metrics['span']}")

        # Test some random combinations for comparison
        print(f"\nTesting 10 new random combinations for comparison...")
        for i in range(10):
            random_combo = sorted(random.sample(self.all_even_heads, 8))
            result = self.test_head_combination(random_combo, n_trials=20)
            result['combo_index'] = i
            result['validation_type'] = 'new_random'
            results.append(result)

            print(f"  Random combo {i+1}: {random_combo} -> {result['success_rate']:.1%}")

        return pd.DataFrame(results)

    def classify_pattern_type(self, pattern_name: str) -> str:
        """Classify pattern type for analysis"""
        if 'consecutive' in pattern_name or 'first_8' in pattern_name or 'last_8' in pattern_name or 'middle_8' in pattern_name:
            return 'consecutive'
        elif 'every' in pattern_name or 'uniform' in pattern_name:
            return 'uniform_spacing'
        elif 'balanced' in pattern_name:
            return 'balanced'
        elif 'irregular' in pattern_name or 'clustered' in pattern_name or 'skewed' in pattern_name:
            return 'irregular'
        else:
            return 'other'

    def create_comprehensive_visualization(self, df_patterns: pd.DataFrame, df_validation: pd.DataFrame, analysis: Dict):
        """Create comprehensive visualization of all results"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))

        # Plot 1: Success rate by pattern type
        ax1 = axes[0, 0]
        pattern_type_success = df_patterns.groupby('pattern_type')['success_rate'].agg(['mean', 'std', 'count'])
        pattern_type_success.plot(kind='bar', y='mean', yerr='std', ax=ax1, capsize=5)
        ax1.set_title('Success Rate by Pattern Type')
        ax1.set_ylabel('Success Rate')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Spatial metrics correlation with success
        ax2 = axes[0, 1]
        if 'spatial_correlations' in analysis:
            metrics = list(analysis['spatial_correlations'].keys())
            correlations = [analysis['spatial_correlations'][m]['correlation'] for m in metrics]
            p_values = [analysis['spatial_correlations'][m]['p_value'] for m in metrics]

            colors = ['red' if p < 0.05 else 'gray' for p in p_values]
            bars = ax2.barh(metrics, correlations, color=colors)
            ax2.set_title('Spatial Metrics vs Success Rate')
            ax2.set_xlabel('Correlation with Success Rate')
            ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax2.grid(True, alpha=0.3)

        # Plot 3: Gap regularity vs success rate
        ax3 = axes[0, 2]
        gap_regularity = [result['spatial_metrics']['gap_regularity'] for result in df_patterns.to_dict('records')]
        success_rates = df_patterns['success_rate'].values
        scatter = ax3.scatter(gap_regularity, success_rates,
                             c=df_patterns['success_rate'], cmap='RdYlGn', s=100, alpha=0.7)
        ax3.set_xlabel('Gap Regularity')
        ax3.set_ylabel('Success Rate')
        ax3.set_title('Gap Regularity vs Success')
        plt.colorbar(scatter, ax=ax3)
        ax3.grid(True, alpha=0.3)

        # Plot 4: Coverage efficiency vs success rate
        ax4 = axes[1, 0]
        coverage_efficiency = [result['spatial_metrics']['coverage_efficiency'] for result in df_patterns.to_dict('records')]
        ax4.scatter(coverage_efficiency, success_rates,
                   c=df_patterns['success_rate'], cmap='RdYlGn', s=100, alpha=0.7)
        ax4.set_xlabel('Coverage Efficiency')
        ax4.set_ylabel('Success Rate')
        ax4.set_title('Coverage Efficiency vs Success')
        ax4.grid(True, alpha=0.3)

        # Plot 5: Number of clusters vs success rate
        ax5 = axes[1, 1]
        n_clusters = [result['spatial_metrics']['n_clusters'] for result in df_patterns.to_dict('records')]
        ax5.scatter(n_clusters, success_rates,
                   c=df_patterns['success_rate'], cmap='RdYlGn', s=100, alpha=0.7)
        ax5.set_xlabel('Number of Clusters')
        ax5.set_ylabel('Success Rate')
        ax5.set_title('Clustering vs Success')
        ax5.grid(True, alpha=0.3)

        # Plot 6: Validation results
        ax6 = axes[1, 2]
        validation_summary = df_validation.groupby('validation_type')['success_rate'].agg(['mean', 'std'])
        validation_summary.plot(kind='bar', y='mean', yerr='std', ax=ax6, capsize=5)
        ax6.set_title('Validation: Failed vs Random')
        ax6.set_ylabel('Success Rate')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3)

        # Plot 7: Head position visualization for successful patterns
        ax7 = axes[2, 0]
        successful_patterns = df_patterns[df_patterns['success_rate'] > 0.8]
        for i, (_, row) in enumerate(successful_patterns.iterrows()):
            heads = row['head_indices']
            y_pos = [i] * len(heads)
            ax7.scatter(heads, y_pos, s=100, alpha=0.7, label=row['pattern_name'])
        ax7.set_xlabel('Head Index')
        ax7.set_ylabel('Pattern')
        ax7.set_title('Successful Pattern Head Positions')
        ax7.set_xlim(-1, 32)
        ax7.grid(True, alpha=0.3)

        # Plot 8: Head position visualization for failed patterns
        ax8 = axes[2, 1]
        failed_patterns = df_patterns[df_patterns['success_rate'] < 0.2]
        for i, (_, row) in enumerate(failed_patterns.iterrows()):
            heads = row['head_indices']
            y_pos = [i] * len(heads)
            ax8.scatter(heads, y_pos, s=100, alpha=0.7, label=row['pattern_name'], color='red')
        ax8.set_xlabel('Head Index')
        ax8.set_ylabel('Pattern')
        ax8.set_title('Failed Pattern Head Positions')
        ax8.set_xlim(-1, 32)
        ax8.grid(True, alpha=0.3)

        # Plot 9: Summary statistics
        ax9 = axes[2, 2]
        ax9.axis('off')

        # Create summary text
        summary_text = "SPATIAL ORGANIZATION INVESTIGATION\n" + "="*40 + "\n\n"

        total_patterns = len(df_patterns)
        successful_patterns = len(df_patterns[df_patterns['success_rate'] > 0.8])
        failed_patterns = len(df_patterns[df_patterns['success_rate'] < 0.2])

        summary_text += f"PATTERN TESTING:\n"
        summary_text += f"  Total patterns tested: {total_patterns}\n"
        summary_text += f"  Successful (>80%): {successful_patterns}\n"
        summary_text += f"  Failed (<20%): {failed_patterns}\n"
        summary_text += f"  Success rate: {successful_patterns/total_patterns:.1%}\n\n"

        # Most important spatial metrics
        if 'spatial_correlations' in analysis:
            summary_text += f"KEY SPATIAL METRICS:\n"
            sorted_correlations = sorted(analysis['spatial_correlations'].items(),
                                       key=lambda x: abs(x[1]['correlation']), reverse=True)
            for metric, data in sorted_correlations[:3]:
                summary_text += f"  {metric}: r={data['correlation']:+.3f}\n"

        summary_text += f"\nVALIDATION RESULTS:\n"
        failed_retest = df_validation[df_validation['validation_type'] == 'failed_retest']
        new_random = df_validation[df_validation['validation_type'] == 'new_random']

        if len(failed_retest) > 0:
            avg_failed = failed_retest['success_rate'].mean()
            summary_text += f"  Known failed: {avg_failed:.1%} avg success\n"
        if len(new_random) > 0:
            avg_random = new_random['success_rate'].mean()
            summary_text += f"  New random: {avg_random:.1%} avg success\n"

        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()
        plt.savefig('/home/paperspace/dev/MATS9/bandwidth/figures/spatial_organization_comprehensive.png',
                   dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Run comprehensive spatial organization investigation"""
    print("="*70)
    print("COMPREHENSIVE SPATIAL ORGANIZATION INVESTIGATION")
    print("="*70)
    print("Testing three hypotheses:")
    print("1. Spatial organization patterns determine success")
    print("2. Specific computational properties require spatial organization")
    print("3. Failed combinations consistently fail")
    print()

    # Initialize investigator
    investigator = SpatialOrganizationInvestigator()

    # Part 1: Test spatial hypothesis
    df_patterns = investigator.part_1_test_spatial_hypothesis()

    # Part 2: Analyze computational properties
    analysis = investigator.part_2_analyze_computational_properties(df_patterns)

    # Part 3: Cross-validate failed combinations
    df_validation = investigator.part_3_cross_validate_failed_combinations()

    # Create comprehensive visualization
    print("\nCreating comprehensive visualization...")
    investigator.create_comprehensive_visualization(df_patterns, df_validation, analysis)

    # Final summary
    print("\n" + "="*70)
    print("INVESTIGATION CONCLUSIONS")
    print("="*70)

    successful_patterns = df_patterns[df_patterns['success_rate'] > 0.8]
    failed_patterns = df_patterns[df_patterns['success_rate'] < 0.2]

    print(f"\nHYPOTHESIS 1 - SPATIAL ORGANIZATION:")
    print(f"  Successful patterns: {len(successful_patterns)}/{len(df_patterns)}")
    print(f"  Failed patterns: {len(failed_patterns)}/{len(df_patterns)}")

    if len(successful_patterns) > 0 and len(failed_patterns) > 0:
        # Compare key metrics
        successful_regularity = np.mean([row['spatial_metrics']['gap_regularity']
                                       for _, row in successful_patterns.iterrows()])
        failed_regularity = np.mean([row['spatial_metrics']['gap_regularity']
                                   for _, row in failed_patterns.iterrows()])

        print(f"  Successful gap regularity: {successful_regularity:.3f}")
        print(f"  Failed gap regularity: {failed_regularity:.3f}")
        print(f"  Hypothesis supported: {'✅' if successful_regularity > failed_regularity else '❌'}")

    print(f"\nHYPOTHESIS 2 - COMPUTATIONAL PROPERTIES:")
    if 'spatial_correlations' in analysis:
        top_correlations = sorted(analysis['spatial_correlations'].items(),
                                key=lambda x: abs(x[1]['correlation']), reverse=True)
        print(f"  Strongest predictor: {top_correlations[0][0]} (r={top_correlations[0][1]['correlation']:+.3f})")

        significant_predictors = [k for k, v in analysis['spatial_correlations'].items()
                                if v['p_value'] < 0.05]
        print(f"  Significant predictors: {len(significant_predictors)}")

    print(f"\nHYPOTHESIS 3 - CONSISTENCY:")
    failed_validation = df_validation[df_validation['validation_type'] == 'failed_retest']
    if len(failed_validation) > 0:
        avg_failed_success = failed_validation['success_rate'].mean()
        consistent_failures = (failed_validation['success_rate'] < 0.2).sum()
        print(f"  Failed combinations re-tested: {len(failed_validation)}")
        print(f"  Average success rate: {avg_failed_success:.1%}")
        print(f"  Consistent failures: {consistent_failures}/{len(failed_validation)}")
        print(f"  Hypothesis supported: {'✅' if consistent_failures/len(failed_validation) > 0.8 else '❌'}")

    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'/home/paperspace/dev/MATS9/bandwidth/spatial_organization_investigation_{timestamp}.json'

    comprehensive_results = {
        'pattern_testing': df_patterns.to_dict('records'),
        'validation_testing': df_validation.to_dict('records'),
        'computational_analysis': analysis,
        'summary': {
            'total_patterns_tested': len(df_patterns),
            'successful_patterns': len(successful_patterns),
            'failed_patterns': len(failed_patterns),
            'spatial_hypothesis_supported': len(successful_patterns) > len(failed_patterns),
            'consistency_validated': avg_failed_success < 0.2 if len(failed_validation) > 0 else False
        }
    }

    with open(results_file, 'w') as f:
        json.dump(comprehensive_results, f, indent=2, default=str)

    print(f"\n✅ Comprehensive investigation complete!")
    print(f"✅ Results saved to {results_file}")
    print(f"✅ Visualization saved to /home/paperspace/dev/MATS9/bandwidth/figures/spatial_organization_comprehensive.png")

    return df_patterns, df_validation, analysis


if __name__ == "__main__":
    main()