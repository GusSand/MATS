#!/usr/bin/env python3
"""
Test Permutation Invariance Argument
===================================

This experiment directly tests the critic's argument that head indices are implementation
artifacts. If the critic is correct, then:

1. Randomly permuting head indices should not affect model performance
2. The "even/odd" pattern should disappear after permutation
3. Success should depend on the ACTUAL heads selected, not their indices

This is the definitive test to silence or validate the critic's argument.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
from contextlib import contextmanager
import json
from datetime import datetime
import time
import matplotlib.pyplot as plt
import os
import copy

class PermutationInvarianceTest:
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

        # Store original weight matrices
        self.original_weights = self.extract_attention_weights()

    def extract_attention_weights(self) -> Dict[str, torch.Tensor]:
        """Extract the original attention weight matrices"""
        attn_module = self.model.model.layers[self.layer_idx].self_attn

        weights = {}
        for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            if hasattr(attn_module, name):
                weights[name] = getattr(attn_module, name).weight.data.clone()

        return weights

    def apply_head_permutation(self, permutation: List[int]):
        """Apply a permutation to attention heads by reordering weight matrices"""
        attn_module = self.model.model.layers[self.layer_idx].self_attn

        # Head dimension
        head_dim = attn_module.head_dim
        hidden_size = attn_module.hidden_size

        print(f"Applying permutation: {permutation[:8]}... to heads")

        # For Q, K, V projections - reshape and permute
        for proj_name in ['q_proj', 'k_proj', 'v_proj']:
            if hasattr(attn_module, proj_name):
                proj = getattr(attn_module, proj_name)

                if proj_name == 'q_proj':
                    # Q projection: [hidden_size, n_heads * head_dim]
                    weight = self.original_weights[proj_name]
                    weight_reshaped = weight.view(hidden_size, self.n_heads, head_dim)

                    # Apply permutation
                    weight_permuted = weight_reshaped[:, permutation, :]
                    weight_flat = weight_permuted.view(hidden_size, self.n_heads * head_dim)

                    proj.weight.data = weight_flat

                elif proj_name in ['k_proj', 'v_proj']:
                    # K, V projections in Llama are grouped - need special handling
                    weight = self.original_weights[proj_name]

                    # For Llama's grouped query attention
                    if weight.shape[0] < hidden_size:  # Grouped case
                        n_groups = weight.shape[0] // head_dim
                        weight_reshaped = weight.view(n_groups, head_dim, hidden_size)

                        # Create permutation for groups (each group of 4 heads)
                        group_permutation = [permutation[i] // 4 for i in range(0, len(permutation), 4)]
                        group_permutation = list(dict.fromkeys(group_permutation))  # Remove duplicates, preserve order

                        if len(group_permutation) == n_groups:
                            weight_permuted = weight_reshaped[group_permutation, :, :]
                            weight_flat = weight_permuted.view(-1, hidden_size)
                            proj.weight.data = weight_flat
                    else:
                        # Non-grouped case (same as Q)
                        weight_reshaped = weight.view(hidden_size, self.n_heads, head_dim)
                        weight_permuted = weight_reshaped[:, permutation, :]
                        weight_flat = weight_permuted.view(hidden_size, self.n_heads * head_dim)
                        proj.weight.data = weight_flat

        # For O projection: [n_heads * head_dim, hidden_size]
        if hasattr(attn_module, 'o_proj'):
            o_proj = attn_module.o_proj
            weight = self.original_weights['o_proj']
            weight_reshaped = weight.view(self.n_heads, head_dim, hidden_size)

            # Apply permutation
            weight_permuted = weight_reshaped[permutation, :, :]
            weight_flat = weight_permuted.view(self.n_heads * head_dim, hidden_size)

            o_proj.weight.data = weight_flat

    def restore_original_weights(self):
        """Restore original weight matrices"""
        attn_module = self.model.model.layers[self.layer_idx].self_attn

        for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            if hasattr(attn_module, name) and name in self.original_weights:
                getattr(attn_module, name).weight.data = self.original_weights[name].clone()

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

        correct_patterns = [
            "9.8 is bigger", "9.8 is larger", "9.8 is greater",
            "9.8 is more", "9.8 is the bigger", "9.8 is the larger"
        ]

        bug_patterns = [
            "9.11 is bigger", "9.11 is larger", "9.11 is greater",
            "9.11 is more", "9.11 is the bigger", "9.11 is the larger"
        ]

        has_correct = any(pattern in output_lower for pattern in correct_patterns)
        has_bug = any(pattern in output_lower for pattern in bug_patterns)

        if not has_correct and not has_bug:
            lines = output.strip().split('\n')
            if lines and '9.8' in lines[0] and '9.11' not in lines[0]:
                has_correct = True

        return has_correct and not has_bug

    def test_head_subset_with_permutation(self, head_indices: List[int], permutation: List[int],
                                        n_trials: int = 20, name: str = "") -> Dict:
        """Test a head subset after applying a permutation"""

        # Apply permutation
        self.apply_head_permutation(permutation)

        try:
            success_count = 0

            correct_prompt = "Which is bigger: 9.8 or 9.11?\nAnswer:"
            buggy_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"

            # Save activation with current (permuted) weights
            with self.save_activation_context(correct_prompt) as saved:
                correct_activation = saved[f"layer_{self.layer_idx}_attention"]

            sample_outputs = []

            for trial in range(n_trials):
                with self.patch_activation_context(correct_activation, head_indices):
                    output = self.generate(buggy_prompt, max_new_tokens=20)

                if self.check_bug_fixed(output):
                    success_count += 1

                if trial < 2:
                    sample_outputs.append(output.strip())

            success_rate = success_count / n_trials

            return {
                'success_rate': success_rate,
                'success_count': success_count,
                'n_trials': n_trials,
                'head_indices': head_indices,
                'permutation': permutation,
                'sample_outputs': sample_outputs
            }

        finally:
            # Always restore original weights
            self.restore_original_weights()

    def run_permutation_test(self) -> Dict:
        """Run the critical permutation test"""

        print("=" * 70)
        print("PERMUTATION INVARIANCE TEST - TESTING CRITIC'S ARGUMENT")
        print("=" * 70)
        print("Testing whether head indices are implementation artifacts")
        print()

        results = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'layer': self.layer_idx,
            'tests': {},
            'analysis': {}
        }

        # Test patterns
        test_patterns = {
            'original_even_8': list(range(0, 16, 2)),  # [0,2,4,6,8,10,12,14]
            'original_distributed': [0, 4, 8, 12, 16, 20, 24, 28],
        }

        # Generate random permutations
        np.random.seed(42)  # For reproducibility
        permutations = []

        # Identity permutation (control)
        permutations.append(list(range(32)))

        # Random permutations
        for i in range(3):
            perm = list(range(32))
            np.random.shuffle(perm)
            permutations.append(perm)

        # Specific "adversarial" permutation that swaps even/odd
        swap_perm = []
        for i in range(16):
            swap_perm.append(2*i + 1)  # Map even indices to odd positions
            swap_perm.append(2*i)      # Map odd indices to even positions
        permutations.append(swap_perm)

        print(f"Testing {len(test_patterns)} patterns with {len(permutations)} permutations...")
        print()

        # Test each pattern with each permutation
        for pattern_name, head_indices in test_patterns.items():
            print(f"TESTING PATTERN: {pattern_name}")
            print(f"Heads: {head_indices}")
            print("-" * 50)

            pattern_results = {}

            for perm_idx, permutation in enumerate(permutations):
                perm_name = f"permutation_{perm_idx}"
                if perm_idx == 0:
                    perm_name = "identity"
                elif perm_idx == len(permutations) - 1:
                    perm_name = "even_odd_swap"

                print(f"  {perm_name}: ", end="")

                # Map original head indices through the permutation
                # If head i was at position j, after permutation it's at position permutation[j]
                # So to select the same functional heads, we need the inverse mapping
                inverse_perm = [0] * 32
                for j, new_pos in enumerate(permutation):
                    inverse_perm[new_pos] = j

                # Map head indices through inverse permutation
                mapped_indices = [inverse_perm[i] for i in head_indices]

                result = self.test_head_subset_with_permutation(
                    mapped_indices, permutation, n_trials=20, name=perm_name
                )

                pattern_results[perm_name] = result
                print(f"{result['success_rate']:.1%} success")

            results['tests'][pattern_name] = pattern_results
            print()

        # Analyze results
        results['analysis'] = self.analyze_permutation_results(results['tests'])

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"permutation_invariance_test_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {output_file}")

        # Create visualization
        self.visualize_permutation_results(results)

        return results

    def analyze_permutation_results(self, test_results: Dict) -> Dict:
        """Analyze permutation test results"""

        analysis = {
            'critic_argument_validated': False,
            'permutation_effects': {},
            'pattern_consistency': {},
            'summary': ""
        }

        for pattern_name, pattern_results in test_results.items():
            success_rates = []
            permutation_names = []

            for perm_name, result in pattern_results.items():
                success_rates.append(result['success_rate'])
                permutation_names.append(perm_name)

            # Check if success rates are consistent across permutations
            rate_variance = np.var(success_rates)
            rate_range = max(success_rates) - min(success_rates)

            analysis['pattern_consistency'][pattern_name] = {
                'success_rates': success_rates,
                'permutation_names': permutation_names,
                'variance': rate_variance,
                'range': rate_range,
                'consistent': rate_range < 0.1  # Less than 10% difference
            }

        # Overall assessment
        all_consistent = all(p['consistent'] for p in analysis['pattern_consistency'].values())

        if all_consistent:
            analysis['critic_argument_validated'] = True
            analysis['summary'] = "CRITIC VALIDATED: Head indices are implementation artifacts. Performance is invariant to permutation."
        else:
            analysis['critic_argument_validated'] = False
            analysis['summary'] = "CRITIC REFUTED: Head indices matter functionally. Performance changes with permutation."

        return analysis

    def visualize_permutation_results(self, results: Dict):
        """Visualize permutation test results"""

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        test_results = results['tests']

        # Plot 1: Success rates by permutation for each pattern
        ax1 = axes[0]

        patterns = list(test_results.keys())
        permutations = list(test_results[patterns[0]].keys())

        x = np.arange(len(permutations))
        width = 0.35

        for i, pattern_name in enumerate(patterns):
            success_rates = [test_results[pattern_name][perm]['success_rate']
                           for perm in permutations]

            ax1.bar(x + i * width, success_rates, width, label=pattern_name, alpha=0.8)

        ax1.set_xlabel('Permutation')
        ax1.set_ylabel('Success Rate')
        ax1.set_title('Success Rate by Permutation', fontweight='bold')
        ax1.set_xticks(x + width / 2)
        ax1.set_xticklabels(permutations, rotation=45)
        ax1.legend()
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Consistency analysis
        ax2 = axes[1]

        pattern_names = []
        ranges = []
        colors = []

        for pattern_name, consistency in results['analysis']['pattern_consistency'].items():
            pattern_names.append(pattern_name)
            ranges.append(consistency['range'])
            colors.append('green' if consistency['consistent'] else 'red')

        bars = ax2.bar(pattern_names, ranges, color=colors, alpha=0.7)
        ax2.set_ylabel('Success Rate Range')
        ax2.set_title('Permutation Consistency\n(Green = Consistent, Red = Inconsistent)', fontweight='bold')
        ax2.set_xticklabels(pattern_names, rotation=45)
        ax2.grid(True, alpha=0.3)

        # Add consistency threshold line
        ax2.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Consistency Threshold (10%)')
        ax2.legend()

        plt.tight_layout()

        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"figures/permutation_test_{timestamp}.png"
        os.makedirs("figures", exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_file}")

        plt.show()

def main():
    """Run the permutation invariance test"""

    start_time = time.time()

    tester = PermutationInvarianceTest(device="cuda")
    results = tester.run_permutation_test()

    print("=" * 70)
    print("FINAL VERDICT:")
    print("=" * 70)

    analysis = results['analysis']
    print(f"\n{analysis['summary']}")

    print(f"\nPattern consistency:")
    for pattern_name, consistency in analysis['pattern_consistency'].items():
        status = "✅ CONSISTENT" if consistency['consistent'] else "❌ INCONSISTENT"
        print(f"  {pattern_name}: {status} (range: {consistency['range']:.2f})")

    if analysis['critic_argument_validated']:
        print(f"\n🎯 The critic is CORRECT: Head indices are implementation artifacts")
        print(f"   Performance remains consistent across permutations")
    else:
        print(f"\n🚫 The critic is WRONG: Head indices have functional meaning")
        print(f"   Performance varies significantly with permutations")

    total_time = time.time() - start_time
    print(f"\nTest completed in {total_time:.1f} seconds")

if __name__ == "__main__":
    main()