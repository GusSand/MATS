#!/usr/bin/env python3
"""
CORRECTED Head Swapping Experiment: Test Index vs Function Dependence

Fixed to use the WORKING methodology from training_dynamics_analysis.
Uses activation patching approach instead of weight manipulation.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import copy
from datetime import datetime
from pathlib import Path
import json
import random
import warnings
warnings.filterwarnings('ignore')

# Setup paths
BASE_DIR = Path("/home/paperspace/dev/MATS9/pythia_clustering_dynamics")
RESULTS_DIR = BASE_DIR / "results"

class CorrectedHeadSwappingExperiment:
    """Head swapping using activation patching (proven working method)"""

    def __init__(self, model_name="EleutherAI/pythia-160m"):
        self.model_name = model_name
        self.target_layer = 6

        print(f"🔧 Loading {model_name} using WORKING methodology...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model using EXACT same settings as working code
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # CRITICAL: Same as working code
            device_map="cuda"           # CRITICAL: Same as working code
        )
        self.model.eval()

        # Prompts
        self.clean_prompt = "Which is bigger: 9.8 or 9.11?"
        self.buggy_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"

        print("  ✅ Model loaded successfully")

    def check_bug_fixed(self, output_text):
        """Exact same bug detection logic as working code"""
        output_lower = output_text.lower()
        correct_patterns = ["9.8 is bigger", "9.8 is larger", "9.8"]
        bug_patterns = ["9.11 is bigger", "9.11 is larger", "9.11"]
        has_correct = any(pattern in output_lower for pattern in correct_patterns)
        has_bug = any(pattern in output_lower for pattern in bug_patterns)
        return has_correct and not has_bug

    def test_baseline(self):
        """Test baseline behavior (should show bug)"""
        print("\n📊 Testing Baseline...")

        inputs = self.tokenizer(self.buggy_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        bug_fixed = self.check_bug_fixed(response)

        print(f"  Response: '{response.strip()}'")
        print(f"  Bug fixed: {'❌ Unexpected!' if bug_fixed else '✅ Has bug (expected)'}")

        return {
            'response': response.strip(),
            'bug_fixed': bug_fixed,
            'has_bug': not bug_fixed
        }

    def get_clean_activation(self):
        """Get clean activation using exact working methodology"""
        print("  💾 Getting clean activation...")

        attention_module = self.model.gpt_neox.layers[self.target_layer].attention
        saved_activation = None

        def save_hook(module, input, output):
            nonlocal saved_activation
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            saved_activation = hidden_states.detach().cpu()

        clean_inputs = self.tokenizer(self.clean_prompt, return_tensors="pt").to("cuda")
        hook = attention_module.register_forward_hook(save_hook)

        with torch.no_grad():
            self.model(**clean_inputs)

        hook.remove()
        print(f"    Saved activation shape: {saved_activation.shape}")
        return saved_activation

    def test_head_group_patching(self, heads_to_patch, group_name, saved_activation):
        """Test patching specific heads using working methodology"""
        print(f"    Testing {group_name}: {heads_to_patch}")

        attention_module = self.model.gpt_neox.layers[self.target_layer].attention

        def patch_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            batch_size, seq_len, hidden_size = hidden_states.shape
            head_dim = hidden_size // 12

            # Reshape to separate heads
            hidden_reshaped = hidden_states.view(batch_size, seq_len, 12, head_dim)
            saved_reshaped = saved_activation.to(hidden_states.device).view(batch_size, -1, 12, head_dim)

            new_hidden = hidden_reshaped.clone()
            min_seq_len = min(seq_len, saved_reshaped.shape[1])

            # Patch specified heads
            for head_idx in heads_to_patch:
                if head_idx < 12:
                    new_hidden[:, :min_seq_len, head_idx, :] = saved_reshaped[:, :min_seq_len, head_idx, :]

            # Reshape back
            new_hidden = new_hidden.view(batch_size, seq_len, hidden_size)

            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden

        # Apply patch
        hook = attention_module.register_forward_hook(patch_hook)

        inputs = self.tokenizer(self.buggy_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        hook.remove()

        bug_fixed = self.check_bug_fixed(response)
        print(f"      Response: '{response.strip()}'")
        print(f"      Bug fixed: {'✅' if bug_fixed else '❌'}")

        return {
            'heads': heads_to_patch,
            'group_name': group_name,
            'response': response.strip(),
            'bug_fixed': bug_fixed
        }

    def test_head_permutation_effect(self, permutation, description, saved_activation):
        """Test effect of virtual head permutation"""
        print(f"\n🧪 Testing {description}")
        print(f"  Permutation: {permutation}")

        # In activation patching, we simulate head permutation by:
        # 1. Testing original even heads [0,2,4,6,8,10]
        # 2. Testing permuted positions that would be "even" after permutation

        # Original even heads
        original_even = [0, 2, 4, 6, 8, 10]

        # After permutation, which heads are in even positions?
        # inverse_perm[i] = j means position i gets head j
        # So heads in even positions are: [inverse_perm[0], inverse_perm[2], ...]
        inverse_perm = [0] * 12
        for new_pos, old_head in enumerate(permutation):
            inverse_perm[new_pos] = old_head

        # Heads that end up in even positions after permutation
        permuted_even_positions = [inverse_perm[i] for i in [0, 2, 4, 6, 8, 10]]

        results = {}

        # Test original even heads (baseline)
        original_result = self.test_head_group_patching(
            original_even,
            "Original Even Heads",
            saved_activation
        )
        results['original_even'] = original_result

        # Test heads that would be in even positions after permutation
        permuted_result = self.test_head_group_patching(
            permuted_even_positions,
            f"Permuted Even Positions ({permuted_even_positions})",
            saved_activation
        )
        results['permuted_even'] = permuted_result

        return {
            'permutation': permutation,
            'description': description,
            'original_even_heads': original_even,
            'permuted_even_positions': permuted_even_positions,
            'results': results
        }

    def run_comprehensive_experiment(self):
        """Run comprehensive head swapping experiment"""
        print("🚀 CORRECTED HEAD SWAPPING EXPERIMENT")
        print("="*60)
        print("Using PROVEN working methodology from training_dynamics_analysis")
        print("="*60)

        all_results = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'target_layer': self.target_layer,
            'methodology': 'activation_patching_corrected'
        }

        # Test baseline first
        baseline = self.test_baseline()
        all_results['baseline'] = baseline

        if not baseline['has_bug']:
            print("\n⚠️  WARNING: Baseline doesn't show bug! Continuing anyway...")

        # Get clean activation
        saved_activation = self.get_clean_activation()

        # Test 1: Baseline even/odd
        print(f"\n⚖️  BASELINE Even/Odd Test")
        even_heads = [0, 2, 4, 6, 8, 10]
        odd_heads = [1, 3, 5, 7, 9, 11]

        baseline_even = self.test_head_group_patching(even_heads, "Even Heads", saved_activation)
        baseline_odd = self.test_head_group_patching(odd_heads, "Odd Heads", saved_activation)

        all_results['baseline_even_odd'] = {
            'even': baseline_even,
            'odd': baseline_odd
        }

        # Test 2: Even/Odd swap (0↔1, 2↔3, 4↔5, 6↔7, 8↔9, 10↔11)
        even_odd_swap = []
        for i in range(0, 12, 2):
            even_odd_swap.extend([i+1, i])  # Swap each pair

        swap_results = self.test_head_permutation_effect(
            even_odd_swap,
            "Even/Odd Swap (0↔1, 2↔3, 4↔5, 6↔7, 8↔9, 10↔11)",
            saved_activation
        )
        all_results['even_odd_swap'] = swap_results

        # Test 3: Even shuffle, Odd shuffle
        even_indices = [0, 2, 4, 6, 8, 10]
        odd_indices = [1, 3, 5, 7, 9, 11]

        random.seed(42)
        shuffled_even = even_indices.copy()
        random.shuffle(shuffled_even)

        shuffled_odd = odd_indices.copy()
        random.shuffle(shuffled_odd)

        even_odd_shuffle = [0] * 12
        for i, new_even in enumerate(shuffled_even):
            even_odd_shuffle[even_indices[i]] = new_even
        for i, new_odd in enumerate(shuffled_odd):
            even_odd_shuffle[odd_indices[i]] = new_odd

        shuffle_results = self.test_head_permutation_effect(
            even_odd_shuffle,
            f"Even Shuffle + Odd Shuffle",
            saved_activation
        )
        all_results['even_odd_shuffle'] = shuffle_results

        # Test 4: Complete random
        random.seed(123)
        random_perm = list(range(12))
        random.shuffle(random_perm)

        random_results = self.test_head_permutation_effect(
            random_perm,
            "Random Permutation",
            saved_activation
        )
        all_results['random_permutation'] = random_results

        return all_results

    def save_results(self, results):
        """Save results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = RESULTS_DIR / f"head_swapping_corrected_{timestamp}.json"

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Saved results to {filepath}")
        return filepath

    def print_summary(self, results):
        """Print summary"""
        print("\n" + "="*60)
        print("🎯 CORRECTED HEAD SWAPPING SUMMARY")
        print("="*60)

        # Baseline check
        baseline = results['baseline']
        print(f"\n📊 Baseline Check:")
        print(f"  Has bug: {'✅' if baseline['has_bug'] else '❌'}")

        # Even/odd baseline
        even_odd = results['baseline_even_odd']
        print(f"\n⚖️  Baseline Even/Odd:")
        print(f"  Even heads fix bug: {'✅' if even_odd['even']['bug_fixed'] else '❌'}")
        print(f"  Odd heads fix bug: {'✅' if even_odd['odd']['bug_fixed'] else '❌'}")

        # Permutation tests
        print(f"\n🔄 Permutation Results:")

        for test_name in ['even_odd_swap', 'even_odd_shuffle', 'random_permutation']:
            if test_name in results:
                test_data = results[test_name]
                original_works = test_data['results']['original_even']['bug_fixed']
                permuted_works = test_data['results']['permuted_even']['bug_fixed']

                print(f"  {test_name}:")
                print(f"    Original even positions: {'✅' if original_works else '❌'}")
                print(f"    Permuted even positions: {'✅' if permuted_works else '❌'}")

                if original_works and permuted_works:
                    print(f"    Result: FUNCTION-DEPENDENT ✅")
                elif original_works and not permuted_works:
                    print(f"    Result: INDEX-DEPENDENT ❌")
                else:
                    print(f"    Result: UNCLEAR")

        print(f"\n✅ Corrected experiment complete!")


def main():
    """Run corrected experiment"""
    experiment = CorrectedHeadSwappingExperiment()
    results = experiment.run_comprehensive_experiment()
    experiment.save_results(results)
    experiment.print_summary(results)


if __name__ == "__main__":
    main()