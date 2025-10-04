#!/usr/bin/env python3
"""
Activation Patching Verification: Test clustering predictions with actual behavioral method

This script uses the proper activation patching methodology to verify:
1. Even/odd head specialization on 9.8 vs 9.11 task
2. Our weight clustering predictions (Head 6 vs others)
3. Consistency between weight structure and behavioral function
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Setup paths
BASE_DIR = Path("/home/paperspace/dev/MATS9/pythia_clustering_dynamics")
RESULTS_DIR = BASE_DIR / "results"

class ActivationPatchingVerifier:
    """Use activation patching to verify clustering predictions"""

    def __init__(self, model_name="EleutherAI/pythia-160m"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_layer = 6

        # Load model and tokenizer
        print(f"🔧 Loading {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="cuda" if torch.cuda.is_available() else None
        )
        self.model.eval()

    def run_clean_prompt(self, prompt):
        """Run clean prompt and save activations"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Store clean activations
        clean_activations = {}

        def hook_fn(module, input, output):
            # Attention module returns a tuple (output, attention_weights)
            if isinstance(output, tuple):
                attention_output = output[0]
            else:
                attention_output = output
            clean_activations['attention_output'] = attention_output.clone()
            return output

        # Register hook
        attention_module = self.model.gpt_neox.layers[self.target_layer].attention
        handle = attention_module.register_forward_hook(hook_fn)

        # Run forward pass
        with torch.no_grad():
            clean_output = self.model(**inputs)

        # Remove hook
        handle.remove()

        return clean_output, clean_activations

    def patch_attention_heads(self, buggy_prompt, clean_activations, heads_to_patch):
        """Patch specific attention heads with clean activations"""
        inputs = self.tokenizer(buggy_prompt, return_tensors="pt").to(self.device)

        def patch_hook_fn(module, input, output):
            # Handle tuple output
            if isinstance(output, tuple):
                attention_output = output[0]
                other_outputs = output[1:]
            else:
                attention_output = output
                other_outputs = ()

            # output shape: [batch_size, seq_len, hidden_size]
            batch_size, seq_len, hidden_size = attention_output.shape
            num_heads = self.model.config.num_attention_heads
            head_dim = hidden_size // num_heads

            # Get clean activations shape
            clean_output = clean_activations['attention_output']
            clean_batch, clean_seq_len, clean_hidden = clean_output.shape

            # Only patch if sequence lengths match or we can handle the difference
            if seq_len == clean_seq_len:
                # Reshape to separate heads: [batch, seq, num_heads, head_dim]
                output_heads = attention_output.view(batch_size, seq_len, num_heads, head_dim)
                clean_heads = clean_output.view(clean_batch, clean_seq_len, num_heads, head_dim)

                # Patch specified heads
                for head_idx in heads_to_patch:
                    if head_idx < num_heads:
                        output_heads[:, :, head_idx, :] = clean_heads[:, :, head_idx, :]

                # Reshape back
                patched_output = output_heads.view(batch_size, seq_len, hidden_size)
            else:
                # If sequence lengths don't match, only patch the last token (most important for next token prediction)
                output_heads = attention_output.view(batch_size, seq_len, num_heads, head_dim)
                clean_heads = clean_output.view(clean_batch, clean_seq_len, num_heads, head_dim)

                # Patch only the last token
                for head_idx in heads_to_patch:
                    if head_idx < num_heads:
                        output_heads[:, -1, head_idx, :] = clean_heads[:, -1, head_idx, :]

                # Reshape back
                patched_output = output_heads.view(batch_size, seq_len, hidden_size)

            if other_outputs:
                return (patched_output,) + other_outputs
            else:
                return patched_output

        # Register patch hook
        attention_module = self.model.gpt_neox.layers[self.target_layer].attention
        handle = attention_module.register_forward_hook(patch_hook_fn)

        # Run patched forward pass
        with torch.no_grad():
            patched_output = self.model(**inputs)

        # Remove hook
        handle.remove()

        return patched_output

    def check_bug_fixed(self, output_text):
        """Check if the decimal comparison bug is fixed"""
        output_lower = output_text.lower()

        # Correct patterns (9.8 is bigger)
        correct_patterns = ["9.8 is bigger", "9.8 is larger", "9.8"]

        # Bug patterns (9.11 is bigger - incorrect)
        bug_patterns = ["9.11 is bigger", "9.11 is larger", "9.11"]

        has_correct = any(pattern in output_lower for pattern in correct_patterns)
        has_bug = any(pattern in output_lower for pattern in bug_patterns)

        if has_correct and not has_bug:
            return "FIXED"
        elif has_bug and not has_correct:
            return "BUG"
        elif has_correct and has_bug:
            return "MIXED"
        else:
            return "UNCLEAR"

    def test_head_group_specialization(self, head_group, group_name):
        """Test if a group of heads fixes the bug when patched"""
        print(f"\n🧪 Testing {group_name}: {head_group}")

        # Define prompts
        clean_prompt = "Which is bigger: 9.8 or 9.11?"
        buggy_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"

        # Get clean activations
        clean_output, clean_activations = self.run_clean_prompt(clean_prompt)

        # Test unpatched (baseline bug)
        inputs = self.tokenizer(buggy_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            baseline_output = self.model(**inputs)

        baseline_text = self.tokenizer.decode(baseline_output.logits[0].argmax(dim=-1), skip_special_tokens=True)
        baseline_status = self.check_bug_fixed(baseline_text)

        # Test with head group patched
        patched_output = self.patch_attention_heads(buggy_prompt, clean_activations, head_group)
        patched_text = self.tokenizer.decode(patched_output.logits[0].argmax(dim=-1), skip_special_tokens=True)
        patched_status = self.check_bug_fixed(patched_text)

        # Get top predictions for comparison
        baseline_probs = F.softmax(baseline_output.logits[0, -1], dim=-1)
        patched_probs = F.softmax(patched_output.logits[0, -1], dim=-1)

        baseline_top5 = torch.topk(baseline_probs, 5)
        patched_top5 = torch.topk(patched_probs, 5)

        baseline_tokens = [self.tokenizer.decode(t) for t in baseline_top5.indices]
        patched_tokens = [self.tokenizer.decode(t) for t in patched_top5.indices]

        result = {
            'head_group': head_group,
            'group_name': group_name,
            'baseline_status': baseline_status,
            'patched_status': patched_status,
            'improvement': baseline_status == "BUG" and patched_status == "FIXED",
            'baseline_top_tokens': baseline_tokens,
            'patched_top_tokens': patched_tokens,
            'baseline_top_probs': baseline_top5.values.cpu().tolist(),
            'patched_top_probs': patched_top5.values.cpu().tolist()
        }

        print(f"  Baseline: {baseline_status}")
        print(f"  Patched: {patched_status}")
        print(f"  Improvement: {'✅' if result['improvement'] else '❌'}")
        print(f"  Baseline top token: {baseline_tokens[0]} ({baseline_top5.values[0]:.3f})")
        print(f"  Patched top token: {patched_tokens[0]} ({patched_top5.values[0]:.3f})")

        return result

    def test_individual_heads(self):
        """Test each head individually"""
        print(f"\n🔬 Testing Individual Head Contributions...")

        individual_results = {}

        for head_idx in range(12):
            result = self.test_head_group_specialization([head_idx], f"Head {head_idx}")
            individual_results[head_idx] = result

        return individual_results

    def test_even_odd_specialization(self):
        """Test the original even/odd hypothesis"""
        print(f"\n⚖️  Testing Even/Odd Specialization...")

        even_heads = [0, 2, 4, 6, 8, 10]
        odd_heads = [1, 3, 5, 7, 9, 11]

        even_result = self.test_head_group_specialization(even_heads, "Even Heads")
        odd_result = self.test_head_group_specialization(odd_heads, "Odd Heads")

        return {
            'even_heads': even_result,
            'odd_heads': odd_result
        }

    def test_weight_clustering_predictions(self):
        """Test our weight clustering predictions"""
        print(f"\n🎯 Testing Weight Clustering Predictions...")

        # Based on our weight clustering results:
        # Query weights: Head 6 vs others
        # Key weights: Head 0 vs others
        # Value weights: Head 3 vs others

        results = {}

        # Test Query clustering prediction
        query_cluster_0 = [6]  # Singleton cluster
        query_cluster_1 = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11]  # Majority cluster

        results['query_singleton'] = self.test_head_group_specialization(
            query_cluster_0, "Query Cluster 0 (Head 6)"
        )
        results['query_majority'] = self.test_head_group_specialization(
            query_cluster_1, "Query Cluster 1 (Others)"
        )

        # Test Key clustering prediction
        key_cluster_0 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        key_cluster_1 = [0]

        results['key_majority'] = self.test_head_group_specialization(
            key_cluster_0, "Key Cluster 0 (Most heads)"
        )
        results['key_singleton'] = self.test_head_group_specialization(
            key_cluster_1, "Key Cluster 1 (Head 0)"
        )

        # Test Value clustering prediction
        value_cluster_0 = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]
        value_cluster_1 = [3]

        results['value_majority'] = self.test_head_group_specialization(
            value_cluster_0, "Value Cluster 0 (Most heads)"
        )
        results['value_singleton'] = self.test_head_group_specialization(
            value_cluster_1, "Value Cluster 1 (Head 3)"
        )

        return results

    def test_baseline_comparison(self):
        """Test baseline behavior without patching"""
        print(f"\n📊 Testing Baseline Behavior...")

        clean_prompt = "Which is bigger: 9.8 or 9.11?"
        buggy_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"

        # Test clean prompt
        clean_inputs = self.tokenizer(clean_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            clean_output = self.model(**clean_inputs)

        clean_probs = F.softmax(clean_output.logits[0, -1], dim=-1)
        clean_top5 = torch.topk(clean_probs, 5)
        clean_tokens = [self.tokenizer.decode(t) for t in clean_top5.indices]
        clean_status = self.check_bug_fixed(self.tokenizer.decode(clean_output.logits[0].argmax(dim=-1), skip_special_tokens=True))

        # Test buggy prompt
        buggy_inputs = self.tokenizer(buggy_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            buggy_output = self.model(**buggy_inputs)

        buggy_probs = F.softmax(buggy_output.logits[0, -1], dim=-1)
        buggy_top5 = torch.topk(buggy_probs, 5)
        buggy_tokens = [self.tokenizer.decode(t) for t in buggy_top5.indices]
        buggy_status = self.check_bug_fixed(self.tokenizer.decode(buggy_output.logits[0].argmax(dim=-1), skip_special_tokens=True))

        print(f"  Clean prompt status: {clean_status}")
        print(f"  Clean top tokens: {clean_tokens}")
        print(f"  Buggy prompt status: {buggy_status}")
        print(f"  Buggy top tokens: {buggy_tokens}")

        return {
            'clean_prompt': {
                'status': clean_status,
                'top_tokens': clean_tokens,
                'top_probs': clean_top5.values.cpu().tolist()
            },
            'buggy_prompt': {
                'status': buggy_status,
                'top_tokens': buggy_tokens,
                'top_probs': buggy_top5.values.cpu().tolist()
            }
        }

    def run_complete_verification(self):
        """Run complete activation patching verification"""
        print("🔍 ACTIVATION PATCHING VERIFICATION")
        print("="*60)

        results = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'target_layer': self.target_layer
        }

        # Test baseline behavior
        results['baseline'] = self.test_baseline_comparison()

        # Test individual heads
        results['individual_heads'] = self.test_individual_heads()

        # Test even/odd specialization
        results['even_odd'] = self.test_even_odd_specialization()

        # Test weight clustering predictions
        results['weight_clustering_predictions'] = self.test_weight_clustering_predictions()

        return results

    def save_results(self, results):
        """Save verification results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            else:
                return obj

        filepath = RESULTS_DIR / f"activation_patching_verification_{timestamp}.json"
        with open(filepath, 'w') as f:
            json.dump(convert_numpy(results), f, indent=2)

        print(f"\n💾 Saved verification results to {filepath}")
        return filepath


def main():
    """Run activation patching verification"""
    verifier = ActivationPatchingVerifier()
    results = verifier.run_complete_verification()
    verifier.save_results(results)

    print("\n" + "="*60)
    print("🎯 ACTIVATION PATCHING VERIFICATION SUMMARY")
    print("="*60)

    # Baseline status
    baseline = results['baseline']
    print(f"\n📊 Baseline Behavior:")
    print(f"  Clean prompt: {baseline['clean_prompt']['status']}")
    print(f"  Buggy prompt: {baseline['buggy_prompt']['status']}")
    print(f"  Bug confirmed: {'✅' if baseline['buggy_prompt']['status'] == 'BUG' else '❌'}")

    # Individual head analysis
    individual = results['individual_heads']
    fixing_heads = [h for h, data in individual.items() if data['improvement']]
    print(f"\n🔬 Individual Head Analysis:")
    print(f"  Heads that fix bug: {fixing_heads}")
    print(f"  Total fixing heads: {len(fixing_heads)}/12")

    # Even/odd analysis
    even_odd = results['even_odd']
    even_fixes = even_odd['even_heads']['improvement']
    odd_fixes = even_odd['odd_heads']['improvement']
    print(f"\n⚖️  Even/Odd Specialization:")
    print(f"  Even heads fix bug: {'✅' if even_fixes else '❌'}")
    print(f"  Odd heads fix bug: {'✅' if odd_fixes else '❌'}")
    print(f"  Even/odd specialization confirmed: {'✅' if even_fixes != odd_fixes else '❌'}")

    # Weight clustering predictions
    clustering = results['weight_clustering_predictions']
    print(f"\n🎯 Weight Clustering Predictions:")
    print(f"  Query singleton (Head 6): {'✅' if clustering['query_singleton']['improvement'] else '❌'}")
    print(f"  Key singleton (Head 0): {'✅' if clustering['key_singleton']['improvement'] else '❌'}")
    print(f"  Value singleton (Head 3): {'✅' if clustering['value_singleton']['improvement'] else '❌'}")

    # Overall conclusions
    print(f"\n💡 Key Findings:")
    if len(fixing_heads) > 0:
        print(f"  • {len(fixing_heads)} heads can fix the bug individually")
        print(f"  • Functional heads: {fixing_heads}")

    if even_fixes != odd_fixes:
        specialization_type = "Even" if even_fixes else "Odd"
        print(f"  • {specialization_type} head specialization confirmed!")
    else:
        print(f"  • No even/odd specialization found")

    singleton_predictions = [
        clustering['query_singleton']['improvement'],
        clustering['key_singleton']['improvement'],
        clustering['value_singleton']['improvement']
    ]
    correct_predictions = sum(singleton_predictions)
    print(f"  • Weight clustering predictions: {correct_predictions}/3 correct")

    print("\n✅ Verification complete!")


if __name__ == "__main__":
    main()