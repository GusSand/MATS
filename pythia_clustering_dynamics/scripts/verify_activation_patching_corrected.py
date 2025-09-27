#!/usr/bin/env python3
"""
Corrected Activation Patching Verification using proper methodology

This script uses the CORRECT methodology from training_dynamics_analysis:
1. Use model.generate() with max_new_tokens=20
2. Check generated text for specific patterns
3. Use step143000 checkpoint where specialization was confirmed
4. Match exact prompt formats and evaluation criteria
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Setup paths
BASE_DIR = Path("/home/paperspace/dev/MATS9/pythia_clustering_dynamics")
RESULTS_DIR = BASE_DIR / "results"

class CorrectedActivationPatchingVerifier:
    """Use the CORRECT activation patching methodology"""

    def __init__(self, model_name="EleutherAI/pythia-160m", use_final_checkpoint=True):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_layer = 6

        print(f"🔧 Loading {model_name} (final checkpoint) on {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model - use final checkpoint to match training dynamics findings
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda" if torch.cuda.is_available() else None
        )
        self.model.eval()

        # Define prompts exactly as in original experiments
        self.clean_prompt = "Which is bigger: 9.8 or 9.11?"
        self.buggy_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"

    def check_bug_fixed(self, output_text):
        """Check if the decimal comparison bug is fixed (exact logic from original)"""
        output_lower = output_text.lower()
        correct_patterns = ["9.8 is bigger", "9.8 is larger", "9.8"]
        bug_patterns = ["9.11 is bigger", "9.11 is larger", "9.11"]

        has_correct = any(pattern in output_lower for pattern in correct_patterns)
        has_bug = any(pattern in output_lower for pattern in bug_patterns)

        return has_correct and not has_bug

    def test_baseline_behavior(self):
        """Test baseline behavior without any patching"""
        print("\n📊 Testing Baseline Behavior...")

        # Test clean prompt
        print("  Testing clean prompt...")
        clean_inputs = self.tokenizer(self.clean_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            clean_outputs = self.model.generate(
                **clean_inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        clean_response = self.tokenizer.decode(
            clean_outputs[0][clean_inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        clean_fixed = self.check_bug_fixed(clean_response)

        print(f"    Response: '{clean_response.strip()}'")
        print(f"    Bug fixed: {clean_fixed}")

        # Test buggy prompt
        print("  Testing buggy prompt...")
        buggy_inputs = self.tokenizer(self.buggy_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            buggy_outputs = self.model.generate(
                **buggy_inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        buggy_response = self.tokenizer.decode(
            buggy_outputs[0][buggy_inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        buggy_fixed = self.check_bug_fixed(buggy_response)

        print(f"    Response: '{buggy_response.strip()}'")
        print(f"    Bug fixed: {buggy_fixed}")

        return {
            'clean_prompt': {
                'text': clean_response.strip(),
                'bug_fixed': clean_fixed
            },
            'buggy_prompt': {
                'text': buggy_response.strip(),
                'bug_fixed': buggy_fixed
            },
            'has_bug': not buggy_fixed
        }

    def get_clean_activation(self):
        """Get clean activation from clean prompt (exact methodology from original)"""
        print("\n💾 Saving clean activation...")

        attention_module = self.model.gpt_neox.layers[self.target_layer].attention
        saved_activation = None

        def save_hook(module, input, output):
            nonlocal saved_activation
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            saved_activation = hidden_states.detach().cpu()

        clean_inputs = self.tokenizer(self.clean_prompt, return_tensors="pt").to(self.device)
        hook = attention_module.register_forward_hook(save_hook)

        with torch.no_grad():
            self.model(**clean_inputs)

        hook.remove()

        print(f"  Saved activation shape: {saved_activation.shape}")
        return saved_activation

    def test_head_group_patching(self, head_group, group_name, saved_activation):
        """Test patching specific head group (exact methodology from original)"""
        print(f"\n🧪 Testing {group_name}: {head_group}")

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
            for head_idx in head_group:
                if head_idx < 12:
                    new_hidden[:, :min_seq_len, head_idx, :] = saved_reshaped[:, :min_seq_len, head_idx, :]

            # Reshape back
            new_hidden = new_hidden.view(batch_size, seq_len, hidden_size)

            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden

        # Apply patch and test
        hook = attention_module.register_forward_hook(patch_hook)

        inputs = self.tokenizer(self.buggy_prompt, return_tensors="pt").to(self.device)
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

        print(f"  Response: '{response.strip()}'")
        print(f"  Bug fixed: {'✅' if bug_fixed else '❌'}")

        return {
            'heads': head_group,
            'group_name': group_name,
            'response': response.strip(),
            'bug_fixed': bug_fixed
        }

    def test_individual_heads(self, saved_activation):
        """Test each head individually"""
        print(f"\n🔬 Testing Individual Heads...")

        results = {}
        fixing_heads = []

        for head_idx in range(12):
            result = self.test_head_group_patching([head_idx], f"Head {head_idx}", saved_activation)
            results[head_idx] = result

            if result['bug_fixed']:
                fixing_heads.append(head_idx)

        print(f"\n  🎯 Heads that fix the bug: {fixing_heads}")
        print(f"  📊 Total fixing heads: {len(fixing_heads)}/12")

        return results, fixing_heads

    def test_even_odd_specialization(self, saved_activation):
        """Test even/odd head specialization (the main hypothesis)"""
        print(f"\n⚖️  Testing Even/Odd Specialization...")

        even_heads = [0, 2, 4, 6, 8, 10]
        odd_heads = [1, 3, 5, 7, 9, 11]

        even_result = self.test_head_group_patching(even_heads, "Even Heads", saved_activation)
        odd_result = self.test_head_group_patching(odd_heads, "Odd Heads", saved_activation)

        # Check for specialization
        even_fixes = even_result['bug_fixed']
        odd_fixes = odd_result['bug_fixed']

        if even_fixes and not odd_fixes:
            specialization = "EVEN specialization confirmed! ✅"
        elif odd_fixes and not even_fixes:
            specialization = "ODD specialization confirmed! ✅"
        elif even_fixes and odd_fixes:
            specialization = "Both groups fix bug - no specialization ❌"
        else:
            specialization = "Neither group fixes bug ❌"

        print(f"\n  🎯 Result: {specialization}")

        return {
            'even_heads': even_result,
            'odd_heads': odd_result,
            'specialization': specialization,
            'even_specializes': even_fixes and not odd_fixes,
            'odd_specializes': odd_fixes and not even_fixes
        }

    def test_weight_clustering_predictions(self, saved_activation):
        """Test our weight clustering predictions"""
        print(f"\n🎯 Testing Weight Clustering Predictions...")

        # Based on our clustering analysis:
        # Query: Head 6 singleton vs others
        # Key: Head 0 singleton vs others
        # Value: Head 3 singleton vs others

        results = {}

        # Query clustering prediction
        query_singleton = self.test_head_group_patching([6], "Query Singleton (Head 6)", saved_activation)
        query_others = self.test_head_group_patching([0,1,2,3,4,5,7,8,9,10,11], "Query Others", saved_activation)

        results['query'] = {
            'singleton': query_singleton,
            'others': query_others,
            'singleton_specializes': query_singleton['bug_fixed'] and not query_others['bug_fixed']
        }

        # Key clustering prediction
        key_singleton = self.test_head_group_patching([0], "Key Singleton (Head 0)", saved_activation)
        key_others = self.test_head_group_patching([1,2,3,4,5,6,7,8,9,10,11], "Key Others", saved_activation)

        results['key'] = {
            'singleton': key_singleton,
            'others': key_others,
            'singleton_specializes': key_singleton['bug_fixed'] and not key_others['bug_fixed']
        }

        # Value clustering prediction
        value_singleton = self.test_head_group_patching([3], "Value Singleton (Head 3)", saved_activation)
        value_others = self.test_head_group_patching([0,1,2,4,5,6,7,8,9,10,11], "Value Others", saved_activation)

        results['value'] = {
            'singleton': value_singleton,
            'others': value_others,
            'singleton_specializes': value_singleton['bug_fixed'] and not value_others['bug_fixed']
        }

        # Summary
        correct_predictions = sum([
            results['query']['singleton_specializes'],
            results['key']['singleton_specializes'],
            results['value']['singleton_specializes']
        ])

        print(f"\n  📊 Weight clustering predictions: {correct_predictions}/3 correct")

        return results

    def run_complete_verification(self):
        """Run complete verification with corrected methodology"""
        print("🔍 CORRECTED ACTIVATION PATCHING VERIFICATION")
        print("="*70)
        print("Using EXACT methodology from training_dynamics_analysis")
        print("="*70)

        results = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'target_layer': self.target_layer,
            'methodology': 'corrected_with_generation'
        }

        # Test baseline
        baseline = self.test_baseline_behavior()
        results['baseline'] = baseline

        if not baseline['has_bug']:
            print("\n⚠️  WARNING: No bug detected in baseline! Cannot test patching effectiveness.")
            print("This might indicate a methodology issue or model difference.")

        # Get clean activation
        saved_activation = self.get_clean_activation()

        # Test individual heads
        individual_results, fixing_heads = self.test_individual_heads(saved_activation)
        results['individual_heads'] = individual_results
        results['fixing_heads'] = fixing_heads

        # Test even/odd specialization
        even_odd_results = self.test_even_odd_specialization(saved_activation)
        results['even_odd'] = even_odd_results

        # Test weight clustering predictions
        clustering_results = self.test_weight_clustering_predictions(saved_activation)
        results['weight_clustering'] = clustering_results

        return results

    def save_results(self, results):
        """Save verification results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = RESULTS_DIR / f"corrected_activation_patching_{timestamp}.json"

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Saved results to {filepath}")
        return filepath


def main():
    """Run corrected activation patching verification"""
    verifier = CorrectedActivationPatchingVerifier()
    results = verifier.run_complete_verification()
    verifier.save_results(results)

    print("\n" + "="*70)
    print("🎯 CORRECTED VERIFICATION SUMMARY")
    print("="*70)

    # Baseline analysis
    baseline = results['baseline']
    print(f"\n📊 Baseline Analysis:")
    print(f"  Clean prompt response: '{baseline['clean_prompt']['text']}'")
    print(f"  Buggy prompt response: '{baseline['buggy_prompt']['text']}'")
    print(f"  Bug exists: {'✅' if baseline['has_bug'] else '❌'}")

    if baseline['has_bug']:
        # Individual head analysis
        fixing_heads = results['fixing_heads']
        print(f"\n🔬 Individual Head Analysis:")
        print(f"  Heads that fix bug: {fixing_heads}")
        print(f"  Count: {len(fixing_heads)}/12")

        # Even/odd specialization
        even_odd = results['even_odd']
        print(f"\n⚖️  Even/Odd Specialization:")
        print(f"  {even_odd['specialization']}")

        if even_odd['even_specializes']:
            print(f"  ✅ EVEN HEAD SPECIALIZATION CONFIRMED!")
        elif even_odd['odd_specializes']:
            print(f"  ✅ ODD HEAD SPECIALIZATION CONFIRMED!")

        # Weight clustering validation
        clustering = results['weight_clustering']
        print(f"\n🎯 Weight Clustering Validation:")
        for weight_type in ['query', 'key', 'value']:
            specializes = clustering[weight_type]['singleton_specializes']
            print(f"  {weight_type.capitalize()} singleton: {'✅' if specializes else '❌'}")

        correct_predictions = sum([
            clustering['query']['singleton_specializes'],
            clustering['key']['singleton_specializes'],
            clustering['value']['singleton_specializes']
        ])
        print(f"  Overall: {correct_predictions}/3 predictions correct")

    else:
        print(f"\n⚠️  Cannot validate specialization - no baseline bug detected!")
        print(f"  This suggests either:")
        print(f"    1. Model difference from original experiments")
        print(f"    2. Prompt format mismatch")
        print(f"    3. Different checkpoint than expected")

    print("\n✅ Corrected verification complete!")


if __name__ == "__main__":
    main()