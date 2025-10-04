#!/usr/bin/env python3
"""
Cross-Model Validation: Pythia-160M
===================================

Test whether the even/odd head specialization pattern discovered in Llama-3.1-8B
generalizes to Pythia-160M, a different architecture and model family.

Key questions:
1. Does Pythia-160M exhibit the 9.8 vs 9.11 bug?
2. Do even-indexed heads fix the bug?
3. Is there a critical mass requirement?
4. Does permutation destroy functionality?
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

class PythiaExperiment:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model_name = "EleutherAI/pythia-160m"

        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        self.model.eval()

        # Pythia-160M has 12 attention heads per layer, 12 layers
        self.n_heads = 12
        self.n_layers = 12

        # We'll test multiple layers to find the best one
        self.target_layers = [6, 7, 8, 9, 10, 11]  # Middle to late layers

        self.saved_activations = {}
        self.hooks = []

    def get_attention_module(self, layer_idx: int):
        """Get attention module for Pythia architecture"""
        return self.model.gpt_neox.layers[layer_idx].attention

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
                if head_idx < self.n_heads:
                    new_hidden[:, :min_seq_len, head_idx, :] = saved_reshaped[:, :min_seq_len, head_idx, :]

            new_hidden = new_hidden.view(batch_size, seq_len, hidden_size)

            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden
        return hook_fn

    @contextmanager
    def save_activation_context(self, prompt: str, layer_idx: int):
        try:
            module = self.get_attention_module(layer_idx)
            key = f"layer_{layer_idx}_attention"

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
    def patch_activation_context(self, saved_activation: torch.Tensor, head_indices: List[int], layer_idx: int):
        try:
            module = self.get_attention_module(layer_idx)
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
            "9.8 is more", "9.8", "9.8 is the bigger"
        ]

        bug_patterns = [
            "9.11 is bigger", "9.11 is larger", "9.11 is greater",
            "9.11 is more", "9.11 is the bigger"
        ]

        has_correct = any(pattern in output_lower for pattern in correct_patterns)
        has_bug = any(pattern in output_lower for pattern in bug_patterns)

        # For small models, also check if 9.8 appears first
        if not has_correct and not has_bug:
            first_line = output.strip().split('\n')[0] if output.strip() else ""
            if '9.8' in first_line and '9.11' not in first_line:
                has_correct = True

        return has_correct and not has_bug

    def test_baseline_bug(self) -> Dict:
        """Test if Pythia-160M exhibits the 9.8 vs 9.11 bug"""
        print("Testing baseline bug in Pythia-160M...")

        test_prompts = [
            "Which is bigger: 9.8 or 9.11?",
            "Q: Which is bigger: 9.8 or 9.11?\nA:",
            "Compare 9.8 and 9.11. Which is larger?",
            "9.8 vs 9.11 - which is greater?"
        ]

        results = {}

        for prompt in test_prompts:
            correct_count = 0
            responses = []

            for trial in range(10):
                response = self.generate(prompt, max_new_tokens=30)
                responses.append(response)

                if self.check_bug_fixed(response):
                    correct_count += 1

            bug_rate = 1.0 - (correct_count / 10)
            results[prompt] = {
                'bug_rate': bug_rate,
                'correct_count': correct_count,
                'sample_responses': responses[:3]
            }

            print(f"  Prompt: {prompt[:30]}...")
            print(f"    Bug rate: {bug_rate:.1%}")
            print(f"    Sample: {responses[0][:50]}...")

        return results

    def test_layer_intervention(self, layer_idx: int, head_indices: List[int], n_trials: int = 20) -> Dict:
        """Test intervention on a specific layer"""

        correct_prompt = "Which is bigger: 9.8 or 9.11?"
        buggy_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"

        # Save activation from correct context
        with self.save_activation_context(correct_prompt, layer_idx) as saved:
            correct_activation = saved[f"layer_{layer_idx}_attention"]

        success_count = 0
        sample_outputs = []

        for trial in range(n_trials):
            with self.patch_activation_context(correct_activation, head_indices, layer_idx):
                output = self.generate(buggy_prompt, max_new_tokens=20)

            if self.check_bug_fixed(output):
                success_count += 1

            if trial < 3:
                sample_outputs.append(output.strip())

        success_rate = success_count / n_trials

        return {
            'layer': layer_idx,
            'heads': head_indices,
            'success_rate': success_rate,
            'success_count': success_count,
            'n_trials': n_trials,
            'sample_outputs': sample_outputs
        }

    def find_best_layer(self) -> Dict:
        """Find which layer responds best to even head intervention"""
        print("Finding best layer for intervention...")

        # Test all even heads on each target layer
        even_heads = list(range(0, self.n_heads, 2))  # [0, 2, 4, 6, 8, 10] for 12 heads

        layer_results = {}

        for layer_idx in self.target_layers:
            print(f"  Testing layer {layer_idx}...")
            result = self.test_layer_intervention(layer_idx, even_heads, n_trials=20)
            layer_results[layer_idx] = result
            print(f"    Success rate: {result['success_rate']:.1%}")

        # Find best layer
        best_layer = max(layer_results.keys(), key=lambda k: layer_results[k]['success_rate'])
        best_rate = layer_results[best_layer]['success_rate']

        print(f"Best layer: {best_layer} with {best_rate:.1%} success")

        return {
            'layer_results': layer_results,
            'best_layer': best_layer,
            'best_success_rate': best_rate
        }

    def test_even_odd_pattern(self, layer_idx: int) -> Dict:
        """Test even vs odd pattern on best layer"""
        print(f"Testing even/odd pattern on layer {layer_idx}...")

        patterns = {
            'all_even': list(range(0, self.n_heads, 2)),
            'all_odd': list(range(1, self.n_heads, 2)),
            'first_half_even': list(range(0, self.n_heads//2, 2)),
            'second_half_even': list(range(self.n_heads//2, self.n_heads, 2)),
            'random_6': [1, 3, 5, 7, 9, 11],  # All odd for comparison
        }

        results = {}

        for pattern_name, head_indices in patterns.items():
            print(f"  Testing {pattern_name}: {head_indices}")
            result = self.test_layer_intervention(layer_idx, head_indices, n_trials=25)
            results[pattern_name] = result
            print(f"    Success rate: {result['success_rate']:.1%}")

        return results

    def run_pythia_validation(self) -> Dict:
        """Run comprehensive Pythia validation"""

        print("=" * 60)
        print("PYTHIA-160M CROSS-MODEL VALIDATION")
        print("=" * 60)
        print(f"Model: {self.model_name}")
        print(f"Architecture: {self.n_heads} heads per layer, {self.n_layers} layers")
        print()

        results = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'baseline_bug_test': {},
            'layer_analysis': {},
            'even_odd_test': {},
            'analysis': {}
        }

        # 1. Test baseline bug
        print("1. BASELINE BUG TEST")
        print("-" * 30)
        baseline_results = self.test_baseline_bug()
        results['baseline_bug_test'] = baseline_results

        # Calculate average bug rate
        avg_bug_rate = np.mean([r['bug_rate'] for r in baseline_results.values()])
        print(f"Average bug rate: {avg_bug_rate:.1%}")

        if avg_bug_rate < 0.3:
            print("⚠️  Model doesn't show strong 9.8 vs 9.11 bug - results may be less meaningful")

        print()

        # 2. Find best layer
        print("2. LAYER ANALYSIS")
        print("-" * 30)
        layer_analysis = self.find_best_layer()
        results['layer_analysis'] = layer_analysis
        best_layer = layer_analysis['best_layer']
        print()

        # 3. Test even/odd pattern on best layer
        print("3. EVEN/ODD PATTERN TEST")
        print("-" * 30)
        even_odd_results = self.test_even_odd_pattern(best_layer)
        results['even_odd_test'] = even_odd_results
        print()

        # 4. Analysis
        results['analysis'] = self.analyze_pythia_results(results)

        # Save results (convert numpy types to native Python)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"pythia_160m_validation_{timestamp}.json"

        def convert_numpy(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        results_converted = convert_numpy(results)

        with open(output_file, 'w') as f:
            json.dump(results_converted, f, indent=2)

        print(f"Results saved to: {output_file}")

        return results

    def analyze_pythia_results(self, results: Dict) -> Dict:
        """Analyze Pythia results for pattern generalization"""

        analysis = {
            'bug_exists': False,
            'even_advantage': False,
            'pattern_generalizes': False,
            'summary': ""
        }

        # Check if bug exists
        avg_bug_rate = np.mean([r['bug_rate'] for r in results['baseline_bug_test'].values()])
        analysis['bug_exists'] = avg_bug_rate > 0.3

        # Check even vs odd performance
        if 'even_odd_test' in results:
            even_rate = results['even_odd_test']['all_even']['success_rate']
            odd_rate = results['even_odd_test']['all_odd']['success_rate']

            analysis['even_advantage'] = even_rate > odd_rate + 0.2  # 20% threshold
            analysis['even_success_rate'] = even_rate
            analysis['odd_success_rate'] = odd_rate

        # Overall pattern generalization
        analysis['pattern_generalizes'] = analysis['bug_exists'] and analysis['even_advantage']

        # Generate summary
        if analysis['pattern_generalizes']:
            analysis['summary'] = "✅ PATTERN GENERALIZES: Pythia-160M shows even-head advantage"
        elif analysis['bug_exists'] and not analysis['even_advantage']:
            analysis['summary'] = "⚠️  BUG EXISTS but no clear even-head advantage"
        elif not analysis['bug_exists']:
            analysis['summary'] = "❌ Bug doesn't manifest strongly in Pythia-160M"
        else:
            analysis['summary'] = "❓ Results inconclusive"

        return analysis

def main():
    """Run Pythia-160M validation"""

    start_time = time.time()

    experiment = PythiaExperiment(device="cuda")
    results = experiment.run_pythia_validation()

    print("=" * 60)
    print("PYTHIA-160M VALIDATION SUMMARY")
    print("=" * 60)

    analysis = results['analysis']
    print(f"\n{analysis['summary']}")

    if 'even_success_rate' in analysis:
        print(f"\nDetailed Results:")
        print(f"  Even heads: {analysis['even_success_rate']:.1%} success")
        print(f"  Odd heads: {analysis['odd_success_rate']:.1%} success")
        print(f"  Advantage: {analysis['even_success_rate'] - analysis['odd_success_rate']:.1%}")

    print(f"\nBest layer: {results['layer_analysis']['best_layer']}")
    print(f"Best success rate: {results['layer_analysis']['best_success_rate']:.1%}")

    total_time = time.time() - start_time
    print(f"\nValidation completed in {total_time:.1f} seconds")

if __name__ == "__main__":
    main()