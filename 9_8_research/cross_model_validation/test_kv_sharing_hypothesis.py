#!/usr/bin/env python3
"""
Test KV Sharing Hypothesis for Pattern Generalization
====================================================

REVISED HYPOTHESIS: The even/odd head specialization pattern fails when there's
extreme Key-Value sharing (high Query/KV ratio), not when heads are independent.

Critical Test Cases:
- Llama-3.1-8B: 32Q/8KV = 4:1 ratio → ✅ Pattern works
- Pythia-160M: 12Q/12KV = 1:1 ratio → ✅ Pattern works
- Gemma-2B: 8Q/1KV = 8:1 ratio → ❌ Pattern fails
- Llama-2-7b: 32Q/32KV = 1:1 ratio → ? Should work if hypothesis correct

Key insight: Extreme KV sharing may prevent specialization by forcing all
query heads to work with identical key/value representations.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import List, Dict
from contextlib import contextmanager
import json
from datetime import datetime
import time

class KVSharingHypothesisTest:
    def __init__(self, device: str = "cuda"):
        self.device = device

        # Critical test models based on KV sharing ratios
        self.test_models = {
            # Low KV sharing (1:1 ratio) - should work
            'low_kv_sharing': [
                "EleutherAI/pythia-160m",      # 12Q/12KV = 1:1 ✅ Confirmed works
                "meta-llama/Llama-2-7b-hf",   # 32Q/32KV = 1:1 ? Critical test
            ],

            # Moderate KV sharing (4:1 ratio) - should work
            'moderate_kv_sharing': [
                "meta-llama/Meta-Llama-3.1-8B-Instruct",  # 32Q/8KV = 4:1 ✅ Confirmed works
            ],

            # Extreme KV sharing (8:1+ ratio) - should fail
            'extreme_kv_sharing': [
                "google/gemma-2b",            # 8Q/1KV = 8:1 ❌ Confirmed fails
                "google/gemma-7b",            # 16Q/16KV = 1:1 ? Unexpected - need to check
            ]
        }

        self.saved_activations = {}
        self.hooks = []

    def analyze_kv_sharing(self, model_name: str) -> Dict:
        """Analyze KV sharing structure of a model"""

        try:
            config = AutoConfig.from_pretrained(model_name)

            analysis = {
                'model_name': model_name,
                'model_type': config.model_type,
                'n_query_heads': getattr(config, 'num_attention_heads', 'Unknown'),
                'n_layers': getattr(config, 'num_hidden_layers', 'Unknown'),
                'kv_sharing': {}
            }

            # Determine KV structure
            if hasattr(config, 'num_key_value_heads'):
                n_kv_heads = config.num_key_value_heads
                n_q_heads = analysis['n_query_heads']

                analysis['kv_sharing'] = {
                    'n_kv_heads': n_kv_heads,
                    'n_query_heads': n_q_heads,
                    'q_kv_ratio': n_q_heads / n_kv_heads,
                    'sharing_level': 'extreme' if n_q_heads / n_kv_heads >= 8 else 'moderate' if n_q_heads / n_kv_heads > 1 else 'none'
                }

                # Predict based on KV sharing hypothesis
                ratio = n_q_heads / n_kv_heads
                if ratio >= 6:  # High threshold for failure
                    analysis['predicted_pattern'] = 'even_odd_fails'
                else:
                    analysis['predicted_pattern'] = 'even_odd_works'

            else:
                # Standard MHA - each head has its own KV
                analysis['kv_sharing'] = {
                    'n_kv_heads': analysis['n_query_heads'],
                    'n_query_heads': analysis['n_query_heads'],
                    'q_kv_ratio': 1.0,
                    'sharing_level': 'none'
                }
                analysis['predicted_pattern'] = 'even_odd_works'

            return analysis

        except Exception as e:
            return {
                'model_name': model_name,
                'error': str(e),
                'predicted_pattern': 'error'
            }

    def get_attention_module(self, model, layer_idx: int):
        """Get attention module for different architectures"""
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return model.model.layers[layer_idx].self_attn
        elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
            return model.gpt_neox.layers[layer_idx].attention
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            return model.transformer.h[layer_idx].attn
        else:
            raise ValueError(f"Unknown model architecture")

    def save_activation_hook(self, key: str):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self.saved_activations[key] = hidden_states.detach().cpu()
        return hook_fn

    def selective_patch_hook(self, saved_activation: torch.Tensor, head_indices: List[int], n_heads: int):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            batch_size, seq_len, hidden_size = hidden_states.shape
            head_dim = hidden_size // n_heads

            hidden_states_reshaped = hidden_states.view(batch_size, seq_len, n_heads, head_dim)
            saved_reshaped = saved_activation.to(hidden_states.device).view(batch_size, -1, n_heads, head_dim)

            new_hidden = hidden_states_reshaped.clone()
            min_seq_len = min(seq_len, saved_reshaped.shape[1])

            for head_idx in head_indices:
                if head_idx < n_heads:
                    new_hidden[:, :min_seq_len, head_idx, :] = saved_reshaped[:, :min_seq_len, head_idx, :]

            new_hidden = new_hidden.view(batch_size, seq_len, hidden_size)

            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden
        return hook_fn

    @contextmanager
    def save_activation_context(self, model, prompt: str, layer_idx: int):
        try:
            module = self.get_attention_module(model, layer_idx)
            key = f"layer_{layer_idx}_attention"

            hook = module.register_forward_hook(self.save_activation_hook(key))
            self.hooks.append(hook)

            tokenizer = model.tokenizer  # Assume tokenizer is attached
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                _ = model(**inputs)

            yield self.saved_activations

        finally:
            for hook in self.hooks:
                hook.remove()
            self.hooks.clear()

    @contextmanager
    def patch_activation_context(self, model, saved_activation: torch.Tensor, head_indices: List[int], layer_idx: int, n_heads: int):
        try:
            module = self.get_attention_module(model, layer_idx)
            hook = module.register_forward_hook(
                self.selective_patch_hook(saved_activation, head_indices, n_heads)
            )
            self.hooks.append(hook)
            yield
        finally:
            for hook in self.hooks:
                hook.remove()
            self.hooks.clear()

    def test_model_pattern(self, model_name: str, max_trials: int = 25) -> Dict:
        """Test even/odd pattern on a specific model using proper activation patching"""

        print(f"\n🧪 Testing: {model_name}")
        print("-" * 50)

        try:
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=self.device
            )
            model.eval()
            model.tokenizer = tokenizer  # Attach for convenience

            # Get model specs
            config = AutoConfig.from_pretrained(model_name)
            n_heads = getattr(config, 'num_attention_heads', 12)
            n_layers = getattr(config, 'num_hidden_layers', 12)

            # Test middle layer
            test_layer = n_layers // 2

            # Define prompts
            correct_prompt = "Which is bigger: 9.8 or 9.11?"
            buggy_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"

            # Generate head lists
            even_heads = list(range(0, n_heads, 2))
            odd_heads = list(range(1, n_heads, 2))

            results = {
                'model_name': model_name,
                'n_heads': n_heads,
                'test_layer': test_layer,
                'even_success_rate': 0.0,
                'odd_success_rate': 0.0,
                'pattern_detected': False,
                'advantage': 'none'
            }

            def check_bug_fixed(output: str) -> bool:
                output_lower = output.lower()
                correct_patterns = ["9.8 is bigger", "9.8 is larger", "9.8"]
                bug_patterns = ["9.11 is bigger", "9.11 is larger", "9.11"]

                has_correct = any(pattern in output_lower for pattern in correct_patterns)
                has_bug = any(pattern in output_lower for pattern in bug_patterns)

                return has_correct and not has_bug

            # Test each head type with activation patching
            for head_type, head_list in [('even', even_heads), ('odd', odd_heads)]:
                print(f"  Testing {head_type} heads: {head_list}")

                success_count = 0
                sample_outputs = []

                # Save clean activation
                with self.save_activation_context(model, correct_prompt, test_layer) as saved:
                    clean_activation = saved[f"layer_{test_layer}_attention"]

                # Test with patching
                for trial in range(max_trials):
                    try:
                        with self.patch_activation_context(model, clean_activation, head_list, test_layer, n_heads):
                            inputs = tokenizer(buggy_prompt, return_tensors="pt").to(self.device)

                            with torch.no_grad():
                                outputs = model.generate(
                                    **inputs,
                                    max_new_tokens=20,
                                    do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id
                                )

                            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

                            if trial < 3:
                                sample_outputs.append(response.strip())

                            if check_bug_fixed(response):
                                success_count += 1

                    except Exception as e:
                        print(f"    Trial {trial} error: {e}")
                        continue

                success_rate = success_count / max_trials
                results[f'{head_type}_success_rate'] = success_rate

                print(f"    Success rate: {success_rate:.1%}")
                print(f"    Samples: {sample_outputs[:2]}")

            # Determine pattern
            even_rate = results['even_success_rate']
            odd_rate = results['odd_success_rate']

            difference = abs(even_rate - odd_rate)
            results['pattern_detected'] = difference > 0.3  # 30% threshold

            if even_rate > odd_rate + 0.2:
                results['advantage'] = 'even'
            elif odd_rate > even_rate + 0.2:
                results['advantage'] = 'odd'
            else:
                results['advantage'] = 'none'

            print(f"  Pattern detected: {results['pattern_detected']} ({results['advantage']} advantage)")

            return results

        except Exception as e:
            print(f"  ❌ Error: {e}")
            return {
                'model_name': model_name,
                'error': str(e),
                'pattern_detected': False
            }

    def run_kv_sharing_test(self) -> Dict:
        """Run comprehensive KV sharing hypothesis test"""

        print("=" * 70)
        print("KV SHARING HYPOTHESIS TEST")
        print("=" * 70)
        print("Testing whether Q/KV ratio determines even/odd pattern emergence")
        print()

        results = {
            'timestamp': datetime.now().isoformat(),
            'hypothesis': 'High Q/KV sharing ratio prevents even/odd pattern emergence',
            'predictions': {},
            'test_results': {},
            'validation': {}
        }

        # Step 1: Analyze KV sharing for all models
        print("STEP 1: KV SHARING ANALYSIS")
        print("=" * 40)

        all_models = []
        for category, models in self.test_models.items():
            all_models.extend(models)

        for model_name in all_models:
            analysis = self.analyze_kv_sharing(model_name)
            results['predictions'][model_name] = analysis

            if 'error' not in analysis:
                kv_info = analysis['kv_sharing']
                print(f"\n{model_name}:")
                print(f"  Q/KV ratio: {kv_info['q_kv_ratio']:.1f}:1")
                print(f"  Sharing level: {kv_info['sharing_level']}")
                print(f"  Prediction: {analysis['predicted_pattern']}")

        # Step 2: Test critical models
        print(f"\n\nSTEP 2: EMPIRICAL TESTING")
        print("=" * 40)

        # Test key models that will validate/refute hypothesis
        critical_tests = [
            "meta-llama/Llama-2-7b-hf",     # 1:1 ratio - should work if hypothesis correct
            "google/gemma-2b",              # 8:1 ratio - confirmed fails
        ]

        for model_name in critical_tests:
            if model_name in results['predictions']:
                test_result = self.test_model_pattern(model_name, max_trials=20)
                results['test_results'][model_name] = test_result

        # Step 3: Validate hypothesis
        print(f"\n\nSTEP 3: HYPOTHESIS VALIDATION")
        print("=" * 40)

        validation = {
            'correct_predictions': 0,
            'total_predictions': 0,
            'hypothesis_supported': False,
            'key_findings': []
        }

        for model_name, test_result in results['test_results'].items():
            if 'error' in test_result:
                continue

            prediction = results['predictions'][model_name]
            if 'error' in prediction:
                continue

            predicted = prediction['predicted_pattern']
            actual = 'even_odd_works' if test_result.get('advantage') == 'even' else 'even_odd_fails'

            is_correct = predicted == actual

            validation['total_predictions'] += 1
            if is_correct:
                validation['correct_predictions'] += 1

            q_kv_ratio = prediction['kv_sharing']['q_kv_ratio']

            print(f"\n{model_name}:")
            print(f"  Q/KV ratio: {q_kv_ratio:.1f}:1")
            print(f"  Predicted: {predicted}")
            print(f"  Actual: {actual}")
            print(f"  Correct: {'✅' if is_correct else '❌'}")

            validation['key_findings'].append({
                'model': model_name,
                'q_kv_ratio': q_kv_ratio,
                'predicted': predicted,
                'actual': actual,
                'correct': is_correct
            })

        if validation['total_predictions'] > 0:
            accuracy = validation['correct_predictions'] / validation['total_predictions']
            validation['accuracy'] = accuracy
            validation['hypothesis_supported'] = accuracy >= 0.75

        results['validation'] = validation

        return results

def main():
    """Run KV sharing hypothesis test"""

    tester = KVSharingHypothesisTest()
    results = tester.run_kv_sharing_test()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"kv_sharing_hypothesis_test_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📁 Results saved to: {output_file}")

    # Final assessment
    print("\n" + "=" * 70)
    print("KV SHARING HYPOTHESIS ASSESSMENT")
    print("=" * 70)

    validation = results['validation']
    if validation.get('hypothesis_supported', False):
        print("✅ HYPOTHESIS SUPPORTED: Q/KV ratio predicts pattern emergence")
        print(f"   Accuracy: {validation.get('accuracy', 0):.1%}")

        # Print key insight
        print(f"\n💡 KEY INSIGHT:")
        print(f"   High Q/KV ratios (≥6:1) prevent even/odd specialization")
        print(f"   because extreme sharing forces all heads to work with identical K,V representations")

    else:
        print("❌ HYPOTHESIS NEEDS FURTHER REFINEMENT")
        print("   Other factors beyond Q/KV ratio are involved")

if __name__ == "__main__":
    main()