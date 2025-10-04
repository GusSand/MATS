#!/usr/bin/env python3
"""
Test Pythia Training Dynamics - When Does Even/Odd Specialization Emerge?
========================================================================

This experiment tests Pythia model checkpoints across training to determine:
1. When does even/odd head specialization first appear?
2. Is emergence gradual or sudden?
3. Does it correlate with other capability milestones?
4. What training dynamics drive this specialization?

Pythia models have checkpoints at: 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k, 256k, 512k, 1000k steps
This gives us unprecedented insight into training dynamics.
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

class PythiaTrainingDynamicsTest:
    def __init__(self, device: str = "cuda"):
        self.device = device

        # Pythia-160M checkpoints (as git revisions/branches)
        # Using log-spaced and key evenly-spaced checkpoints available
        self.checkpoints = [
            ("step1000", "step1000"),
            ("step2000", "step2000"),
            ("step4000", "step4000"),
            ("step8000", "step8000"),
            ("step16000", "step16000"),
            ("step32000", "step32000"),
            ("step64000", "step64000"),
            ("step80000", "step80000"),   # mid training
            ("step100000", "step100000"), # late-mid training
            ("step120000", "step120000"), # late training
            ("step143000", "main")        # final model - step143000 = main branch
        ]

        self.base_model = "EleutherAI/pythia-160m"
        self.n_heads = 12
        self.n_layers = 12
        self.target_layer = 6  # We know this works for final model

        self.saved_activations = {}
        self.hooks = []

    def get_model_name_and_revision(self, checkpoint_tuple: tuple) -> tuple:
        """Get the model name and revision for a specific checkpoint"""
        checkpoint_name, revision = checkpoint_tuple
        return self.base_model, revision

    def get_attention_module(self, model, layer_idx: int):
        """Get attention module for Pythia architecture"""
        return model.gpt_neox.layers[layer_idx].attention

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
    def save_activation_context(self, model, prompt: str):
        try:
            module = self.get_attention_module(model, self.target_layer)
            key = f"layer_{self.target_layer}_attention"

            hook = module.register_forward_hook(self.save_activation_hook(key))
            self.hooks.append(hook)

            tokenizer = model.tokenizer
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                _ = model(**inputs)

            yield self.saved_activations

        finally:
            for hook in self.hooks:
                hook.remove()
            self.hooks.clear()

    @contextmanager
    def patch_activation_context(self, model, saved_activation: torch.Tensor, head_indices: List[int]):
        try:
            module = self.get_attention_module(model, self.target_layer)
            hook = module.register_forward_hook(
                self.selective_patch_hook(saved_activation, head_indices)
            )
            self.hooks.append(hook)
            yield
        finally:
            for hook in self.hooks:
                hook.remove()
            self.hooks.clear()

    def test_checkpoint_specialization(self, checkpoint_tuple: tuple, max_trials: int = 15) -> Dict:
        """Test even/odd specialization for a specific checkpoint"""

        checkpoint_name, revision = checkpoint_tuple
        model_name, model_revision = self.get_model_name_and_revision(checkpoint_tuple)
        print(f"\n🧪 Testing {checkpoint_name} ({model_name} @ {model_revision})")
        print("-" * 60)

        try:
            # Load model and tokenizer with specific revision
            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                revision=model_revision,
                torch_dtype=torch.float16,
                device_map=self.device
            )
            model.eval()
            model.tokenizer = tokenizer

            # Test prompts
            correct_prompt = "Which is bigger: 9.8 or 9.11?"
            buggy_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"

            # Generate head lists
            even_heads = list(range(0, self.n_heads, 2))  # [0, 2, 4, 6, 8, 10]
            odd_heads = list(range(1, self.n_heads, 2))   # [1, 3, 5, 7, 9, 11]

            results = {
                'checkpoint': checkpoint_name,
                'model_name': model_name,
                'even_success_rate': 0.0,
                'odd_success_rate': 0.0,
                'baseline_success_rate': 0.0,
                'specialization_strength': 0.0,
                'pattern_detected': False
            }

            def check_bug_fixed(output: str) -> bool:
                output_lower = output.lower()
                correct_patterns = ["9.8 is bigger", "9.8 is larger", "9.8"]
                bug_patterns = ["9.11 is bigger", "9.11 is larger", "9.11"]

                has_correct = any(pattern in output_lower for pattern in correct_patterns)
                has_bug = any(pattern in output_lower for pattern in bug_patterns)

                return has_correct and not has_bug

            # Test baseline (no patching)
            baseline_success = 0
            for trial in range(max_trials):
                try:
                    inputs = tokenizer(buggy_prompt, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=20,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id
                        )
                    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                    if check_bug_fixed(response):
                        baseline_success += 1
                except:
                    continue

            results['baseline_success_rate'] = baseline_success / max_trials
            print(f"  Baseline success: {results['baseline_success_rate']:.1%}")

            # Test each head type with patching
            for head_type, head_list in [('even', even_heads), ('odd', odd_heads)]:
                success_count = 0
                sample_outputs = []

                # Save clean activation
                try:
                    with self.save_activation_context(model, correct_prompt) as saved:
                        clean_activation = saved[f"layer_{self.target_layer}_attention"]

                    # Test with patching
                    for trial in range(max_trials):
                        try:
                            with self.patch_activation_context(model, clean_activation, head_list):
                                inputs = tokenizer(buggy_prompt, return_tensors="pt").to(self.device)

                                with torch.no_grad():
                                    outputs = model.generate(
                                        **inputs,
                                        max_new_tokens=20,
                                        do_sample=False,
                                        pad_token_id=tokenizer.pad_token_id
                                    )

                                response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

                                if trial < 2:
                                    sample_outputs.append(response.strip()[:50])

                                if check_bug_fixed(response):
                                    success_count += 1

                        except Exception as e:
                            continue

                except Exception as e:
                    print(f"  Error testing {head_type} heads: {e}")
                    success_count = 0

                success_rate = success_count / max_trials
                results[f'{head_type}_success_rate'] = success_rate

                print(f"  {head_type.title()} heads: {success_rate:.1%} success")
                if sample_outputs:
                    print(f"    Sample: {sample_outputs[0]}")

            # Calculate specialization metrics
            even_rate = results['even_success_rate']
            odd_rate = results['odd_success_rate']

            # Specialization strength (how much even outperforms odd)
            results['specialization_strength'] = even_rate - odd_rate

            # Pattern detected if even significantly outperforms odd
            results['pattern_detected'] = (even_rate - odd_rate) > 0.3

            print(f"  Specialization strength: {results['specialization_strength']:+.2f}")
            print(f"  Pattern detected: {results['pattern_detected']}")

            # Clean up model to save memory
            del model
            del tokenizer
            torch.cuda.empty_cache()

            return results

        except Exception as e:
            print(f"  ❌ Error loading {checkpoint_name}: {e}")
            return {
                'checkpoint': checkpoint_name,
                'error': str(e),
                'pattern_detected': False,
                'specialization_strength': 0.0
            }

    def run_training_dynamics_analysis(self) -> Dict:
        """Run comprehensive training dynamics analysis"""

        print("=" * 80)
        print("PYTHIA TRAINING DYNAMICS ANALYSIS")
        print("=" * 80)
        print("Testing when even/odd head specialization emerges during training")
        print(f"Model: {self.base_model}")
        print(f"Checkpoints: {len(self.checkpoints)} from 1k to 1000k steps")
        print()

        results = {
            'timestamp': datetime.now().isoformat(),
            'base_model': self.base_model,
            'checkpoints_tested': self.checkpoints,
            'checkpoint_results': {},
            'analysis': {}
        }

        # Test each checkpoint
        for i, checkpoint_tuple in enumerate(self.checkpoints):
            checkpoint_name = checkpoint_tuple[0]
            print(f"\n{'='*20} CHECKPOINT {i+1}/{len(self.checkpoints)} {'='*20}")

            checkpoint_result = self.test_checkpoint_specialization(checkpoint_tuple)
            results['checkpoint_results'][checkpoint_name] = checkpoint_result

            # Show progress
            if 'specialization_strength' in checkpoint_result:
                strength = checkpoint_result['specialization_strength']
                detected = "✅" if checkpoint_result['pattern_detected'] else "❌"
                print(f"  Result: {detected} Specialization = {strength:+.2f}")

        # Analyze emergence patterns
        print(f"\n\n{'='*20} EMERGENCE ANALYSIS {'='*20}")
        analysis = self.analyze_emergence_pattern(results['checkpoint_results'])
        results['analysis'] = analysis

        # Create visualization
        self.visualize_training_dynamics(results)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"pythia_training_dynamics_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n📁 Results saved to: {output_file}")

        return results

    def analyze_emergence_pattern(self, checkpoint_results: Dict) -> Dict:
        """Analyze when and how specialization emerges"""

        analysis = {
            'emergence_point': None,
            'emergence_type': 'unknown',
            'specialization_timeline': [],
            'key_insights': []
        }

        # Extract timeline data
        valid_checkpoints = []
        specialization_scores = []

        for checkpoint_tuple in self.checkpoints:
            checkpoint_name = checkpoint_tuple[0]
            if checkpoint_name in checkpoint_results:
                result = checkpoint_results[checkpoint_name]
                if 'specialization_strength' in result and 'error' not in result:
                    valid_checkpoints.append(checkpoint_name)
                    specialization_scores.append(result['specialization_strength'])

        analysis['specialization_timeline'] = list(zip(valid_checkpoints, specialization_scores))

        if len(specialization_scores) < 3:
            analysis['key_insights'].append("Insufficient data for timeline analysis")
            return analysis

        # Find emergence point (first checkpoint with strong specialization)
        emergence_threshold = 0.5  # 50% difference between even and odd

        for i, (checkpoint, score) in enumerate(zip(valid_checkpoints, specialization_scores)):
            if score > emergence_threshold:
                analysis['emergence_point'] = checkpoint
                analysis['emergence_step'] = i
                break

        # Analyze emergence type
        if analysis['emergence_point']:
            emergence_idx = analysis['emergence_step']

            if emergence_idx == 0:
                analysis['emergence_type'] = 'early'
                analysis['key_insights'].append("Specialization emerges very early in training")
            elif emergence_idx < len(specialization_scores) // 2:
                analysis['emergence_type'] = 'early-mid'
                analysis['key_insights'].append("Specialization emerges in early-middle training")
            elif emergence_idx < 3 * len(specialization_scores) // 4:
                analysis['emergence_type'] = 'late-mid'
                analysis['key_insights'].append("Specialization emerges in late-middle training")
            else:
                analysis['emergence_type'] = 'late'
                analysis['key_insights'].append("Specialization emerges late in training")

            # Check if emergence is gradual or sudden
            if emergence_idx > 0:
                pre_score = specialization_scores[emergence_idx - 1]
                post_score = specialization_scores[emergence_idx]
                jump = post_score - pre_score

                if jump > 0.4:
                    analysis['key_insights'].append("Emergence appears sudden (large jump)")
                else:
                    analysis['key_insights'].append("Emergence appears gradual")

        else:
            analysis['emergence_type'] = 'none'
            analysis['key_insights'].append("No clear specialization emergence detected")

        # Analyze final strength
        if specialization_scores:
            final_score = specialization_scores[-1]
            if final_score > 0.8:
                analysis['key_insights'].append("Strong final specialization achieved")
            elif final_score > 0.3:
                analysis['key_insights'].append("Moderate final specialization achieved")
            else:
                analysis['key_insights'].append("Weak or no final specialization")

        print(f"\nEMERGENCE ANALYSIS:")
        print(f"  Emergence point: {analysis['emergence_point']}")
        print(f"  Emergence type: {analysis['emergence_type']}")
        for insight in analysis['key_insights']:
            print(f"  • {insight}")

        return analysis

    def visualize_training_dynamics(self, results: Dict):
        """Create visualization of specialization emergence"""

        checkpoint_results = results['checkpoint_results']

        # Extract data for plotting
        steps = []
        specialization_scores = []
        even_rates = []
        odd_rates = []
        baseline_rates = []

        step_mapping = {
            'step1000': 1000,
            'step2000': 2000,
            'step4000': 4000,
            'step8000': 8000,
            'step16000': 16000,
            'step32000': 32000,
            'step64000': 64000,
            'step80000': 80000,
            'step100000': 100000,
            'step120000': 120000,
            'step143000': 143000  # Final model
        }

        for checkpoint_tuple in self.checkpoints:
            checkpoint_name = checkpoint_tuple[0]
            if checkpoint_name in checkpoint_results:
                result = checkpoint_results[checkpoint_name]
                if 'specialization_strength' in result and 'error' not in result:
                    steps.append(step_mapping[checkpoint_name])
                    specialization_scores.append(result['specialization_strength'])
                    even_rates.append(result['even_success_rate'])
                    odd_rates.append(result['odd_success_rate'])
                    baseline_rates.append(result['baseline_success_rate'])

        if len(steps) < 2:
            print("Insufficient data for visualization")
            return

        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Specialization strength over training
        ax1.plot(steps, specialization_scores, 'bo-', linewidth=2, markersize=6, label='Specialization Strength')
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Strong Specialization Threshold')
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Even - Odd Success Rate')
        ax1.set_title('Even/Odd Head Specialization Emergence During Training', fontweight='bold')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Individual success rates
        ax2.plot(steps, even_rates, 'go-', linewidth=2, markersize=6, label='Even Heads', alpha=0.8)
        ax2.plot(steps, odd_rates, 'ro-', linewidth=2, markersize=6, label='Odd Heads', alpha=0.8)
        ax2.plot(steps, baseline_rates, 'ko-', linewidth=2, markersize=6, label='Baseline (No Patch)', alpha=0.6)
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Success Rate')
        ax2.set_title('Individual Head Type Performance', fontweight='bold')
        ax2.set_xscale('log')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"figures/pythia_training_dynamics_{timestamp}.png"
        os.makedirs("figures", exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to: {output_file}")

        plt.show()

def main():
    """Run Pythia training dynamics analysis"""

    print("🧬 PYTHIA TRAINING DYNAMICS EXPERIMENT")
    print("Investigating when even/odd head specialization emerges during training")
    print()

    start_time = time.time()

    tester = PythiaTrainingDynamicsTest()
    results = tester.run_training_dynamics_analysis()

    total_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("TRAINING DYNAMICS ANALYSIS COMPLETE")
    print("=" * 80)

    analysis = results['analysis']
    print(f"\nKey Findings:")
    print(f"  Emergence Point: {analysis['emergence_point']}")
    print(f"  Emergence Type: {analysis['emergence_type']}")

    for insight in analysis['key_insights']:
        print(f"  • {insight}")

    print(f"\nAnalysis completed in {total_time:.1f} seconds")
    print("This provides unprecedented insight into when and how attention head specialization emerges!")

if __name__ == "__main__":
    main()