#!/usr/bin/env python3
"""
Activation Steering Analysis

Unlike patching (which replaces activations), steering ADDS a direction vector
to shift the model's behavior. This is: output = original + alpha * steering_vector

Key questions:
1. Can we extract a "security direction" from activations?
2. Can we steer the neutral prompt toward snprintf by adding this direction?
3. At which layers does steering work best?
4. What steering multipliers (alpha) are most effective?
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import warnings
import os
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Prompts
SECURE_PROMPT = '''// WARNING: Use snprintf to prevent buffer overflows
int list_item_to_string(node *head, char *str, size_t str_size) {
    if (head == NULL) return EXIT_FAILURE;
    '''

NEUTRAL_PROMPT = '''int list_item_to_string(node *head, char *str, size_t str_size) {
    if (head == NULL) return EXIT_FAILURE;
    '''


class ActivationSteering:
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        print("Loading model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()

        self.n_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size

        # Token IDs
        self.snprintf_token = self.tokenizer.encode(" snprintf", add_special_tokens=False)[0]
        self.sprintf_token = self.tokenizer.encode(" sprintf", add_special_tokens=False)[0]

        print(f"Model loaded: {self.n_layers} layers, hidden_size={self.hidden_size}")
        print(f"snprintf token: {self.snprintf_token}, sprintf token: {self.sprintf_token}")

        # Storage for activations and hooks
        self.activations = {}
        self.hooks = []

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def collect_activations(self, prompt: str) -> dict:
        """Collect residual stream activations at each layer for the last token."""
        self.activations = {}

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                # Store last token activation
                self.activations[layer_idx] = h[:, -1, :].detach().clone()
                return output
            return hook_fn

        # Register hooks
        self.clear_hooks()
        for layer_idx in range(self.n_layers):
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(hook)

        # Forward pass
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get final probabilities
        final_logits = outputs.logits[0, -1, :]
        probs = torch.softmax(final_logits, dim=-1)

        result = {
            'activations': {k: v.cpu() for k, v in self.activations.items()},
            'snprintf_prob': probs[self.snprintf_token].item(),
            'sprintf_prob': probs[self.sprintf_token].item(),
            'seq_len': inputs['input_ids'].shape[1]
        }

        self.clear_hooks()
        return result

    def compute_steering_vector(self, secure_acts: dict, neutral_acts: dict) -> dict:
        """
        Compute steering vector = secure_activation - neutral_activation at each layer.
        This is the "security direction" that shifts behavior toward secure outputs.
        """
        steering_vectors = {}

        for layer_idx in range(self.n_layers):
            secure_act = secure_acts['activations'][layer_idx]
            neutral_act = neutral_acts['activations'][layer_idx]

            # Steering vector: direction from neutral to secure
            steering_vec = secure_act - neutral_act
            steering_vectors[layer_idx] = steering_vec

        return steering_vectors

    def steer_at_layer(
        self,
        prompt: str,
        steering_vectors: dict,
        target_layer: int,
        alpha: float = 1.0
    ) -> dict:
        """
        Apply steering at a specific layer.

        new_activation = original + alpha * steering_vector

        alpha=1: Full steering (equivalent to replacing with secure activation)
        alpha>1: Over-steering (pushing further in security direction)
        alpha<0: Anti-steering (pushing toward insecure)
        """
        steering_vec = steering_vectors[target_layer].to(self.device)

        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output

            # Add steering vector to last token position
            h[:, -1, :] = h[:, -1, :] + alpha * steering_vec

            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h

        # Register hook at target layer
        self.clear_hooks()
        layer = self.model.model.layers[target_layer]
        hook = layer.register_forward_hook(steering_hook)
        self.hooks.append(hook)

        # Forward pass
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get probabilities
        final_logits = outputs.logits[0, -1, :]
        probs = torch.softmax(final_logits, dim=-1)

        result = {
            'snprintf_prob': probs[self.snprintf_token].item(),
            'sprintf_prob': probs[self.sprintf_token].item(),
            'target_layer': target_layer,
            'alpha': alpha
        }

        self.clear_hooks()
        return result

    def steer_at_layers(
        self,
        prompt: str,
        steering_vectors: dict,
        target_layers: list,
        alpha: float = 1.0
    ) -> dict:
        """Apply steering at multiple layers simultaneously."""
        def make_steering_hook(layer_idx):
            steering_vec = steering_vectors[layer_idx].to(self.device)

            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output

                h[:, -1, :] = h[:, -1, :] + alpha * steering_vec

                if isinstance(output, tuple):
                    return (h,) + output[1:]
                return h
            return hook_fn

        # Register hooks at all target layers
        self.clear_hooks()
        for layer_idx in target_layers:
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(make_steering_hook(layer_idx))
            self.hooks.append(hook)

        # Forward pass
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get probabilities
        final_logits = outputs.logits[0, -1, :]
        probs = torch.softmax(final_logits, dim=-1)

        result = {
            'snprintf_prob': probs[self.snprintf_token].item(),
            'sprintf_prob': probs[self.sprintf_token].item(),
            'target_layers': target_layers,
            'alpha': alpha
        }

        self.clear_hooks()
        return result

    def layer_sweep(self, neutral_prompt: str, steering_vectors: dict, alpha: float = 1.0) -> list:
        """Test steering at each layer individually."""
        results = []

        for layer_idx in range(self.n_layers):
            result = self.steer_at_layer(
                neutral_prompt, steering_vectors, layer_idx, alpha
            )
            results.append(result)

        return results

    def alpha_sweep(
        self,
        neutral_prompt: str,
        steering_vectors: dict,
        target_layer: int,
        alphas: list = None
    ) -> list:
        """Test different steering strengths at a specific layer."""
        if alphas is None:
            alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

        results = []
        for alpha in alphas:
            result = self.steer_at_layer(
                neutral_prompt, steering_vectors, target_layer, alpha
            )
            results.append(result)

        return results

    def visualize(self, layer_results: list, alpha_results: dict, output_path: Path,
                  baseline_secure: float, baseline_neutral: float):
        """Create visualization of steering results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Layer sweep - P(snprintf) at each layer
        ax1 = axes[0, 0]
        layers = [r['target_layer'] for r in layer_results]
        probs = [r['snprintf_prob'] * 100 for r in layer_results]

        ax1.bar(layers, probs, color='blue', alpha=0.7)
        ax1.axhline(y=baseline_secure * 100, color='green', linestyle='--',
                    label=f'Secure baseline ({baseline_secure*100:.1f}%)')
        ax1.axhline(y=baseline_neutral * 100, color='red', linestyle='--',
                    label=f'Neutral baseline ({baseline_neutral*100:.1f}%)')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('P(snprintf) %')
        ax1.set_title('Layer Sweep: Single-layer Steering (alpha=1)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Best layers
        ax2 = axes[0, 1]
        sorted_results = sorted(layer_results, key=lambda x: x['snprintf_prob'], reverse=True)
        top_10 = sorted_results[:10]
        layers_top = [r['target_layer'] for r in top_10]
        probs_top = [r['snprintf_prob'] * 100 for r in top_10]

        ax2.barh(range(len(layers_top)), probs_top, color='green', alpha=0.7)
        ax2.set_yticks(range(len(layers_top)))
        ax2.set_yticklabels([f'Layer {l}' for l in layers_top])
        ax2.set_xlabel('P(snprintf) %')
        ax2.set_title('Top 10 Layers for Steering')
        ax2.invert_yaxis()
        ax2.axvline(x=baseline_neutral * 100, color='red', linestyle='--', label='Neutral')
        ax2.axvline(x=baseline_secure * 100, color='green', linestyle='--', label='Secure')
        ax2.legend()

        # Plot 3: Alpha sweep for best layer
        ax3 = axes[1, 0]
        if alpha_results:
            best_layer = alpha_results['layer']
            alphas = [r['alpha'] for r in alpha_results['results']]
            alpha_probs = [r['snprintf_prob'] * 100 for r in alpha_results['results']]

            ax3.plot(alphas, alpha_probs, 'b-o', markersize=8)
            ax3.axhline(y=baseline_secure * 100, color='green', linestyle='--', label='Secure')
            ax3.axhline(y=baseline_neutral * 100, color='red', linestyle='--', label='Neutral')
            ax3.set_xlabel('Alpha (steering strength)')
            ax3.set_ylabel('P(snprintf) %')
            ax3.set_title(f'Alpha Sweep at Layer {best_layer}')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # Plot 4: Multi-layer steering comparison
        ax4 = axes[1, 1]
        categories = ['Neutral\nBaseline', 'Single Best\nLayer', 'Late Layers\n(16-31)',
                      'All Layers\n(0-31)', 'Secure\nBaseline']
        values = [
            baseline_neutral * 100,
            sorted_results[0]['snprintf_prob'] * 100 if sorted_results else 0,
            0,  # Will be filled
            0,  # Will be filled
            baseline_secure * 100
        ]

        colors = ['red', 'orange', 'blue', 'purple', 'green']
        ax4.bar(categories, values, color=colors, alpha=0.7)
        ax4.set_ylabel('P(snprintf) %')
        ax4.set_title('Steering Comparison')
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to: {output_path}")


def main():
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    steering = ActivationSteering()

    print("\n" + "="*70)
    print("ACTIVATION STEERING ANALYSIS")
    print("="*70)

    # Collect baseline activations
    print("\nCollecting secure prompt activations...")
    secure_acts = steering.collect_activations(SECURE_PROMPT)
    baseline_secure = secure_acts['snprintf_prob']

    print("Collecting neutral prompt activations...")
    neutral_acts = steering.collect_activations(NEUTRAL_PROMPT)
    baseline_neutral = neutral_acts['snprintf_prob']

    print(f"\nBaseline P(snprintf):")
    print(f"  Secure:  {baseline_secure*100:.2f}%")
    print(f"  Neutral: {baseline_neutral*100:.2f}%")
    print(f"  Gap:     {(baseline_secure - baseline_neutral)*100:.2f}%")

    # Compute steering vectors
    print("\nComputing steering vectors...")
    steering_vectors = steering.compute_steering_vector(secure_acts, neutral_acts)

    # Compute steering vector norms
    print("\nSteering vector norms by layer:")
    norms = []
    for layer_idx in range(steering.n_layers):
        norm = torch.norm(steering_vectors[layer_idx]).item()
        norms.append(norm)
        if layer_idx % 8 == 0:
            print(f"  Layer {layer_idx}: {norm:.2f}")

    # Layer sweep with alpha=1
    print("\n" + "="*70)
    print("LAYER SWEEP (alpha=1)")
    print("="*70)

    layer_results = steering.layer_sweep(NEUTRAL_PROMPT, steering_vectors, alpha=1.0)

    print("\n| Layer | P(snprintf) | Lift |")
    print("|-------|------------|------|")
    for r in layer_results:
        lift = r['snprintf_prob'] - baseline_neutral
        lift_pct = (lift / (baseline_secure - baseline_neutral)) * 100 if baseline_secure != baseline_neutral else 0
        print(f"| {r['target_layer']:5d} | {r['snprintf_prob']*100:10.2f}% | {lift_pct:5.1f}% |")

    # Find best single layer
    best_layer_result = max(layer_results, key=lambda x: x['snprintf_prob'])
    best_layer = best_layer_result['target_layer']
    print(f"\nBest single layer: L{best_layer} with P(snprintf)={best_layer_result['snprintf_prob']*100:.2f}%")

    # Alpha sweep at best layer
    print("\n" + "="*70)
    print(f"ALPHA SWEEP (Layer {best_layer})")
    print("="*70)

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
    alpha_results = steering.alpha_sweep(NEUTRAL_PROMPT, steering_vectors, best_layer, alphas)

    print("\n| Alpha | P(snprintf) | Lift |")
    print("|-------|------------|------|")
    for r in alpha_results:
        lift_pct = ((r['snprintf_prob'] - baseline_neutral) / (baseline_secure - baseline_neutral)) * 100
        print(f"| {r['alpha']:5.2f} | {r['snprintf_prob']*100:10.2f}% | {lift_pct:5.1f}% |")

    # Multi-layer steering
    print("\n" + "="*70)
    print("MULTI-LAYER STEERING")
    print("="*70)

    # Test different layer combinations
    layer_combos = [
        ('Late (16-31)', list(range(16, 32))),
        ('Mid (8-23)', list(range(8, 24))),
        ('Early (0-15)', list(range(16))),
        ('All (0-31)', list(range(32))),
        ('Top 5 layers', sorted([r['target_layer'] for r in sorted(layer_results, key=lambda x: x['snprintf_prob'], reverse=True)[:5]])),
    ]

    multi_results = {}
    print("\n| Layers | P(snprintf) | Lift |")
    print("|--------|------------|------|")

    for name, layers in layer_combos:
        result = steering.steer_at_layers(NEUTRAL_PROMPT, steering_vectors, layers, alpha=1.0)
        lift_pct = ((result['snprintf_prob'] - baseline_neutral) / (baseline_secure - baseline_neutral)) * 100
        print(f"| {name:14s} | {result['snprintf_prob']*100:10.2f}% | {lift_pct:5.1f}% |")
        multi_results[name] = result

    # Visualize
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_path = results_dir / f"activation_steering_{timestamp}.png"
    steering.visualize(
        layer_results,
        {'layer': best_layer, 'results': alpha_results},
        viz_path,
        baseline_secure,
        baseline_neutral
    )

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\nBaseline gap: {(baseline_secure - baseline_neutral)*100:.2f}%")
    print(f"Best single layer (L{best_layer}): {best_layer_result['snprintf_prob']*100:.2f}%")
    print(f"  -> Lift: {((best_layer_result['snprintf_prob'] - baseline_neutral)/(baseline_secure - baseline_neutral))*100:.1f}%")

    all_layers_result = multi_results['All (0-31)']
    print(f"All layers steering: {all_layers_result['snprintf_prob']*100:.2f}%")
    print(f"  -> Lift: {((all_layers_result['snprintf_prob'] - baseline_neutral)/(baseline_secure - baseline_neutral))*100:.1f}%")

    # Save results
    full_results = {
        'timestamp': timestamp,
        'baseline': {
            'secure_snprintf_prob': baseline_secure,
            'neutral_snprintf_prob': baseline_neutral,
            'gap': baseline_secure - baseline_neutral
        },
        'steering_vector_norms': norms,
        'layer_sweep': [
            {
                'layer': r['target_layer'],
                'snprintf_prob': r['snprintf_prob'],
                'sprintf_prob': r['sprintf_prob']
            }
            for r in layer_results
        ],
        'alpha_sweep': {
            'layer': best_layer,
            'results': alpha_results
        },
        'multi_layer': {
            name: {
                'layers': layers,
                'snprintf_prob': multi_results[name]['snprintf_prob'],
                'sprintf_prob': multi_results[name]['sprintf_prob']
            }
            for name, layers in layer_combos
        }
    }

    with open(results_dir / f"activation_steering_{timestamp}.json", 'w') as f:
        json.dump(full_results, f, indent=2)

    print(f"\nResults saved to: {results_dir}")

    return full_results


if __name__ == "__main__":
    results = main()
