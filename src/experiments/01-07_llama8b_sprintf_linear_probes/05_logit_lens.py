#!/usr/bin/env python3
"""
Logit Lens Analysis

Project intermediate representations at each layer to vocabulary space.
Track when "snprintf" vs "sprintf" probability emerges through the layers.

This reveals the representation→computation trajectory:
- When does the model "know" it will output snprintf?
- How does this differ between secure and neutral contexts?
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


class LogitLens:
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

        # Get the unembedding matrix and layer norm
        self.lm_head = self.model.lm_head
        self.final_norm = self.model.model.norm

        # Token IDs
        self.snprintf_token = self.tokenizer.encode(" snprintf", add_special_tokens=False)[0]
        self.sprintf_token = self.tokenizer.encode(" sprintf", add_special_tokens=False)[0]

        print(f"Model loaded: {self.n_layers} layers")
        print(f"snprintf token: {self.snprintf_token}, sprintf token: {self.sprintf_token}")

        self.residual_stream = {}
        self.hooks = []

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.residual_stream = {}

    def save_residual_hook(self, layer_idx: int):
        """Save the residual stream after each layer."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            # Save last token position
            self.residual_stream[layer_idx] = h[:, -1, :].detach().clone()
            return output
        return hook_fn

    def get_logit_lens(self, prompt: str) -> dict:
        """
        Apply logit lens at each layer.
        Project residual stream through final LayerNorm and unembedding.
        """
        self.clear_hooks()

        # Register hooks on all layers
        for layer_idx in range(self.n_layers):
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(self.save_residual_hook(layer_idx))
            self.hooks.append(hook)

        # Forward pass
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get final logits for comparison
        final_logits = outputs.logits[0, -1, :]
        final_probs = torch.softmax(final_logits, dim=-1)

        results = {
            'layers': {},
            'final': {
                'snprintf_prob': final_probs[self.snprintf_token].item(),
                'sprintf_prob': final_probs[self.sprintf_token].item(),
                'snprintf_rank': (final_probs > final_probs[self.snprintf_token]).sum().item(),
                'sprintf_rank': (final_probs > final_probs[self.sprintf_token]).sum().item(),
            }
        }

        # Apply logit lens at each layer
        for layer_idx in range(self.n_layers):
            residual = self.residual_stream[layer_idx]

            # Apply final layer norm
            normed = self.final_norm(residual)

            # Project to vocabulary
            logits = self.lm_head(normed).squeeze(0)
            probs = torch.softmax(logits, dim=-1)

            # Get snprintf and sprintf probabilities
            snprintf_prob = probs[self.snprintf_token].item()
            sprintf_prob = probs[self.sprintf_token].item()

            # Get ranks
            snprintf_rank = (probs > probs[self.snprintf_token]).sum().item()
            sprintf_rank = (probs > probs[self.sprintf_token]).sum().item()

            # Get top 5 tokens
            top_probs, top_ids = torch.topk(probs, 5)
            top_tokens = [self.tokenizer.decode([t.item()]) for t in top_ids]

            results['layers'][layer_idx] = {
                'snprintf_prob': snprintf_prob,
                'sprintf_prob': sprintf_prob,
                'snprintf_rank': snprintf_rank,
                'sprintf_rank': sprintf_rank,
                'snprintf_logit': logits[self.snprintf_token].item(),
                'sprintf_logit': logits[self.sprintf_token].item(),
                'top_5_tokens': top_tokens,
                'top_5_probs': [p.item() for p in top_probs]
            }

        self.clear_hooks()
        return results

    def analyze_trajectory(self, secure_results: dict, neutral_results: dict) -> dict:
        """Analyze the trajectory of snprintf probability through layers."""
        analysis = {
            'secure': {'snprintf_probs': [], 'sprintf_probs': [], 'snprintf_ranks': []},
            'neutral': {'snprintf_probs': [], 'sprintf_probs': [], 'snprintf_ranks': []}
        }

        for layer_idx in range(self.n_layers):
            analysis['secure']['snprintf_probs'].append(
                secure_results['layers'][layer_idx]['snprintf_prob']
            )
            analysis['secure']['sprintf_probs'].append(
                secure_results['layers'][layer_idx]['sprintf_prob']
            )
            analysis['secure']['snprintf_ranks'].append(
                secure_results['layers'][layer_idx]['snprintf_rank']
            )

            analysis['neutral']['snprintf_probs'].append(
                neutral_results['layers'][layer_idx]['snprintf_prob']
            )
            analysis['neutral']['sprintf_probs'].append(
                neutral_results['layers'][layer_idx]['sprintf_prob']
            )
            analysis['neutral']['snprintf_ranks'].append(
                neutral_results['layers'][layer_idx]['snprintf_rank']
            )

        # Find divergence point
        secure_probs = np.array(analysis['secure']['snprintf_probs'])
        neutral_probs = np.array(analysis['neutral']['snprintf_probs'])
        diff = secure_probs - neutral_probs

        # Find when difference becomes significant (>1%)
        divergence_layer = None
        for i, d in enumerate(diff):
            if d > 0.01:
                divergence_layer = i
                break

        analysis['divergence_layer'] = divergence_layer
        analysis['max_diff'] = float(diff.max())
        analysis['max_diff_layer'] = int(diff.argmax())

        return analysis

    def visualize(self, secure_results: dict, neutral_results: dict,
                  analysis: dict, output_path: Path):
        """Create visualization of logit lens results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        layers = list(range(self.n_layers))

        # Plot 1: snprintf probability trajectory
        ax1 = axes[0, 0]
        ax1.plot(layers, analysis['secure']['snprintf_probs'], 'g-o',
                 label='Secure context', markersize=4)
        ax1.plot(layers, analysis['neutral']['snprintf_probs'], 'r-o',
                 label='Neutral context', markersize=4)
        if analysis['divergence_layer'] is not None:
            ax1.axvline(x=analysis['divergence_layer'], color='purple',
                        linestyle='--', label=f'Divergence (L{analysis["divergence_layer"]})')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('P(snprintf)')
        ax1.set_title('Logit Lens: snprintf Probability')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # Plot 2: sprintf probability trajectory
        ax2 = axes[0, 1]
        ax2.plot(layers, analysis['secure']['sprintf_probs'], 'g-o',
                 label='Secure context', markersize=4)
        ax2.plot(layers, analysis['neutral']['sprintf_probs'], 'r-o',
                 label='Neutral context', markersize=4)
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('P(sprintf)')
        ax2.set_title('Logit Lens: sprintf Probability')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

        # Plot 3: Probability difference (secure - neutral)
        ax3 = axes[1, 0]
        diff = np.array(analysis['secure']['snprintf_probs']) - \
               np.array(analysis['neutral']['snprintf_probs'])
        ax3.bar(layers, diff, color='blue', alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-')
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('P(snprintf) difference')
        ax3.set_title('Secure - Neutral Difference')
        ax3.grid(True, alpha=0.3)

        # Plot 4: snprintf rank trajectory
        ax4 = axes[1, 1]
        ax4.plot(layers, analysis['secure']['snprintf_ranks'], 'g-o',
                 label='Secure context', markersize=4)
        ax4.plot(layers, analysis['neutral']['snprintf_ranks'], 'r-o',
                 label='Neutral context', markersize=4)
        ax4.set_xlabel('Layer')
        ax4.set_ylabel('Rank of snprintf')
        ax4.set_title('snprintf Token Rank (lower = better)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.invert_yaxis()  # Lower rank is better

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Visualization saved to: {output_path}")


def main():
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    lens = LogitLens()

    print("\n" + "="*70)
    print("LOGIT LENS ANALYSIS")
    print("="*70)

    # Analyze secure prompt
    print("\nAnalyzing secure prompt...")
    secure_results = lens.get_logit_lens(SECURE_PROMPT)

    print("\nAnalyzing neutral prompt...")
    neutral_results = lens.get_logit_lens(NEUTRAL_PROMPT)

    # Compare trajectories
    print("\n" + "="*70)
    print("TRAJECTORY ANALYSIS")
    print("="*70)

    analysis = lens.analyze_trajectory(secure_results, neutral_results)

    print(f"\nDivergence layer (>1% diff): {analysis['divergence_layer']}")
    print(f"Max difference: {analysis['max_diff']*100:.2f}% at layer {analysis['max_diff_layer']}")

    # Print layer-by-layer
    print("\n| Layer | Secure P(snprintf) | Neutral P(snprintf) | Difference |")
    print("|-------|-------------------|--------------------| -----------|")

    for i in range(0, lens.n_layers, 4):  # Print every 4th layer
        s_prob = analysis['secure']['snprintf_probs'][i]
        n_prob = analysis['neutral']['snprintf_probs'][i]
        diff = s_prob - n_prob
        print(f"| {i:5d} | {s_prob*100:17.4f}% | {n_prob*100:18.4f}% | {diff*100:+10.4f}% |")

    # Final layer
    print(f"| {lens.n_layers-1:5d} | {analysis['secure']['snprintf_probs'][-1]*100:17.4f}% | "
          f"{analysis['neutral']['snprintf_probs'][-1]*100:18.4f}% | "
          f"{(analysis['secure']['snprintf_probs'][-1] - analysis['neutral']['snprintf_probs'][-1])*100:+10.4f}% |")

    # Print final output comparison
    print("\n" + "="*70)
    print("FINAL OUTPUT COMPARISON")
    print("="*70)

    print("\nSecure prompt final output:")
    print(f"  P(snprintf) = {secure_results['final']['snprintf_prob']*100:.2f}%, rank = {secure_results['final']['snprintf_rank']}")
    print(f"  P(sprintf) = {secure_results['final']['sprintf_prob']*100:.2f}%, rank = {secure_results['final']['sprintf_rank']}")

    print("\nNeutral prompt final output:")
    print(f"  P(snprintf) = {neutral_results['final']['snprintf_prob']*100:.2f}%, rank = {neutral_results['final']['snprintf_rank']}")
    print(f"  P(sprintf) = {neutral_results['final']['sprintf_prob']*100:.2f}%, rank = {neutral_results['final']['sprintf_rank']}")

    # Visualize
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_path = results_dir / f"logit_lens_{timestamp}.png"
    lens.visualize(secure_results, neutral_results, analysis, viz_path)

    # Save results
    full_results = {
        'timestamp': timestamp,
        'secure_prompt': SECURE_PROMPT[:100] + '...',
        'neutral_prompt': NEUTRAL_PROMPT[:100] + '...',
        'secure_results': secure_results,
        'neutral_results': neutral_results,
        'analysis': {
            'divergence_layer': analysis['divergence_layer'],
            'max_diff': analysis['max_diff'],
            'max_diff_layer': analysis['max_diff_layer'],
            'secure_snprintf_probs': analysis['secure']['snprintf_probs'],
            'neutral_snprintf_probs': analysis['neutral']['snprintf_probs']
        }
    }

    with open(results_dir / f"logit_lens_{timestamp}.json", 'w') as f:
        json.dump(full_results, f, indent=2, default=float)

    print(f"\n💾 Results saved to: {results_dir}")

    return analysis


if __name__ == "__main__":
    analysis = main()
