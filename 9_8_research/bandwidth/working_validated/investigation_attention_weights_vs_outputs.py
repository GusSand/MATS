#!/usr/bin/env python3
"""
Investigation: Attention Weights vs Attention Outputs
====================================================

This script investigates the difference between:
1. Attention weights (what we measured in Exp 1.1) - WHO pays attention to WHOM
2. Attention outputs (what successful experiments patch) - the PROCESSED information

The successful head patching experiments work on attention outputs, not weights.
This could explain why our bandwidth analysis contradicted the even/odd head findings.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
from datetime import datetime
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Model configuration
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LAYER_OF_INTEREST = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TokenCategory:
    """Categorize tokens for bandwidth analysis"""
    format_tokens: List[str] = None
    numerical_tokens: List[str] = None

    def __post_init__(self):
        if self.format_tokens is None:
            self.format_tokens = ['Q', ':', 'A', 'Which', 'is', 'bigger', '?',
                                 '<|start_header_id|>', '<|end_header_id|>',
                                 'user', 'assistant']
        if self.numerical_tokens is None:
            self.numerical_tokens = ['9', '.', '8', '11', '0', '1', '2', '3',
                                    '4', '5', '6', '7']


class AttentionComparator:
    """Compare attention weights vs attention outputs for bandwidth analysis"""

    def __init__(self, model_name: str = MODEL_NAME, device: str = DEVICE):
        self.device = device
        self.model_name = model_name

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager"
        )
        self.model.eval()

        self.n_heads = 32
        self.layer_idx = LAYER_OF_INTEREST
        self.saved_outputs = {}
        self.hooks = []

    def save_attention_output_hook(self, key: str):
        """Hook to save attention module OUTPUT (what gets patched in successful experiments)"""
        def hook_fn(module, input, output):
            # Save the attention output (hidden states after attention processing)
            if isinstance(output, tuple):
                hidden_states = output[0]  # Main output
            else:
                hidden_states = output
            self.saved_outputs[key] = hidden_states.detach().cpu()
        return hook_fn

    def categorize_tokens(self, tokens: List[str], token_cat: TokenCategory) -> Dict[str, List[int]]:
        """Categorize token positions by type"""
        positions = {
            'format': [],
            'numerical': [],
            'other': []
        }

        for i, token in enumerate(tokens):
            token_str = token.replace('▁', '').strip()
            if any(fmt in token_str for fmt in token_cat.format_tokens):
                positions['format'].append(i)
            elif any(num in token_str for num in token_cat.numerical_tokens):
                positions['numerical'].append(i)
            else:
                positions['other'].append(i)

        return positions

    def analyze_attention_weights_bandwidth(self, prompt: str) -> Dict:
        """Analyze bandwidth using attention WEIGHTS (Experiment 1.1 method)"""
        token_cat = TokenCategory()

        # Get attention weights
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        token_positions = self.categorize_tokens(tokens, token_cat)

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True, return_dict=True)

        attention_weights = outputs.attentions[self.layer_idx]  # [batch, heads, seq_len, seq_len]
        last_token_idx = attention_weights.shape[2] - 1

        results = {'even': [], 'odd': []}

        for head_idx in range(self.n_heads):
            head_attn = attention_weights[0, head_idx].cpu().numpy()

            # Calculate bandwidth from last token to numerical tokens
            numerical_attn = sum(head_attn[last_token_idx, pos] for pos in token_positions['numerical']
                               if pos < attention_weights.shape[3])
            total_attn = head_attn[last_token_idx].sum()
            numerical_bandwidth = numerical_attn / total_attn if total_attn > 0 else 0

            head_type = 'even' if head_idx % 2 == 0 else 'odd'
            results[head_type].append(numerical_bandwidth)

        return {
            'method': 'attention_weights',
            'even_mean': np.mean(results['even']),
            'odd_mean': np.mean(results['odd']),
            'even_std': np.std(results['even']),
            'odd_std': np.std(results['odd']),
            'even_values': results['even'],
            'odd_values': results['odd'],
            'tokens': tokens,
            'token_positions': token_positions
        }

    def analyze_attention_outputs_bandwidth(self, prompt: str) -> Dict:
        """Analyze bandwidth using attention OUTPUTS (successful experiment method)"""
        token_cat = TokenCategory()

        # Set up hook to capture attention outputs
        attention_module = self.model.model.layers[self.layer_idx].self_attn
        hook = attention_module.register_forward_hook(self.save_attention_output_hook('attn_output'))
        self.hooks.append(hook)

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
            token_positions = self.categorize_tokens(tokens, token_cat)

            with torch.no_grad():
                _ = self.model(**inputs)

            # Analyze the attention output
            attn_output = self.saved_outputs['attn_output']  # [batch, seq_len, hidden_dim]
            seq_len, hidden_dim = attn_output.shape[1], attn_output.shape[2]
            head_dim = hidden_dim // self.n_heads

            # Reshape to separate heads: [seq_len, n_heads, head_dim]
            attn_reshaped = attn_output[0].view(seq_len, self.n_heads, head_dim)

            results = {'even': [], 'odd': []}

            # For each head, calculate how much "numerical information" it carries
            for head_idx in range(self.n_heads):
                head_output = attn_reshaped[:, head_idx, :].numpy()  # [seq_len, head_dim]

                # Calculate the magnitude of activations at numerical token positions
                numerical_activation = 0
                total_activation = 0

                for pos in range(seq_len):
                    activation_magnitude = np.linalg.norm(head_output[pos])
                    total_activation += activation_magnitude

                    if pos in token_positions['numerical']:
                        numerical_activation += activation_magnitude

                numerical_bandwidth = numerical_activation / total_activation if total_activation > 0 else 0

                head_type = 'even' if head_idx % 2 == 0 else 'odd'
                results[head_type].append(numerical_bandwidth)

        finally:
            # Clean up hooks
            for hook in self.hooks:
                hook.remove()
            self.hooks.clear()

        return {
            'method': 'attention_outputs',
            'even_mean': np.mean(results['even']),
            'odd_mean': np.mean(results['odd']),
            'even_std': np.std(results['even']),
            'odd_std': np.std(results['odd']),
            'even_values': results['even'],
            'odd_values': results['odd'],
            'tokens': tokens,
            'token_positions': token_positions
        }

    def compare_methods(self, prompts: Dict[str, str]) -> pd.DataFrame:
        """Compare both methods across different prompt formats"""
        results = []

        for format_name, prompt in prompts.items():
            print(f"\nAnalyzing format: {format_name}")
            print(f"Prompt: {prompt}")

            # Method 1: Attention weights (Experiment 1.1)
            weights_result = self.analyze_attention_weights_bandwidth(prompt)
            results.append({
                'format': format_name,
                'method': 'attention_weights',
                'even_mean': weights_result['even_mean'],
                'odd_mean': weights_result['odd_mean'],
                'even_std': weights_result['even_std'],
                'odd_std': weights_result['odd_std'],
                'difference': weights_result['odd_mean'] - weights_result['even_mean']
            })

            # Method 2: Attention outputs (successful experiments)
            outputs_result = self.analyze_attention_outputs_bandwidth(prompt)
            results.append({
                'format': format_name,
                'method': 'attention_outputs',
                'even_mean': outputs_result['even_mean'],
                'odd_mean': outputs_result['odd_mean'],
                'even_std': outputs_result['even_std'],
                'odd_std': outputs_result['odd_std'],
                'difference': outputs_result['odd_mean'] - outputs_result['even_mean']
            })

            print(f"  Weights - Even: {weights_result['even_mean']:.4f}, Odd: {weights_result['odd_mean']:.4f}")
            print(f"  Outputs - Even: {outputs_result['even_mean']:.4f}, Odd: {outputs_result['odd_mean']:.4f}")

        return pd.DataFrame(results)

    def visualize_comparison(self, df: pd.DataFrame):
        """Create visualization comparing both methods"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Mean bandwidth by method and head type
        ax1 = axes[0, 0]
        methods = df['method'].unique()
        formats = df['format'].unique()

        x = np.arange(len(formats))
        width = 0.15

        for i, method in enumerate(methods):
            method_data = df[df['method'] == method]
            even_means = method_data['even_mean'].values
            odd_means = method_data['odd_mean'].values

            ax1.bar(x + i*width*2, even_means, width, label=f'{method} - Even', alpha=0.8)
            ax1.bar(x + i*width*2 + width, odd_means, width, label=f'{method} - Odd', alpha=0.8)

        ax1.set_xlabel('Format')
        ax1.set_ylabel('Numerical Bandwidth')
        ax1.set_title('Bandwidth Comparison: Weights vs Outputs')
        ax1.set_xticks(x + width*1.5)
        ax1.set_xticklabels(formats)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Difference (Odd - Even) by method
        ax2 = axes[0, 1]
        for i, method in enumerate(methods):
            method_data = df[df['method'] == method]
            differences = method_data['difference'].values
            ax2.bar(x + i*width, differences, width, label=method, alpha=0.8)

        ax2.set_xlabel('Format')
        ax2.set_ylabel('Bandwidth Difference (Odd - Even)')
        ax2.set_title('Head Type Preference by Method')
        ax2.set_xticks(x + width/2)
        ax2.set_xticklabels(formats)
        ax2.legend()
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Method comparison for Q&A format
        ax3 = axes[1, 0]
        qa_data = df[df['format'] == 'Q&A']
        methods_qa = qa_data['method'].values
        even_qa = qa_data['even_mean'].values
        odd_qa = qa_data['odd_mean'].values

        x_qa = np.arange(len(methods_qa))
        ax3.bar(x_qa - 0.2, even_qa, 0.4, label='Even Heads', alpha=0.8)
        ax3.bar(x_qa + 0.2, odd_qa, 0.4, label='Odd Heads', alpha=0.8)
        ax3.set_xlabel('Method')
        ax3.set_ylabel('Numerical Bandwidth')
        ax3.set_title('Q&A Format: Method Comparison')
        ax3.set_xticks(x_qa)
        ax3.set_xticklabels(methods_qa, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')

        # Create summary text
        summary_text = "INVESTIGATION SUMMARY\n" + "="*30 + "\n\n"

        for method in methods:
            method_data = df[df['method'] == method]
            avg_even = method_data['even_mean'].mean()
            avg_odd = method_data['odd_mean'].mean()
            avg_diff = method_data['difference'].mean()

            summary_text += f"{method.upper()}:\n"
            summary_text += f"  Avg Even: {avg_even:.4f}\n"
            summary_text += f"  Avg Odd:  {avg_odd:.4f}\n"
            summary_text += f"  Avg Diff: {avg_diff:+.4f}\n\n"

        # Determine which method aligns with theory
        weights_supports_theory = df[df['method'] == 'attention_weights']['difference'].mean() < 0
        outputs_supports_theory = df[df['method'] == 'attention_outputs']['difference'].mean() < 0

        summary_text += "THEORY ALIGNMENT:\n"
        summary_text += f"Weights support theory: {'✅' if weights_supports_theory else '❌'}\n"
        summary_text += f"Outputs support theory: {'✅' if outputs_supports_theory else '❌'}\n"

        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        plt.savefig('/home/paperspace/dev/MATS9/bandwidth/figures/weights_vs_outputs_comparison.png',
                   dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Run the investigation comparing attention weights vs outputs"""
    print("="*70)
    print("INVESTIGATION: Attention Weights vs Attention Outputs")
    print("="*70)
    print("Comparing two measurement approaches:")
    print("1. Attention weights (Experiment 1.1 method)")
    print("2. Attention outputs (successful patching experiments method)")
    print()

    # Test prompts
    prompts = {
        'Simple': '9.8 or 9.11? Answer:',
        'Q&A': 'Q: Which is bigger: 9.8 or 9.11? A:',
        'Chat': '<|start_header_id|>user<|end_header_id|>\n\nWhich is bigger: 9.8 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    }

    # Initialize comparator
    comparator = AttentionComparator()

    # Run comparison
    print("Running comparison analysis...")
    df_comparison = comparator.compare_methods(prompts)

    # Create visualization
    print("\nCreating comparison visualization...")
    comparator.visualize_comparison(df_comparison)

    # Print detailed results
    print("\n" + "="*70)
    print("DETAILED COMPARISON RESULTS")
    print("="*70)
    print(df_comparison.to_string(index=False))

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'/home/paperspace/dev/MATS9/bandwidth/investigation_weights_vs_outputs_{timestamp}.json'

    results_dict = {
        'comparison_data': df_comparison.to_dict('records'),
        'summary': {
            'weights_method': {
                'avg_even': df_comparison[df_comparison['method'] == 'attention_weights']['even_mean'].mean(),
                'avg_odd': df_comparison[df_comparison['method'] == 'attention_weights']['odd_mean'].mean(),
                'supports_theory': df_comparison[df_comparison['method'] == 'attention_weights']['difference'].mean() < 0
            },
            'outputs_method': {
                'avg_even': df_comparison[df_comparison['method'] == 'attention_outputs']['even_mean'].mean(),
                'avg_odd': df_comparison[df_comparison['method'] == 'attention_outputs']['odd_mean'].mean(),
                'supports_theory': df_comparison[df_comparison['method'] == 'attention_outputs']['difference'].mean() < 0
            }
        }
    }

    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)

    print(f"\n✅ Investigation complete!")
    print(f"✅ Results saved to {results_file}")
    print(f"✅ Visualization saved to /home/paperspace/dev/MATS9/bandwidth/figures/weights_vs_outputs_comparison.png")

    # Final analysis
    print(f"\n" + "="*70)
    print("INVESTIGATION CONCLUSION")
    print("="*70)

    weights_diff = df_comparison[df_comparison['method'] == 'attention_weights']['difference'].mean()
    outputs_diff = df_comparison[df_comparison['method'] == 'attention_outputs']['difference'].mean()

    print(f"Average difference (Odd - Even):")
    print(f"  Attention weights: {weights_diff:+.4f}")
    print(f"  Attention outputs: {outputs_diff:+.4f}")
    print()

    if abs(weights_diff) < abs(outputs_diff):
        print("🔍 Attention weights show smaller head-type differences")
    else:
        print("🔍 Attention outputs show larger head-type differences")

    if weights_diff > 0 and outputs_diff < 0:
        print("🚨 CRITICAL: Methods show OPPOSITE patterns!")
        print("   This could explain the contradiction with successful patching experiments.")
    elif weights_diff < 0 and outputs_diff > 0:
        print("🚨 CRITICAL: Methods show OPPOSITE patterns!")
        print("   Attention outputs align with theory while weights don't.")
    else:
        print("📊 Both methods show similar directional patterns")

    return df_comparison


if __name__ == "__main__":
    main()