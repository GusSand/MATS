#!/usr/bin/env python3
"""
Bandwidth Competition Theory - Experiment 1.1: Attention Bandwidth Distribution
===============================================================================

This experiment measures how attention is distributed across token categories
(format tokens vs numerical tokens) in different prompt formats.

Expected result:
- Even heads maintain >40% attention on numerical tokens
- Odd heads drop below threshold in Q&A format

Adapted from bandwidth_competition_experiments.md to use transformers hooks
instead of nnsight, following patterns from working_scripts/
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional
from dataclasses import dataclass
from contextlib import contextmanager
import json
from datetime import datetime
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Visualization settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Model configuration
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LAYER_OF_INTEREST = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TokenCategory:
    """Categorize tokens for bandwidth analysis"""
    format_tokens: List[str] = None
    numerical_tokens: List[str] = None
    other_tokens: List[str] = None

    def __post_init__(self):
        if self.format_tokens is None:
            self.format_tokens = ['Q', ':', 'A', 'Which', 'is', 'bigger', '?',
                                 '<|start_header_id|>', '<|end_header_id|>',
                                 'user', 'assistant']
        if self.numerical_tokens is None:
            self.numerical_tokens = ['9', '.', '8', '11', '0', '1', '2', '3',
                                    '4', '5', '6', '7']


class BandwidthAnalyzer:
    """Analyze attention bandwidth distribution using transformers hooks"""

    def __init__(self, model_name: str = MODEL_NAME, device: str = DEVICE):
        self.device = device
        self.model_name = model_name

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        self.model.eval()

        self.n_heads = 32
        self.layer_idx = LAYER_OF_INTEREST
        self.saved_attention = {}
        self.hooks = []

    def save_attention_hook(self, key: str):
        """Create hook to save attention weights"""
        def hook_fn(module, input, output):
            # Attention module output: (hidden_states, attention_weights, past_key_values)
            if isinstance(output, tuple) and len(output) >= 2:
                attention_weights = output[1]  # [batch, n_heads, seq_len, seq_len]
                if attention_weights is not None:
                    self.saved_attention[key] = attention_weights.detach().cpu()
        return hook_fn

    @contextmanager
    def capture_attention_context(self, prompt: str, key: str):
        """Context manager to capture attention weights"""
        try:
            # Get attention module for layer of interest
            attention_module = self.model.model.layers[self.layer_idx].self_attn

            # Register hook
            hook = attention_module.register_forward_hook(self.save_attention_hook(key))
            self.hooks.append(hook)

            # Forward pass
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                _ = self.model(**inputs, output_attentions=True)

            yield self.saved_attention

        finally:
            # Clean up hooks
            for hook in self.hooks:
                hook.remove()
            self.hooks.clear()

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

    def calculate_bandwidth(self, attention_weights: torch.Tensor, token_positions: Dict[str, List[int]]) -> Dict[str, float]:
        """Calculate attention bandwidth for each token category"""
        # attention_weights: [batch, n_heads, seq_len, seq_len]
        batch_size, n_heads, seq_len, _ = attention_weights.shape

        bandwidths = {'format': [], 'numerical': [], 'other': []}

        for head_idx in range(n_heads):
            head_attn = attention_weights[0, head_idx].numpy()  # [seq_len, seq_len]

            # Calculate attention FROM the last token (where prediction happens) TO each token category
            # This is more meaningful for understanding what the model "pays attention to"
            last_token_idx = seq_len - 1

            format_attn = 0
            numerical_attn = 0
            other_attn = 0

            for pos in token_positions['format']:
                if pos < seq_len:
                    format_attn += head_attn[last_token_idx, pos]
            for pos in token_positions['numerical']:
                if pos < seq_len:
                    numerical_attn += head_attn[last_token_idx, pos]
            for pos in token_positions['other']:
                if pos < seq_len:
                    other_attn += head_attn[last_token_idx, pos]

            # Normalize (should sum to 1 for the last token's attention)
            total = format_attn + numerical_attn + other_attn
            if total > 0:
                bandwidths['format'].append(format_attn / total)
                bandwidths['numerical'].append(numerical_attn / total)
                bandwidths['other'].append(other_attn / total)
            else:
                bandwidths['format'].append(0)
                bandwidths['numerical'].append(0)
                bandwidths['other'].append(0)

        return bandwidths

    def analyze_attention_bandwidth(self, prompts: Dict[str, str]) -> pd.DataFrame:
        """
        Measure how attention is distributed across token categories.

        Expected result:
        - Even heads maintain >40% attention on numerical tokens
        - Odd heads drop below threshold in Q&A format
        """
        results = []
        token_cat = TokenCategory()

        for format_name, prompt in prompts.items():
            print(f"Analyzing format: {format_name}")

            with self.capture_attention_context(prompt, f"attn_{format_name}") as saved:
                # Get tokens and categorize them
                inputs = self.tokenizer(prompt, return_tensors="pt")
                tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
                token_positions = self.categorize_tokens(tokens, token_cat)

                print(f"  Tokens: {tokens}")
                print(f"  Format positions: {token_positions['format']}")
                print(f"  Numerical positions: {token_positions['numerical']}")

                # Get attention weights
                if f"attn_{format_name}" in saved:
                    attention_weights = saved[f"attn_{format_name}"]
                    bandwidths = self.calculate_bandwidth(attention_weights, token_positions)

                    # Store results for each head
                    for head_idx in range(self.n_heads):
                        results.append({
                            'format': format_name,
                            'head_idx': head_idx,
                            'head_type': 'even' if head_idx % 2 == 0 else 'odd',
                            'format_bandwidth': bandwidths['format'][head_idx],
                            'numerical_bandwidth': bandwidths['numerical'][head_idx],
                            'other_bandwidth': bandwidths['other'][head_idx]
                        })

        return pd.DataFrame(results)

    def visualize_bandwidth_results(self, df: pd.DataFrame, prompts: Dict[str, str]):
        """Create visualization of bandwidth distribution"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Bandwidth distribution by head type
        for idx, head_type in enumerate(['even', 'odd']):
            ax = axes[idx]
            data = df[df['head_type'] == head_type]

            x = np.arange(len(prompts))
            width = 0.25

            for i, metric in enumerate(['format_bandwidth', 'numerical_bandwidth', 'other_bandwidth']):
                means = data.groupby('format')[metric].mean()
                stds = data.groupby('format')[metric].std()
                ax.bar(x + i*width, means.values, width,
                      yerr=stds.values, label=metric.replace('_bandwidth', ''),
                      alpha=0.8, capsize=5)

            ax.set_xlabel('Format')
            ax.set_ylabel('Bandwidth Proportion')
            ax.set_title(f'{head_type.capitalize()} Heads - Bandwidth Distribution')
            ax.set_xticks(x + width)
            ax.set_xticklabels(prompts.keys())
            ax.legend()
            ax.axhline(y=0.4, color='r', linestyle='--', alpha=0.7, label='40% threshold')

        plt.tight_layout()
        plt.savefig('/home/paperspace/dev/MATS9/bandwidth/figures/bandwidth_distribution.png',
                   dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Run bandwidth analysis experiment"""
    print("="*60)
    print("BANDWIDTH COMPETITION THEORY - EXPERIMENT 1.1")
    print("Attention Bandwidth Distribution Analysis")
    print("="*60)

    # Test prompts with increasing format complexity
    prompts = {
        'Simple': '9.8 or 9.11? Answer:',
        'Q&A': 'Q: Which is bigger: 9.8 or 9.11? A:',
        'Chat': '<|start_header_id|>user<|end_header_id|>\n\nWhich is bigger: 9.8 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    }

    # Initialize analyzer
    analyzer = BandwidthAnalyzer()

    # Run analysis
    print("\nRunning bandwidth analysis...")
    df_bandwidth = analyzer.analyze_attention_bandwidth(prompts)

    # Create visualizations
    print("\nCreating visualizations...")
    analyzer.visualize_bandwidth_results(df_bandwidth, prompts)

    # Print summary statistics
    print("\n" + "="*60)
    print("BANDWIDTH ANALYSIS SUMMARY")
    print("="*60)

    print("\nNumerical bandwidth by format and head type:")
    summary = df_bandwidth.groupby(['format', 'head_type'])['numerical_bandwidth'].agg(['mean', 'std'])
    print(summary)

    # Test key hypotheses
    print("\n" + "-"*40)
    print("HYPOTHESIS TESTING")
    print("-"*40)

    # Hypothesis 1: Even heads maintain >40% numerical bandwidth
    even_heads = df_bandwidth[df_bandwidth['head_type'] == 'even']
    even_above_threshold = (even_heads['numerical_bandwidth'] > 0.4).groupby(even_heads['format']).mean()
    print(f"\nHypothesis 1 - Even heads >40% numerical bandwidth:")
    for format_name, pct in even_above_threshold.items():
        print(f"  {format_name}: {pct:.1%} of even heads above threshold")

    # Hypothesis 2: Odd heads drop below 40% in Q&A format
    odd_heads = df_bandwidth[df_bandwidth['head_type'] == 'odd']
    odd_below_threshold = (odd_heads['numerical_bandwidth'] < 0.4).groupby(odd_heads['format']).mean()
    print(f"\nHypothesis 2 - Odd heads <40% numerical bandwidth:")
    for format_name, pct in odd_below_threshold.items():
        print(f"  {format_name}: {pct:.1%} of odd heads below threshold")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'/home/paperspace/dev/MATS9/bandwidth/set1_bandwidth_competition/bandwidth_results_{timestamp}.json'

    # Convert DataFrame to JSON-serializable format
    results_dict = {
        'summary_stats': {str(k): v for k, v in summary.to_dict().items()},
        'even_above_threshold': {str(k): float(v) for k, v in even_above_threshold.to_dict().items()},
        'odd_below_threshold': {str(k): float(v) for k, v in odd_below_threshold.to_dict().items()},
        'raw_data': df_bandwidth.to_dict('records')
    }

    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)

    print(f"\n✅ Results saved to {results_file}")
    print(f"✅ Visualization saved to /home/paperspace/dev/MATS9/bandwidth/figures/bandwidth_distribution.png")


if __name__ == "__main__":
    main()