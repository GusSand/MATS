#!/usr/bin/env python3
"""
Bandwidth Competition Theory - Experiment 1.1: Attention Bandwidth Distribution
===============================================================================

This experiment measures how attention is distributed across token categories
(format tokens vs numerical tokens) in different prompt formats using nnsight.

Expected result:
- Even heads maintain >40% attention on numerical tokens
- Odd heads drop below threshold in Q&A format

Implementation follows bandwidth_competition_experiments.md specification.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

# Model and intervention tools
from nnsight import LanguageModel

# Statistical analysis
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm

# Visualization settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Model configuration
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LAYER_OF_INTEREST = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import json
from datetime import datetime
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'


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


def analyze_attention_bandwidth(model, prompts: Dict[str, str]) -> pd.DataFrame:
    """
    Measure how attention is distributed across token categories.

    Expected result:
    - Even heads maintain >40% attention on numerical tokens
    - Odd heads drop below threshold in Q&A format
    """
    results = []
    token_cat = TokenCategory()

    # Configure model to output attention weights
    model.config.output_attentions = True
    model.config.return_dict = True

    for format_name, prompt in prompts.items():
        print(f"Analyzing format: {format_name}")

        # Tokenize input to understand token structure
        tokens = model.tokenizer.encode(prompt)
        token_strings = model.tokenizer.convert_ids_to_tokens(tokens)

        print(f"  Tokens: {token_strings}")

        # Run forward pass with attention capture using nnsight trace
        with model.trace(prompt) as tracer:
            outputs = model.forward(output_attentions=True, return_dict=True)
            attentions = outputs.attentions

        # Get attention weights for our layer of interest
        attn = attentions[LAYER_OF_INTEREST]  # Shape: [batch, heads, seq_len, seq_len]
        print(f"  Attention shape: {attn.shape}")

        # Categorize token positions
        format_positions = []
        numerical_positions = []
        other_positions = []

        for i, token in enumerate(tokens):
            token_str = token.replace('▁', '').strip()
            if any(fmt in token_str for fmt in token_cat.format_tokens):
                format_positions.append(i)
            elif any(num in token_str for num in token_cat.numerical_tokens):
                numerical_positions.append(i)
            else:
                other_positions.append(i)

        print(f"  Format positions: {format_positions}")
        print(f"  Numerical positions: {numerical_positions}")
        print(f"  Other positions: {other_positions}")

        for head_idx in range(attn.shape[1]):
            head_attn = attn[0, head_idx, :, :].cpu().numpy()

            # Calculate attention bandwidth from last token to each category
            # This represents what the model "pays attention to" when making predictions
            last_token_idx = attn.shape[2] - 1

            format_attn = 0
            numerical_attn = 0
            other_attn = 0

            for pos in format_positions:
                if pos < attn.shape[3]:
                    format_attn += head_attn[last_token_idx, pos]

            for pos in numerical_positions:
                if pos < attn.shape[3]:
                    numerical_attn += head_attn[last_token_idx, pos]

            for pos in other_positions:
                if pos < attn.shape[3]:
                    other_attn += head_attn[last_token_idx, pos]

            # Normalize
            total = format_attn + numerical_attn + other_attn
            if total > 0:
                format_attn /= total
                numerical_attn /= total
                other_attn /= total

            results.append({
                'format': format_name,
                'head_idx': head_idx,
                'head_type': 'even' if head_idx % 2 == 0 else 'odd',
                'format_bandwidth': format_attn,
                'numerical_bandwidth': numerical_attn,
                'other_bandwidth': other_attn
            })

    df = pd.DataFrame(results)

    # Visualize results
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

    return df


def main():
    """Run bandwidth analysis experiment"""
    print("="*60)
    print("BANDWIDTH COMPETITION THEORY - EXPERIMENT 1.1")
    print("Attention Bandwidth Distribution Analysis (NNsight)")
    print("="*60)

    # Load model using nnsight
    print(f"Loading model: {MODEL_NAME}")
    model = LanguageModel(MODEL_NAME, device_map="auto")

    # Test prompts with increasing format complexity
    prompts = {
        'Simple': '9.8 or 9.11? Answer:',
        'Q&A': 'Q: Which is bigger: 9.8 or 9.11? A:',
        'Chat': '<|start_header_id|>user<|end_header_id|>\n\nWhich is bigger: 9.8 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    }

    # Run analysis
    print("\nRunning bandwidth analysis...")
    df_bandwidth = analyze_attention_bandwidth(model, prompts)

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

    # Additional analysis: Mean differences
    print(f"\nAdditional Analysis - Mean numerical bandwidth:")
    for format_name in prompts.keys():
        format_data = df_bandwidth[df_bandwidth['format'] == format_name]
        even_mean = format_data[format_data['head_type'] == 'even']['numerical_bandwidth'].mean()
        odd_mean = format_data[format_data['head_type'] == 'odd']['numerical_bandwidth'].mean()
        print(f"  {format_name}: Even={even_mean:.3f}, Odd={odd_mean:.3f}, Diff={even_mean-odd_mean:.3f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'/home/paperspace/dev/MATS9/bandwidth/set1_bandwidth_competition/bandwidth_results_{timestamp}.json'

    # Convert DataFrame to JSON-serializable format
    results_dict = {
        'summary_stats': {str(k): {str(k2): float(v2) for k2, v2 in v.items()}
                         for k, v in summary.to_dict().items()},
        'even_above_threshold': {str(k): float(v) for k, v in even_above_threshold.to_dict().items()},
        'odd_below_threshold': {str(k): float(v) for k, v in odd_below_threshold.to_dict().items()},
        'raw_data': df_bandwidth.to_dict('records')
    }

    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)

    print(f"\n✅ Results saved to {results_file}")
    print(f"✅ Visualization saved to /home/paperspace/dev/MATS9/bandwidth/figures/bandwidth_distribution.png")

    # Quick validation of results
    print(f"\n" + "="*60)
    print("THEORY VALIDATION")
    print("="*60)

    # Check if theory predictions hold
    simple_even = df_bandwidth[(df_bandwidth['format'] == 'Simple') &
                              (df_bandwidth['head_type'] == 'even')]['numerical_bandwidth'].mean()
    qa_even = df_bandwidth[(df_bandwidth['format'] == 'Q&A') &
                          (df_bandwidth['head_type'] == 'even')]['numerical_bandwidth'].mean()
    qa_odd = df_bandwidth[(df_bandwidth['format'] == 'Q&A') &
                         (df_bandwidth['head_type'] == 'odd')]['numerical_bandwidth'].mean()

    print(f"Key metrics:")
    print(f"  Even heads (Simple format): {simple_even:.3f}")
    print(f"  Even heads (Q&A format): {qa_even:.3f}")
    print(f"  Odd heads (Q&A format): {qa_odd:.3f}")

    theory_supported = simple_even > 0.3 and qa_even > qa_odd
    print(f"\nBandwidth competition theory supported: {theory_supported}")

    return df_bandwidth


if __name__ == "__main__":
    main()