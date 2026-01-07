#!/usr/bin/env python3
"""
Attention Pattern Analysis (IOI-style)

Analyzes which attention heads attend to security-relevant tokens.
Goal: Identify "Security Context Heads" similar to IOI's "Name Mover Heads"

Key questions:
1. Which heads attend to "WARNING", "snprintf", "buffer", "overflow"?
2. How do attention patterns differ between secure vs neutral contexts?
3. Which heads at the final position attend back to security tokens?
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
import seaborn as sns

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

# Security-relevant keywords to track
SECURITY_KEYWORDS = ['WARNING', 'snprintf', 'buffer', 'overflow', 'prevent']


class AttentionAnalyzer:
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        print("Loading model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            output_attentions=True  # Enable attention output
        )
        self.model.eval()

        self.n_layers = self.model.config.num_hidden_layers
        self.n_heads = self.model.config.num_attention_heads
        print(f"Model loaded: {self.n_layers} layers, {self.n_heads} heads")

    def get_token_positions(self, prompt: str, keywords: list) -> dict:
        """Find positions of security-relevant tokens."""
        tokens = self.tokenizer.encode(prompt)
        token_strs = [self.tokenizer.decode([t]) for t in tokens]

        positions = {}
        for keyword in keywords:
            keyword_lower = keyword.lower()
            for i, tok_str in enumerate(token_strs):
                if keyword_lower in tok_str.lower():
                    if keyword not in positions:
                        positions[keyword] = []
                    positions[keyword].append(i)

        return positions, token_strs

    def get_attention_patterns(self, prompt: str) -> tuple:
        """Get attention patterns for all layers and heads."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        # attentions is a tuple of (n_layers) tensors, each of shape (batch, n_heads, seq_len, seq_len)
        attentions = outputs.attentions

        # Stack into single tensor: (n_layers, n_heads, seq_len, seq_len)
        attention_tensor = torch.stack([a.squeeze(0) for a in attentions])

        return attention_tensor.cpu().numpy(), inputs['input_ids'].shape[1]

    def analyze_attention_to_keywords(self, prompt: str, keywords: list) -> dict:
        """
        Analyze how much each head attends to keyword tokens from the final position.
        This identifies heads that "read" security context when making predictions.
        """
        positions, token_strs = self.get_token_positions(prompt, keywords)
        attention, seq_len = self.get_attention_patterns(prompt)

        results = {
            'prompt': prompt[:100] + '...',
            'seq_len': seq_len,
            'token_strs': token_strs,
            'keyword_positions': positions,
            'heads': {}
        }

        # For each head, compute attention from final position to keyword positions
        for layer in range(self.n_layers):
            for head in range(self.n_heads):
                head_key = f"L{layer}H{head}"

                # Attention from final position to all positions
                final_attn = attention[layer, head, -1, :]  # Shape: (seq_len,)

                # Compute attention to each keyword
                keyword_attention = {}
                total_keyword_attn = 0.0

                for keyword, pos_list in positions.items():
                    attn_sum = sum(final_attn[p] for p in pos_list)
                    keyword_attention[keyword] = float(attn_sum)
                    total_keyword_attn += attn_sum

                results['heads'][head_key] = {
                    'layer': layer,
                    'head': head,
                    'keyword_attention': keyword_attention,
                    'total_keyword_attention': float(total_keyword_attn),
                    'max_attention': float(final_attn.max()),
                    'max_attention_pos': int(final_attn.argmax())
                }

        return results

    def compare_attention_patterns(self, secure_prompt: str, neutral_prompt: str) -> dict:
        """
        Compare attention patterns between secure and neutral prompts.
        Identify heads that behave differently.
        """
        print("\nAnalyzing secure prompt...")
        secure_attn, secure_len = self.get_attention_patterns(secure_prompt)
        secure_positions, secure_tokens = self.get_token_positions(secure_prompt, SECURITY_KEYWORDS)

        print("Analyzing neutral prompt...")
        neutral_attn, neutral_len = self.get_attention_patterns(neutral_prompt)
        neutral_positions, neutral_tokens = self.get_token_positions(neutral_prompt, SECURITY_KEYWORDS)

        print(f"\nSecure prompt: {secure_len} tokens")
        print(f"Neutral prompt: {neutral_len} tokens")
        print(f"Security keywords found in secure: {list(secure_positions.keys())}")

        results = {
            'secure_tokens': secure_tokens,
            'neutral_tokens': neutral_tokens,
            'secure_keyword_positions': secure_positions,
            'head_differences': {}
        }

        # Compare final-position attention patterns
        # Focus on where each head attends from the last token
        for layer in range(self.n_layers):
            for head in range(self.n_heads):
                head_key = f"L{layer}H{head}"

                secure_final = secure_attn[layer, head, -1, :]
                neutral_final = neutral_attn[layer, head, -1, :]

                # Compute attention to security keyword positions (secure prompt only)
                security_attn = 0.0
                for keyword, pos_list in secure_positions.items():
                    for p in pos_list:
                        if p < len(secure_final):
                            security_attn += secure_final[p]

                # Compute entropy (measure of attention spread)
                def entropy(attn):
                    attn = attn + 1e-10  # Avoid log(0)
                    attn = attn / attn.sum()
                    return -np.sum(attn * np.log(attn))

                secure_entropy = entropy(secure_final)
                neutral_entropy = entropy(neutral_final)

                results['head_differences'][head_key] = {
                    'layer': layer,
                    'head': head,
                    'security_token_attention': float(security_attn),
                    'secure_entropy': float(secure_entropy),
                    'neutral_entropy': float(neutral_entropy),
                    'entropy_diff': float(secure_entropy - neutral_entropy),
                    'secure_max_pos': int(secure_final.argmax()),
                    'neutral_max_pos': int(neutral_final.argmax())
                }

        return results

    def find_security_context_heads(self, comparison: dict, top_k: int = 20) -> list:
        """
        Identify heads that attend most to security tokens.
        These are candidate "Security Context Heads" (analogous to IOI's Name Mover Heads).
        """
        heads = comparison['head_differences']

        # Sort by attention to security tokens
        sorted_heads = sorted(
            heads.items(),
            key=lambda x: x[1]['security_token_attention'],
            reverse=True
        )

        return sorted_heads[:top_k]

    def visualize_top_heads(self, comparison: dict, output_path: Path):
        """Create visualization of security context heads."""
        heads = comparison['head_differences']

        # Create heatmap of security attention by layer and head
        security_attn_matrix = np.zeros((self.n_layers, self.n_heads))

        for head_key, data in heads.items():
            security_attn_matrix[data['layer'], data['head']] = data['security_token_attention']

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Heatmap
        ax1 = axes[0]
        sns.heatmap(security_attn_matrix, ax=ax1, cmap='Reds',
                    xticklabels=range(self.n_heads),
                    yticklabels=range(self.n_layers))
        ax1.set_xlabel('Head')
        ax1.set_ylabel('Layer')
        ax1.set_title('Attention to Security Tokens\n(from final position)')

        # Bar chart of top heads
        ax2 = axes[1]
        top_heads = self.find_security_context_heads(comparison, top_k=15)
        names = [h[0] for h in top_heads]
        values = [h[1]['security_token_attention'] for h in top_heads]

        ax2.barh(range(len(names)), values, color='coral')
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names)
        ax2.set_xlabel('Attention to Security Tokens')
        ax2.set_title('Top 15 Security Context Heads')
        ax2.invert_yaxis()

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n📊 Visualization saved to: {output_path}")

    def visualize_attention_pattern(self, prompt: str, layer: int, head: int, output_path: Path):
        """Visualize attention pattern for a specific head."""
        attention, seq_len = self.get_attention_patterns(prompt)
        positions, token_strs = self.get_token_positions(prompt, SECURITY_KEYWORDS)

        # Get attention matrix for this head
        attn_matrix = attention[layer, head]  # (seq_len, seq_len)

        # Truncate token strings for display
        short_tokens = [t[:8] for t in token_strs]

        fig, ax = plt.subplots(figsize=(14, 12))

        sns.heatmap(attn_matrix, ax=ax, cmap='Blues',
                    xticklabels=short_tokens,
                    yticklabels=short_tokens)

        # Mark security keyword positions
        for keyword, pos_list in positions.items():
            for p in pos_list:
                ax.axvline(x=p + 0.5, color='red', linewidth=2, alpha=0.5)
                ax.axhline(y=p + 0.5, color='red', linewidth=2, alpha=0.5)

        ax.set_xlabel('Key (attending to)')
        ax.set_ylabel('Query (attending from)')
        ax.set_title(f'Attention Pattern: Layer {layer}, Head {head}\n(Red lines = security keywords)')

        plt.xticks(rotation=90, fontsize=6)
        plt.yticks(fontsize=6)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    analyzer = AttentionAnalyzer()

    # Compare attention patterns
    print("\n" + "="*60)
    print("ATTENTION PATTERN ANALYSIS")
    print("="*60)

    comparison = analyzer.compare_attention_patterns(SECURE_PROMPT, NEUTRAL_PROMPT)

    # Find top security context heads
    print("\n" + "="*60)
    print("TOP SECURITY CONTEXT HEADS")
    print("="*60)
    print("\nHeads with highest attention to security tokens (from final position):")

    top_heads = analyzer.find_security_context_heads(comparison, top_k=20)

    print("\n| Rank | Head | Layer | Attn to Security | Entropy Diff |")
    print("|------|------|-------|------------------|--------------|")
    for i, (head_key, data) in enumerate(top_heads):
        print(f"| {i+1:4d} | {head_key:6s} | {data['layer']:5d} | {data['security_token_attention']:16.4f} | {data['entropy_diff']:12.4f} |")

    # Identify candidate circuit
    print("\n" + "="*60)
    print("CANDIDATE SECURITY CIRCUIT")
    print("="*60)

    # Group by layer
    layer_counts = {}
    for head_key, data in top_heads[:15]:
        layer = data['layer']
        if layer not in layer_counts:
            layer_counts[layer] = []
        layer_counts[layer].append(data['head'])

    print("\nTop heads by layer:")
    for layer in sorted(layer_counts.keys()):
        heads = layer_counts[layer]
        print(f"  Layer {layer:2d}: Heads {heads}")

    # Visualize
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    viz_path = results_dir / f"security_heads_heatmap_{timestamp}.png"
    analyzer.visualize_top_heads(comparison, viz_path)

    # Visualize top 3 heads' attention patterns
    print("\nVisualizing attention patterns for top 3 heads...")
    for i, (head_key, data) in enumerate(top_heads[:3]):
        pattern_path = results_dir / f"attention_pattern_{head_key}_{timestamp}.png"
        analyzer.visualize_attention_pattern(
            SECURE_PROMPT,
            data['layer'],
            data['head'],
            pattern_path
        )
        print(f"  {head_key}: {pattern_path.name}")

    # Save results
    results = {
        'timestamp': timestamp,
        'secure_prompt': SECURE_PROMPT,
        'neutral_prompt': NEUTRAL_PROMPT,
        'security_keywords': SECURITY_KEYWORDS,
        'comparison': {
            'secure_tokens': comparison['secure_tokens'],
            'neutral_tokens': comparison['neutral_tokens'],
            'keyword_positions': {k: v for k, v in comparison['secure_keyword_positions'].items()}
        },
        'top_security_heads': [
            {'head': h[0], **h[1]} for h in top_heads
        ],
        'layer_distribution': {str(k): v for k, v in layer_counts.items()}
    }

    with open(results_dir / f"attention_analysis_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {results_dir}")

    return comparison, top_heads


if __name__ == "__main__":
    comparison, top_heads = main()
