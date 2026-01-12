#!/usr/bin/env python3
"""
Integrated Gradients Analysis

Compute token-level attribution for the snprintf vs sprintf decision.
This reveals which input tokens most influence the security-aware output.

Key questions:
1. Which tokens have highest attribution to P(snprintf)?
2. Do security keywords (WARNING, buffer, overflow) have high attribution?
3. How does attribution differ between secure and neutral contexts?
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


class IntegratedGradients:
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        print("Loading model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Need float32 for gradients
            device_map="auto"
        )
        # Keep model in eval mode but enable gradients for embeddings
        self.model.eval()

        # Token IDs
        self.snprintf_token = self.tokenizer.encode(" snprintf", add_special_tokens=False)[0]
        self.sprintf_token = self.tokenizer.encode(" sprintf", add_special_tokens=False)[0]

        print(f"snprintf token: {self.snprintf_token}, sprintf token: {self.sprintf_token}")

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings for input IDs."""
        return self.model.model.embed_tokens(input_ids)

    def forward_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass starting from embeddings."""
        # Create position IDs
        seq_len = embeddings.shape[1]
        position_ids = torch.arange(seq_len, device=embeddings.device).unsqueeze(0)

        # Forward through the transformer
        outputs = self.model.model(
            inputs_embeds=embeddings,
            position_ids=position_ids,
            output_hidden_states=False,
            return_dict=True
        )

        # Get logits
        hidden_states = outputs.last_hidden_state
        logits = self.model.lm_head(hidden_states)

        return logits

    def compute_input_x_gradient(
        self,
        prompt: str,
        target_token: int
    ) -> dict:
        """
        Compute Input x Gradient attribution (simpler, more stable than IG).

        Attribution = input_embedding * gradient

        This gives a quick approximation of token importance.
        """
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs['input_ids']

        # Get embeddings with gradient tracking
        embeddings = self.get_embeddings(input_ids)
        embeddings = embeddings.clone().detach().requires_grad_(True)

        # Forward pass
        logits = self.forward_from_embeddings(embeddings)

        # Get target logit (last position, target token)
        target_logit = logits[0, -1, target_token]

        # Backward pass
        self.model.zero_grad()
        target_logit.backward()

        # Get gradient w.r.t. embeddings
        grad = embeddings.grad  # (1, seq_len, hidden_size)

        # Input x Gradient: element-wise multiply, then sum over hidden dim
        attributions = (embeddings * grad).sum(dim=-1).squeeze(0)  # (seq_len,)
        attributions = attributions.detach().cpu().numpy()

        # Get token strings
        tokens = [self.tokenizer.decode([t.item()]) for t in input_ids[0]]

        # Compute final probability
        with torch.no_grad():
            new_embeddings = self.get_embeddings(input_ids)
            logits = self.forward_from_embeddings(new_embeddings)
            probs = torch.softmax(logits[0, -1, :], dim=-1)
            target_prob = probs[target_token].item()

        return {
            'tokens': tokens,
            'attributions': attributions.tolist(),
            'target_token': target_token,
            'target_token_str': self.tokenizer.decode([target_token]),
            'target_prob': target_prob,
            'method': 'input_x_gradient'
        }

    def compute_gradient_norm(
        self,
        prompt: str,
        target_token: int
    ) -> dict:
        """
        Compute gradient L2 norm at each token position.
        Higher gradient norm = more sensitive to that token.
        """
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs['input_ids']

        # Get embeddings with gradient tracking
        embeddings = self.get_embeddings(input_ids)
        embeddings = embeddings.clone().detach().requires_grad_(True)

        # Forward pass
        logits = self.forward_from_embeddings(embeddings)

        # Get target logit
        target_logit = logits[0, -1, target_token]

        # Backward pass
        self.model.zero_grad()
        target_logit.backward()

        # Get gradient L2 norm at each position
        grad = embeddings.grad  # (1, seq_len, hidden_size)
        grad_norms = torch.norm(grad, dim=-1).squeeze(0)  # (seq_len,)
        grad_norms = grad_norms.detach().cpu().numpy()

        # Get token strings
        tokens = [self.tokenizer.decode([t.item()]) for t in input_ids[0]]

        # Compute final probability
        with torch.no_grad():
            new_embeddings = self.get_embeddings(input_ids)
            logits = self.forward_from_embeddings(new_embeddings)
            probs = torch.softmax(logits[0, -1, :], dim=-1)
            target_prob = probs[target_token].item()

        return {
            'tokens': tokens,
            'gradient_norms': grad_norms.tolist(),
            'target_token': target_token,
            'target_token_str': self.tokenizer.decode([target_token]),
            'target_prob': target_prob,
            'method': 'gradient_norm'
        }

    def analyze_security_attribution(self, prompt: str) -> dict:
        """
        Analyze attributions for both snprintf and sprintf tokens.
        Uses gradient norm (more stable than integrated gradients).
        """
        print(f"\nComputing gradient attribution for snprintf...")
        snprintf_result = self.compute_gradient_norm(prompt, self.snprintf_token)

        print(f"Computing gradient attribution for sprintf...")
        sprintf_result = self.compute_gradient_norm(prompt, self.sprintf_token)

        # Also compute input x gradient for comparison
        print(f"Computing input x gradient for snprintf...")
        snprintf_ixg = self.compute_input_x_gradient(prompt, self.snprintf_token)

        print(f"Computing input x gradient for sprintf...")
        sprintf_ixg = self.compute_input_x_gradient(prompt, self.sprintf_token)

        # Compute difference: grad norm for snprintf minus sprintf
        grad_diff = np.array(snprintf_result['gradient_norms']) - np.array(sprintf_result['gradient_norms'])

        # Compute input x gradient difference
        ixg_diff = np.array(snprintf_ixg['attributions']) - np.array(sprintf_ixg['attributions'])

        return {
            'snprintf_grad_norm': snprintf_result,
            'sprintf_grad_norm': sprintf_result,
            'snprintf_ixg': snprintf_ixg,
            'sprintf_ixg': sprintf_ixg,
            'grad_norm_diff': grad_diff.tolist(),
            'ixg_diff': ixg_diff.tolist(),
            'tokens': snprintf_result['tokens']
        }

    def visualize(self, secure_results: dict, neutral_results: dict, output_path: Path):
        """Create visualization of gradient attribution."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Helper to plot attributions
        def plot_attributions(ax, tokens, attributions, title, color='blue'):
            x = range(len(tokens))
            colors = [color if a >= 0 else 'red' for a in attributions]
            ax.bar(x, attributions, color=colors, alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(tokens, rotation=90, fontsize=6)
            ax.set_ylabel('Attribution')
            ax.set_title(title)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Plot 1: Secure prompt - gradient norm difference
        ax1 = axes[0, 0]
        plot_attributions(
            ax1,
            secure_results['tokens'][:50],
            secure_results['grad_norm_diff'][:50],
            f"Secure: Gradient Norm (snprintf - sprintf)\nP(snprintf)={secure_results['snprintf_grad_norm']['target_prob']:.3f}",
            color='green'
        )

        # Plot 2: Secure prompt - input x gradient difference
        ax2 = axes[0, 1]
        plot_attributions(
            ax2,
            secure_results['tokens'][:50],
            secure_results['ixg_diff'][:50],
            "Secure: Input x Gradient (snprintf - sprintf)",
            color='blue'
        )

        # Plot 3: Neutral prompt - gradient norm
        ax3 = axes[1, 0]
        plot_attributions(
            ax3,
            neutral_results['tokens'][:40],
            neutral_results['grad_norm_diff'][:40],
            f"Neutral: Gradient Norm (snprintf - sprintf)\nP(snprintf)={neutral_results['snprintf_grad_norm']['target_prob']:.3f}",
            color='orange'
        )

        # Plot 4: Compare key token attributions
        ax4 = axes[1, 1]

        # Find security keywords in secure prompt
        secure_tokens = secure_results['tokens']
        keywords = ['WARNING', 'snprintf', 'buffer', 'overflow']

        keyword_attrs = []
        keyword_labels = []

        for i, tok in enumerate(secure_tokens):
            tok_clean = tok.strip().upper()
            for kw in keywords:
                if kw.upper() in tok_clean:
                    keyword_attrs.append(secure_results['ixg_diff'][i])
                    keyword_labels.append(f"{tok.strip()} (pos {i})")
                    break

        if keyword_attrs:
            x = range(len(keyword_attrs))
            colors = ['green' if a >= 0 else 'red' for a in keyword_attrs]
            ax4.barh(x, keyword_attrs, color=colors, alpha=0.7)
            ax4.set_yticks(x)
            ax4.set_yticklabels(keyword_labels)
            ax4.set_xlabel('Input x Gradient (snprintf - sprintf)')
            ax4.set_title('Security Keyword Attribution')
            ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        else:
            ax4.text(0.5, 0.5, "No keywords found", ha='center', va='center')
            ax4.set_title('Security Keyword Attribution')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to: {output_path}")


def main():
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    ig = IntegratedGradients()

    print("\n" + "="*70)
    print("GRADIENT ATTRIBUTION ANALYSIS")
    print("="*70)

    # Analyze secure prompt
    print("\n--- Analyzing Secure Prompt ---")
    secure_results = ig.analyze_security_attribution(SECURE_PROMPT)

    # Analyze neutral prompt
    print("\n--- Analyzing Neutral Prompt ---")
    neutral_results = ig.analyze_security_attribution(NEUTRAL_PROMPT)

    # Print results
    print("\n" + "="*70)
    print("TOKEN ATTRIBUTION ANALYSIS")
    print("="*70)

    print("\n### Secure Prompt - Top Attribution Tokens (Input x Gradient, snprintf - sprintf):")
    tokens = secure_results['tokens']
    ixg_diff = secure_results['ixg_diff']

    # Sort by absolute attribution
    sorted_indices = sorted(range(len(ixg_diff)), key=lambda i: abs(ixg_diff[i]), reverse=True)

    print("\n| Rank | Token | Position | IxG Diff |")
    print("|------|-------|----------|----------|")
    for rank, idx in enumerate(sorted_indices[:15]):
        print(f"| {rank+1:4d} | {tokens[idx]:12s} | {idx:8d} | {ixg_diff[idx]:+8.4f} |")

    # Find security keyword attributions
    print("\n### Security Keyword Attributions:")
    keywords = ['WARNING', 'snprintf', 'buffer', 'overflow', 'prevent']

    for i, tok in enumerate(tokens):
        tok_clean = tok.strip().upper()
        for kw in keywords:
            if kw.upper() in tok_clean:
                snprintf_attr = secure_results['snprintf_ixg']['attributions'][i]
                sprintf_attr = secure_results['sprintf_ixg']['attributions'][i]
                grad_diff = secure_results['grad_norm_diff'][i]
                print(f"  {tok.strip():15s} (pos {i:2d}): IxG snprintf={snprintf_attr:+.4f}, "
                      f"sprintf={sprintf_attr:+.4f}, diff={ixg_diff[i]:+.4f}, grad_norm_diff={grad_diff:+.4f}")

    print("\n### Neutral Prompt - Top Attribution Tokens:")
    neutral_tokens = neutral_results['tokens']
    neutral_ixg = neutral_results['ixg_diff']
    sorted_neutral = sorted(range(len(neutral_ixg)), key=lambda i: abs(neutral_ixg[i]), reverse=True)

    print("\n| Rank | Token | Position | IxG Diff |")
    print("|------|-------|----------|----------|")
    for rank, idx in enumerate(sorted_neutral[:10]):
        print(f"| {rank+1:4d} | {neutral_tokens[idx]:12s} | {idx:8d} | {neutral_ixg[idx]:+8.4f} |")

    # Compare probabilities
    print("\n" + "="*70)
    print("PROBABILITY COMPARISON")
    print("="*70)
    print(f"\nSecure prompt:")
    print(f"  P(snprintf) = {secure_results['snprintf_grad_norm']['target_prob']*100:.2f}%")
    print(f"  P(sprintf)  = {secure_results['sprintf_grad_norm']['target_prob']*100:.2f}%")

    print(f"\nNeutral prompt:")
    print(f"  P(snprintf) = {neutral_results['snprintf_grad_norm']['target_prob']*100:.2f}%")
    print(f"  P(sprintf)  = {neutral_results['sprintf_grad_norm']['target_prob']*100:.2f}%")

    # Visualize
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_path = results_dir / f"gradient_attribution_{timestamp}.png"
    ig.visualize(secure_results, neutral_results, viz_path)

    # Save results
    full_results = {
        'timestamp': timestamp,
        'secure_prompt': SECURE_PROMPT[:100] + '...',
        'neutral_prompt': NEUTRAL_PROMPT[:100] + '...',
        'secure_results': {
            'tokens': secure_results['tokens'],
            'snprintf_ixg': secure_results['snprintf_ixg']['attributions'],
            'sprintf_ixg': secure_results['sprintf_ixg']['attributions'],
            'ixg_diff': secure_results['ixg_diff'],
            'grad_norm_diff': secure_results['grad_norm_diff'],
            'snprintf_prob': secure_results['snprintf_grad_norm']['target_prob'],
            'sprintf_prob': secure_results['sprintf_grad_norm']['target_prob']
        },
        'neutral_results': {
            'tokens': neutral_results['tokens'],
            'snprintf_ixg': neutral_results['snprintf_ixg']['attributions'],
            'sprintf_ixg': neutral_results['sprintf_ixg']['attributions'],
            'ixg_diff': neutral_results['ixg_diff'],
            'grad_norm_diff': neutral_results['grad_norm_diff'],
            'snprintf_prob': neutral_results['snprintf_grad_norm']['target_prob'],
            'sprintf_prob': neutral_results['sprintf_grad_norm']['target_prob']
        }
    }

    with open(results_dir / f"gradient_attribution_{timestamp}.json", 'w') as f:
        json.dump(full_results, f, indent=2)

    print(f"\nResults saved to: {results_dir}")

    return secure_results, neutral_results


if __name__ == "__main__":
    secure_results, neutral_results = main()
