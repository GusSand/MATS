#!/usr/bin/env python3
"""
Experiment 5 (Alternative): Head Logit Lens Analysis

Since SAEs require Python 3.10+, we use logit lens instead:
For each head, project its output through the unembedding matrix to see
what tokens it promotes/suppresses.

This gives interpretable evidence for what each head "does".
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import json

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LAYER_IDX = 10
DEVICE = "cuda"

EVEN_HEADS = list(range(0, 32, 2))
ODD_HEADS = list(range(1, 32, 2))

PROMPT = "Q: Which is bigger: 9.8 or 9.11?\nA:"


class HeadLogitLens:
    def __init__(self):
        print(f"Loading {MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map=DEVICE
        )
        self.model.eval()

        self.n_heads = 32
        self.head_dim = self.model.config.hidden_size // self.n_heads
        self.hooks = []

        # Get unembedding matrix
        self.unembed = self.model.lm_head.weight.detach()  # [vocab, hidden]

    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_head_outputs(self, prompt: str):
        """Get per-head attention outputs."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)

        attn_output = {}

        def capture_attn(module, input, output):
            attn_output['out'] = output[0].detach()

        hook = self.model.model.layers[LAYER_IDX].self_attn.register_forward_hook(capture_attn)
        self.hooks.append(hook)

        try:
            with torch.no_grad():
                _ = self.model(**inputs)

            # Reshape to per-head: [batch, seq, n_heads, head_dim]
            out = attn_output['out']
            batch, seq, hidden = out.shape
            per_head = out.view(batch, seq, self.n_heads, self.head_dim)

            return per_head[0, -1, :, :]  # [n_heads, head_dim] at last position
        finally:
            self.clear_hooks()

    def project_to_vocab(self, head_output, top_k=10):
        """
        Project head output to vocabulary space.

        The o_proj combines heads, so we need to handle this carefully.
        We'll project each head's output through its slice of o_proj,
        then through the unembedding.
        """
        o_proj = self.model.model.layers[LAYER_IDX].self_attn.o_proj.weight.detach()
        # o_proj is [hidden, hidden], takes concatenated heads

        results = []
        for head_idx in range(self.n_heads):
            # Get this head's slice of o_proj
            start = head_idx * self.head_dim
            end = (head_idx + 1) * self.head_dim
            head_oproj = o_proj[:, start:end]  # [hidden, head_dim]

            # Project head output to hidden space
            head_out = head_output[head_idx]  # [head_dim]
            hidden_contrib = torch.matmul(head_oproj.float(), head_out.float())  # [hidden]

            # Project to vocabulary
            logits = torch.matmul(hidden_contrib, self.unembed.T.float())  # [vocab]

            # Get top tokens
            top_vals, top_idx = torch.topk(logits, k=top_k)
            top_tokens = [self.tokenizer.decode([idx]) for idx in top_idx.cpu().tolist()]

            # Get bottom tokens (most suppressed)
            bot_vals, bot_idx = torch.topk(-logits, k=top_k)
            bot_tokens = [self.tokenizer.decode([idx]) for idx in bot_idx.cpu().tolist()]

            results.append({
                'head_idx': head_idx,
                'top_tokens': top_tokens,
                'top_logits': top_vals.cpu().tolist(),
                'bottom_tokens': bot_tokens,
                'bottom_logits': (-bot_vals).cpu().tolist()
            })

        return results

    def analyze_numerical_bias(self, vocab_results):
        """Check if heads promote numerical vs non-numerical tokens."""
        numerical_chars = set('0123456789.')

        results = []
        for r in vocab_results:
            head_idx = r['head_idx']
            top_tokens = r['top_tokens']

            # Count numerical tokens in top-10
            num_numerical = sum(1 for t in top_tokens if any(c in numerical_chars for c in t))

            results.append({
                'head_idx': head_idx,
                'numerical_in_top10': num_numerical,
                'top_tokens': top_tokens[:5]
            })

        return results

    def run_analysis(self):
        """Run full analysis."""
        results = {
            'model': MODEL_NAME,
            'layer': LAYER_IDX,
            'prompt': PROMPT,
            'timestamp': datetime.now().isoformat()
        }

        print("\n" + "="*70)
        print("EXPERIMENT 5: HEAD LOGIT LENS ANALYSIS")
        print("="*70)

        print(f"\nPrompt: {PROMPT}")

        # Get head outputs
        print("\nGetting head outputs...")
        head_outputs = self.get_head_outputs(PROMPT)

        # Project to vocabulary
        print("Projecting to vocabulary...")
        vocab_results = self.project_to_vocab(head_outputs, top_k=20)

        # Analyze numerical bias
        print("Analyzing numerical bias...")
        bias_results = self.analyze_numerical_bias(vocab_results)

        # Aggregate by even/odd
        even_numerical = np.mean([r['numerical_in_top10'] for r in bias_results if r['head_idx'] in EVEN_HEADS])
        odd_numerical = np.mean([r['numerical_in_top10'] for r in bias_results if r['head_idx'] in ODD_HEADS])

        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")

        print(f"\nNumerical tokens in top-10 promoted tokens:")
        print(f"  Even heads: {even_numerical:.2f} avg")
        print(f"  Odd heads:  {odd_numerical:.2f} avg")

        # Per-head details
        print("\n--- PER-HEAD TOP TOKENS ---")
        print(f"{'Head':<6} {'Type':<5} {'#Num':<5} {'Top 5 Promoted Tokens'}")
        print("-" * 70)
        for r in bias_results:
            htype = 'Even' if r['head_idx'] % 2 == 0 else 'Odd'
            tokens_str = ', '.join(f"'{t}'" for t in r['top_tokens'])
            print(f"H{r['head_idx']:<4} {htype:<5} {r['numerical_in_top10']:<5} {tokens_str}")

        # Check specific tokens
        print("\n--- CHECKING SPECIFIC TOKENS ---")
        target_tokens = ['9', '.', '8', '11', '1', '>', '<', 'bigger', 'larger']

        for target in target_tokens:
            token_ids = self.tokenizer.encode(target, add_special_tokens=False)
            if not token_ids:
                continue

            print(f"\nToken '{target}' (id={token_ids[0]}):")
            token_logits = []

            for r in vocab_results:
                head_idx = r['head_idx']
                # Re-compute logit for this specific token
                head_out = head_outputs[head_idx]
                o_proj = self.model.model.layers[LAYER_IDX].self_attn.o_proj.weight.detach()
                start = head_idx * self.head_dim
                end = (head_idx + 1) * self.head_dim
                head_oproj = o_proj[:, start:end]
                hidden_contrib = torch.matmul(head_oproj.float(), head_out.float())
                logit = torch.matmul(hidden_contrib, self.unembed[token_ids[0]].float()).item()
                token_logits.append(logit)

            even_logits = [token_logits[i] for i in EVEN_HEADS]
            odd_logits = [token_logits[i] for i in ODD_HEADS]

            print(f"  Even heads: mean={np.mean(even_logits):.3f}, std={np.std(even_logits):.3f}")
            print(f"  Odd heads:  mean={np.mean(odd_logits):.3f}, std={np.std(odd_logits):.3f}")

        # Save results
        results['vocab_projections'] = vocab_results
        results['numerical_bias'] = bias_results
        results['even_numerical_avg'] = float(even_numerical)
        results['odd_numerical_avg'] = float(odd_numerical)

        with open('head_logit_lens_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        self.write_markdown(results, bias_results, even_numerical, odd_numerical)

        return results

    def write_markdown(self, results, bias_results, even_num, odd_num):
        md = f"""# Head Logit Lens Analysis Results

**Model**: {results['model']}
**Layer**: {results['layer']}
**Prompt**: `{results['prompt']}`
**Date**: {results['timestamp']}

---

## Summary

| Head Type | Numerical Tokens in Top-10 |
|-----------|---------------------------|
| Even heads | {even_num:.2f} avg |
| Odd heads | {odd_num:.2f} avg |

---

## Method

For each attention head:
1. Get head's output at last token position
2. Project through o_proj (that head's slice)
3. Project to vocabulary via unembedding matrix
4. Count numerical tokens (0-9, .) in top-10 promoted tokens

---

## Per-Head Results

| Head | Type | #Numerical | Top Promoted Tokens |
|------|------|------------|---------------------|
"""
        for r in bias_results:
            htype = 'Even' if r['head_idx'] % 2 == 0 else 'Odd'
            tokens = ', '.join(f"'{t}'" for t in r['top_tokens'][:3])
            md += f"| H{r['head_idx']} | {htype} | {r['numerical_in_top10']} | {tokens} |\n"

        md += f"""
---

## One-Slide Summary

```
HEAD LOGIT LENS: What tokens do heads promote?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Method: Project each head's output to vocabulary space

Numerical tokens (0-9, .) in top-10:
  Even heads: {even_num:.2f} avg
  Odd heads:  {odd_num:.2f} avg

Interpretation:
  [Based on results]
```
"""

        with open('head_logit_lens_results.md', 'w') as f:
            f.write(md)

        print("\nResults written to: head_logit_lens_results.md")


def main():
    analyzer = HeadLogitLens()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
