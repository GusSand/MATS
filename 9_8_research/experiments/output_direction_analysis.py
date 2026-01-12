#!/usr/bin/env python3
"""
Experiment 6: Output Direction Analysis

Key insight: We checked magnitudes (identical) but not DIRECTIONS.
Maybe even heads output in a specific direction that matters for correction.

Method:
1. Compute "correction vector" = (correct state) - (buggy state)
2. For each head, measure alignment with this correction vector
3. Compare even vs odd heads

If even heads are more aligned with the correction direction,
that would explain why patching them works.
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

PROMPT_BUGGY = "Q: Which is bigger: 9.8 or 9.11?\nA:"
PROMPT_CORRECT = "Which is bigger: 9.8 or 9.11?\nAnswer:"


class DirectionAnalysis:
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
        self.hidden_size = self.model.config.hidden_size
        self.hooks = []

    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_attention_output(self, prompt: str):
        """Get attention output (before adding to residual) at layer 10."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)

        attn_out = {}

        def hook(module, input, output):
            attn_out['out'] = output[0].detach()

        h = self.model.model.layers[LAYER_IDX].self_attn.register_forward_hook(hook)
        self.hooks.append(h)

        try:
            with torch.no_grad():
                _ = self.model(**inputs)
            return attn_out['out'][0, -1, :].cpu()  # [hidden_size] at last position
        finally:
            self.clear_hooks()

    def get_residual_stream(self, prompt: str):
        """Get residual stream state AFTER layer 10 attention (before MLP)."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)

        residual = {}

        def hook(module, input, output):
            # Input to MLP is the residual after attention + layernorm
            residual['state'] = input[0].detach()

        h = self.model.model.layers[LAYER_IDX].mlp.register_forward_hook(hook)
        self.hooks.append(h)

        try:
            with torch.no_grad():
                _ = self.model(**inputs)
            return residual['state'][0, -1, :].cpu()  # [hidden_size]
        finally:
            self.clear_hooks()

    def get_per_head_outputs(self, prompt: str):
        """Get per-head attention outputs."""
        attn_out = self.get_attention_output(prompt)
        # Reshape to [n_heads, head_dim]
        return attn_out.view(self.n_heads, self.head_dim)

    def run_analysis(self):
        """Run direction analysis."""
        results = {
            'model': MODEL_NAME,
            'layer': LAYER_IDX,
            'timestamp': datetime.now().isoformat()
        }

        print("\n" + "="*70)
        print("EXPERIMENT 6: OUTPUT DIRECTION ANALYSIS")
        print("="*70)

        # Get attention outputs for both prompts
        print("\nGetting attention outputs...")
        buggy_attn = self.get_attention_output(PROMPT_BUGGY)
        correct_attn = self.get_attention_output(PROMPT_CORRECT)

        buggy_per_head = buggy_attn.view(self.n_heads, self.head_dim)
        correct_per_head = correct_attn.view(self.n_heads, self.head_dim)

        # Get residual stream states
        print("Getting residual stream states...")
        buggy_residual = self.get_residual_stream(PROMPT_BUGGY)
        correct_residual = self.get_residual_stream(PROMPT_CORRECT)

        # =========================================================
        # ANALYSIS 1: Correction vector in full hidden space
        # =========================================================
        print("\n--- ANALYSIS 1: Full Hidden Space Correction ---")

        correction_full = correct_attn - buggy_attn
        correction_norm = torch.norm(correction_full)
        correction_unit = correction_full / correction_norm

        print(f"Correction vector norm: {correction_norm:.4f}")

        # How much does each head's output align with correction?
        # Project each head through o_proj to get full hidden contribution
        o_proj = self.model.model.layers[LAYER_IDX].self_attn.o_proj.weight.detach().cpu()

        head_alignments = []
        head_contributions = []

        for head_idx in range(self.n_heads):
            start = head_idx * self.head_dim
            end = (head_idx + 1) * self.head_dim
            head_oproj = o_proj[:, start:end].float()

            # Head's contribution to correction
            buggy_head = buggy_per_head[head_idx].float()
            correct_head = correct_per_head[head_idx].float()

            buggy_contrib = torch.matmul(head_oproj, buggy_head)
            correct_contrib = torch.matmul(head_oproj, correct_head)
            head_correction = correct_contrib - buggy_contrib

            # Alignment with full correction vector
            alignment = torch.nn.functional.cosine_similarity(
                head_correction.unsqueeze(0),
                correction_full.float().unsqueeze(0)
            ).item()

            # Contribution magnitude in correction direction
            contribution = torch.dot(head_correction, correction_unit.float()).item()

            head_alignments.append(alignment)
            head_contributions.append(contribution)

        even_alignments = [head_alignments[i] for i in EVEN_HEADS]
        odd_alignments = [head_alignments[i] for i in ODD_HEADS]
        even_contributions = [head_contributions[i] for i in EVEN_HEADS]
        odd_contributions = [head_contributions[i] for i in ODD_HEADS]

        print(f"\nAlignment with correction direction:")
        print(f"  Even heads: mean={np.mean(even_alignments):.4f}, std={np.std(even_alignments):.4f}")
        print(f"  Odd heads:  mean={np.mean(odd_alignments):.4f}, std={np.std(odd_alignments):.4f}")

        print(f"\nContribution in correction direction:")
        print(f"  Even heads: mean={np.mean(even_contributions):.4f}, std={np.std(even_contributions):.4f}")
        print(f"  Odd heads:  mean={np.mean(odd_contributions):.4f}, std={np.std(odd_contributions):.4f}")

        results['alignment'] = {
            'all': head_alignments,
            'even_mean': float(np.mean(even_alignments)),
            'odd_mean': float(np.mean(odd_alignments))
        }
        results['contribution'] = {
            'all': head_contributions,
            'even_mean': float(np.mean(even_contributions)),
            'odd_mean': float(np.mean(odd_contributions))
        }

        # =========================================================
        # ANALYSIS 2: Per-head direction in head_dim space
        # =========================================================
        print("\n--- ANALYSIS 2: Per-Head Correction Directions ---")

        per_head_corrections = correct_per_head - buggy_per_head
        per_head_norms = torch.norm(per_head_corrections, dim=1)

        even_norms = per_head_norms[EVEN_HEADS].numpy()
        odd_norms = per_head_norms[ODD_HEADS].numpy()

        print(f"\nPer-head correction magnitude (in head_dim space):")
        print(f"  Even heads: mean={np.mean(even_norms):.4f}, std={np.std(even_norms):.4f}")
        print(f"  Odd heads:  mean={np.mean(odd_norms):.4f}, std={np.std(odd_norms):.4f}")

        # =========================================================
        # ANALYSIS 3: Residual stream direction
        # =========================================================
        print("\n--- ANALYSIS 3: Residual Stream Analysis ---")

        residual_correction = correct_residual - buggy_residual
        residual_correction_norm = torch.norm(residual_correction)

        print(f"Residual stream correction norm: {residual_correction_norm:.4f}")

        # How much does each head contribute to residual correction?
        residual_alignments = []
        for head_idx in range(self.n_heads):
            start = head_idx * self.head_dim
            end = (head_idx + 1) * self.head_dim
            head_oproj = o_proj[:, start:end].float()

            head_diff = (correct_per_head[head_idx] - buggy_per_head[head_idx]).float()
            head_contrib_to_residual = torch.matmul(head_oproj, head_diff)

            alignment = torch.nn.functional.cosine_similarity(
                head_contrib_to_residual.unsqueeze(0),
                residual_correction.float().unsqueeze(0)
            ).item()
            residual_alignments.append(alignment)

        even_res_align = [residual_alignments[i] for i in EVEN_HEADS]
        odd_res_align = [residual_alignments[i] for i in ODD_HEADS]

        print(f"\nAlignment with residual correction:")
        print(f"  Even heads: mean={np.mean(even_res_align):.4f}, std={np.std(even_res_align):.4f}")
        print(f"  Odd heads:  mean={np.mean(odd_res_align):.4f}, std={np.std(odd_res_align):.4f}")

        results['residual_alignment'] = {
            'all': residual_alignments,
            'even_mean': float(np.mean(even_res_align)),
            'odd_mean': float(np.mean(odd_res_align))
        }

        # =========================================================
        # Per-head breakdown
        # =========================================================
        print("\n--- PER-HEAD BREAKDOWN ---")
        print(f"{'Head':<6} {'Type':<5} {'Align':<10} {'Contrib':<10} {'Res Align':<10}")
        print("-" * 50)
        for h in range(self.n_heads):
            htype = 'Even' if h % 2 == 0 else 'Odd'
            print(f"H{h:<4} {htype:<5} {head_alignments[h]:<10.4f} {head_contributions[h]:<10.4f} {residual_alignments[h]:<10.4f}")

        # Save results
        with open('direction_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        self.write_markdown(results, head_alignments, head_contributions, residual_alignments)

        return results

    def write_markdown(self, results, alignments, contributions, res_alignments):
        even_align = results['alignment']['even_mean']
        odd_align = results['alignment']['odd_mean']
        even_contrib = results['contribution']['even_mean']
        odd_contrib = results['contribution']['odd_mean']
        even_res = results['residual_alignment']['even_mean']
        odd_res = results['residual_alignment']['odd_mean']

        md = f"""# Output Direction Analysis Results

**Model**: {results['model']}
**Layer**: {results['layer']}
**Date**: {results['timestamp']}

---

## Summary

| Metric | Even Heads | Odd Heads | Difference |
|--------|------------|-----------|------------|
| Alignment with correction | {even_align:.4f} | {odd_align:.4f} | {even_align - odd_align:+.4f} |
| Contribution in correction dir | {even_contrib:.4f} | {odd_contrib:.4f} | {even_contrib - odd_contrib:+.4f} |
| Alignment with residual correction | {even_res:.4f} | {odd_res:.4f} | {even_res - odd_res:+.4f} |

---

## Method

1. Compute "correction vector" = correct_output - buggy_output
2. For each head, compute its contribution through o_proj
3. Measure alignment (cosine similarity) with correction vector
4. Measure contribution magnitude in correction direction

---

## Interpretation

"""
        if abs(even_align - odd_align) > 0.1:
            if even_align > odd_align:
                md += "**Even heads are MORE aligned with the correction direction!**\n"
            else:
                md += "**Odd heads are MORE aligned with the correction direction!**\n"
        else:
            md += "**Even and odd heads have similar alignment with correction direction.**\n"

        md += f"""
---

## Per-Head Data

| Head | Type | Alignment | Contribution | Residual Align |
|------|------|-----------|--------------|----------------|
"""
        for h in range(32):
            htype = 'Even' if h % 2 == 0 else 'Odd'
            md += f"| H{h} | {htype} | {alignments[h]:.4f} | {contributions[h]:.4f} | {res_alignments[h]:.4f} |\n"

        md += f"""
---

## One-Slide Summary

```
OUTPUT DIRECTION ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━

Question: Do even heads output in a "correction direction"
          that odd heads don't?

Alignment with correction vector:
  Even heads: {even_align:.4f}
  Odd heads:  {odd_align:.4f}
  Difference: {even_align - odd_align:+.4f}

Contribution in correction direction:
  Even heads: {even_contrib:.4f}
  Odd heads:  {odd_contrib:.4f}
  Difference: {even_contrib - odd_contrib:+.4f}

Conclusion: {"DIFFERENT!" if abs(even_align - odd_align) > 0.1 else "Similar (no clear difference)"}
```
"""

        with open('direction_analysis_results.md', 'w') as f:
            f.write(md)

        print("\nResults written to: direction_analysis_results.md")


def main():
    analyzer = DirectionAnalysis()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
