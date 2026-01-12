#!/usr/bin/env python3
"""
Experiment 2: Analyze head OUTPUT contributions (not attention patterns).

Key insight: Attention patterns show what heads LOOK at.
But the causal effect comes from what heads WRITE to the residual stream.

This experiment measures the actual output contribution of each head.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import json

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LAYER_IDX = 10
DEVICE = "cuda"

PROMPT_BUGGY = "Q: Which is bigger: 9.8 or 9.11?\nA:"
PROMPT_CORRECT = "Which is bigger: 9.8 or 9.11?\nAnswer:"

EVEN_HEADS = list(range(0, 32, 2))
ODD_HEADS = list(range(1, 32, 2))


def main():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=DEVICE
    )
    model.eval()

    n_heads = 32
    head_dim = model.config.hidden_size // n_heads

    results = {
        'model': MODEL_NAME,
        'layer': LAYER_IDX,
        'timestamp': datetime.now().isoformat(),
        'prompts': {'buggy': PROMPT_BUGGY, 'correct': PROMPT_CORRECT}
    }

    # Capture attention outputs
    def get_head_outputs(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs_dict = {}

        def hook(module, input, output):
            # output[0] is attention output: [batch, seq, hidden]
            outputs_dict['attn_out'] = output[0].detach().cpu()

        hook_handle = model.model.layers[LAYER_IDX].self_attn.register_forward_hook(hook)

        with torch.no_grad():
            _ = model(**inputs)

        hook_handle.remove()

        # Reshape to per-head: [batch, seq, n_heads, head_dim]
        attn_out = outputs_dict['attn_out']
        batch, seq_len, hidden = attn_out.shape
        per_head = attn_out.view(batch, seq_len, n_heads, head_dim)

        return per_head, tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

    print("\n" + "="*60)
    print("EXPERIMENT 2: HEAD OUTPUT ANALYSIS")
    print("="*60)

    # Get outputs for both prompts
    buggy_heads, buggy_tokens = get_head_outputs(PROMPT_BUGGY)
    correct_heads, correct_tokens = get_head_outputs(PROMPT_CORRECT)

    # Analyze at last position (where prediction happens)
    buggy_last = buggy_heads[0, -1, :, :]  # [n_heads, head_dim]
    correct_last = correct_heads[0, -1, :, :]

    # METRIC 1: Output norms (how much each head contributes)
    print("\n--- METRIC 1: Output Norms ---")
    buggy_norms = torch.norm(buggy_last, dim=1).numpy()
    correct_norms = torch.norm(correct_last, dim=1).numpy()

    print(f"\nBuggy prompt - Head output norms:")
    print(f"  Even heads: mean={np.mean(buggy_norms[EVEN_HEADS]):.4f}, std={np.std(buggy_norms[EVEN_HEADS]):.4f}")
    print(f"  Odd heads:  mean={np.mean(buggy_norms[ODD_HEADS]):.4f}, std={np.std(buggy_norms[ODD_HEADS]):.4f}")

    print(f"\nCorrect prompt - Head output norms:")
    print(f"  Even heads: mean={np.mean(correct_norms[EVEN_HEADS]):.4f}, std={np.std(correct_norms[EVEN_HEADS]):.4f}")
    print(f"  Odd heads:  mean={np.mean(correct_norms[ODD_HEADS]):.4f}, std={np.std(correct_norms[ODD_HEADS]):.4f}")

    results['output_norms'] = {
        'buggy': {
            'all': buggy_norms.tolist(),
            'even_mean': float(np.mean(buggy_norms[EVEN_HEADS])),
            'odd_mean': float(np.mean(buggy_norms[ODD_HEADS]))
        },
        'correct': {
            'all': correct_norms.tolist(),
            'even_mean': float(np.mean(correct_norms[EVEN_HEADS])),
            'odd_mean': float(np.mean(correct_norms[ODD_HEADS]))
        }
    }

    # METRIC 2: How much does each head CHANGE between buggy and correct?
    print("\n--- METRIC 2: Output Change (Correct - Buggy) ---")
    diff = correct_last - buggy_last
    diff_norms = torch.norm(diff, dim=1).numpy()

    print(f"\nHow much each head's output changes:")
    print(f"  Even heads: mean={np.mean(diff_norms[EVEN_HEADS]):.4f}, std={np.std(diff_norms[EVEN_HEADS]):.4f}")
    print(f"  Odd heads:  mean={np.mean(diff_norms[ODD_HEADS]):.4f}, std={np.std(diff_norms[ODD_HEADS]):.4f}")

    results['output_change'] = {
        'all': diff_norms.tolist(),
        'even_mean': float(np.mean(diff_norms[EVEN_HEADS])),
        'odd_mean': float(np.mean(diff_norms[ODD_HEADS]))
    }

    # METRIC 3: Cosine similarity (do heads output similar things?)
    print("\n--- METRIC 3: Cosine Similarity (Buggy vs Correct) ---")
    cos_sims = []
    for h in range(n_heads):
        cos = torch.nn.functional.cosine_similarity(
            buggy_last[h].unsqueeze(0).float(),
            correct_last[h].unsqueeze(0).float()
        ).item()
        cos_sims.append(cos)

    cos_sims = np.array(cos_sims)

    print(f"\nHow similar each head's output is between prompts:")
    print(f"  Even heads: mean={np.mean(cos_sims[EVEN_HEADS]):.4f}, std={np.std(cos_sims[EVEN_HEADS]):.4f}")
    print(f"  Odd heads:  mean={np.mean(cos_sims[ODD_HEADS]):.4f}, std={np.std(cos_sims[ODD_HEADS]):.4f}")

    results['cosine_similarity'] = {
        'all': cos_sims.tolist(),
        'even_mean': float(np.mean(cos_sims[EVEN_HEADS])),
        'odd_mean': float(np.mean(cos_sims[ODD_HEADS]))
    }

    # Per-head breakdown
    print("\n--- PER-HEAD DETAILS ---")
    print(f"{'Head':<6} {'Type':<5} {'Buggy Norm':<12} {'Correct Norm':<12} {'Change':<10} {'Cos Sim':<10}")
    print("-" * 60)
    for h in range(n_heads):
        htype = 'Even' if h % 2 == 0 else 'Odd'
        print(f"H{h:<4} {htype:<5} {buggy_norms[h]:<12.4f} {correct_norms[h]:<12.4f} {diff_norms[h]:<10.4f} {cos_sims[h]:<10.4f}")

    # Save JSON
    with open('head_output_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Write markdown report
    write_markdown_report(results, buggy_norms, correct_norms, diff_norms, cos_sims)

    print("\n" + "="*60)
    print("DONE")
    print("="*60)


def write_markdown_report(results, buggy_norms, correct_norms, diff_norms, cos_sims):
    """Write results to markdown file."""

    even_buggy = np.mean(buggy_norms[EVEN_HEADS])
    odd_buggy = np.mean(buggy_norms[ODD_HEADS])
    even_correct = np.mean(correct_norms[EVEN_HEADS])
    odd_correct = np.mean(correct_norms[ODD_HEADS])
    even_change = np.mean(diff_norms[EVEN_HEADS])
    odd_change = np.mean(diff_norms[ODD_HEADS])
    even_cos = np.mean(cos_sims[EVEN_HEADS])
    odd_cos = np.mean(cos_sims[ODD_HEADS])

    md = f"""# Head Output Analysis Results

**Model**: {results['model']}
**Layer**: {results['layer']}
**Date**: {results['timestamp']}

## Prompts
- **Buggy**: `{results['prompts']['buggy']}`
- **Correct**: `{results['prompts']['correct']}`

---

## Key Finding

| Metric | Even Heads | Odd Heads | Difference |
|--------|------------|-----------|------------|
| Output norm (buggy) | {even_buggy:.4f} | {odd_buggy:.4f} | {even_buggy - odd_buggy:+.4f} |
| Output norm (correct) | {even_correct:.4f} | {odd_correct:.4f} | {even_correct - odd_correct:+.4f} |
| **Output change** | **{even_change:.4f}** | **{odd_change:.4f}** | **{even_change - odd_change:+.4f}** |
| Cosine similarity | {even_cos:.4f} | {odd_cos:.4f} | {even_cos - odd_cos:+.4f} |

---

## Interpretation

### What we measured:
1. **Output norm**: How much each head writes to the residual stream (magnitude)
2. **Output change**: How much each head's output differs between buggy vs correct prompt
3. **Cosine similarity**: How similar each head's output is between the two prompts

### Key insight:
- **Attention patterns** show what heads LOOK at
- **Head outputs** show what heads WRITE

The causal effect (even heads fix the bug) comes from what heads **write**, not what they attend to.

---

## One-Slide Summary

```
WHY ARE EVEN HEADS NECESSARY?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CAUSAL FACT (validated):
  Even heads only → 100% correct
  Odd heads only  → 0% correct

MECHANISM (this experiment):
  We measured head OUTPUTS (what they write)
  not attention patterns (what they look at)

  Output Change (correct - buggy):
    Even heads: {even_change:.4f}
    Odd heads:  {odd_change:.4f}

  Cosine Similarity (buggy vs correct):
    Even heads: {even_cos:.4f}
    Odd heads:  {odd_cos:.4f}

CONCLUSION:
  [To be filled based on results]
```

---

## Raw Data

### Output Norms (Buggy Prompt)
```
{_format_per_head(buggy_norms)}
```

### Output Norms (Correct Prompt)
```
{_format_per_head(correct_norms)}
```

### Output Change (|Correct - Buggy|)
```
{_format_per_head(diff_norms)}
```

### Cosine Similarity
```
{_format_per_head(cos_sims)}
```
"""

    with open('head_output_analysis_results.md', 'w') as f:
        f.write(md)

    print("\nResults written to: head_output_analysis_results.md")


def _format_per_head(arr):
    lines = []
    for i in range(0, 32, 8):
        heads = [f"H{j}:{arr[j]:.3f}" for j in range(i, min(i+8, 32))]
        lines.append("  ".join(heads))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
