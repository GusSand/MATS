#!/usr/bin/env python3
"""
Experiment 3: Analyze o_proj weights for even vs odd heads.

The o_proj combines all head outputs into the final attention output.
If even and odd heads have different weight patterns, that could explain
why patching even heads works but odd heads don't.

Structure of attention:
  1. Each head produces output: [head_dim]
  2. Concatenate all heads: [n_heads * head_dim] = [hidden_size]
  3. o_proj projects: [hidden_size] -> [hidden_size]

The o_proj weight matrix is [hidden_size, hidden_size].
We can decompose it by head: each head's contribution is a [head_dim, hidden_size] slice.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM
from datetime import datetime
import json

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LAYER_IDX = 10
DEVICE = "cuda"

EVEN_HEADS = list(range(0, 32, 2))
ODD_HEADS = list(range(1, 32, 2))


def main():
    print(f"Loading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=DEVICE
    )

    n_heads = 32
    head_dim = model.config.hidden_size // n_heads  # 128
    hidden_size = model.config.hidden_size  # 4096

    print(f"\nModel config:")
    print(f"  n_heads: {n_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  hidden_size: {hidden_size}")

    # Get o_proj weights from layer 10
    o_proj = model.model.layers[LAYER_IDX].self_attn.o_proj.weight.detach().cpu()
    print(f"\no_proj shape: {o_proj.shape}")  # [hidden_size, hidden_size]

    # The o_proj takes concatenated head outputs [n_heads * head_dim]
    # and projects to [hidden_size]
    #
    # We can slice it by head:
    # Head i's contribution: o_proj[:, i*head_dim : (i+1)*head_dim]
    # This is a [hidden_size, head_dim] matrix for each head

    results = {
        'model': MODEL_NAME,
        'layer': LAYER_IDX,
        'timestamp': datetime.now().isoformat()
    }

    print("\n" + "="*60)
    print("EXPERIMENT 3: O_PROJ WEIGHT ANALYSIS")
    print("="*60)

    # Extract per-head weight slices
    head_weights = []
    for h in range(n_heads):
        start = h * head_dim
        end = (h + 1) * head_dim
        hw = o_proj[:, start:end]  # [hidden_size, head_dim]
        head_weights.append(hw)

    # METRIC 1: Frobenius norm of each head's o_proj slice
    print("\n--- METRIC 1: O_proj Weight Norms ---")
    weight_norms = [torch.norm(hw).item() for hw in head_weights]

    even_norms = [weight_norms[i] for i in EVEN_HEADS]
    odd_norms = [weight_norms[i] for i in ODD_HEADS]

    print(f"\nFrobenius norm of o_proj weights per head:")
    print(f"  Even heads: mean={np.mean(even_norms):.4f}, std={np.std(even_norms):.4f}")
    print(f"  Odd heads:  mean={np.mean(odd_norms):.4f}, std={np.std(odd_norms):.4f}")
    print(f"  Difference: {np.mean(even_norms) - np.mean(odd_norms):.4f}")

    results['weight_norms'] = {
        'all': weight_norms,
        'even_mean': float(np.mean(even_norms)),
        'odd_mean': float(np.mean(odd_norms)),
        'diff': float(np.mean(even_norms) - np.mean(odd_norms))
    }

    # METRIC 2: Spectral norm (largest singular value) - measures max amplification
    print("\n--- METRIC 2: Spectral Norm (Max Singular Value) ---")
    spectral_norms = []
    for hw in head_weights:
        # SVD to get largest singular value
        s = torch.linalg.svdvals(hw.float())
        spectral_norms.append(s[0].item())

    even_spectral = [spectral_norms[i] for i in EVEN_HEADS]
    odd_spectral = [spectral_norms[i] for i in ODD_HEADS]

    print(f"\nSpectral norm (max amplification) per head:")
    print(f"  Even heads: mean={np.mean(even_spectral):.4f}, std={np.std(even_spectral):.4f}")
    print(f"  Odd heads:  mean={np.mean(odd_spectral):.4f}, std={np.std(odd_spectral):.4f}")
    print(f"  Difference: {np.mean(even_spectral) - np.mean(odd_spectral):.4f}")

    results['spectral_norms'] = {
        'all': spectral_norms,
        'even_mean': float(np.mean(even_spectral)),
        'odd_mean': float(np.mean(odd_spectral)),
        'diff': float(np.mean(even_spectral) - np.mean(odd_spectral))
    }

    # METRIC 3: Effective rank (how many dimensions the head uses)
    print("\n--- METRIC 3: Effective Rank ---")
    effective_ranks = []
    for hw in head_weights:
        s = torch.linalg.svdvals(hw.float())
        s_norm = s / s.sum()
        entropy = -torch.sum(s_norm * torch.log(s_norm + 1e-10))
        eff_rank = torch.exp(entropy).item()
        effective_ranks.append(eff_rank)

    even_ranks = [effective_ranks[i] for i in EVEN_HEADS]
    odd_ranks = [effective_ranks[i] for i in ODD_HEADS]

    print(f"\nEffective rank (dimension utilization) per head:")
    print(f"  Even heads: mean={np.mean(even_ranks):.4f}, std={np.std(even_ranks):.4f}")
    print(f"  Odd heads:  mean={np.mean(odd_ranks):.4f}, std={np.std(odd_ranks):.4f}")
    print(f"  Difference: {np.mean(even_ranks) - np.mean(odd_ranks):.4f}")

    results['effective_ranks'] = {
        'all': effective_ranks,
        'even_mean': float(np.mean(even_ranks)),
        'odd_mean': float(np.mean(odd_ranks)),
        'diff': float(np.mean(even_ranks) - np.mean(odd_ranks))
    }

    # METRIC 4: Cosine similarity between adjacent even-odd pairs
    print("\n--- METRIC 4: Similarity Between Adjacent Head Pairs ---")
    pair_similarities = []
    for i in range(0, 32, 2):
        even_hw = head_weights[i].flatten()
        odd_hw = head_weights[i+1].flatten()
        cos_sim = torch.nn.functional.cosine_similarity(
            even_hw.unsqueeze(0).float(),
            odd_hw.unsqueeze(0).float()
        ).item()
        pair_similarities.append(cos_sim)

    print(f"\nCosine similarity between adjacent (even, odd) pairs:")
    print(f"  Mean: {np.mean(pair_similarities):.4f}")
    print(f"  Std:  {np.std(pair_similarities):.4f}")
    print(f"  Min:  {np.min(pair_similarities):.4f}")
    print(f"  Max:  {np.max(pair_similarities):.4f}")

    results['pair_similarities'] = {
        'all': pair_similarities,
        'mean': float(np.mean(pair_similarities)),
        'std': float(np.std(pair_similarities))
    }

    # METRIC 5: Average similarity within even vs within odd
    print("\n--- METRIC 5: Within-Group Similarity ---")

    def avg_pairwise_sim(indices):
        sims = []
        for i, idx1 in enumerate(indices):
            for idx2 in indices[i+1:]:
                hw1 = head_weights[idx1].flatten()
                hw2 = head_weights[idx2].flatten()
                sim = torch.nn.functional.cosine_similarity(
                    hw1.unsqueeze(0).float(),
                    hw2.unsqueeze(0).float()
                ).item()
                sims.append(sim)
        return np.mean(sims), np.std(sims)

    even_sim_mean, even_sim_std = avg_pairwise_sim(EVEN_HEADS)
    odd_sim_mean, odd_sim_std = avg_pairwise_sim(ODD_HEADS)

    print(f"\nWithin-group pairwise similarity:")
    print(f"  Even-Even: mean={even_sim_mean:.4f}, std={even_sim_std:.4f}")
    print(f"  Odd-Odd:   mean={odd_sim_mean:.4f}, std={odd_sim_std:.4f}")

    results['within_group_similarity'] = {
        'even_mean': float(even_sim_mean),
        'odd_mean': float(odd_sim_mean)
    }

    # METRIC 6: Cross-group similarity
    cross_sims = []
    for even_idx in EVEN_HEADS:
        for odd_idx in ODD_HEADS:
            hw1 = head_weights[even_idx].flatten()
            hw2 = head_weights[odd_idx].flatten()
            sim = torch.nn.functional.cosine_similarity(
                hw1.unsqueeze(0).float(),
                hw2.unsqueeze(0).float()
            ).item()
            cross_sims.append(sim)

    print(f"  Even-Odd:  mean={np.mean(cross_sims):.4f}, std={np.std(cross_sims):.4f}")

    results['cross_group_similarity'] = {
        'mean': float(np.mean(cross_sims)),
        'std': float(np.std(cross_sims))
    }

    # Per-head breakdown
    print("\n--- PER-HEAD DETAILS ---")
    print(f"{'Head':<6} {'Type':<5} {'Frob Norm':<12} {'Spectral':<12} {'Eff Rank':<10}")
    print("-" * 50)
    for h in range(n_heads):
        htype = 'Even' if h % 2 == 0 else 'Odd'
        print(f"H{h:<4} {htype:<5} {weight_norms[h]:<12.4f} {spectral_norms[h]:<12.4f} {effective_ranks[h]:<10.4f}")

    # Save results
    with open('oproj_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Write markdown
    write_markdown(results, weight_norms, spectral_norms, effective_ranks)

    print("\n" + "="*60)
    print("DONE")
    print("="*60)


def write_markdown(results, weight_norms, spectral_norms, effective_ranks):
    even_frob = results['weight_norms']['even_mean']
    odd_frob = results['weight_norms']['odd_mean']
    even_spec = results['spectral_norms']['even_mean']
    odd_spec = results['spectral_norms']['odd_mean']
    even_rank = results['effective_ranks']['even_mean']
    odd_rank = results['effective_ranks']['odd_mean']

    md = f"""# O_Proj Weight Analysis Results

**Model**: {results['model']}
**Layer**: {results['layer']}
**Date**: {results['timestamp']}

---

## Key Finding

| Metric | Even Heads | Odd Heads | Difference |
|--------|------------|-----------|------------|
| Frobenius norm | {even_frob:.4f} | {odd_frob:.4f} | {even_frob - odd_frob:+.4f} |
| Spectral norm | {even_spec:.4f} | {odd_spec:.4f} | {even_spec - odd_spec:+.4f} |
| Effective rank | {even_rank:.4f} | {odd_rank:.4f} | {even_rank - odd_rank:+.4f} |

### Similarity Analysis

| Comparison | Mean Similarity |
|------------|-----------------|
| Within Even heads | {results['within_group_similarity']['even_mean']:.4f} |
| Within Odd heads | {results['within_group_similarity']['odd_mean']:.4f} |
| Cross Even-Odd | {results['cross_group_similarity']['mean']:.4f} |
| Adjacent pairs | {results['pair_similarities']['mean']:.4f} |

---

## Interpretation

The o_proj weight matrix determines how each head's output contributes to the
final attention output. Key metrics:

1. **Frobenius norm**: Total magnitude of weights - how much influence the head has
2. **Spectral norm**: Maximum amplification factor - how much the head can boost a signal
3. **Effective rank**: How many dimensions the head's output projection uses

---

## Conclusion

[To be filled based on results]

---

## Raw Per-Head Data

### Frobenius Norms
```
{_format_per_head(weight_norms)}
```

### Spectral Norms
```
{_format_per_head(spectral_norms)}
```

### Effective Ranks
```
{_format_per_head(effective_ranks)}
```
"""

    with open('oproj_analysis_results.md', 'w') as f:
        f.write(md)

    print("\nResults written to: oproj_analysis_results.md")


def _format_per_head(arr):
    lines = []
    for i in range(0, 32, 8):
        heads = [f"H{j}:{arr[j]:.3f}" for j in range(i, min(i+8, 32))]
        lines.append("  ".join(heads))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
