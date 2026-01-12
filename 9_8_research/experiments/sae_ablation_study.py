#!/usr/bin/env python3
"""
Experiment 5: SAE Feature Ablation Study

Core idea: For each of the 32 attention heads, measure what happens to
SAE features when you "turn off" that head.

This gives CAUSAL evidence for head→feature relationships.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE
from datetime import datetime
import json
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LAYER_IDX = 10
DEVICE = "cuda"

EVEN_HEADS = list(range(0, 32, 2))
ODD_HEADS = list(range(1, 32, 2))

PROMPT = "Q: Which is bigger: 9.8 or 9.11?\nA:"


class SAEAblationStudy:
    def __init__(self):
        print(f"Loading model: {MODEL_NAME}")
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

        print(f"Loading SAE for layer {LAYER_IDX}...")
        # Try different SAE sources
        try:
            self.sae = SAE.from_pretrained(
                release="llama_scope_lxm_8x",
                sae_id=f"l{LAYER_IDX}m_8x",
                device=DEVICE
            )[0]
            print("  Loaded from llama_scope_lxm_8x")
        except Exception as e:
            print(f"  llama_scope failed: {e}")
            # Try alternative
            try:
                self.sae = SAE.from_pretrained(
                    release="sae_bench_llama31_8b_layer10_width_2pow14_date_0901",
                    sae_id="layer_10/width_16k/average_l0_68",
                    device=DEVICE
                )[0]
                print("  Loaded from sae_bench")
            except Exception as e2:
                print(f"  sae_bench failed: {e2}")
                raise RuntimeError("Could not load any SAE for layer 10")

        self.hooks = []

    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_sae_features(self, prompt: str, ablate_head: int = None):
        """Get SAE features, optionally ablating a head."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)

        mlp_output = {}

        def capture_mlp(module, input, output):
            mlp_output['out'] = output.detach()

        def ablate_attention(module, input, output):
            if ablate_head is None:
                return output

            hidden = output[0]
            batch, seq_len, hidden_size = hidden.shape
            hidden_reshaped = hidden.view(batch, seq_len, self.n_heads, self.head_dim)
            hidden_reshaped[:, :, ablate_head, :] = 0
            hidden = hidden_reshaped.view(batch, seq_len, hidden_size)
            return (hidden,) + output[1:]

        attn_hook = self.model.model.layers[LAYER_IDX].self_attn.register_forward_hook(ablate_attention)
        mlp_hook = self.model.model.layers[LAYER_IDX].mlp.register_forward_hook(capture_mlp)
        self.hooks.extend([attn_hook, mlp_hook])

        try:
            with torch.no_grad():
                _ = self.model(**inputs)

            mlp_last = mlp_output['out'][0, -1, :].unsqueeze(0)

            with torch.no_grad():
                features = self.sae.encode(mlp_last).squeeze(0)

            return features.cpu()
        finally:
            self.clear_hooks()

    def run_ablation_study(self, prompt: str, top_k: int = 100):
        """Run ablation study."""
        print(f"\nPrompt: {prompt[:50]}...")

        print("  Getting baseline features...")
        baseline = self.get_sae_features(prompt, ablate_head=None)

        # Get top-K active features
        top_vals, top_indices = torch.topk(baseline, k=top_k)
        top_indices = top_indices.numpy()
        print(f"  Top {top_k} features identified (max activation: {top_vals[0]:.2f})")

        # Ablate each head
        print("  Ablating heads...")
        effects = np.zeros((self.n_heads, top_k))

        for head_idx in range(self.n_heads):
            ablated = self.get_sae_features(prompt, ablate_head=head_idx)

            for i, feat_idx in enumerate(top_indices):
                effects[head_idx, i] = (baseline[feat_idx] - ablated[feat_idx]).item()

            if (head_idx + 1) % 8 == 0:
                print(f"    {head_idx + 1}/32 heads done")

        return effects, top_indices, baseline[top_indices].numpy()

    def classify_features(self, effects, top_indices, baseline_vals, threshold=1.5):
        """Classify features as even/odd dominated."""
        classifications = []

        for i, feat_idx in enumerate(top_indices):
            even_effect = np.mean(np.abs(effects[EVEN_HEADS, i]))
            odd_effect = np.mean(np.abs(effects[ODD_HEADS, i]))

            if even_effect > threshold * odd_effect:
                category = "EVEN"
            elif odd_effect > threshold * even_effect:
                category = "ODD"
            else:
                category = "Mixed"

            classifications.append({
                'feature_idx': int(feat_idx),
                'baseline': float(baseline_vals[i]),
                'even_effect': float(even_effect),
                'odd_effect': float(odd_effect),
                'ratio': float(even_effect / odd_effect) if odd_effect > 0.001 else 999,
                'category': category
            })

        return classifications

    def run_analysis(self, top_k=100):
        """Run full analysis."""
        results = {
            'model': MODEL_NAME,
            'layer': LAYER_IDX,
            'prompt': PROMPT,
            'timestamp': datetime.now().isoformat()
        }

        print("\n" + "="*70)
        print("EXPERIMENT 5: SAE FEATURE ABLATION STUDY")
        print("="*70)

        effects, top_indices, baseline_vals = self.run_ablation_study(PROMPT, top_k)
        classifications = self.classify_features(effects, top_indices, baseline_vals)

        # Count
        even_dom = [c for c in classifications if c['category'] == 'EVEN']
        odd_dom = [c for c in classifications if c['category'] == 'ODD']
        mixed = [c for c in classifications if c['category'] == 'Mixed']

        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print(f"\n  EVEN-dominated: {len(even_dom)} features")
        print(f"  ODD-dominated:  {len(odd_dom)} features")
        print(f"  Mixed:          {len(mixed)} features")

        results['summary'] = {
            'even_count': len(even_dom),
            'odd_count': len(odd_dom),
            'mixed_count': len(mixed)
        }

        # Top even-dominated
        print("\n--- TOP EVEN-DOMINATED FEATURES ---")
        even_sorted = sorted(even_dom, key=lambda x: x['ratio'], reverse=True)[:10]
        for c in even_sorted:
            print(f"  Feature {c['feature_idx']:5d}: ratio={c['ratio']:.1f}x, even={c['even_effect']:.3f}, odd={c['odd_effect']:.3f}")

        # Top odd-dominated
        print("\n--- TOP ODD-DOMINATED FEATURES ---")
        odd_sorted = sorted(odd_dom, key=lambda x: x['odd_effect']/max(x['even_effect'], 0.001), reverse=True)[:10]
        for c in odd_sorted:
            odd_ratio = c['odd_effect'] / max(c['even_effect'], 0.001)
            print(f"  Feature {c['feature_idx']:5d}: ratio={odd_ratio:.1f}x, even={c['even_effect']:.3f}, odd={c['odd_effect']:.3f}")

        # Save
        results['classifications'] = classifications
        results['effects'] = effects.tolist()

        with open('sae_ablation_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        self.write_markdown(results, classifications)

        return results

    def write_markdown(self, results, classifications):
        even_dom = [c for c in classifications if c['category'] == 'EVEN']
        odd_dom = [c for c in classifications if c['category'] == 'ODD']
        mixed = [c for c in classifications if c['category'] == 'Mixed']
        total = len(classifications)

        md = f"""# SAE Feature Ablation Study Results

**Model**: {results['model']}
**Layer**: {results['layer']}
**Prompt**: `{results['prompt']}`
**Date**: {results['timestamp']}

---

## Summary

| Category | Count | Percentage |
|----------|-------|------------|
| EVEN-dominated | {len(even_dom)} | {100*len(even_dom)/total:.1f}% |
| ODD-dominated | {len(odd_dom)} | {100*len(odd_dom)/total:.1f}% |
| Mixed | {len(mixed)} | {100*len(mixed)/total:.1f}% |

---

## Method

For each head h (0 to 31):
1. Run model normally → MLP output → SAE encode → baseline features
2. Zero out head h → MLP output → SAE encode → ablated features
3. effect[h] = baseline - ablated

Classification (threshold = 1.5x):
- EVEN-dominated: mean(|even effects|) > 1.5 × mean(|odd effects|)
- ODD-dominated: mean(|odd effects|) > 1.5 × mean(|even effects|)

---

## Top EVEN-Dominated Features

| Feature | Baseline | Even Effect | Odd Effect | Ratio |
|---------|----------|-------------|------------|-------|
"""
        even_sorted = sorted(even_dom, key=lambda x: x['ratio'], reverse=True)
        for c in even_sorted[:15]:
            md += f"| {c['feature_idx']} | {c['baseline']:.2f} | {c['even_effect']:.4f} | {c['odd_effect']:.4f} | {c['ratio']:.1f}x |\n"

        md += """
## Top ODD-Dominated Features

| Feature | Baseline | Even Effect | Odd Effect | Ratio |
|---------|----------|-------------|------------|-------|
"""
        odd_sorted = sorted(odd_dom, key=lambda x: x['odd_effect']/max(x['even_effect'],0.001), reverse=True)
        for c in odd_sorted[:15]:
            ratio = c['odd_effect']/max(c['even_effect'], 0.001)
            md += f"| {c['feature_idx']} | {c['baseline']:.2f} | {c['even_effect']:.4f} | {c['odd_effect']:.4f} | {ratio:.1f}x |\n"

        md += f"""
---

## One-Slide Summary

```
SAE ABLATION STUDY: Which heads drive which features?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Method: Zero each head → measure SAE feature change

Results (top 100 features):
  EVEN-dominated: {len(even_dom)} features ({100*len(even_dom)/total:.0f}%)
  ODD-dominated:  {len(odd_dom)} features ({100*len(odd_dom)/total:.0f}%)
  Mixed:          {len(mixed)} features ({100*len(mixed)/total:.0f}%)

This is CAUSAL: ablating even heads changes different
features than ablating odd heads.
```
"""

        with open('sae_ablation_results.md', 'w') as f:
            f.write(md)

        print("\nResults written to: sae_ablation_results.md")


def main():
    study = SAEAblationStudy()
    study.run_analysis(top_k=100)


if __name__ == "__main__":
    main()
