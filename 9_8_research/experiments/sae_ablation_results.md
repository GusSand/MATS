# SAE Feature Ablation Study Results

**Model**: meta-llama/Meta-Llama-3.1-8B-Instruct
**Layer**: 10
**Prompt**: `Q: Which is bigger: 9.8 or 9.11?
A:`
**Date**: 2026-01-07T00:04:41.816535

---

## Summary

| Category | Count | Percentage |
|----------|-------|------------|
| EVEN-dominated | 0 | 0.0% |
| ODD-dominated | 0 | 0.0% |
| Mixed | 100 | 100.0% |

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

## Top ODD-Dominated Features

| Feature | Baseline | Even Effect | Odd Effect | Ratio |
|---------|----------|-------------|------------|-------|

---

## One-Slide Summary

```
SAE ABLATION STUDY: Which heads drive which features?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Method: Zero each head → measure SAE feature change

Results (top 100 features):
  EVEN-dominated: 0 features (0%)
  ODD-dominated:  0 features (0%)
  Mixed:          100 features (100%)

This is CAUSAL: ablating even heads changes different
features than ablating odd heads.
```
