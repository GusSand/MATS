# Layer Testing Results

**Model**: meta-llama/Meta-Llama-3.1-8B-Instruct
**Date**: 2026-01-07T00:12:45.701245
**Trials per test**: 10

---

## Results by Layer

| Layer | Even Heads | Odd Heads | All Heads | Even-Odd Diff |
|-------|------------|-----------|-----------|---------------|
| 8 | 0% | 0% | 0% | +0% |
| 9 | 0% | 0% | 0% | +0% |
| 10 | 100% | 0% | 100% | +100% |
| 11 | 0% | 0% | 0% | +0% |
| 12 | 0% | 0% | 0% | +0% |
| 15 | 100% | 0% | 100% | +100% |
| 20 | 0% | 0% | 0% | +0% |

---

## Interpretation

If the even/odd pattern appears at multiple layers → it's a general property
If it only appears at layer 10 → layer 10 is special for this task

---

## One-Slide Summary

```
LAYER TESTING: Is layer 10 special?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Results will show if even/odd pattern is layer-specific]
```
