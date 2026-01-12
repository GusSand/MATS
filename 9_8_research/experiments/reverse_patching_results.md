# Reverse Patching Test Results

**Model**: meta-llama/Meta-Llama-3.1-8B-Instruct
**Layer**: 10
**Date**: 2026-01-06T23:51:27.516758

---

## Question

Why do even heads fix the bug but odd heads don't?

- **Hypothesis A (POSITION)**: Even-indexed positions are processed differently by downstream layers
- **Hypothesis B (CONTENT)**: Even heads produce outputs with specific information that helps

---

## Results

| Test | Source → Target | Success Rate |
|------|-----------------|--------------|
| Baseline Even | Even → Even | 100% |
| Baseline Odd | Odd → Odd | 0% |
| **Cross-patch A** | **Even → Odd** | **0%** |
| **Cross-patch B** | **Odd → Even** | **0%** |
| Sanity Check | All → All | 100% |

---

## Interpretation

**BOTH POSITION and CONTENT matter.**

- Even content in odd positions fails
- Odd content in even positions also fails
- Need both: the RIGHT content in the RIGHT position

---

## One-Slide Summary

```
IS IT POSITION OR CONTENT?
━━━━━━━━━━━━━━━━━━━━━━━━━━

Baseline:
  Even→Even: 100%
  Odd→Odd:   0%

Cross-patch:
  Even content → Odd positions: 0%
  Odd content → Even positions: 0%

CONCLUSION: BOTH position AND content matter!
  - Even content alone doesn't work (needs even position)
  - Even position alone doesn't work (needs even content)
  - There's a tight coupling between head index and content
```
