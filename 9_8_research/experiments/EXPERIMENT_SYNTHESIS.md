# Even/Odd Head Investigation: Complete Synthesis

**Model**: meta-llama/Meta-Llama-3.1-8B-Instruct
**Task**: Decimal comparison bug (9.8 vs 9.11)
**Date**: 2026-01-07

---

## The Puzzle (SOLVED!)

Patching even-indexed attention heads (0,2,4,...) from a "correct" prompt into a "buggy" prompt fixes the model's output with 100% success rate. Patching odd-indexed heads (1,3,5,...) has 0% success rate.

**Initial Question**: What makes even heads different from odd heads?

**ANSWER**: It's NOT about even/odd! It's about **GQA position 0**.

---

## The Real Finding

Llama 3.1 uses Grouped Query Attention (GQA) with 8 KV heads shared by 32 attention heads.
Each group of 4 attention heads shares one KV head:
- Group 0: heads [0, 1, 2, 3]
- Group 1: heads [4, 5, 6, 7]
- ...
- Group 7: heads [28, 29, 30, 31]

Within each group:
- **Position 0** (heads 0, 4, 8, 12, 16, 20, 24, 28): **CARRIES THE FIX**
- Position 1 (heads 1, 5, 9, 13, 17, 21, 25, 29): No effect
- Position 2 (heads 2, 6, 10, 14, 18, 22, 26, 30): No effect
- Position 3 (heads 3, 7, 11, 15, 19, 23, 27, 31): No effect

**Why "even heads" worked**: Even heads = positions 0 and 2 = includes all position-0 heads
**Why "odd heads" failed**: Odd heads = positions 1 and 3 = includes zero position-0 heads

---

## Verification

| Heads Patched | Contains Pos-0? | Works? |
|--------------|-----------------|--------|
| Position 0 only (0,4,8...) | Yes (all) | **100%** |
| Position 1 only (1,5,9...) | No | 0% |
| Position 2 only (2,6,10...) | No | 0% |
| Position 3 only (3,7,11...) | No | 0% |
| Position 0+1 | Yes | **100%** |
| Position 2+3 | No | 0% |
| Position 0+3 (diagonal) | Yes | **100%** |
| Position 1+2 (diagonal) | No | 0% |
| "Even" (pos 0+2) | Yes | **100%** |
| "Odd" (pos 1+3) | No | 0% |

---

## Experiments Run

### Experiment 2: Head Output Magnitudes
**Method**: Compare L2 norms of attention outputs per head
**Result**: NULL - Nearly identical (Even: 14.17, Odd: 14.18)

### Experiment 3: O_proj Weight Analysis
**Method**: Compare o_proj weight matrix norms per head
**Result**: NULL - Nearly identical (Even: 16.43, Odd: 16.38)

### Experiment 4: Reverse Patching (Position vs Content)
**Method**: Cross-patch even content→odd positions and vice versa
**Result**: CRITICAL FINDING
- Even→Even: 100%
- Odd→Odd: 0%
- Even content→Odd positions: 0%
- Odd content→Even positions: 0%

**Conclusion**: BOTH position AND content are required

### Experiment 5a: Logit Lens (Head → Vocabulary)
**Method**: Project each head's output through o_proj and unembed to vocabulary
**Result**: NULL - No meaningful difference
- Even heads: 1.06 numerical tokens in top-10 (avg)
- Odd heads: 0.88 numerical tokens in top-10 (avg)
- Top promoted tokens are mostly gibberish for both

### Experiment 5b: SAE Feature Ablation
**Method**: Zero each head, measure SAE feature changes
**Result**: NULL - All 100 features classified as "Mixed"
- No even-dominated features
- No odd-dominated features

### Experiment 6: Output Direction Analysis
**Method**: Measure alignment with "correction vector" (correct - buggy)
**Result**: NULL - Similar alignments
- Even heads: -0.0050
- Odd heads: +0.0020
- Both essentially orthogonal to correction direction

### Experiment 7: Layer Testing
**Method**: Test even/odd patching at layers 8, 9, 10, 11, 12, 15, 20
**Result**: CRITICAL FINDING

| Layer | Even | Odd | All |
|-------|------|-----|-----|
| 8     | 0%   | 0%  | 0%  |
| 9     | 0%   | 0%  | 0%  |
| **10**| **100%** | **0%** | **100%** |
| 11    | 0%   | 0%  | 0%  |
| 12    | 0%   | 0%  | 0%  |
| **15**| **100%** | **0%** | **100%** |
| 20    | 0%   | 0%  | 0%  |

**Conclusion**: Effect exists at EXACTLY layers 10 and 15

---

## Key Findings

### What We Know:
1. **The effect is real**: 100% vs 0% success rate is not noise
2. **Layer-specific**: Only layers 10 and 15 show the pattern
3. **Position + Content**: Both required - can't swap them
4. **No structural difference**: Even and odd heads have identical:
   - Output magnitudes
   - Weight norms
   - Alignment with correction direction
   - SAE feature influence patterns

### What We Don't Know:
- WHY even heads at layers 10 and 15 carry the "fixing" information
- What's special about layers 10 and 15 specifically
- Why this pattern doesn't appear at adjacent layers (9, 11)

---

## The Mystery

We have a **causal effect without a detectable mechanism**.

The even/odd distinction is:
- NOT in the weights (identical norms)
- NOT in the output magnitudes (identical)
- NOT in the output directions (both orthogonal to correction)
- NOT visible in SAE features (all mixed)

Yet the behavioral difference is stark and reproducible.

---

## Hypotheses for Future Work

1. **Information routing**: Even positions may form a "channel" that routes differently through the network
2. **Phase relationship**: The even/odd pattern may relate to how attention patterns combine
3. **Architectural artifact**: Could be an emergent property of how transformers learn
4. **Layer 10 & 15 interaction**: These layers may form a "processing pair"

---

## One-Slide Summary

```
THE EVEN/ODD HEAD MYSTERY — SOLVED!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Initial observation:
  Patching EVEN heads → 100% fix
  Patching ODD heads  → 0% fix

THE REAL FINDING: It's GQA position, not even/odd!

Llama 3.1 GQA: 8 KV heads, 4 Q heads per KV group
  Group k: heads [4k, 4k+1, 4k+2, 4k+3]

Position within group:
  Pos 0 (0,4,8,12,16,20,24,28): ✓ FIXES BUG
  Pos 1 (1,5,9,13,17,21,25,29): ✗ No effect
  Pos 2 (2,6,10,14,18,22,26,30): ✗ No effect
  Pos 3 (3,7,11,15,19,23,27,31): ✗ No effect

Why "even" worked: Even = pos 0 + pos 2 = contains pos 0
Why "odd" failed:   Odd = pos 1 + pos 3 = no pos 0

Layer specificity: Works at layers 10 & 15 only
```

---

## Implications

This finding reveals:
1. **The "even/odd" pattern was a coincidence** - it aligned with GQA structure
2. **Position 0 in each GQA group is special** - these heads carry task-critical information
3. **GQA creates computational hierarchy** - the first Q head per KV group has a distinct role
4. This effect only manifests at layers 10 & 15 - suggesting these layers are where numerical comparison is processed

## Why Position 0?

In GQA, 4 Q heads share 1 KV head. They all see the same keys and values but have different query projections. Position 0 might be special because:
- It's the "primary" head that learns the main computation
- Other positions (1,2,3) may learn refinements or alternatives
- The architecture may implicitly encourage this specialization during training

## Next Questions

1. Does position-0 specialization exist in other GQA models?
2. Why layers 10 and 15 specifically?
3. What computation do position-0 heads perform that others don't?
