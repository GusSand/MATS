# Head Output Analysis Results

**Model**: meta-llama/Meta-Llama-3.1-8B-Instruct
**Layer**: 10
**Date**: 2026-01-06T23:44:09.784330

## Prompts
- **Buggy**: `Q: Which is bigger: 9.8 or 9.11?
A:`
- **Correct**: `Which is bigger: 9.8 or 9.11?
Answer:`

---

## Key Finding

| Metric | Even Heads | Odd Heads | Difference |
|--------|------------|-----------|------------|
| Output norm (buggy) | 0.3413 | 0.3550 | -0.0137 |
| Output norm (correct) | 0.2976 | 0.3169 | -0.0193 |
| **Output change** | **0.1992** | **0.2000** | **-0.0007** |
| Cosine similarity | 0.8144 | 0.8217 | -0.0073 |

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
    Even heads: 0.1992
    Odd heads:  0.2000

  Cosine Similarity (buggy vs correct):
    Even heads: 0.8144
    Odd heads:  0.8217

CONCLUSION:
  Even and odd heads produce NEARLY IDENTICAL outputs!
  - Same magnitude (diff: -0.01)
  - Same change between contexts (diff: -0.0007)
  - Same similarity (diff: -0.007)

  The mechanism is NOT in the head outputs themselves.
  The causal effect must come from HOW downstream
  layers process even vs odd head outputs differently.
```

---

## Raw Data

### Output Norms (Buggy Prompt)
```
H0:0.319  H1:0.367  H2:0.357  H3:0.369  H4:0.368  H5:0.360  H6:0.316  H7:0.314
H8:0.331  H9:0.364  H10:0.352  H11:0.333  H12:0.337  H13:0.325  H14:0.339  H15:0.326
H16:0.372  H17:0.338  H18:0.318  H19:0.336  H20:0.349  H21:0.358  H22:0.328  H23:0.317
H24:0.342  H25:0.360  H26:0.325  H27:0.326  H28:0.345  H29:0.309  H30:0.363  H31:0.577
```

### Output Norms (Correct Prompt)
```
H0:0.276  H1:0.304  H2:0.285  H3:0.308  H4:0.317  H5:0.302  H6:0.287  H7:0.269
H8:0.284  H9:0.311  H10:0.292  H11:0.282  H12:0.310  H13:0.306  H14:0.302  H15:0.287
H16:0.308  H17:0.293  H18:0.311  H19:0.303  H20:0.304  H21:0.291  H22:0.314  H23:0.309
H24:0.296  H25:0.319  H26:0.287  H27:0.285  H28:0.286  H29:0.260  H30:0.301  H31:0.642
```

### Output Change (|Correct - Buggy|)
```
H0:0.189  H1:0.216  H2:0.187  H3:0.218  H4:0.208  H5:0.206  H6:0.191  H7:0.205
H8:0.222  H9:0.219  H10:0.200  H11:0.204  H12:0.203  H13:0.182  H14:0.204  H15:0.175
H16:0.224  H17:0.192  H18:0.194  H19:0.187  H20:0.185  H21:0.189  H22:0.203  H23:0.209
H24:0.205  H25:0.190  H26:0.169  H27:0.198  H28:0.189  H29:0.184  H30:0.214  H31:0.225
```

### Cosine Similarity
```
H0:0.807  H1:0.808  H2:0.854  H3:0.808  H4:0.825  H5:0.820  H6:0.802  H7:0.763
H8:0.749  H9:0.800  H10:0.824  H11:0.793  H12:0.806  H13:0.835  H14:0.804  H15:0.845
H16:0.799  H17:0.824  H18:0.810  H19:0.835  H20:0.849  H21:0.850  H22:0.802  H23:0.777
H24:0.803  H25:0.850  H26:0.853  H27:0.799  H28:0.837  H29:0.805  H30:0.807  H31:0.937
```
