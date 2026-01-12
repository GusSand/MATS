# Output Direction Analysis Results

**Model**: meta-llama/Meta-Llama-3.1-8B-Instruct
**Layer**: 10
**Date**: 2026-01-07T00:11:27.870635

---

## Summary

| Metric | Even Heads | Odd Heads | Difference |
|--------|------------|-----------|------------|
| Alignment with correction | -0.0050 | 0.0020 | -0.0070 |
| Contribution in correction dir | -0.0005 | 0.0000 | -0.0005 |
| Alignment with residual correction | 0.0032 | 0.0000 | +0.0032 |

---

## Method

1. Compute "correction vector" = correct_output - buggy_output
2. For each head, compute its contribution through o_proj
3. Measure alignment (cosine similarity) with correction vector
4. Measure contribution magnitude in correction direction

---

## Interpretation

**Even and odd heads have similar alignment with correction direction.**

---

## Per-Head Data

| Head | Type | Alignment | Contribution | Residual Align |
|------|------|-----------|--------------|----------------|
| H0 | Even | -0.0739 | -0.0092 | -0.0006 |
| H1 | Odd | -0.0338 | -0.0036 | 0.0071 |
| H2 | Even | 0.0235 | 0.0026 | -0.0178 |
| H3 | Odd | -0.0362 | -0.0069 | -0.0062 |
| H4 | Even | 0.0338 | 0.0049 | 0.0192 |
| H5 | Odd | -0.0242 | -0.0037 | -0.0158 |
| H6 | Even | 0.0041 | 0.0007 | -0.0091 |
| H7 | Odd | -0.0247 | -0.0034 | 0.0076 |
| H8 | Even | 0.0324 | 0.0045 | 0.0156 |
| H9 | Odd | 0.0304 | 0.0043 | -0.0064 |
| H10 | Even | 0.0188 | 0.0025 | 0.0101 |
| H11 | Odd | 0.0116 | 0.0018 | -0.0003 |
| H12 | Even | 0.0100 | 0.0013 | 0.0070 |
| H13 | Odd | -0.0132 | -0.0013 | -0.0024 |
| H14 | Even | 0.0159 | 0.0016 | -0.0055 |
| H15 | Odd | -0.0031 | -0.0003 | -0.0109 |
| H16 | Even | -0.0566 | -0.0069 | 0.0090 |
| H17 | Odd | -0.0218 | -0.0037 | 0.0192 |
| H18 | Even | 0.0028 | 0.0003 | 0.0262 |
| H19 | Odd | 0.0203 | 0.0020 | -0.0059 |
| H20 | Even | -0.0283 | -0.0029 | -0.0203 |
| H21 | Odd | 0.0441 | 0.0063 | 0.0291 |
| H22 | Even | -0.0171 | -0.0023 | -0.0021 |
| H23 | Odd | 0.0191 | 0.0028 | 0.0182 |
| H24 | Even | 0.0024 | 0.0004 | -0.0085 |
| H25 | Odd | 0.0164 | 0.0017 | -0.0164 |
| H26 | Even | -0.0112 | -0.0013 | 0.0008 |
| H27 | Odd | -0.0053 | -0.0006 | -0.0062 |
| H28 | Even | 0.0232 | 0.0028 | 0.0031 |
| H29 | Odd | 0.0452 | 0.0044 | -0.0130 |
| H30 | Even | -0.0596 | -0.0068 | 0.0240 |
| H31 | Odd | 0.0069 | 0.0010 | 0.0027 |

---

## One-Slide Summary

```
OUTPUT DIRECTION ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━

Question: Do even heads output in a "correction direction"
          that odd heads don't?

Alignment with correction vector:
  Even heads: -0.0050
  Odd heads:  0.0020
  Difference: -0.0070

Contribution in correction direction:
  Even heads: -0.0005
  Odd heads:  0.0000
  Difference: -0.0005

Conclusion: Similar (no clear difference)
```
