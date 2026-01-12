# O_Proj Weight Analysis Results

**Model**: meta-llama/Meta-Llama-3.1-8B-Instruct
**Layer**: 10
**Date**: 2026-01-06T23:47:18.154402

---

## Key Finding

| Metric | Even Heads | Odd Heads | Difference |
|--------|------------|-----------|------------|
| Frobenius norm | 7.2393 | 7.3154 | -0.0762 |
| Spectral norm | 1.1206 | 1.0554 | +0.0652 |
| Effective rank | 125.0230 | 125.0120 | +0.0111 |

### Similarity Analysis

| Comparison | Mean Similarity |
|------------|-----------------|
| Within Even heads | -0.0082 |
| Within Odd heads | -0.0102 |
| Cross Even-Odd | -0.0223 |
| Adjacent pairs | -0.2648 |

---

## Interpretation

The o_proj weight matrix determines how each head's output contributes to the
final attention output. Key metrics:

1. **Frobenius norm**: Total magnitude of weights - how much influence the head has
2. **Spectral norm**: Maximum amplification factor - how much the head can boost a signal
3. **Effective rank**: How many dimensions the head's output projection uses

---

## Conclusion

**Another null result: O_proj weights are nearly identical for even vs odd heads.**

- Frobenius norm difference: -0.08 (negligible)
- Spectral norm difference: +0.07 (negligible)
- Effective rank difference: +0.01 (negligible)

One interesting observation: **Adjacent even-odd pairs are anti-correlated** (cosine sim = -0.26),
but within-group similarities are near zero, so heads are generally orthogonal to each other.

**The mechanism is NOT in the o_proj weights.**

---

## Raw Per-Head Data

### Frobenius Norms
```
H0:7.094  H1:5.699  H2:7.270  H3:9.586  H4:7.773  H5:8.398  H6:9.797  H7:7.410
H8:7.219  H9:7.449  H10:7.879  H11:8.547  H12:7.398  H13:6.250  H14:5.543  H15:6.637
H16:6.285  H17:10.062  H18:6.441  H19:5.973  H20:6.129  H21:8.148  H22:7.430  H23:7.910
H24:8.453  H25:6.316  H26:7.504  H27:6.375  H28:7.410  H29:5.855  H30:6.203  H31:6.430
```

### Spectral Norms
```
H0:1.287  H1:0.778  H2:1.008  H3:1.303  H4:1.178  H5:1.134  H6:1.826  H7:1.427
H8:1.086  H9:0.956  H10:2.131  H11:1.074  H12:1.093  H13:0.928  H14:0.871  H15:1.050
H16:0.733  H17:1.324  H18:0.758  H19:0.760  H20:0.760  H21:1.110  H22:0.898  H23:1.101
H24:1.046  H25:0.802  H26:0.969  H27:0.887  H28:1.211  H29:1.007  H30:1.076  H31:1.244
```

### Effective Ranks
```
H0:125.477  H1:125.970  H2:125.894  H3:124.943  H4:125.075  H5:124.956  H6:122.549  H7:123.794
H8:124.164  H9:125.461  H10:125.052  H11:125.962  H12:124.792  H13:124.022  H14:124.322  H15:123.197
H16:126.736  H17:125.773  H18:126.700  H19:126.614  H20:126.601  H21:125.427  H22:126.269  H23:125.172
H24:126.097  H25:125.937  H26:125.573  H27:125.480  H28:122.273  H29:123.506  H30:122.795  H31:123.978
```
