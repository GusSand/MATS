# Experiment 12: Mistral-7B Linear Probe Layer Sweep (Mechanistic Replication)

**Date**: 2026-02-15
**Model**: mistralai/Mistral-7B-Instruct-v0.3 (fp16, 32 layers, 4096 hidden dim)
**GPU**: A100-80GB
**Datasets**: CWE-787 (C sprintf/snprintf, 105 pairs, 7 base_ids), CWE-89 (Python SQL injection, 105 pairs, 7 base_ids)

---

## Research Question

Does the "hierarchical convergence" pattern observed in Llama-3.1-8B-Instruct (early-encoding of security context by linear probes, late-emergence in generation behavior, invisible to logit lens) generalize to Mistral-7B-Instruct-v0.3?

## Method

- **Probing**: LogisticRegression(max_iter=1000, C=1.0) with StandardScaler
- **Validation**: LOBO cross-validation (7 folds, leave-one-base_id-out)
- **Layers probed**: [0, 4, 8, 12, 16, 20, 24, 28, 31]
- **Additional analyses**: Logit lens (unembedding projection), steering vector norms per layer

## Results

### Probe Accuracy: CWE-787 (C buffer overflow)

| Layer | Accuracy | Std | Vector Norm |
|-------|----------|-----|-------------|
| 0 | 0.8762 | 0.1294 | 0.01 |
| 4 | 0.9429 | 0.0811 | 0.05 |
| 8 | 0.9524 | 0.0431 | 0.13 |
| 12 | 0.8952 | 0.1327 | 0.26 |
| 16 | 0.8619 | 0.1452 | 0.84 |
| 20 | 0.8905 | 0.1065 | 1.29 |
| 24 | 0.8667 | 0.1369 | 1.76 |
| 28 | 0.8667 | 0.1543 | 2.76 |
| 31 | 0.8524 | 0.1552 | 3.80 |

### Probe Accuracy: CWE-89 (SQL injection)

| Layer | Accuracy | Std | Vector Norm |
|-------|----------|-----|-------------|
| 0 | 0.9571 | 0.0610 | 0.00 |
| 4 | 0.9667 | 0.0356 | 0.03 |
| 8 | 0.9952 | 0.0117 | 0.07 |
| 12 | 0.9762 | 0.0294 | 0.12 |
| 16 | 1.0000 | 0.0000 | 0.34 |
| 20 | 0.9905 | 0.0233 | 0.72 |
| 24 | 0.9810 | 0.0350 | 0.95 |
| 28 | 0.9810 | 0.0350 | 1.36 |
| 31 | 0.9857 | 0.0243 | 2.08 |

### Logit Lens Results (Original — BUGGED)

- **CWE-787**: P(snprintf) stays near zero (~0.0001) across all layers for both secure and vulnerable prompts. No meaningful emergence even at final layer.
- **CWE-89**: P(?) stays near zero (~0.001 max at L8) across all layers. No meaningful emergence at final layer.

**NOTE**: These results contained two bugs identified in the Exp 12b investigation (2026-02-17). See corrected results below.

### Logit Lens Results (Corrected — Exp 12b, 2026-02-17)

**Bugs fixed**:
1. **Tokenization splitting**: `"snprintf"` is 2 tokens on Mistral (`["sn", "printf"]`) but 1 token on Llama. The original script tracked P("sn") instead of P("snprintf"). Fixed by tracking P("sprintf") (single token on Mistral) and P("sn") separately.
2. **Chat template mismatch**: Original used chat-templated prompts where the next predicted token is "Here"/"\`\`\`", not code. Fixed by using raw completion-style prompts (matching the Llama reference).

**Architecture note**: Mistral has `tie_word_embeddings: False` (separate lm_head, cosine similarity with embed_tokens ≈ 0.001). The script correctly used `model.lm_head`, so the unembedding matrix was NOT the issue.

**Corrected CWE-787 results (static prompts, P("sn") = first subtoken of "snprintf")**:

| Layer | P("sn") Secure | P("sn") Vulnerable | P("sprintf") Secure | P("sprintf") Vulnerable |
|-------|---------------|-------------------|--------------------|-----------------------|
| 0 | 0.005% | 0.005% | 0.001% | 0.001% |
| 8 | 0.001% | 0.002% | 0.001% | 0.000% |
| 16 | 0.065% | 0.016% | 0.008% | 0.011% |
| 19 | 0.421% | 0.229% | 0.143% | 0.143% |
| 21 | **6.49%** | 2.39% | 0.117% | 0.125% |
| 24 | **13.1%** | 4.93% | 0.215% | 0.516% |
| 28 | **96.4%** | **75.0%** | 0.002% | 0.157% |
| 31 | **34.3%** | **13.6%** | **0.70%** | **2.71%** |

**Key observations**:
- P("sn") shows massive emergence starting at L19, peaking at L28 (96.4% for secure prompt!)
- Secure prompts have higher P("sn") at most layers → model planning "snprintf" output
- P("sprintf") is higher for **vulnerable** prompts at L31 (2.71% vs 0.70%) — correct directional signal
- Unlike Llama's clean single-token jump at L31, Mistral plans the multi-token "sn"→"printf" sequence across middle layers (L21-28), with partial decay by L31

## Key Findings

1. Probes achieve high accuracy from Layer 0 onward (87.6% CWE-787, 95.7% CWE-89), peaking at Layer 8 (CWE-787: 95.2%) and Layer 16 (CWE-89: 100%).
2. This IS consistent with the Llama pattern: early-layer encoding of secure vs insecure context.
3. **Corrected logit lens** shows Mistral DOES exhibit emergence, but with a different mechanism than Llama:
   - **Llama**: Single-token "snprintf" (ID 37546) jumps from ~0.15% → 37% at L31
   - **Mistral**: Multi-token "sn"+"printf" — P("sn") emerges at L21-28 (up to 96.4%), requiring the model to plan multi-token outputs earlier in the forward pass
4. Vector norms increase monotonically from ~0 at Layer 0 to ~3.8 (CWE-787) and ~2.1 (CWE-89) at Layer 31, showing the representations become more separated in later layers even as probe accuracy stays relatively flat.
5. CWE-787 shows more variance across folds (std 0.08-0.15) compared to CWE-89 (std 0.00-0.06), suggesting the Python SQL injection distinction is more uniformly encoded than the C buffer overflow distinction.

## Conclusion

Mistral-7B shows the SAME hierarchical convergence pattern as Llama-3.1-8B: early-layer encoding (probes), late-layer emergence (logit lens). However, the emergence mechanism differs due to tokenization: Llama concentrates the security decision in a single-token probability jump at L31, while Mistral distributes it across layers L21-28 as a multi-token planning process (P("sn") peaks at 96.4% at L28, then decays). This is a genuine cross-architecture mechanistic difference, not just a replication artifact.

## Code

- [01_probe_sweep.py](../../src/experiments/02-15_mistral_probe_sweep/01_probe_sweep.py) — Original experiment (probes + bugged logit lens)
- [02_logit_lens_corrected.py](../../src/experiments/02-15_mistral_probe_sweep/02_logit_lens_corrected.py) — Corrected logit lens (Exp 12b)
- [investigate_logit_lens.py](../../src/experiments/02-15_mistral_probe_sweep/investigate_logit_lens.py) — Bug investigation (Phase 1-3 diagnostic)

## Files Generated

- `results/probe_sweep_results_20260215_223524.json` — Full Exp 12 results (probes + original logit lens)
- `results/logit_lens_corrected_20260217_020743.json` — Corrected logit lens results (Exp 12b)
- `results/activations_CWE-787_20260215_223524.npz` — CWE-787 activations at all layers
- `results/activations_CWE-89_20260215_223524.npz` — CWE-89 activations at all layers
- `results/metadata_CWE-787_20260215_223524.json` — CWE-787 metadata
- `results/metadata_CWE-89_20260215_223524.json` — CWE-89 metadata
