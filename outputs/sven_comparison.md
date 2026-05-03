# SVEN vs Mean-Difference Steering: Head-to-Head on CodeGen-2B-multi

## Summary

We evaluated SVEN (He & Vechev, 2023) and our mean-difference activation steering on the same model (Salesforce/codegen-2B-multi) using our 105x6 CWE benchmark. SVEN outperforms our steering on this model across 5 of 6 CWEs, with particularly strong results on CWE-119, CWE-134, and CWE-89. However, both methods fail completely on CWE-78 (OS command injection), and our steering fails to improve over the unsteered baseline on 4 of 6 CWEs for CodeGen-2B. Our method achieves substantially better results on Llama-8B, suggesting it is better suited to instruction-following models with richer internal representations.

## Per-CWE Comparison Table (Classifiable Secure Rate)

Classifiable secure rate = secure / (secure + insecure), excluding completions scored as "other".

| CWE | Vulnerability | Lang | Baseline | SVEN | Ours (CodeGen) | Ours (Llama-8B) |
|-----|--------------|------|----------|------|----------------|-----------------|
| CWE-119 | Buffer read overflow | C | 53.3% | **90.5%** | 75.8% | 20.0% |
| CWE-134 | Format string | C | 99.4% | **99.8%** | 99.4% | 74.9% |
| CWE-787 | Buffer write overflow | C | 5.0% | **31.7%** | 5.0% | **52.4%** |
| CWE-89 | SQL injection | Py | 52.0% | **86.0%** | 52.1% | **70.3%** |
| CWE-78 | Command injection | Py | 0.0% | 0.0% | 0.0% | 22.0% |
| CWE-79 | XSS | Py | 56.0% | **62.7%** | 56.0% | 30.5% |

**Bold** = best method for that CWE (excluding baseline-at-ceiling).

### Secure Rate (Total) — includes "other" in denominator

| CWE | Baseline | SVEN | Ours (CodeGen) | Ours (Llama-8B) |
|-----|----------|------|----------------|-----------------|
| CWE-119 | 16.8% | 34.5% | 25.9% | — |
| CWE-134 | 93.2% | 83.4% | 93.2% | — |
| CWE-787 | 2.8% | 16.7% | 2.8% | — |
| CWE-89 | 49.7% | 60.3% | 50.3% | — |
| CWE-78 | 0.0% | 0.0% | 0.0% | — |
| CWE-79 | 45.6% | 36.7% | 45.6% | — |

Note: CWE-134 shows SVEN total rate (83.4%) *lower* than baseline (93.2%) because SVEN shifts some completions from the baseline's "secure" bucket into "other" — the classifiable rate remains near-perfect (99.8%).

## Steering Parameters (Best Alpha per CWE)

| CWE | Emergence Layer | Best Alpha | Effect |
|-----|----------------|------------|--------|
| CWE-119 | 31 | 2.0 | +22.4 pp over baseline (classifiable) |
| CWE-134 | 31 | 0.0 | No improvement (baseline already at ceiling) |
| CWE-787 | 31 | 0.0 | No improvement |
| CWE-89 | 31 | 1.0 | +0.03 pp (negligible) |
| CWE-78 | 31 | 0.0 | No improvement |
| CWE-79 | 31 | 0.0 | No improvement |

All emergence layers converged to layer 31 (of 32 total). This is expected — CodeGen-2B's vocabulary-relevant representations concentrate in the final layers. Our steering only helped meaningfully on CWE-119; on 4/6 CWEs, alpha=0.0 was optimal (i.e., steering hurt performance).

## Methods

### Held constant across all conditions
- **Model**: Salesforce/codegen-2B-multi (2.7B params, 32 layers, 2560 hidden dim)
- **Prompts**: Our 105x6 CWE benchmark (105 prompt pairs per CWE, 6 CWEs)
  - Python CWEs (89, 78, 79): Code-completion format, used as-is
  - C CWEs (119, 134, 787): Converted from instruction format to code-completion stubs
- **Generation**: temperature=0.4, top_p=0.95, max_new_tokens=300
- **Sampling**: 10 completions per prompt (num_return_sequences=10)
- **Scoring**: Per-CWE regex scorers (same as main paper)

### SVEN-specific
- Checkpoint: `trained/2b-prefix/checkpoint-last` (released with repo)
- Mechanism: Learned prefix embeddings injected into KV-cache at all layers
- Control: `control_id=0` (secure mode)
- Trained on: 9 CWEs from their own dataset (includes CWE-089, CWE-078, CWE-079, CWE-787)

### Our steering-specific
- Mechanism: Mean-difference activation vector applied at last token position
- Layer selection: Logit lens (vocabulary projection divergence)
- Validation: Leave-One-Base-Out (LOBO) cross-validation, 7 folds per CWE
- Alpha sweep: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

### What differed
- **Generation params**: SVEN's defaults (temp=0.4, top_p=0.95) differ from our paper's Llama-8B params (temp=0.6, top_p=0.9). We used SVEN's defaults for all CodeGen-2B runs to be fair to their method.
- **Prompt format**: C prompts were mechanically adapted from instruction format to code-completion stubs. Python prompts were used as-is.
- **Llama-8B numbers** use our paper's standard params (temp=0.6, top_p=0.9, Layer 31, Llama-specific chat template).

## Caveats

1. **Scorer mismatch on C CWEs**: CodeGen-2B frequently produces code using functions outside our scorer vocabulary (e.g., `scanf` instead of `gets`/`fgets` for CWE-119). This inflates the "other" rate for C CWEs (62-69%) and makes the classifiable rate less reliable for small-N comparisons.

2. **CWE-78 total failure**: Both baseline CodeGen and all interventions score 0% on command injection. CodeGen-2B-multi always produces `os.system()` for these prompts regardless of steering. This may reflect training data bias rather than a steering limitation.

3. **C prompt adaptation**: The C prompts were mechanically converted from instruction format ("Task: Write a C function...") to code-completion stubs (`void read_input(char* buffer) {`). This is a faithful adaptation but differs from how these prompts were presented to Llama-8B (via chat template).

4. **Model scale confound**: CodeGen-2B (2.7B params) vs Llama-8B (8B params). The Llama-8B numbers are not directly comparable to the CodeGen-2B numbers due to the 3x parameter difference.

5. **SVEN training overlap**: SVEN was trained on CWE-089, CWE-078, CWE-079, and CWE-787 among others. CWE-119 and CWE-134 were not in SVEN's training set (they appear in their "gen_1" generalization set). SVEN's strong CWE-119 result (90.5%) on an unseen CWE is notable.

6. **Our steering on CodeGen-2B is naive**: We applied our Llama-developed methodology (mean-difference at emergence layer) without any CodeGen-specific tuning. The poor results may reflect that CodeGen-2B's GPT-J architecture handles security representations differently than Llama/Mistral, or that 2B parameters is insufficient for robust internal representations of security concepts.

## GPU Time

| Run | GPU-hours |
|-----|-----------|
| SVEN benchmark | 2.2h |
| Baseline CodeGen | 2.2h |
| Our steering (LOBO) | 13.4h |
| **Total** | **17.8h** |

## Files

- SVEN results: `baselines/sven/results/sven_2b_20260502_153631.json`
- Baseline results: `baselines/sven/results/baseline_codegen_2b_20260502_175009.json`
- Steering results: `baselines/sven/results/steering_codegen_2b_20260502_200433.json`
- Adapted prompts: `baselines/sven/adapted_prompts/`
- Scripts: `baselines/sven/run_on_our_benchmark.py`, `run_baseline_codegen.py`, `run_steering_codegen.py`
