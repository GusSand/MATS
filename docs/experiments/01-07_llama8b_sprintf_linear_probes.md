# Linear Probes & Circuit Analysis: sprintf vs snprintf Security Decision

**Date**: 2026-01-07
**Model**: meta-llama/Meta-Llama-3.1-8B-Instruct
**Experiment**: Linear probing and IOI-style circuit analysis for security code generation

## Research Question

Can we identify a sparse "security circuit" (analogous to IOI's Name Mover circuit) that is responsible for the model's decision to use secure `snprintf` vs insecure `sprintf`?

## Methods

### 1. Linear Probes
- **Probe A (Context)**: Classify secure vs neutral prompt from activations
- **Probe B (Behavior)**: Predict snprintf vs sprintf output from activations
- Trained logistic regression at each layer (0-31)
- 200 samples for context, 184 samples for behavior

### 2. Attention Pattern Analysis (IOI-style)
- Measured attention from final token to security keywords ("WARNING", "snprintf", "buffer", "overflow")
- Identified heads with highest attention to security tokens
- Compared attention patterns between secure and neutral prompts

### 3. Causal Verification of Candidate Heads
Three tests on the 8 heads with highest security-token attention:
- **Ablation**: Zero out head → measure P(snprintf) drop
- **Path Patching**: Patch head from secure→neutral → measure P(snprintf) lift
- **Output Analysis**: Compare head output vectors between conditions

---

## Results

### Experiment 1: Linear Probes

| Probe | Layer 0 | Layer 15 | Layer 31 | Pattern |
|-------|---------|----------|----------|---------|
| Context (secure vs neutral) | 100% | 100% | 100% | Flat - perfect everywhere |
| Behavior (snprintf vs sprintf) | 91.9% | 91.9% | 91.9% | Flat - high everywhere |

**Finding**: Both context and behavior are linearly decodable from Layer 0 onward. No layer-specific emergence pattern.

**Interpretation**: The security context creates an immediately distinguishable representation. The information is PRESENT early, but (from patching experiments) must be PROCESSED through all layers to influence output.

---

### Experiment 2: Attention Pattern Analysis

**Top heads by attention to security tokens (from final position):**

| Rank | Head | Attention to Security Tokens |
|------|------|------------------------------|
| 1 | L20H24 | 61.1% |
| 2 | L25H13 | 47.7% |
| 3 | L17H29 | 44.0% |
| 4 | L16H8 | 42.2% |
| 5 | L22H14 | 40.8% |

**Layer distribution of top 15 heads:**
- Layer 0: 1 head
- Layers 16-17: 4 heads
- Layers 20-22: 5 heads
- Layers 24-26: 4 heads

---

### Experiment 3: Causal Verification

#### Ablation Results (on secure prompt)

| Head | P(snprintf) After Ablation | Drop |
|------|---------------------------|------|
| Baseline | 37.08% | - |
| L17H29 | 34.95% | +5.7% |
| L25H13 | 35.72% | +3.7% |
| L16H8 | 36.32% | +2.0% |
| L20H24 | 37.30% | -0.6% |
| **All 8 heads** | 33.89% | **+8.6%** |

#### Path Patching Results (secure → neutral)

| Head | P(snprintf) After Patch | Lift (% of gap) |
|------|------------------------|-----------------|
| Baseline neutral | 3.21% | - |
| L17H29 | 3.37% | +0.5% |
| L17H25 | 3.39% | +0.5% |
| L25H13 | 3.30% | +0.3% |
| **All 8 heads** | 3.69% | **+1.4%** |

#### Output Similarity (secure vs neutral)

| Head | Cosine Similarity | Interpretation |
|------|-------------------|----------------|
| L0H11 | 0.995 | Nearly identical (not causal) |
| L20H24 | 0.571 | Very different |
| L25H13 | 0.746 | Different |
| L17H29 | 0.833 | Different |

#### Causal Verdict Summary

| Head | Ablation | Patching | Output Diff | Verdict |
|------|----------|----------|-------------|---------|
| L17H29 | +5.7% ✓ | +0.5% | Different | ✅ VERIFIED |
| L25H13 | +3.7% | +0.3% | Different | 🟡 MAYBE |
| L20H24 | -0.6% | +0.1% | Different | 🟡 MAYBE |
| L0H11 | -1.8% | +0.0% | Same | ❌ NOT CAUSAL |

---

## Key Findings

### 1. Attention ≠ Causation

**L20H24** attends 61% to security tokens but has **zero causal impact**. High attention does not imply causal relevance.

### 2. No Sparse Circuit Exists

Unlike IOI's clean circuit of 26 heads, the security decision is **distributed**:

| Metric | IOI (Name Mover) | Security Context |
|--------|------------------|------------------|
| Core heads | ~26 identified | None sufficient |
| Path patching | High lift from few heads | 1.4% from top 8 |
| Ablation | Large effects | Small effects |
| Circuit structure | Clear functional groups | Diffuse |

### 3. Distribution Characteristics

**Across layers (vertical):**
- Layers 0-15: 7% of effect
- Layers 16-31: 95% of effect
- All 32 layers needed for 100%

**Within layers (horizontal):**
- Even heads (512 total): 46% effect
- Odd heads (512 total): 28% effect
- Top 8 heads: 1.4% effect

**Interpretation**: This is "many heads doing a little bit" across late layers, not a sparse circuit.

### 4. Information vs Processing

| Aspect | When Available |
|--------|----------------|
| Information PRESENT | Layer 0 (linear probe 100%) |
| Information USED | Requires all 32 layers (patching) |

The model can "see" the security context immediately but needs full network depth to act on it.

---

## Comparison to Prior Work

### vs IOI (Wang et al. 2022)

| Aspect | IOI | Security Context |
|--------|-----|------------------|
| Task | Indirect object identification | Security-aware code generation |
| Circuit size | 26 heads, 7 functional classes | No identifiable circuit |
| Interpretability | High - clear roles | Low - distributed |
| Intervention | Targeted heads sufficient | Full network needed |

### Possible Explanations

1. **Task complexity**: IOI is syntactic (subject/object tracking). Security awareness may be a higher-level concept requiring distributed processing.

2. **Training signal**: IOI task is implicit in natural text. "Use secure functions" may be learned from diverse sources (docs, code review comments, security advisories).

3. **Representation type**: IOI uses positional/syntactic features. Security may be represented as a "soft instruction" that modulates behavior globally.

---

## Files Generated

### Scripts
- [01_collect_activations.py](../../src/experiments/01-07_llama8b_sprintf_linear_probes/01_collect_activations.py) - Activation collection
- [02_train_probes.py](../../src/experiments/01-07_llama8b_sprintf_linear_probes/02_train_probes.py) - Linear probe training
- [03_attention_patterns.py](../../src/experiments/01-07_llama8b_sprintf_linear_probes/03_attention_patterns.py) - Attention analysis
- [04_verify_security_heads.py](../../src/experiments/01-07_llama8b_sprintf_linear_probes/04_verify_security_heads.py) - Causal verification

### Results
- `results/probe_results_*.json` - Linear probe accuracies
- `results/attention_analysis_*.json` - Attention patterns
- `results/head_verification_*.json` - Causal verification results
- `results/*.png` - Visualizations

---

## Conclusions

1. **Negative result for circuit identification**: No sparse "security circuit" analogous to IOI exists for the sprintf/snprintf decision.

2. **Positive finding about representation**: Security context is immediately linearly decodable but requires distributed processing to influence behavior.

3. **Methodological lesson**: Attention patterns are unreliable indicators of causal importance. Causal verification (ablation + path patching) is essential.

4. **Implication for interpretability**: Some model behaviors may be fundamentally distributed and resist sparse circuit analysis. This doesn't mean they're uninterpretable, but may require different techniques (e.g., probing directions, distributed interventions).

---

## Next Steps (Potential)

- [ ] Logit lens analysis - track when snprintf probability emerges
- [ ] Difference-in-means direction - find the "security direction" in activation space
- [ ] MLP neuron analysis - check if signal is in MLPs rather than attention
- [ ] Sparse probing - L1-regularized probe to find minimal features
- [ ] Test on simpler security behaviors that might have sparser circuits
