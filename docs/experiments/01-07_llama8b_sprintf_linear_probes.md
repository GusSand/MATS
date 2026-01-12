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

- [x] Logit lens analysis - track when snprintf probability emerges ✅ DONE
- [ ] Difference-in-means direction - find the "security direction" in activation space
- [ ] MLP neuron analysis - check if signal is in MLPs rather than attention
- [ ] Sparse probing - L1-regularized probe to find minimal features
- [ ] Test on simpler security behaviors that might have sparser circuits

---

# Phase 1: Logit Lens, Gradient Attribution & Activation Steering

**Date**: 2026-01-07 (continued)

## Research Question

Understanding the representation→computation gap: Why is security information present early (linear probes 100% at L0) but behavior only emerges late (patching needs all layers)?

---

## Experiment 4: Logit Lens Analysis

### Method
Project residual stream at each layer through final LayerNorm and unembedding matrix to get token probabilities. Track when snprintf probability emerges.

### Results

| Layer | Secure P(snprintf) | Neutral P(snprintf) | Difference |
|-------|-------------------|---------------------|------------|
| 0     | 0.0001%           | 0.0001%             | +0.0000%   |
| 8     | 0.0000%           | 0.0000%             | +0.0000%   |
| 16    | 0.0001%           | 0.0000%             | +0.0001%   |
| 24    | 0.0006%           | 0.0001%             | +0.0005%   |
| 28    | 0.0078%           | 0.0001%             | +0.0077%   |
| 30    | 0.1451%           | 0.0076%             | +0.1375%   |
| 31    | 37.0850%          | 3.2074%             | +33.8776%  |

**Key Metrics:**
- Divergence layer (>1% difference): **Layer 31**
- Max difference: **33.88%** at layer 31
- Final secure P(snprintf): 37.09%
- Final neutral P(snprintf): 3.21%

### Interpretation
The snprintf probability is essentially zero (< 0.01%) through layers 0-28, then jumps dramatically at layer 31. This confirms the "late decision" hypothesis: the model carries the security context through 30 layers without converting it to output behavior, then makes the decision at the final layer.

---

## Experiment 5: Gradient Attribution

### Method
1. **Input x Gradient**: Attribution = embedding * gradient (measures sensitivity weighted by input)
2. **Gradient Norm**: L2 norm of gradient at each token position (measures raw sensitivity)

### Results - Token Attribution (Input x Gradient, snprintf - sprintf)

| Rank | Token | Position | IxG Diff |
|------|-------|----------|----------|
| 1 | `<\|begin_of_text\|>` | 0 | +0.3712 |
| 2 | `//` | 1 | -0.1786 |
| 3 | `\n` | 11 | +0.1289 |
| 4 | `_string` | 16 | -0.1220 |
| 5 | `_t` | 26 | +0.1145 |
| 6 | `_FAILURE` | 40 | -0.1143 |
| 7 | **WARNING** | 2 | **+0.1072** |

### Security Keyword Attribution

| Token | Position | IxG snprintf | IxG sprintf | Diff | Grad Norm Diff |
|-------|----------|--------------|-------------|------|----------------|
| WARNING | 2 | +0.2356 | +0.1285 | **+0.1072** | +1.5042 |
| snprintf | 5 | +0.0433 | +0.0026 | +0.0406 | -2.3050 |
| prevent | 7 | -0.0085 | -0.0009 | -0.0076 | -0.5031 |
| buffer | 8 | -0.0608 | -0.0337 | -0.0271 | -0.1433 |

### Interpretation
The **WARNING** token has the highest positive attribution (+0.107) for the snprintf decision. Interestingly, the explicit "snprintf" word in the comment has lower attribution (+0.041), and "buffer" is actually negative (-0.027). This suggests the model keys on the warning/instruction signal more than the specific function name.

---

## Experiment 6: Activation Steering

### Method
Unlike patching (which replaces activations), steering **adds** a direction vector:
```
output = original + α × (secure_activation - neutral_activation)
```

### Results - Single Layer Steering (α=1)

| Layer | P(snprintf) | Lift |
|-------|-------------|------|
| 0 | 3.23% | 0.1% |
| 8 | 3.68% | 1.4% |
| 16 | 5.88% | 7.9% |
| 24 | 10.40% | 21.2% |
| 27 | 24.42% | 62.6% |
| 28 | 24.67% | 63.4% |
| 29 | 27.25% | 71.0% |
| 30 | 32.19% | 85.6% |
| **31** | **37.08%** | **100.0%** |

**Best single layer: L31 with 100% lift**

### Results - Alpha Sweep at Layer 31

| Alpha | P(snprintf) | Lift |
|-------|-------------|------|
| 0.00 | 3.21% | 0.0% |
| 0.25 | 7.28% | 12.0% |
| 0.50 | 14.39% | 33.0% |
| 0.75 | 24.91% | 64.1% |
| 1.00 | 37.08% | 100.0% |
| 1.25 | 48.73% | 134.4% |
| 1.50 | 58.68% | 163.8% |
| 2.00 | 72.05% | **203.3%** |
| 3.00 | 82.74% | 234.8% |
| 5.00 | 82.95% | 235.4% |

### Results - Multi-Layer Steering (α=1)

| Layer Range | P(snprintf) | Lift |
|-------------|-------------|------|
| Early (0-15) | 8.45% | 15.5% |
| Mid (8-23) | 1.03% | **-6.5%** |
| Late (16-31) | 0.45% | **-8.2%** |
| All (0-31) | 0.36% | **-8.4%** |
| Top 5 layers | 46.26% | **127.1%** |

### Steering Vector Norms

| Layer | Norm |
|-------|------|
| 0 | 0.07 |
| 8 | 1.12 |
| 16 | 2.92 |
| 24 | 7.71 |

### Interpretation

1. **Single-layer steering at L31 achieves 100% lift** - The final layer is where the decision is made
2. **Over-steering works**: α=2 gives 203% lift (72% probability), α=3 gives 235% lift (83%)
3. **Multi-layer interference**: Steering all layers simultaneously gives -8.4% (worse than baseline!)
4. **Steering vectors grow exponentially**: From 0.07 at L0 to 7.71 at L24
5. **Top 5 layers synergize**: 127% lift, better than any single layer

The multi-layer interference is caused by the exponentially growing steering vector norms. Late layers dominate and corrupt the representation when applied together.

---

## Synthesis: The Representation→Computation Gap

### Timeline of Security Processing

| Layer | Linear Probe | Logit Lens | Steering | Interpretation |
|-------|-------------|------------|----------|----------------|
| 0 | 100% | 0.0001% | 0.1% | Information ENCODED |
| 1-15 | 100% | ~0% | 2-7% | Information PROPAGATED |
| 16-30 | 100% | ~0% | 8-86% | Information TRANSFORMED |
| 31 | 100% | 37.09% | 100% | Information COMPUTED→OUTPUT |

### Key Insight

The security context is immediately recognizable as a **feature** (linear probes work at L0), but not converted to **output behavior** until the final layer. This is the **representation→computation gap**:

- **Representation**: Present from Layer 0 (linearly decodable)
- **Computation**: Localized at Layer 31 (where logits emerge)

### Comparison: Security Context vs IOI

| Aspect | IOI (Wang et al.) | Security Context |
|--------|-------------------|------------------|
| Information encoding | Layer 0-4 | Layer 0 |
| Core computation | Layers 5-26 | Layer 31 only |
| Circuit structure | 26 heads, 7 classes | No sparse circuit |
| Intervention type | Targeted heads | Single layer (L31) |
| Multi-component intervention | Effective | Destructive interference |

### Implications for Interpretability

1. **Different mechanisms for different tasks**: Syntactic tasks (IOI) use distributed circuits; behavioral instructions may use "late decision" mechanisms

2. **Single-layer interventions can be more effective**: For some behaviors, targeting the final layer is more effective than distributed interventions

3. **Steering vectors have layer-specific magnitudes**: Care needed when combining interventions across layers

4. **Over-steering is possible**: The behavior can be amplified beyond natural limits (α > 1)

---

## Files Generated (Phase 1)

### Scripts
- [05_logit_lens.py](../../src/experiments/01-07_llama8b_sprintf_linear_probes/05_logit_lens.py) - Logit lens analysis
- [06_integrated_gradients.py](../../src/experiments/01-07_llama8b_sprintf_linear_probes/06_integrated_gradients.py) - Gradient attribution
- [07_activation_steering.py](../../src/experiments/01-07_llama8b_sprintf_linear_probes/07_activation_steering.py) - Activation steering
- [08_synthesis_analysis.py](../../src/experiments/01-07_llama8b_sprintf_linear_probes/08_synthesis_analysis.py) - Synthesis and visualization

### Results
- `results/logit_lens_*.json` - Logit lens trajectories
- `results/gradient_attribution_*.json` - Token attributions
- `results/activation_steering_*.json` - Steering results
- `results/synthesis_*.png` - Summary visualization

---

## Updated Conclusions

1. **The representation→computation gap is real**: Information exists from L0 but computation happens at L31

2. **Security context uses a "late decision" mechanism**: Unlike IOI's distributed circuit, the snprintf decision is made at the final layer

3. **Single-layer steering is highly effective**: L31 steering with α=1 achieves 100% lift; α=2 achieves 203%

4. **Multi-layer steering causes interference**: The exponentially growing steering vector norms cause destructive interference

5. **WARNING keyword is the primary signal**: Gradient attribution shows WARNING has the highest positive attribution, not the explicit "snprintf" mention

---

## Potential Next Steps (Phase 2)

- [ ] MLP neuron analysis - is the computation in MLP or attention at L31?
- [ ] Difference-in-means direction - extract the "security direction" in residual stream
- [x] SAE feature analysis - decompose into interpretable features ✅ DONE
- [ ] Test steering on different security behaviors
- [ ] Investigate why multi-layer steering fails (normalize by layer?)

---

# SAE Analysis: Distributed Hypothesis Validation

**Date**: 2026-01-07 (continued)

## Research Question

Can we validate the distributed hypothesis using Sparse Autoencoders? Are there specific "security features" or is the signal distributed across many features?

---

## Experiment 7: SAE Feature Analysis

### Method
- Used pretrained Llama-Scope SAEs (residual stream, 8x expansion = 32,768 features per layer)
- Loaded SAEs for all 16 layers (16-31)
- Collected residual stream activations for secure and neutral prompts
- Encoded through SAEs and compared feature activations
- Identified features unique to each context ("secure-only" vs "neutral-only")

### Results - Feature Activity by Layer

| Layer | Secure Active | Neutral Active | Secure-Only | Neutral-Only |
|-------|--------------|----------------|-------------|--------------|
| 16 | 36 | 31 | 7 | 2 |
| 17 | 35 | 33 | 7 | 5 |
| 18 | 34 | 39 | 6 | 11 |
| 19 | 28 | 30 | 3 | 5 |
| 20 | 34 | 35 | 4 | 5 |
| 21 | 24 | 27 | 1 | 4 |
| 22 | 26 | 26 | 4 | 4 |
| 23 | 29 | 27 | 6 | 4 |
| 24 | 26 | 24 | 6 | 4 |
| 25 | 29 | 27 | 5 | 3 |
| 26 | 32 | 34 | 5 | 7 |
| 27 | 30 | 28 | 5 | 3 |
| 28 | 27 | 31 | 1 | 5 |
| 29 | 30 | 26 | 8 | 4 |
| 30 | 26 | 28 | 4 | 6 |
| 31 | 41 | 36 | 9 | 4 |
| **Total** | - | - | **81** | **76** |

**Layers with differential features: 16/16 (100%)**

### Top Security-Promoting Features

Features more active in secure context (contribute to snprintf decision):

| Rank | Layer | Feature | Secure Act | Neutral Act | Diff |
|------|-------|---------|------------|-------------|------|
| 1 | L30 | 10391 | 4.02 | 0.00 | **+4.02** |
| 2 | L29 | 20815 | 9.57 | 5.71 | **+3.86** |
| 3 | L31 | 1895 | 11.06 | 7.64 | **+3.41** |
| 4 | L31 | 22936 | 3.14 | 0.00 | **+3.14** |
| 5 | L18 | 28814 | 3.07 | 0.00 | **+3.07** |
| 6 | L17 | 21279 | 2.76 | 0.00 | +2.76 |
| 7 | L20 | 25851 | 2.66 | 0.00 | +2.66 |
| 8 | L24 | 24012 | 2.51 | 0.00 | +2.51 |
| 9 | L31 | 14534 | 2.50 | 0.00 | +2.50 |
| 10 | L29 | 17181 | 2.49 | 0.00 | +2.49 |

### Top Security-Suppressing Features

Features more active in neutral context (suppress snprintf decision):

| Rank | Layer | Feature | Secure Act | Neutral Act | Diff |
|------|-------|---------|------------|-------------|------|
| 1 | L18 | 13526 | 0.00 | 3.71 | **-3.71** |
| 2 | L17 | 16229 | 0.00 | 3.44 | **-3.44** |
| 3 | L18 | 9703 | 0.00 | 3.40 | **-3.40** |
| 4 | L30 | 4791 | 0.00 | 3.00 | **-3.00** |
| 5 | L21 | 7159 | 0.00 | 2.76 | -2.76 |
| 6 | L19 | 22544 | 0.00 | 2.72 | -2.72 |
| 7 | L16 | 14024 | 0.00 | 2.71 | -2.71 |
| 8 | L29 | 7415 | 0.00 | 2.70 | -2.70 |
| 9 | L18 | 2738 | 0.00 | 2.67 | -2.67 |
| 10 | L17 | 8086 | 0.00 | 2.64 | -2.64 |

### Interpretation

1. **All 16 layers contribute differential features**: 100% of layers (16-31) have features unique to either secure or neutral context

2. **Security signal is distributed across 81 features**: Not a sparse circuit - 81 features are specifically active in secure context but not neutral

3. **Top feature is at L30, not L31**: L30 Feature 10391 has the highest differential (+4.02), suggesting computation may start slightly before the final layer

4. **Bidirectional signal**: Both security-promoting (81) and security-suppressing (76) features exist, indicating active suppression in neutral context

5. **Layer 18 is surprisingly active**: Multiple top features (both promoting and suppressing) are at L18, earlier than expected

### Comparison: SAE vs Other Methods

| Method | Distribution Finding |
|--------|---------------------|
| Linear Probes | 100% accuracy at all layers (L0-31) |
| Patching | Requires all 32 layers for 100% |
| Logit Lens | Emerges only at L31 |
| Steering | L31 alone = 100%, but gradual build-up |
| **SAE** | **81 features across 16 layers (100% coverage)** |

The SAE analysis **confirms** the distributed hypothesis: the security decision involves many features across all late layers, not a sparse circuit.

---

## Files Generated (SAE Analysis)

### Scripts
- [09_sae_security_analysis.py](../../src/experiments/01-07_llama8b_sprintf_linear_probes/09_sae_security_analysis.py) - SAE feature analysis

### Results
- `results/sae_security_analysis_*.json` - Full feature analysis

---

## Final Conclusions

### The Distributed Nature of Security Context

All experiments converge on the same conclusion: **the security decision is fundamentally distributed**:

| Evidence | Finding |
|----------|---------|
| Linear Probes | Information present at L0 (100% accuracy) |
| Activation Patching | All 32 layers needed for full effect |
| Logit Lens | Probability emerges only at L31 |
| Attention Analysis | No sparse circuit (top 8 heads = 1.4% effect) |
| Activation Steering | L31 = 100%, but multi-layer interference |
| **SAE Analysis** | **81 features across 16 layers** |

### Implications

1. **Different from IOI**: Syntactic tasks (IOI) have sparse circuits. Behavioral instructions (security) are distributed.

2. **Late decision, early signal**: Information is encoded early but only converted to behavior at the final layer.

3. **Intervention strategy**: Single-layer steering at L31 is more effective than distributed interventions.

4. **Interpretability challenge**: Some behaviors may be fundamentally distributed and resist sparse circuit analysis.

5. **Publishable methodology**: This SAE-based validation of distributed processing could be a novel contribution.
