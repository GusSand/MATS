# Research Journal

## 2026-01-07: sprintf vs snprintf Security Localization

### Prompt
> Run an experiment similar to the 9.8 vs 9.11 mechanistic analysis, but for security code. Have LLaMA-8B complete a C function and determine where the model decides to use sprintf (insecure) vs snprintf (secure).

### Research Question
Where in LLaMA-8B does the model decide to use insecure `sprintf` vs secure `snprintf`?

### Methods
- **Model**: meta-llama/Meta-Llama-3.1-8B-Instruct
- **Task**: C code completion for string formatting function
- **Measurement**: Logit probability shift for snprintf token
- **Technique**: Last-token activation patching across layers/heads

### Results (No Interpretation)

| Experiment | Result |
|------------|--------|
| Baseline without security context | 0% snprintf, 100% sprintf |
| Baseline with security warning | 100% snprintf, 0% sprintf |
| P(snprintf) gap | 33.9% (secure 37.1% vs neutral 3.2%) |
| Single layer patching (best L25) | 6.1% lift |
| All 32 layers last-token patching | 100% lift |
| Layers 16-31 | 94.7% lift |
| Layers 0-15 | 7.0% lift |
| All even heads (32 layers) | 46.1% lift |
| All odd heads (32 layers) | 28.4% lift |

### Interpretation (Claude's)
The security context ("use snprintf for buffer overflow prevention") is encoded as a **distributed representation** across all 32 layers, concentrated at the last token position. This is fundamentally different from the 9.8 decimal bug, which was localized to Layer 10 attention.

This suggests that high-level behavioral instructions (like "use secure code patterns") involve the entire model rather than specific circuits. This has implications for AI safety: complex behavioral properties may be harder to mechanistically interpret/edit than simple processing errors.

### Detailed Report
See: [docs/experiments/01-07_llama8b_security_sprintf_localization.md](experiments/01-07_llama8b_security_sprintf_localization.md)

### Code Location
`src/experiments/01-07_llama8b_sprintf_security/`

---

## 2026-01-07: Linear Probes & IOI-style Circuit Analysis

### Prompt
> Can we identify a sparse "security circuit" (like IOI's Name Mover circuit) for the sprintf/snprintf decision? Use linear probes and attention pattern analysis.

### Research Question
Is there a sparse, identifiable circuit responsible for security-aware code generation, or is it distributed?

### Methods
1. **Linear Probes**: Train logistic regression at each layer to classify context (secure vs neutral) and predict behavior (snprintf vs sprintf)
2. **Attention Pattern Analysis**: Identify heads attending to security tokens ("WARNING", "snprintf", "buffer", "overflow")
3. **Causal Verification**: Test candidate heads via ablation, path patching, and output analysis

### Results (No Interpretation)

**Linear Probes:**
| Probe | Accuracy (all layers) |
|-------|----------------------|
| Context (secure vs neutral) | 100% |
| Behavior (snprintf vs sprintf) | 91.9% |

**Top Attention Heads (to security tokens):**
| Head | Attention | Ablation Drop | Path Patch Lift |
|------|-----------|---------------|-----------------|
| L20H24 | 61.1% | -0.6% | +0.1% |
| L25H13 | 47.7% | +3.7% | +0.3% |
| L17H29 | 44.0% | +5.7% | +0.5% |

**Combined Effects:**
| Intervention | Effect |
|--------------|--------|
| Ablate all 8 top heads | 8.6% drop |
| Patch all 8 top heads | 1.4% lift (vs 33.9% gap) |

### Key Findings (No Interpretation)
1. **Attention ≠ Causation**: L20H24 attends 61% to security tokens but has zero causal impact
2. **No sparse circuit**: Top 8 heads account for only 1.4% of the effect
3. **Only L17H29 verified** as having measurable causal impact (5.7% ablation drop)
4. **Distribution**: ~512 heads across layers 16-31 each contribute small amounts

### Interpretation (Claude's)
Unlike IOI which found a clean circuit of 26 heads in 7 functional classes, the security decision has **no identifiable sparse circuit**. The effect is "many heads doing a little bit" - a diffuse representation rather than a localized circuit.

This is a **negative result for circuit identification** but a **positive finding about representation**: security context is immediately linearly decodable (layer 0) but requires distributed processing across the full network to influence behavior.

**Methodological lesson**: Attention patterns are unreliable indicators of causal importance. Heads that "look at" security tokens may not "use" that information.

### Detailed Report
See: [docs/experiments/01-07_llama8b_sprintf_linear_probes.md](experiments/01-07_llama8b_sprintf_linear_probes.md)

### Code Location
`src/experiments/01-07_llama8b_sprintf_linear_probes/`

---
