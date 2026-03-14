# R9: Prior Literature Engagement & Cross-Model Framing

## Purpose
Address reviewer WzrL's critique: "The paper does not engage enough with prior literature on mechanistic interpretability of arithmetic/numerical reasoning."

## 1. Key Prior Work Comparison

### Mueller et al. — "Quest for the Right Mediator" (2024)
**Relevance**: Directly addresses our intervention granularity findings.

| Aspect | Mueller et al. | Our Work |
|---|---|---|
| **Focus** | Identifying minimal mediator sets for causal mediation | Identifying minimal head subsets for bug fixing |
| **Key concept** | "Mediator granularity" — too coarse loses info, too fine has low power | "Goldilocks zone" — too many/few heads both fail |
| **Finding** | Mid-level mediators (attention heads) outperform neurons or full layers | 8 even heads at Layer 10 outperform full layer or single heads |

**How to cite**: "Our finding that 8 even-indexed heads constitute a sufficient mediator set aligns with Mueller et al.'s (2024) observation that mediator granularity critically determines intervention success. However, our discovery that head parity matters extends their framework — the mediator must be not just the right size, but the right *composition*."

**UPDATE based on R8 results**: The threshold is NOT sharp — 40% of 7-even combos and 59% of 8-even combos succeed. This is more consistent with Mueller et al.'s continuum view than a discrete threshold. Revise language from "Goldilocks principle" to "gradual improvement with combo-dependent variance."

### Transluce/Monitor — Neuron Steering (2024)
**Relevance**: Alternative intervention approach for model behavior modification.

| Aspect | Transluce/Monitor | Our Work |
|---|---|---|
| **Method** | Neuron-level steering vectors | Attention output patching |
| **Granularity** | Individual neurons across layers | Head subsets at single layer |
| **Error reduction** | 21% on arithmetic tasks | 100% on the specific 9.8 vs 9.11 bug |
| **Generality** | Broad arithmetic improvement | Extremely narrow (single phrasing) |

**How to cite**: "While neuron steering approaches (Transluce, 2024) achieve modest but broad error reduction (21%), our attention patching achieves complete correction for a specific format-dependent failure. This highlights a tradeoff between intervention specificity and generality."

**UPDATE based on R2 results**: Our intervention's narrowness (only works on exact "9.8 vs 9.11" phrasing) is a significant limitation. The 100% fix rate is less impressive given it applies to exactly one bug instance.

### Stolfo et al. — "Mechanistic Interpretation of Arithmetic Reasoning" (2023)
**Relevance**: Attention head roles in arithmetic.

- Found that specific attention heads in GPT-2 specialize for different arithmetic operations
- Our even/odd head distinction is related but different — it's about format processing, not arithmetic operations
- R1 results (p<0.0001 for even vs odd) support genuine functional specialization, not random grouping

### Hanna et al. — "How does GPT-2 compute greater-than?" (2023)
**Relevance**: Closest methodological comparison — also studies comparison operations.

- Found attention heads that implement ordering/comparison logic
- Our Layer 10 attention may perform a similar function but in the context of decimal parsing
- Key difference: they study a computation the model performs correctly; we study a *failure mode*

### Nanda et al. — "Progress Measures for Grokking" / Mechanistic Interpretability of Modular Addition
**Relevance**: Understanding when/how models learn arithmetic patterns.

- Models can learn structured algorithms (e.g., Fourier features for modular addition)
- Our finding that the bug is essentially memorized to one phrasing (R2) suggests Llama did NOT learn a general decimal comparison algorithm

## 2. Cross-Model Framing

Based on existing cross-model results (Experiment 10):

| Model | Even/Odd Pattern? | Bug Exists? | Notes |
|---|---|---|---|
| Llama-3.1-8B-Instruct | YES (strong) | Yes (single phrasing) | Primary study model |
| Pythia-160M | N/A (pure memorization) | Yes | Too small for head specialization |
| Gemma-2B | NO | Varies | Even/odd pattern does NOT hold |

**Framing**: "The even-head specialization pattern is specific to Llama-3.1-8B's architecture and training. Cross-model testing reveals this is not a universal feature of transformer attention — Gemma-2B shows no such pattern. This limits our claims to architectural observations within a single model family, consistent with the growing evidence that mechanistic findings often do not transfer across architectures (Variengien & Winsor, 2023)."

## 3. Revised Claims (Based on All R-Experiment Results)

### Original Paper Claims → Revised Claims

1. **"provide one of the most comprehensive mechanistic analysis to date"**
   → "provide a detailed mechanistic case study of a format-dependent failure in Llama-3.1-8B"

2. **"75% reduction in attention computation during inference"**
   → "75% reduction in attention computation for this specific comparison task in this model — though the fix applies only to the exact original phrasing (R2)"

3. **"architectural design principles"**
   → "observations about head organization in Llama-3.1-8B that do not generalize to other architectures (Gemma)"

4. **"Goldilocks principle" as novel discovery**
   → "A gradual threshold effect consistent with Mueller et al.'s mediator granularity framework. The 7→8 even-head transition is not sharp (40%→59% success) and is highly combo-dependent (R8)."

5. **"Even heads process numerical features, odd heads process format features"**
   → "Even-indexed heads are significantly more effective for bug correction (70% vs 0%, p<0.0001, R1), supporting functional specialization. However, this pattern is not all-or-nothing — specific combinations matter more than simple parity counts (R8)."

## 4. Suggested Paper Section Structure

### Section 6: Related Work (Revised)

- **Paragraph 1**: Mechanistic interpretability of arithmetic (Stolfo et al., Hanna et al., Nanda et al.)
- **Paragraph 2**: Intervention methods and granularity (Mueller et al., Transluce/Monitor)
- **Paragraph 3**: Cross-model generalization limitations (Variengien & Winsor, our Gemma/Pythia results)
- **Paragraph 4**: Format sensitivity in LLMs (position our bug as an instance of broader format-dependence literature)

## 5. Limitations Section Additions

Based on R1-R8 results, add:

1. **Single effective bug instance**: The format-dependent bug exists only for the exact "9.8 vs 9.11" phrasing. All variations (synonyms, rephrasings, different decimals, 10.9 vs 10.11) either work in both formats or fail in both (R2, R4).

2. **Model-specific pattern**: The even/odd head specialization is specific to Llama-3.1-8B. Gemma-2B shows no such pattern (Exp 10 cross-model).

3. **Gradual threshold**: The 7→8 even-head threshold is gradual (40%→59%), not sharp. Both 7-even and 8-even subsets show high variance in success rates (R8).

4. **Diff-of-means failure**: The reviewer-suggested diff-of-means approach (compute mean activation difference and add as steering vector) completely fails (0% everywhere, R7). This suggests the intervention requires precise activation replacement, not additive correction.

5. **MLP contribution**: MLP patching at several layers (5, 6, 7) also fixes the bug for the original pair (R3). The mechanism is not purely attention-mediated.
