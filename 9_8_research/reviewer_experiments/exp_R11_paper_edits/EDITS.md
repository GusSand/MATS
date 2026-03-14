# R11: Paper Edits — Final (All R1-R10 Results Incorporated)

**Status**: FINAL — all experiments complete as of 2026-03-14

## 1. Table 2 Caption Fix

**Current (incorrect)**: "only patching all attention heads in Layer 10 achieves perfect success"

**Corrected**: "Three configurations achieve perfect success: all 32 heads, all 16 even-indexed heads, and subsets of 8 even-indexed heads. The table shows that intervention granularity determines success — too coarse (full layers) or too narrow (single heads) both fail."

## 2. Typo Fixes

- Line 46: "discovered that that" → "discovered that"
- Line 47: "has remarkably precise requirement" → "has a remarkably precise requirement"
- Line 91: "Using logit lens nostalgebraist" → "Using logit lens (nostalgebraist, 2020)"
- Line 112: "the model incoherent repetitive text" → "the model produced incoherent repetitive text"
- Line 113: "exist in in" → "exist in"

## 3. Patching Specification (Section 3.2)

**Add**: "We patch the output of the multi-head attention computation after the W_O projection but before addition to the residual stream. Specifically, we hook `model.model.layers[10].self_attn` using PyTorch's `register_forward_hook`, capturing the first element of the output tuple (shape [batch, seq_len, 4096]). For head-specific patching, we reshape this tensor to [batch, seq_len, 32, 128] and replace only the selected head slices."

**Add (from R6)**: "We verified that `self_attn` patching (attention output only) achieves 100% success, while full decoder layer patching (`layers[10]`) achieves 0%. This confirms the intervention targets the attention computation specifically, not the MLP or residual stream."

## 4. Tone-Down Claims (Updated with R1-R10 results)

| Original | Revised | Evidence |
|---|---|---|
| "provide one of the most comprehensive mechanistic analysis to date" | "provide a detailed mechanistic case study of a single format-dependent failure" | R2: bug unique to exact phrasing |
| "75% reduction in attention computation during inference" | "75% reduction in attention computation for this specific comparison task, though the fix applies only to the original phrasing" | R2: 0/70 other pairs have the bug |
| "architectural design principles" | "observations about head organization in Llama-3.1-8B that do not generalize to other architectures" | Exp 10: Gemma shows no even/odd pattern |
| "Goldilocks principle" (as novel) | "gradual threshold effect consistent with Mueller et al.'s (2024) mediator granularity framework" | R8: 7-even 40% → 8-even 59%, not sharp |
| "even heads process numerical features" | "even-indexed heads are significantly more effective for correction (70% vs 0%, p<0.0001), supporting functional specialization that is combo-dependent" | R1: parity matters; R8: specific combos matter |

## 5. Additional Limitations (Finalized)

- [x] **Single bug instance**: "The format-dependent decimal comparison bug exists only for the exact phrasing 'Q: Which is bigger: 9.8 or 9.11?\nA:' on Llama-3.1-8B-Instruct. Testing 9 prompt variations and 70 decimal pairs found no other instances of format-dependent failure (R2). The bug is more specific than initially characterized."

- [x] **Parity is real but not absolute**: "Random 8-even-head subsets succeed 70% of the time on average, while random odd-only subsets succeed 0% (p<0.0001, R1). However, success is highly combo-dependent — 59% of random 8-even combos achieve ≥80% success, while 40% of 7-even combos do too (R8). The threshold is gradual, not sharp."

- [x] **Single model**: "The even/odd head specialization is specific to Llama-3.1-8B. Gemma-2B shows no such pattern. Pythia-160M shows pure memorization. Cross-model generalization remains untested on larger models."

- [x] **MLP contribution**: "MLP patching at layers 5, 6, and 7 also fixes the bug for the original pair (R3), suggesting the mechanism is not purely attention-mediated. However, the paper's focus on attention is justified by the head-specific granularity of the intervention."

- [x] **SAE limitations**: "Llama-Scope SAEs (TopK, 8x expansion) are too lossy for causal intervention (reconstruction error 19,000-24,000 L2). SAE features identify format-specific activation patterns but cannot faithfully roundtrip activations for patching (R10)."

- [x] **Diff-of-means failure**: "The reviewer-suggested diff-of-means approach (additive steering vector from mean format difference) achieves 0% success at all layers and scales (R7). The bug correction requires precise activation replacement, not additive adjustment."

## 6. Prior Literature Engagement (From R9)

### New Related Work Paragraphs

**Intervention Granularity**: "Our finding that 8 even-indexed heads constitute a sufficient mediator set aligns with Mueller et al.'s (2024) observation that mediator granularity critically determines intervention success. However, our R8 results show the transition is gradual (40%→59%) rather than sharp, consistent with their continuum view. The specific combination of heads matters more than the count alone."

**Alternative Interventions**: "Neuron-level steering approaches (Transluce, 2024) achieve broad but modest error reduction (21% on arithmetic tasks). Our attention patching achieves complete correction but for an extremely narrow failure mode — a single phrasing on a single model. This highlights a fundamental tradeoff between intervention specificity and generality."

**Arithmetic Mechanisms**: "Prior work on arithmetic reasoning in transformers (Stolfo et al., 2023; Hanna et al., 2023) identified attention heads specializing for arithmetic operations. Our work differs in studying a *failure mode* rather than correct computation, and in finding that the failure is format-dependent rather than arithmetic-dependent. The bug appears to be an artifact of training data patterns (R2 memorization evidence) rather than a fundamental limitation of the arithmetic circuitry."

## 7. Figure Updates

### Figure 1: Add Logit Difference Panels (R5)
- Add panel showing logit(8) - logit(11) across layers for both formats
- Simple format: logit difference crosses zero around layer 15, stabilizes positive by layer 25
- Q&A format: logit difference remains negative throughout
- Key divergence layers: 10-15 (where intervention works)

### New Figure: R1 Even vs Odd Head Analysis
- Bar chart: even head subsets vs odd head subsets success rates
- Box plot: distribution of success rates for random 8-even vs 8-odd combos
- Statistical annotation: p<0.0001

### New Figure: R8 Threshold Analysis
- Histogram: success rate distribution for 500 random 7-even and 500 random 8-even combos
- Transition plot: success rate vs number of odd heads added to 7-even base

## 8. Section-by-Section Edit Plan

### Abstract
- Remove "comprehensive mechanistic analysis" → "detailed case study"
- Add: "though the format-dependent bug is highly specific to one phrasing"

### Section 1 (Introduction)
- Soften all generalization claims
- Add cross-model caveat upfront

### Section 3.2 (Intervention)
- Add patching specification from R6
- Add R7 result: diff-of-means fails completely

### Section 4 (Even/Odd Analysis)
- Add R1 results: p<0.0001 for even vs odd
- Add R8 results: gradual threshold, not sharp
- Cite Mueller et al. for mediator granularity

### Section 5 (Generalization)
- Major revision needed: R2 shows bug is unique to exact phrasing
- Add R4 results: 10.9 vs 10.11 has no format-dependent bug
- Reframe from "generalizable mechanism" to "specific but precise finding"

### Section 6 (Related Work)
- Add Mueller et al., Transluce/Monitor, Stolfo et al., Hanna et al.
- See R9 analysis for full framing

### Section 7 (Limitations)
- Expand significantly with all R-experiment findings
- See Section 5 above for full list

### Section 8 (Conclusion)
- Temper claims significantly
- Emphasize the precision of the finding (exact phrasing, exact heads, exact layer) as both a strength (specificity) and limitation (narrowness)
