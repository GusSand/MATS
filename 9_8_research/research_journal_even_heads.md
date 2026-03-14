# Research Journal: Decimal Comparison Bug (9_8_research)

**Paper**: ["Even Heads Fix Odd Errors: Mechanistic Discovery and Surgical Repair in Transformer Attention"](https://arxiv.org/html/2508.19414v1)
**Author**: Gustavo A. Sandoval (NYU Tandon)
**Primary Model**: Llama-3.1-8B-Instruct
**Research Period**: August–September 2025

---

## Experiment 1: Bug Discovery & Verification

**Research Question**: Does Llama-3.1-8B-Instruct exhibit a format-dependent decimal comparison bug?

**Methods**:
- Model: Llama-3.1-8B-Instruct
- Tested 3 prompt formats: Chat template, Q&A ("Q: ... A:"), Simple ("... Answer:")
- Greedy decoding (temperature=0.0)
- n=1000 trials

**Results**:
- Q&A format: 100% error rate (says 9.11 > 9.8)
- Chat format: 95% error rate
- Simple format: 0% error rate
- 100% correlation between format and bug occurrence

**Scripts**: `working_scripts/verify_llama_bug.py`, `working_scripts/format_comparison.py`

---

## Experiment 2: Layer-by-Layer Intervention (Logit Lens & Attribution)

**Research Question**: At which layer does the model commit to the wrong answer?

**Methods**:
- Logit lens: layer-wise vocabulary projections
- Logit attribution: layer contributions to final logits
- All 32 layers tested

**Results**:
- Complete divergence between formats occurs at Layer 25
- Layer 10 identified as the critical intervention point
- Full-layer patching (Layers 20-30) causes incoherent outputs — too coarse

**Scripts**: `experimental/logitlens/`

---

## Experiment 3: Attention Output Patching (Main Breakthrough)

**Research Question**: Can we surgically fix the bug by patching attention outputs at a single layer?

**Methods**:
- Attention output patching using nnsight library
- Replaced attention outputs from buggy format with correct format outputs
- Tested all 32 layers individually
- n=1000 trials for Layer 10

**Results**:
- **Layer 10 attention output patching: 100% success rate** (the only single-layer fix)
- Bidirectional causality confirmed: forward patching fixes bug (100%), reverse patching induces it (100%)
- p-value: 9.33 × 10⁻³⁰²
- Key insight: attention OUTPUTS work, attention WEIGHTS do not

**Scripts**: `working_scripts/bidirectional_patching.py`, `working_scripts/attention_control_experiment.py`
**Documentation**: `experimental/attention/BREAKTHROUGH_FINDINGS.md`

---

## Experiment 4: Even/Odd Head Specialization Discovery

**Research Question**: Which attention heads at Layer 10 are responsible?

**Methods**:
- Systematic testing of all head combinations at Layer 10 (32 heads total)
- Tested even-indexed vs odd-indexed heads separately
- Head subset testing with varying counts

**Results**:
- **Even-indexed heads (0,2,4,...,30)**: 100% success at fixing bug
- **Odd-indexed heads (1,3,5,...,31)**: 0% success
- All 16 even heads are perfectly interchangeable
- Sharp computational threshold: exactly 8 even heads required
  - 7 even heads: 0% success
  - 8 even heads: 100% success (no gradual degradation)

**Scripts**: `working_scripts/`, `repro/head_analysis/`

---

## Experiment 5: Pattern Replacement Threshold

**Research Question**: How much of the attention pattern needs to be replaced for the fix to work?

**Methods**:
- Ablation parameter sweep: 0–100% replacement rates
- n=1000 per threshold level

**Results**:
- Below 60% replacement: 0% success
- At/above 60% replacement: 100% success
- Sharp binary transition — discrete computational modes, not gradual

**Scripts**: `experimental/statistical_validation/comprehensive_validation.py`
**Documentation**: `experimental/statistical_validation/FINAL_RESULTS.md`

---

## Experiment 6: SAE Feature Analysis (Layer 10 Bottleneck)

**Research Question**: What happens at the feature level that makes Layer 10 special?

**Methods**:
- Llama-Scope SAEs with 32,768 features
- Analyzed all 32 layers for feature overlap between formats
- Identified numerical processing features vs format-discriminative features

**Results**:
- Three-phase progression discovered:
  - **Phase 1 (Layers 0-6)**: Features initially entangled (60% overlap)
  - **Phase 2 (Layers 7-8)**: Maximum separation (10% overlap)
  - **Phase 3 (Layer 10)**: Sharp re-entanglement (80% overlap) — the bottleneck
- Format-specific features show 1.5× amplification in failing formats
- Layer 10 = "re-entanglement bottleneck" where format bias locks in

**Scripts**: `experimental/sae/all_layers_batched.py`, `experimental/sae/layer_10_focused_analysis.py`
**Data**: `experimental/sae/all_32_layers_analysis.json`
**Documentation**: `experimental/sae/LAYER_10_CRITICAL_DISCOVERY.md`

---

## Experiment 7: Causal Validation (Format Dominance Hypothesis)

**Research Question**: Is the bug caused by format tokens "dominating" attention bandwidth?

**Methods**:
- Boosted format token contribution in simple format (to try to induce bug)
- Reduced format token contribution in Q&A format (to try to fix bug)
- Tested multiple threshold levels

**Results**:
- **Format dominance is NOT causal**
- Boosting format contribution in simple format: does NOT induce bug
- Reducing format contribution in Q&A format: does NOT fix bug
- Conclusion: formats create qualitatively different attention computations, not just quantitative differences

**Scripts**: `experimental/causal_validation/format_dominance_validation.py`
**Documentation**: `experimental/causal_validation/causal_validation.md`

---

## Experiment 8: Sparse Editing & Neuron-Level Intervention (Failed)

**Research Question**: Can we fix the bug by targeting individual neurons?

**Methods**:
- Three approaches attempted:
  1. Neuron ablation
  2. Steering vectors
  3. Sparse activation editing

**Results**:
- **All three approaches FAILED**
- "Irremediable entanglement": critical neurons are active in both correct AND incorrect responses
- Example: L14/N12639 active in both buggy and correct paths — cannot ablate one without breaking the other
- Steering vectors too small (minimal activation differences between formats)
- Conclusion: entanglement requires appropriately scoped intervention (full attention submodule), not point interventions

**Scripts**: `experimental/submission/`
**Documentation**: `experimental/submission/EXPERIMENT_SUMMARY.md`, `experimental/submission/sparse_editing_final_conclusions.md`

---

## Experiment 9: Generalization Across Decimal Pairs

**Research Question**: Does the Layer 10 intervention generalize beyond 9.8 vs 9.11?

**Methods**:
- Tested 5 decimal pairs with the same intervention
- n=1000 per pair

**Results**:
| Decimal Pair | Bug Present? | Intervention Success |
|---|---|---|
| 9.8 vs 9.11 | Yes | 100% |
| 8.7 vs 8.12 | No bug | N/A |
| 7.85 vs 7.9 | No bug | N/A |
| 3.4 vs 3.25 | No bug | N/A |
| 10.9 vs 10.11 | Yes | 0% (tokenization effects) |

- 4/5 decimal pairs: intervention successful or no bug to fix
- 10.9 vs 10.11 failure attributed to different tokenization

---

## Experiment 10: Cross-Model Validation (27 Models)

**Research Question**: Do other models exhibit the bug? Does even/odd head specialization generalize?

**Methods**:
- Tested 27 models across Pythia, Gemma, Llama, GPT-2, OPT, GPT-Neo families
- For models with the bug, tested even/odd head intervention

**Results**:

### Bug Prevalence (6/27 models = 22%)
| Model | Base Error | Instruct Error |
|---|---|---|
| Pythia-160M | 90% | 95% |
| Gemma-1B | 85% | 10% |
| Gemma-2B | 80% | 5% |
| Llama-3.1-8B | ~90% | ~100% |
| Llama-3.1-70B | 60% | 70% |
| Mistral-7B | 20% | 25% |

### Even/Odd Specialization
| Model | Even Success | Odd Success | Pattern Holds? |
|---|---|---|---|
| Llama-3.1-8B | 100% | 0% | YES |
| Pythia-160M | 100% | 0% | YES |
| Gemma-2B | 100% | 100% | NO (both work) |

### Instruction Tuning Paradox
- **Gemma**: IT FIXES the bug (90% → 0%)
- **Llama**: IT AMPLIFIES/maintains the bug

**Scripts**: `cross_model_validation/test_pythia_160m.py`, `cross_model_validation/test_gemma_2b.py`
**Documentation**: `cross_model_validation/CROSS_MODEL_VALIDATION_RESULTS.md`, `experimental/ablations/COMPLETE_RESULTS_ALL_MODELS.md`

---

## Experiment 11: Pythia Training Dynamics (When Does Specialization Emerge?)

**Research Question**: When during pretraining does the even/odd head specialization emerge?

**Methods**:
- Pythia-160M checkpoints analyzed across training
- Tested even/odd head intervention at each checkpoint
- Format sensitivity analysis at each checkpoint

**Results**:
- **Specialization emerges ONLY at the final checkpoint** (step 143k of 143k)
- 0% → 100% specialization between steps 120k–143k (final 15% of training)
- **Late sudden phase transition**, not gradual development
- Implication: late optimization phenomenon, NOT architectural bias

**Scripts**: `training_dynamics_analysis/scripts/test_pythia_training_dynamics.py`
**Documentation**: `training_dynamics_analysis/HEAD_SPECIALIZATION_README.md`

---

## Experiment 12: Memorization vs. Generalization (Critical Finding)

**Research Question**: Is the even-head specialization general mathematical reasoning or memorization?

**Methods**:
- Tested Pythia-160M on variations of the prompt:
  - Different numbers (9.9 vs 9.11 instead of 9.8 vs 9.11)
  - Different word order (9.11 vs 9.8 instead of 9.8 vs 9.11)
  - Different wording ("larger" instead of "bigger")
  - 50 different decimal comparison pairs
- Large-sample format sensitivity (25 cases per format)

**Results**:
- Even-head trick works ONLY for the exact phrase "Q: Which is bigger: 9.8 or 9.11?\nA:"
- Different numbers: FAILS
- Different word order: FAILS
- Different wording: FAILS
- 0% accuracy on 50 different decimal pairs
- Memorization score: 0.14/1.0 (strong memorization evidence)
- **Conclusion: The "specialization" is ultra-specific memorization of one training example, not general numerical reasoning**

**Paradigm Shift**: What appeared to be learned attention head specialization is memorization with zero generalization.

**Documentation**: `training_dynamics_analysis/COMPREHENSIVE_FINDINGS_REPORT.md`, `training_dynamics_analysis/SIMPLE_SUMMARY.md`

---

## Experiment 13: Bandwidth Competition & Spatial Organization

**Research Question**: Does the even/odd claim hold up under scrutiny? Is it really about index parity?

**Methods**:
- Tested random combinations of even heads (not just "any 8")
- Analyzed spatial organization (consecutive vs spread patterns)
- Permutation invariance testing
- Clustering analysis (ARI metric)

**Results**:
- Original claim "ANY 8 even heads work" **partially refuted**: only 63% of random even head combos succeed (19/30)
- **Spatial organization matters more than parity**:
  - Consecutive head patterns: 100% success
  - Uniform spacing: 100% success
  - Irregular spacing: 33% success
- No functional clustering by even/odd (ARI = -0.060)
- BUT permutation invariance test confirms indices ARE functionally meaningful

**Scripts**: `bandwidth/working_validated/`, `bandwidth/test_permutation_invariance.py`
**Documentation**: `bandwidth/BANDWIDTH_COMPETITION_INVESTIGATION_COMPLETE.md`

---

## Experiment 14: Pythia Clustering Dynamics

**Research Question**: How are attention heads organized functionally? Is it pure index-based or function-based?

**Methods**:
- Multi-method clustering verification
- Weight similarity vs behavioral similarity analysis
- Head swapping experiments (virtual permutation testing)
- Activation patching across head groups

**Results**:
- **Hybrid Index/Function Dependence Model**:
  - Position type level: Even vs Odd groups (INDEX-dependent)
  - Within groups: Heads are interchangeable (FUNCTION-dependent)
- Within-group flexibility: even heads shufflable among themselves
- Between-group constraint: even/odd heads cannot be swapped
- Weight clustering ≠ behavioral clustering (static similarity doesn't predict dynamic function)

**Scripts**: `pythia_clustering_dynamics/scripts/`
**Documentation**: `pythia_clustering_dynamics/docs/EXPERIMENT_SUMMARY_COMPLETE.md`

---

## Experiment 15: Publication Visualizations

**Research Question**: N/A — figure generation for paper

**Methods**:
- Generated publication-ready figures with real experimental data
- Multiple visualization types

**Outputs**:
- `main_results_figure.py` — 3-panel: bug rates & intervention success
- `mechanism_figure.py` — 4-panel: attention mechanism analysis
- `surgical_precision_figure.py` — heatmap of intervention precision
- `attention_pattern_comparison.py` — detailed attention pattern analysis

**Scripts**: `experimental/paper_visualizations/`

---

## Summary of Key Paradigm Shifts

1. **"Specialization" → "Memorization"**: What appeared to be learned head specialization is ultra-specific training data memorization with zero generalization (Pythia).

2. **"Small Samples Are Fine" → "Statistical Power Matters"**: Small samples (8-12 cases) led to completely wrong conclusions; large samples (25+) revealed robust performance.

3. **"Distributed Mechanism" → "Bottleneck Theory"**: The bug concentrates at a single bottleneck (Layer 10) where format-separated representations re-entangle.

4. **"Linear Modulation" → "Qualitative Computation"**: Format dominance (token importance) is correlational but NOT causal. The mechanism involves qualitatively different computations.

5. **"Architectural Bias" → "Late Optimization"**: Specialization emerges in the final 15% of training as a sudden phase transition.

---

## Open Questions from the Paper

1. Is even/odd specialization universal across architectures? (Partially answered: NO for Gemma)
2. Why exactly 8 heads and 60% threshold mathematically?
3. When does specialization emerge during training? (Answered for Pythia: very late, as memorization)
4. Can format-invariant attention mechanisms prevent such failures?

---

## Reviewer Response Experiments (March 2026)

Experiments R1-R11 address reviewer comments from joQU and WzrL.

### Experiment R1: Arbitrary Head Combinations (COMPLETE)

**Prompt**: Address joQU's central criticism — does even/odd parity matter, or is it a threshold effect?

**Research Question**: Do arbitrary 8-head subsets (mixing even and odd) succeed at the same rate as even-only subsets?

**Methods**:
- Model: Llama-3.1-8B-Instruct, Layer 10 attention patching
- Section 1: 200 random mixed-parity 8-head subsets, n=50 trials each
- Section 2: Varying subset sizes k={4,6,7,8,9,10,12,16}, 100 random subsets each, n=50
- Section 3: Odd-only subsets at k=8,10,12,14,16, n=50
- Section 4: 30 matched even/odd control pairs (same spatial pattern, shifted by 1), n=50
- Wilson CIs, paired t-test, Wilcoxon signed-rank test
- Total runtime: 516.6 minutes (~8.6 hours)

**Results**:

*Section 1 — Random mixed-parity 8-head subsets*:
- 13/200 = **6.5%** success rate (vs **63%** for even-only 8-head subsets from Exp 13)
- Correlation between #even heads and success rate confirmed

*Section 2 — Success rate by subset size k (random mixed-parity)*:

| k | Mean Success | Fraction ≥80% |
|---|---|---|
| 4 | 0% | 0% |
| 6 | 2% | 2% |
| 7 | 4% | 4% |
| 8 | 5% | 5% |
| 9 | 14% | 14% |
| 10 | 10% | 10% |
| 12 | 21% | 21% |
| 16 | **36%** | **36%** |

Even with ALL 16 heads (k=16, guaranteed to include all even heads), only 36% of random subsets succeed — because the odd heads actively interfere.

*Section 3 — All-odd subsets*:

| k | Success Rate |
|---|---|
| 8 | **0%** [0%, 7%] |
| 10 | **0%** [0%, 7%] |
| 12 | **0%** [0%, 7%] |
| 14 | **0%** [0%, 7%] |
| 16 (ALL odd) | **0%** [0%, 7%] |

**All 16 odd heads together = 0% success.** Odd heads cannot fix the bug at any count.

*Section 4 — Matched even/odd controls (30 pairs)*:
- Mean even rate: **70%**
- Mean odd rate: **0%**
- Mean advantage: **+70%**
- Paired t-test: p < 0.0001
- Wilcoxon signed-rank: p < 0.0001

**Key Findings**:
1. **Parity STRONGLY matters** — not just a threshold effect. Even-only 8-head subsets succeed 63% vs 6.5% for random mixed 8-head subsets (10x difference).
2. **Odd heads are inert**: 0% success at ANY subset size, even all 16 together.
3. **Odd heads interfere**: At k=16 (which includes all 16 even heads), only 36% succeed — odd heads actively corrupt the fix.
4. **Matched controls prove causation**: Same spatial pattern, shifted by 1 (even→odd), produces +70% advantage (p<0.0001).
5. More heads help somewhat (0% at k=4 → 36% at k=16) but even doubling the count from 8→16 only reaches 36% for mixed subsets vs 63% for 8 even-only.

**Scripts**: `reviewer_experiments/exp_R1_arbitrary_heads/run_experiment.py`
**Data**: `reviewer_experiments/exp_R1_arbitrary_heads/results_20260314_023132.json`

---

### Experiment R2: Memorization vs Generalization on Llama (EARLY RESULTS)

**Prompt**: Both reviewers' concern about narrow scope; extends Exp 12 (Pythia memorization) to Llama.

**Research Question**: Does the even-head intervention generalize across prompt variations on Llama-3.1-8B?

**Methods**:
- 9 prompt variations (synonyms, rephrasings, order swaps, different decimals)
- 70 decimal pairs for broader bug hunting
- n=100 trials per intervention test, n=20 baselines
- Intervention: first 8 even heads at Layer 10

**Results (Section 1 — Prompt Variations — COMPLETE)**:

| Variation | Q&A Bug Rate | Simple Correct Rate | Format-Dependent Bug? | Intervention Fix |
|---|---|---|---|---|
| original (9.8 vs 9.11) | 100% | 100% | **YES** | **100%** |
| reversed_order (9.11 vs 9.8) | 0% | 100% | No | N/A |
| synonym_larger | 0% | 100% | No | N/A |
| synonym_greater | 100% | 0% | No* | N/A |
| rephrased_1 | 100% | 0% | No* | N/A |
| rephrased_2 | 100% | 0% | No* | N/A |
| trailing_zero (9.80) | 0% | 100% | No | N/A |
| diff_decimal_97 (9.7 vs 9.11) | 0% | 100% | No | N/A |
| diff_decimal_93 (9.3 vs 9.11) | 100% | 0% | No* | N/A |

*Model fails in BOTH formats (not format-dependent)

**Key Finding**: Only the **exact original phrasing** shows a format-dependent bug on Llama. Synonyms and rephrasings either work in both formats or fail in both. The intervention works perfectly (100%) on the one case where the bug exists, but there's only one case to fix.

**Interpretation**: This is more nuanced than Pythia's pure memorization — Llama can answer some variations correctly in both formats. But the format-dependent failure is highly specific to the exact phrasing, similar to the Pythia finding.

**Results (Section 2 — Bug Hunting — COMPLETE)**:
- 70 decimal pairs tested (X.Y vs X.YZ format where X.Y > X.YZ)
- Categories: 1-digit vs 2-digit decimal, 2-digit integer, edge cases
- **0 out of 70 pairs have a format-dependent bug**
- None had the Q&A-fails/Simple-works pattern beyond the original 9.8 vs 9.11

**Generalization Score**: 1.00 (1 fixable / 1 buggy) — trivially perfect because only 1 bug exists

**Overall Interpretation**: The format-dependent decimal comparison bug is essentially **unique to 9.8 vs 9.11** on Llama-3.1-8B-Instruct. This is more nuanced than Pythia's memorization (Llama handles many variations correctly in both formats), but the format-dependent failure is extremely specific. The paper's claims about generalization need significant caveating.

**Scripts**: `reviewer_experiments/exp_R2_generalization/run_experiment.py`
**Data**: `reviewer_experiments/exp_R2_generalization/results_20260313_184514.json`

---

### Experiment R3: MLP Contribution Analysis (COMPLETE)

**Prompt**: Reviewer joQU — "the authors focus on attention throughout and fail to rationalize why they believe attention is the focal mechanism"

**Research Question**: Does MLP patching also fix the bug? Is attention truly the only mechanism?

**Methods**:
- MLP-only patching at Layers 6-15, 20, 25 (n=100 per layer)
- Combined Attn+MLP patching at same layers
- Attention-only baseline at Layer 10
- MLP knockout (zeroing) at Layer 10
- Logit lens decomposition (attention vs MLP contribution)

**Results**:

| Layer | MLP-only | Attn+MLP | Attn-only |
|-------|----------|----------|-----------|
| 6 | 0% | 0% | |
| 7 | 0% | 0% | |
| **8** | **100%** | **100%** | |
| 9 | 0% | 0% | |
| **10** | **0%** | **100%** | **100%** |
| 11 | 0% | 0% | |
| 12-14 | 0% | 0% | |
| **15** | **100%** | **100%** | |
| **20** | **100%** | **100%** | |
| 25 | 0% | 0% | |

MLP Knockout at Layer 10: Q&A still buggy (0% correct), Simple still works (100%).

**Key Findings**:
1. **MLP patching ALSO fixes the bug** at Layers 8, 15, and 20 (100% each)
2. At Layer 10 specifically, only attention works (MLP=0%), confirming the paper's L10 finding
3. The paper's claim that "only attention" matters is incorrect — the correct framing is that Layer 10 attention is the cleanest SINGLE-LAYER intervention, but MLP at other layers also works
4. MLP knockout confirms Layer 10's MLP is NOT needed for the bug mechanism

**Scripts**: `reviewer_experiments/exp_R3_mlp_analysis/run_experiment.py`
**Data**: `reviewer_experiments/exp_R3_mlp_analysis/results_20260313_200039.json`

---

### Experiments R4-R8: Prepared / Running

- **R4** (10.9 vs 10.11 Deep Dive): `reviewer_experiments/exp_R4_decimal_pairs/run_experiment.py` — READY
- **R5** (Logit Difference): COMPLETE — see below
### Experiment R5: Logit Difference Analysis (COMPLETE)

**Prompt**: Reviewer joQU — "The authors did not analyze when the '.8' and '.11' tokens are boosted"

**Research Question**: At which layer do the logits for token "8" (id=23) vs "11" (id=806) diverge between formats?

**Methods**:
- Logit lens at all 32 layers, final token position
- Track logit("8") - logit("11") in both Simple and Q&A formats
- Full (layer × position) heatmap

**Results** (logit difference "8" minus "11" at final token):

| Layer | Simple | Q&A | Gap (Simple - Q&A) |
|-------|--------|-----|---------------------|
| 0 | -1.03 | -1.47 | +0.44 |
| 4 | -3.60 | +1.32 | -4.92 *** |
| 5 | -1.78 | +3.56 | -5.34 *** |
| 10 | -0.95 | -3.15 | +2.20 *** |
| 16 | -0.43 | -2.75 | +2.32 *** |
| 25 | +1.09 | -0.11 | +1.19 *** |
| 30 | +2.48 | +0.91 | +1.57 *** |
| 31 | +1.61 | +0.56 | +1.05 *** |

**Key Findings**:
1. Layers 4-5: Q&A format initially FAVORS correct answer (positive logit for "8"), then reverses
2. Layer 10: Major divergence (+2.20 gap) — Q&A format strongly suppresses "8" token
3. Layers 16-24: Consistent ~+1.5 to +2.3 gap accumulates
4. Layers 25-31: Simple format finally pushes "8" logit positive; Q&A stays weak
5. The format divergence is NOT concentrated at one layer — it accumulates across many layers, with Layer 10 being one important but not unique inflection point

**Scripts**: `reviewer_experiments/exp_R5_logit_difference/run_experiment.py`
**Data**: `reviewer_experiments/exp_R5_logit_difference/results_20260313_202523.json`

---

### Experiment R6: Patching Clarification (COMPLETE)

**Prompt**: Reviewer joQU — "they don't make it clear what part of the attn layer was patched - pre-W_out?"

**Research Question**: At which point in the attention computation does patching work?

**Methods**: Patched 4 different points, n=100 each:

**Results**:

| Patching Point | Success |
|---|---|
| Post-W_O (self_attn output — paper's method) | **100%** [96%, 100%] |
| Post-o_proj (output projection only) | **100%** [96%, 100%] |
| Post-v_proj (value projection) | **100%** [96%, 100%] |
| Post-q_proj (query projection — control) | **0%** [0%, 4%] |

**Key Finding**: Patching works at any point AFTER the value computation (V, O_proj, full output) but NOT at the query projection. The format corruption affects how values are combined/routed, not query formation.

**Technical spec for paper**: "We hook `model.model.layers[10].self_attn`, replacing the first element of the output tuple (shape [1, seq_len, 4096]). For head-specific patching, we reshape to [1, seq_len, 32, 128]."

**Scripts**: `reviewer_experiments/exp_R6_patching_clarification/run_experiment.py`
**Data**: `reviewer_experiments/exp_R6_patching_clarification/results_20260313_203505.json`

---

### Experiment R7: Diff-of-Means Patching (COMPLETE)

**Prompt**: Reviewer joQU — "A more nuanced approach would capture mean activations... and apply that 'adjustment' vector (diff of means)"

**Research Question**: Can the bug be fixed with a simple additive diff-of-means intervention instead of direct activation patching?

**Methods**:
- Compute adjustment vector: mean(Simple activations) - mean(Q&A activations) at final token
- Apply adjustment to Q&A format during generation
- Tested at 8 layers (6-12, 25) on residual stream (n=200 each)
- Decomposed: attention-only and MLP-only at Layers 10 and 25
- Scaled: 0.5x, 1.0x, 2.0x, 5.0x, 10.0x at Layer 10 residual

**Results**:

| Condition | Success Rate | CI |
|---|---|---|
| Layer 6 residual | **0%** | [0%, 2%] |
| Layer 7 residual | **0%** | [0%, 2%] |
| Layer 8 residual | **0%** | [0%, 2%] |
| Layer 9 residual | **0%** | [0%, 2%] |
| Layer 10 residual | **0%** | [0%, 2%] |
| Layer 11 residual | **0%** | [0%, 2%] |
| Layer 12 residual | **0%** | [0%, 2%] |
| Layer 25 residual | **0%** | [0%, 2%] |
| Layer 10 attn decomposed | **0%** | [0%, 2%] |
| Layer 10 MLP decomposed | **0%** | [0%, 2%] |
| Layer 25 attn decomposed | **0%** | [0%, 2%] |
| Layer 25 MLP decomposed | **0%** | [0%, 2%] |
| All 5 scale factors (0.5-10x) | **0%** | [0%, 2%] |

**Key Findings**:
1. **Diff-of-means completely fails** — 0/200 = 0% success at EVERY layer, component, and scale tested
2. The bug mechanism is NOT a simple additive bias in the residual stream
3. Direct activation replacement (the paper's method) captures nonlinear/distributional information that a mean shift cannot
4. This strongly validates the paper's patching approach over the reviewer's suggested alternative
5. Adjustment norms were reasonable (||adj|| ≈ 4-5 for most layers, 12.8 for L25), ruling out magnitude issues

**Scripts**: `reviewer_experiments/exp_R7_diff_of_means/run_experiment.py`
**Data**: `reviewer_experiments/exp_R7_diff_of_means/results_20260313_214414.json`

---

### Experiment R4: 10.9 vs 10.11 Deep Dive (COMPLETE)

**Prompt**: Reviewer joQU — "if 10.9 vs. 10.11 had the same bug, it would show that the phenomenon is not isolated"

**Research Question**: Does the format-dependent bug generalize to 10.9 vs 10.11? Does the same Layer 10 attention mechanism apply?

**Methods**:
- Tokenization analysis: 10.9 → [10, ., 9], 10.11 → [10, ., 11]
- Baseline: 20 trials each format
- Logit lens at all 32 layers
- Attention patching at all 32 layers (50 trials each)
- MLP patching at all 32 layers (50 trials each)

**Results**:

| Test | Result |
|---|---|
| Q&A format baseline | 100% bug (says 10.11 > 10.9) |
| Simple format baseline | **100% bug** (says 10.11 > 10.9) |
| Format-dependent bug exists? | **NO** |
| Attention patching (any layer) | **0%** — no working layers |
| MLP patching Layer 5 | **100%** |

**Key Findings**:
1. **10.9 vs 10.11 has NO format-dependent bug** — the model answers wrong in BOTH formats
2. Attention patching cannot fix it because there's no "correct format" to patch FROM
3. MLP patching at Layer 5 CAN correct it (100%), suggesting MLP has a separate correction mechanism
4. This fundamentally limits the paper's generalization claims — the format-dependent mechanism is specific to 9.8 vs 9.11
5. Tokenization is analogous ([10, ., 9] vs [10, ., 11]), so the difference isn't tokenization structure

**Scripts**: `reviewer_experiments/exp_R4_decimal_pairs/run_experiment.py`
**Data**: `reviewer_experiments/exp_R4_decimal_pairs/results_20260313_224520.json`

---

### Experiment R8: Threshold Confidence Intervals (COMPLETE)

**Prompt**: Paper claims sharp 7→8 even-head threshold; reviewers question whether this is robust.

**Research Question**: How sharp is the 7→8 even-head threshold? What fraction of random 7-even and 8-even subsets succeed?

**Methods**:
- 500 random 7-even-head combos (from C(16,7)=11,440 possible), 50 trials each
- 500 random 8-even-head combos (from C(16,8)=12,870 possible), 50 trials each
- Transition: 7 even heads + N odd heads (N=0,1,2,3,5,9,16), 50 trials each
- Seed=42 for reproducibility

**Results**:

| Metric | 7-even | 8-even |
|---|---|---|
| Mean success rate | 40.4% | 59.4% |
| Fraction ≥80% success | 40.4% | 59.4% |
| Max/Min success | max=100% | min=0% |
| Sharp threshold? | **No** — gradual improvement |

**Transition (7 even + N odd heads)**:

| Config | Success Rate | 95% CI |
|---|---|---|
| 7 even + 0 odd | 0% | [0%, 7%] |
| 7 even + 1 odd | 0% | [0%, 7%] |
| 7 even + 2 odd | 0% | [0%, 7%] |
| 7 even + 3 odd | 0% | [0%, 7%] |
| 7 even + 5 odd | **100%** | [93%, 100%] |
| 7 even + 9 odd | 0% | [0%, 7%] |
| 7 even + 16 odd (all) | 0% | [0%, 7%] |

**Key Findings**:
1. The 7→8 threshold is **not sharp** — it's a gradual improvement (40% → 59%), not a cliff
2. Both 7-even and 8-even combos show high variance — some 7-even combos succeed (100%), some 8-even combos fail (0%)
3. The transition section shows bizarre behavior: adding 5 odd heads to 7 even heads gives 100%, but adding 9 or 16 odd heads gives 0%. This suggests the specific combination of heads matters more than the count
4. **Implication for the paper**: The "Goldilocks" threshold claim needs significant softening. The effect is real (more even heads helps) but the transition is gradual and highly combo-dependent

**Scripts**: `reviewer_experiments/exp_R8_threshold_analysis/run_experiment.py`
**Data**: `reviewer_experiments/exp_R8_threshold_analysis/results_20260314_091723.json`

---

### Experiment R9: Prior Literature Engagement (COMPLETE)

**Prompt**: Reviewer WzrL's critique about insufficient engagement with prior literature on mechanistic interpretability of arithmetic reasoning.

**Research Question**: How do our findings relate to existing work on arithmetic reasoning, intervention methods, and cross-model generalization?

**Methods**: Literature review and systematic comparison with key prior work, informed by all R1-R8 experimental results.

**Key Comparisons**:
1. **Mueller et al. (2024)** — "Quest for the Right Mediator": Our gradual threshold (R8) is consistent with their "mediator granularity" framework. Revise "Goldilocks principle" to cite their work.
2. **Transluce/Monitor (2024)** — Neuron steering: Their 21% broad improvement vs our 100% narrow fix highlights a specificity-generality tradeoff.
3. **Stolfo et al. (2023)** — Head specialization for arithmetic: Related but different — our heads specialize for format processing, not arithmetic operations.
4. **Hanna et al. (2023)** — Greater-than comparison: Closest methodological parallel but they study correct computation, we study failure modes.

**Revised Claims**: All paper claims significantly softened based on R1-R8:
- "comprehensive analysis" → "detailed case study"
- "architectural design principles" → "model-specific observations"
- "Goldilocks principle" → "gradual threshold consistent with Mueller et al."
- 100% fix rate contextualized as applying to exactly one bug instance (R2)

**Analysis**: `reviewer_experiments/exp_R9_prior_literature/analysis.md`

---

### Experiment R10: SAE Causal Validation (COMPLETE — NEGATIVE)

**Research Question**: Can SAE feature-level interventions at Layer 10 causally fix the bug?

**Methods**:
- Llama-Scope residual stream SAEs (TopK, 8x expansion = 32,768 features)
- Section 1: Feature identification (Simple vs Q&A SAE feature differences)
- Section 2: Full SAE feature swap (encode Simple→decode→inject into Q&A)
- Section 3: Selective feature clamping (top N differential features)
- Section 4: Feature necessity (ablation from full swap)
- Section 5: Multi-layer SAE comparison (layers 8, 9, 10, 11, 12, 25)
- n=50 trials per condition

**Results**: **0% success across ALL conditions.**

| Section | Condition | Success Rate |
|---|---|---|
| S1 | Feature identification | 40% overlap, 4 simple-dominant, 10 qa-dominant features |
| S2 | SAE-reconstructed swap | **0%** |
| S2 | Direct residual swap | **0%** |
| S3 | Selective clamping (4-50 features) | **0%** (all conditions) |
| S4 | Necessity ablation (keep 0-500 at Q&A) | **0%** (all conditions) |
| S5 | Multi-layer SAE swap | **0%** (all 6 layers) |

**SAE Reconstruction Errors** (L2 norm):

| Layer | Reconstruction Error |
|---|---|
| 8 | 24,640 |
| 9 | 22,848 |
| 10 | 19,360 |
| 11 | 21,696 |
| 12 | ~20,000 |
| 25 | ~20,000 |

**Key Finding**: The Llama-Scope SAEs are **too lossy for causal intervention**. The reconstruction error (19,000-24,000 L2) destroys the information needed for bug correction. SAE features are useful for *observational* analysis (identifying format-specific features) but cannot faithfully reconstruct activations for *causal* patching.

**Important caveat**: This experiment patches the **full decoder layer output** (residual stream after attention + MLP), not just the attention output. The known working intervention patches `self_attn` output only. There is no attention-specific SAE available, so SAE-based causal validation of the attention mechanism is not currently feasible with Llama-Scope SAEs.

**Scripts**: `reviewer_experiments/exp_R10_sae_causal/run_experiment.py`
**Data**: `reviewer_experiments/exp_R10_sae_causal/results_20260314_095232.json`

---

### Experiment R11: Paper Edits (FINAL)

**Status**: Complete — all R1-R10 results incorporated.

**Summary of required edits**:
1. Table 2 caption fix (3 configs achieve 100%, not just 1)
2. 5 typo fixes
3. Patching specification with R6 verification
4. Tone-down 5 major claims (with evidence references)
5. 6 new limitations from R1-R10 findings
6. 3 new related work paragraphs (Mueller et al., Transluce, Stolfo/Hanna)
7. 3 new/updated figures (logit difference, even vs odd, threshold)
8. Section-by-section edit plan for revised submission

**Key framing shift**: Paper moves from "comprehensive analysis with general principles" to "precise case study of a specific, narrow failure mode with model-specific mechanistic insights."

**Document**: `reviewer_experiments/exp_R11_paper_edits/EDITS.md`

---

## Directory Map

```
9_8_research/
├── working_scripts/              # Verified working implementations
├── repro/                        # Reproduction package with statistical validation
├── experimental/
│   ├── attention/                # Attention mechanism analysis (Exp 3)
│   ├── ablations/                # 27-model ablation study (Exp 10)
│   ├── sae/                      # SAE feature analysis (Exp 6)
│   ├── causal_validation/        # Format dominance hypothesis (Exp 7)
│   ├── statistical_validation/   # n=1000 validation (Exp 5)
│   ├── submission/               # Sparse editing attempts (Exp 8)
│   ├── paper_visualizations/     # Publication figures (Exp 15)
│   ├── logitlens/                # Logit lens analysis (Exp 2)
│   ├── acdc/                     # ACDC circuit discovery
│   ├── GPT2/                     # GPT-2 specific tests
│   ├── llama/                    # Llama-specific tests
│   ├── multi/                    # Multi-model comparisons
│   ├── layer25/                  # Layer 25 experiments
│   ├── attention_output_patching/# Initial patching work
│   └── attribution/              # Attribution analysis
├── bandwidth/                    # Bandwidth competition theory (Exp 13)
├── cross_model_validation/       # Cross-model testing (Exp 10)
├── training_dynamics_analysis/   # Training emergence (Exp 11-12)
├── pythia_clustering_dynamics/   # Head clustering (Exp 14)
├── experiments/                  # Ad-hoc explorations
└── reviewer_experiments/         # Reviewer response experiments (R1-R11)
    ├── exp_R1_arbitrary_heads/   # Mixed-parity head testing
    ├── exp_R2_generalization/    # Llama memorization/generalization
    ├── exp_R3_mlp_analysis/      # MLP vs attention contribution
    ├── exp_R4_decimal_pairs/     # 10.9 vs 10.11 deep dive
    ├── exp_R5_logit_difference/  # Logit difference analysis
    ├── exp_R6_patching_clarification/ # Patching point disambiguation
    ├── exp_R7_diff_of_means/     # Reviewer's suggested method
    ├── exp_R8_threshold_analysis/ # 7→8 threshold CIs
    ├── exp_R10_sae_causal/       # SAE feature ablation/activation
    └── exp_R11_paper_edits/      # Paper corrections (draft)
```
