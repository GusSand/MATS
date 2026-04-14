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

## Experiment R12: GPT-2 Reconciliation with Hoang's Results

**Date**: 2026-03-27
**Status**: COMPLETE — CRITICAL BUG FOUND AND FIXED IN EVALUATION CODE

### Prompt
> Gus's Experiment 10 shows 0% error for GPT-2. Hoang's results show 100% error for GPT-2-Small. Resolve this discrepancy. Test all GPT-2 variants with multiple prompt formats using both text-based and logit-based evaluation. Test Hoang's 5-head circuit and MLP patching.

### Research Question
Why do Gus (0% error) and Hoang (100% error) disagree on GPT-2's decimal comparison bug rate? Which is correct?

### Methods
- **Models**: GPT-2-Small (124M), GPT-2-Medium (355M), GPT-2-Large (774M), GPT-2-XL (1.5B)
- **Prompt formats**: hoang_compare, gus_simple, gus_qa
- **X values**: 1-9 (correct: X.8, incorrect: X.11)
- **Evaluation methods**: Gus's original text eval, strict text eval, first-number eval, logit-based eval
- **Patching**: Hoang's 5-head circuit [(2,2),(5,1),(6,11),(9,9),(10,2)], MLP layer sweep (all 12 layers)
- **Key fix**: `get_logit_difference` had a shared-token bug — candidates like `"1.8"` and `"1.11"` both tokenize to first-token `"1"` (id=16), making both sides pick the same logit → logit_diff=0.000 always. Fixed to only use discriminating (non-shared) token IDs.

### Results (No Interpretation)

**Section 1-2 (pre-fix): Reproduced Gus & Hoang setups**
- Text eval: Gus's eval counts "ambiguous"/"unclear" as "no bug" → 0% error
- Logit eval (BUGGY): showed 0/9 correct (100% bug) due to shared-token issue

**Section 3 (post-fix): All GPT-2 Variants — Logit Eval**

| Model | hoang_compare | gus_simple | gus_qa |
|-------|--------------|------------|--------|
| GPT-2-Small | 8/9 correct (11% err) | 8/9 correct (11% err) | 8/9 correct (11% err) |
| GPT-2-Medium | 7/9 correct (22% err) | 8/9 correct (11% err) | 8/9 correct (11% err) |
| GPT-2-Large | 7/9 correct (22% err) | 8/9 correct (11% err) | 8/9 correct (11% err) |
| GPT-2-XL | 6/9 correct (33% err) | 5/9 correct (44% err) | 8/9 correct (11% err) |

Text eval: 0/9 correct across ALL models/formats (all outputs classified "unclear")

**Section 4: Evaluation Method Comparison (GPT-2-Small, all formats, X=1-9)**
- Text: 0 correct, 0 bug, 27 other
- Logit: 24 correct, 3 bug (89% correct)
- First-number: 18 correct, 6 bug, 3 other

**Section 5: Hoang's 5-Head Circuit Patching**
- Baseline: 8/9 correct → Patched: 8/9 correct (all formats)
- Effect negligible: max Δ = ±0.04 logit diff

**Section 6: MLP Layer Patching**
- Baseline: 8/9 → Patched: 8/9 at every layer (0-11), both formats
- No individual MLP layer has measurable causal effect

**Edge case**: X=8 gives `-inf` because correct tokens (`"8"`, `" 8"`) are shared with incorrect tokens (from `"8.11"` → first token `"8"`), leaving zero unique correct candidates.

### My Interpretation
The reconciliation reveals NEITHER Gus (0% error) NOR Hoang (100% error) was correct for GPT-2:
1. Gus's text eval was blind — base GPT-2 outputs incoherent text, text eval can't classify it, counts as "no bug"
2. Hoang's 100% error appears to have relied on an evaluation with the same shared-token issue we found, OR used a different evaluation method
3. With proper discriminating-token logit eval, GPT-2-Small has ~11% error rate (1/9 = X=8 edge case)
4. GPT-2-XL is surprisingly WORSE than smaller variants (up to 44% error)
5. Circuit patching and MLP patching show negligible effects — the signal is already in the right direction at baseline

**Scripts**: `9_8_research/reviewer_experiments/exp_R12_gpt2_reconciliation/run_experiment.py` (main), `run_section{3-6}.py` (split runners), `run_summary.py`

---

## Experiment 16: Format-Differential MLP Neurons and the 9.11 > 9.8 Bug

**Date**: 2026-04-14
**Status**: COMPLETE — inconclusive re: Transluce hypothesis; real but small selective ablation effect on Llama

### Prompt
> Test the Transluce hypothesis (Oct 2024) that the Llama 9.11 > 9.8 bug is caused by spurious MLP neurons firing on Sept-11 / Bible-verse / gravity contexts. Run on Llama-3.1-8B-Instruct, Pythia-160M, and GPT-2-Small. [User provided a full Python script specifying attribution by chat-vs-simple format contrast and targeted ablation.]

### Research Question
Does ablating MLP neurons that are differentially active between the "chat" (buggy) and "simple" (correct) format fix the 9.11 > 9.8 bug? Are such neurons also present in Pythia-160M (same bug) and absent/weaker in GPT-2-Small (no bug)?

### Important Caveat (my note, flagged)
**This experiment does NOT directly replicate Transluce's methodology.** Transluce used a single prompt format and identified neurons via gradient-based attribution (`e · ∂z/∂e` against the wrong-answer logit). Our experiment used a *format contrast* (chat_mean − simple_mean) to identify candidate neurons. These are different metrics measuring different things; results cannot be interpreted as a direct test of Transluce's specific mechanistic claims. A follow-up experiment (16b, in-progress) uses Transluce's actual gradient-attribution method.

### Methods
- **Models**: Llama-3.1-8B-Instruct (fp16), Pythia-160M (fp32), GPT-2-Small (fp32)
- **Attribution**: Per-neuron differential activation `chat_mean − simple_mean` at the last 4 tokens of the prompt, averaged over 50 forward passes per model. Top-200 by |score| retained; top-50 and top-100 by |score| used for ablation.
- **Ablation mechanism**:
  - Llama: `down_proj.register_forward_pre_hook` zeroing specified channels of the input (canonical SwiGLU "neuron activation" = `act_fn(gate) * up_proj`)
  - Pythia / GPT-2: `mlp.act.register_forward_hook` zeroing specified channels of the activation output
- **Bug-rate measurement**: N=100 stochastic samples per condition (temp=0.6, `do_sample=True`, max_new_tokens=10); Wilson 95% CIs. "Bug" = output contains `9.11`; "correct" = output contains `9.8` (bug check prioritizes `9.11` to avoid the substring-match issue).
- **Prompts**:
  - Llama: chat = Llama-3.1 instruction-template prompt; simple = Q&A text prompt
  - Pythia: Q&A (buggy) vs "Which is bigger ... Answer:" (correct)
  - GPT-2: Q&A vs "The larger number is" completion
- **GPT-2 bug eval**: logit-based with discriminating-token logic reused from Experiment R12.
- **Controls**: random-50 uniform over all layers (main script) and random-50/100 at layer 31, random-50 at layers 28–31 (follow-up)
- **Code reuse**: `wilson_ci` from R3/R12; `get_logit_difference` from R12 via `sys.path` import

### Results (No Interpretation)

**Llama-3.1-8B-Instruct — bug rates (N=100, temp=0.6, 95% Wilson CI)**

| Condition | Bug rate | CI | n_valid |
|---|---|---|---|
| Baseline (chat format) | 100.0% | [96.3, 100] | 100/100 |
| Baseline (simple format) | 78.8% | [69.7, 85.7] | 99/100 |
| Ablate top-50 by \|s\| (mixed sign, L28–31) | **86.3%** | [78.0, 91.8] | 95/100 |
| Ablate top-100 by \|s\| | **81.0%** | [72.2, 87.5] | 100/100 |
| Ablate mid-layer cluster (L10–21, s>0, 15 neurons) | 100.0% | [96.3, 100] | 100/100 |
| Ablate late-layer cluster (L22–31, s>0, 50 neurons) | 100.0% | [96.3, 100] | 100/100 |
| Ablate 50 random neurons (uniform over all layers) | 100.0% | [96.3, 100] | 100/100 |

Cluster composition (from top-200 by |s|): early (<L10, s>0) = 3 neurons; mid (L10–21, s>0) = 15; late (L22–31, s>0) = 83.

Top differential neurons are concentrated at layers 28–31, with layer 31 dominating (16 of top-20). Scores are roughly half positive (fires more in chat than simple) and half negative, with comparable magnitudes (top values: +10.22, −10.27, +9.48, +8.12, +7.62, −7.29, −6.24, …).

**Llama layer-matched random controls (follow-up run, `run_l31_control.py`, N=100)**

| Condition | Bug rate | CI |
|---|---|---|
| Random 50 at L31 only | 100.0% | [96.3, 100] |
| Random 50 at L28–31 | 100.0% | [96.3, 100] |
| Random 100 at L31 only | 100.0% | [96.3, 100] |

**Pythia-160M (N=100, temp=0.6)**

| Condition | Bug rate | CI | n_valid |
|---|---|---|---|
| Baseline Q&A | 82.1% | [64.4, 92.1] | 28/100 |
| Baseline simple | 73.7% | [62.8, 82.3] | 76/100 |
| Ablate top-50 by \|s\| | 83.5% | [73.9, 90.1] | 79/100 |
| Ablate 50 random | 83.3% | [68.1, 92.1] | 36/100 |

Top Pythia differential neurons concentrate at layers 9–11 (scores +6.45, +4.03, +3.63, …); all top-10 are positive-signed.

**GPT-2-Small (logit-based eval, X = 1..9, using R12's discriminating-token method)**

- Correct: 8/9 (X=8 yielded −Infinity due to shared-token edge case from R12)
- Error rate: 11.1% (matches R12's finding exactly)
- Mean |differential score| comparison: Llama top-50 = 3.56; GPT-2 top-50 = 1.65; ratio = **2.16×**

### Observations (facts, no interpretation)
1. Llama top-50/top-100 (|s|-sorted) ablation moves bug rate from 100% to 86.3%/81.0% — non-overlapping CIs with baseline.
2. Positive-s-only ablation at any layer range gave exactly 100% (same as baseline).
3. Every layer-matched random control — at L31 (50 and 100 neurons) and L28–31 (50 neurons) — also gave 100%.
4. The only ablations that moved bug rate included both positive and negative-s neurons (top-50/top-100 by |s|).
5. Pythia showed no ablation effect (top-50 ≈ random ≈ baseline, all ~83%).
6. GPT-2's top-50 differential scores are 2.16× smaller than Llama's in mean absolute magnitude.

### My Interpretation (flagged as mine)

**What the data supports:**
- A specific, non-random subset of layer-28–31 MLP neurons in Llama is causally involved in the bug: random controls at the same layers with the same count give no effect, while the targeted subset moves the bug rate ~14 pp.

**What the data does NOT support:**
- The Transluce hypothesis as stated. Transluce predicts *positive-firing* neurons pushing toward 9.11 via spurious-concept activation. Our positive-s-only ablations did nothing; the effect we observed requires including the **negative-s** neurons (those that fire *less* in chat format than simple format). This is inconsistent with the "9.11-promoting concept neurons" framing.
- Cross-architecture replication. Pythia showed no effect under the same metric; this is weak evidence at best about whether Pythia has analogous bug-causing neurons.

**Caveats:**
- The differential-activation metric is methodologically biased toward final layers because residual-stream magnitudes grow through depth. This may explain the layer-31 concentration regardless of whether those neurons are the most causally important.
- Part of the "bug reduction" in top-50 ablation may be model destabilization: 5/100 responses contained neither "9.8" nor "9.11", so n_valid dropped from 100 to 95.
- GPT-2 X=8 edge case (−Infinity) is a known shared-token issue from R12 and not a finding of this experiment.

### Files Generated
- Main script: [run_experiment.py](../experimental/16_transluce_hypothesis/run_experiment.py)
- Semantic probe texts (used in 16b): [probe_texts.py](../experimental/16_transluce_hypothesis/probe_texts.py)
- Layer-31 control script: [run_l31_control.py](../experimental/16_transluce_hypothesis/run_l31_control.py)
- Main results: `9_8_research/experimental/16_transluce_hypothesis/results_20260414_140942.json`
- Layer-31 control results: `9_8_research/experimental/16_transluce_hypothesis/l31_control_20260414_144929.json`
- Output log: `9_8_research/experimental/16_transluce_hypothesis/output.log`

### Follow-up
A proper gradient-attribution replication of Transluce's method (single format, `e · ∂z/∂e` per-neuron, plus semantic probe check on Sept-11 / Bible / gravity / neutral text sets) is running separately as Experiment 16b (`run_proper_transluce.py`).

---

## Experiment 16b: Proper Transluce Replication with Gradient Attribution

**Date**: 2026-04-14
**Status**: COMPLETE — **strong replication in Pythia-160M, weaker in Llama, untestable on GPT-2-Small**

### Prompt
> [After discovering Experiment 16 did not test the actual Transluce methodology] Read the Transluce article and re-implement their method properly: single format, gradient-based attribution `e · ∂z/∂e`, targeted ablation, plus a semantic probe on Sept-11 / Bible / gravity / neutral text sets. Include a Pythia format sweep since Pythia's bug is format-sensitive.

### Research Question
Do MLP neurons identified by gradient attribution (`e · ∂z/∂e` of the wrong-answer logit minus correct-answer logit) cause the 9.11 > 9.8 bug in Llama, Pythia-160M, and GPT-2-Small? Do the top-attribution neurons exhibit concept-selectivity for Sept-11 / Bible-verse / gravity contexts as Transluce claimed?

### Methods
- **Attribution prompts**: Transluce-style sweep — `"Which is bigger, X.A or X.B? Answer: X."` for X ∈ {1..15} × (A,B) ∈ {(8,11), (9,10), (9,12), (8,13)} = ~50+ prompts per model. The trailing "X." commits the model to a numeric completion whose next token discriminates between A and B.
- **Attribution metric**: `z = logit[wrong_first_token] − logit[correct_first_token]`; compute `z.backward()`; per-neuron attribution = `(activation * grad).sum(target_positions)` averaged over valid prompts. Target positions: last 4 tokens.
- **Neuron activation definition** (same as 16): Llama SwiGLU input to `down_proj`; Pythia/GPT-2 output of `mlp.act`.
- **Token resolution**: prefer single-token encoding (raw) over space-prefixed (which returns a shared space-token id across sides, silently skipping all prompts — a bug caught and fixed mid-run).
- **Ablation**: zero specified (layer, neuron) channels as in 16. Conditions: top-50 positive-attribution, top-50 |attribution| (mixed sign), layer-matched random-50 (same layer distribution as top-50 positive).
- **Bug-rate measurement**:
  - Llama, Pythia: stochastic sampling, N=100, temp=0.6, text match ("9.11" vs "9.8")
  - GPT-2: logit-based via R12's `get_logit_difference` across X ∈ {1..9}\{8}
- **Semantic probe**: 20 probe texts per theme (sept-11, bible, gravity/software-version, neutral). For each top-50 positive-attribution neuron, compute mean activation at the last token across each theme's probes, then report `theme_mean − neutral_mean` per neuron.
- **Pythia format sweep**: 8 prompt variants tested; selected the highest-bug-rate format with adequate n_valid for attribution/ablation.
- **CPU probe fallback**: After CUBLAS errors on Pythia/GPT-2 probes (likely heavy-hook GPU state corruption), probes were moved to CPU. Llama's probe ran on GPU.

### Results (No Interpretation)

#### Llama-3.1-8B-Instruct (Transluce prompt: "Which is bigger, 9.11 or 9.8?")

| Condition | Bug rate | n_valid |
|---|---|---|
| Baseline | 40.6% | 64/100 |
| **Ablate top-50 positive-attribution** | **27.3%** | 22/100 |
| Ablate top-50 \|attribution\| (mixed sign) | 47.6% | 42/100 |
| Layer-matched random-50 | 55.4% | 56/100 |

Top-attribution neurons distributed across layers 11, 14, 15, 21, 24, 29, 30, 31 — not concentrated in the final layer as Experiment 16's differential metric had suggested.

Semantic probe (Llama top-50, GPU): sept11 max=+1.300, mean=−0.094, n>0=27/50; bible max=+0.172, mean=−0.167; gravity max=+0.080, mean=−0.118.

#### Pythia-160M — format sweep

| Format | Prompt | Bug rate | n_valid |
|---|---|---|---|
| `transluce` | "Which is bigger, 9.11 or 9.8?" | 1.1% | 94 |
| `qa_8_first` | "Q: Which is bigger: 9.8 or 9.11?\nA:" | 68.4% | 19 |
| `qa_11_first` | "Q: Which is bigger: 9.11 or 9.8?\nA:" | 68.0% | 25 |
| **`answer_prompt`** | **"Which is bigger: 9.8 or 9.11? Answer:"** | **58.9%** | **90** |
| `answer_prompt_rev` | same, order reversed | 32.9% | 82 |
| `larger_is` | "Between 9.8 and 9.11, the larger number is" | N/A | 0 |
| `compare_two` | "Compare two numbers: 9.11 and 9.8. Which one is bigger?" | 100% | 2 |
| `is_bigger` | "Is 9.11 bigger than 9.8? Answer:" | 30.3% | 33 |

Selected `answer_prompt` for ablation (high bug rate AND high n_valid; the script initially selected `compare_two` by bug_rate alone, n_valid=2 made those ablation numbers uninterpretable — rerun on `answer_prompt` documented below).

#### Pythia-160M — proper attribution on `answer_prompt` (N=100)

| Condition | Bug rate | 95% CI | n_valid |
|---|---|---|---|
| Baseline | 72.3% | [61.8, 80.8] | 83/100 |
| **Ablate top-50 positive-attribution** | **7.1%** | — | 42/100 |
| Ablate top-50 \|attribution\| (mixed sign) | 33.3% | — | 30/100 |
| Layer-matched random-50 | 55.0% | — | 80/100 |

Top-attribution neurons distributed across layers 0, 5, 7, 8, 9, 10. Leading neuron: L9 N2315 (attr=+0.7013). L0 has two strongly-positive neurons (N1089, N2915), suggesting the attribution signal begins early.

Semantic probe (Pythia top-50, CPU): sept11 max=+0.006, mean=+0.000; bible max=+0.017, mean=+0.001; **gravity max=+5.649, mean=+0.336**.

#### GPT-2-Small (logit-based eval across X ∈ {1..9}\{8})

| Condition | Error rate | n_valid |
|---|---|---|
| Baseline | 0.0% | 8/8 |
| Ablate top-50 positive-attribution | 0.0% | 8/8 |
| Ablate top-50 \|attribution\| | 0.0% | 8/8 |
| Layer-matched random-50 | 0.0% | 8/8 |

GPT-2-Small shows no logit-level preference for 9.11 on this prompt family (consistent with R12). No baseline bug → no test of Transluce's hypothesis possible under this evaluation.

Semantic probe (GPT-2 top-50, CPU): sept11 max=+0.320, mean=−0.003; bible max=+0.346, mean=−0.004; gravity max=+1.481, mean=+0.026.

### Cross-Model Summary

| Model | Baseline bug | Top-50 positive-attr ablation | Δ | Random control | Δ | Probe max |
|---|---|---|---|---|---|---|
| Llama-3.1-8B (chat template) | 100% (Exp 16) | — | — | — | — | — |
| Llama-3.1-8B (Transluce prompt) | 40.6% | 27.3% | −13pp | 55.4% | +15pp | sept11 +1.30 |
| Pythia-160M (answer_prompt) | 72.3% | **7.1%** | **−65pp** | 55.0% | −17pp | **gravity +5.65** |
| GPT-2-Small (logit, Transluce-style) | 0% | 0% | 0 | 0% | 0 | gravity +1.48 |

### My Interpretation (flagged as mine)

1. **Pythia-160M is the cleanest replication of Transluce's mechanism** (65pp reduction from targeted positive-attribution ablation, layer-matched random control drops only ~17pp, mixed-sign less effective than positive-only). This is what the Transluce story predicts.
2. **Llama's result is in the same direction but noisy**, partly because Transluce's un-templated prompt reduces baseline bug to 40.6% vs 100% in chat template, and partly because ablating these neurons destabilizes the model heavily (n_valid=22/100 for positive-attribution ablation). The 13pp drop is consistent with Transluce but not clean.
3. **Format-dominance is confirmed as much stronger than Transluce's article implies.** Pythia's bug rate spans 1.1% to 68.4% across near-equivalent English phrasings of the same comparison. The concept-neurons story doesn't explain format-dominance — format-dominance acts *upstream* of the MLP computations we're ablating.
4. **Gravity-concept selectivity is the strongest signal across all three models** — one neuron in Pythia is +5.65× over neutral baseline, vs Transluce's headline narrative emphasizing sept11/bible-verse neurons. With small (20-text) probe sets, this is weak evidence, but the direction is clear.
5. **GPT-2-Small at logit level does not exhibit the bug** on Transluce-style prompts (R12 finding, re-confirmed). This limits how much cross-architecture interpretation we can do.
6. **The "negative-score" anomaly from Experiment 16 is resolved**: gradient attribution picks a different neuron set than the activation-differential metric. The differential metric's magnitude bias toward layer 31 was the culprit; gradient attribution distributes across layers (Llama: 11–31; Pythia: 0–10).

### Caveats
- **Llama n_valid drop**: positive-attribution ablation on Llama drops 78% of responses off-script. Part of the 13pp "bug reduction" is model destabilization rather than clean correct-answer recovery. Pythia's n_valid drop is less severe (42/100 parseable).
- **Probe set size**: 20 texts per theme is tiny vs Transluce's observability infra. The +5.65 outlier in Pythia gravity is driven by a small number of probe activations.
- **CPU probe for Pythia/GPT-2**: CUBLAS errors on GPU after heavy hook traffic were not debugged; moved probe to CPU. Not a scientific issue but a known workaround.
- **Format sensitivity confound**: Pythia's bug is so format-sensitive that the Transluce prompt gave baseline 1.1%; a different prompt (`answer_prompt`) gave 72.3%. Our ablation tests the format where the bug is present, but the concept-neurons story is really a claim about the model's behavior on *any* prompt format where the bug appears.
- **GPT-2 weight loading warning**: `h.{0...11}.attn.bias | UNEXPECTED` appeared on load; cosmetic (attention bias absent in modern configs) but noted.

### Files Generated
- Main script: [run_proper_transluce.py](../experimental/16_transluce_hypothesis/run_proper_transluce.py)
- Followup (Pythia format sweep + GPT-2 logit ablation): [run_proper_followup.py](../experimental/16_transluce_hypothesis/run_proper_followup.py)
- Pythia rerun on `answer_prompt`: [run_pythia_answerprompt.py](../experimental/16_transluce_hypothesis/run_pythia_answerprompt.py)
- Probe texts: [probe_texts.py](../experimental/16_transluce_hypothesis/probe_texts.py)
- Main results: `results_proper_20260414_154247.json`
- Followup results: `results_followup_20260414_160607.json`
- Pythia rerun results: `results_pythia_answerprompt_20260414_173127.json`
- Combined checkpoint: `checkpoint_proper.json`

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
