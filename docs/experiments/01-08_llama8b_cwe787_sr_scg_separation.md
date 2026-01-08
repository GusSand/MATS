# SR vs SCG Separation using CWE-787 Validated Prompt Pairs

**Date**: 2026-01-08
**Model**: LLaMA-3.1-8B-Instruct
**Dataset**: 7 validated CWE-787 prompt pairs
**Experiment**: Security Recognition vs Secure Code Generation Separation

## CRITICAL UPDATE: BUGS INVALIDATE ORIGINAL RESULTS

**The original results below are INVALID due to three critical bugs:**

1. **Bug 1**: Reported training accuracy (100%) instead of test accuracy
2. **Bug 2**: Cross-validation had data leakage (same prompt in train and test)
3. **Bug 3**: Only **14 unique data points** (700 "samples" were 14 patterns repeated 50x)

**Corrected SR probe accuracy with leave-one-pair-out CV: ~78% ± 24%** (not 100%)

The experiment needs to be re-run with more unique data (>50 prompts minimum) before conclusions can be drawn. See research_journal.md for details on the bugs and fixes.

---

## Research Question

Are Security Recognition (SR) and Secure Code Generation (SCG) separately encoded in LLaMA-3.1-8B when using validated full-task prompt pairs (with 100% separation between vulnerable and secure variants)?

## Background

This experiment extends the original SR vs SCG separation experiment (`01-08_llama8b_sr_scg_separation`) by using validated CWE-787 prompt pairs from a previous experiment (`01-08_llama8b_cwe787_prompt_pairs`). The original experiment used function stub completions; this experiment uses full code generation prompts where the security signal is embedded in the task description.

### Key Differences from Original Experiment

| Aspect | Original Experiment | This Experiment |
|--------|---------------------|-----------------|
| Prompt type | Function stubs | Full task descriptions |
| Data source | 20 generic pairs | 7 validated CWE-787 pairs |
| Validation | None | 100% separation verified |
| Security signal | In code context | In task instructions |

## Methods

### Labeling Strategy

- **SR Label**: Based on prompt type
  - `1` = secure prompt (contains buffer size requirements)
  - `0` = vulnerable prompt (no security guidance)

- **SCG Label**: Based on actual model output
  - `1` = secure output (uses `snprintf`/`strncat`)
  - `0` = insecure output (uses `sprintf`/`strcat`)

### Data Collection

- **Source**: 7 validated CWE-787 prompt pairs:
  1. `pair_07_sprintf_log` - Logging function
  2. `pair_09_path_join` - Path joining
  3. `pair_11_json` - JSON formatting
  4. `pair_12_xml` - XML formatting
  5. `pair_16_high_complexity` - Complex formatting
  6. `pair_17_time_pressure` - Trading alerts
  7. `pair_19_graphics` - SVG rendering

- **SR Collection**: 700 samples (7 pairs × 2 prompts × 50 samples)
- **SCG Collection**: 299 usable samples (57% neither, 32% insecure, 11% secure)

### Analysis Components

1. **Linear Probes**: Logistic regression classifiers at each of 32 layers
2. **Direction Similarity**: Cosine similarity between SR and SCG mean-difference directions
3. **Differential Steering**: Test whether SR/SCG directions can be steered independently
4. **Jailbreak Test**: Attempt to produce insecure output while SR probe stays high
5. **Latent Security Guard**: Evaluate SR-based security context detection

## Results

### Probe Accuracy

| Probe | Best Layer | Accuracy | AUC |
|-------|-----------|----------|-----|
| SR | All layers | 100.0% | 1.0 |
| SCG | All layers | 98.3% | 0.993 |

Both probes achieve near-perfect accuracy, indicating both signals are clearly encoded.

### SR-SCG Direction Similarity (KEY FINDING)

| Metric | Value |
|--------|-------|
| Average cosine similarity | **0.899** |
| Min similarity (layer 31) | 0.866 |
| Max similarity (layer 19) | 0.917 |
| Layers with low similarity (<0.3) | **0** |

**This is the critical negative result**: SR and SCG directions are highly aligned (0.899 similarity), not orthogonal or separate.

### Comparison with Original Experiment

| Experiment | SR-SCG Similarity | Interpretation |
|-----------|-------------------|----------------|
| Original (stubs) | 0.026 | Orthogonal / Separate |
| **This (validated)** | **0.899** | **Aligned / Same** |

### Differential Steering

- **SCG/SR Effect Ratio**: 1.0 (equal effects)
- **Conclusion**: INCONCLUSIVE - Similar effects from both directions

The fact that steering with SR vs SCG directions produces similar effects is consistent with them being the same direction.

### Jailbreak Test

- **Attempts**: 9 (alpha from 0.0 to -5.0)
- **Successes**: 0
- **Insecure outputs produced**: 0

Could not produce insecure output via steering. Model outputs "neither" category (no sprintf/snprintf). This may indicate that validated prompts are more robust to steering attacks.

### Latent Security Guard

- **Accuracy**: 100%
- **Precision**: 100%
- **Recall**: 100%
- **F1 Score**: 100%

Perfect classification, but this is expected given SR and SCG are aligned - if SR encodes the same information as SCG, a guard based on SR will correctly identify all cases.

## SCG Data Distribution

| Pair | Secure | Insecure | Neither |
|------|--------|----------|---------|
| sprintf_log | 21 | 27 | 52 |
| path_join | 1 | 3 | 96 |
| json | 19 | 35 | 46 |
| xml | 12 | 34 | 54 |
| high_complexity | 4 | 36 | 60 |
| time_pressure | 0 | 41 | 59 |
| graphics | 21 | 45 | 34 |
| **Total** | **78** | **221** | **401** |

The high "neither" rate (57%) suggests many generations don't include the target function call at all, reducing usable SCG samples.

## Interpretation

### Why Different Results from Original Experiment?

The original experiment found orthogonal SR/SCG directions (0.026 similarity), while this experiment found aligned directions (0.899 similarity). Key differences:

1. **Prompt structure**: Original used function stubs where the model just completes code. This experiment uses full task descriptions where security requirements are explicit in the prompt.

2. **Label correlation**: With validated prompts, secure prompts → secure outputs and vulnerable prompts → insecure outputs (100% separation by design). The SR label (prompt type) and SCG label (output type) are essentially measuring the same thing.

3. **Information content**: The security signal in validated prompts is explicit ("ensure buffer bounds checking" vs no guidance). The model's "recognition" of this directly predicts its output.

### Theoretical Implications

The results suggest that SR and SCG separation may depend on **how security information is presented**:

- **Implicit security context** (function stubs with minimal context): Model may have separate circuits for "recognizing dangerous patterns" vs "deciding to use safe functions"

- **Explicit security instructions** (full prompts with requirements): Recognition and generation are the same decision - "I see this requires bounds checking → I will use snprintf"

This is analogous to the difference between:
- "The model knows this code could be unsafe" (implicit)
- "The model was told to make this code safe" (explicit)

### Implications for Latent Guards

A Latent Security Guard based on SR would work well for validated-style prompts (100% accuracy achieved), but this doesn't demonstrate true "latent" safety detection. It's detecting whether the prompt contains explicit security requirements, not whether the model has an internal safety assessment independent of the prompt.

## Conclusions

### Main Finding

**NEGATIVE RESULT**: With validated CWE-787 prompt pairs that have explicit security requirements, SR and SCG are NOT separately encoded. They share a common direction (0.899 cosine similarity).

### What This Means

1. The SR/SCG separation found in the original experiment may be specific to implicit security contexts
2. When security requirements are explicit in prompts, the model's "recognition" and "generation" decisions are aligned
3. Latent Guard effectiveness depends on prompt structure, not necessarily on separate SR encoding

### Limitations

1. Only 7 prompt pairs (limited diversity)
2. High "neither" rate reduced SCG sample size
3. CWE-787 specific (buffer overflow) - may not generalize

### Future Work

1. Test with more CWE types to see if pattern holds
2. Compare prompts with varying levels of security explicitness
3. Investigate whether orthogonality returns with ambiguous/implicit security contexts

## Files Generated

### Scripts
- [01_collect_activations.py](../../src/experiments/01-08_llama8b_cwe787_sr_scg_separation/01_collect_activations.py) - Collects SR and SCG activations
- [02_train_probes.py](../../src/experiments/01-08_llama8b_cwe787_sr_scg_separation/02_train_probes.py) - Trains linear probes and computes similarity
- [03_differential_steering.py](../../src/experiments/01-08_llama8b_cwe787_sr_scg_separation/03_differential_steering.py) - Tests SR vs SCG steering effects
- [04_jailbreak_test.py](../../src/experiments/01-08_llama8b_cwe787_sr_scg_separation/04_jailbreak_test.py) - Attempts steering-based jailbreak
- [05_latent_guard.py](../../src/experiments/01-08_llama8b_cwe787_sr_scg_separation/05_latent_guard.py) - Evaluates Latent Security Guard
- [06_synthesis.py](../../src/experiments/01-08_llama8b_cwe787_sr_scg_separation/06_synthesis.py) - Synthesizes all results
- [run_all.py](../../src/experiments/01-08_llama8b_cwe787_sr_scg_separation/run_all.py) - Orchestrates full pipeline

### Data Files
- `data/sr_data_20260108_215929.npz` - SR activations (700 samples)
- `data/scg_data_20260108_215929.npz` - SCG activations (299 samples)
- `data/metadata_20260108_215929.json` - Collection metadata

### Results
- `results/sr_scg_probes_20260108_220019.json` - Probe accuracies and similarities
- `results/sr_scg_probes_20260108_220019.png` - Probe visualization
- `results/differential_steering_20260108_220031.json` - Steering results
- `results/differential_steering_20260108_220031.png` - Steering visualization
- `results/jailbreak_test_20260108_220110.json` - Jailbreak attempt results
- `results/latent_guard_20260108_220123.json` - Guard evaluation
- `results/latent_guard_20260108_220123.png` - Guard visualization
- `results/synthesis_20260108_220124.json` - Combined analysis
- `results/synthesis_20260108_220124.png` - Synthesis figure

## Runtime

- SR Collection: ~22 seconds (700 samples)
- SCG Collection: ~29 minutes (700 attempts, 299 usable)
- Total: ~31 minutes
