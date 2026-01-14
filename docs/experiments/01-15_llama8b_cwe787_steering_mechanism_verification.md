# Steering Mechanism Verification Experiment

**Date**: 2026-01-15
**Model**: meta-llama/Meta-Llama-3.1-8B-Instruct
**Dataset**: CWE-787 Expanded (105 pairs)
**Experiment**: Mechanistic verification of activation steering

## Research Question

Does steering at Layer 31 shift the model's internal representations toward the "secure" direction identified by our probes? More specifically: can we demonstrate that the behavioral change (more secure code) corresponds to a measurable shift in activation space toward the natural secure representation?

## Experimental Design

### Three Conditions
| Condition | Prompts | Steering (α) | Purpose |
|-----------|---------|--------------|---------|
| A | Vulnerable | 0.0 | Baseline (no steering) |
| B | Vulnerable | 3.5 | Steered intervention |
| C | Secure | 0.0 | Natural reference |

### Metrics
1. **Probe projections**: `dot(activation, probe_direction)` at layers [0, 8, 16, 24, 28, 30, 31]
   - Probe direction = `mean(secure_activations) - mean(vulnerable_activations)` (normalized)
2. **Steering alignment**: Decompose activation change into parallel/orthogonal to steering vector
3. **Gap closure**: `(proj(B) - proj(A)) / (proj(C) - proj(A)) × 100%`

### Success Criteria

| Level | Criterion | Threshold | Result |
|-------|-----------|-----------|--------|
| **Primary** | Probe B > A at L31 | p < 0.05, d > 0.5 | **PASS** |
| **Secondary** | Gap closure | ≥ 30% | **PASS** (299.5%) |
| **Secondary** | Alignment ratio | > 1.0 | **PASS** (1711.99) |
| Tertiary | SAE features | Move in predicted direction | N/A |

## Results

### Probe Projections by Layer

| Layer | A (baseline) | B (steered) | C (natural) | Gap Closure | Cohen's d |
|-------|--------------|-------------|-------------|-------------|-----------|
| 0 | 0.266 ± 0.108 | 0.266 ± 0.108 | 0.329 ± 0.038 | 0.0% | 0.000 |
| 8 | 0.182 ± 0.155 | 0.182 ± 0.155 | 0.278 ± 0.079 | 0.0% | 0.000 |
| 16 | 0.089 ± 0.139 | 0.089 ± 0.139 | 0.184 ± 0.096 | 0.0% | 0.000 |
| 24 | 0.121 ± 0.108 | 0.121 ± 0.108 | 0.218 ± 0.061 | 0.0% | 0.000 |
| 28 | 0.055 ± 0.077 | 0.055 ± 0.077 | 0.173 ± 0.053 | 0.0% | 0.000 |
| 30 | 0.053 ± 0.062 | 0.053 ± 0.062 | 0.191 ± 0.045 | 0.0% | 0.000 |
| **31** | **0.066 ± 0.056** | **0.476 ± 0.051** | **0.203 ± 0.041** | **299.5%** | **7.599** |

**Key observation**: The effect is entirely localized to Layer 31 (where steering is applied). No detectable change at upstream layers, confirming the intervention is targeted.

### Statistical Tests

**Primary Hypothesis (B > A at L31):**
- t-statistic: -37.997
- p-value: 9.44e-61
- Cohen's d: 7.599 (far exceeds "large effect" threshold of 0.8)

**Steering Alignment:**
- Parallel component magnitude: 27.206
- Orthogonal component magnitude: 0.016
- Ratio: 1711.989

This means 99.99% of the activation change is in the direction of the steering vector.

### Interpretation

1. **Mechanism Verified**: The steering intervention shifts internal representations toward the "secure" direction in activation space. This is not just a surface behavioral change — the model's internal state is measurably different.

2. **Overshoot Phenomenon**: Gap closure of 299.5% means steered activations (B) project *further* in the secure direction than naturally secure prompts (C). This explains:
   - Why high α values cause model degeneracy
   - Why α=3.5 may be too aggressive for optimal behavioral outcomes
   - The model is being pushed past the "natural" secure state

3. **Surgical Precision**: The alignment ratio of 1711 confirms the intervention is highly targeted. Steering does almost nothing orthogonal to the intended direction — it's a pure shift along the probe axis.

## Files Generated

### Code
- [experiment_config.py](../../src/experiments/01-15_steering_mechanism_verification/experiment_config.py) - Configuration and paths
- [01_collect_activations.py](../../src/experiments/01-15_steering_mechanism_verification/01_collect_activations.py) - Hook-based activation collection
- [02_compute_metrics.py](../../src/experiments/01-15_steering_mechanism_verification/02_compute_metrics.py) - Probe projections and alignment
- [03_statistical_analysis.py](../../src/experiments/01-15_steering_mechanism_verification/03_statistical_analysis.py) - Significance tests
- [04_visualizations.py](../../src/experiments/01-15_steering_mechanism_verification/04_visualizations.py) - Publication figures
- [run_experiment.py](../../src/experiments/01-15_steering_mechanism_verification/run_experiment.py) - Orchestrator script

### Data
- `results/activations_*.npz` - Collected activations (50 samples × 3 conditions × 7 layers)
- `results/metrics_*.json` - Computed metrics (probe projections, alignments)
- `results/statistics_*.json` - Statistical analysis results

## Conclusions

1. **Primary claim supported**: Activation steering works through the predicted mechanism — it shifts representations toward the "secure" direction identified by the probe.

2. **Effect is large and significant**: Cohen's d = 7.6, p < 1e-59. This is not a subtle effect.

3. **Intervention is targeted**: 99.99% of the activation change is parallel to the steering vector.

4. **Overshoot at α=3.5**: The steered condition overshoots the natural secure condition, suggesting lower α values might be more appropriate for behavioral outcomes.

## Implications for Future Work

1. **α Calibration**: Consider calibrating α such that gap closure ≈ 100% instead of 300%
2. **Lower layers**: Could steering at earlier layers produce smoother behavioral changes?
3. **SAE analysis**: Still worth investigating which specific features are activated/suppressed
