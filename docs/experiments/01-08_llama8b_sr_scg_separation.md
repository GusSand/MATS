# SR vs SCG Separation Experiment

**Date**: 2026-01-08
**Model**: meta-llama/Meta-Llama-3.1-8B-Instruct
**Duration**: 34 minutes
**Inspired by**: arxiv 2507.11878 (Harmfulness vs Refusal Separation)

## Research Question

Are **Security Recognition** (SR) and **Secure Code Generation** (SCG) separately encoded in LLaMA-8B, analogous to how harmfulness and refusal are separately encoded according to the arxiv paper?

## Background

Previous experiments (01-07) found:
- Security context is encoded early (linear probes 100% at L0)
- Behavior emerges late (logit lens diverges at L31)
- Signal is distributed across 81 features in 16 layers
- No sparse circuit like IOI

The arxiv paper 2507.11878 found that:
- Harmfulness and refusal are **separately encoded**
- Jailbreaks work by reducing refusal while leaving harmfulness intact
- Adversarial finetuning has minimal impact on latent harmfulness beliefs

This experiment tests if the same separation exists for security.

## Methodology

### Definitions

| Concept | Definition | Probe Label |
|---------|------------|-------------|
| **SR (Security Recognition)** | Does the context contain security-relevant information? | 1 = security warning present, 0 = neutral |
| **SCG (Secure Code Generation)** | Will the model output secure code? | 1 = secure function, 0 = insecure function |

### Security Pairs Tested

| # | Insecure | Secure | Vulnerability |
|---|----------|--------|---------------|
| 1 | sprintf | snprintf | Buffer overflow |
| 2 | strcpy | strncpy | Buffer overflow |
| 3 | gets | fgets | Buffer overflow |
| 4 | atoi | strtol | No error handling |
| 5 | rand() | getrandom() | Weak randomness |

### Data Collection

- **SR Dataset**: 450 samples (90 per pair × 5 pairs)
  - 45 secure context + 45 neutral context per pair
- **SCG Dataset**: 437 samples (varying per pair based on model output)
  - Label based on actual model generation

### Experiments

1. **Train Separate Probes**: Logistic regression for SR and SCG at each layer
2. **Compute Direction Similarity**: Cosine similarity between probe coefficient vectors
3. **Differential Steering**: Steer SR direction → measure SCG effect (and vice versa)
4. **Jailbreak Test**: Apply negative SCG steering to produce insecure output while SR stays high
5. **Latent Security Guard**: Use SR probes to detect security contexts regardless of output

## Results

### 1. Probe Accuracy

**SR Probes (Security Recognition):**
| Layer Range | Accuracy | AUC |
|-------------|----------|-----|
| 0-7 | 100% | 1.000 |
| 8-15 | 100% | 1.000 |
| 16-23 | 100% | 1.000 |
| 24-31 | 100% | 1.000 |

**SCG Probes (Secure Code Generation):**
| Layer Range | Accuracy | AUC |
|-------------|----------|-----|
| 0-7 | 83% | 0.86 |
| 8-15 | 83% | 0.85 |
| 16-23 | 83% | 0.87 |
| 24-31 | 83% | 0.87 |

**Key Observation**: SR is perfectly linearly decodable at all layers. SCG is harder to predict (83%), suggesting the model's decision to output secure code involves more than just recognizing the context.

### 2. Direction Similarity

| Layer | Cosine Similarity | Interpretation |
|-------|-------------------|----------------|
| 0 | 0.138 | SEPARATE |
| 5 | 0.026 | SEPARATE |
| 10 | 0.093 | SEPARATE |
| 15 | -0.002 | SEPARATE |
| 17 | **-0.047** | SEPARATE (most negative) |
| 20 | -0.037 | SEPARATE |
| 25 | 0.006 | SEPARATE |
| 31 | 0.029 | SEPARATE |

**Summary Statistics:**
- Average similarity: **0.026**
- Min similarity: -0.047 (L17)
- Max similarity: 0.138 (L0)
- Layers with low similarity (<0.5): **32/32 (100%)**

**Conclusion**: SR and SCG are encoded as **nearly orthogonal directions**. This is strong evidence for separate encoding, analogous to harmfulness vs refusal.

### 3. Differential Steering

Testing if steering one direction affects the other:

| Layer | SR Max Effect | SCG Max Effect | SCG/SR Ratio |
|-------|---------------|----------------|--------------|
| 16 | 0.124 | 0.042 | 0.34x |
| 20 | 0.051 | 0.043 | 0.84x |
| 24 | 0.057 | 0.043 | 0.75x |
| 28 | 0.073 | 0.051 | 0.70x |
| 31 | 0.052 | **0.142** | **2.73x** |

**Key Finding**: At Layer 31, SCG steering is **2.73x more effective** than SR steering at changing P(secure). This suggests:
- The "decision to write secure code" is made at L31
- Steering the SCG direction at L31 has the strongest effect on output

### 4. Jailbreak Test

Attempting to produce insecure output while SR probe remains high:

| Alpha | SR Probe | Output | Result |
|-------|----------|--------|--------|
| 0.0 | 1.000 | SECURE | Baseline |
| -0.5 | 1.000 | neither | Disrupted |
| -1.0 | 1.000 | SECURE | Resistant |
| -2.0 | 1.000 | neither | Disrupted |
| -3.0 | 1.000 | neither | Disrupted |
| -5.0 | 1.000 | neither | Disrupted |

**Result**: **0 successful jailbreaks**

The model resisted steering toward insecure output. Instead of outputting insecure code, it either:
- Continued outputting secure code
- Generated incomplete/unrelated code ("neither")

This differs from the arxiv paper where jailbreaks could reduce refusal while leaving harmfulness intact.

### 5. Latent Security Guard

Using SR probes as a defense mechanism:

| Metric | Value |
|--------|-------|
| Training Accuracy (all layers) | 100% |
| Test Accuracy | 100% |
| F1 Score | 100% |
| Guard-Output Mismatches | 13 |

**Mismatches Analysis**: 13 cases where:
- Guard correctly detected security-relevant context (SR probe high)
- But model output was classified as insecure

This demonstrates the guard can catch cases where the model "knows" about security but doesn't act on it.

## Visualizations

Generated figures:
- `results/sr_scg_comparison_*.png` - Probe accuracy and similarity across layers
- `results/differential_steering_*.png` - Steering effects at each layer
- `results/latent_guard_*.png` - Guard evaluation metrics
- `results/synthesis_*.png` - Combined summary figure

## Conclusions

### Evidence FOR Separation

1. **Near-orthogonal directions** (cosine sim = 0.026)
2. **All 32 layers show separate encoding**
3. **L31 shows differential steering effect** (SCG 2.73x more effective)
4. **Latent Guard works perfectly** (100% accuracy)

### Evidence AGAINST Separation

1. **Jailbreak failed** - couldn't produce insecure output
2. **Earlier layers show SR steering stronger** (contradicts expectation)
3. **Average steering ratio is 1.07x** (inconclusive)

### Overall Conclusion

**MODERATE EVIDENCE FOR SEPARATION**

SR and SCG appear to be separately encoded (orthogonal directions), but the separation doesn't enable the same kind of "jailbreaks" seen in the harmfulness/refusal paper. This may be because:

1. Security code generation is more tightly coupled to context than refusal
2. The model has stronger safeguards around code security
3. Steering disrupts generation rather than redirecting it

### Implications

1. **Security recognition is robust**: Can be detected at any layer with 100% accuracy
2. **Latent Security Guard is viable**: Can detect security contexts regardless of output
3. **Different from harmfulness/refusal**: Security behavior may be harder to manipulate
4. **Defense opportunity**: The 13 mismatches show where additional safeguards could help

## Code Files

| Script | Purpose |
|--------|---------|
| [01_generate_prompts.py](../../src/experiments/01-08_llama8b_sr_scg_separation/01_generate_prompts.py) | Validate security pairs |
| [02_collect_activations.py](../../src/experiments/01-08_llama8b_sr_scg_separation/02_collect_activations.py) | Collect SR and SCG data |
| [03_train_separate_probes.py](../../src/experiments/01-08_llama8b_sr_scg_separation/03_train_separate_probes.py) | Train probes, compute similarity |
| [04_differential_steering.py](../../src/experiments/01-08_llama8b_sr_scg_separation/04_differential_steering.py) | Test steering independence |
| [05_jailbreak_test.py](../../src/experiments/01-08_llama8b_sr_scg_separation/05_jailbreak_test.py) | Attempt jailbreak |
| [06_latent_security_guard.py](../../src/experiments/01-08_llama8b_sr_scg_separation/06_latent_security_guard.py) | Evaluate guard |
| [07_synthesis.py](../../src/experiments/01-08_llama8b_sr_scg_separation/07_synthesis.py) | Generate summary |
| [run_all.py](../../src/experiments/01-08_llama8b_sr_scg_separation/run_all.py) | Run full pipeline |

## Data Files

Located in `src/experiments/01-08_llama8b_sr_scg_separation/data/`:
- `sr_merged_*.npz` - SR activations (450 samples)
- `scg_merged_*.npz` - SCG activations (437 samples)
- `sr_<pair>_*.npz` - Per-pair SR data
- `scg_<pair>_*.npz` - Per-pair SCG data

## Future Work

1. **Try more aggressive steering** - Higher alpha values, combined layer steering
2. **Test on more security pairs** - Expand to all 14 pairs
3. **Compare with other models** - Does this separation exist in GPT-2, Mistral?
4. **Probe the "neither" outputs** - What happens when steering disrupts generation?
5. **Train adversarial examples** - Can we find prompts that bypass the guard?
