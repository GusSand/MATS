# Head Swapping Experiment: Index vs Function Dependence

## Overview

This experiment tests whether the even/odd head specialization in Pythia-160M is **index-dependent** (tied to specific positions 0,2,4,6,8,10) or **function-dependent** (tied to learned functions that can be moved).

## Research Question

**Are attention head specializations fundamentally tied to their position indices, or to their learned functions?**

This has profound implications for:
- Understanding the nature of learned attention patterns
- Model editing and intervention techniques
- Architectural design principles
- Mechanistic interpretability methodology

## Experimental Design

### Core Hypothesis

**If function-dependent**: Head specializations should be preserved when we move the functional units to different positions, as long as we properly adjust the output projection weights (W^O) to maintain mathematical equivalence.

**If index-dependent**: Head specializations should break completely when moved to different positions, regardless of mathematical equivalence.

### Mathematical Framework

For attention layer with heads h₀, h₁, ..., h₁₁:

```
Output = W^O · concat(h₀, h₁, ..., h₁₁)
```

When we permute heads according to permutation π:

```
Output = W^O_permuted · concat(h_π(0), h_π(1), ..., h_π(11))
```

For mathematical equivalence, we must adjust W^O accordingly.

### Permutation Types

1. **Baseline**: No permutation (control)
2. **Even/Odd Swap**: (0↔1, 2↔3, 4↔5, 6↔7, 8↔9, 10↔11)
3. **Even/Odd Shuffle**: Randomly shuffle even heads among themselves, odd heads among themselves
4. **Random Permutation**: Completely random head ordering
5. **Reverse Order**: [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

### Test Tasks

#### Primary Task: 9.8 vs 9.11 Decimal Bug
- **Clean prompt**: "Which is bigger: 9.8 or 9.11?"
- **Buggy prompt**: "Q: Which is bigger: 9.8 or 9.11?\nA:"
- **Success criteria**: Model generates "9.8" (correct answer)

#### Secondary Task: Even/Odd Specialization Test
- Use activation patching with even heads [0,2,4,6,8,10]
- **Success criteria**: Patching even heads fixes the bug

#### Tertiary Task: Basic Math
- Simple arithmetic: "2+3", "10-4", "5*2", "8/2"
- **Success criteria**: Correct numerical answers

## Implementation Details

### Weight Manipulation

1. **Extract weights**:
   - QKV weights: `[3, num_heads, head_dim, hidden_size]`
   - Output projection: `[hidden_size, num_heads, head_dim]`

2. **Apply permutation**:
   - Permute QKV weights by head dimension
   - **Crucially**: Permute output projection weights to maintain mathematical equivalence

3. **Restore to model**: Reshape and apply to attention layers

### Mathematical Equivalence

The key insight is that attention output is:
```
output = W^O @ concat(head_0, head_1, ..., head_11)
```

If we permute heads: `π(head_0), π(head_1), ..., π(head_11)`

We must permute W^O accordingly: `W^O_new[:, i, :] = W^O_old[:, π(i), :]`

This ensures the mathematical computation remains identical.

## Expected Results

### Scenario 1: Index-Dependent Specialization
- **Even/odd swap**: ❌ Completely breaks specialization
- **Random permutation**: ❌ Destroys all functionality
- **Even/odd shuffle**: ❌ Breaks specialization
- **Interpretation**: Specialization is hardcoded to specific index positions

### Scenario 2: Function-Dependent Specialization
- **Even/odd swap**: ✅ Preserves functionality (functions moved to odd positions)
- **Random permutation**: ✅ Preserves overall functionality
- **Even/odd shuffle**: ✅ Preserves specialization within groups
- **Interpretation**: Specialization is based on learned functions, not positions

### Scenario 3: Mixed Dependencies
- **Even/odd swap**: ⚠️ Partial preservation
- **Different permutations**: Different levels of preservation
- **Interpretation**: Some aspects are index-dependent, others function-dependent

## Scientific Significance

### For Mechanistic Interpretability
- **Method validation**: Tests whether current interpretability techniques capture true functional organization
- **Intervention design**: Informs how to design effective model interventions
- **Causal understanding**: Distinguishes correlation from causation in attention patterns

### For Model Architecture
- **Position encoding**: Insights into how positional information interacts with learned functions
- **Architectural constraints**: Understanding fundamental vs accidental structural dependencies
- **Design principles**: Guidelines for building more interpretable architectures

### For AI Safety
- **Robustness**: Understanding fragility of learned behaviors to structural changes
- **Predictability**: Whether model behaviors are tied to interpretable functional units
- **Intervention safety**: Reliability of model editing techniques

## Implications by Result

### If Index-Dependent
- Current interpretability methods may be **position-confounded**
- Model editing techniques are **fragile and unreliable**
- Attention patterns may be **architectural artifacts** rather than learned functions
- **Positional information** plays a larger role than previously thought

### If Function-Dependent
- Attention head specializations are **genuine functional units**
- Model editing can be made **robust through proper weight adjustment**
- Current interpretability methods **correctly identify functional organization**
- **Transfer learning** of attention patterns may be possible

### If Mixed
- **Hybrid approach** needed for model intervention
- Some behaviors are **fundamental**, others are **accidental**
- **Careful analysis** required to distinguish index vs function dependencies
- **Different layers/heads** may have different dependency types

## Related Work

This experiment builds on:
- **Activation patching** methodology (Vig et al., 2020)
- **Attention head analysis** (Clark et al., 2019; Kovaleva et al., 2019)
- **Model editing** techniques (Meng et al., 2022)
- **Mechanistic interpretability** frameworks (Olah et al., 2020)

## Limitations

1. **Single model**: Only tests Pythia-160M
2. **Single layer**: Only examines layer 6
3. **Single task focus**: Primary evaluation on decimal comparison bug
4. **Mathematical equivalence assumption**: Assumes perfect weight adjustment preserves all behaviors

## Future Extensions

1. **Multi-model**: Test Llama, GPT-2, other architectures
2. **Multi-layer**: Examine all attention layers
3. **Comprehensive benchmarks**: Full evaluation suites
4. **Partial permutations**: Test subsets of heads
5. **Training dynamics**: Study how dependencies emerge during training

---

This experiment represents a **fundamental test** of our understanding of attention head specialization and provides crucial insights for the future of mechanistic interpretability research.