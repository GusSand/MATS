# Attention Output Patching Experiments

This directory contains the successful implementations of attention output patching that demonstrate Layer 10 attention output is causally responsible for the decimal comparison bug in Llama-3.1-8B.

## ✅ Working Scripts

### 1. `bidirectional_patching.py` (MAIN RESULT)
- **Status**: ✅ **100% SUCCESS**
- **What it does**: Tests both forward and reverse patching of Layer 10 attention outputs
- **Key Results**:
  - Forward Patching (Buggy format + Correct attention): **100% fixes the bug**
  - Reverse Patching (Correct format + Buggy attention): **100% induces the bug**
- **Conclusion**: Proves bidirectional causality of Layer 10 attention output

### 2. Original Working Implementation
- **Location**: `/home/paperspace/dev/MATS9/layer25/attention_control_experiment.py`
- **Status**: ✅ Working
- **Key Finding**: Layer 10 attention-only patching achieves 100% success

## ❌ Failed Attempts (For Reference)

### 1. `attention_output_patch.py`
- **Status**: ❌ Did not achieve expected results
- **Issue**: Implementation details differed from the working version
- **Result**: Forward patching failed, only reverse patching worked

## Key Findings

1. **Layer 10 is Special**: Only Layer 10 attention output patching works cleanly. Other layers produce gibberish.

2. **Bidirectional Causality Confirmed**:
   - Can fix the bug by replacing buggy attention with correct attention
   - Can induce the bug by replacing correct attention with buggy attention

3. **Attention Output ≠ Attention Weights**:
   - Patching attention weights (attention patterns) does NOT work
   - Patching attention outputs (processed information) DOES work

## How to Run

```bash
# Run the main bidirectional experiment
python bidirectional_patching.py
```

## Results Summary

| Experiment | Success Rate | Conclusion |
|------------|-------------|------------|
| Forward Patch (Fix Bug) | 100% | ✅ Causal |
| Reverse Patch (Induce Bug) | 100% | ✅ Causal |
| Layer 10 Only | 100% | ✅ Specific to Layer 10 |

## Related Documentation

- See `BREAKTHROUGH_FINDINGS.md` in the parent attention directory for theoretical background
- See `layer25/attention_control_experiment.py` for the original working implementation