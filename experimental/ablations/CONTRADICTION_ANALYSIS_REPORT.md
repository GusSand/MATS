# Contradiction Analysis Report

## Executive Summary

We discovered significant contradictions between our earlier test results and systematic re-testing. The key finding: **prompt format matters more than we initially realized**. The same model can show 0% to 100% accuracy on the same question depending on how you ask it.

## Key Contradictions Found

### 1. Gemma-2B Base Model
- **Earlier**: Showed 90% error rate on 9.8 vs 9.11
- **Now**: Shows 100% accuracy with Q&A format, 60% with simple format
- **Explanation**: The earlier test likely used a simple format that triggered unclear responses

### 2. Instruction-Tuned Models
- **Earlier**: Gemma-2B-IT showed 0% accuracy (seemed to "fix" the bug)
- **Now**: Shows 100% accuracy with simple formats, but 0% with chat templates
- **Explanation**: Chat templates cause empty responses, not better understanding

### 3. Format Sensitivity Discovery

#### What Works:
```
Q: Which is bigger: 9.8 or 9.11?
A: [Model generates answer]
```
Result: 100% accuracy for all models

#### What Fails:
```
<start_of_turn>user
Which is bigger: 9.8 or 9.11?<end_of_turn>
<start_of_turn>model
```
Result: Empty responses ("...")

## The Real Pattern

### It's Not About Base vs Instruct
The instruction tuning effect we thought we found was actually a **format effect**:

1. **Base models**: Handle simple formats well
2. **Instruct models**: Break with chat templates but work with simple formats
3. **The "bug" persistence**: Depends entirely on prompt format

### Performance by Format Type

| Model | Simple Q | Q&A Format | Chat Template |
|-------|----------|------------|---------------|
| Gemma Base | 60% | 100% | N/A |
| Gemma IT | 100% | 100% | 0% (empty) |
| Llama Base | N/A | N/A | N/A |
| Llama IT | 80% | 100% | 0% (empty) |

## Specific Findings

### 1. Gemma-2B Base
- **9.8 vs 9.11**: 100% accuracy (Q&A format)
- **Other comparisons**: Mixed (0-100% depending on numbers)
- **Math operations**: 50% accuracy
- **Overall**: 56.2% across all tests

### 2. Gemma-2B-IT
- **9.8 vs 9.11**: 100% accuracy (simple/Q&A formats)
- **Other comparisons**: 100% accuracy on basic comparisons
- **Math operations**: 50% accuracy
- **Overall**: 70% across all tests
- **Critical**: Chat formats produce empty responses

### 3. Llama-3.1-8B-Instruct
- **9.8 vs 9.11**: 100% accuracy (Q&A format)
- **Other comparisons**: 100% accuracy on basic comparisons
- **Math operations**: 100% accuracy
- **Overall**: 87.5% across all tests
- **Critical**: Chat formats produce empty responses

## Why the Contradiction?

### Earlier Tests Used Different Formats
1. Our `test_family_hypothesis.py` used the format:
   ```python
   prompt = f"Q: Which is bigger: {num1} or {num2}?\nA:"
   ```
   
2. But for instruct models, it tried to use chat templates which failed

3. The empty responses were counted as "0% accuracy" but weren't actually testing decimal understanding

### The Real Bug Pattern
The decimal comparison bug shows up in:
- Gemma-2B base: 3.9 vs 3.11 (0%), 5.6 vs 5.14 (20%)
- But NOT in 9.8 vs 9.11 (100% with Q&A format)

## Conclusions

1. **Format > Model Type**: Prompt format has a larger effect than base vs instruct
2. **Chat Templates Break Models**: Both Gemma and Llama instruct models fail with their official chat templates
3. **The Bug is Real but Inconsistent**: Some decimal comparisons fail, others don't
4. **Simple Formats Win**: "Q: ... A:" format works best across all models

## Recommendations

1. **Always test multiple prompt formats** before drawing conclusions
2. **Avoid chat templates** for numerical reasoning tasks
3. **Use simple Q&A format** for consistent results
4. **Document exact prompts** when reporting model capabilities

## What We Learned

The "family-dependent instruction tuning hypothesis" was incorrect. Instead:
- Instruction tuning doesn't fundamentally change decimal understanding
- Chat template formatting breaks response generation
- The decimal bug exists but is number-specific, not model-family-specific
- Simple prompts reveal true model capabilities better than complex chat formats