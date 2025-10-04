# Transluce Study Comparison: Bible Verses vs Even/Odd Heads

**Date**: September 26, 2025
**Context**: Comparing our Pythia memorization discovery with Transluce's Bible verse hypothesis
**URL**: https://transluce.org/observability-interface

---

## Key Findings Comparison

### Our Discovery (Pythia-160M)
- **Pattern**: Even/odd head specialization for 9.8 vs 9.11
- **Scope**: Ultra-specific memorization ("Q: Which is bigger: 9.8 or 9.11?\nA:")
- **Evidence**: Training data memorization, not reasoning
- **Mechanism**: Attention head patching fixes specific phrase

### Transluce Study (Llama-3.1)
- **Pattern**: Bible verse neurons affect decimal comparison accuracy
- **Scope**: Broader numerical comparison issues (1280 test cases)
- **Evidence**: 55% baseline accuracy on X.Y vs X.Z comparisons
- **Mechanism**: Neuron steering improves performance to 76-79%

---

## Critical Connections

### 1. **Broader Numerical Reasoning Deficits**
**Transluce Finding**: Llama-3.1 gets only 55% of decimal comparisons correct across systematic testing

**Our Finding**: Pythia-160M has specific memorized pattern for 9.8 vs 9.11 only

**Synthesis**: Our "specialization" may be a **memorized patch** for a broader systematic deficit in decimal comparison reasoning

### 2. **Training Data Artifacts**
**Transluce Hypothesis**: Bible verses (0.4% of training data) interfere with numerical reasoning
- 9.8 vs 9.11 → interpreted as Bible verses 9:8 vs 9:11
- Sequential interpretation: 9:8 comes before 9:11 in Bible

**Our Finding**: Exact phrase memorization from training data

**Synthesis**: Both findings point to **training data contamination** affecting numerical reasoning in different ways

### 3. **Intervention Mechanisms**
**Transluce**: Neuron steering (suppressing Bible verse neurons) → 76% accuracy

**Ours**: Attention head patching (activating even heads) → 100% accuracy for specific case

**Key Difference**: Their intervention is **generalizable**, ours is **ultra-specific**

---

## Reinterpreting Our Findings

### The Bible Verse Hypothesis Applied to Pythia

**Speculation**: Could our Pythia pattern also relate to Bible verse interpretation?

**Test**: Does Pythia interpret "9.8 vs 9.11" as "9:8 vs 9:11" (Bible verses)?

**Evidence to Check**:
- Do other Biblical ratios show similar patterns? (e.g., 3.16, 1.23)
- Does adding "verse" or "chapter" context break the pattern?
- Are there other X:Y vs X:Z patterns in Pythia?

### Memorization vs Interference

**Two Competing Hypotheses**:

1. **Pure Memorization** (our original conclusion):
   - Pythia memorized exact phrase "Q: Which is bigger: 9.8 or 9.11?\nA:"
   - No generalization because it's phrase-level pattern matching

2. **Bible Verse Interference + Memorized Fix** (new hypothesis):
   - Pythia has same Bible verse interference as Llama
   - "Even head specialization" is memorized correction for this specific interference
   - Doesn't generalize because the memorized fix is ultra-specific

---

## Experimental Implications

### Tests We Should Run

1. **Bible Verse Context Testing**:
   ```
   "In chapter 9, verse 8 vs verse 11, which is bigger?"
   "Comparing 9:8 and 9:11 as Bible verses..."
   "John 9.8 vs John 9.11"
   ```

2. **Other Biblical Ratios**:
   ```
   "Which is bigger: 3.16 or 3.17?" (John 3:16 reference)
   "Which is bigger: 1.1 or 1.23?" (Psalm references)
   ```

3. **Bible Verse Pattern Testing**:
   ```
   Test all X.Y vs X.Z where Y < Z < 20 (Biblical verse ranges)
   ```

4. **Sequential Interpretation Test**:
   ```
   If 9.8 → 9:8 and 9.11 → 9:11, does Pythia think 9:8 comes before 9:11?
   ```

### Predictions

**If Bible Verse Hypothesis is Correct**:
- Other biblical ratios might show similar even/odd patterns
- Adding biblical context might break or change the pattern
- Pattern might extend to other X:Y vs X:Z comparisons

**If Pure Memorization Hypothesis is Correct**:
- No patterns for other biblical ratios
- Biblical context won't affect the pattern
- Pattern remains limited to exact 9.8 vs 9.11 phrase

---

## Synthesis: Training Data Contamination

### The Bigger Picture

Both studies reveal **training data contamination** affecting numerical reasoning:

**Llama-3.1 (Transluce)**:
- Bible verse neurons interfere with decimal comparison
- Systematic deficit across many X.Y vs X.Z cases
- Generalizable intervention (neuron steering)

**Pythia-160M (Our Study)**:
- Ultra-specific memorized pattern for 9.8 vs 9.11
- No generalization to similar cases
- Non-generalizable intervention (attention patching)

### Common Underlying Issue

**Training Data Bias**: Both models learned unhelpful associations from training data that interfere with mathematical reasoning

**Different Manifestations**:
- **Llama**: Broad biblical interference with decimal comparisons
- **Pythia**: Specific memorized patch for one comparison

---

## Scientific Implications

### 1. **Numerical Reasoning is Fundamentally Broken**
Both studies suggest decimal comparison is a **systematically difficult task** for LLMs, not just edge cases

### 2. **Training Data Quality Matters**
Bible verses, version numbers, and other structured data in training corpora create unexpected interference patterns

### 3. **Model-Specific Artifacts**
Different models develop different maladaptive patterns:
- Some develop systematic biases (Llama)
- Some develop memorized patches (Pythia)

### 4. **Intervention Limitations**
- **Generalizable interventions** (neuron steering) may be more valuable
- **Specific interventions** (attention patching) may be artifacts rather than insights

---

## Revised Understanding

### What We Thought
"Even/odd attention heads develop numerical reasoning specialization"

### What Transluce + Our Study Suggests
"LLMs have systematic numerical reasoning deficits from training data contamination. Some models (Pythia) develop ultra-specific memorized patches, while others (Llama) have broader interference patterns that can be addressed with generalizable interventions."

### Key Insight
**Our "specialization" may be a memorized workaround for the same underlying biblical interference that Transluce identified in Llama.**

---

## Next Steps

### Immediate Tests
1. Run biblical context tests on our Pythia pattern
2. Test other biblical ratios in Pythia
3. Check if Pythia has the same 55% baseline issue on broader decimal comparisons

### Research Questions
1. Do all transformer models have biblical interference with numerical reasoning?
2. Why do some models develop memorized patches vs systematic biases?
3. Can we identify training data sources for these specific patterns?

### Methodological Implications
1. Always test **systematic baselines** (like Transluce's 1280 cases)
2. Consider **training data contamination** as default hypothesis
3. Compare **intervention generalizability** as key metric

---

## Conclusion

The Transluce study **revolutionizes our interpretation** of the Pythia findings:

**Before**: "Even heads develop sophisticated numerical reasoning"
**After**: "Even heads may implement memorized patches for systematic biblical interference"

This connection suggests our discovery is part of a **broader landscape of training data contamination** affecting numerical reasoning across multiple model families.

**The field needs systematic studies of numerical reasoning deficits and their training data origins.**

---

*This analysis connects our specific Pythia findings to broader patterns of numerical reasoning interference in transformer models*