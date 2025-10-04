# Causal Validation Experiments

This directory contains experiments testing causal hypotheses about the decimal comparison bug mechanism.

## 📁 Contents

- `format_dominance_validation.py` - Main experiment script testing format token dominance hypothesis
- `causal_validation.md` - Comprehensive report of findings
- `format_dominance_results_*.json` - Raw experimental results
- `format_hijacking_validation_*.png` - Visualization of results

## ❌ Status: FAILED HYPOTHESIS

These experiments tested whether manipulating the relative contribution of format tokens could causally affect the bug. **All experiments failed**, demonstrating that format dominance is not the causal mechanism.

## 🔬 Experiments Run

1. **Format Dominance Induction** - Tried to induce bug in correct format
2. **Format Influence Reduction** - Tried to fix bug in buggy format  
3. **Threshold Discovery** - Searched for critical format dominance level

## 📊 Key Finding

Format token contribution differences are **correlational, not causal**. The bug requires qualitatively different attention patterns, not just different token weightings.

## ✅ What Works Instead

See `/working_scripts/bidirectional_patching.py` for the successful approach: complete attention output replacement at Layer 10.

## 📝 Lessons Learned

- Correlation ≠ Causation in neural mechanisms
- Attention outputs are not simply compositional
- Successful interventions require complete pattern replacement
- Format creates fundamentally different computational modes

---

*These experiments were valuable in ruling out a plausible but incorrect hypothesis.*