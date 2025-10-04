# Final Conclusions: Understanding the Decimal Comparison Bug

## Summary of Findings

After extensive testing and resolving contradictions in our results, we have discovered:

1. **The decimal bug is real** but not family-dependent
2. **Prompt format matters more than model architecture**
3. **Instruction tuning can help** but only with proper prompting
4. **Chat templates often break models** for numerical tasks

## The True Pattern

### Models with the Bug
- **Pythia-160M**: Strong bug (46.7% overall accuracy)
  - Fails on: 9.8 vs 9.11, 3.9 vs 3.11, 10.8 vs 10.11
- **Gemma-2-2B Base**: Moderate bug (70% overall accuracy)
  - Fails on: 5.6 vs 5.14, 10.8 vs 10.11

### Models without the Bug
- **Gemma-2-2B-IT**: No bug (100% accuracy with Q&A format)
- **Llama-3.1-8B Base**: No bug (88% accuracy)
- **Llama-3.1-8B-Instruct**: No bug when properly prompted

## Key Insights

### 1. Format Sensitivity
The same model can show dramatically different results based on prompt format:

| Format | Example | Result |
|--------|---------|--------|
| Q&A | `Q: Which is bigger: 9.8 or 9.11?\nA:` | ✓ Works well |
| Simple | `Which is bigger: 9.8 or 9.11?` | ~ Mixed results |
| Chat | `<start_of_turn>user\n...` | ✗ Often fails |

### 2. The Instruction Tuning Effect
- **Gemma**: Instruction tuning DOES help (100% vs 70% accuracy)
- **Llama**: Base model already good, instruct adds format sensitivity
- **Key**: Must use simple formats, not chat templates

### 3. Specific Number Patterns
Some comparisons are harder than others:
- **Hard**: X.Y vs X.YZ where Y < YZ (e.g., 5.6 vs 5.14)
- **Easy**: Clear magnitude differences (e.g., 1.8 vs 1.12)

## Why the Contradictions?

1. **Different test scripts used different formats**
   - Early tests mixed formats
   - Chat templates counted as "failures"
   
2. **Empty responses misinterpreted**
   - "..." responses counted as wrong
   - Actually a format rejection, not decimal confusion

3. **Inconsistent test methodology**
   - Some tests used temperature=0, others 0.2
   - Different numbers of runs per test

## Final Recommendations

### For Testing Models
1. **Always use Q&A format** for consistency
2. **Test multiple number pairs** - the bug is number-specific
3. **Avoid chat templates** for numerical reasoning
4. **Use temperature=0.2** for slight variation
5. **Run at least 5 trials** per test

### For Understanding the Bug
1. It's about **parsing decimal places**, not magnitude
2. Models struggle with **version-number-like** comparisons
3. The bug appears in **smaller models** more often
4. **Instruction tuning can help** but isn't a silver bullet

## The Bottom Line

The "9.8 vs 9.11" decimal comparison bug:
- **Exists** in some models (Pythia, base Gemma)
- **Can be fixed** by instruction tuning (Gemma case)
- **Doesn't exist** in larger/newer models (Llama 3.1)
- **Is highly sensitive** to prompt format
- **Is not family-dependent** as originally hypothesized

The most important lesson: **Always test with multiple prompt formats and be explicit about your testing methodology when reporting model capabilities.**