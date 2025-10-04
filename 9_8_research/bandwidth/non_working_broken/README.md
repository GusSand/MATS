# Non-Working / Broken Scripts

This directory contains scripts with methodological issues that produced incorrect or unreliable results.

## Broken Scripts

### ❌ **`test_patching_distributed_coverage.py`**
- **Issue**: Incorrect patching methodology
- **Problem**: Was zeroing out heads instead of proper activation patching
- **Result**: All patterns showed identical 50% performance (random chance)
- **Symptom**: Generated forum-style responses instead of clean answers
- **Fix**: Use proper methodology from `../working_validated/test_distributed_coverage_proper.py`

### ❌ **`test_distributed_coverage_hypothesis.py`**
- **Issue**: No actual model intervention
- **Problem**: Only generated predictions without testing real patching
- **Result**: Theoretical predictions that weren't empirically validated
- **Fix**: Implement actual patching as in working scripts

### ❌ **`comprehensive_summary_analysis.py`**
- **Issue**: Based on incorrect/incomplete data
- **Problem**: Generated summary figures before having validated results
- **Result**: Misleading visualizations
- **Fix**: Summary is now in the main markdown documentation

## Lessons Learned

1. **Proper Activation Patching Critical**: Must use exact methodology from `../working_scripts/`
2. **String Matching Insufficient**: Need proper evaluation beyond simple pattern matching
3. **Model Loading Issues**: Framework compatibility problems with some approaches
4. **Methodology Validation**: Always validate core intervention works before scaling

## What Not to Use

- Don't use the patching methodology from broken scripts
- Don't trust results showing identical performance across all patterns
- Don't use scripts that generate forum-style outputs instead of clean responses

## Migration Notes

If you need functionality from these scripts:
1. Use the working validated versions instead
2. Adapt any useful analysis code to the proper patching framework
3. Validate results manually before trusting them