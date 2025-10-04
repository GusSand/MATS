# Bandwidth Competition Theory - Experiment 1.1 Results

## Attention Bandwidth Distribution Analysis

**Date:** September 26, 2025
**Experiment:** Set 1.1 - Attention Bandwidth Distribution
**Model:** meta-llama/Meta-Llama-3.1-8B-Instruct
**Layer Analyzed:** Layer 10

---

## Executive Summary

**Experiment Status:** ✅ **COMPLETED** - Unexpected results requiring further investigation

**Key Finding:** The bandwidth competition theory's core predictions were **not supported** by the data. Contrary to expectations, **odd heads show higher numerical bandwidth than even heads** across all format types.

---

## Methodology

### Test Prompts
- **Simple:** `9.8 or 9.11? Answer:`
- **Q&A:** `Q: Which is bigger: 9.8 or 9.11? A:`
- **Chat:** `<|start_header_id|>user<|end_header_id|>\n\nWhich is bigger: 9.8 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n`

### Token Categorization
- **Format tokens:** Q, :, A, Which, is, bigger, ?, user, assistant, header tags
- **Numerical tokens:** 9, ., 8, 11, 0-7
- **Other tokens:** Everything else (including <|begin_of_text|>)

### Bandwidth Calculation
- Measured attention **FROM** the last token **TO** each token category
- Normalized to sum to 1.0 for each head
- Analyzed all 32 attention heads at Layer 10

---

## Results

### Numerical Bandwidth by Head Type and Format

| Format | Head Type | Mean Bandwidth | Std Dev |
|--------|-----------|----------------|---------|
| Simple | Even      | 8.23%          | 5.83%   |
| Simple | Odd       | **10.34%**     | 4.99%   |
| Q&A    | Even      | 4.22%          | 4.49%   |
| Q&A    | Odd       | **6.51%**      | 5.84%   |
| Chat   | Even      | 1.59%          | 2.47%   |
| Chat   | Odd       | **3.23%**      | 5.83%   |

### Hypothesis Testing Results

#### Hypothesis 1: Even heads maintain >40% numerical bandwidth
- **Simple format:** 0.0% of even heads above threshold ❌
- **Q&A format:** 0.0% of even heads above threshold ❌
- **Chat format:** 0.0% of even heads above threshold ❌

**Result:** **REJECTED** - No heads (even or odd) reached the predicted 40% threshold

#### Hypothesis 2: Odd heads drop below 40% in Q&A format
- **Simple format:** 100.0% of odd heads below threshold ✅
- **Q&A format:** 100.0% of odd heads below threshold ✅
- **Chat format:** 100.0% of odd heads below threshold ✅

**Result:** **SUPPORTED** - But this is trivial since ALL heads were below 40%

### Key Observations

1. **Odd heads consistently outperform even heads** in numerical bandwidth
2. **Overall numerical bandwidth is much lower than predicted** (8-10% vs 40%+)
3. **Clear format complexity effect:** Simple → Q&A → Chat shows declining attention
4. **Pattern is consistent across all formats**

---

## Unexpected Findings

### 🚨 Theory Contradiction: Odd > Even

The most significant finding is that **odd heads consistently show higher numerical bandwidth than even heads**, which directly contradicts the bandwidth competition theory's core prediction.

**Numerical Bandwidth Differences (Odd - Even):**
- Simple: +2.11 percentage points
- Q&A: +2.29 percentage points
- Chat: +1.64 percentage points

### 🔍 Low Overall Bandwidth

The absolute bandwidth values (8-10%) are much lower than the theory's predicted 40%+ threshold. This suggests either:
1. **Measurement methodology needs adjustment**
2. **Token categorization is too restrictive**
3. **Different attention positions should be analyzed**
4. **The 40% threshold assumption needs revision**

---

## Possible Explanations

### 1. Measurement Methodology Issues
- **Last token focus:** We measured attention FROM the last token, but numerical reasoning might happen at different positions
- **Aggregation method:** Maybe we should look at maximum attention or attention patterns across multiple positions

### 2. Token Categorization Problems
- **Too restrictive:** Our numerical token list might be missing important tokens
- **Context dependency:** The same token might serve different functions in different contexts

### 3. Layer Selection
- **Layer 10 might not be optimal:** The numerical reasoning might happen at different layers
- **Need multi-layer analysis:** Important patterns might emerge when looking across layers

### 4. Alternative Theoretical Framework
- **Odd heads might specialize in numerical tasks:** Contrary to the original theory
- **Even heads might handle format/structural tasks:** While odd heads handle content

---

## Technical Details

### Files Generated
- **Script:** `bandwidth_analysis_working.py` (successful implementation)
- **Results:** `bandwidth_results_20250926_154825.json`
- **Visualization:** `bandwidth_distribution.png`

### Model Configuration
- **Attention implementation:** `eager` (required for `output_attentions=True`)
- **Precision:** `torch.float16`
- **Device:** CUDA auto-mapped

### Data Quality
- **All experiments completed successfully** ✅
- **No technical errors or missing data** ✅
- **Consistent results across format types** ✅

---

## Next Steps & Recommendations

### Immediate Investigation Needed

1. **🔍 Verify with existing successful experiments**
   - Compare our bandwidth measurements with the successful head patching results
   - Check if the 8 even heads that fix the bug actually show higher bandwidth

2. **📊 Expand measurement methodology**
   - Analyze attention from different token positions (not just last)
   - Try different aggregation methods (max, mean, weighted)
   - Look at attention TO numerical tokens FROM all other positions

3. **🎯 Cross-validate with known working interventions**
   - Use the exact same 8 even heads identified in successful experiments
   - Measure their bandwidth specifically compared to failing heads

### Potential Experiment Modifications

1. **Multi-position analysis:** Look at attention patterns across all token positions
2. **Dynamic token categorization:** Adjust categories based on token context/position
3. **Multi-layer bandwidth:** Analyze bandwidth distribution across multiple layers
4. **Intervention correlation:** Directly correlate bandwidth with intervention success rates

---

## Conclusion

**Experiment 1.1 was technically successful but theoretically surprising.** The results suggest that either:

1. **The bandwidth competition theory needs significant revision**
2. **Our measurement methodology needs refinement**
3. **The numerical reasoning mechanism works differently than predicted**

The most important finding is that **odd heads consistently outperform even heads in numerical bandwidth**, which is the opposite of what the theory predicts. This warrants immediate investigation before proceeding with additional experiments.

**Recommendation:** Before implementing additional experiments, we should investigate why our bandwidth measurements contradict the successful head patching results from previous experiments.

---

*Generated: September 26, 2025*
*Model: meta-llama/Meta-Llama-3.1-8B-Instruct*
*Layer: 10 (attention weights analysis)*
*Status: Completed with unexpected results requiring investigation*