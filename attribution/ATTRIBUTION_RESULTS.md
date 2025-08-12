# Logit Attribution Analysis Results

## Executive Summary

We performed a comprehensive logit attribution analysis to understand how each transformer layer contributes to the decimal comparison bug in Llama-3.1-8B-Instruct. The analysis traces the model's internal predictions layer-by-layer, revealing the mechanistic path of how different prompt formats lead to opposite answers.

## Key Findings

### 1. Layer 25 is the Critical Divergence Point (Confirmed)

At Layer 25, we observe the critical divergence between formats:

**Q&A Format (Produces Wrong Answer):**
- Delta Logit: 0.314
- P("Both"): 0.129 (12.9% - still hedging)
- P("9"): 0.003 (0.3% - almost no commitment)

**Simple Format (Produces Correct Answer):**
- Delta Logit: 0.836
- P("Both"): 0.048 (4.8% - reduced hedging)
- P("9"): 0.221 (22.1% - strong commitment)

**Critical Insight**: At Layer 25, the Simple format commits to starting with "9" (22.1% probability) while the Q&A format continues hedging with "Both" tokens (12.9% probability).

### 2. Largest Overall Divergence at Layer 6

Surprisingly, the largest contribution difference occurs earlier at Layer 6:
- Q&A contribution: 2.365
- Simple contribution: -0.135
- Difference: 2.499

This suggests that format-dependent processing begins very early in the network, well before the visible divergence at Layer 25.

### 3. Token Probability Evolution

The analysis tracked how token probabilities evolve through layers:

**"Both" Token Evolution:**
- Both formats start with near-zero "Both" probability
- Q&A format peaks at ~14.8% around layers 23-24
- Simple format peaks lower at ~11.1%
- Both drop to near-zero by final layer

**"9" Token Evolution:**
- Both formats start at zero
- Simple format shows dramatic spike to 22.1% at Layer 25
- Q&A format barely reaches 0.3% at Layer 25
- Critical commitment happens specifically at Layer 25

### 4. Top Contributing Neurons at Layer 25

**Q&A Format Top Neurons:**
1. N13: activation=-0.228, impact=0.176
2. N43: activation=-0.157, impact=0.123
3. N53: activation=-0.153, impact=0.120

**Simple Format Top Neurons:**
1. N13: activation=-0.197, impact=0.152 (shared with Q&A)
2. N27: activation=0.125, impact=0.095
3. N38: activation=-0.125, impact=0.088

**Observation**: Neuron N13 is highly active in both formats but with different activation levels, suggesting it plays a role in format processing.

## Technical Configuration

- **Model**: meta-llama/Llama-3.1-8B-Instruct
- **Temperature**: 0.0 (deterministic)
- **Sampling**: do_sample=False
- **Prompts**:
  - Q&A (Wrong): "Q: Which is bigger: 9.8 or 9.11?\nA:"
  - Simple (Correct): "Which is bigger: 9.8 or 9.11?\nAnswer:"

## Verified Outputs

- **Q&A Format**: "9.11 is bigger than 9.8." ✗ (Wrong)
- **Simple Format**: "9.8 is bigger than 9.11." ✓ (Correct)

## Layer-by-Layer Contribution Pattern

The cumulative logit difference (logit(9) - logit(8)) shows:

1. **Layers 0-5**: Initial processing, format detection begins
2. **Layer 6**: Major divergence in contributions
3. **Layers 7-22**: Gradual buildup of format-specific patterns
4. **Layers 23-24**: Hedging behavior emerges ("Both" tokens)
5. **Layer 25**: Critical commitment point - Simple format commits to "9"
6. **Layers 26-31**: Strengthening of chosen path
7. **Layer 32**: Final output projection

## Mechanistic Interpretation

### The Bug's Causal Chain

1. **Format Detection (Layers 0-6)**: The model recognizes the prompt format very early, with Layer 6 showing the largest contribution difference.

2. **Path Establishment (Layers 7-22)**: Format-specific processing paths are established and strengthened.

3. **Hedging Phase (Layers 23-24)**: Q&A format triggers hedging behavior with "Both" tokens.

4. **Commitment Point (Layer 25)**: 
   - Simple format breaks through to commit to "9"
   - Q&A format remains stuck in hedging mode

5. **Answer Generation (Layers 26-32)**: The committed path is reinforced, leading to the final answer.

## Why Interventions Failed

Our previous intervention attempts failed because:

1. **Early Commitment**: Format processing begins as early as Layer 6, making Layer 25 interventions too late
2. **Distributed Processing**: The bug involves contributions from many layers, not just Layer 25
3. **Path Dependency**: Once the format-specific path is established, it's difficult to redirect

## Visualizations Created

1. **Cumulative Logit Difference**: Shows how predictions evolve through layers
2. **Per-Layer Contributions**: Identifies which layers contribute most to the bug
3. **Token Probability Evolution**: Tracks "Both" and "9" token probabilities
4. **Hedging Zone Zoom (L20-30)**: Detailed view of the critical region
5. **Top Neurons Analysis**: MLP neuron contributions at Layer 25

All visualizations saved in high-resolution PNG and PDF formats with publication-ready formatting.

## Conclusions

1. **Layer 25 is the observation point, not the cause**: While Layer 25 shows the clearest divergence in token probabilities, the format-dependent processing begins much earlier (Layer 6).

2. **The bug is a multi-layer phenomenon**: Contributions from many layers accumulate to produce the wrong answer, explaining why single-layer interventions fail.

3. **Hedging is the key mechanism**: The Q&A format triggers a hedging response ("Both" tokens) that prevents commitment to the correct answer.

4. **Format detection happens early**: The model determines its processing strategy within the first 6 layers, well before any visible token predictions.

## Recommendations

1. **Multi-layer interventions**: Target layers 6-25 simultaneously for any intervention attempts
2. **Format preprocessing**: Modify input formats before they reach the model
3. **Fine-tuning focus**: Target early layers (0-6) for format-independent processing
4. **Prompt engineering**: Use formats that avoid triggering hedging behavior

## Files Generated

- `logit_attribution_analysis_20250811_204500.png` - High-resolution visualization
- `logit_attribution_analysis_20250811_204500.pdf` - Publication-ready PDF
- `logit_attribution_results_20250811_204502.json` - Complete numerical results
- `logit_attribution_20250811_*.log` - Detailed execution logs

---

*Analysis completed: August 11, 2025*  
*Temperature: 0.0 (deterministic)*  
*Model: Llama-3.1-8B-Instruct*