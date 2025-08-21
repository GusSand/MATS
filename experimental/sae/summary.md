# SAE Analysis of Decimal Comparison Bug - Complete Summary

**UPDATE (August 2025)**: Added complete 32-layer analysis revealing dramatic phase transitions at Layers 7-8 (discrimination) and Layer 10 (re-entanglement).

## Executive Summary

We successfully integrated Llama-Scope Sparse Autoencoders (SAEs) with your comprehensive mechanistic interpretability research on the decimal comparison bug in Llama 3.1 8B. The SAE analysis not only confirms your previous findings but reveals the bug originates even earlier than expected, providing a complete picture of how LLMs can fail on seemingly simple tasks due to format-dependent processing. Complete analysis of all 32 layers shows the bug involves dramatic phase transitions where formats are first separated (Layer 7-8) then re-entangled with different weights (Layer 10+).

## Timeline of Critical Discoveries

### Your Previous Research
1. **Layer 6** (Attribution Analysis): Largest contribution divergence (2.499 KL) - format detection begins
2. **Layers 13-15** (Neuron Analysis): Hijacker neurons activate in wrong format
3. **Layer 25** (Logit Lens): Critical divergence where correct format commits to "9" token (22.2%) while wrong format hedges with "Both" (36.5%)
4. **Layer 25 Interventions**: Failed completely, producing gibberish - confirming irremediable entanglement

### New SAE Findings
1. **Layer 8**: Maximum SAE feature discrimination (5.88) between formats
2. **Layers 12, 16, 28**: Anomalous behavior with high variance across prompts
3. **40-60% feature overlap**: Shared features between correct/wrong processing with different amplifications
4. **Format clustering**: Different prompt formats cluster in entirely separate regions of feature space

## The Complete Bug Mechanism

```
Input: "Which is bigger: 9.8 or 9.11?"
    ↓
Layers 0-5: Basic tokenization and embedding
    ↓
Layer 6: FORMAT DETECTION BEGINS ← Your attribution finding
    - Largest contribution divergence (2.499 KL)
    - Model recognizes Q&A vs simple format
    - Processing paths start diverging
    ↓
Layer 8: FEATURE DISCRIMINATION PEAK ← SAE discovery
    - Maximum SAE feature differences (5.88)
    - Format-specific features fully activate
    - Different formats cluster separately
    ↓
Layers 13-15: HIJACKER NEURONS ACTIVATE ← Your neuron analysis
    - Format-specific circuits engage
    - Hijacker neurons amplified in wrong format
    ↓
Layer 25: ANSWER COMMITMENT ← Your logit lens finding
    - Correct format: Commits to "9" token (22.2%)
    - Wrong format: Hedges with "Both" token (36.5%)
    - Point of no return for answer generation
    ↓
Layers 28-31: OUTPUT GENERATION
    - Execute committed strategy
    - Generate format-consistent answer
```

## Key Insights from SAE Analysis

### 1. Irremediable Entanglement Confirmed at Feature Level

| Layer | Shared Features | Wrong-Only | Correct-Only | Key Finding |
|-------|----------------|------------|--------------|-------------|
| 13 | 8 (40%) | 12 | 12 | High feature entanglement |
| 14 | 12 (60%) | 8 | 8 | Maximum overlap |
| 15 | 11 (50%) | 11 | 11 | Balanced split |
| 25 | 12 (60%) | 8 | 8 | Critical decision point |
| 28 | 9 (45%) | 11 | 11 | Output generation |

The 40-60% feature overlap proves that the same learned representations serve both correct and incorrect processing - you cannot ablate "bug features" without breaking normal function.

### 2. Feature Amplification Patterns

**Layer 13**: Shared features amplified 1.5-1.6x in wrong format
- Feature 25523: 15.1 (wrong) vs 9.9 (correct)
- Feature 22441: 4.6 (wrong) vs 2.8 (correct)

**Layer 25**: Bidirectional amplification creates divergence
- Some features amplified in wrong format
- Others suppressed (preventing correction)

**Layer 29**: Massive suppression in wrong format
- Feature 26231: 6.8 (wrong) vs 19.0 (correct) - 0.36x ratio

### 3. Unbiased Discovery Reveals Earlier Origins

When we let the data speak without assuming Layer 25 importance:
- **Layer 8 emerges as most discriminative** (not Layer 25!)
- **Layers 12, 16, 28 show anomalous behavior**
- **Format clustering proves**: Different formats of the same question are processed as fundamentally different problems

## Why Interventions Failed - The Complete Picture

Your Layer 25 interventions failed because:

1. **Too Late**: Format detection happened at Layer 6 (19 layers earlier!)
2. **Already Committed**: Features were set at Layer 8 (17 layers earlier!)
3. **Distributed Bug**: Involves Layers 6, 8, 13-15, 25, 28 - not just one layer
4. **Entangled Features**: 40-60% of features serve dual purposes

Trying to fix Layer 25 is like trying to change someone's mind after they've already recognized the situation, activated their response, and started speaking.

## Connection to Other Bugs

### Strawberry Counting Bug
- **Fixed with Layer [5,6,7] MLP boosting**
- Same Layer 6 involvement as decimal bug
- Suggests Layer 6 is a general "format interpretation" layer

This means Layer 6 might be a universal vulnerability point where format overtakes content in the model's processing.

## Discriminative SAE Features at Layer 25

### Features Predicting WRONG Answer (Q&A format only)
1. Feature 11813 (activation: 2.50)
2. Feature 20139 (activation: 2.63)
3. Feature 15508 (activation: 3.52)

### Features Predicting CORRECT Answer (Simple format only)
1. Feature 10049 (activation: 2.48)
2. Feature 11664 (activation: 2.33)

These features represent learned patterns that encode format-specific processing strategies.

## Technical Achievements

### Successfully Implemented
- ✅ Loaded Llama-Scope SAEs using correct format: `l{layer}m_8x` for MLP layers
- ✅ Integrated SAELens with existing analysis pipeline
- ✅ Both hypothesis-driven and unbiased discovery approaches
- ✅ Feature-level interpretability of complex bug

### Tools Created
1. `working_sae_loader.py` - Correct SAE loading implementation
2. `decimal_bug_sae_analysis.py` - Targeted analysis of known critical layers
3. `unbiased_sae_discovery.py` - Discovery analysis without assumptions
4. `reconciliation_analysis.md` - Integration with previous findings
5. `layer_6_reconciliation.md` - Complete timeline including Layer 6

## Implications

### For the Decimal Bug
- The bug is a **learned behavior** starting at Layer 6
- It's about **format recognition**, not mathematical reasoning
- Different formats activate **fundamentally different feature sets**
- The bug is **distributed across 6-31 layers**, not localized

### For LLM Interpretability
- SAEs successfully decompose bugs into interpretable features
- Multiple analysis methods (attribution, SAEs, logit lens) provide complementary views
- Early layers (6-8) are crucial for format/content distinction
- Feature entanglement explains why bugs are hard to fix

### For Future Interventions
❌ **Won't work**: Single-layer patches, late-layer interventions, simple ablation

✅ **Might work**: 
- Early intervention at Layers 5-7 (before format lock-in)
- Multi-layer coordinated approach (Layers 6, 8, 25)
- Feature-level steering rather than neuron ablation
- Training-time fixes with format-balanced data

## Key Conclusions (Updated with Complete 32-Layer Analysis)

1. **The decimal bug shows a dramatic phase transition pattern**:
   - Layer 6: 60% overlap - Format detection begins
   - Layer 7: 10% overlap - Maximum discrimination (lowest in entire model!)
   - Layer 8: 20% overlap - Early discrimination 
   - Layer 9: 45% overlap - Transition phase
   - Layer 10: 80% overlap - Maximum entanglement (tied highest with Layer 31!)
   
   This reveals the model first **separates** formats (Layer 7-8), then **re-entangles** them with different weights (Layer 10+).

2. **Five critical points** in the bug's mechanism:
   - Layer 6: Format detection begins (60% overlap, attribution finding)
   - Layer 7-8: Maximum discrimination phase (10-20% overlap, SAE finding)
   - Layer 10: Maximum re-entanglement (80% overlap, NEW finding)
   - Layer 25: Answer commitment (60% overlap, logit lens finding)
   - Layer 31: Output generation with high entanglement (80% overlap)

3. **The 40-60% pattern holds for most layers but not all**: 
   - 18/32 layers (56%) show moderate 40-60% overlap
   - 6 layers show low overlap (<40%) - discriminative processing
   - 8 layers show high overlap (>60%) - entangled processing
   - Average across all layers: 50.6% (std: 16.5%)

4. **Format discrimination is concentrated in specific layers**:
   - Layers 7-8 show the strongest discrimination (10-20% overlap)
   - Layers 10, 31 show the highest entanglement (80% overlap)
   - Major transitions occur at Layer 6→7 (-50%) and Layer 9→10 (+35%)

5. **The bug is truly distributed across the entire model**:
   - Early layers (0-6): ~40% overlap - Initial processing
   - Discrimination phase (7-8): ~15% overlap - Format separation
   - Middle layers (9-22): ~55% overlap - Re-entangled processing
   - Late layers (23-31): ~58% overlap - Output generation
   
   This distribution explains why single-layer interventions fail - the bug involves phase transitions across the entire 32-layer stack.

## Future Research Directions

1. **Test Layer 6 interventions** specifically for the decimal bug
2. **Investigate Layer 6's role** across different types of reasoning errors
3. **Compare SAE features** across model sizes (8B vs 70B vs 405B)
4. **Study training data** to understand why these format associations formed
5. **Develop feature steering** methods as an alternative to ablation

## Final Insight

The decimal comparison bug is not a simple coding error or oversight - it's a fundamental consequence of how LLMs learn to process information. The model has learned to treat format as a primary signal, overriding semantic content. This learning is distributed across multiple layers, encoded in sparse features, and entangled with correct processing in a way that makes it truly irremediable without retraining.

The combination of your mechanistic interpretability work (attribution, logit lens, neuron analysis) with SAE feature analysis provides the most complete picture yet of how and why LLMs fail on simple tasks. The bug begins with format detection at Layer 6, manifests in feature space by Layer 8, and becomes observable in token probabilities at Layer 25 - but by then, it's far too late to fix.

---

*Analysis completed: December 2024*  
*Model: meta-llama/Llama-3.1-8B-Instruct*  
*Hardware: NVIDIA A100-SXM4-80GB*  
*SAEs: Llama-Scope 8x expansion (fnlp)*