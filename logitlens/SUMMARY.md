# Logit Lens Analysis of the Decimal Comparison Bug in Llama-3.1-8B-Instruct

## What is Logit Lens Analysis?

Logit lens is a powerful interpretability technique for understanding how language models process information layer by layer. The method works by:

1. **Extracting Hidden States**: Capturing the model's internal representations at each transformer layer
2. **Applying the Output Head**: Taking each layer's hidden state and applying the model's final output projection (language model head) to see what tokens would be predicted if the model stopped at that layer
3. **Tracking Evolution**: Observing how predictions change as information flows through the network

This allows us to see the model's "intermediate thoughts" - what it would output at each processing stage. It's like placing a lens at different depths in the model to observe how the answer forms progressively.

## Executive Summary

We conducted a comprehensive logit lens analysis to understand how Llama-3.1-8B-Instruct processes the question "Which is bigger: 9.8 or 9.11?" differently depending on prompt format. Our analysis reveals that subtle prompt variations lead to dramatically different processing paths through the model's layers, resulting in opposite answers.

## Key Findings

### 1. The Bug Manifests Differently Based on Temperature and Format

| Format | Temperature | Result |
|--------|------------|--------|
| Chat Template | 0.0 (deterministic) | Wrong answer: "9.11 is bigger" |
| Chat Template | 0.2 (stochastic) | Empty response |
| Q&A Format (`Q: ... A:`) | 0.0 | Wrong answer: "9.11 is bigger" |
| Simple Format (`... Answer:`) | 0.0 | **Correct answer: "9.8 is bigger"** |

**Critical Discovery**: The bug is temperature-dependent. This was discovered by examining `/home/paperspace/dev/MATS9/submission/test_empty_response_bug.py`, which revealed that temperature=0.2 produces empty responses while temperature=0.0 produces wrong answers. The script `submission/sparse_edit_simple.py` also helped identify the correct vs wrong format patterns.

### 2. Layer 25 is the Critical Divergence Point

#### How We Discovered Layer 25

Through systematic logit lens analysis, we traced the model's predictions at each of its 32 transformer layers. By applying the language model head to each layer's output, we could see what token the model would predict if it stopped processing at that point.

#### The Exact Probability Values That Revealed the Divergence

**Correct Format (Simple "Answer:"):**
- Layer 20: Top token = "neither", P(token "9") = 0.000
- **Layer 25: Top token = "9", P(token "9") = 0.222** ← First appearance of "9" as top prediction!
- Layer 30: Top token = "9", P(token "9") = 0.585 ← Confidence increases

**Wrong Format (Q&A "Q:...A:"):**
- Layer 20: Top token = "Both", P(token "9") = 0.000
- **Layer 25: Top token = "Both", P(token "9") = 0.003** ← Still hedging with "Both"
- Layer 30: Top token = "They", P(token "9") = 0.087 ← Never commits to "9"

#### Why Layer 25 is Critical

At Layer 25, we observe the exact moment where the two processing paths diverge:

1. **Correct format**: Model commits to token "9" with 22.2% probability (its top prediction)
2. **Wrong format**: Model hedges with "Both" at 36.5% probability, while "9" remains at only 0.3%

This is the decisive moment in processing:
- **Before Layer 25**: Both formats show similar uncertainty patterns
- **At Layer 25**: Correct format commits to "9", wrong format commits to "Both"
- **After Layer 25**: The paths are set - correct format strengthens "9" to 58.5%, wrong format never recovers

#### Discovery Methodology

The discovery came from:
1. Running comprehensive logit lens analysis using `create_final_visualizations.py`
2. Plotting P(token "9") across all 32 layers for both formats
3. Observing the dramatic spike in the correct format starting precisely at Layer 25
4. Examining token evolution tables showing "9" first appearing at Layer 25 for correct format
5. Confirming with direct probability measurements showing the 0.222 probability at Layer 25

This systematic analysis revealed that **Layer 25 is where the model "decides" whether to start the answer with "9" (leading to correct answer) or hedge with "Both" (leading to wrong answer)**.

### 3. Token Evolution Reveals Different Processing Strategies

**Wrong Format Evolution** (Q&A):
```
L0: "greg" → L7: "createAction" → L14: "ApplicationExc" → L20: "Both" → L25: "Both" → L31: ""
```
The model hedges with "Both" at layers 20-25, then produces empty tokens, eventually generating "9.11 is bigger".

**Correct Format Evolution** (Simple):
```
L0: "greg" → L7: "IDO" → L14: "sched" → L20: "neither" → L25: "9" → L31: ""
```
The model commits to "9" at layer 25, leading to the correct "9.8 is bigger" answer.

### 4. Token Preference Analysis

When examining P("11") - P("8"), both formats show minimal preference until late layers:
- Values remain near zero (±0.002) for most layers
- Slight negative values in layers 26-29 for correct format (preferring "8")
- The differences are subtle (×1000 scaling needed for visibility)

## Technical Details

### Model Behavior

1. **The model doesn't directly predict "8" or "11" tokens** - Instead, it generates complete phrases
2. **Empty final layer predictions** - Both formats predict whitespace at layer 31, suggesting generation happens through a different mechanism
3. **Format-dependent activation patterns** - Different prompt structures activate entirely different neural pathways

### Reproduction Instructions

To reproduce the bug consistently:
```python
# Wrong answer setup
prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"
temperature = 0.0
do_sample = False

# Correct answer setup  
prompt = "Which is bigger: 9.8 or 9.11?\nAnswer:"
temperature = 0.0
do_sample = False
```

### Performance Metrics

- **GPU**: NVIDIA A100-SXM4-80GB
- **Memory Usage**: ~30.2 GB
- **Analysis Runtime**: ~24 seconds total
- **Model Loading**: ~10 seconds (checkpoint shards)
- **Per-prompt Analysis**: ~12-16 seconds

## Implications

### 1. Prompt Engineering Matters
Small variations in prompt format can completely change model behavior. The difference between "Q: ... A:" and "... Answer:" determines whether the model gives the correct answer.

### 2. Temperature Effects
Temperature doesn't just affect randomness - at 0.2, it triggers different behavior patterns (empty responses) that might be safety-related.

### 3. Deep Processing Differences
The bug isn't a simple surface-level error. Different formats activate different processing pathways from early layers, with divergence clearly visible by layer 25.

### 4. Irremediable Entanglement
As noted in the submission materials, this bug exhibits "irremediable entanglement" - the neurons involved in producing wrong answers are also essential for correct processing, making surgical fixes impossible.

## Visualizations Created

1. **Token "9" Probability** (`viz_1_token_9_probability_*`)
   - Shows dramatic divergence at layers 25-30
   - Correct format shows strong commitment to "9"

2. **Token Preference** (`viz_2_token_preference_*`)
   - Reveals subtle differences in decimal ending preference
   - Scale adjusted to -2.5 to 2.5 (×1000) for visibility

3. **Token Evolution - Wrong** (`viz_3_token_evolution_wrong_*`)
   - Multi-color visualization of token progression
   - Shows hedging behavior with "Both" tokens

4. **Token Evolution - Correct** (`viz_4_token_evolution_correct_*`)
   - Highlights critical Layer 25 where "9" appears
   - Shows clear progression to correct answer

5. **Combined Comparison** (`viz_5_combined_comparison_*`)
   - Overview of all metrics in one figure
   - Useful for paper presentations

## Conclusions

The decimal comparison bug in Llama-3.1-8B-Instruct is a fascinating example of how LLMs can fail in unexpected ways:

1. **Format Sensitivity**: The exact same question produces opposite answers based on subtle prompt differences
2. **Temperature Dependency**: The bug manifests differently at different temperatures
3. **Deep Neural Divergence**: Different formats activate fundamentally different processing paths
4. **Critical Layer Identification**: Layer 25 is the key divergence point where correct/wrCaong paths separate

This analysis demonstrates that LLM bugs are not simple coding errors but complex emergent behaviors arising from the interaction of training data, model architecture, and prompt formatting. The logit lens technique proves invaluable for understanding these deep processing differences.

## Files in This Directory

- `create_final_visualizations.py` - Script to generate all visualizations
- `logitlens_deterministic.py` - Main analysis script with temperature=0.0
- `logitlens.py` - Original analysis script with temperature=0.2
- `DISCREPANCY_NOTES.md` - Detailed notes on temperature/format dependencies
- `SUMMARY.md` - This comprehensive summary
- `viz_*.png/pdf` - Generated visualizations (5 pairs)

## Key Scripts from Submission Directory That Led to Discoveries

1. **`/home/paperspace/dev/MATS9/submission/test_empty_response_bug.py`**
   - Revealed the temperature dependency (0.0 → wrong answer, 0.2 → empty response)
   - Helped us understand why initial runs showed empty responses

2. **`/home/paperspace/dev/MATS9/submission/sparse_edit_simple.py`**
   - Identified correct vs wrong format patterns
   - Showed that "Q: ... A:" format gives wrong answers
   - Demonstrated that simple "Answer:" format gives correct answers

3. **`/home/paperspace/dev/MATS9/submission/test_verbosity_bug.py`**
   - Additional testing of format variations
   - Confirmed the pattern across different prompting styles

## Recommendations for Future Work

1. **Test more decimal pairs** to see if the pattern holds (e.g., 8.9 vs 8.11)
2. **Analyze other model sizes** (70B, 405B) for similar behaviors
3. **Investigate the "Both" hedging tokens** that appear in wrong formats
4. **Explore why temperature=0.2 triggers empty responses** in chat formats
5. **Study Layer 25's special role** in decimal comparison tasks

---

*Analysis completed: August 11, 2025*  
*Model: meta-llama/Llama-3.1-8B-Instruct*  
*Hardware: NVIDIA A100-SXM4-80GB*