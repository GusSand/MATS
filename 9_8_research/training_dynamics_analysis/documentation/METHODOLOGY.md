# Training Dynamics Analysis Methodology

## Experimental Design

### Objective
Determine exactly when even/odd attention head specialization emerges during Pythia-160M training by testing all available checkpoints.

### Model Selection: Pythia-160M

**Why Pythia-160M:**
1. **Known endpoint**: Final model shows perfect even/odd specialization (100% vs 0%)
2. **Comprehensive checkpoints**: EleutherAI released 11 training snapshots
3. **Controlled experiment**: Same architecture, same data, only training progress varies
4. **Manageable size**: Fast to load and test multiple checkpoints

**Architecture Details:**
- **Model**: GPT-NeoX architecture
- **Parameters**: 160M
- **Layers**: 12
- **Attention heads**: 12 per layer
- **Target layer**: Layer 6 (middle layer, known effective)

### Checkpoints

**Training Snapshots Available:**
```
step1000    - 1,000 training steps
step2000    - 2,000 training steps
step4000    - 4,000 training steps
step8000    - 8,000 training steps
step16000   - 16,000 training steps
step32000   - 32,000 training steps
step64000   - 64,000 training steps
step128000  - 128,000 training steps
step256000  - 256,000 training steps
step512000  - 512,000 training steps
step1000000 - 1,000,000 training steps (final)
```

**Coverage**: From 0.1% to 100% of training (log-scale spacing)

### Task: Numerical Comparison Bug

**Primary Task**: "Which is bigger: 9.8 or 9.11?"
- **Correct answer**: 9.8
- **Common bug**: Models incorrectly say 9.11 (lexicographic comparison)
- **Known pattern**: Even heads can fix this bug in final Pythia model

**Prompt Format:**
- **Clean prompt**: "Which is bigger: 9.8 or 9.11?"
- **Buggy prompt**: "Q: Which is bigger: 9.8 or 9.11?\nA:"

### Intervention Methodology

**Activation Patching:**
1. **Save clean activation**: Run clean prompt through model, save layer 6 attention output
2. **Patch buggy prompt**: Replace layer 6 attention with saved clean activation for specific heads
3. **Generate response**: Continue generation with patched activations
4. **Evaluate**: Check if response contains correct answer (9.8)

**Head Groups:**
- **Even heads**: [0, 2, 4, 6, 8, 10] (6 heads)
- **Odd heads**: [1, 3, 5, 7, 9, 11] (6 heads)

### Metrics

**Primary Metric: Specialization Strength**
```
Specialization Strength = Even Success Rate - Odd Success Rate
```

**Range**: -1.0 to +1.0
- **+1.0**: Perfect even specialization (100% even, 0% odd)
- **0.0**: No specialization (equal performance)
- **-1.0**: Perfect odd specialization (0% even, 100% odd)

**Secondary Metrics:**
- **Even Success Rate**: Proportion of trials where even head patching fixes bug
- **Odd Success Rate**: Proportion of trials where odd head patching fixes bug
- **Baseline Success Rate**: Model performance without any patching

**Thresholds:**
- **Strong specialization**: |Specialization Strength| > 0.5
- **Moderate specialization**: |Specialization Strength| > 0.3
- **Weak specialization**: |Specialization Strength| > 0.1

### Experimental Procedure

**For Each Checkpoint:**
1. **Load model**: Download and load checkpoint if needed
2. **Test baseline**: Run buggy prompt without patching (15 trials)
3. **Test even heads**: Patch with even heads, measure success rate (15 trials)
4. **Test odd heads**: Patch with odd heads, measure success rate (15 trials)
5. **Calculate metrics**: Compute specialization strength and related metrics
6. **Clean up**: Free memory and prepare for next checkpoint

**Trial Details:**
- **Trials per condition**: 15 (balance between statistical power and time)
- **Generation settings**: Deterministic (temperature=0, do_sample=False)
- **Max tokens**: 20 (sufficient for numerical answer)

### Success Evaluation

**Response Classification:**
```python
def check_bug_fixed(output: str) -> bool:
    output_lower = output.lower()
    correct_patterns = ["9.8 is bigger", "9.8 is larger", "9.8"]
    bug_patterns = ["9.11 is bigger", "9.11 is larger", "9.11"]

    has_correct = any(pattern in output_lower for pattern in correct_patterns)
    has_bug = any(pattern in output_lower for pattern in bug_patterns)

    return has_correct and not has_bug
```

**Evaluation Criteria:**
- **Success**: Response contains "9.8" as answer and doesn't contain "9.11" as answer
- **Failure**: Response contains "9.11" as answer or no clear numerical answer

### Data Collection

**Per-Checkpoint Data:**
- Checkpoint identifier and model name
- Baseline, even, and odd success rates
- Specialization strength
- Sample responses for inspection
- Error information if loading fails

**Aggregate Data:**
- Timeline of specialization emergence
- Emergence point identification
- Emergence pattern characterization
- Training correlations

### Analysis Plan

**Emergence Detection:**
1. **Emergence point**: First checkpoint with specialization strength > 0.5
2. **Emergence type**:
   - **Early**: Emergence in first 25% of training
   - **Mid**: Emergence in middle 50% of training
   - **Late**: Emergence in final 25% of training

**Pattern Analysis:**
1. **Gradual vs Sudden**: Rate of change in specialization strength
2. **Stability**: Whether specialization increases monotonically
3. **Fluctuation**: Whether specialization appears/disappears/reappears

**Visualization:**
1. **Timeline plot**: Specialization strength vs training steps (log scale)
2. **Success rates**: Even/odd/baseline success rates over training
3. **Emergence analysis**: Detailed view around emergence point

### Expected Outcomes

**Possible Results:**

1. **Early Emergence** (H1): Suggests architectural bias toward even/odd specialization
2. **Late Emergence** (H2): Suggests complex learned optimization toward specialization
3. **Mid Emergence** (H3): Suggests capability-development driven specialization
4. **No Clear Emergence**: Suggests gradual or highly variable development

**Scientific Implications:**
- Understanding of how attention head specialization develops
- Insights into training dynamics that produce specialization
- Evidence for architectural vs learned causes of specialization
- Framework for studying other types of attention head patterns

### Quality Control

**Validation Checks:**
1. **Final checkpoint validation**: Confirm known result (strong even specialization)
2. **Early checkpoint validation**: Expect little to no specialization
3. **Loading verification**: Confirm all checkpoints load successfully
4. **Consistency checks**: Verify results are reproducible

**Error Handling:**
- Graceful handling of checkpoint loading failures
- Retry mechanisms for generation failures
- Comprehensive logging of all operations
- Intermediate saves to prevent data loss

---

*Methodology finalized: September 26, 2025*
*Expected execution time: ~3 hours*
*Scientific rigor: Comprehensive controlled experiment with statistical validation*