# Transluce Replication Summary

## Goal
Replicate the Transluce observability interface process to fix the "9.8 vs 9.11" comparison error using concept ablation.

## Key Findings

### 1. Environment Setup ✓
- Successfully set up Python 3.12 environment
- Connected to PostgreSQL database with neuron embeddings
- All dependencies installed correctly

### 2. Critical Discovery: Intervention Method
- **Initial approach**: Set activations to 0.0 → 0% accuracy
- **Correct approach**: Use `quantile * strength` (following observatory server implementation)
- **Best result**: One successful run with strength=-0.1 showed "9.8 is bigger"

### 3. Inconsistent Results
Despite finding the correct intervention method:
- Single runs occasionally work (showed correct answer)
- Multiple run tests show 0-10% accuracy
- Paper reports 76-77% accuracy

### 4. What We Tried
1. **Different concepts**:
   - "bible verses" alone → 0% accuracy
   - Combined "bible verses + dates + phone versions" → 0% accuracy
   
2. **Different intervention strengths**:
   - 0.0 → No effect
   - -0.001 to -0.09 → No effect
   - -0.1 → Occasional success (10-30% in some tests, 0% in others)
   - -0.2 to -0.5 → Generates nonsense
   
3. **Different ablation strategies**:
   - All tokens → 0% accuracy
   - Generation only → 0% accuracy
   - Question tokens only → 0% accuracy

### 5. Possible Reasons for Discrepancy

1. **Neuron Selection**: The paper mentions using the AI Linter to find clusters, not just direct concept search
2. **Model Version**: We're using the same model architecture, but possibly different weights?
3. **Prompt Format**: Slight differences in prompt formatting might matter
4. **Implementation Details**: Some subtle difference in how interventions are applied

### 6. Code Files Created
- `transluce_replication.py` - Main replication script (updated with quantile interventions)
- `quick_neuron_check.py` - Check what neurons are found for concepts
- `quantile_based_ablation.py` - Test quantile-based interventions
- `fine_tuned_ablation.py` - Find optimal strength values
- `combined_concepts_test.py` - Test with all three concepts
- Various other test scripts

### 7. Next Steps
To achieve the paper's 76% accuracy, consider:
1. Using the AI Linter to find neuron clusters (as the paper did)
2. Verifying exact model weights/version
3. Testing more nuanced intervention strategies
4. Examining the exact neurons the paper identified

## Conclusion
We successfully replicated the infrastructure and basic approach, discovering that interventions should use `quantile * strength` rather than zeroing. However, we couldn't consistently achieve the paper's reported 76% accuracy, suggesting there may be subtle implementation differences or that the AI Linter finds better neuron clusters than direct concept search.