# Decimal Comparison Bug Testing Results

## Summary of Results

| Model | Parameters | Bug Present | Error Rate | Notes |
|-------|------------|-------------|------------|-------|
| GPT-2 | 124M | ❌ No | 0% | Gives unclear responses |
| GPT-2-medium | 355M | ❌ No | 0% | Sometimes correct |
| **Pythia-160M** | 160M | **✅ Yes** | **40%** | **Bug present!** |
| Pythia-410M | 410M | ❌ No | 0% | 100% correct |
| Pythia-1B | 1B | ❌ No | 0% | 100% correct |
| Gemma-2-2B | 2B | ⚠️ | N/A | Requires access |
| Gemma-2-9B | 9B | ⚠️ | N/A | Requires access |
| Llama-3-8B | 8B | ⚠️ | N/A | Requires access |
| Llama-3.1-8B-Instruct | 8B | ✅ Yes | ~100% | Strong bug (from llama tests) |

## Key Insights

1. **Bug is NOT simply a function of scale**: 
   - Pythia-160M (smallest) has the bug
   - Pythia-410M and 1B (larger) don't have the bug

2. **Possible factors**:
   - Training data quality/curation
   - Model architecture details
   - Training methodology
   - Instruction tuning (Llama-3.1-Instruct shows strong bug)

3. **Pattern observed**:
   - Base models: Mixed results (Pythia-160M has bug, others don't)
   - Instruction-tuned models: Strong bug (Llama-3.1-8B-Instruct)

## Pythia Results Detail

```
Pythia-160M: 40% error rate (says "9.11 is bigger")
Pythia-410M: 0% error rate (correctly says "9.8 is bigger")  
Pythia-1B: 0% error rate (correctly says "9.8 is bigger")
```

This suggests the decimal comparison bug might be:
1. Present in some smaller models due to insufficient training
2. Corrected with moderate scale increase
3. Re-introduced during instruction tuning (as seen in Llama-3.1-8B-Instruct)