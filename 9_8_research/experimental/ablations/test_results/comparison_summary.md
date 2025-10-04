# Family-Dependent Hypothesis Test Results

## Hypothesis

- **Gemma**: Instruction tuning FIXES the decimal bug
- **Llama**: Instruction tuning CREATES the decimal bug

## Results Summary

| Model | Type | 9.8 vs 9.11 | Overall Accuracy | Hypothesis Support |
|-------|------|-------------|------------------|--------------------|
| Gemma-2-2B Base | base | 100.0% | 61.9% | ✅ Base has bug |
| Gemma-2-2B-IT | instruct | 0.0% | 3.1% | ❓ Mixed |
| Llama-3.1-8B Base | base | 90.0% | 88.1% | ✅ Base is better |
| Llama-3.1-8B-Instruct | instruct | 0.0% | 4.4% | ✅ IT creates bug |
