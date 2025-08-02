# Complete Decimal Comparison Bug Testing Results

## The Bug
Models are asked: "Which is bigger: 9.8 or 9.11?"
- Correct answer: 9.8
- Bug behavior: Model says 9.11

## Complete Results Table

| Model Family | Model | Parameters | Bug Present | Error Rate | Notes |
|--------------|-------|------------|-------------|------------|-------|
| **GPT-2** | GPT-2 | 124M | ❌ No | 0% | Gives unclear/evasive responses |
| | GPT-2-medium | 355M | ❌ No | 0% | Sometimes correct, no consistent bug |
| **Pythia** | **Pythia-160M** | 160M | **✅ Yes** | **40%** | **Bug present! Says 9.11** |
| | Pythia-410M | 410M | ❌ No | 0% | 100% correct |
| | Pythia-1B | 1B | ❌ No | 0% | 100% correct |
| **OPT** | OPT-125M | 125M | ❌ No | 0% | 100% correct |
| | OPT-350M | 350M | ❌ No | 0% | 100% correct |
| | OPT-1.3B | 1.3B | ❌ No | 0% | 100% correct with Q&A format |
| **GPT-Neo** | GPT-Neo-125M | 125M | ❌ No | 0% | 100% correct with proper format |
| | GPT-Neo-1.3B | 1.3B | ❌ No | 0% | 100% correct with Q&A format |
| **Llama** | Llama-3-8B | 8B | ⚠️ | N/A | Requires HF access |
| | **Llama-3.1-8B-Instruct** | 8B | **✅ Yes** | **~100%** | **Strong bug (from llama/ tests)** |
| **Gemma** | All Gemma models | 2B-9B | ⚠️ | N/A | All require HF access |

## Key Findings

### 1. **Bug is NOT correlated with model scale**
- Smallest tested model with bug: Pythia-160M (40% error)
- Larger Pythia models (410M, 1B): No bug
- Many small models (GPT-2, OPT-125M, GPT-Neo-125M): No bug

### 2. **Pattern Analysis**
- **Pythia-160M**: Only base model showing the bug
- **Llama-3.1-8B-Instruct**: Instruction-tuned model with strong bug
- All other base models tested: No bug

### 3. **Possible Hypotheses**

#### A. Training Data Quality
- Pythia-160M might have insufficient training or poor data filtering
- Larger Pythia models had better training

#### B. Instruction Tuning Effect
- Llama-3.1-8B-Instruct shows very strong bug
- Base models generally don't show the bug (except Pythia-160M)
- Instruction tuning might reinforce certain patterns

#### C. Model-Specific Factors
- Could be related to specific training hyperparameters
- Tokenization differences
- Architecture details

### 4. **Prompt Sensitivity**
Different models responded better to different prompts:
- GPT-2: Unclear with most formats
- Pythia: Best with "Question: ... Answer:" format
- OPT: Consistent across formats
- Llama: Requires specific chat template

## Conclusion

The decimal comparison bug is **not simply a function of scale**. Instead, it appears to be:
1. Present in specific undertrained models (Pythia-160M)
2. Strongly present in instruction-tuned models (Llama-3.1-8B-Instruct)
3. Absent in most well-trained base models regardless of size

This suggests the bug might emerge from:
- Poor training data or insufficient training (Pythia-160M)
- Instruction tuning that reinforces problematic patterns (Llama-3.1-8B-Instruct)
- NOT from model scale alone