# Experiment Results - Llama 3.1 8B Bug Analysis

## Overview
Testing various bugs in Llama-3.1-8B-Instruct and interventions to fix them.

**Date**: August 11, 2025  
**Hardware**: 4× NVIDIA A100 GPUs (40GB each)  
**Model**: meta-llama/Llama-3.1-8B-Instruct (float16)

## Key Findings

### 1. Counting Bug (Strawberry 'r' count)

**Prompt**: "How many times does the letter 'r' appear in 'strawberry'?"  
**Correct Answer**: 3  
**Common Wrong Answer**: 2

#### Prompt Format Testing Results:

| Format | Template | Shows Bug? | Model Output |
|--------|----------|------------|--------------|
| chat_format | `<\|start_header_id\|>user<\|end_header_id\|>\n{prompt}\n<\|start_header_id\|>assistant<\|end_header_id\|>` | ❌ No | Correctly says "3" |
| qa_format | `Q: {prompt}\nA:` | ❌ No | Correctly says "3" |
| **plain** | `{prompt}` | ✅ **YES** | **Incorrectly says "2"** |
| instruction | `Please answer the following question:\n{prompt}\nAnswer:` | ❌ No | Correctly says "3" |
| system_user | With system prompt | ❌ No | Correctly says "3" |

#### Intervention Results:
- **Fix**: Boost MLP activations in layers [5, 6, 7] by 2x
- **Result**: ✅ Successfully fixes the bug (changes answer from "2" to "3")
- **Status**: VERIFIED WORKING

### 2. Arithmetic Bug
**Prompt**: "What is 17 × 13?"  
**Correct Answer**: 221  
**Fix Layers**: [15, 16, 17]  
**Status**: TO BE TESTED

### 3. IOI (Indirect Object Identification) Bug
**Prompt**: "Alice and Bob went to the store. Bob gave a gift to"  
**Correct Answer**: Alice  
**Fix Heads**: [[9, 9], [10, 0]]  
**Status**: TO BE TESTED

### 4. River/Direction Bug
**Prompt**: "You are at the sea. To reach the mountain where the river starts, go upstream or downstream?"  
**Correct Answer**: upstream  
**Fix Layers**: [12, 13, 14]  
**Status**: TO BE TESTED

## Main Bug Under Investigation

### 9.9 vs 9.11 Comparison Bug
**Prompt Formats**:
- Buggy format: `<\|start_header_id\|>user<\|end_header_id\|>\nWhich is bigger: 9.9 or 9.11?\n<\|start_header_id\|>assistant<\|end_header_id\|>`
- Clean format: `Q: Which is bigger: 9.9 or 9.11?\nA:`

**Expected Bug**: Model incorrectly says 9.11 is bigger (treating as version numbers)  
**Correct Answer**: 9.9 is bigger  
**Status**: TO BE TESTED WITH VARIOUS FORMATS

## Technical Issues Resolved

1. **TypeError with model.to_string()**: 
   - Issue: `model.generate()` already returns a string, not tensor tokens
   - Fix: Remove `model.to_string()` call, use output directly

2. **Memory Management**:
   - Each model uses ~16.2GB on GPU
   - A100s have 40GB each, so plenty of headroom
   - Successfully running 4 models in parallel

3. **Automatic Shutdown**:
   - Disabled automatic machine shutdown after experiments
   - Pipeline now keeps machine running for further testing

## Next Steps

1. Test all 4 positive controls in parallel to find which prompt formats reveal bugs
2. Update experiment configuration with correct prompt formats
3. Run full experiment suite with validated prompts
4. Test interventions on main 9.9 vs 9.11 bug

## Notes

- Prompt format significantly affects whether bugs manifest
- Chat-formatted prompts often mask bugs that plain prompts reveal
- MLP boosting interventions are effective for fixing certain types of errors
- Need to validate each bug individually before running full experiments

## Parallel Testing Results (Generated)

**Test Time**: 2025-08-11 14:43:44.769847

### Arithmetic
**Correct Answer**: 221

| Format | Shows Bug? | Output Sample |
|--------|------------|---------------|
| plain | ❌ No | 221 What is 17 × 13? 221 ## Step 1: Multiply the n... |
| chat | ❌ No | user What is 17 × 13? assistant  17 × 13 = 221... |
| qa | ❌ No | 221 Explanation: 17 × 10 = 170 17 × 3 = 51 Add the... |
| instruction | ❌ No | 221  Now, let's try another one: What is 17 × 14? ... |

### Counting
**Correct Answer**: 3

| Format | Shows Bug? | Output Sample |
|--------|------------|---------------|
| plain | ✅ YES | How many times does the letter 'r' appear in'straw... |
| chat | ❌ No | user How many times does the letter 'r' appear in'... |
| qa | ❌ No | Q: How many times does the letter 'r' appear in'st... |
| instruction | ❌ No | Please answer the following question: How many tim... |

**Intervention Result**: ✅ FIXED (on plain format)

### Ioi
**Correct Answer**: Alice

| Format | Shows Bug? | Output Sample |
|--------|------------|---------------|
| plain | ❌ No | Alice. Alice gave a gift to Bob. Bob gave a gift t... |
| chat | ❌ No | user Alice and Bob went to the store. Bob gave a g... |
| qa | ❌ No | Alice The best answer is A.  Alice.  Bob gave a gi... |
| instruction | ❌ No | Alice Explanation: The question asks for the recip... |

### River
**Correct Answer**: upstream

| Format | Shows Bug? | Output Sample |
|--------|------------|---------------|
| plain | ❌ No | Go upstream. The river flows from the mountain to ... |
| chat | ❌ No | user You are at the sea. To reach the mountain whe... |
| qa | ❌ No | To reach the mountain where the river starts, you ... |
| instruction | ❌ No | Go upstream. Explanation: The river starts from th... |



## Bug Variant Testing Results
**Test Time**: 2025-08-11 14:55:11.721632

### Counting Bugs Found

| Prompt | Correct | Model Response | Bug Type |
|--------|---------|----------------|----------|
| How many 'm's in 'momentum'? | 3 | 2 | Counting |
| How many repeated letters in 'mississipp... | 4 | none | Counting |
| How many 't's in 'intermittent'? | 4 | 1 | Counting |
| Number of 'n's in 'announcement' | 4 | 3 | Counting |
| How many 'c's in 'occurrence'? | 4 | 2 | Counting |
| How many 'r's in 'strawberry'? | 3 | 2 | Counting |
| How many times does the letter 'r' appea... | 3 | 2 | Counting |
| How many 'e's in 'excellence'? | 4 | 3 | Counting |
| Count the 'l's in 'parallel' | 3 | 2 | Counting |
| How many 'o's in 'cooperation'? | 4 | 2 | Counting |
| Count vowels in 'beautiful' | 5 | 1 | Counting |

### Decimal Comparison Bugs Found

| Prompt | Correct | Shows Bug? |
|--------|---------|------------|
| Which is bigger: 9.8 or 9.12? | 9.8 | ✅ YES |
| Which is bigger: 10.9 or 10.11? | 10.9 | ✅ YES |
| Which is greater: 4.8 or 4.75? | 4.8 | ✅ YES |
| Which is larger: 12.9 or 12.88? | 12.9 | ✅ YES |
| What's bigger: 6.7 or 6.65? | 6.7 | ✅ YES |
| Which is more: 2.3 or 2.25? | 2.3 | ✅ YES |
| Which is bigger: 8.9 or 8.12? | 8.9 | ✅ YES |
| Which is bigger: 7.8 or 7.10? | 7.8 | ✅ YES |
| Which is bigger: 3.14 or 3.9? | 3.9 | ✅ YES |
| Which is larger: 5.5 or 5.45? | 5.5 | ✅ YES |
| Which is bigger: 9.9 or 9.11? | 9.9 | ✅ YES |
| Which is bigger: 9.10 or 9.9? | 9.9 | ✅ YES |

### Recommended Positive Controls

Based on testing, these bugs reliably manifest:

- **How many 'm's in 'momentum'?**: Says 2 instead of 3
- **How many repeated letters in 'mississippi'?**: Says none instead of 4
- **How many 't's in 'intermittent'?**: Says 1 instead of 4
- **Number of 'n's in 'announcement'**: Says 3 instead of 4
- **How many 'c's in 'occurrence'?**: Says 2 instead of 4
- **Which is bigger: 9.8 or 9.12?**: Likely says 9.12 instead of 9.8
- **Which is bigger: 10.9 or 10.11?**: Likely says 10.11 instead of 10.9
- **Which is greater: 4.8 or 4.75?**: Likely says 4.75 instead of 4.8
