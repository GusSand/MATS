# Discrepancy Analysis: Chat Format Bug Behavior

## Summary of Discrepancy

There's a significant discrepancy in how the decimal comparison bug manifests:

1. **Submission claims**: Chat format → Wrong answer ("9.11 is bigger"), Simple format → Correct answer ("9.8 is bigger")
2. **Our tests show**: Chat format → Empty response, Simple format → Wrong answer ("9.11 is bigger")
3. **BUT**: When running submission scripts, they DO get wrong answers from chat format!

## Root Cause Identified: Temperature Settings

### Key Finding
The critical difference is **temperature=0.0 (deterministic) vs temperature=0.2 (stochastic)**:

1. **Temperature=0.0 (deterministic)**: 
   - Chat format → **Wrong answer** ("9.11 is bigger than 9.8")
   - Used in submission scripts
   - File: `record_neurons_chat_template.py` line 74: `temperature=0.0`

2. **Temperature=0.2 (stochastic)**:
   - Chat format → **Empty response**
   - Used in our tests (as requested by user)
   - This is what we observed in our testing

## Evidence

### From Submission Scripts (temperature=0.0)
```python
# record_neurons_chat_template.py
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    temperature=0.0,  # DETERMINISTIC!
    pad_token_id=tokenizer.eos_token_id,
    do_sample=False,  # NO SAMPLING
)
# Result: "9.11 is bigger than 9.8."
```

### From Our Tests (temperature=0.2)
```python
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    temperature=0.2,  # STOCHASTIC!
    do_sample=True,   # SAMPLING ENABLED
)
# Result: Empty response
```

## Bug Behavior Matrix

| Format | Temperature | Result |
|--------|------------|--------|
| Chat Template | 0.0 | Wrong answer: "9.11 is bigger" |
| Chat Template | 0.2 | Empty response |
| Simple Q&A | 0.0 | Wrong answer: "9.11 is bigger" |
| Simple Q&A | 0.2 | Wrong answer: "9.11 is bigger" |

## Why This Matters

1. **The bug manifests differently** depending on generation parameters
2. **Temperature affects chat format more** than simple format
3. **At temperature=0.0**, the chat format reliably produces the wrong answer
4. **At temperature=0.2**, the chat format produces empty responses (possible safety filtering)

## Reconciliation with Documentation

The submission documentation states:
- "Chat Template: 90% bug rate" 
- "Simple Format: 0% bug rate"

This appears to be based on **temperature=0.0 testing**, where:
- Chat format consistently gives wrong answers
- The "Simple Format: 0% bug rate" claim needs verification

## Testing Anomaly

When we tested "Simple format" with both temperatures, it consistently gave **wrong answers**, not correct ones. This suggests either:
1. The submission used a different "simple format" prompt
2. There was an error in the documentation
3. Model behavior has changed

## The Actual Prompts Used

### Chat Format (from submission)
```python
messages = [{"role": "user", "content": "Which is bigger: 9.8 or 9.11?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# Results in: <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date...
```

### Simple Format Variants
1. **Q&A format**: "Q: Which is bigger: 9.8 or 9.11?\nA:"
2. **Direct format**: "Which is bigger: 9.8 or 9.11?\nAnswer:"

Both "simple" formats give wrong answers in our tests.

## Conclusion

The discrepancy arises from:
1. **Different temperature settings** (0.0 vs 0.2)
2. **Possible confusion about what constitutes "simple format"**
3. **Temperature-dependent behavior** in chat templates

The decimal comparison bug is **real and reproducible**, but its manifestation depends critically on:
- Exact prompt format
- Temperature setting
- Sampling parameters

## Recommendations

1. **For reproducing submission results**: Use `temperature=0.0, do_sample=False`
2. **For understanding bug behavior**: Test across multiple temperatures
3. **For paper clarity**: Specify exact generation parameters used
4. **For robustness**: The bug exists across settings but manifests differently

## Key Takeaway

The "empty response" behavior at temperature=0.2 might actually be the model's safety system preventing it from giving confidently wrong answers when there's uncertainty. At temperature=0.0, it deterministically follows the wrong path without hesitation.