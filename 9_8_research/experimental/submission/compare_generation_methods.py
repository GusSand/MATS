#!/usr/bin/env python
"""
Compare generation methods between verify_llama_bug.py and our neuron recording scripts
"""

print("="*80)
print("KEY DIFFERENCES IN GENERATION METHODS")
print("="*80)

print("\n1. TEMPERATURE SETTING:")
print("-"*40)
print("verify_llama_bug.py: temperature=0.2")
print("record_neurons_qa_correct.py (initial): temperature=0.2")
print("record_neurons_qa_correct.py (modified): temperature=0.7")
print("\n→ Higher temperature (0.7) makes outputs more random")

print("\n2. GENERATION PARAMETERS:")
print("-"*40)
print("\nverify_llama_bug.py uses:")
print("- do_sample=temperature > 0 (True when temp=0.2)")
print("- No return_dict_in_generate")
print("- No output_scores")
print("- Simple generation")

print("\nrecord_neurons scripts use:")
print("- do_sample=True")
print("- return_dict_in_generate=True")
print("- output_scores=True")
print("- More complex generation for neuron tracking")

print("\n3. TOKENIZATION DIFFERENCES:")
print("-"*40)
print("\nverify_llama_bug.py:")
print("- Uses skip_special_tokens=True in decode")
print("- Returns full text including prompt")

print("\nrecord_neurons scripts:")
print("- Also uses skip_special_tokens=True")
print("- But extracts only generated portion")

print("\n4. CRITICAL FINDING:")
print("-"*40)
print("The main issue is likely the temperature!")
print("- verify_llama_bug.py uses 0.2 (more deterministic)")
print("- We changed to 0.7 to find correct answers (more random)")
print("- This explains why verify_llama_bug.py gets more consistent results")

print("\n5. RECOMMENDATIONS:")
print("-"*40)
print("1. Use temperature=0.2 to match verify_llama_bug.py behavior")
print("2. Run many more iterations to find correct answers")
print("3. Or use temperature=0.0 for fully deterministic output")