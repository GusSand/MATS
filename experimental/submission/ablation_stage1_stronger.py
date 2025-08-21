#!/usr/bin/env python
"""
Stage 1 Ablation with stronger interventions:
1. Ablate ALL 10 hijacker neurons from chat format
2. If that doesn't work, try negative values
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import os
import json

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

print("🧪 Stage 1: Stronger Hijacker Circuit Ablation")
print("="*70)

# Load the identified circuits
with open("identified_circuits.json", "r") as f:
    circuits = json.load(f)

# Get ALL hijacker neurons from chat format (the buggy one)
all_hijackers = circuits["chat"]["hijacker_cluster"]

print(f"ALL Hijacker Neurons from chat format: {len(all_hijackers)}")
for layer, neuron in sorted(all_hijackers):
    print(f"  L{layer}/N{neuron}")

# Load model
print("\n" + "-"*70)
print("Loading model...")
model_path = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# Global variables for ablation
NEURONS_TO_ABLATE = set()
ABLATION_VALUE = 0.0  # Will be 0 or negative

def ablation_hook(module, input, output, layer_idx):
    """Hook that sets specific neurons to ablation value"""
    if hasattr(output, 'clone'):
        new_output = output.clone()
        
        # Check if this layer has neurons to ablate
        for neuron_layer, neuron_idx in NEURONS_TO_ABLATE:
            if neuron_layer == layer_idx:
                # Set to ablation value
                new_output[:, :, neuron_idx] = ABLATION_VALUE
        
        return new_output
    return output

def run_generation_test(prompt_messages, ablation_neurons=None, ablation_value=0.0, description="Baseline"):
    """Run generation with neuron ablation."""
    global NEURONS_TO_ABLATE, ABLATION_VALUE
    
    print(f"\n{'='*60}")
    print(f"Test: {description}")
    print(f"{'='*60}")
    
    # Set ablation parameters
    ABLATION_VALUE = ablation_value
    if ablation_neurons:
        NEURONS_TO_ABLATE = set(tuple(n) if isinstance(n, list) else n for n in ablation_neurons)
        print(f"Ablating {len(ablation_neurons)} neurons with value: {ablation_value}")
    else:
        NEURONS_TO_ABLATE = set()
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    
    # Register hooks if ablating
    hooks = []
    if ablation_neurons:
        for idx, layer in enumerate(model.model.layers):
            hook = layer.mlp.act_fn.register_forward_hook(
                lambda module, input, output, idx=idx: ablation_hook(module, input, output, idx)
            )
            hooks.append(hook)
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Run 5 times to check consistency
    results = []
    for i in range(5):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.2,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True
            )
        
        # Decode
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract generated part
        if "assistant" in full_text:
            generated_text = full_text.split("assistant")[-1].strip()
        else:
            generated_text = full_text
        
        # Quick check
        if "9.8" in generated_text and "bigger" in generated_text.lower():
            results.append("CORRECT")
        elif "9.11" in generated_text and "bigger" in generated_text.lower():
            results.append("INCORRECT")
        else:
            results.append("UNCLEAR")
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    
    # Report results
    print(f"\nResults over 5 runs: {results}")
    correct_count = results.count("CORRECT")
    incorrect_count = results.count("INCORRECT")
    
    if correct_count > incorrect_count:
        final_result = "CORRECT"
        print(f"✅ FIXED! {correct_count}/5 correct answers")
    elif incorrect_count > correct_count:
        final_result = "INCORRECT"
        print(f"❌ Still buggy: {incorrect_count}/5 incorrect answers")
    else:
        final_result = "MIXED"
        print(f"🤔 Mixed results: {correct_count} correct, {incorrect_count} incorrect")
    
    return {
        "description": description,
        "ablation_value": ablation_value,
        "results": results,
        "final_result": final_result,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count
    }

# Test prompt
test_prompt = [{"role": "user", "content": "Which is bigger: 9.8 or 9.11?"}]

# Run experiments
all_results = []

# BASELINE
print("\n" + "="*70)
print("BASELINE TEST")
baseline = run_generation_test(test_prompt, 
                             ablation_neurons=None,
                             description="Baseline (no ablation)")
all_results.append(baseline)

# ABLATION 1: All 10 neurons to zero
print("\n" + "="*70)
print("ABLATION 1: ALL 10 HIJACKER NEURONS → 0")
ablation1 = run_generation_test(test_prompt,
                               ablation_neurons=all_hijackers,
                               ablation_value=0.0,
                               description="All 10 hijackers → 0")
all_results.append(ablation1)

# If not fixed, try negative values
if ablation1["final_result"] != "CORRECT":
    print("\n" + "="*70)
    print("ABLATION 2: ALL 10 HIJACKER NEURONS → -1")
    ablation2 = run_generation_test(test_prompt,
                                   ablation_neurons=all_hijackers,
                                   ablation_value=-1.0,
                                   description="All 10 hijackers → -1")
    all_results.append(ablation2)
    
    if ablation2["final_result"] != "CORRECT":
        print("\n" + "="*70)
        print("ABLATION 3: ALL 10 HIJACKER NEURONS → -5")
        ablation3 = run_generation_test(test_prompt,
                                       ablation_neurons=all_hijackers,
                                       ablation_value=-5.0,
                                       description="All 10 hijackers → -5")
        all_results.append(ablation3)
        
        if ablation3["final_result"] != "CORRECT":
            print("\n" + "="*70)
            print("ABLATION 4: ALL 10 HIJACKER NEURONS → -10")
            ablation4 = run_generation_test(test_prompt,
                                           ablation_neurons=all_hijackers,
                                           ablation_value=-10.0,
                                           description="All 10 hijackers → -10")
            all_results.append(ablation4)

# Final Summary
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

for result in all_results:
    print(f"\n{result['description']}:")
    print(f"  Final result: {result['final_result']}")
    print(f"  Correct: {result['correct_count']}/5, Incorrect: {result['incorrect_count']}/5")
    print(f"  Details: {result['results']}")

# Save results
with open("stage1_stronger_results.json", "w") as f:
    json.dump({
        "all_hijackers": all_hijackers,
        "results": all_results
    }, f, indent=2)

print(f"\n✅ Results saved to: stage1_stronger_results.json")

# Analysis
print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

success = False
for result in all_results:
    if result["final_result"] == "CORRECT":
        print(f"🎉 SUCCESS! The bug was fixed by: {result['description']}")
        print(f"   Ablation value: {result['ablation_value']}")
        print(f"   This proves the hijacker circuit hypothesis!")
        success = True
        break

if not success:
    print("❌ The hijacker circuit ablation did not fix the bug.")
    print("   Possible explanations:")
    print("   1. The bug involves more neurons than identified")
    print("   2. The bug is in earlier layers (< layer 7)")
    print("   3. The mechanism is more complex than simple activation")
    print("   4. We need different intervention points (e.g., attention heads)")