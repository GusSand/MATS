#!/usr/bin/env python
"""
Robust Stage 1 Ablation Test with many iterations to get accurate bug rates
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import os
import json
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

print("🧪 Stage 1: Robust Ablation Test (20 iterations)")
print("="*70)

# Load the identified circuits
with open("identified_circuits.json", "r") as f:
    circuits = json.load(f)

# Get ALL hijacker neurons from chat format
all_hijackers = circuits["chat"]["hijacker_cluster"]

print(f"ALL Hijacker Neurons from chat format: {len(all_hijackers)}")
for layer, neuron in sorted(set(tuple(n) for n in all_hijackers)):
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

# Global variables
NEURONS_TO_ABLATE = set()
ABLATION_VALUE = 0.0

def ablation_hook(module, input, output, layer_idx):
    """Hook that sets specific neurons to ablation value"""
    if hasattr(output, 'clone'):
        new_output = output.clone()
        
        for neuron_layer, neuron_idx in NEURONS_TO_ABLATE:
            if neuron_layer == layer_idx:
                new_output[:, :, neuron_idx] = ABLATION_VALUE
        
        return new_output
    return output

def run_multiple_tests(prompt_messages, ablation_neurons=None, ablation_value=0.0, 
                      description="Test", num_iterations=20):
    """Run generation test multiple times to get accurate statistics."""
    global NEURONS_TO_ABLATE, ABLATION_VALUE
    
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    
    # Set ablation parameters
    ABLATION_VALUE = ablation_value
    if ablation_neurons:
        NEURONS_TO_ABLATE = set(tuple(n) for n in ablation_neurons)
        print(f"Ablating {len(NEURONS_TO_ABLATE)} unique neurons with value: {ablation_value}")
    else:
        NEURONS_TO_ABLATE = set()
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize once
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Run multiple iterations
    results = []
    generated_texts = []
    
    for i in tqdm(range(num_iterations), desc="Running tests"):
        # Register hooks
        hooks = []
        if ablation_neurons:
            for idx, layer in enumerate(model.model.layers):
                hook = layer.mlp.act_fn.register_forward_hook(
                    lambda module, input, output, idx=idx: ablation_hook(module, input, output, idx)
                )
                hooks.append(hook)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.2,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True
            )
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        
        # Decode
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract generated part
        if "assistant" in full_text:
            generated_text = full_text.split("assistant")[-1].strip()
        else:
            generated_text = full_text
        
        generated_texts.append(generated_text)
        
        # Analyze
        generated_lower = generated_text.lower()
        
        # More careful analysis
        if "9.11" in generated_text and "is bigger" in generated_lower:
            # Check if 9.11 comes before "is bigger"
            pos_911 = generated_text.find("9.11")
            pos_bigger = generated_lower.find("is bigger")
            if pos_911 < pos_bigger and pos_bigger - pos_911 < 20:
                results.append("INCORRECT")
            else:
                results.append("UNCLEAR")
        elif "9.8" in generated_text and "is bigger" in generated_lower:
            # Check if 9.8 comes before "is bigger"
            pos_98 = generated_text.find("9.8")
            pos_bigger = generated_lower.find("is bigger")
            if pos_98 < pos_bigger and pos_bigger - pos_98 < 20:
                results.append("CORRECT")
            else:
                results.append("UNCLEAR")
        else:
            results.append("UNCLEAR")
    
    # Calculate statistics
    correct_count = results.count("CORRECT")
    incorrect_count = results.count("INCORRECT")
    unclear_count = results.count("UNCLEAR")
    
    error_rate = (incorrect_count / num_iterations) * 100
    correct_rate = (correct_count / num_iterations) * 100
    
    print(f"\nResults: {correct_count} correct, {incorrect_count} incorrect, {unclear_count} unclear")
    print(f"Error rate: {error_rate:.1f}%")
    print(f"Correct rate: {correct_rate:.1f}%")
    
    # Show some examples
    print("\nExample outputs:")
    for i in range(min(3, len(generated_texts))):
        print(f"  {i+1}. {generated_texts[i][:60]}... [{results[i]}]")
    
    return {
        "description": description,
        "ablation_value": ablation_value,
        "num_iterations": num_iterations,
        "results": results,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "unclear_count": unclear_count,
        "error_rate": error_rate,
        "correct_rate": correct_rate,
        "example_outputs": generated_texts[:5]
    }

# Test prompt
test_prompt = [{"role": "user", "content": "Which is bigger: 9.8 or 9.11?"}]

# Run experiments
all_results = []

# BASELINE
print("\n" + "="*70)
baseline = run_multiple_tests(test_prompt, 
                            ablation_neurons=None,
                            description="BASELINE (no ablation)",
                            num_iterations=20)
all_results.append(baseline)

# Only continue if baseline shows the bug
if baseline["error_rate"] > 30:  # At least 30% error rate to consider it buggy
    print(f"\n✓ Baseline shows bug with {baseline['error_rate']:.1f}% error rate")
    
    # ABLATION 1: All neurons to zero
    print("\n" + "="*70)
    ablation1 = run_multiple_tests(test_prompt,
                                 ablation_neurons=all_hijackers,
                                 ablation_value=0.0,
                                 description="ABLATION: All 10 hijackers → 0",
                                 num_iterations=20)
    all_results.append(ablation1)
    
    # Check if fixed
    if ablation1["error_rate"] < baseline["error_rate"] / 2:
        print(f"\n🎉 SUCCESS! Error rate reduced from {baseline['error_rate']:.1f}% to {ablation1['error_rate']:.1f}%")
    else:
        print(f"\n❌ Not fixed. Trying negative values...")
        
        # Try negative values
        for neg_val in [-1.0, -5.0, -10.0]:
            print("\n" + "="*70)
            ablation = run_multiple_tests(test_prompt,
                                        ablation_neurons=all_hijackers,
                                        ablation_value=neg_val,
                                        description=f"ABLATION: All 10 hijackers → {neg_val}",
                                        num_iterations=20)
            all_results.append(ablation)
            
            if ablation["error_rate"] < baseline["error_rate"] / 2:
                print(f"\n🎉 SUCCESS! Error rate reduced from {baseline['error_rate']:.1f}% to {ablation['error_rate']:.1f}%")
                break
else:
    print(f"\n⚠️ Baseline doesn't show consistent bug (only {baseline['error_rate']:.1f}% error rate)")

# Final Summary
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

for result in all_results:
    print(f"\n{result['description']}:")
    print(f"  Error rate: {result['error_rate']:.1f}% ({result['incorrect_count']}/{result['num_iterations']} incorrect)")
    print(f"  Correct rate: {result['correct_rate']:.1f}% ({result['correct_count']}/{result['num_iterations']} correct)")

# Save results
with open("stage1_robust_results.json", "w") as f:
    json.dump({
        "all_hijackers": list(set(tuple(n) for n in all_hijackers)),
        "results": all_results
    }, f, indent=2)

print(f"\n✅ Results saved to: stage1_robust_results.json")

# Analysis
if len(all_results) > 1:
    baseline_error = all_results[0]["error_rate"]
    best_result = min(all_results[1:], key=lambda x: x["error_rate"])
    
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    if best_result["error_rate"] < baseline_error / 2:
        print(f"✅ ABLATION SUCCESSFUL!")
        print(f"   Best result: {best_result['description']}")
        print(f"   Error reduction: {baseline_error:.1f}% → {best_result['error_rate']:.1f}%")
        print(f"   This confirms the hijacker circuit hypothesis!")
    else:
        print(f"❌ ABLATION UNSUCCESSFUL")
        print(f"   Baseline error: {baseline_error:.1f}%")
        print(f"   Best ablation error: {best_result['error_rate']:.1f}%")
        print(f"   The identified neurons may not be causal for the bug")