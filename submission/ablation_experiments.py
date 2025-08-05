#!/usr/bin/env python
"""
Ablation experiments using nnsight to test the hijacker circuit hypothesis.

Stage 1: Ablate core hijacker neurons (common to both formats)
Stage 2: Ablate ONLY tie-breaker neurons (unique to buggy format)

Based on our circuit identification, we'll test if we can fix the decimal bug
by precisely targeting specific neurons.
"""

import torch
from nnsight import LanguageModel
import warnings
import os
import json
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("🧪 Ablation Experiments: Hijacker Circuit vs Tie-Breaker Neurons")
print("="*70)

# Load the identified circuits from our previous analysis
with open("identified_circuits.json", "r") as f:
    circuits = json.load(f)

# Extract neuron lists
chat_hijackers = circuits["chat"]["hijacker_cluster"]
simple_hijackers = circuits["simple"]["hijacker_cluster"]

# Find core hijacker neurons (common to both formats)
# Convert to sets of tuples for comparison
chat_set = set(tuple(n) for n in chat_hijackers)
simple_set = set(tuple(n) for n in simple_hijackers)

# Core neurons appear in BOTH formats
core_hijackers = list(chat_set.intersection(simple_set))

# Tie-breaker neurons appear ONLY in chat format
tiebreaker_neurons = list(chat_set - simple_set)

print(f"Core Hijacker Neurons (in both formats): {len(core_hijackers)}")
for layer, neuron in sorted(core_hijackers):
    print(f"  L{layer}/N{neuron}")

print(f"\nTie-breaker Neurons (chat format only): {len(tiebreaker_neurons)}")
for layer, neuron in sorted(tiebreaker_neurons):
    print(f"  L{layer}/N{neuron}")

# Load model using nnsight
print("\n" + "-"*70)
print("Loading model with nnsight...")
model = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", device_map="auto")

# Test prompts - we'll use chat template since it exhibits the bug
test_prompt_template = [{"role": "user", "content": "Which is bigger: 9.8 or 9.11?"}]

def run_generation_test(model, prompt_messages, ablation_neurons=None, description="Baseline"):
    """
    Run generation with optional neuron ablation.
    
    Args:
        model: nnsight LanguageModel
        prompt_messages: Chat messages for template
        ablation_neurons: List of (layer, neuron) tuples to ablate
        description: Description of this test
    
    Returns:
        dict with generated text and analysis
    """
    print(f"\n{'='*50}")
    print(f"Test: {description}")
    print(f"{'='*50}")
    
    if ablation_neurons:
        print(f"Ablating {len(ablation_neurons)} neurons:")
        for layer, neuron in sorted(ablation_neurons):
            print(f"  L{layer}/N{neuron}")
    
    # Apply chat template
    prompt = model.tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize
    inputs = model.tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate with or without ablations
    if ablation_neurons:
        # Use nnsight intervention during generation
        with model.generate(max_new_tokens=50, temperature=0.2, pad_token_id=model.tokenizer.eos_token_id, do_sample=True) as generator:
            with generator.invoke(inputs["input_ids"]):
                # Apply ablations by setting specific neuron activations to zero
                for layer_idx, neuron_idx in ablation_neurons:
                    # Access the MLP activation output for this layer
                    # In Llama, the activation is after the activation function
                    mlp_act = model.model.layers[layer_idx].mlp.act_fn.output
                    
                    # Set specific neuron to zero across all positions
                    # mlp_act shape is [batch, seq_len, hidden_dim]
                    mlp_act[:, :, neuron_idx] = 0.0
                
                # Save the output
                output = model.output.save()
        
        # Get generated tokens
        generated_ids = output.value
    else:
        # Normal generation without intervention
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.2,
                pad_token_id=model.tokenizer.eos_token_id,
                do_sample=True
            )
        generated_ids = output
    
    # Decode full response
    full_text = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Extract just the generated part
    prompt_text = prompt.replace('<|begin_of_text|>', '').replace('<|eot_id|>', '').strip()
    generated_text = full_text[len(prompt_text):].strip() if len(full_text) > len(prompt_text) else full_text
    
    print(f"\nGenerated: {generated_text[:100]}...")
    
    # Analyze the answer
    generated_lower = generated_text.lower()
    
    # Simple check for which number is claimed to be bigger
    says_9_8_bigger = (
        ("9.8" in generated_text and any(w in generated_lower for w in ["bigger", "larger", "greater"])) or
        ("bigger than 9.11" in generated_lower) or
        ("larger than 9.11" in generated_lower) or
        ("greater than 9.11" in generated_lower)
    )
    
    says_9_11_bigger = (
        ("9.11" in generated_text and any(w in generated_lower for w in ["bigger", "larger", "greater"])) or
        ("bigger than 9.8" in generated_lower) or
        ("larger than 9.8" in generated_lower) or
        ("greater than 9.8" in generated_lower)
    )
    
    # Determine answer
    if says_9_8_bigger and not says_9_11_bigger:
        answer = "CORRECT (9.8 is bigger)"
        correct = True
    elif says_9_11_bigger and not says_9_8_bigger:
        answer = "INCORRECT (9.11 is bigger)"
        correct = False
    else:
        answer = "UNCLEAR"
        correct = None
    
    print(f"Answer: {answer}")
    
    return {
        "description": description,
        "generated": generated_text,
        "correct": correct,
        "answer": answer
    }

# Run experiments
results = []

# BASELINE: No ablation
print("\n" + "="*70)
print("BASELINE TEST")
print("="*70)
baseline = run_generation_test(model, test_prompt_template, 
                             ablation_neurons=None,
                             description="Baseline (no ablation)")
results.append(baseline)

# STAGE 1: Ablate core hijacker circuit
print("\n" + "="*70)
print("STAGE 1: CORE HIJACKER ABLATION")
print("="*70)
stage1 = run_generation_test(model, test_prompt_template,
                           ablation_neurons=core_hijackers,
                           description="Core hijacker ablation")
results.append(stage1)

# STAGE 2: Ablate ONLY tie-breaker neurons
print("\n" + "="*70)
print("STAGE 2: TIE-BREAKER NEURON ABLATION")
print("="*70)
stage2 = run_generation_test(model, test_prompt_template,
                           ablation_neurons=tiebreaker_neurons,
                           description="Tie-breaker only ablation")
results.append(stage2)

# CONTROL: Ablate some random neurons from reasoning circuit (shouldn't fix bug)
print("\n" + "="*70)
print("CONTROL: REASONING CIRCUIT ABLATION")
print("="*70)
reasoning_neurons = circuits["chat"]["reasoning_cluster"][:3]  # Just 3 neurons
control = run_generation_test(model, test_prompt_template,
                            ablation_neurons=reasoning_neurons,
                            description="Control (reasoning circuit)")
results.append(control)

# Summary
print("\n" + "="*70)
print("EXPERIMENT SUMMARY")
print("="*70)

for result in results:
    print(f"\n{result['description']}:")
    print(f"  Result: {result['answer']}")
    print(f"  Generated: {result['generated'][:50]}...")

# Analysis
print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

if baseline['correct'] is False:
    print("✓ Baseline confirms the bug (9.11 > 9.8)")
    
    if stage1['correct'] is True:
        print("✓ Stage 1 SUCCESS: Core hijacker ablation fixes the bug!")
        print("  → The shared neurons (L15/N3136, L14/N13315) are key!")
    else:
        print("✗ Stage 1: Core hijacker ablation did not fix the bug")
    
    if stage2['correct'] is True:
        print("🌟 Stage 2 SPECTACULAR: Tie-breaker neurons are master switches!")
        print(f"  → Just {len(tiebreaker_neurons)} neurons control the entire bug!")
        for layer, neuron in sorted(tiebreaker_neurons):
            print(f"    Master switch: L{layer}/N{neuron}")
    else:
        print("✗ Stage 2: Tie-breaker ablation did not fix the bug")
    
    if control['correct'] is False:
        print("✓ Control: Reasoning circuit ablation maintains the bug (as expected)")
    else:
        print("? Control: Unexpected result from reasoning circuit ablation")
else:
    print("⚠️ Baseline did not exhibit the bug - results may be unreliable")

# Save detailed results
with open("ablation_results.json", "w") as f:
    json.dump({
        "core_hijackers": core_hijackers,
        "tiebreaker_neurons": tiebreaker_neurons,
        "results": results
    }, f, indent=2)

print(f"\n✅ Detailed results saved to: ablation_results.json")