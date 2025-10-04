#!/usr/bin/env python
"""
Simplified ablation experiments using direct PyTorch hooks.

Stage 1: Ablate core hijacker neurons (common to both formats)
Stage 2: Ablate ONLY tie-breaker neurons (unique to buggy format)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import os
import json
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

print("🧪 Ablation Experiments: Hijacker Circuit vs Tie-Breaker Neurons")
print("="*70)

# Load the identified circuits from our previous analysis
with open("identified_circuits.json", "r") as f:
    circuits = json.load(f)

# Extract neuron lists
chat_hijackers = circuits["chat"]["hijacker_cluster"]
simple_hijackers = circuits["simple"]["hijacker_cluster"]

# Find core hijacker neurons (common to both formats)
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

# Global variable to track which neurons to ablate
NEURONS_TO_ABLATE = set()

def ablation_hook(module, input, output, layer_idx):
    """Hook that zeros out specific neurons"""
    if hasattr(output, 'clone'):
        # Clone to avoid in-place modification issues
        new_output = output.clone()
        
        # Check if this layer has neurons to ablate
        for neuron_layer, neuron_idx in NEURONS_TO_ABLATE:
            if neuron_layer == layer_idx:
                # Zero out the specific neuron across all positions
                new_output[:, :, neuron_idx] = 0.0
        
        return new_output
    return output

def run_generation_test(prompt_messages, ablation_neurons=None, description="Baseline"):
    """
    Run generation with optional neuron ablation.
    """
    global NEURONS_TO_ABLATE
    
    print(f"\n{'='*50}")
    print(f"Test: {description}")
    print(f"{'='*50}")
    
    # Set neurons to ablate
    if ablation_neurons:
        # Convert list of lists to set of tuples
        NEURONS_TO_ABLATE = set(tuple(n) if isinstance(n, list) else n for n in ablation_neurons)
        print(f"Ablating {len(ablation_neurons)} neurons:")
        for neuron in sorted(ablation_neurons):
            if isinstance(neuron, (list, tuple)):
                layer, n = neuron
                print(f"  L{layer}/N{n}")
    else:
        NEURONS_TO_ABLATE = set()
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    
    # Register hooks if ablating
    hooks = []
    if ablation_neurons:
        for idx, layer in enumerate(model.model.layers):
            # Hook the MLP activation function output
            hook = layer.mlp.act_fn.register_forward_hook(
                lambda module, input, output, idx=idx: ablation_hook(module, input, output, idx)
            )
            hooks.append(hook)
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
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
    prompt_clean = prompt.replace('<|begin_of_text|>', '').replace('<|end_header_id|>', '').replace('<|eot_id|>', '').replace('\n\n', '\n').strip()
    if prompt_clean in full_text:
        generated_text = full_text.split(prompt_clean)[-1].strip()
    else:
        # Fallback: just take everything after "assistant"
        if "assistant" in full_text:
            generated_text = full_text.split("assistant")[-1].strip()
        else:
            generated_text = full_text
    
    print(f"\nGenerated: {generated_text[:100]}...")
    
    # Analyze the answer
    generated_lower = generated_text.lower()
    
    # Check for which number is claimed to be bigger
    says_9_8_bigger = False
    says_9_11_bigger = False
    
    # Look for explicit statements
    if "9.8" in generated_text and any(phrase in generated_lower for phrase in ["is bigger", "is larger", "is greater"]):
        # Check if 9.8 is the subject
        if generated_lower.index("9.8") < generated_lower.index("is"):
            says_9_8_bigger = True
    
    if "9.11" in generated_text and any(phrase in generated_lower for phrase in ["is bigger", "is larger", "is greater"]):
        # Check if 9.11 is the subject
        if "9.11" in generated_lower and "is" in generated_lower:
            pos_911 = generated_lower.index("9.11")
            pos_is = generated_lower.find("is", pos_911)
            if pos_is > pos_911 and pos_is - pos_911 < 10:  # Close proximity
                says_9_11_bigger = True
    
    # Also check for "X bigger than Y" patterns
    if "9.8" in generated_text and ("bigger than 9.11" in generated_lower or "larger than 9.11" in generated_lower or "greater than 9.11" in generated_lower):
        says_9_8_bigger = True
        says_9_11_bigger = False
    elif "9.11" in generated_text and ("bigger than 9.8" in generated_lower or "larger than 9.8" in generated_lower or "greater than 9.8" in generated_lower):
        says_9_11_bigger = True
        says_9_8_bigger = False
    
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

# Test prompts - chat template that exhibits the bug
test_prompt = [{"role": "user", "content": "Which is bigger: 9.8 or 9.11?"}]

# Run experiments
results = []

# BASELINE: No ablation
print("\n" + "="*70)
print("BASELINE TEST")
print("="*70)
baseline = run_generation_test(test_prompt, 
                             ablation_neurons=None,
                             description="Baseline (no ablation)")
results.append(baseline)

# STAGE 1: Ablate core hijacker circuit
print("\n" + "="*70)
print("STAGE 1: CORE HIJACKER ABLATION")
print("="*70)
stage1 = run_generation_test(test_prompt,
                           ablation_neurons=core_hijackers,
                           description="Core hijacker ablation")
results.append(stage1)

# STAGE 2: Ablate ONLY tie-breaker neurons
print("\n" + "="*70)
print("STAGE 2: TIE-BREAKER NEURON ABLATION")
print("="*70)
stage2 = run_generation_test(test_prompt,
                           ablation_neurons=tiebreaker_neurons,
                           description="Tie-breaker only ablation")
results.append(stage2)

# CONTROL: Ablate some random neurons from reasoning circuit
print("\n" + "="*70)
print("CONTROL: REASONING CIRCUIT ABLATION")
print("="*70)
reasoning_neurons = circuits["chat"]["reasoning_cluster"][:3]
control = run_generation_test(test_prompt,
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
        print("✅ Stage 1 SUCCESS: Core hijacker ablation fixes the bug!")
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