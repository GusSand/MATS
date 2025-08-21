#!/usr/bin/env python
"""
Fine-grained parameter sweep around the transition point (-4.0 to -4.5)
to find if there's a sweet spot we missed
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

print("🔬 Fine-Grained Ablation Sweep: -4.0 to -4.5")
print("="*70)

# Load the identified circuits
with open("identified_circuits.json", "r") as f:
    circuits = json.load(f)

# Get hijacker neurons
all_hijackers = circuits["chat"]["hijacker_cluster"]
unique_hijackers = list(set(tuple(n) for n in all_hijackers))

print(f"Targeting {len(unique_hijackers)} unique hijacker neurons")

# Load model
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
NEURONS_TO_ABLATE = set(unique_hijackers)
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

def analyze_output(text):
    """Analyze output for bug presence and correctness"""
    text_lower = text.lower()
    
    # Check what the model claims
    says_911_bigger = False
    says_98_bigger = False
    
    if "9.11" in text and any(phrase in text_lower for phrase in ["is bigger", "is larger", "is greater"]):
        pos_911 = text.find("9.11")
        pos_phrase = max([text_lower.find(p) for p in ["is bigger", "is larger", "is greater"] if p in text_lower])
        if pos_911 < pos_phrase and pos_phrase - pos_911 < 30:
            says_911_bigger = True
    
    if "9.8" in text and any(phrase in text_lower for phrase in ["is bigger", "is larger", "is greater"]):
        pos_98 = text.find("9.8")
        pos_phrase = max([text_lower.find(p) for p in ["is bigger", "is larger", "is greater"] if p in text_lower])
        if pos_98 < pos_phrase and pos_phrase - pos_98 < 30:
            says_98_bigger = True
    
    # Determine result
    if says_911_bigger and not says_98_bigger:
        return "BUG"
    elif says_98_bigger and not says_911_bigger:
        return "CORRECT"
    else:
        return "UNCLEAR"

def test_ablation_value(ablation_val, num_tests=20):
    """Test a specific ablation value"""
    global ABLATION_VALUE
    ABLATION_VALUE = ablation_val
    
    # Prepare prompt
    test_prompt = [{"role": "user", "content": "Which is bigger: 9.8 or 9.11?"}]
    prompt = tokenizer.apply_chat_template(test_prompt, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    results = {"BUG": 0, "CORRECT": 0, "UNCLEAR": 0}
    outputs = []
    
    for _ in range(num_tests):
        # Register hooks
        hooks = []
        for idx, layer in enumerate(model.model.layers):
            hook = layer.mlp.act_fn.register_forward_hook(
                lambda module, input, output, idx=idx: ablation_hook(module, input, output, idx)
            )
            hooks.append(hook)
        
        # Generate
        with torch.no_grad():
            output = model.generate(
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
        full_text = tokenizer.decode(output[0], skip_special_tokens=True)
        if "assistant" in full_text:
            generated = full_text.split("assistant")[-1].strip()
        else:
            generated = full_text
        
        # Analyze
        result = analyze_output(generated)
        results[result] += 1
        
        if len(outputs) < 3:
            outputs.append(generated)
    
    bug_rate = (results["BUG"] / num_tests) * 100
    correct_rate = (results["CORRECT"] / num_tests) * 100
    unclear_rate = (results["UNCLEAR"] / num_tests) * 100
    
    return {
        "ablation_value": ablation_val,
        "bug_rate": bug_rate,
        "correct_rate": correct_rate,
        "unclear_rate": unclear_rate,
        "examples": outputs
    }

# Fine-grained sweep
ablation_values = [-4.0, -4.05, -4.1, -4.15, -4.2, -4.25, -4.3, -4.35, -4.4, -4.45, -4.5]
results = []

print("\nRunning fine-grained sweep...")
for val in tqdm(ablation_values, desc="Testing values"):
    result = test_ablation_value(val, num_tests=20)
    results.append(result)
    print(f"\nAblation {val}: Bug={result['bug_rate']:.0f}%, Correct={result['correct_rate']:.0f}%, Unclear={result['unclear_rate']:.0f}%")
    if result['correct_rate'] > 0:
        print(f"  ✅ FOUND CORRECT ANSWERS!")
    print(f"  Example: {result['examples'][0][:80]}...")

# Find the best value
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

print("\n{:<8} {:<10} {:<12} {:<12}".format("Value", "Bug %", "Correct %", "Unclear %"))
print("-" * 45)
for r in results:
    print("{:<8} {:<10.0f} {:<12.0f} {:<12.0f}".format(
        r['ablation_value'], r['bug_rate'], r['correct_rate'], r['unclear_rate']
    ))

# Look for sweet spot
sweet_spots = []
for r in results:
    if r['bug_rate'] < 50 and r['correct_rate'] > 0:
        sweet_spots.append(r)

if sweet_spots:
    best = max(sweet_spots, key=lambda x: x['correct_rate'])
    print(f"\n🎯 SWEET SPOT FOUND!")
    print(f"   Ablation value: {best['ablation_value']}")
    print(f"   Bug rate: {best['bug_rate']:.0f}%")
    print(f"   CORRECT rate: {best['correct_rate']:.0f}%")
    print(f"   Unclear rate: {best['unclear_rate']:.0f}%")
    print(f"\n   Example outputs:")
    for i, ex in enumerate(best['examples']):
        print(f"   {i+1}. {ex}")
else:
    print("\n❌ No true sweet spot found")
    print("The transition is too sharp - the model goes from buggy to incoherent")

# Save results
with open("fine_sweep_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
vals = [r['ablation_value'] for r in results]
bugs = [r['bug_rate'] for r in results]
corrects = [r['correct_rate'] for r in results]
unclears = [r['unclear_rate'] for r in results]

ax.plot(vals, bugs, 'o-', label='Bug Rate %', color='red', linewidth=2, markersize=8)
ax.plot(vals, corrects, 's-', label='Correct Rate %', color='green', linewidth=2, markersize=8)
ax.plot(vals, unclears, '^-', label='Unclear Rate %', color='gray', linewidth=2, markersize=8)

ax.set_xlabel('Ablation Value', fontsize=12)
ax.set_ylabel('Percentage', fontsize=12)
ax.set_title('Fine-Grained Ablation Sweep Results', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-5, 105)

plt.tight_layout()
plt.savefig('fine_sweep_plot.png', dpi=150)
print(f"\n📊 Plot saved to: fine_sweep_plot.png")

print("\n✅ Fine sweep complete!")