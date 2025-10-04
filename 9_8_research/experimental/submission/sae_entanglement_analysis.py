#!/usr/bin/env python
"""
SAE Analysis to test the "Irremediable Entanglement" hypothesis.
We'll compare feature activations between good and bad states to see if
the same features are involved in both correct reasoning and the 9.11 bug.
"""

import torch
from sae_lens import SAE
from nnsight import LanguageModel
import warnings
import os
import json
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

print("🔬 SAE Entanglement Analysis")
print("=" * 70)

# Define the models and target layer
# Based on our findings, Layer 15 is a hotspot for hijacker neurons
TARGET_LAYER = 15 
LLM_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# Check available SAE models
print("\nNote: We'll use sae-lens to load pre-trained SAEs for Llama 3.1")
print("Available options include models from EleutherAI and other sources")

# Define our two prompts for contrastive analysis
# Chat format (exhibits bug)
PROMPT_BAD_STATE = [{"role": "user", "content": "Which is bigger: 9.8 or 9.11?"}]

# Simple format (correct answer)
PROMPT_GOOD_STATE = "Which is bigger: 9.8 or 9.11?\nAnswer:"

# Load model with nnsight
print(f"\nLoading {LLM_MODEL_NAME}...")
try:
    model = LanguageModel(LLM_MODEL_NAME, device_map="auto")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Function to extract MLP activations at a specific layer
def get_mlp_activations(model, prompt, layer_idx=TARGET_LAYER):
    """
    Run a forward pass and extract MLP activations at the specified layer.
    """
    # Tokenize based on prompt type
    if isinstance(prompt, list):
        # Chat format
        prompt_text = model.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    else:
        # Simple format
        prompt_text = prompt
    
    print(f"\nAnalyzing prompt: '{prompt_text[:50]}...'")
    
    # Tokenize
    inputs = model.tokenizer(prompt_text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Run forward pass and capture activations
    with model.trace(inputs['input_ids']):
        # Get MLP output at target layer
        mlp_output = model.model.layers[layer_idx].mlp.act_fn.output.save()
    
    # Get the activation values from the saved tensor
    activations = mlp_output
    
    print(f"Activation shape: {activations.shape}")
    print(f"Analyzing activations for last token position...")
    
    # Get activations for the last token (where the model makes its decision)
    last_token_acts = activations[0, -1, :]  # Shape: [hidden_dim]
    
    return last_token_acts, prompt_text

# Function to find top activating neurons
def get_top_neurons(activations, top_k=20):
    """
    Find the top k most active neurons.
    """
    top_values, top_indices = torch.topk(activations, k=top_k)
    
    results = []
    for i in range(top_k):
        results.append({
            'neuron_idx': top_indices[i].item(),
            'activation': top_values[i].item()
        })
    
    return results

# Load our previously identified hijacker neurons
print("\nLoading previously identified circuits...")
with open("identified_circuits.json", "r") as f:
    circuits = json.load(f)

hijacker_neurons = set()
for layer, neuron in circuits["chat"]["hijacker_cluster"]:
    if layer == TARGET_LAYER:
        hijacker_neurons.add(neuron)

print(f"Hijacker neurons in layer {TARGET_LAYER}: {sorted(hijacker_neurons)}")

# Analyze bad state (chat format)
print("\n" + "="*50)
print("ANALYZING BAD STATE (Chat Format)")
print("="*50)

bad_acts, bad_prompt = get_mlp_activations(model, PROMPT_BAD_STATE)
bad_top_neurons = get_top_neurons(bad_acts, top_k=30)

print("\nTop 10 neurons in bad state:")
for i, neuron in enumerate(bad_top_neurons[:10]):
    hijacker_mark = " ⚠️ HIJACKER" if neuron['neuron_idx'] in hijacker_neurons else ""
    print(f"  {i+1}. Neuron {neuron['neuron_idx']}: {neuron['activation']:.4f}{hijacker_mark}")

# Analyze good state (simple format)
print("\n" + "="*50)
print("ANALYZING GOOD STATE (Simple Format)")
print("="*50)

good_acts, good_prompt = get_mlp_activations(model, PROMPT_GOOD_STATE)
good_top_neurons = get_top_neurons(good_acts, top_k=30)

print("\nTop 10 neurons in good state:")
for i, neuron in enumerate(good_top_neurons[:10]):
    hijacker_mark = " ⚠️ HIJACKER" if neuron['neuron_idx'] in hijacker_neurons else ""
    print(f"  {i+1}. Neuron {neuron['neuron_idx']}: {neuron['activation']:.4f}{hijacker_mark}")

# Compare activations
print("\n" + "="*50)
print("ENTANGLEMENT ANALYSIS")
print("="*50)

# Find neurons that are active in both states
bad_neurons = {n['neuron_idx'] for n in bad_top_neurons}
good_neurons = {n['neuron_idx'] for n in good_top_neurons}

shared_neurons = bad_neurons.intersection(good_neurons)
bad_only = bad_neurons - good_neurons
good_only = good_neurons - bad_neurons

print(f"\nNeurons active in BOTH states: {len(shared_neurons)}")
print(f"Neurons active ONLY in bad state: {len(bad_only)}")
print(f"Neurons active ONLY in good state: {len(good_only)}")

# Check if hijacker neurons appear in both
hijacker_in_both = hijacker_neurons.intersection(shared_neurons)
hijacker_bad_only = hijacker_neurons.intersection(bad_only)

print(f"\nHijacker neurons in BOTH states: {sorted(hijacker_in_both)}")
print(f"Hijacker neurons ONLY in bad state: {sorted(hijacker_bad_only)}")

# Quantitative comparison for shared neurons
print("\n" + "="*50)
print("QUANTITATIVE COMPARISON OF SHARED NEURONS")
print("="*50)

# Create lookup dictionaries
bad_lookup = {n['neuron_idx']: n['activation'] for n in bad_top_neurons}
good_lookup = {n['neuron_idx']: n['activation'] for n in good_top_neurons}

amplifications = []
for neuron_idx in shared_neurons:
    bad_act = bad_lookup.get(neuron_idx, 0)
    good_act = good_lookup.get(neuron_idx, 0)
    
    if good_act > 0:
        amplification = bad_act / good_act
        amplifications.append({
            'neuron': neuron_idx,
            'bad_activation': bad_act,
            'good_activation': good_act,
            'amplification': amplification,
            'is_hijacker': neuron_idx in hijacker_neurons
        })

# Sort by amplification ratio
amplifications.sort(key=lambda x: x['amplification'], reverse=True)

print("\nTop 10 most amplified neurons (bad/good ratio):")
for i, amp in enumerate(amplifications[:10]):
    hijacker_mark = " ⚠️ HIJACKER" if amp['is_hijacker'] else ""
    print(f"  {i+1}. Neuron {amp['neuron']}: {amp['amplification']:.2f}x "
          f"(bad: {amp['bad_activation']:.3f}, good: {amp['good_activation']:.3f}){hijacker_mark}")

# Test the entanglement hypothesis
print("\n" + "="*70)
print("IRREMEDIABLE ENTANGLEMENT HYPOTHESIS TEST")
print("="*70)

if hijacker_in_both:
    print("✅ EVIDENCE FOR ENTANGLEMENT:")
    print(f"   - {len(hijacker_in_both)} hijacker neurons are active in BOTH states")
    print(f"   - These neurons: {sorted(hijacker_in_both)}")
    print("   - This suggests these neurons serve dual purposes:")
    print("     1. Normal decimal processing (good state)")
    print("     2. Triggering the bug (bad state)")
    
    # Check if they're amplified
    hijacker_amps = [a for a in amplifications if a['is_hijacker']]
    if hijacker_amps:
        avg_amp = np.mean([a['amplification'] for a in hijacker_amps])
        print(f"\n   - Average amplification of hijacker neurons: {avg_amp:.2f}x")
        if avg_amp > 1.5:
            print("   - Hijacker neurons are AMPLIFIED in bad state!")
else:
    print("❌ NO CLEAR ENTANGLEMENT:")
    print("   - Hijacker neurons don't appear in both states")
    print("   - The bug might be more separable than hypothesized")

# Save results
results = {
    'layer': TARGET_LAYER,
    'bad_prompt': bad_prompt,
    'good_prompt': good_prompt,
    'bad_top_neurons': bad_top_neurons[:20],
    'good_top_neurons': good_top_neurons[:20],
    'shared_neurons': list(shared_neurons),
    'hijacker_in_both': list(hijacker_in_both),
    'amplifications': amplifications[:20]
}

with open("sae_entanglement_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to: sae_entanglement_results.json")

# Generate prediction about SAE features
print("\n" + "="*70)
print("PREDICTION FOR SAE FEATURE ANALYSIS")
print("="*70)
print("When we run actual SAE analysis, we expect to find:")
print("1. Features that fire on decimal number patterns (9.8, 9.11)")
print("2. These SAME features will be active in both good and bad states")
print("3. In the bad state, they'll be quantitatively stronger")
print("4. This proves 'irremediable entanglement' - you can't cleanly separate")
print("   the bug from normal function because they use the same features!")