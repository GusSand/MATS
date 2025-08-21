import torch
from nnsight import LanguageModel
import os
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress ALL warnings
warnings.filterwarnings('ignore')

# Suppress transformers logging
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
import logging
logging.getLogger('transformers').setLevel(logging.ERROR)

# --- Configuration ---
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct" 
# From the Transluce investigator, we identified L2/N1054 as a key "9/11" neuron.
ABLATION_LAYER = 2
ABLATION_NEURON = 1054

# --- Prompt ---
# The prompt where the model makes a mistake. We will intervene on this pathway.
prompt = "<|start_header_id|>user<|end_header_id|>\\n\\nWhich is bigger: 9.9 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"


print("⚙️ Initializing model on your GPU...")
# NNSight will automatically use your powerful GPU.

# Temporarily suppress stdout during model loading
import sys
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
    model = LanguageModel(MODEL_NAME, device_map='auto')
    
print("✅ Model loaded.")

# ==============================================================================
# ACT I: ABLATION & COMPARISON
# We run two scenarios: one unaltered, one with a neuron ablated.
# ==============================================================================
print("\n--- Act I: Performing Ablation and Generating Outputs ---")

# Suppress output during generation
with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
    # --- 1. UNALTERED (FLAWED PATHWAY) ---
    with model.generate(max_new_tokens=45, temperature=0.2) as generator:
        with generator.invoke(prompt):
            unaltered_output = model.generator.output.save()

    # --- 2. INTERVENED (NEURON ABLATED) ---
    with model.generate(max_new_tokens=45, temperature=0.2) as generator:
        with generator.invoke(prompt):

            # VERIFICATION: Save the neuron's value BEFORE the intervention.
            pre_ablation_value = model.model.layers[ABLATION_LAYER].mlp.output[:, :, ABLATION_NEURON].mean().save()

            # SURGERY: Zero out the activation of the target neuron.
            # We apply this to all tokens in the sequence.
            model.model.layers[ABLATION_LAYER].mlp.output[:, :, ABLATION_NEURON] = 0

            # VERIFICATION: Save the neuron's value AFTER the intervention.
            post_ablation_value = model.model.layers[ABLATION_LAYER].mlp.output[:, :, ABLATION_NEURON].mean().save()
            
            intervened_output = model.generator.output.save()


# ==============================================================================
# FINAL RESULTS
# ==============================================================================
print("\n\n" + "="*30)
print("--- 📊 FINAL RESULTS 📊 ---")
print("="*30)

# Extract tokens from the tensor outputs
unaltered_tokens = unaltered_output
intervened_tokens = intervened_output

# Get original prompt lengths to identify only the NEW tokens
prompt_len = len(model.tokenizer(prompt)['input_ids'])

print("\n🔍 ABLATION ANALYSIS:")
print("="*30)

print("\n 1️⃣  Unaltered (Flawed Pathway):")
answer_only = model.tokenizer.decode(unaltered_tokens[0][prompt_len:].tolist(), skip_special_tokens=True)
# Clean up the response - remove common unwanted tokens
clean_answer = answer_only.strip()
for unwanted in ["assistant", "Answer:", "\\n\\n", "  "]:
    clean_answer = clean_answer.replace(unwanted, "")
clean_answer = clean_answer.strip()
print(f"Answer: {clean_answer}")
print("-" * 30)

print("\n 2️⃣  Intervened (Neuron Ablated):")
answer_only = model.tokenizer.decode(intervened_tokens[0][prompt_len:].tolist(), skip_special_tokens=True)
# Clean up the response
clean_answer = answer_only.strip()
for unwanted in ["assistant", "Answer:", "\\n\\n", "  "]:
    clean_answer = clean_answer.replace(unwanted, "")
clean_answer = clean_answer.strip()
print(f"Answer: {clean_answer}")

print("\n🔬 VERIFICATION:")
print(f"• Pre-ablation value of L{ABLATION_LAYER}/N{ABLATION_NEURON}:  {pre_ablation_value.item():.4f}")
print(f"• Post-ablation value of L{ABLATION_LAYER}/N{ABLATION_NEURON}: {post_ablation_value.item():.4f}")

print("\n🧠 INTERPRETATION:")
print(f"• We are testing the hypothesis that neuron {ABLATION_NEURON} in layer {ABLATION_LAYER} is responsible for the error.")
print("• Unaltered: Shows the model's original, flawed reasoning.")
print(f"• Intervened: Shows the result after zeroing out the activation of neuron L{ABLATION_LAYER}/N{ABLATION_NEURON}.")
print("• Goal: To see if removing this single, spurious concept is enough to correct the model's behavior.")
print("="*30)
