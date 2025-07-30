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
# A middle layer is a great place to start for interventions.
INTERVENTION_LAYER = 30
INTERVENTION_STRENGTH = 30.0 # How strongly to apply the correction vector.

# --- Prompts ---
# We will use two very similar prompts to create a "correction vector".
# The only difference is '9.9' vs '9.90'.
# This helps us isolate the model's knowledge about that specific equivalence.

# The prompt where the model makes a mistake. This is also what we'll intervene on.
unprimed_prompt = "<|start_header_id|>user<|end_header_id|>\\n\\nWhich is bigger: 9.9 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"

# The prompt that provides the 'correct' reasoning pathway.
primed_prompt = "<|start_header_id|>user<|end_header_id|>\\n\\nWhich is bigger: 9.90 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"


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
# ACT I: RECORD THE ACTIVATIONS
# ==============================================================================
print("\n--- Act I: Recording Activations ---")

# Suppress output during activation recording
with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
    with model.generate(max_new_tokens=1) as generator:
        # Get unprimed activations
        with generator.invoke(unprimed_prompt) as invoker:
            # Tokenize the prompt to get sequence length
            unprimed_tokens = model.tokenizer(unprimed_prompt, return_tensors="pt")
            last_token_idx = unprimed_tokens['input_ids'].shape[1] - 1
            #Get the activations for the last token. 
            unprimed_activations = model.model.layers[INTERVENTION_LAYER].mlp.output[:, last_token_idx, :].save()

        #Get primed activations
        with generator.invoke(primed_prompt) as invoker:
            # Tokenize the prompt to get sequence length
            primed_tokens = model.tokenizer(primed_prompt, return_tensors="pt")
            last_token_idx = primed_tokens['input_ids'].shape[1] - 1
            #Get the activations for the last token. 
            primed_activations = model.model.layers[INTERVENTION_LAYER].mlp.output[:, last_token_idx, :].save()

print("✅ Activations for both pathways recorded.")

# ==============================================================================
# ACT II: CALCULATE THE CONTAMINATION VECTOR
# ==============================================================================
print("\n--- Act II: Calculating the Contamination Vector ---")

contamination_vector = primed_activations - unprimed_activations
print(f"✅ Contamination vector calculated. Shape: {contamination_vector.shape}")

# ==============================================================================
# ACT III: THE INTERVENTION & COMPARISON
# We run all three scenarios to get a clean, direct comparison.
# ==============================================================================
print("\n--- Act III: Performing Intervention and Generating Outputs ---")

# Suppress output during generation
with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
    # --- 1. UNPRIMED (FLAWED PATHWAY) ---
    with model.generate(max_new_tokens=45, temperature=0.2) as generator:
        with generator.invoke(unprimed_prompt):
            unprimed_output = model.generator.output.save()

    # --- 2. PRIMED (CORRECT PATHWAY) ---  
    with model.generate(max_new_tokens=100, temperature=0.2) as generator:
        with generator.invoke(primed_prompt):
            primed_output = model.generator.output.save()

    # --- 3. INTERVENED (MIND-SWAPPED) ---
    with model.generate(max_new_tokens=45, temperature=0.2) as generator:
        # We are now intervening on the UNPRIMED prompt to see if we can fix it
        with generator.invoke(unprimed_prompt):
            # Tokenize the prompt to get sequence length for intervention
            unprimed_tokens = model.tokenizer(unprimed_prompt, return_tensors="pt")
            last_token_idx = unprimed_tokens['input_ids'].shape[1] - 1
            
            # SURGERY: ADD the clean "correction vector" with a bit more strength
            # We .clone() the activations to prevent a self-referential loop in the computation graph.
            original_activations = model.model.layers[INTERVENTION_LAYER].mlp.output[:, last_token_idx, :].clone()
            model.model.layers[INTERVENTION_LAYER].mlp.output[:, last_token_idx, :] = original_activations + (INTERVENTION_STRENGTH * contamination_vector)
            
            intervened_output = model.generator.output.save()


# ==============================================================================
# FINAL RESULTS
# ==============================================================================
print("\\n\\n" + "="*30)
print("--- 📊 FINAL RESULTS 📊 ---")
print("="*30)

# Extract tokens from the tensor outputs
unprimed_tokens = unprimed_output
primed_tokens = primed_output
intervened_tokens = intervened_output

# Get original prompt lengths to identify only the NEW tokens
unprimed_prompt_len = len(model.tokenizer(unprimed_prompt)['input_ids'])
primed_prompt_len = len(model.tokenizer(primed_prompt)['input_ids'])

print("\n🔍 MIND-SWAP ANALYSIS:")
print("="*30)

print("\n1️⃣  Flawed Pathway (Unprimed):")
answer_only = model.tokenizer.decode(unprimed_tokens[0][unprimed_prompt_len:].tolist(), skip_special_tokens=True)
# Clean up the response - remove common unwanted tokens
clean_answer = answer_only.strip()
for unwanted in ["assistant", "Answer:", "\\n\\n", "  "]:
    clean_answer = clean_answer.replace(unwanted, "")
clean_answer = clean_answer.strip()
print(f"Answer: {clean_answer}")


print("\n2️⃣  Correct Pathway (Primed with '9.90'):")
answer_only = model.tokenizer.decode(primed_tokens[0][primed_prompt_len:].tolist(), skip_special_tokens=True)
# Clean up the response  
clean_answer = answer_only.strip()
for unwanted in ["assistant", "Answer:", "\\n\\n", "  "]:
    clean_answer = clean_answer.replace(unwanted, "")
clean_answer = clean_answer.strip()
print(f"Answer: {clean_answer}")
print("-" * 30)

print("\n3️⃣  Intervened Pathway (Surgical Fix):")
answer_only = model.tokenizer.decode(intervened_tokens[0][unprimed_prompt_len:].tolist(), skip_special_tokens=True)
# Clean up the response
clean_answer = answer_only.strip()
for unwanted in ["assistant", "Answer:", "\\n\\n", "  "]:
    clean_answer = clean_answer.replace(unwanted, "")
clean_answer = clean_answer.strip()
print(f"Answer: {clean_answer}")

print("\n🧠 INTERPRETATION:")
print("• We created a 'correction vector' by finding the difference in activations between a prompt with '9.9' and '9.90'.")
print("• Unprimed (Flawed): Shows the model's mistake when comparing 9.9 and 9.11 directly.")
print("• Primed (Correct): Shows how using '9.90' leads to the correct answer.")
print(f"• Intervened: We add the correction vector (strength: {INTERVENTION_STRENGTH}x) to the flawed pathway's activations.")
print("• Goal: Surgically implant the reasoning that '9.9' is like '9.90' to fix the model's comparison.")
print("="*30)
