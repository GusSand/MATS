import torch
from nnsight import LanguageModel
import os
import warnings

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
import logging
logging.getLogger('transformers').setLevel(logging.ERROR)

# --- Configuration ---
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# We will intervene at the source of the error, Layer 2.
INTERVENTION_LAYER = 2
# We are intervening at a very early, sensitive layer. Reset strength to 1.0.
SUBTRACTION_STRENGTH = 1.0

# --- Prompts ---
# 1. The prompt where the model makes a mistake.
base_prompt = "<|start_header_id|>user<|end_header_id|>\\n\\nWhich is bigger: 9.9 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"
# 2. A neutral prompt to establish a baseline activation state.
neutral_prompt = "<|start_header_id|>user<|end_header_id|>\\n\\nToday is a beautiful day.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"
# 3. A prompt designed to isolate the "9/11" concept.
contamination_prompt = "<|start_header_id|>user<|end_header_id|>\\n\\nToday is a beautiful day, it reminds me of September 11th.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"

print("⚙️ Initializing model on your GPU...")
# Temporarily suppress stdout during model loading
import sys
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
    model = LanguageModel(MODEL_NAME, device_map='auto')
print("✅ Model loaded.")

# ==============================================================================
# ACT I: CALCULATE THE "BAD CONCEPT" VECTOR
# We subtract the activations of a neutral prompt from a prompt that has
# the contaminating concept to isolate the vector for that concept.
# ==============================================================================
print("\n--- Act I: Calculating the 'Bad Concept' Vector ---")

with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
    with model.generate(max_new_tokens=1) as generator:
        # Get neutral activations
        with generator.invoke(neutral_prompt) as invoker:
            last_token_idx = len(model.tokenizer(neutral_prompt)['input_ids']) - 1
            neutral_activations = model.model.layers[INTERVENTION_LAYER].input[:, last_token_idx, :].save()

        # Get contamination activations
        with generator.invoke(contamination_prompt) as invoker:
            last_token_idx = len(model.tokenizer(contamination_prompt)['input_ids']) - 1
            contamination_activations = model.model.layers[INTERVENTION_LAYER].input[:, last_token_idx, :].save()

# The "bad vector" is the difference, representing the isolated concept of "9/11".
bad_concept_vector = contamination_activations - neutral_activations
print(f"✅ 'Bad Concept' vector calculated. Shape: {bad_concept_vector.shape}")

# ==============================================================================
# ACT II: APPLY THE ANTIDOTE AND COMPARE
# We run the original flawed prompt with and without subtracting the bad concept.
# ==============================================================================
print("\n--- Act II: Applying Antidote and Generating Outputs ---")

with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
    # --- 1. UNALTERED (FLAWED PATHWAY) ---
    with model.generate(max_new_tokens=45, temperature=0.2) as generator:
        with generator.invoke(base_prompt):
            unaltered_output = model.generator.output.save()

    # --- 2. INTERVENED (CONCEPT SUBTRACTED) ---
    with model.generate(max_new_tokens=45, temperature=0.2) as generator:
        with generator.invoke(base_prompt):
            # SURGERY: Subtract the "bad concept" vector from the residual stream of ALL tokens.
            # The vector will be broadcast across the sequence length dimension.
            original_activations = model.model.layers[INTERVENTION_LAYER].input.clone()
            model.model.layers[INTERVENTION_LAYER].input = original_activations - (SUBTRACTION_STRENGTH * bad_concept_vector)
            
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

prompt_len = len(model.tokenizer(base_prompt)['input_ids'])

print("\n🔍 CONCEPT SUBTRACTION ANALYSIS:")
print("="*30)

print("\n1️⃣  Unaltered (Flawed Pathway):")
answer_only = model.tokenizer.decode(unaltered_tokens[0][prompt_len:].tolist(), skip_special_tokens=True)
clean_answer = answer_only.strip().replace("assistant", "").replace("Answer:", "").replace("\\n\\n", "  ").strip()
print(f"Answer: {clean_answer}")
print("-" * 30)

print("\n2️⃣  Intervened (Concept Subtracted):")
answer_only = model.tokenizer.decode(intervened_tokens[0][prompt_len:].tolist(), skip_special_tokens=True)
clean_answer = answer_only.strip().replace("assistant", "").replace("Answer:", "").replace("\\n\\n", "  ").strip()
print(f"Answer: {clean_answer}")

print("\n🧠 INTERPRETATION:")
print(f"• We created an 'antidote' vector by isolating the activations for the 'September 11th' concept.")
print("• Unaltered: Shows the model's original, flawed reasoning.")
print(f"• Intervened: Shows the result after subtracting the 'antidote' vector from the activations.")
print("• Goal: To see if surgically removing the spurious concept corrects the model's behavior.")
print("="*30)
