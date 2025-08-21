import torch
from nnsight import LanguageModel
import os
import warnings
import sys

# --- Setup ---
OBSERVATORY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'observatory'))
PROJECT_PATH = os.path.join(OBSERVATORY_PATH, 'project')
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

from neurondb.postgres import DBManager
from util.subject import Subject, llama31_8B_instruct_config
from neurondb.view import NeuronView
from util.chat_input import make_chat_conversation
from dotenv import load_dotenv
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

DOTENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'observatory', '.env'))
load_dotenv(DOTENV_PATH)


# --- Suppress Warnings ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
import logging
logging.getLogger('transformers').setLevel(logging.ERROR)

# --- Configuration ---
ADDITION_STRENGTH = 2.0
INTERVENTION_LAYER = 15

# --- Prompts ---
UNPRIMED_PROMPT = "<|start_header_id|>user<|end_header_id|>\\n\\nWhich is bigger: 9.9 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"
PRIMED_PROMPT = "<|start_header_id|>user<|end_header_id|>\\n\\nWhich is bigger: 9.90 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"


print("⚙️ Initializing model, database, and subject...")
with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
    subject = Subject(llama31_8B_instruct_config, nnsight_lm_kwargs={"dispatch": True})

print("✅ Setup complete.")

# ==============================================================================
# ACT I: CALCULATE THE "CORRECT CONCEPT" VECTOR
# ==============================================================================
print("\\n--- Act I: Calculating the 'Correct Concept' Vector ---")

with subject.model.generate(max_new_tokens=1) as generator:
    with generator.invoke(UNPRIMED_PROMPT) as invoker:
        last_token_idx = len(subject.tokenizer(UNPRIMED_PROMPT)['input_ids']) - 1
        unprimed_activations = subject.model.model.layers[INTERVENTION_LAYER].input[:, last_token_idx, :].save()

    with generator.invoke(PRIMED_PROMPT) as invoker:
        last_token_idx = len(subject.tokenizer(PRIMED_PROMPT)['input_ids']) - 1
        primed_activations = subject.model.model.layers[INTERVENTION_LAYER].input[:, last_token_idx, :].save()

correct_concept_vector = primed_activations - unprimed_activations
print(f"✅ 'Correct Concept' vector calculated. Shape: {correct_concept_vector.shape}")


# ==============================================================================
# ACT II: APPLY THE "CORRECT CONCEPT" AND COMPARE
# ==============================================================================
print("\\n--- Act II: Applying 'Correct Concept' and Generating Outputs ---")

with subject.model.generate(max_new_tokens=45, temperature=0.2) as generator:
    with generator.invoke(UNPRIMED_PROMPT):
        unaltered_output = subject.model.generator.output.save()

    with generator.invoke(UNPRIMED_PROMPT):
        last_token_idx = len(subject.tokenizer(UNPRIMED_PROMPT)['input_ids']) - 1
        original_activations = subject.model.model.layers[INTERVENTION_LAYER].input[:, last_token_idx, :].clone()
        subject.model.model.layers[INTERVENTION_LAYER].input[:, last_token_idx, :] = original_activations + (ADDITION_STRENGTH * correct_concept_vector)
        intervened_output = subject.model.generator.output.save()


# ==============================================================================
# FINAL RESULTS
# ==============================================================================
print("\\n\\n" + "="*30)
print("--- 📊 FINAL RESULTS 📊 ---")
print("="*30)

prompt_len = len(subject.tokenizer(UNPRIMED_PROMPT)['input_ids'])

print("\\n🔍 CONCEPT ADDITION ANALYSIS:")
print("="*30)

print("\\n1️⃣  Unaltered (Flawed Pathway):")
answer_only = subject.decode(unaltered_output.tolist()[0][prompt_len:])
print(f"Answer: {answer_only}")
print("-" * 30)

print("\\n2️⃣  Intervened (Concept Added):")
answer_only = subject.decode(intervened_output.tolist()[0][prompt_len:])
print(f"Answer: {answer_only}")

print("\\n🧠 INTERPRETATION:")
print("• We created a 'correct concept' vector by finding the difference between a primed and unprimed prompt.")
print("• Unaltered: Shows the model's original, flawed reasoning.")
print("• Intervened: Shows the result after adding the 'correct concept' vector.")
print("• Goal: To see if adding the correct reasoning process fixes the model's behavior.")
print("="*30)
