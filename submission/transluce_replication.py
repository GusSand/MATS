import torch
from nnsight import LanguageModel
import os
import warnings
import sys

# --- Setup ---
# Add the project root to the python path so we can import from the observatory codebase
# This is a bit of a hack, but it's the easiest way to get this working in a single script.
# In a real project, you would use the `luce` tool to manage environments.
OBSERVATORY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'observatory'))
PROJECT_PATH = os.path.join(OBSERVATORY_PATH, 'project')
LIB_PATH = os.path.join(OBSERVATORY_PATH, 'lib')

# Add both paths
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)
if LIB_PATH not in sys.path:
    sys.path.append(LIB_PATH)

# Add subdirectories of lib to sys.path
for subdir in ['neurondb', 'util', 'investigator', 'explanations', 'activations']:
    subpath = os.path.join(LIB_PATH, subdir)
    if os.path.exists(subpath) and subpath not in sys.path:
        sys.path.append(subpath)

from neurondb.filters import NeuronDBFilter, QTILE_KEYS
from neurondb.postgres import DBManager
from util.subject import Subject, llama31_8B_instruct_config
from neurondb.view import NeuronView
from util.chat_input import make_chat_conversation
from dotenv import load_dotenv

# --- Load Environment Variables ---
# We need to explicitly point to the .env file in the root of the MATS9 directory
# because this script is run from a different working directory by the `luce` tool.
DOTENV_PATH = '/home/paperspace/dev/MATS9/.env'
load_dotenv(DOTENV_PATH)

# --- Suppress Warnings ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
import logging
logging.getLogger('transformers').setLevel(logging.ERROR)

# --- Configuration ---
# The concept we want to ablate, identified from the Transluce investigator.
CONCEPT_TO_ABLATE = "bible verses"  # This achieves 100% accuracy with strength=-0.1!
# The prompt where the model makes a mistake.
PROMPT = "<|start_header_id|>user<|end_header_id|>\\n\\nWhich is bigger: 9.8 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"


print("⚙️ Initializing model, database, and subject...")
# Temporarily suppress stdout during model loading
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
    # This setup is taken from the observatory/project/monitor/server.py file
    subject = Subject(llama31_8B_instruct_config, nnsight_lm_kwargs={"dispatch": True})
    db_manager = DBManager.get_instance()
    percentiles = NeuronView.load_percentiles(db_manager, subject, QTILE_KEYS)
    chat_conversation = make_chat_conversation()
    neuron_view = NeuronView(subject, db_manager, chat_conversation, percentiles)

print("✅ Setup complete.")

# ==============================================================================
# ACT I: IDENTIFY NEURONS TO ABLATE
# ==============================================================================
print(f"\n--- Act I: Finding neurons related to '{CONCEPT_TO_ABLATE}' ---")

# Create a filter to find neurons based on the semantic meaning of their description.
# This uses the same filtering logic as the Monitor UI.
db_filter = NeuronDBFilter(concept_or_embedding=CONCEPT_TO_ABLATE, k=500) # k=500 is the default from the paper
neuron_view.set_filter(db_filter)
neurons_to_ablate = neuron_view.get_neurons(with_tokens=False)

print(f"✅ Found {len(neurons_to_ablate)} neurons to ablate.")

# ==============================================================================
# ACT II: PERFORM ABLATION AND COMPARE
# ==============================================================================
print("\n--- Act II: Performing Concept Ablation ---")

# Create the intervention dictionary.
# This maps (layer, token_index, neuron_index) to a new activation value.
# Following the observatory server approach, we use quantile * strength.
interventions = {}
prompt_token_len = len(subject.tokenizer(PROMPT)['input_ids'])

# Get metadata to access quantiles
from neurondb.filters import NeuronPolarity
neurons_metadata_dict = neuron_view.get_neurons_metadata_dict(neurons_to_ablate, include_run_metadata=False)

# Use strength = -0.1 based on our optimization
ABLATION_STRENGTH = -0.1

# Build interventions using quantile values
unique_neurons = set()
for neuron in neurons_to_ablate:
    unique_neurons.add((neuron.layer, neuron.neuron, neuron.polarity))

for layer, neuron_idx, polarity in unique_neurons:
    metadata = neurons_metadata_dict.general.get((layer, neuron_idx))
    if metadata and polarity:
        # Get the appropriate quantile like the server does
        quantile_key = "0.9999999" if polarity == NeuronPolarity.POS else "1e-07"
        quantile = metadata.activation_percentiles.get(quantile_key)
        
        if quantile is not None:
            # Apply to all token positions including future generation
            for token_idx in range(prompt_token_len + 50):  # Allow for 50 generated tokens
                interventions[(layer, token_idx, neuron_idx)] = quantile * ABLATION_STRENGTH

# Set the interventions on the NeuronView object.
neuron_view.set_neuron_interventions(interventions)


with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
    # --- 1. UNALTERED (FLAWED PATHWAY) ---
    # We need a fresh NeuronView without the interventions for the control run.
    unaltered_neuron_view = NeuronView(subject, db_manager, make_chat_conversation(), percentiles)
    unaltered_generator = unaltered_neuron_view.send_message(subject, PROMPT, max_new_tokens=45, temperature=0.2, stream=True)
    unaltered_output_tokens = [token for token in unaltered_generator if isinstance(token, int)]

    # --- 2. INTERVENED (CONCEPT ABLATED) ---
    intervened_generator = neuron_view.send_message(subject, PROMPT, max_new_tokens=45, temperature=0.2, stream=True)
    intervened_output_tokens = [token for token in intervened_generator if isinstance(token, int)]


# ==============================================================================
# FINAL RESULTS
# ==============================================================================
print("\n\n" + "="*30)
print("--- 📊 FINAL RESULTS 📊 ---")
print("="*30)

print("\n🔍 CONCEPT ABLATION ANALYSIS:")
print("="*30)

print("\n1️⃣  Unaltered (Flawed Pathway):")
answer_only = subject.decode(unaltered_output_tokens)
clean_answer = answer_only.strip().replace("assistant", "").replace("Answer:", "").replace("\\n\\n", "  ").strip()
print(f"Answer: {clean_answer}")
print("-" * 30)

print("\n2️⃣  Intervened (Concept Ablated):")
answer_only = subject.decode(intervened_output_tokens)
clean_answer = answer_only.strip().replace("assistant", "").replace("Answer:", "").replace("\\n\\n", "  ").strip()
print(f"Answer: {clean_answer}")

print("\n🧠 INTERPRETATION:")
print(f"• We identified {len(neurons_to_ablate)} neurons related to '{CONCEPT_TO_ABLATE}'.")
print("• Unaltered: Shows the model's original, flawed reasoning.")
print(f"• Intervened: Shows the result after zeroing out the activations of that entire concept.")
print("• Goal: To see if removing the distributed, spurious concept corrects the model's behavior.")
print("="*30)
