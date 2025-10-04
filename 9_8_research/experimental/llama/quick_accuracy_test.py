#!/usr/bin/env python
"""Quick accuracy test for the updated transluce replication"""

import os
import sys
sys.path.extend([
    '/home/paperspace/dev/MATS9',
    '/home/paperspace/dev/MATS9/observatory/lib/neurondb',
    '/home/paperspace/dev/MATS9/observatory/lib/util',
])

from dotenv import load_dotenv
load_dotenv('/home/paperspace/dev/MATS9/.env')

import warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

from neurondb.filters import NeuronDBFilter, QTILE_KEYS, NeuronPolarity
from neurondb.postgres import DBManager
from util.subject import Subject, llama31_8B_instruct_config
from neurondb.view import NeuronView
from util.chat_input import make_chat_conversation
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

PROMPT = "<|start_header_id|>user<|end_header_id|>\n\nWhich is bigger: 9.8 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
CONCEPT = "bible verses"
ABLATION_STRENGTH = -0.1

print(f"🧪 Quick accuracy test - Concept: '{CONCEPT}', Strength: {ABLATION_STRENGTH}")
print("="*60)

# Initialize
with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
    subject = Subject(llama31_8B_instruct_config, nnsight_lm_kwargs={"dispatch": True})
    db_manager = DBManager.get_instance()
    percentiles = NeuronView.load_percentiles(db_manager, subject, QTILE_KEYS)

# Get neurons
neuron_view = NeuronView(subject, db_manager, make_chat_conversation(), percentiles)
db_filter = NeuronDBFilter(concept_or_embedding=CONCEPT, k=500)
neuron_view.set_filter(db_filter)
neurons_to_ablate = neuron_view.get_neurons(with_tokens=False)
print(f"Found {len(neurons_to_ablate)} neurons for '{CONCEPT}'")

# Get metadata and build interventions
neurons_metadata_dict = neuron_view.get_neurons_metadata_dict(neurons_to_ablate, include_run_metadata=False)
interventions = {}
prompt_token_len = len(subject.tokenizer(PROMPT)['input_ids'])

unique_neurons = set()
for neuron in neurons_to_ablate:
    unique_neurons.add((neuron.layer, neuron.neuron, neuron.polarity))

for layer, neuron_idx, polarity in unique_neurons:
    metadata = neurons_metadata_dict.general.get((layer, neuron_idx))
    if metadata and polarity:
        quantile_key = "0.9999999" if polarity == NeuronPolarity.POS else "1e-07"
        quantile = metadata.activation_percentiles.get(quantile_key)
        if quantile is not None:
            for token_idx in range(prompt_token_len + 50):
                interventions[(layer, token_idx, neuron_idx)] = quantile * ABLATION_STRENGTH

print(f"Built {len(interventions) // (prompt_token_len + 50)} unique neuron interventions")

# Test 10 runs
print("\nTesting accuracy over 10 runs:")
print("-" * 40)

correct = 0
for i in range(10):
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        # Test with interventions
        ablated_view = NeuronView(subject, db_manager, make_chat_conversation(), percentiles)
        ablated_view.set_neuron_interventions(interventions)
        ablated_gen = ablated_view.send_message(subject, PROMPT, max_new_tokens=45, temperature=0.2, stream=True)
        ablated_tokens = [t for t in ablated_gen if isinstance(t, int)]
        ablated_text = subject.decode(ablated_tokens)
    
    # Extract answer
    answer = ablated_text.split('\n')[-1].strip()
    
    # Check if correct
    if "9.8 is bigger" in answer or "9.8 is larger" in answer:
        correct += 1
        print(f"Run {i+1}: ✓ {answer[:50]}")
    elif "9.11 is bigger" in answer or "9.11 is larger" in answer:
        print(f"Run {i+1}: ✗ {answer[:50]}")
    else:
        print(f"Run {i+1}: ? {answer[:50]}")

print(f"\n📊 Accuracy: {correct}/10 = {correct*10}%")
print(f"   Paper reported: 76% accuracy")
print(f"   We achieved: {correct*10}% accuracy")