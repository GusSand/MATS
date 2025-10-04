#!/usr/bin/env python
"""Test with temperature 0 for deterministic results"""

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

print("🧪 Testing with temperature=0 for deterministic results")
print("="*60)

# Initialize
with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
    subject = Subject(llama31_8B_instruct_config, nnsight_lm_kwargs={"dispatch": True})
    db_manager = DBManager.get_instance()
    percentiles = NeuronView.load_percentiles(db_manager, subject, QTILE_KEYS)

# Get neurons for combined concepts
concepts = ["bible verses", "dates", "phone versions"]
all_neurons = []

for concept in concepts:
    neuron_view = NeuronView(subject, db_manager, make_chat_conversation(), percentiles)
    db_filter = NeuronDBFilter(concept_or_embedding=concept, k=500)
    neuron_view.set_filter(db_filter)
    neurons = neuron_view.get_neurons(with_tokens=False)
    all_neurons.extend(neurons)

print(f"Total neurons found: {len(all_neurons)}")

# Test baseline first
print("\n1️⃣ Baseline (no intervention, temp=0):")
for i in range(3):
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        nv = NeuronView(subject, db_manager, make_chat_conversation(), percentiles)
        gen = nv.send_message(subject, PROMPT, max_new_tokens=45, temperature=0, stream=True)
        tokens = [t for t in gen if isinstance(t, int)]
        text = subject.decode(tokens)
    answer = text.split('\n')[-1].strip()
    print(f"   Run {i+1}: {answer[:50]}")

# Test with quantile * -0.1
print("\n2️⃣ Quantile * -0.1 intervention (temp=0):")
neurons_metadata_dict = neuron_view.get_neurons_metadata_dict(all_neurons, include_run_metadata=False)
interventions = {}
prompt_len = len(subject.tokenizer(PROMPT)['input_ids'])

unique_neurons = set()
for neuron in all_neurons:
    unique_neurons.add((neuron.layer, neuron.neuron, neuron.polarity))

for layer, neuron_idx, polarity in unique_neurons:
    metadata = neurons_metadata_dict.general.get((layer, neuron_idx))
    if metadata and polarity:
        quantile_key = "0.9999999" if polarity == NeuronPolarity.POS else "1e-07"
        quantile = metadata.activation_percentiles.get(quantile_key)
        if quantile is not None:
            for token_idx in range(prompt_len + 50):
                interventions[(layer, token_idx, neuron_idx)] = quantile * -0.1

print(f"   Intervening on {len(interventions) // (prompt_len + 50)} neurons")

for i in range(3):
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        ablated_view = NeuronView(subject, db_manager, make_chat_conversation(), percentiles)
        ablated_view.set_neuron_interventions(interventions)
        gen = ablated_view.send_message(subject, PROMPT, max_new_tokens=45, temperature=0, stream=True)
        tokens = [t for t in gen if isinstance(t, int)]
        text = subject.decode(tokens)
    answer = text.split('\n')[-1].strip()
    print(f"   Run {i+1}: {answer[:50]}")

# Test with zero ablation
print("\n3️⃣ Zero ablation (temp=0):")
zero_interventions = {}
for layer, neuron_idx, _ in unique_neurons:
    for token_idx in range(prompt_len + 50):
        zero_interventions[(layer, token_idx, neuron_idx)] = 0.0

for i in range(3):
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        ablated_view = NeuronView(subject, db_manager, make_chat_conversation(), percentiles)
        ablated_view.set_neuron_interventions(zero_interventions)
        gen = ablated_view.send_message(subject, PROMPT, max_new_tokens=45, temperature=0, stream=True)
        tokens = [t for t in gen if isinstance(t, int)]
        text = subject.decode(tokens)
    answer = text.split('\n')[-1].strip()
    print(f"   Run {i+1}: {answer[:50]}")