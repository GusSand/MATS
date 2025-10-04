#!/usr/bin/env python
"""Test with combined concepts as mentioned in the paper"""

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
CONCEPTS = ["bible verses", "dates", "phone versions"]  # As mentioned in paper
ABLATION_STRENGTH = -0.1

print(f"🧪 Testing with combined concepts: {', '.join(CONCEPTS)}")
print(f"   Ablation strength: {ABLATION_STRENGTH}")
print("="*60)

# Initialize
with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
    subject = Subject(llama31_8B_instruct_config, nnsight_lm_kwargs={"dispatch": True})
    db_manager = DBManager.get_instance()
    percentiles = NeuronView.load_percentiles(db_manager, subject, QTILE_KEYS)

# Get neurons for all concepts
all_neurons = []
for concept in CONCEPTS:
    neuron_view = NeuronView(subject, db_manager, make_chat_conversation(), percentiles)
    db_filter = NeuronDBFilter(concept_or_embedding=concept, k=500)
    neuron_view.set_filter(db_filter)
    neurons = neuron_view.get_neurons(with_tokens=False)
    all_neurons.extend(neurons)
    print(f"Found {len(neurons)} neurons for '{concept}'")

# Get unique neurons
unique_neurons = set()
for neuron in all_neurons:
    unique_neurons.add((neuron.layer, neuron.neuron, neuron.polarity))
print(f"Total unique neurons: {len(unique_neurons)}")

# Get metadata and build interventions
neurons_metadata_dict = neuron_view.get_neurons_metadata_dict(all_neurons, include_run_metadata=False)
interventions = {}
prompt_token_len = len(subject.tokenizer(PROMPT)['input_ids'])

for layer, neuron_idx, polarity in unique_neurons:
    metadata = neurons_metadata_dict.general.get((layer, neuron_idx))
    if metadata and polarity:
        quantile_key = "0.9999999" if polarity == NeuronPolarity.POS else "1e-07"
        quantile = metadata.activation_percentiles.get(quantile_key)
        if quantile is not None:
            for token_idx in range(prompt_token_len + 50):
                interventions[(layer, token_idx, neuron_idx)] = quantile * ABLATION_STRENGTH

print(f"Built interventions for {len(interventions) // (prompt_token_len + 50)} neurons")

# Test multiple runs
print("\nTesting accuracy over 20 runs:")
print("-" * 40)

correct = 0
nonsense = 0
for i in range(20):
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        # Test with interventions
        ablated_view = NeuronView(subject, db_manager, make_chat_conversation(), percentiles)
        ablated_view.set_neuron_interventions(interventions)
        ablated_gen = ablated_view.send_message(subject, PROMPT, max_new_tokens=45, temperature=0.2, stream=True)
        ablated_tokens = [t for t in ablated_gen if isinstance(t, int)]
        ablated_text = subject.decode(ablated_tokens)
    
    # Extract answer
    answer = ablated_text.split('\n')[-1].strip()
    
    # Check result
    if "9.8 is bigger" in answer or "9.8 is larger" in answer:
        correct += 1
        symbol = "✓"
    elif "9.11 is bigger" in answer or "9.11 is larger" in answer:
        symbol = "✗"
    else:
        nonsense += 1
        symbol = "?"
    
    if i < 5:  # Show first 5
        print(f"Run {i+1}: {symbol} {answer[:50]}...")

print(f"\n📊 Final Results:")
print(f"   Correct (9.8):   {correct}/20 = {correct*5}%")
print(f"   Incorrect (9.11): {20-correct-nonsense}/20 = {(20-correct-nonsense)*5}%")
print(f"   Nonsense:         {nonsense}/20 = {nonsense*5}%")
print(f"\n   Paper reported: 77% accuracy with combined concepts")
print(f"   We achieved:    {correct*5}% accuracy")