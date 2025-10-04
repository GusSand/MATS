#!/usr/bin/env python
"""Inspect what neurons we're actually finding"""

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

from neurondb.filters import NeuronDBFilter, QTILE_KEYS
from neurondb.postgres import DBManager
from util.subject import Subject, llama31_8B_instruct_config
from neurondb.view import NeuronView
from util.chat_input import make_chat_conversation
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

print("🔍 Inspecting neurons found for each concept")
print("="*60)

# Initialize
with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
    subject = Subject(llama31_8B_instruct_config, nnsight_lm_kwargs={"dispatch": True})
    db_manager = DBManager.get_instance()
    percentiles = NeuronView.load_percentiles(db_manager, subject, QTILE_KEYS)

concepts = ["bible verses", "dates", "phone versions", "9.11", "decimal numbers", "numerical comparison"]

for concept in concepts:
    print(f"\n📌 Concept: '{concept}'")
    print("-" * 40)
    
    neuron_view = NeuronView(subject, db_manager, make_chat_conversation(), percentiles)
    db_filter = NeuronDBFilter(concept_or_embedding=concept, k=10)  # Just top 10
    neuron_view.set_filter(db_filter)
    neurons = neuron_view.get_neurons(with_tokens=False)
    
    # Get metadata to see descriptions
    metadata = neuron_view.get_neurons_metadata_dict(neurons)
    
    print(f"Found {len(neurons)} neurons. Top examples:")
    
    for i, neuron in enumerate(neurons[:5]):
        meta = metadata.general.get((neuron.layer, neuron.neuron))
        if meta and neuron.polarity:
            desc = meta.descriptions.get(neuron.polarity)
            if desc:
                # Show both summary and full text
                print(f"\n  {i+1}. Layer {neuron.layer}, Neuron {neuron.neuron}")
                if desc.summary:
                    print(f"     Summary: {desc.summary[:100]}...")
                if desc.text:
                    print(f"     Full: {desc.text[:150]}...")

# Now let's check what neurons activate on our specific prompt
print("\n\n🎯 Checking what neurons activate on our prompt")
print("="*60)

PROMPT = "<|start_header_id|>user<|end_header_id|>\n\nWhich is bigger: 9.8 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

# Generate and see what activates
neuron_view = NeuronView(subject, db_manager, make_chat_conversation(), percentiles)

with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
    generator = neuron_view.send_message(subject, PROMPT, max_new_tokens=30, temperature=0, stream=True)
    tokens = []
    for token in generator:
        if isinstance(token, int):
            tokens.append(token)
    response = subject.decode(tokens)

print(f"Model response: {response.split(chr(10))[-1]}")

# Check highly activated neurons
from neurondb.filters import ActivationPercentileFilter
activation_filter = ActivationPercentileFilter(min_activation_percentile=0.99)
neuron_view.set_filter(activation_filter)
highly_activated = neuron_view.get_neurons(with_tokens=True)

print(f"\n🔥 Top 10 most activated neurons:")
metadata = neuron_view.get_neurons_metadata_dict(highly_activated[:10])

for i, neuron in enumerate(highly_activated[:10]):
    meta = metadata.general.get((neuron.layer, neuron.neuron))
    if meta and neuron.polarity:
        desc = meta.descriptions.get(neuron.polarity)
        if desc and desc.summary:
            print(f"  {i+1}. L{neuron.layer}N{neuron.neuron}: {desc.summary[:80]}...")