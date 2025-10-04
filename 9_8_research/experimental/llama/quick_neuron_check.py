#!/usr/bin/env python
"""Quick check of neurons found for different concepts"""

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

print("🔍 Quick neuron check for concepts")
print("="*60)

# Initialize
with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
    subject = Subject(llama31_8B_instruct_config, nnsight_lm_kwargs={"dispatch": True})
    db_manager = DBManager.get_instance()
    percentiles = NeuronView.load_percentiles(db_manager, subject, QTILE_KEYS)

# Check each concept
concepts = ["bible verses", "dates", "phone versions", "decimal numbers", "9.11", "9.8"]

for concept in concepts:
    neuron_view = NeuronView(subject, db_manager, make_chat_conversation(), percentiles)
    db_filter = NeuronDBFilter(concept_or_embedding=concept, k=100)
    neuron_view.set_filter(db_filter)
    neurons = neuron_view.get_neurons(with_tokens=False)
    
    print(f"\n📌 '{concept}': Found {len(neurons)} neurons")
    
    # Show first few neurons
    metadata = neuron_view.get_neurons_metadata_dict(neurons[:3])
    for i, neuron in enumerate(neurons[:3]):
        meta = metadata.general.get((neuron.layer, neuron.neuron))
        if meta and neuron.polarity:
            desc = meta.descriptions.get(neuron.polarity)
            if desc and desc.summary:
                print(f"   L{neuron.layer}N{neuron.neuron}: {desc.summary[:60]}...")

print("\n✅ Done")