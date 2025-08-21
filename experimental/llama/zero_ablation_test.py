#!/usr/bin/env python
"""Test setting activations to zero as mentioned in the blog"""

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

PROMPT = "<|start_header_id|>user<|end_header_id|>\n\nWhich is bigger: 9.8 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

print("🧪 Testing zero ablation as mentioned in blog")
print("="*60)

# Initialize
with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
    subject = Subject(llama31_8B_instruct_config, nnsight_lm_kwargs={"dispatch": True})
    db_manager = DBManager.get_instance()
    percentiles = NeuronView.load_percentiles(db_manager, subject, QTILE_KEYS)

# Test with different k values (number of neurons)
k_values = [100, 500, 1000]

for k in k_values:
    print(f"\n📊 Testing with k={k} neurons per concept")
    print("-" * 40)
    
    # Get neurons for combined concepts
    concepts = ["bible verses", "dates", "phone versions"]
    all_neurons = []
    
    for concept in concepts:
        neuron_view = NeuronView(subject, db_manager, make_chat_conversation(), percentiles)
        db_filter = NeuronDBFilter(concept_or_embedding=concept, k=k)
        neuron_view.set_filter(db_filter)
        neurons = neuron_view.get_neurons(with_tokens=False)
        all_neurons.extend(neurons)
    
    # Get unique neurons
    unique_neurons = set()
    for neuron in all_neurons:
        unique_neurons.add((neuron.layer, neuron.neuron))
    
    print(f"Total unique neurons: {len(unique_neurons)}")
    
    # Build interventions - set to 0
    interventions = {}
    prompt_len = len(subject.tokenizer(PROMPT)['input_ids'])
    
    # Try only ablating during generation, not prompt processing
    for layer, neuron_idx in unique_neurons:
        # Start from prompt_len to only affect generation
        for token_idx in range(prompt_len, prompt_len + 50):
            interventions[(layer, token_idx, neuron_idx)] = 0.0
    
    # Test 10 runs
    correct = 0
    for run in range(10):
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            # Test with interventions
            ablated_view = NeuronView(subject, db_manager, make_chat_conversation(), percentiles)
            ablated_view.set_neuron_interventions(interventions)
            ablated_gen = ablated_view.send_message(subject, PROMPT, max_new_tokens=45, temperature=0.2, stream=True)
            ablated_tokens = [t for t in ablated_gen if isinstance(t, int)]
            ablated_text = subject.decode(ablated_tokens)
        
        # Check if correct
        answer = ablated_text.split('\n')[-1].strip()
        if "9.8 is bigger" in answer or "9.8 is larger" in answer:
            correct += 1
            symbol = "✓"
        elif "9.11 is bigger" in answer or "9.11 is larger" in answer:
            symbol = "✗"
        else:
            symbol = "?"
        
        if run < 3:
            print(f"  Run {run+1}: {symbol} {answer[:40]}...")
    
    print(f"\n  Accuracy: {correct}/10 = {correct*10}%")

print("\n" + "="*60)
print("💡 Testing different approaches to match the paper's method")