#!/usr/bin/env python
"""Optimize ablation strength around -0.1"""

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

print("🎯 Optimizing ablation strength around -0.1")
print("="*60)

# Initialize
with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
    subject = Subject(llama31_8B_instruct_config, nnsight_lm_kwargs={"dispatch": True})
    db_manager = DBManager.get_instance()
    percentiles = NeuronView.load_percentiles(db_manager, subject, QTILE_KEYS)

# Test values around -0.1
strengths = [-0.06, -0.07, -0.08, -0.09, -0.10, -0.11, -0.12, -0.13, -0.14, -0.15]

best_strength = None
best_correct_rate = 0

for strength in strengths:
    print(f"\n📊 Testing with strength = {strength}")
    print("-" * 40)
    
    # Get neurons for combined concepts
    concepts = ["bible verses", "dates", "phone versions"]
    all_neurons = []
    
    for concept in concepts:
        neuron_view = NeuronView(subject, db_manager, make_chat_conversation(), percentiles)
        db_filter = NeuronDBFilter(concept_or_embedding=concept, k=500)
        neuron_view.set_filter(db_filter)
        neurons = neuron_view.get_neurons(with_tokens=False)
        all_neurons.extend(neurons)
    
    # Get metadata to access quantiles
    neurons_metadata_dict = neuron_view.get_neurons_metadata_dict(all_neurons, include_run_metadata=False)
    
    # Build interventions using quantile values
    interventions = {}
    prompt_len = len(subject.tokenizer(PROMPT)['input_ids'])
    
    unique_neurons = set()
    for neuron in all_neurons:
        unique_neurons.add((neuron.layer, neuron.neuron, neuron.polarity))
    
    for layer, neuron_idx, polarity in unique_neurons:
        metadata = neurons_metadata_dict.general.get((layer, neuron_idx))
        if metadata and polarity:
            # Get the appropriate quantile like the server does
            quantile_key = "0.9999999" if polarity == NeuronPolarity.POS else "1e-07"
            quantile = metadata.activation_percentiles.get(quantile_key)
            
            if quantile is not None:
                # Apply to all token positions (including future generation)
                for token_idx in range(prompt_len + 50):
                    interventions[(layer, token_idx, neuron_idx)] = quantile * strength
    
    # Test 20 times to get better statistics
    correct_9_8 = 0  # Says 9.8 is bigger
    incorrect_9_11 = 0  # Says 9.11 is bigger  
    nonsense = 0  # Generates nonsense
    
    for run in range(20):
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            # Test with interventions
            ablated_view = NeuronView(subject, db_manager, make_chat_conversation(), percentiles)
            ablated_view.set_neuron_interventions(interventions)
            ablated_gen = ablated_view.send_message(subject, PROMPT, max_new_tokens=45, temperature=0.2, stream=True)
            ablated_tokens = [t for t in ablated_gen if isinstance(t, int)]
            ablated_text = subject.decode(ablated_tokens)
        
        # Categorize the answer
        answer = ablated_text.split('\n')[-1].strip()
        
        if "9.8 is bigger" in answer or "9.8 is larger" in answer:
            correct_9_8 += 1
        elif "9.11 is bigger" in answer or "9.11 is larger" in answer:
            incorrect_9_11 += 1
        else:
            nonsense += 1
    
    correct_rate = correct_9_8 / 20 * 100
    
    print(f"  Results over 20 runs:")
    print(f"    Correct (9.8):   {correct_9_8}/20 = {correct_rate:.0f}%")
    print(f"    Incorrect (9.11): {incorrect_9_11}/20 = {incorrect_9_11/20*100:.0f}%")
    print(f"    Nonsense:         {nonsense}/20 = {nonsense/20*100:.0f}%")
    
    if correct_rate > best_correct_rate:
        best_correct_rate = correct_rate
        best_strength = strength

print("\n" + "="*60)
print(f"💡 Best strength: {best_strength} with {best_correct_rate:.0f}% correct answers!")
print(f"   Compare to paper's 76% accuracy - we're getting closer!")