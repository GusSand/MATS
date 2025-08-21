#!/usr/bin/env python
"""Fine-tune the ablation strength to get correct answers"""

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

print("🎯 Fine-tuning ablation strength for correct answers")
print("="*60)

# Initialize
with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
    subject = Subject(llama31_8B_instruct_config, nnsight_lm_kwargs={"dispatch": True})
    db_manager = DBManager.get_instance()
    percentiles = NeuronView.load_percentiles(db_manager, subject, QTILE_KEYS)

# Test a range of smaller negative strengths
strengths = [0.0, -0.001, -0.01, -0.05, -0.1, -0.2, -0.3, -0.4]

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
    
    # Test 10 times to get better statistics
    correct_9_8 = 0  # Says 9.8 is bigger
    incorrect_9_11 = 0  # Says 9.11 is bigger  
    nonsense = 0  # Generates nonsense
    
    for run in range(10):
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
            symbol = "✓"
        elif "9.11 is bigger" in answer or "9.11 is larger" in answer:
            incorrect_9_11 += 1
            symbol = "✗"
        else:
            nonsense += 1
            symbol = "?"
        
        if run < 3:  # Show first 3 examples
            print(f"  Run {run+1}: {answer[:50]}... {symbol}")
    
    print(f"\n  Results over 10 runs:")
    print(f"    Correct (9.8):   {correct_9_8}/10 = {correct_9_8*10}%")
    print(f"    Incorrect (9.11): {incorrect_9_11}/10 = {incorrect_9_11*10}%")
    print(f"    Nonsense:         {nonsense}/10 = {nonsense*10}%")

print("\n" + "="*60)
print("💡 Finding the right strength is crucial for meaningful ablation!")