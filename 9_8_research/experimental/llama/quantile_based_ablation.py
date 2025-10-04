#!/usr/bin/env python
"""Test ablation using quantile-based values like the observatory server"""

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

print("🧪 Testing quantile-based ablation (like observatory server)")
print("="*60)

# Initialize
with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
    subject = Subject(llama31_8B_instruct_config, nnsight_lm_kwargs={"dispatch": True})
    db_manager = DBManager.get_instance()
    percentiles = NeuronView.load_percentiles(db_manager, subject, QTILE_KEYS)

# Test different strengths
strengths = [0.0, -1.0, -0.5, 1.0, 2.0]

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
    
    print(f"Total interventions: {len(interventions) // (prompt_len + 50)} unique neurons")
    
    # Test 5 times
    correct_count = 0
    for run in range(5):
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            # Test with interventions
            ablated_view = NeuronView(subject, db_manager, make_chat_conversation(), percentiles)
            ablated_view.set_neuron_interventions(interventions)
            ablated_gen = ablated_view.send_message(subject, PROMPT, max_new_tokens=45, temperature=0.2, stream=True)
            ablated_tokens = [t for t in ablated_gen if isinstance(t, int)]
            ablated_text = subject.decode(ablated_tokens)
        
        # Check if correct
        answer = ablated_text.split('\n')[-1].strip()
        is_wrong = "9.11 is bigger" in answer or "9.11 is larger" in answer
        
        if not is_wrong:
            correct_count += 1
        
        print(f"  Run {run+1}: {answer[:50]}... {'✗' if is_wrong else '✓'}")
    
    print(f"  Accuracy: {correct_count}/5 = {correct_count/5*100:.0f}%")

print("\n" + "="*60)
print("💡 Using quantile-based intervention values might be the key!")