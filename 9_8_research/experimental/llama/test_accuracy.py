#!/usr/bin/env python
"""Test accuracy over multiple runs like in the paper"""

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

def test_once(concept_to_ablate: str, prompt: str) -> tuple[bool, bool]:
    """
    Test once and return (unaltered_correct, ablated_correct)
    """
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        subject = Subject(llama31_8B_instruct_config, nnsight_lm_kwargs={"dispatch": True})
        db_manager = DBManager.get_instance()
        percentiles = NeuronView.load_percentiles(db_manager, subject, QTILE_KEYS)
        neuron_view = NeuronView(subject, db_manager, make_chat_conversation(), percentiles)
    
    # Find neurons
    db_filter = NeuronDBFilter(concept_or_embedding=concept_to_ablate, k=500)
    neuron_view.set_filter(db_filter)
    neurons_to_ablate = neuron_view.get_neurons(with_tokens=False)
    
    # Set interventions
    interventions = {}
    prompt_token_len = len(subject.tokenizer(prompt)['input_ids'])
    for neuron in neurons_to_ablate:
        for token_idx in range(prompt_token_len):
            interventions[(neuron.layer, token_idx, neuron.neuron)] = 0.0
    
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        # Unaltered
        unaltered_view = NeuronView(subject, db_manager, make_chat_conversation(), percentiles)
        unaltered_gen = unaltered_view.send_message(subject, prompt, max_new_tokens=45, temperature=0.2, stream=True)
        unaltered_tokens = [t for t in unaltered_gen if isinstance(t, int)]
        unaltered_text = subject.decode(unaltered_tokens)
        
        # Ablated
        neuron_view.set_neuron_interventions(interventions)
        ablated_gen = neuron_view.send_message(subject, prompt, max_new_tokens=45, temperature=0.2, stream=True)
        ablated_tokens = [t for t in ablated_gen if isinstance(t, int)]
        ablated_text = subject.decode(ablated_tokens)
    
    # Check correctness (9.8 should be identified as bigger)
    unaltered_correct = "9.8" in unaltered_text and ("bigger" in unaltered_text or "larger" in unaltered_text)
    ablated_correct = "9.8" in ablated_text and ("bigger" in ablated_text or "larger" in ablated_text)
    
    return unaltered_correct, ablated_correct

def main():
    prompt = "<|start_header_id|>user<|end_header_id|>\n\nWhich is bigger: 9.8 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    concept = "bible verses"
    n_runs = 10
    
    print(f"Testing '{concept}' ablation over {n_runs} runs...")
    print("Temperature: 0.2 (same as paper)")
    print("="*60)
    
    unaltered_correct_count = 0
    ablated_correct_count = 0
    
    for i in range(n_runs):
        print(f"\nRun {i+1}/{n_runs}...", end='', flush=True)
        try:
            unaltered_correct, ablated_correct = test_once(concept, prompt)
            
            unaltered_correct_count += unaltered_correct
            ablated_correct_count += ablated_correct
            
            print(f" Unaltered: {'✓' if unaltered_correct else '✗'}, Ablated: {'✓' if ablated_correct else '✗'}")
            
        except Exception as e:
            print(f" Error: {e}")
    
    print("\n" + "="*60)
    print("RESULTS:")
    print(f"Unaltered accuracy: {unaltered_correct_count}/{n_runs} = {unaltered_correct_count/n_runs*100:.1f}%")
    print(f"Ablated accuracy:   {ablated_correct_count}/{n_runs} = {ablated_correct_count/n_runs*100:.1f}%")
    print(f"Improvement:        {(ablated_correct_count - unaltered_correct_count)/n_runs*100:+.1f}%")

if __name__ == "__main__":
    main()