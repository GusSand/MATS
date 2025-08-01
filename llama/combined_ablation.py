#!/usr/bin/env python
"""Test combined ablation of multiple concepts as in the paper"""

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

# The three concepts from the paper
CONCEPTS = ["bible verses", "dates", "phone versions"]
PROMPT = "<|start_header_id|>user<|end_header_id|>\n\nWhich is bigger: 9.8 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

print("🧠 COMBINED CONCEPT ABLATION (as in Transluce paper)")
print("="*60)
print(f"Concepts: {', '.join(CONCEPTS)}")
print("="*60)

# Initialize
print("\n⚙️ Initializing...")
with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
    subject = Subject(llama31_8B_instruct_config, nnsight_lm_kwargs={"dispatch": True})
    db_manager = DBManager.get_instance()
    percentiles = NeuronView.load_percentiles(db_manager, subject, QTILE_KEYS)

# Find neurons for each concept and combine
all_neurons = set()
for concept in CONCEPTS:
    print(f"\n🔍 Finding neurons for '{concept}'...")
    neuron_view = NeuronView(subject, db_manager, make_chat_conversation(), percentiles)
    db_filter = NeuronDBFilter(concept_or_embedding=concept, k=500)
    neuron_view.set_filter(db_filter)
    neurons = neuron_view.get_neurons(with_tokens=False)
    
    # Add to combined set
    for neuron in neurons:
        all_neurons.add((neuron.layer, neuron.neuron))
    
    print(f"   Found {len(neurons)} neurons")

print(f"\n📊 Total unique neurons to ablate: {len(all_neurons)}")

# Test with and without ablation
print("\n🧪 Testing...")

# Prepare interventions
interventions = {}
prompt_token_len = len(subject.tokenizer(PROMPT)['input_ids'])
for layer, neuron_idx in all_neurons:
    for token_idx in range(prompt_token_len):
        interventions[(layer, token_idx, neuron_idx)] = 0.0

# Run 10 times to get accuracy
correct_unaltered = 0
correct_ablated = 0
n_runs = 10

for i in range(n_runs):
    print(f"\nRun {i+1}/{n_runs}:", end=' ')
    
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        # Unaltered
        unaltered_view = NeuronView(subject, db_manager, make_chat_conversation(), percentiles)
        unaltered_gen = unaltered_view.send_message(subject, PROMPT, max_new_tokens=45, temperature=0.2, stream=True)
        unaltered_tokens = [t for t in unaltered_gen if isinstance(t, int)]
        unaltered_text = subject.decode(unaltered_tokens)
        
        # Ablated
        ablated_view = NeuronView(subject, db_manager, make_chat_conversation(), percentiles)
        ablated_view.set_neuron_interventions(interventions)
        ablated_gen = ablated_view.send_message(subject, PROMPT, max_new_tokens=45, temperature=0.2, stream=True)
        ablated_tokens = [t for t in ablated_gen if isinstance(t, int)]
        ablated_text = subject.decode(ablated_tokens)
    
    # Extract answers
    unaltered_answer = unaltered_text.split('\n')[-1].strip()
    ablated_answer = ablated_text.split('\n')[-1].strip()
    
    # Check correctness
    unaltered_wrong = "9.11 is bigger" in unaltered_answer or "9.11 is larger" in unaltered_answer
    ablated_wrong = "9.11 is bigger" in ablated_answer or "9.11 is larger" in ablated_answer
    
    if not unaltered_wrong:
        correct_unaltered += 1
    if not ablated_wrong:
        correct_ablated += 1
        
    print(f"Unaltered: {'✗' if unaltered_wrong else '✓'} | Ablated: {'✗' if ablated_wrong else '✓'}")

# Results
print("\n" + "="*60)
print("📊 RESULTS (Combined Ablation)")
print("="*60)
print(f"Concepts ablated: {', '.join(CONCEPTS)}")
print(f"Total neurons ablated: {len(all_neurons)}")
print(f"\nAccuracy over {n_runs} runs:")
print(f"  Unaltered: {correct_unaltered}/{n_runs} = {correct_unaltered/n_runs*100:.0f}%")
print(f"  Ablated:   {correct_ablated}/{n_runs} = {correct_ablated/n_runs*100:.0f}%")
print(f"  Improvement: {(correct_ablated - correct_unaltered)/n_runs*100:+.0f}%")

if correct_ablated > correct_unaltered:
    print("\n✅ Combined ablation improved accuracy!")
else:
    print("\n❌ Combined ablation did not improve accuracy")