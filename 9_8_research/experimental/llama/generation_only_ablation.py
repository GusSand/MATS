#!/usr/bin/env python
"""Test ablation only during generation, not prompt processing"""

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

print("🧪 Testing generation-only ablation")
print("="*60)

# Initialize
with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
    subject = Subject(llama31_8B_instruct_config, nnsight_lm_kwargs={"dispatch": True})
    db_manager = DBManager.get_instance()
    percentiles = NeuronView.load_percentiles(db_manager, subject, QTILE_KEYS)

# Get prompt length
prompt_tokens = subject.tokenizer(PROMPT)['input_ids']
prompt_len = len(prompt_tokens)
print(f"Prompt has {prompt_len} tokens")

# Find neurons for combined concepts
concepts = ["bible verses", "dates", "phone versions"]
all_neurons = set()

for concept in concepts:
    neuron_view = NeuronView(subject, db_manager, make_chat_conversation(), percentiles)
    db_filter = NeuronDBFilter(concept_or_embedding=concept, k=500)
    neuron_view.set_filter(db_filter)
    neurons = neuron_view.get_neurons(with_tokens=False)
    for neuron in neurons:
        all_neurons.add((neuron.layer, neuron.neuron))

print(f"Total neurons to ablate: {len(all_neurons)}")

# Test different ablation strategies
strategies = [
    ("All tokens", lambda idx: True),
    ("Generation only", lambda idx: idx >= prompt_len),
    ("Question only", lambda idx: 5 <= idx <= 18),  # "Which is bigger: 9.8 or 9.11?"
    ("Numbers only", lambda idx: idx in [10, 11, 12, 15, 16, 17]),  # Just the number tokens
]

for strategy_name, should_ablate in strategies:
    print(f"\n📊 Testing strategy: {strategy_name}")
    print("-" * 40)
    
    correct_count = 0
    n_runs = 5
    
    for run in range(n_runs):
        # Prepare interventions based on strategy
        interventions = {}
        for layer, neuron_idx in all_neurons:
            # We need to ablate future tokens too (up to some reasonable max)
            for token_idx in range(prompt_len + 50):  # Allow for 50 generated tokens
                if should_ablate(token_idx):
                    interventions[(layer, token_idx, neuron_idx)] = 0.0
        
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            # Test with ablation
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
    
    print(f"  Accuracy: {correct_count}/{n_runs} = {correct_count/n_runs*100:.0f}%")

print("\n" + "="*60)
print("💡 Key insight: Different token ablation strategies may have different effects!")