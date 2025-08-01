#!/usr/bin/env python
"""Test different prompt formats to see if that's the issue"""

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

print("🧪 Testing different prompt formats")
print("="*60)

# Initialize
with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
    subject = Subject(llama31_8B_instruct_config, nnsight_lm_kwargs={"dispatch": True})
    db_manager = DBManager.get_instance()
    percentiles = NeuronView.load_percentiles(db_manager, subject, QTILE_KEYS)

# Different prompt formats to test
prompts = [
    ("Chat format", "<|start_header_id|>user<|end_header_id|>\n\nWhich is bigger: 9.8 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
    ("Simple question", "Which is bigger: 9.8 or 9.11?\n\nAnswer:"),
    ("Q/A format", "Q: Which is bigger: 9.8 or 9.11?\nA:"),
    ("Direct", "9.8 vs 9.11. The bigger number is"),
    ("Complete sentence", "Between 9.8 and 9.11, the larger number is"),
]

# Get combined neurons for ablation
concepts = ["bible verses", "dates", "phone versions"]
all_neurons = set()

for concept in concepts:
    neuron_view = NeuronView(subject, db_manager, make_chat_conversation(), percentiles)
    db_filter = NeuronDBFilter(concept_or_embedding=concept, k=500)
    neuron_view.set_filter(db_filter)
    neurons = neuron_view.get_neurons(with_tokens=False)
    for neuron in neurons:
        all_neurons.add((neuron.layer, neuron.neuron))

print(f"Ablating {len(all_neurons)} neurons from: {', '.join(concepts)}\n")

# Test each prompt format
for prompt_name, prompt in prompts:
    print(f"\n📝 {prompt_name}")
    print("-" * 40)
    print(f"Prompt: {repr(prompt[:50])}...")
    
    # Test without ablation
    unaltered_correct = 0
    ablated_correct = 0
    
    for run in range(3):
        # Prepare interventions
        interventions = {}
        prompt_len = len(subject.tokenizer(prompt)['input_ids'])
        for layer, neuron_idx in all_neurons:
            for token_idx in range(prompt_len + 50):  # Include generation tokens
                interventions[(layer, token_idx, neuron_idx)] = 0.0
        
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            # Unaltered
            unaltered_view = NeuronView(subject, db_manager, make_chat_conversation(), percentiles)
            unaltered_gen = unaltered_view.send_message(subject, prompt, max_new_tokens=20, temperature=0.2, stream=True)
            unaltered_tokens = [t for t in unaltered_gen if isinstance(t, int)]
            unaltered_text = subject.decode(unaltered_tokens).strip()
            
            # Ablated
            ablated_view = NeuronView(subject, db_manager, make_chat_conversation(), percentiles)
            ablated_view.set_neuron_interventions(interventions)
            ablated_gen = ablated_view.send_message(subject, prompt, max_new_tokens=20, temperature=0.2, stream=True)
            ablated_tokens = [t for t in ablated_gen if isinstance(t, int)]
            ablated_text = subject.decode(ablated_tokens).strip()
        
        # Extract the key part of the answer
        unaltered_answer = unaltered_text.replace(prompt, "").strip()[:50]
        ablated_answer = ablated_text.replace(prompt, "").strip()[:50]
        
        # Check correctness - 9.8 IS bigger
        unaltered_correct += 1 if ("9.8" in unaltered_answer and ("9.8" in unaltered_answer.split()[0:3])) else 0
        ablated_correct += 1 if ("9.8" in ablated_answer and ("9.8" in ablated_answer.split()[0:3])) else 0
        
        print(f"  Run {run+1}:")
        print(f"    Unaltered: {unaltered_answer}")
        print(f"    Ablated:   {ablated_answer}")
    
    print(f"  Accuracy: Unaltered {unaltered_correct}/3, Ablated {ablated_correct}/3")

print("\n" + "="*60)
print("💡 Perhaps the prompt format matters for this task!")