#!/usr/bin/env python
"""
Simplified AI Linter - Find and ablate highly activated neurons
"""

import os
import sys

# Add paths
sys.path.extend([
    '/home/paperspace/dev/MATS9',
    '/home/paperspace/dev/MATS9/observatory/lib/neurondb',
    '/home/paperspace/dev/MATS9/observatory/lib/util',
])

from dotenv import load_dotenv
load_dotenv('/home/paperspace/dev/MATS9/.env')

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

from neurondb.filters import QTILE_KEYS, ActivationPercentileFilter
from neurondb.postgres import DBManager
from util.subject import Subject, llama31_8B_instruct_config
from neurondb.view import NeuronView
from util.chat_input import make_chat_conversation
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

def test_linter_ablation(prompt: str, activation_threshold: float = 0.95):
    """
    Simplified version: Find highly activated neurons and ablate them
    """
    print("\n" + "="*60)
    print("🤖 SIMPLIFIED AI LINTER ABLATION")
    print("="*60)
    
    # Initialize
    print("Initializing...")
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        subject = Subject(llama31_8B_instruct_config, nnsight_lm_kwargs={"dispatch": True})
        db_manager = DBManager.get_instance()
        percentiles = NeuronView.load_percentiles(db_manager, subject, QTILE_KEYS)
    
    # First, generate normally and get activations
    print("\n1️⃣ Generating original response...")
    neuron_view = NeuronView(subject, db_manager, make_chat_conversation(), percentiles)
    
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        generator = neuron_view.send_message(
            subject, prompt, max_new_tokens=30, temperature=0, stream=True
        )
        tokens = []
        for token in generator:
            if isinstance(token, int):
                tokens.append(token)
        
        original_response = subject.decode(tokens).strip()
    
    original_answer = original_response.split('\n')[-1].strip()
    print(f"Original: {original_answer}")
    
    # Find highly activated neurons
    print(f"\n2️⃣ Finding neurons with activation > {activation_threshold}...")
    activation_filter = ActivationPercentileFilter(
        min_activation_percentile=activation_threshold
    )
    neuron_view.set_filter(activation_filter)
    highly_activated = neuron_view.get_neurons(with_tokens=True)
    
    print(f"Found {len(highly_activated)} highly activated neurons")
    
    # Get metadata to see what concepts they represent
    metadata = neuron_view.get_neurons_metadata_dict(highly_activated[:20])  # Just look at top 20
    
    print("\n📊 Top activated concepts:")
    concept_counts = {}
    for i, neuron in enumerate(highly_activated[:20]):
        meta = metadata.general.get((neuron.layer, neuron.neuron))
        if meta and neuron.polarity:
            desc = meta.descriptions.get(neuron.polarity)
            if desc and desc.summary:
                summary = desc.summary[:80]
                print(f"  L{neuron.layer}N{neuron.neuron}: {summary}...")
                
                # Simple concept extraction
                for word in ["bible", "verse", "date", "September", "11", "calendar", "version", "chapter"]:
                    if word.lower() in summary.lower():
                        concept_counts[word] = concept_counts.get(word, 0) + 1
    
    if concept_counts:
        print("\n🚨 Suspicious concepts found:")
        for concept, count in sorted(concept_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {concept}: {count} neurons")
    
    # Ablate highly activated neurons
    print(f"\n3️⃣ Ablating {len(highly_activated)} highly activated neurons...")
    
    interventions = {}
    prompt_len = len(subject.tokenizer(prompt)['input_ids'])
    
    for neuron in highly_activated:
        for idx in range(prompt_len):
            interventions[(neuron.layer, idx, neuron.neuron)] = 0.0
    
    # Create new neuron view for ablation
    ablated_view = NeuronView(subject, db_manager, make_chat_conversation(), percentiles)
    ablated_view.set_neuron_interventions(interventions)
    
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        generator = ablated_view.send_message(
            subject, prompt, max_new_tokens=30, temperature=0, stream=True
        )
        tokens = []
        for token in generator:
            if isinstance(token, int):
                tokens.append(token)
        
        ablated_response = subject.decode(tokens).strip()
    
    ablated_answer = ablated_response.split('\n')[-1].strip()
    print(f"\nAblated: {ablated_answer}")
    
    # Check results
    print("\n📋 RESULTS:")
    print("="*60)
    print(f"Original: {original_answer}")
    print(f"Ablated:  {ablated_answer}")
    
    if original_answer != ablated_answer:
        print("\n✅ Ablation changed the output!")
        
        # Check if it's correct now
        if "9.8" in prompt or "9.9" in prompt:
            if ("9.8" in ablated_answer and "bigger" in ablated_answer.lower()) or \
               ("9.9" in ablated_answer and "bigger" in ablated_answer.lower()):
                print("🎉 AND IT FIXED THE ERROR!")
    else:
        print("\n❌ No change from ablation")
    
    return original_answer, ablated_answer

# Test
if __name__ == "__main__":
    prompts = [
        "<|start_header_id|>user<|end_header_id|>\n\nWhich is bigger: 9.8 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "<|start_header_id|>user<|end_header_id|>\n\nWhich is bigger: 9.9 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    ]
    
    for prompt in prompts:
        question = prompt.split('user')[-1].split('assistant')[0].strip()
        print(f"\n🔍 Testing: {question}")
        
        try:
            test_linter_ablation(prompt)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()