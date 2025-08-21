#!/usr/bin/env python
"""
AI Linter-Guided Ablation
Use the AI Linter to automatically identify and ablate problematic neurons
"""

import os
import sys
import asyncio

# Add paths
sys.path.extend([
    '/home/paperspace/dev/MATS9',
    '/home/paperspace/dev/MATS9/observatory/lib/neurondb',
    '/home/paperspace/dev/MATS9/observatory/lib/util',
    '/home/paperspace/dev/MATS9/observatory/lib/investigator',
])

from dotenv import load_dotenv
load_dotenv('/home/paperspace/dev/MATS9/.env')

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

from neurondb.filters import QTILE_KEYS, ActivationPercentileFilter, Neuron
from neurondb.postgres import DBManager
from util.subject import Subject, llama31_8B_instruct_config
from neurondb.view import NeuronView
from util.chat_input import make_chat_conversation
from investigator.clustering import cluster_neurons, Cluster
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from typing import List, Dict, Tuple

async def find_suspicious_neurons(prompt: str, activation_threshold: float = 0.9) -> Tuple[List[Cluster], str, NeuronView]:
    """
    Use AI Linter to find suspicious neurons for a given prompt
    Returns: (clusters, original_response, neuron_view)
    """
    print("🔍 Phase 1: AI Linter Analysis")
    print("-"*40)
    
    # Initialize
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        subject = Subject(llama31_8B_instruct_config, nnsight_lm_kwargs={"dispatch": True})
        db_manager = DBManager.get_instance()
        percentiles = NeuronView.load_percentiles(db_manager, subject, QTILE_KEYS)
        neuron_view = NeuronView(subject, db_manager, make_chat_conversation(), percentiles)
    
    # Generate response and collect activations
    print("Generating response and analyzing activations...")
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        generator = neuron_view.send_message(
            subject, prompt, max_new_tokens=30, temperature=0, stream=True
        )
        tokens = []
        for token in generator:
            if isinstance(token, int):
                tokens.append(token)
        
        response = subject.decode(tokens).strip()
    
    # Extract answer
    answer_lines = response.split('\n')
    answer = ""
    for line in answer_lines:
        if line and "header" not in line and "eot_id" not in line:
            answer = line.strip()
            break
    
    print(f"Original answer: {answer}")
    
    # Find highly activated neurons
    activation_filter = ActivationPercentileFilter(
        min_activation_percentile=activation_threshold
    )
    neuron_view.set_filter(activation_filter)
    activated_neurons = neuron_view.get_neurons(with_tokens=True)
    
    # Get metadata and filter interesting neurons
    neurons_metadata = neuron_view.get_neurons_metadata_dict(
        activated_neurons, include_run_metadata=True
    )
    
    interesting_neurons = []
    for neuron in activated_neurons:
        meta = neurons_metadata.general.get((neuron.layer, neuron.neuron))
        if meta and neuron.polarity:
            desc = meta.descriptions.get(neuron.polarity)
            if desc and desc.is_interesting:
                interesting_neurons.append(neuron)
    
    print(f"Found {len(interesting_neurons)} interesting highly-activated neurons")
    
    # Cluster neurons
    clusters, _ = await cluster_neurons(
        interesting_neurons,
        neurons_metadata,
        max_similarity_score=2,
        min_size=3
    )
    
    print(f"\n📊 Found {len(clusters)} suspicious concept clusters:")
    for i, cluster in enumerate(clusters, 1):
        print(f"  {i}. {cluster.description} ({len(cluster.neurons)} neurons)")
    
    return clusters, answer, neuron_view

def ablate_cluster(cluster: Cluster, prompt: str, neuron_view: NeuronView, subject: Subject) -> str:
    """
    Ablate all neurons in a cluster and generate response
    """
    # Prepare interventions for all neurons in cluster
    interventions = {}
    prompt_len = len(subject.tokenizer(prompt)['input_ids'])
    
    for neuron in cluster.neurons:
        for idx in range(prompt_len):
            interventions[(neuron.layer, neuron.neuron, idx)] = 0.0
    
    # Apply interventions
    neuron_view.set_neuron_interventions(interventions)
    
    # Generate with ablation
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        generator = neuron_view.send_message(
            subject, prompt, max_new_tokens=30, temperature=0, stream=True
        )
        tokens = []
        for token in generator:
            if isinstance(token, int):
                tokens.append(token)
        
        response = subject.decode(tokens).strip()
    
    # Clear interventions
    neuron_view.clear_neuron_interventions()
    
    # Extract answer
    answer_lines = response.split('\n')
    answer = ""
    for line in answer_lines:
        if line and "header" not in line and "eot_id" not in line:
            answer = line.strip()
            break
    
    return answer

async def linter_guided_ablation(prompt: str):
    """
    Complete pipeline: AI Linter → Identify clusters → Ablate → Compare
    """
    print("\n" + "="*60)
    print("🤖 AI LINTER-GUIDED ABLATION")
    print("="*60)
    print(f"Prompt: {prompt.split('user')[-1].split('assistant')[0].strip()}")
    print("="*60)
    
    # Phase 1: Find suspicious neurons
    clusters, original_answer, neuron_view = await find_suspicious_neurons(prompt)
    
    if not clusters:
        print("\n✅ No suspicious clusters found - model might be working correctly!")
        return
    
    # Phase 2: Test ablation of each cluster
    print("\n\n🔧 Phase 2: Testing Ablations")
    print("-"*40)
    
    # Get subject for ablation
    subject = neuron_view._subject
    
    results = []
    for i, cluster in enumerate(clusters, 1):
        print(f"\nTesting cluster {i}: {cluster.description}")
        print(f"Ablating {len(cluster.neurons)} neurons...")
        
        ablated_answer = ablate_cluster(cluster, prompt, neuron_view, subject)
        
        print(f"  Original: {original_answer}")
        print(f"  Ablated:  {ablated_answer}")
        
        # Check if it improved
        improved = False
        if "9.8" in prompt or "9.9" in prompt:
            # Check if the answer is now correct
            original_correct = ("9.8" in original_answer and "bigger" in original_answer.lower()) or \
                              ("9.9" in original_answer and "bigger" in original_answer.lower())
            ablated_correct = ("9.8" in ablated_answer and "bigger" in ablated_answer.lower()) or \
                             ("9.9" in ablated_answer and "bigger" in ablated_answer.lower())
            
            improved = not original_correct and ablated_correct
        
        if improved:
            print("  ✅ FIXED THE ERROR!")
        elif ablated_answer != original_answer:
            print("  🔄 Changed but didn't fix")
        else:
            print("  ❌ No change")
        
        results.append({
            'cluster': cluster.description,
            'neurons': len(cluster.neurons),
            'improved': improved,
            'ablated_answer': ablated_answer
        })
    
    # Phase 3: Try ablating ALL suspicious clusters together
    print("\n\n🎯 Phase 3: Combined Ablation")
    print("-"*40)
    print("Ablating ALL suspicious clusters together...")
    
    # Combine all neurons from all clusters
    all_interventions = {}
    prompt_len = len(subject.tokenizer(prompt)['input_ids'])
    total_neurons = 0
    
    for cluster in clusters:
        for neuron in cluster.neurons:
            for idx in range(prompt_len):
                all_interventions[(neuron.layer, neuron.neuron, idx)] = 0.0
            total_neurons += 1
    
    print(f"Ablating {total_neurons} neurons total...")
    
    # Apply all interventions
    neuron_view.set_neuron_interventions(all_interventions)
    
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        generator = neuron_view.send_message(
            subject, prompt, max_new_tokens=30, temperature=0, stream=True
        )
        tokens = []
        for token in generator:
            if isinstance(token, int):
                tokens.append(token)
        
        combined_response = subject.decode(tokens).strip()
    
    neuron_view.clear_neuron_interventions()
    
    # Extract answer
    combined_answer = ""
    for line in combined_response.split('\n'):
        if line and "header" not in line and "eot_id" not in line:
            combined_answer = line.strip()
            break
    
    print(f"\nOriginal: {original_answer}")
    print(f"Combined: {combined_answer}")
    
    # Final summary
    print("\n\n📋 SUMMARY")
    print("="*60)
    print(f"Original answer: {original_answer}")
    print(f"\nSuspicious clusters found:")
    for i, result in enumerate(results, 1):
        status = "✅ FIXED" if result['improved'] else "❌ No fix"
        print(f"  {i}. {result['cluster']} ({result['neurons']} neurons) - {status}")
    
    print(f"\nCombined ablation result: {combined_answer}")
    
    # Check final result
    if "9.8" in prompt or "9.9" in prompt:
        combined_correct = ("9.8" in combined_answer and "bigger" in combined_answer.lower()) or \
                          ("9.9" in combined_answer and "bigger" in combined_answer.lower())
        if combined_correct:
            print("\n🎉 SUCCESS: Combined ablation fixed the error!")
        else:
            print("\n❌ Combined ablation didn't fix the error")

# Test cases
async def main():
    test_prompts = [
        # The classic problematic comparison
        "<|start_header_id|>user<|end_header_id|>\n\nWhich is bigger: 9.8 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        
        # Alternative version
        "<|start_header_id|>user<|end_header_id|>\n\nWhich is bigger: 9.9 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    ]
    
    for prompt in test_prompts:
        try:
            await linter_guided_ablation(prompt)
            print("\n" + "="*80 + "\n")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())