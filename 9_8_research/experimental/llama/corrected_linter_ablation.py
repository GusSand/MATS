#!/usr/bin/env python
"""
Corrected AI Linter-Guided Ablation with proper intervention format
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

from neurondb.filters import QTILE_KEYS, ActivationPercentileFilter, NeuronPolarity
from neurondb.postgres import DBManager
from util.subject import Subject, llama31_8B_instruct_config
from neurondb.view import NeuronView
from util.chat_input import make_chat_conversation
from investigator.clustering import cluster_neurons
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

PROMPT = "<|start_header_id|>user<|end_header_id|>\n\nWhich is bigger: 9.8 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

async def run_linter_ablation():
    """
    Use AI Linter to find and ablate suspicious neurons
    """
    print("🤖 AI LINTER-GUIDED ABLATION (Corrected)")
    print("="*60)
    
    # Initialize
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        subject = Subject(llama31_8B_instruct_config, nnsight_lm_kwargs={"dispatch": True})
        db_manager = DBManager.get_instance()
        percentiles = NeuronView.load_percentiles(db_manager, subject, QTILE_KEYS)
    
    # Step 1: Generate baseline response and collect activations
    print("\n1️⃣ Generating baseline response...")
    neuron_view = NeuronView(subject, db_manager, make_chat_conversation(), percentiles)
    
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        generator = neuron_view.send_message(subject, PROMPT, max_new_tokens=30, temperature=0.2, stream=True)
        tokens = []
        for token in generator:
            if isinstance(token, int):
                tokens.append(token)
        baseline_response = subject.decode(tokens)
    
    baseline_answer = baseline_response.split('\n')[-1].strip()
    print(f"Baseline: {baseline_answer}")
    
    # Step 2: Find highly activated neurons
    print("\n2️⃣ Finding highly activated neurons...")
    # Use the correct filter parameters - must be one of: "1e-7", "1e-6", "1e-5", "1e-4"
    activation_filter = ActivationPercentileFilter(
        percentile="1e-5",  # Top 0.001% neurons
        direction="top"  # Top percentile neurons
    )
    neuron_view.set_filter(activation_filter)
    activated_neurons = neuron_view.get_neurons(with_tokens=True)
    
    # Get metadata and filter interesting neurons
    neurons_metadata = neuron_view.get_neurons_metadata_dict(activated_neurons, include_run_metadata=True)
    
    interesting_neurons = []
    for neuron in activated_neurons:
        meta = neurons_metadata.general.get((neuron.layer, neuron.neuron))
        if meta and neuron.polarity:
            desc = meta.descriptions.get(neuron.polarity)
            if desc and desc.is_interesting:
                interesting_neurons.append(neuron)
    
    print(f"Found {len(interesting_neurons)} interesting highly-activated neurons")
    
    if not interesting_neurons:
        print("No interesting neurons found!")
        return
    
    # Step 3: Cluster neurons
    print("\n3️⃣ Clustering neurons to find concepts...")
    try:
        clusters, n_failures = await cluster_neurons(
            interesting_neurons,
            neurons_metadata,
            max_similarity_score=2,
            min_size=2  # Lower threshold to get more clusters
        )
        
        print(f"Found {len(clusters)} concept clusters (failures: {n_failures})")
        
        for i, cluster in enumerate(clusters[:5], 1):  # Show top 5
            print(f"  Cluster {i}: {cluster.description} ({len(cluster.neurons)} neurons)")
    
    except Exception as e:
        print(f"Clustering failed: {e}")
        print("Falling back to using all interesting neurons as one group")
        clusters = []
    
    # Step 4: Ablate neurons using correct method
    print("\n4️⃣ Ablating neurons with quantile-based interventions...")
    
    # Get all neurons to ablate (from clusters or all interesting)
    neurons_to_ablate = []
    if clusters:
        for cluster in clusters:
            neurons_to_ablate.extend(cluster.neurons)
    else:
        neurons_to_ablate = interesting_neurons
    
    print(f"Total neurons to ablate: {len(neurons_to_ablate)}")
    
    # Get metadata for quantiles
    neurons_metadata_dict = neuron_view.get_neurons_metadata_dict(neurons_to_ablate, include_run_metadata=False)
    
    # Build interventions using quantile * strength
    interventions = {}
    prompt_len = len(subject.tokenizer(PROMPT)['input_ids'])
    ABLATION_STRENGTH = -0.1
    
    unique_neurons = set()
    for neuron in neurons_to_ablate:
        unique_neurons.add((neuron.layer, neuron.neuron, neuron.polarity))
    
    for layer, neuron_idx, polarity in unique_neurons:
        metadata = neurons_metadata_dict.general.get((layer, neuron_idx))
        if metadata and polarity:
            quantile_key = "0.9999999" if polarity == NeuronPolarity.POS else "1e-07"
            quantile = metadata.activation_percentiles.get(quantile_key)
            if quantile is not None:
                # Correct format: (layer, token_idx, neuron_idx)
                for token_idx in range(prompt_len + 50):
                    interventions[(layer, token_idx, neuron_idx)] = quantile * ABLATION_STRENGTH
    
    print(f"Built {len(interventions) // (prompt_len + 50)} unique neuron interventions")
    
    # Test multiple runs
    print("\n5️⃣ Testing ablation accuracy (10 runs)...")
    correct = 0
    
    for i in range(10):
        ablated_view = NeuronView(subject, db_manager, make_chat_conversation(), percentiles)
        ablated_view.set_neuron_interventions(interventions)
        
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            generator = ablated_view.send_message(subject, PROMPT, max_new_tokens=30, temperature=0.2, stream=True)
            tokens = []
            for token in generator:
                if isinstance(token, int):
                    tokens.append(token)
            ablated_response = subject.decode(tokens)
        
        ablated_answer = ablated_response.split('\n')[-1].strip()
        
        if "9.8 is bigger" in ablated_answer or "9.8 is larger" in ablated_answer:
            correct += 1
            symbol = "✓"
        elif "9.11 is bigger" in ablated_answer or "9.11 is larger" in ablated_answer:
            symbol = "✗"
        else:
            symbol = "?"
        
        if i < 3:
            print(f"  Run {i+1}: {symbol} {ablated_answer[:50]}")
    
    print(f"\n📊 Results:")
    print(f"  Baseline: {baseline_answer}")
    print(f"  Accuracy with linter-guided ablation: {correct}/10 = {correct*10}%")
    
    if clusters:
        print(f"\n💡 Key insight: AI Linter found {len(clusters)} concept clusters")
        print("   that might be causing the error.")

if __name__ == "__main__":
    asyncio.run(run_linter_ablation())