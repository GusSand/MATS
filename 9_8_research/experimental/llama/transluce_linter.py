#!/usr/bin/env python
"""
AI Linter - Automatically find suspicious concept clusters in model activations
Based on Transluce's Monitor interface
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

from neurondb.filters import QTILE_KEYS, ActivationPercentileFilter
from neurondb.postgres import DBManager
from util.subject import Subject, llama31_8B_instruct_config
from neurondb.view import NeuronView
from util.chat_input import make_chat_conversation
from investigator.clustering import cluster_neurons
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

async def run_ai_linter(prompt: str, activation_threshold: float = 0.9):
    """
    Run the AI Linter on a given prompt to find suspicious concept clusters
    
    Args:
        prompt: The prompt to analyze
        activation_threshold: Minimum activation percentile to consider (0.9 = top 10%)
    """
    print("🤖 AI LINTER - Concept Cluster Analysis")
    print("="*60)
    print(f"Analyzing: {prompt[:50]}...")
    print("="*60)
    
    # Initialize
    print("\n⚙️ Initializing...")
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        subject = Subject(llama31_8B_instruct_config, nnsight_lm_kwargs={"dispatch": True})
        db_manager = DBManager.get_instance()
        percentiles = NeuronView.load_percentiles(db_manager, subject, QTILE_KEYS)
        neuron_view = NeuronView(subject, db_manager, make_chat_conversation(), percentiles)
    
    # Generate response to get activations
    print("📝 Generating response and collecting activations...")
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        # Send message to activate neurons
        generator = neuron_view.send_message(
            subject, prompt, max_new_tokens=30, temperature=0, stream=True
        )
        tokens = []
        for token in generator:
            if isinstance(token, int):
                tokens.append(token)
        
        response = subject.decode(tokens).strip()
    
    print(f"Response: {response.split(chr(10))[-1][:100]}...")
    
    # Filter highly activated neurons
    print(f"\n🔍 Finding neurons with activation > {activation_threshold}...")
    activation_filter = ActivationPercentileFilter(
        min_activation_percentile=activation_threshold
    )
    neuron_view.set_filter(activation_filter)
    activated_neurons = neuron_view.get_neurons(with_tokens=True)
    
    print(f"Found {len(activated_neurons)} highly activated neurons")
    
    # Get metadata for clustering
    neurons_metadata = neuron_view.get_neurons_metadata_dict(
        activated_neurons, include_run_metadata=True
    )
    
    # Filter to only interesting neurons
    interesting_neurons = []
    for neuron in activated_neurons:
        meta = neurons_metadata.general.get((neuron.layer, neuron.neuron))
        if meta and neuron.polarity:
            desc = meta.descriptions.get(neuron.polarity)
            if desc and desc.is_interesting:
                interesting_neurons.append(neuron)
    
    print(f"Filtered to {len(interesting_neurons)} interesting neurons")
    
    # Run clustering
    print("\n🔬 Clustering neurons by semantic similarity...")
    clusters, failures = await cluster_neurons(
        interesting_neurons,
        neurons_metadata,
        max_similarity_score=2,  # Only very similar concepts
        min_size=3  # At least 3 neurons per cluster
    )
    
    # Display results
    print(f"\n📊 FOUND {len(clusters)} CONCEPT CLUSTERS:")
    print("="*60)
    
    for i, cluster in enumerate(clusters, 1):
        print(f"\n🚨 Cluster {i}: {cluster.description}")
        print(f"   Similarity: {cluster.similarity}/4 (1=most similar)")
        print(f"   Neurons: {len(cluster.neurons)}")
        print("   Examples:")
        
        # Show first 3 neuron descriptions
        for j, neuron in enumerate(cluster.neurons[:3]):
            print(f"     - L{neuron.layer}N{neuron.neuron}: {neuron.description[:80]}...")
            
        if len(cluster.neurons) > 3:
            print(f"     ... and {len(cluster.neurons) - 3} more")
    
    if not clusters:
        print("\n✅ No suspicious concept clusters found!")
    else:
        print(f"\n⚠️  These {len(clusters)} concept clusters might explain unexpected behavior!")
    
    return clusters, response

# Test prompts
TEST_PROMPTS = [
    # The classic example from the paper
    "<|start_header_id|>user<|end_header_id|>\n\nWhich is bigger: 9.8 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    
    # Variations
    "<|start_header_id|>user<|end_header_id|>\n\nWhat is larger: 9.9 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    
    # Control - should work correctly
    "<|start_header_id|>user<|end_header_id|>\n\nWhich is bigger: 9.8 or 9.2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
]

async def main():
    print("🔧 TRANSLUCE AI LINTER DEMO")
    print("Automatically finding suspicious concept clusters\n")
    
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}/{len(TEST_PROMPTS)}")
        print('='*60)
        
        try:
            clusters, response = await run_ai_linter(prompt)
            
            # Check if response is correct
            if "9.8" in prompt or "9.9" in prompt:
                correct = ("9.8" in response and "bigger" in response.lower()) or \
                         ("9.9" in response and "bigger" in response.lower())
                if not correct and clusters:
                    print("\n💡 The AI Linter found concept clusters that might explain this error!")
                    
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())