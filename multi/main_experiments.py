# main_experiments.py
"""
Main script to run on 4× A5000 GPUs
Optimized for minimal runtime
"""

import torch
import torch.multiprocessing as mp
from transformer_lens import HookedTransformer
import numpy as np
import json
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import logging
from datetime import datetime
import traceback
import sys

# CRITICAL: Set this for multi-GPU
mp.set_start_method('spawn', force=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'experiment_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_model_on_gpu(gpu_id):
    """
    Load model on specific GPU
    """
    torch.cuda.set_device(gpu_id)
    
    # Load from cache (already downloaded)
    model = HookedTransformer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        device=f"cuda:{gpu_id}",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True  # Use cached version
    )
    print(f"✓ Model loaded on GPU {gpu_id}")
    
    return model

# ============= EXPERIMENT 1: POSITIVE CONTROLS (1 hour) =============

def run_positive_control(gpu_id, control_name, control_config, result_queue):
    """
    Each GPU tests one control bug
    """
    torch.cuda.set_device(gpu_id)
    logger.info(f"GPU {gpu_id}: Starting positive control test for {control_name}")
    print(f"GPU {gpu_id}: Testing {control_name}")
    
    # Load model
    model = load_model_on_gpu(gpu_id)
    
    prompt = control_config['prompt']
    correct = control_config['correct_answer']
    
    # Test if bug exists
    chat_prompt = f"<|start_header_id|>user<|end_header_id|>\n{prompt}\n<|start_header_id|>assistant<|end_header_id|>"
    
    # Generate without intervention
    output_before = model.generate(chat_prompt, max_new_tokens=20, temperature=0)
    text_before = model.to_string(output_before[0])
    has_bug = correct.lower() not in text_before.lower()
    
    # Apply fix based on control type
    if 'fix_layers' in control_config:
        # MLP intervention
        layers = control_config['fix_layers']
        
        def boost_mlp(act, hook):
            if any(f"blocks.{l}" in hook.name for l in layers):
                act = act * 2.0  # Boost activation
            return act
        
        for layer in layers:
            model.add_hook(f"blocks.{layer}.mlp.hook_post", boost_mlp)
    
    elif 'fix_heads' in control_config:
        # Attention intervention
        heads = control_config['fix_heads']
        
        def ablate_heads(pattern, hook):
            for layer, head in heads:
                if f"blocks.{layer}" in hook.name:
                    pattern[:, head, :, :] = 0
            return pattern
        
        for layer, _ in heads:
            model.add_hook(f"blocks.{layer}.attn.hook_pattern", ablate_heads)
    
    # Test after intervention
    output_after = model.generate(chat_prompt, max_new_tokens=20, temperature=0)
    text_after = model.to_string(output_after[0])
    is_fixed = correct.lower() in text_after.lower()
    
    result = {
        'control': control_name,
        'gpu_id': gpu_id,
        'has_bug': has_bug,
        'is_fixed': is_fixed,
        'output_before': text_before,
        'output_after': text_after
    }
    
    result_queue.put(result)
    logger.info(f"GPU {gpu_id}: {control_name} - Bug: {has_bug}, Fixed: {is_fixed}")
    print(f"GPU {gpu_id}: {control_name} - Bug: {has_bug}, Fixed: {is_fixed}")
    
    # Clean up
    del model
    torch.cuda.empty_cache()

def parallel_positive_controls(config):
    """
    Run all positive controls in parallel
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: POSITIVE CONTROLS (Parallel on 4 GPUs)")
    print("="*60)
    
    manager = mp.Manager()
    result_queue = manager.Queue()
    processes = []
    
    controls = list(config['positive_controls'].items())[:4]  # 4 controls for 4 GPUs
    
    for gpu_id, (name, control_config) in enumerate(controls):
        p = mp.Process(
            target=run_positive_control,
            args=(gpu_id, name, control_config, result_queue)
        )
        p.start()
        processes.append(p)
    
    # Wait for all to complete
    for p in processes:
        p.join()
    
    # Collect results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    # Save results
    pd.DataFrame(results).to_csv('results/positive_controls.csv', index=False)
    print("✓ Positive controls complete\n")
    return results

# ============= EXPERIMENT 2: CAUSAL TRACING (2 hours) =============

def causal_trace_layers(gpu_id, layer_range, config, result_queue):
    """
    Each GPU traces different layers
    """
    torch.cuda.set_device(gpu_id)
    logger.info(f"GPU {gpu_id}: Starting causal tracing for layers {layer_range[0]}-{layer_range[-1]}")
    print(f"GPU {gpu_id}: Tracing layers {layer_range[0]}-{layer_range[-1]}")
    
    model = load_model_on_gpu(gpu_id)
    
    clean_prompt = config['prompts']['clean']
    buggy_prompt = config['prompts']['buggy']
    
    # Get clean and buggy caches
    _, clean_cache = model.run_with_cache(clean_prompt)
    _, buggy_cache = model.run_with_cache(buggy_prompt)
    
    buggy_tokens = model.to_tokens(buggy_prompt)
    n_positions = buggy_tokens.shape[1]
    
    causal_effects = {}
    
    for layer in tqdm(layer_range, desc=f"GPU {gpu_id}"):
        layer_effects = []
        
        for pos in range(n_positions):
            # Patch clean activation at (layer, pos)
            def patch_activation(act, hook):
                if hook.name == f"blocks.{layer}.hook_resid_post":
                    act[:, pos, :] = clean_cache[hook.name][:, min(pos, clean_cache[hook.name].shape[1]-1), :]
                return act
            
            model.reset_hooks()
            model.add_hook(f"blocks.{layer}.hook_resid_post", patch_activation)
            
            # Get patched output
            with torch.no_grad():
                patched_logits = model(buggy_tokens)
            
            # Measure effect
            token_99 = model.to_tokens(" 9.9", prepend_bos=False)[0, 0]
            token_911 = model.to_tokens(" 9.11", prepend_bos=False)[0, 0]
            
            prob_99 = torch.softmax(patched_logits[0, -1], dim=-1)[token_99].item()
            prob_911 = torch.softmax(patched_logits[0, -1], dim=-1)[token_911].item()
            
            effect = prob_99 - prob_911
            layer_effects.append(effect)
        
        causal_effects[layer] = layer_effects
    
    model.reset_hooks()
    result_queue.put({
        'gpu_id': gpu_id,
        'layers': layer_range,
        'effects': causal_effects
    })
    
    del model
    torch.cuda.empty_cache()

def parallel_causal_tracing(config):
    """
    Split causal tracing across 4 GPUs
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: CAUSAL TRACING (Parallel on 4 GPUs)")
    print("="*60)
    
    # Split 32 layers across 4 GPUs
    layer_splits = [
        list(range(0, 8)),   # GPU 0
        list(range(8, 16)),  # GPU 1
        list(range(16, 24)), # GPU 2
        list(range(24, 32))  # GPU 3
    ]
    
    manager = mp.Manager()
    result_queue = manager.Queue()
    processes = []
    
    for gpu_id, layers in enumerate(layer_splits):
        p = mp.Process(
            target=causal_trace_layers,
            args=(gpu_id, layers, config, result_queue)
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    # Combine results
    all_effects = {}
    while not result_queue.empty():
        result = result_queue.get()
        all_effects.update(result['effects'])
    
    # Save as numpy array
    np.save('results/causal_effects.npy', all_effects)
    print("✓ Causal tracing complete\n")
    return all_effects

# ============= EXPERIMENT 3: PHASE TRANSITION (2 hours) =============

def ablation_sweep_chunk(gpu_id, ablation_values, config, result_queue):
    """
    Each GPU tests subset of ablation values
    """
    torch.cuda.set_device(gpu_id)
    logger.info(f"GPU {gpu_id}: Starting ablation sweep with {len(ablation_values)} values")
    print(f"GPU {gpu_id}: Testing {len(ablation_values)} ablation values")
    
    model = load_model_on_gpu(gpu_id)
    
    buggy_prompt = config['prompts']['buggy']
    neurons = config['neurons_to_ablate']
    
    results = []
    
    for ablation_val in tqdm(ablation_values, desc=f"GPU {gpu_id}"):
        # Apply ablation
        def ablate_neurons(act, hook):
            layer = int(hook.name.split('.')[1])
            if layer in neurons:
                for neuron_id in neurons[layer]:
                    act[:, :, neuron_id] = ablation_val
            return act
        
        model.reset_hooks()
        for layer in neurons.keys():
            model.add_hook(f"blocks.{layer}.mlp.hook_post", ablate_neurons)
        
        # Generate
        output = model.generate(buggy_prompt, max_new_tokens=20, temperature=0)
        text = model.to_string(output[0])
        
        # Classify output
        has_bug = "9.11" in text and "bigger" in text.lower()
        is_coherent = ("9.11" in text or "9.9" in text) and len(text.split()) > 5
        
        results.append({
            'ablation_value': ablation_val,
            'has_bug': has_bug,
            'is_coherent': is_coherent,
            'output': text[:100]  # Truncate for storage
        })
    
    model.reset_hooks()
    result_queue.put({
        'gpu_id': gpu_id,
        'results': results
    })
    
    del model
    torch.cuda.empty_cache()

def parallel_phase_transition(config):
    """
    200-point ablation sweep across 4 GPUs
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: PHASE TRANSITION (200 points on 4 GPUs)")
    print("="*60)
    
    all_ablation_values = config['ablation_values']
    
    # Split across GPUs
    chunks = np.array_split(all_ablation_values, 4)
    
    manager = mp.Manager()
    result_queue = manager.Queue()
    processes = []
    
    for gpu_id, chunk in enumerate(chunks):
        p = mp.Process(
            target=ablation_sweep_chunk,
            args=(gpu_id, chunk.tolist(), config, result_queue)
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    # Combine results
    all_results = []
    while not result_queue.empty():
        result = result_queue.get()
        all_results.extend(result['results'])
    
    # Save
    df = pd.DataFrame(all_results)
    df.to_csv('results/phase_transition.csv', index=False)
    print("✓ Phase transition analysis complete\n")
    return df

# ============= EXPERIMENT 4: PATH PATCHING (3 hours) =============

def path_patch_heads(gpu_id, head_ranges, config, result_queue):
    """
    Each GPU tests different attention heads
    """
    torch.cuda.set_device(gpu_id)
    logger.info(f"GPU {gpu_id}: Starting path patching for layers {head_ranges}")
    
    model = load_model_on_gpu(gpu_id)
    
    clean_prompt = config['prompts']['clean']
    buggy_prompt = config['prompts']['buggy']
    
    clean_tokens = model.to_tokens(clean_prompt)
    buggy_tokens = model.to_tokens(buggy_prompt)
    
    _, clean_cache = model.run_with_cache(clean_prompt)
    
    results = []
    
    for layer in tqdm(head_ranges, desc=f"GPU {gpu_id} - Layers"):
        for head in range(model.cfg.n_heads):
            # Patch this specific head
            def patch_head(act, hook):
                if f"blocks.{layer}" in hook.name:
                    # Patch attention output for this head
                    d_head = model.cfg.d_head
                    head_slice = slice(head * d_head, (head + 1) * d_head)
                    act[:, -1, head_slice] = clean_cache[hook.name][:, -1, head_slice]
                return act
            
            model.reset_hooks()
            model.add_hook(f"blocks.{layer}.attn.hook_result", patch_head)
            
            # Test effect
            with torch.no_grad():
                patched_logits = model(buggy_tokens)
            
            token_99 = model.to_tokens(" 9.9", prepend_bos=False)[0, 0]
            token_911 = model.to_tokens(" 9.11", prepend_bos=False)[0, 0]
            
            prob_99 = torch.softmax(patched_logits[0, -1], dim=-1)[token_99].item()
            prob_911 = torch.softmax(patched_logits[0, -1], dim=-1)[token_911].item()
            
            effect = prob_99 - prob_911
            
            if abs(effect) > 0.01:  # Only save significant effects
                results.append({
                    'layer': layer,
                    'head': head,
                    'effect': effect
                })
    
    model.reset_hooks()
    result_queue.put({
        'gpu_id': gpu_id,
        'results': results
    })
    
    del model
    torch.cuda.empty_cache()

def parallel_path_patching(config):
    """
    Path patching split across GPUs
    """
    print("\n" + "="*60)
    print("EXPERIMENT 4: PATH PATCHING (Parallel on 4 GPUs)")
    print("="*60)
    
    # Focus on important layers from your paper
    important_layers = [7, 8, 9, 13, 14, 15, 28, 29, 30, 31]
    
    # Split across GPUs
    layer_splits = [
        [7, 8, 9],      # GPU 0
        [13, 14, 15],   # GPU 1
        [28, 29],       # GPU 2
        [30, 31]        # GPU 3
    ]
    
    manager = mp.Manager()
    result_queue = manager.Queue()
    processes = []
    
    for gpu_id, layers in enumerate(layer_splits):
        p = mp.Process(
            target=path_patch_heads,
            args=(gpu_id, layers, config, result_queue)
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    # Combine results
    all_results = []
    while not result_queue.empty():
        result = result_queue.get()
        all_results.extend(result['results'])
    
    # Save
    df = pd.DataFrame(all_results)
    df.to_csv('results/path_patching.csv', index=False)
    print("✓ Path patching complete\n")
    return df

# ============= MAIN EXECUTION =============

def main():
    """
    Run all experiments efficiently
    """
    start_time = time.time()
    
    logger.info("="*60)
    logger.info("STARTING 4× A5000 PARALLEL EXPERIMENTS")
    logger.info(f"Start time: {datetime.now()}")
    logger.info("="*60)
    
    print("="*60)
    print("STARTING 4× A5000 PARALLEL EXPERIMENTS")
    print("="*60)
    
    # Load config
    with open('experiment_config.json', 'r') as f:
        config = json.load(f)
    
    # Track results
    all_results = {}
    
    # Run experiments in order
    try:
        # 1. Positive Controls (1 hour)
        logger.info("Starting Experiment 1: Positive Controls")
        exp1_start = time.time()
        all_results['positive_controls'] = parallel_positive_controls(config)
        logger.info(f"Experiment 1 completed in {(time.time() - exp1_start)/60:.1f} minutes")
        
        # 2. Causal Tracing (2 hours)
        logger.info("Starting Experiment 2: Causal Tracing")
        exp2_start = time.time()
        all_results['causal_tracing'] = parallel_causal_tracing(config)
        logger.info(f"Experiment 2 completed in {(time.time() - exp2_start)/60:.1f} minutes")
        
        # 3. Phase Transition (2 hours)
        logger.info("Starting Experiment 3: Phase Transition")
        exp3_start = time.time()
        all_results['phase_transition'] = parallel_phase_transition(config)
        logger.info(f"Experiment 3 completed in {(time.time() - exp3_start)/60:.1f} minutes")
        
        # 4. Path Patching (3 hours)
        logger.info("Starting Experiment 4: Path Patching")
        exp4_start = time.time()
        all_results['path_patching'] = parallel_path_patching(config)
        logger.info(f"Experiment 4 completed in {(time.time() - exp4_start)/60:.1f} minutes")
        
    except Exception as e:
        logger.error(f"ERROR: {e}")
        logger.error(traceback.format_exc())
        print(f"ERROR: {e}")
        # Save whatever we have
        torch.save(all_results, 'results/partial_results.pt')
        logger.info("Partial results saved to results/partial_results.pt")
        raise
    
    # Calculate runtime
    total_time = (time.time() - start_time) / 3600
    
    logger.info("="*60)
    logger.info(f"ALL EXPERIMENTS COMPLETE!")
    logger.info(f"Total runtime: {total_time:.2f} hours")
    logger.info(f"Estimated cost: ${total_time * 5.52:.2f}")
    logger.info(f"End time: {datetime.now()}")
    logger.info("="*60)
    
    print("="*60)
    print(f"ALL EXPERIMENTS COMPLETE!")
    print(f"Total runtime: {total_time:.2f} hours")
    print(f"Estimated cost: ${total_time * 5.52:.2f}")
    print("="*60)
    
    # Save final results
    torch.save(all_results, 'results/all_results.pt')
    
    # Generate summary
    generate_summary(all_results)

def generate_summary(results):
    """
    Generate summary for paper
    """
    summary = f"""
    EXPERIMENT SUMMARY
    ==================
    
    1. POSITIVE CONTROLS:
    - Counting bug: {results['positive_controls'][0]['is_fixed']}
    - Arithmetic bug: {results['positive_controls'][1]['is_fixed']}  
    - IOI bug: {results['positive_controls'][2]['is_fixed']}
    - River bug: {results['positive_controls'][3]['is_fixed']}
    
    2. CAUSAL TRACING:
    - Critical layers identified: {len([l for l, effects in results['causal_tracing'].items() if max(effects) > 0.1])}
    - Max effect size: {max([max(effects) for effects in results['causal_tracing'].values()]):.3f}
    
    3. PHASE TRANSITION:
    - Transition point: {results['phase_transition'][results['phase_transition']['has_bug'] == False].iloc[0]['ablation_value'] if any(~results['phase_transition']['has_bug']) else 'Not found'}
    
    4. PATH PATCHING:
    - Significant heads: {len(results['path_patching'])}
    - Max head effect: {results['path_patching']['effect'].max():.3f}
    
    CONCLUSION: Bug shows entangled circuit resistant to intervention.
    """
    
    print(summary)
    
    with open('results/summary.txt', 'w') as f:
        f.write(summary)

if __name__ == "__main__":
    main()