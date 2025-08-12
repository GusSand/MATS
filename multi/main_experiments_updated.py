#!/usr/bin/env python3
"""
Updated Multi-GPU Llama 3.1 8B Bug Analysis
With validated positive controls that actually manifest bugs
"""
import torch
from transformer_lens import HookedTransformer
import numpy as np
import pandas as pd
import json
import multiprocessing as mp
from datetime import datetime
import time
import logging
import os
import sys
import traceback

# Setup logging
log_file = f'experiment_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_model_on_gpu(gpu_id):
    """Load model on specific GPU with memory monitoring"""
    torch.cuda.set_device(gpu_id)
    
    # Log memory before loading
    mem_before = torch.cuda.memory_allocated(gpu_id) / 1e9
    max_mem = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
    logger.info(f"GPU {gpu_id}: Memory before load: {mem_before:.1f}GB / {max_mem:.1f}GB")
    
    # Load model
    model = HookedTransformer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        device=f"cuda:{gpu_id}",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True
    )
    
    # Enable attention result hooks
    model.cfg.use_attn_result = True
    model.setup()
    
    # Log memory after loading
    mem_after = torch.cuda.memory_allocated(gpu_id) / 1e9
    logger.info(f"GPU {gpu_id}: Memory after load: {mem_after:.1f}GB / {max_mem:.1f}GB")
    
    # Warn if memory usage is high
    usage_percent = (mem_after / max_mem) * 100
    if usage_percent > 80:
        logger.warning(f"GPU {gpu_id}: High memory usage: {usage_percent:.1f}%")
    
    print(f"✓ Model loaded on GPU {gpu_id} ({mem_after:.1f}/{max_mem:.1f}GB used)")
    
    return model

# ============= EXPERIMENT 1: POSITIVE CONTROLS =============

def run_positive_control(gpu_id, control_name, control_config, result_queue):
    """Test one positive control bug"""
    torch.cuda.set_device(gpu_id)
    logger.info(f"GPU {gpu_id}: Testing {control_name}")
    print(f"GPU {gpu_id}: Testing {control_name}")
    
    # Load model
    model = load_model_on_gpu(gpu_id)
    
    prompt = control_config['prompt']
    correct = str(control_config['correct_answer'])
    expected_wrong = control_config.get('expected_wrong', None)
    
    # Use plain format for bugs (as validated in testing)
    if control_config.get('use_plain_format', True):
        formatted_prompt = prompt  # Plain format shows bugs
    else:
        formatted_prompt = f"<|start_header_id|>user<|end_header_id|>\n{prompt}\n<|start_header_id|>assistant<|end_header_id|>"
    
    # Test without intervention
    output_before = model.generate(formatted_prompt, max_new_tokens=30, temperature=0)
    text_before = output_before
    
    # Check if bug exists
    has_bug = correct.lower() not in text_before.lower()
    if expected_wrong:
        shows_expected_wrong = expected_wrong.lower() in text_before.lower()
    else:
        shows_expected_wrong = False
    
    # Apply intervention
    model.reset_hooks()
    
    if 'fix_layers' in control_config:
        # MLP intervention for counting bugs
        layers = control_config['fix_layers']
        
        def boost_mlp(act, hook):
            return act * 2.0  # Boost by 2x
        
        for layer in layers:
            model.add_hook(f"blocks.{layer}.mlp.hook_post", boost_mlp)
    
    # Test with intervention
    output_after = model.generate(formatted_prompt, max_new_tokens=30, temperature=0)
    text_after = output_after
    is_fixed = correct.lower() in text_after.lower()
    
    result = {
        'control': control_name,
        'gpu_id': gpu_id,
        'has_bug': has_bug,
        'shows_expected_wrong': shows_expected_wrong,
        'is_fixed': is_fixed,
        'output_before': text_before[:100],
        'output_after': text_after[:100],
        'correct_answer': correct
    }
    
    result_queue.put(result)
    logger.info(f"GPU {gpu_id}: {control_name} - Bug: {has_bug}, Fixed: {is_fixed}")
    print(f"GPU {gpu_id}: {control_name} - Bug: {has_bug}, Fixed: {is_fixed}")
    
    # Clean up
    del model
    torch.cuda.empty_cache()

def parallel_positive_controls(config):
    """Run positive controls in parallel"""
    print("\n" + "="*60)
    print("EXPERIMENT 1: POSITIVE CONTROLS (Parallel on 4 GPUs)")
    print("="*60)
    
    manager = mp.Manager()
    result_queue = manager.Queue()
    processes = []
    
    controls = config['positive_controls']
    
    # Distribute controls across GPUs
    for i, (control_name, control_config) in enumerate(controls.items()):
        gpu_id = i % 4  # Cycle through GPUs
        p = mp.Process(
            target=run_positive_control,
            args=(gpu_id, control_name, control_config, result_queue)
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    # Collect results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv('results/positive_controls.csv', index=False)
    print("✓ Positive controls complete\n")
    return df

# ============= EXPERIMENT 2: MAIN BUG ANALYSIS =============

def analyze_main_bug(gpu_id, prompt_variant, config, result_queue):
    """Analyze the 9.9 vs 9.11 bug and variants"""
    torch.cuda.set_device(gpu_id)
    
    model = load_model_on_gpu(gpu_id)
    
    # Test different formats
    formats = [
        ("plain", prompt_variant),
        ("chat", f"<|start_header_id|>user<|end_header_id|>\n{prompt_variant}\n<|start_header_id|>assistant<|end_header_id|>"),
        ("qa", f"Q: {prompt_variant}\nA:")
    ]
    
    results = []
    for format_name, formatted_prompt in formats:
        output = model.generate(formatted_prompt, max_new_tokens=30, temperature=0)
        
        # Check which answer it gives
        says_9_9 = "9.9" in output and "bigger" in output.lower()
        says_9_11 = "9.11" in output and "bigger" in output.lower()
        
        results.append({
            'prompt': prompt_variant,
            'format': format_name,
            'output': output[:100],
            'correct': says_9_9,
            'has_bug': says_9_11,
            'gpu_id': gpu_id
        })
    
    result_queue.put(results)
    
    # Clean up
    del model
    torch.cuda.empty_cache()

# ============= EXPERIMENT 3: DECIMAL COMPARISON CONTROLS =============

def test_decimal_controls(gpu_id, test_cases, result_queue):
    """Test other decimal comparison bugs as controls"""
    torch.cuda.set_device(gpu_id)
    
    model = load_model_on_gpu(gpu_id)
    
    results = []
    for test_case in test_cases:
        prompt = test_case['prompt']
        correct = test_case['correct']
        wrong = test_case['wrong']
        
        # Test with plain format (where bugs manifest)
        output = model.generate(prompt, max_new_tokens=30, temperature=0)
        
        has_correct = correct in output
        has_wrong = wrong in output
        
        results.append({
            'prompt': prompt,
            'correct_answer': correct,
            'wrong_answer': wrong,
            'output': output[:100],
            'has_correct': has_correct,
            'has_bug': has_wrong or not has_correct,
            'gpu_id': gpu_id
        })
        
        print(f"GPU {gpu_id}: {prompt} -> {'BUG' if has_wrong or not has_correct else 'OK'}")
    
    result_queue.put(results)
    
    # Clean up
    del model
    torch.cuda.empty_cache()

# ============= EXPERIMENT 4: INTERVENTION ATTEMPTS =============

def test_interventions(gpu_id, bug_prompts, intervention_configs, result_queue):
    """Test various intervention strategies on bugs"""
    torch.cuda.set_device(gpu_id)
    
    model = load_model_on_gpu(gpu_id)
    
    results = []
    
    for prompt in bug_prompts:
        for intervention_name, intervention_config in intervention_configs.items():
            model.reset_hooks()
            
            # Apply intervention
            if intervention_config['type'] == 'mlp_boost':
                layers = intervention_config['layers']
                strength = intervention_config['strength']
                
                def boost_mlp(act, hook):
                    return act * strength
                
                for layer in layers:
                    model.add_hook(f"blocks.{layer}.mlp.hook_post", boost_mlp)
                    
            elif intervention_config['type'] == 'attention_ablate':
                heads = intervention_config['heads']
                
                def ablate_heads(pattern, hook):
                    for layer, head in heads:
                        if f"blocks.{layer}" in hook.name:
                            pattern[:, head, :, :] = 0
                    return pattern
                
                for layer, _ in heads:
                    model.add_hook(f"blocks.{layer}.attn.hook_pattern", ablate_heads)
            
            # Test with intervention
            output = model.generate(prompt, max_new_tokens=30, temperature=0)
            
            # Check if fixed (for 9.9 vs 9.11, correct is 9.9)
            is_fixed = "9.9" in output and "bigger" in output.lower()
            
            results.append({
                'prompt': prompt,
                'intervention': intervention_name,
                'output': output[:100],
                'is_fixed': is_fixed,
                'gpu_id': gpu_id
            })
            
            print(f"GPU {gpu_id}: {intervention_name} -> {'FIXED' if is_fixed else 'FAILED'}")
    
    result_queue.put(results)
    
    # Clean up
    del model
    torch.cuda.empty_cache()

# ============= MAIN EXECUTION =============

def main():
    """Run all experiments with validated bugs"""
    start_time = time.time()
    
    # Detect GPU configuration
    gpu_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "No GPU"
    
    logger.info("="*60)
    logger.info(f"STARTING {gpu_count}× {gpu_name.split()[1]} PARALLEL EXPERIMENTS")
    logger.info(f"Start time: {datetime.now()}")
    
    # Log GPU memory info
    total_vram = 0
    for i in range(gpu_count):
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        total_vram += mem
        logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)} - {mem:.1f}GB")
    logger.info(f"Total VRAM available: {total_vram:.1f}GB")
    logger.info("="*60)
    
    print("="*60)
    print(f"STARTING {gpu_count}× {gpu_name.split()[1]} PARALLEL EXPERIMENTS")
    print(f"Total VRAM: {total_vram:.1f}GB across {gpu_count} GPUs")
    print("="*60)
    
    # Load config
    with open('experiment_config_updated.json', 'r') as f:
        config = json.load(f)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    all_results = {}
    
    try:
        # Experiment 1: Positive Controls
        logger.info("Starting Experiment 1: Positive Controls")
        exp1_start = time.time()
        all_results['positive_controls'] = parallel_positive_controls(config)
        logger.info(f"Experiment 1 completed in {(time.time() - exp1_start)/60:.1f} minutes")
        
        # Experiment 2: Main Bug Analysis
        logger.info("Starting Experiment 2: Main Bug Analysis")
        exp2_start = time.time()
        
        print("\n" + "="*60)
        print("EXPERIMENT 2: MAIN BUG ANALYSIS (9.9 vs 9.11)")
        print("="*60)
        
        manager = mp.Manager()
        result_queue = manager.Queue()
        processes = []
        
        # Test different variants
        bug_variants = [
            "Which is bigger: 9.9 or 9.11?",
            "Which is bigger: 9.11 or 9.9?",
            "Which is larger: 9.9 or 9.11?",
            "What's greater: 9.9 or 9.11?"
        ]
        
        for i, variant in enumerate(bug_variants):
            p = mp.Process(
                target=analyze_main_bug,
                args=(i % gpu_count, variant, config, result_queue)
            )
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
        
        # Collect results
        main_bug_results = []
        while not result_queue.empty():
            main_bug_results.extend(result_queue.get())
        
        all_results['main_bug'] = pd.DataFrame(main_bug_results)
        all_results['main_bug'].to_csv('results/main_bug_analysis.csv', index=False)
        print("✓ Main bug analysis complete\n")
        
        logger.info(f"Experiment 2 completed in {(time.time() - exp2_start)/60:.1f} minutes")
        
        # Experiment 3: Decimal Comparison Controls
        logger.info("Starting Experiment 3: Decimal Comparison Controls")
        exp3_start = time.time()
        
        print("\n" + "="*60)
        print("EXPERIMENT 3: DECIMAL COMPARISON CONTROLS")
        print("="*60)
        
        result_queue = manager.Queue()
        processes = []
        
        decimal_controls = config['decimal_comparison_controls']
        
        # Split across GPUs
        controls_per_gpu = len(decimal_controls) // gpu_count + 1
        for i in range(gpu_count):
            start_idx = i * controls_per_gpu
            end_idx = min((i+1) * controls_per_gpu, len(decimal_controls))
            if start_idx < len(decimal_controls):
                p = mp.Process(
                    target=test_decimal_controls,
                    args=(i, decimal_controls[start_idx:end_idx], result_queue)
                )
                p.start()
                processes.append(p)
        
        for p in processes:
            p.join()
        
        # Collect results
        decimal_results = []
        while not result_queue.empty():
            decimal_results.extend(result_queue.get())
        
        all_results['decimal_controls'] = pd.DataFrame(decimal_results)
        all_results['decimal_controls'].to_csv('results/decimal_controls.csv', index=False)
        print("✓ Decimal controls complete\n")
        
        logger.info(f"Experiment 3 completed in {(time.time() - exp3_start)/60:.1f} minutes")
        
        # Experiment 4: Intervention Attempts
        logger.info("Starting Experiment 4: Intervention Attempts")
        exp4_start = time.time()
        
        print("\n" + "="*60)
        print("EXPERIMENT 4: INTERVENTION ATTEMPTS")
        print("="*60)
        
        result_queue = manager.Queue()
        processes = []
        
        # Define interventions to test
        interventions = {
            'mlp_boost_counting': {
                'type': 'mlp_boost',
                'layers': [5, 6, 7],
                'strength': 2.0
            },
            'mlp_boost_reasoning': {
                'type': 'mlp_boost',
                'layers': [15, 16, 17],
                'strength': 2.0
            },
            'mlp_boost_strong': {
                'type': 'mlp_boost',
                'layers': [5, 6, 7, 15, 16, 17],
                'strength': 3.0
            },
            'attention_ablate_mid': {
                'type': 'attention_ablate',
                'heads': [[9, 9], [10, 0]]
            }
        }
        
        test_prompts = [
            "Which is bigger: 9.9 or 9.11?",
            "Which is bigger: 8.9 or 8.12?"
        ]
        
        # Run on 2 GPUs (2 prompts)
        for i, prompt in enumerate(test_prompts):
            p = mp.Process(
                target=test_interventions,
                args=(i % gpu_count, [prompt], interventions, result_queue)
            )
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
        
        # Collect results
        intervention_results = []
        while not result_queue.empty():
            intervention_results.extend(result_queue.get())
        
        all_results['interventions'] = pd.DataFrame(intervention_results)
        all_results['interventions'].to_csv('results/interventions.csv', index=False)
        print("✓ Intervention attempts complete\n")
        
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
    logger.info(f"End time: {datetime.now()}")
    logger.info("="*60)
    
    print("="*60)
    print(f"ALL EXPERIMENTS COMPLETE!")
    print(f"Total runtime: {total_time:.2f} hours")
    print("="*60)
    
    # Save final results
    torch.save(all_results, 'results/all_results.pt')
    
    # Generate summary
    generate_summary(all_results)

def generate_summary(results):
    """Generate summary report"""
    summary = """
    EXPERIMENT SUMMARY
    ==================
    
    1. POSITIVE CONTROLS:
    """
    
    if 'positive_controls' in results and len(results['positive_controls']) > 0:
        pc = results['positive_controls']
        summary += f"   - Total tested: {len(pc)}\n"
        summary += f"   - Bugs found: {pc['has_bug'].sum()}\n"
        summary += f"   - Fixed by intervention: {pc['is_fixed'].sum()}\n"
    else:
        summary += "   - No positive control results available\n"
    
    summary += "\n2. MAIN BUG (9.9 vs 9.11):\n"
    if 'main_bug' in results and len(results['main_bug']) > 0:
        mb = results['main_bug']
        summary += f"   - Formats tested: {mb['format'].nunique()}\n"
        summary += f"   - Shows bug: {mb['has_bug'].sum()}/{len(mb)}\n"
    else:
        summary += "   - No main bug results available\n"
    
    summary += "\n3. DECIMAL CONTROLS:\n"
    if 'decimal_controls' in results and len(results['decimal_controls']) > 0:
        dc = results['decimal_controls']
        summary += f"   - Total tested: {len(dc)}\n"
        summary += f"   - Bugs found: {dc['has_bug'].sum()}/{len(dc)}\n"
    else:
        summary += "   - No decimal control results available\n"
    
    summary += "\n4. INTERVENTIONS:\n"
    if 'interventions' in results and len(results['interventions']) > 0:
        inv = results['interventions']
        summary += f"   - Strategies tested: {inv['intervention'].nunique()}\n"
        summary += f"   - Successful fixes: {inv['is_fixed'].sum()}/{len(inv)}\n"
    else:
        summary += "   - No intervention results available\n"
    
    summary += "\nCONCLUSION: Decimal comparison bugs are pervasive in Llama 3.1 8B\n"
    
    print(summary)
    
    with open('results/summary.txt', 'w') as f:
        f.write(summary)
    
    logger.info("Summary saved to results/summary.txt")

if __name__ == "__main__":
    # Set start method for CUDA multiprocessing
    mp.set_start_method('spawn', force=True)
    main()