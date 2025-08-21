#!/usr/bin/env python3
"""
Test all 4 positive controls in parallel on 4 GPUs to find which formats show bugs
"""
import torch
from transformer_lens import HookedTransformer
import json
import multiprocessing as mp
from datetime import datetime

def test_control_formats(gpu_id, control_name, control_config, result_queue):
    """Test one control on one GPU with multiple formats"""
    torch.cuda.set_device(gpu_id)
    
    print(f"GPU {gpu_id}: Testing {control_name}")
    
    # Load model
    model = HookedTransformer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        device=f"cuda:{gpu_id}",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True
    )
    model.cfg.use_attn_result = True
    model.setup()
    
    prompt = control_config['prompt']
    correct_answer = str(control_config['correct_answer']).lower()
    
    # Test different formats
    formats = {
        "plain": prompt,
        "chat": f"<|start_header_id|>user<|end_header_id|>\n{prompt}\n<|start_header_id|>assistant<|end_header_id|>",
        "qa": f"Q: {prompt}\nA:",
        "instruction": f"Please answer the following question:\n{prompt}\nAnswer:",
    }
    
    results = {}
    for format_name, formatted_prompt in formats.items():
        # Generate response
        output = model.generate(formatted_prompt, max_new_tokens=30, temperature=0)
        
        # Check if it contains the correct answer
        has_correct = correct_answer in output.lower()
        
        # Store just the generated part (after the prompt)
        if formatted_prompt in output:
            response = output[len(formatted_prompt):].strip()
        else:
            response = output.strip()
        
        results[format_name] = {
            "has_correct": has_correct,
            "output": response[:100],  # First 100 chars
            "full_output": output
        }
        
        print(f"GPU {gpu_id} - {control_name} - {format_name}: {'✓' if has_correct else '✗'}")
    
    # Now test intervention on the format that shows the bug
    bug_format = None
    for fmt, res in results.items():
        if not res['has_correct']:
            bug_format = fmt
            break
    
    intervention_result = None
    if bug_format:
        print(f"GPU {gpu_id} - {control_name}: Bug found in {bug_format} format, testing intervention...")
        
        # Reset hooks
        model.reset_hooks()
        
        # Apply intervention based on control type
        if 'fix_layers' in control_config:
            # MLP intervention
            layers = control_config['fix_layers']
            
            def boost_mlp(act, hook):
                return act * 2.0
            
            for layer in layers:
                model.add_hook(f"blocks.{layer}.mlp.hook_post", boost_mlp)
                
        elif 'fix_heads' in control_config:
            # Attention head intervention
            heads = control_config['fix_heads']
            
            def ablate_heads(pattern, hook):
                for layer, head in heads:
                    if f"blocks.{layer}" in hook.name:
                        pattern[:, head, :, :] = 0
                return pattern
            
            for layer, _ in heads:
                model.add_hook(f"blocks.{layer}.attn.hook_pattern", ablate_heads)
        
        # Test with intervention
        output_fixed = model.generate(formats[bug_format], max_new_tokens=30, temperature=0)
        is_fixed = correct_answer in output_fixed.lower()
        
        intervention_result = {
            "format": bug_format,
            "is_fixed": is_fixed,
            "output": output_fixed
        }
        
        print(f"GPU {gpu_id} - {control_name}: Intervention {'✓ FIXED' if is_fixed else '✗ FAILED'}")
    else:
        print(f"GPU {gpu_id} - {control_name}: No bug found in any format!")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    # Return results
    result_queue.put({
        'control': control_name,
        'gpu_id': gpu_id,
        'formats': results,
        'intervention': intervention_result,
        'correct_answer': control_config['correct_answer']
    })

def main():
    print("="*60)
    print("TESTING ALL POSITIVE CONTROLS IN PARALLEL")
    print(f"Time: {datetime.now()}")
    print("="*60)
    
    # Load config
    with open('experiment_config.json', 'r') as f:
        config = json.load(f)
    
    controls = config['positive_controls']
    
    # Setup multiprocessing
    manager = mp.Manager()
    result_queue = manager.Queue()
    processes = []
    
    # Launch parallel tests
    for gpu_id, (control_name, control_config) in enumerate(controls.items()):
        p = mp.Process(
            target=test_control_formats,
            args=(gpu_id, control_name, control_config, result_queue)
        )
        p.start()
        processes.append(p)
        print(f"Launched {control_name} test on GPU {gpu_id}")
    
    # Wait for all to complete
    for p in processes:
        p.join()
    
    # Collect results
    all_results = []
    while not result_queue.empty():
        all_results.append(result_queue.get())
    
    # Display summary
    print("\n" + "="*60)
    print("SUMMARY OF ALL CONTROLS")
    print("="*60)
    
    for result in sorted(all_results, key=lambda x: x['gpu_id']):
        control = result['control']
        correct = result['correct_answer']
        print(f"\n{control.upper()} (correct: {correct}):")
        
        # Show format results
        for fmt, res in result['formats'].items():
            status = "✓" if res['has_correct'] else "✗ BUG"
            print(f"  {fmt:12} {status}")
        
        # Show intervention result
        if result['intervention']:
            int_res = result['intervention']
            status = "✓ FIXED" if int_res['is_fixed'] else "✗ FAILED"
            print(f"  Intervention on {int_res['format']}: {status}")
        else:
            print(f"  No intervention needed (no bug found)")
    
    # Update experiment_results.md
    print("\n" + "="*60)
    print("Updating experiment_results.md...")
    
    with open('experiment_results.md', 'a') as f:
        f.write("\n\n## Parallel Testing Results (Generated)\n\n")
        f.write(f"**Test Time**: {datetime.now()}\n\n")
        
        for result in sorted(all_results, key=lambda x: x['control']):
            control = result['control']
            f.write(f"### {control.title()}\n")
            f.write(f"**Correct Answer**: {result['correct_answer']}\n\n")
            f.write("| Format | Shows Bug? | Output Sample |\n")
            f.write("|--------|------------|---------------|\n")
            
            for fmt, res in result['formats'].items():
                bug = "✅ YES" if not res['has_correct'] else "❌ No"
                output = res['output'][:50].replace('\n', ' ')
                f.write(f"| {fmt} | {bug} | {output}... |\n")
            
            if result['intervention']:
                int_res = result['intervention']
                f.write(f"\n**Intervention Result**: ")
                f.write(f"{'✅ FIXED' if int_res['is_fixed'] else '❌ FAILED'} ")
                f.write(f"(on {int_res['format']} format)\n")
            f.write("\n")
    
    print("Results saved to experiment_results.md")
    
    return all_results

if __name__ == "__main__":
    # Set start method for CUDA
    mp.set_start_method('spawn', force=True)
    results = main()