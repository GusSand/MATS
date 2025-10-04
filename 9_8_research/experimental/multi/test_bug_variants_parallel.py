#!/usr/bin/env python3
"""
Test multiple counting bugs and decimal comparison variants in parallel on 4 GPUs
"""
import torch
from transformer_lens import HookedTransformer
import json
import multiprocessing as mp
from datetime import datetime

def test_bugs_on_gpu(gpu_id, test_cases, test_type, result_queue):
    """Test a subset of bugs on one GPU"""
    torch.cuda.set_device(gpu_id)
    
    print(f"GPU {gpu_id}: Testing {len(test_cases)} {test_type} cases")
    
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
    
    results = []
    
    for prompt, correct, expected_wrong in test_cases:
        # Test with plain format (most likely to show bugs)
        output = model.generate(prompt, max_new_tokens=30, temperature=0)
        
        # Extract just the response
        response = output[len(prompt):].strip() if prompt in output else output.strip()
        
        # Check results
        has_correct = correct.lower() in output.lower()
        has_wrong = expected_wrong.lower() in output.lower() if expected_wrong else False
        
        # For counting bugs, also check if any number is mentioned
        if test_type == "counting":
            # Look for any number in response
            import re
            numbers = re.findall(r'\d+', response)
            first_number = numbers[0] if numbers else "none"
        else:
            first_number = None
        
        result = {
            'prompt': prompt,
            'correct': correct,
            'expected_wrong': expected_wrong,
            'response': response[:100],
            'has_correct': has_correct,
            'has_bug': has_wrong or not has_correct,
            'first_number': first_number,
            'gpu_id': gpu_id
        }
        
        results.append(result)
        
        # Quick output
        status = "✗ BUG" if result['has_bug'] else "✓"
        print(f"GPU {gpu_id}: {status} - {prompt[:40]}... -> {first_number if first_number else response[:20]}")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    result_queue.put((test_type, gpu_id, results))

def main():
    print("="*60)
    print("TESTING BUG VARIANTS IN PARALLEL")
    print(f"Time: {datetime.now()}")
    print("="*60)
    
    # Define all test cases
    counting_tests = [
        # Original
        ("How many 'r's in 'strawberry'?", "3", "2"),
        ("How many times does the letter 'r' appear in 'strawberry'?", "3", "2"),
        
        # New variants
        ("How many 's's in 'mississippi'?", "4", "3"),
        ("How many 'e's in 'excellence'?", "4", "3"),
        ("Count the 'l's in 'parallel'", "3", "2"),
        ("How many 'o's in 'cooperation'?", "4", "3"),
        ("Count vowels in 'beautiful'", "5", "4"),
        ("How many 'm's in 'momentum'?", "3", "2"),
        
        # Double letters
        ("How many double letters in 'committee'?", "3", "2"),
        ("How many repeated letters in 'mississippi'?", "4", "3"),
        
        # More variants
        ("Count the letter 'p' in 'hippopotamus'", "3", "2"),
        ("How many 't's in 'intermittent'?", "4", "3"),
        ("Number of 'n's in 'announcement'", "4", "3"),
        ("How many 'c's in 'occurrence'?", "4", "3"),
    ]
    
    decimal_tests = [
        # Controls (should work)
        ("Which is bigger: 8.9 or 8.12?", "8.9", "8.12"),
        ("Which is bigger: 7.8 or 7.10?", "7.8", "7.10"),
        ("Which is bigger: 3.14 or 3.9?", "3.9", "3.14"),
        ("Which is larger: 5.5 or 5.45?", "5.5", "5.45"),
        
        # Main bug and variants
        ("Which is bigger: 9.9 or 9.11?", "9.9", "9.11"),
        ("Which is bigger: 9.10 or 9.9?", "9.9", "9.10"),
        ("Which is bigger: 9.8 or 9.12?", "9.8", "9.12"),
        ("Which is bigger: 10.9 or 10.11?", "10.9", "10.11"),
        
        # More decimal comparisons
        ("Which is greater: 4.8 or 4.75?", "4.8", "4.75"),
        ("Which is larger: 12.9 or 12.88?", "12.9", "12.88"),
        ("What's bigger: 6.7 or 6.65?", "6.7", "6.65"),
        ("Which is more: 2.3 or 2.25?", "2.3", "2.25"),
    ]
    
    # Split tests across GPUs
    counting_per_gpu = len(counting_tests) // 2
    decimal_per_gpu = len(decimal_tests) // 2
    
    gpu_assignments = [
        (0, counting_tests[:counting_per_gpu], "counting"),
        (1, counting_tests[counting_per_gpu:], "counting"),
        (2, decimal_tests[:decimal_per_gpu], "decimal"),
        (3, decimal_tests[decimal_per_gpu:], "decimal"),
    ]
    
    # Setup multiprocessing
    manager = mp.Manager()
    result_queue = manager.Queue()
    processes = []
    
    # Launch parallel tests
    for gpu_id, test_cases, test_type in gpu_assignments:
        p = mp.Process(
            target=test_bugs_on_gpu,
            args=(gpu_id, test_cases, test_type, result_queue)
        )
        p.start()
        processes.append(p)
        print(f"Launched {test_type} tests on GPU {gpu_id} ({len(test_cases)} cases)")
    
    # Wait for completion
    for p in processes:
        p.join()
    
    # Collect results
    counting_results = []
    decimal_results = []
    
    while not result_queue.empty():
        test_type, gpu_id, results = result_queue.get()
        if test_type == "counting":
            counting_results.extend(results)
        else:
            decimal_results.extend(results)
    
    # Display summary
    print("\n" + "="*60)
    print("COUNTING BUG RESULTS")
    print("="*60)
    
    bugs_found = []
    for result in counting_results:
        if result['has_bug']:
            bugs_found.append(result)
            print(f"✗ BUG FOUND: {result['prompt']}")
            print(f"  Expected: {result['correct']}, Got: {result['first_number']}")
        else:
            print(f"✓ No bug: {result['prompt'][:50]}... (correct: {result['correct']})")
    
    print(f"\nTotal counting bugs found: {len(bugs_found)}/{len(counting_results)}")
    
    print("\n" + "="*60)
    print("DECIMAL COMPARISON RESULTS")
    print("="*60)
    
    decimal_bugs = []
    for result in decimal_results:
        if result['has_bug']:
            decimal_bugs.append(result)
            print(f"✗ BUG FOUND: {result['prompt']}")
            print(f"  Expected: {result['correct']}, Wrong answer: {result['expected_wrong']}")
        else:
            print(f"✓ Correct: {result['prompt']} -> {result['correct']}")
    
    print(f"\nTotal decimal bugs found: {len(decimal_bugs)}/{len(decimal_results)}")
    
    # Save to experiment_results.md
    print("\n" + "="*60)
    print("Updating experiment_results.md...")
    
    with open('experiment_results.md', 'a') as f:
        f.write(f"\n\n## Bug Variant Testing Results\n")
        f.write(f"**Test Time**: {datetime.now()}\n\n")
        
        f.write("### Counting Bugs Found\n\n")
        if bugs_found:
            f.write("| Prompt | Correct | Model Response | Bug Type |\n")
            f.write("|--------|---------|----------------|----------|\n")
            for bug in bugs_found:
                prompt_short = bug['prompt'][:40] + "..." if len(bug['prompt']) > 40 else bug['prompt']
                response_short = bug['response'][:30] + "..." if len(bug['response']) > 30 else bug['response']
                f.write(f"| {prompt_short} | {bug['correct']} | {bug['first_number']} | Counting |\n")
        else:
            f.write("No counting bugs found!\n")
        
        f.write("\n### Decimal Comparison Bugs Found\n\n")
        if decimal_bugs:
            f.write("| Prompt | Correct | Shows Bug? |\n")
            f.write("|--------|---------|------------|\n")
            for bug in decimal_bugs:
                f.write(f"| {bug['prompt']} | {bug['correct']} | ✅ YES |\n")
        else:
            f.write("No decimal comparison bugs found!\n")
        
        # Good candidates for positive controls
        f.write("\n### Recommended Positive Controls\n\n")
        f.write("Based on testing, these bugs reliably manifest:\n\n")
        for bug in bugs_found[:5]:  # Top 5
            f.write(f"- **{bug['prompt']}**: Says {bug['first_number']} instead of {bug['correct']}\n")
        for bug in decimal_bugs[:3]:  # Top 3
            f.write(f"- **{bug['prompt']}**: Likely says {bug['expected_wrong']} instead of {bug['correct']}\n")
    
    print("Results saved to experiment_results.md")
    
    return counting_results, decimal_results

if __name__ == "__main__":
    # Set start method for CUDA
    mp.set_start_method('spawn', force=True)
    counting, decimal = main()