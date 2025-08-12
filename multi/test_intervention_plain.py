#!/usr/bin/env python3
"""
Test if MLP intervention can fix the counting bug with plain prompt
"""
import torch
from transformer_lens import HookedTransformer
import json

def test_intervention():
    """Test if intervention fixes the plain prompt bug"""
    print("="*60)
    print("TESTING INTERVENTION ON PLAIN PROMPT (WHERE BUG EXISTS)")
    print("="*60)
    
    # Load config
    with open('experiment_config.json', 'r') as f:
        config = json.load(f)
    
    control = config['positive_controls']['counting']
    prompt = control['prompt']
    correct_answer = control['correct_answer']
    fix_layers = control['fix_layers']
    
    # Use PLAIN format where the bug exists!
    plain_prompt = prompt  # Just the question, no formatting
    
    print(f"\nUsing PLAIN prompt: {plain_prompt}")
    print(f"Expected answer: {correct_answer}")
    print(f"Fix layers: {fix_layers}")
    
    # Load model
    print("\nLoading model...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    model = HookedTransformer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        device=device,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True
    )
    
    # Enable attention result hooks
    model.cfg.use_attn_result = True
    model.setup()
    
    print(f"Model loaded on {device}")
    
    # Test 1: Without intervention (should show bug)
    print("\n" + "="*60)
    print("TEST 1: Without intervention (expecting bug)")
    print("="*60)
    
    output_before = model.generate(plain_prompt, max_new_tokens=30, temperature=0)
    print(f"Output: {output_before}")
    
    has_bug = "2" in output_before or "two" in output_before.lower()
    has_correct = "3" in output_before or "three" in output_before.lower()
    
    print(f"\n✗ Has bug (says 2): {has_bug}")
    print(f"✓ Has correct (says 3): {has_correct}")
    
    if not has_bug:
        print("\nWARNING: Bug not reproduced! The model might have been updated.")
    
    # Test 2: With MLP intervention
    print("\n" + "="*60)
    print("TEST 2: With MLP intervention")
    print("="*60)
    
    # Reset hooks
    model.reset_hooks()
    
    # Add intervention hook
    def boost_mlp(act, hook):
        """Boost MLP activation for specific layers"""
        return act * 2.0  # Boost by 2x
    
    # Add hooks to all fix layers
    for layer in fix_layers:
        hook_name = f"blocks.{layer}.mlp.hook_post"
        model.add_hook(hook_name, boost_mlp)
        print(f"Added boost hook to layer {layer}")
    
    # Generate with intervention
    output_after = model.generate(plain_prompt, max_new_tokens=30, temperature=0)
    print(f"\nOutput with intervention: {output_after}")
    
    # Check if fixed
    is_fixed = "3" in output_after or "three" in output_after.lower()
    still_has_bug = "2" in output_after or "two" in output_after.lower()
    
    print(f"\n✓ Is fixed (says 3): {is_fixed}")
    print(f"✗ Still has bug (says 2): {still_has_bug}")
    
    # Test 3: Try stronger intervention
    if not is_fixed:
        print("\n" + "="*60)
        print("TEST 3: With STRONGER intervention (3x boost)")
        print("="*60)
        
        model.reset_hooks()
        
        def boost_mlp_strong(act, hook):
            return act * 3.0  # Stronger boost
        
        for layer in fix_layers:
            model.add_hook(f"blocks.{layer}.mlp.hook_post", boost_mlp_strong)
        
        output_strong = model.generate(plain_prompt, max_new_tokens=30, temperature=0)
        print(f"Output with 3x boost: {output_strong}")
        
        is_fixed_strong = "3" in output_strong or "three" in output_strong.lower()
        print(f"\n✓ Fixed with stronger intervention: {is_fixed_strong}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"1. Bug exists with plain prompt: {has_bug}")
    print(f"2. 2x MLP boost fixes it: {is_fixed}")
    if not is_fixed:
        print(f"3. Stronger intervention needed")
    print(f"\nSUCCESS: {has_bug and is_fixed}")
    
    # Clean up
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    test_intervention()