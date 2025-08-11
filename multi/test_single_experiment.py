#!/usr/bin/env python3
"""
Test a single positive control experiment


We are testing the counting bug in the Llama-3.1-8B-Instruct model.

What the Counting Bug Is:

  - Prompt: "How many times does the letter 'r' appear in 'strawberry'?"
  - Correct Answer: "3" (strawberry has 3 r's)
  - The Bug: LLMs often get this wrong, saying 2 or some other number
  - The Fix: Boost MLP activations in layers 5, 6, and 7

  What the Test File Does:

  1. Loads the model (Llama-3.1-8B-Instruct)
  2. Tests WITHOUT intervention:
    - Asks the model the strawberry question
    - Checks if it gets it wrong (the bug exists)
  3. Tests WITH intervention:
    - Adds hooks to layers 5, 6, 7 that multiply MLP activations by 2
    - Asks the same question again
    - Checks if it now gets it right (the fix works)
  4. Reports results: Shows whether the bug existed and whether our intervention fixed it

This is a test of the model's ability to learn a function and fix it, as the original counting bug paper showed to be impossible with a single prompt. 

Thus, we are looking for a small (or possibly zero) probability of success with the original prompt.




"""
import torch
from transformer_lens import HookedTransformer
import json

def test_counting_bug():
    """Test the counting bug (strawberry 'r' count)"""
    print("="*60)
    print("TESTING SINGLE POSITIVE CONTROL: Counting Bug")
    print("="*60)
    
    # Load config
    with open('experiment_config.json', 'r') as f:
        config = json.load(f)
    
    control = config['positive_controls']['counting']
    prompt = control['prompt']
    correct_answer = control['correct_answer']
    fix_layers = control['fix_layers']
    
    print(f"\nPrompt: {prompt}")
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
    
    # Create chat prompt
    chat_prompt = f"<|start_header_id|>user<|end_header_id|>\n{prompt}\n<|start_header_id|>assistant<|end_header_id|>"
    
    # Test 1: Generate without intervention
    print("\n" + "-"*40)
    print("TEST 1: Without intervention")
    print("-"*40)
    
    output_before = model.generate(chat_prompt, max_new_tokens=20, temperature=0)
    print(f"Debug - output type: {type(output_before)}, shape: {output_before.shape if hasattr(output_before, 'shape') else 'N/A'}")
    
    # The generate function returns tokens, we need to decode them
    # If output_before is a tensor of shape [batch, seq_len], we need to handle it properly
    if len(output_before.shape) == 2:
        text_before = model.to_string(output_before[0])  # Take first batch item
    else:
        text_before = model.to_string(output_before)
    
    print(f"Input: {chat_prompt}")
    print(f"Output: {text_before}")
    
    # Check if bug exists
    has_bug = correct_answer.lower() not in text_before.lower()
    print(f"Has bug (doesn't contain '{correct_answer}'): {has_bug}")
    
    # Test 2: Apply MLP intervention
    print("\n" + "-"*40)
    print("TEST 2: With MLP intervention")
    print("-"*40)
    
    # Reset hooks
    model.reset_hooks()
    
    # Add intervention hook
    def boost_mlp(act, hook):
        """Boost MLP activation for specific layers"""
        for layer in fix_layers:
            if f"blocks.{layer}" in hook.name:
                print(f"  Boosting layer {layer} MLP activation by 2x")
                act = act * 2.0
        return act
    
    # Add hooks to all fix layers
    for layer in fix_layers:
        model.add_hook(f"blocks.{layer}.mlp.hook_post", boost_mlp)
    
    print(f"Added MLP boost hooks to layers: {fix_layers}")
    
    # Generate with intervention
    output_after = model.generate(chat_prompt, max_new_tokens=20, temperature=0)
    if len(output_after.shape) == 2:
        text_after = model.to_string(output_after[0])  # Take first batch item
    else:
        text_after = model.to_string(output_after)
    
    print(f"Output with intervention: {text_after}")
    
    # Check if fixed
    is_fixed = correct_answer.lower() in text_after.lower()
    print(f"Is fixed (contains '{correct_answer}'): {is_fixed}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Bug exists: {has_bug}")
    print(f"Intervention fixed it: {is_fixed}")
    print(f"Success: {has_bug and is_fixed}")
    
    # Clean up
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    test_counting_bug()