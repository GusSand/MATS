#!/usr/bin/env python3
"""Test order dependency - does 9.11 vs 9.8 show the same pattern?"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_case_with_order(num1, num2, description):
    """Test both orders of a number pair"""
    print(f"\n🔄 {description}")
    print("-" * 50)

    for order_desc, n1, n2 in [("Original", num1, num2), ("Reversed", num2, num1)]:
        print(f"\n📋 {order_desc} Order: {n1} vs {n2}")

        model_name = "EleutherAI/pythia-160m"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda"
        )
        model.eval()

        correct_num = max(float(n1), float(n2))
        correct_answer = n1 if float(n1) == correct_num else n2

        clean_prompt = f"Which is bigger: {n1} or {n2}?"
        buggy_prompt = f"Q: Which is bigger: {n1} or {n2}?\nA:"

        def check_correct(output: str) -> bool:
            return correct_answer in output

        # Get clean activation
        attention_module = model.gpt_neox.layers[6].attention
        saved_activation = None

        def save_hook(module, input, output):
            nonlocal saved_activation
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            saved_activation = hidden_states.detach().cpu()

        clean_inputs = tokenizer(clean_prompt, return_tensors="pt").to("cuda")
        hook = attention_module.register_forward_hook(save_hook)
        with torch.no_grad():
            model(**clean_inputs)
        hook.remove()

        # Test baseline
        inputs = tokenizer(buggy_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        baseline_response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        baseline_correct = check_correct(baseline_response)

        # Test even heads
        even_heads = [0, 2, 4, 6, 8, 10]

        def patch_hook_even(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            batch_size, seq_len, hidden_size = hidden_states.shape
            head_dim = hidden_size // 12
            hidden_reshaped = hidden_states.view(batch_size, seq_len, 12, head_dim)
            saved_reshaped = saved_activation.to(hidden_states.device).view(batch_size, -1, 12, head_dim)
            new_hidden = hidden_reshaped.clone()
            min_seq_len = min(seq_len, saved_reshaped.shape[1])

            for head_idx in even_heads:
                new_hidden[:, :min_seq_len, head_idx, :] = saved_reshaped[:, :min_seq_len, head_idx, :]

            new_hidden = new_hidden.view(batch_size, seq_len, hidden_size)
            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden

        hook = attention_module.register_forward_hook(patch_hook_even)
        inputs = tokenizer(buggy_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        even_response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        hook.remove()
        even_correct = check_correct(even_response)

        # Test odd heads
        odd_heads = [1, 3, 5, 7, 9, 11]

        def patch_hook_odd(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            batch_size, seq_len, hidden_size = hidden_states.shape
            head_dim = hidden_size // 12
            hidden_reshaped = hidden_states.view(batch_size, seq_len, 12, head_dim)
            saved_reshaped = saved_activation.to(hidden_states.device).view(batch_size, -1, 12, head_dim)
            new_hidden = hidden_reshaped.clone()
            min_seq_len = min(seq_len, saved_reshaped.shape[1])

            for head_idx in odd_heads:
                new_hidden[:, :min_seq_len, head_idx, :] = saved_reshaped[:, :min_seq_len, head_idx, :]

            new_hidden = new_hidden.view(batch_size, seq_len, hidden_size)
            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden

        hook = attention_module.register_forward_hook(patch_hook_odd)
        inputs = tokenizer(buggy_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        odd_response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        hook.remove()
        odd_correct = check_correct(odd_response)

        specialization = (1 if even_correct else 0) - (1 if odd_correct else 0)

        # Print results
        print(f"  Correct answer: {correct_answer}")
        print(f"  Baseline: {'✓' if baseline_correct else '✗'} - {baseline_response.strip()[:30]}...")
        print(f"  Even:     {'✓' if even_correct else '✗'} - {even_response.strip()[:30]}...")
        print(f"  Odd:      {'✓' if odd_correct else '✗'} - {odd_response.strip()[:30]}...")
        print(f"  Specialization: {specialization:+.0f}")

        if specialization > 0:
            print(f"  🎯 EVEN HEAD SPECIALIZATION DETECTED")
        elif specialization < 0:
            print(f"  🔴 ODD HEAD SPECIALIZATION DETECTED")
        else:
            print(f"  ⚪ NO SPECIALIZATION")

        del model
        del tokenizer
        torch.cuda.empty_cache()

        return specialization

# Test the key cases with order dependency
print("🔄 ORDER DEPENDENCY ANALYSIS")
print("="*60)
print("Testing whether pattern depends on number order")

# Test the main working case
spec1 = test_case_with_order("9.8", "9.11", "Main Working Case")

# Test a known failing case for comparison
spec2 = test_case_with_order("9.9", "9.11", "Known Failing Case (for comparison)")

print(f"\n🎯 ORDER DEPENDENCY SUMMARY:")
print("="*40)
print(f"9.8 vs 9.11: {spec1:+.0f} | 9.11 vs 9.8: TBD")
print(f"9.9 vs 9.11: {spec2:+.0f} | 9.11 vs 9.9: TBD")

# Based on results, determine if order matters
print(f"\n💡 CONCLUSION:")
if abs(spec1) != abs(spec2):
    print("Order dependency may exist - patterns differ between arrangements")
else:
    print("Order dependency unclear from this test - need more data")