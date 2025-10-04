#!/usr/bin/env python3
"""Test pattern boundaries - is it specific to 9.8 vs 9.11?"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_single_case(num1, num2):
    """Test a single number pair quickly"""
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

    correct_num = max(float(num1), float(num2))
    correct_answer = num1 if float(num1) == correct_num else num2

    clean_prompt = f"Which is bigger: {num1} or {num2}?"
    buggy_prompt = f"Q: Which is bigger: {num1} or {num2}?\nA:"

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

    del model
    del tokenizer
    torch.cuda.empty_cache()

    return {
        'baseline': baseline_correct,
        'even': even_correct,
        'odd': odd_correct,
        'specialization': specialization,
        'baseline_response': baseline_response.strip()[:40],
        'even_response': even_response.strip()[:40],
        'odd_response': odd_response.strip()[:40]
    }

# Test strategic cases
test_cases = [
    # Exact variations of 9.8 vs 9.11
    ('9.8', '9.11'),  # Original - should work
    ('9.8', '9.12'),  # Change last digit
    ('9.7', '9.11'),  # Change first decimal
    ('9.8', '9.10'),  # Change to .10

    # Same structure, different numbers
    ('8.8', '8.11'),  # Same pattern, different integer
    ('5.8', '5.11'),  # Same pattern, different integer

    # Different structures
    ('9.9', '9.11'),  # Different first decimal
    ('9.8', '9.21'),  # Different second decimal structure
]

print("🔬 TESTING PATTERN BOUNDARIES")
print("="*50)
print("Investigating what makes 9.8 vs 9.11 special\n")

results = []
for num1, num2 in test_cases:
    print(f"Testing {num1} vs {num2}...", end=" ")
    result = test_single_case(num1, num2)
    results.append((f"{num1} vs {num2}", result))

    if result['specialization'] > 0:
        print("✅ EVEN")
    elif result['specialization'] < 0:
        print("🔴 ODD")
    else:
        print("⚪ NONE")

print(f"\n📊 DETAILED RESULTS:")
print("-" * 70)
print(f"{'Case':<15} {'Base':<4} {'Even':<4} {'Odd':<4} {'Spec':<4} {'Pattern'}")
print("-" * 70)

working_cases = []
for case, result in results:
    base_icon = "✓" if result['baseline'] else "✗"
    even_icon = "✓" if result['even'] else "✗"
    odd_icon = "✓" if result['odd'] else "✗"
    spec = result['specialization']

    if spec > 0:
        pattern = "✅ EVEN"
        working_cases.append(case)
    elif spec < 0:
        pattern = "🔴 ODD"
    else:
        pattern = "⚪ NONE"

    print(f"{case:<15} {base_icon:<4} {even_icon:<4} {odd_icon:<4} {spec:+2.0f}   {pattern}")

print(f"\n🎯 KEY FINDINGS:")
print(f"Working cases with even specialization: {len(working_cases)}")
for case in working_cases:
    print(f"  • {case}")

if len(working_cases) == 1:
    print(f"\n🚨 CRITICAL FINDING: Pattern appears highly specific to 9.8 vs 9.11!")
elif len(working_cases) > 1:
    print(f"\n🔍 PATTERN IDENTIFIED: Multiple cases show even specialization")
else:
    print(f"\n❌ NO PATTERNS: Even the original case failed to replicate")