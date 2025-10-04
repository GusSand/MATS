#!/usr/bin/env python3
"""Test Type 1 cases only to validate pattern"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_number_pair(num1, num2):
    print(f"\n🔢 Testing: {num1} vs {num2}")

    # Load fresh model for each test
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

    # Determine correct answer
    correct_num = max(float(num1), float(num2))
    correct_answer = num1 if float(num1) == correct_num else num2

    clean_prompt = f"Which is bigger: {num1} or {num2}?"
    buggy_prompt = f"Q: Which is bigger: {num1} or {num2}?\nA:"

    def check_correct(output: str) -> bool:
        return correct_answer in output

    # Test baseline
    inputs = tokenizer(buggy_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    baseline_response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    baseline_correct = check_correct(baseline_response)

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

    # Test even heads
    even_heads = [0, 2, 4, 6, 8, 10]

    def patch_hook(module, input, output):
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

    hook = attention_module.register_forward_hook(patch_hook)
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

    print(f"  Baseline: {baseline_correct} | Even: {even_correct} | Odd: {odd_correct} | Spec: {specialization:+.0f}")
    print(f"  Baseline: {baseline_response.strip()[:30]}...")
    print(f"  Even: {even_response.strip()[:30]}...")
    print(f"  Odd: {odd_response.strip()[:30]}...")

    # Cleanup
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return specialization

# Test Type 1 cases
type1_cases = [
    ('9.8', '9.11'),
    ('8.9', '8.10'),
    ('7.8', '7.11'),
    ('6.9', '6.12')
]

print("🧪 TESTING TYPE 1 CASES (Single vs Double Digit Decimals)")
print("="*60)

specializations = []
for num1, num2 in type1_cases:
    spec = test_number_pair(num1, num2)
    specializations.append(spec)

print(f"\n📊 SUMMARY:")
print(f"Average specialization: {sum(specializations)/len(specializations):+.2f}")
print(f"Cases with even preference: {sum(1 for s in specializations if s > 0)}/{len(specializations)}")
print(f"Cases with odd preference: {sum(1 for s in specializations if s < 0)}/{len(specializations)}")
print(f"Cases with no preference: {sum(1 for s in specializations if s == 0)}/{len(specializations)}")