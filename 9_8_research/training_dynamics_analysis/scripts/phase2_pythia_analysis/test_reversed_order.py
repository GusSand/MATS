#!/usr/bin/env python3
"""Quick test of 9.11 vs 9.8 (reversed order)"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("🔄 TESTING REVERSED ORDER: 9.11 vs 9.8")
print("="*50)

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

# Test 9.11 vs 9.8 (reversed from our standard 9.8 vs 9.11)
num1, num2 = "9.11", "9.8"
correct_answer = "9.11"  # 9.11 > 9.8

clean_prompt = f"Which is bigger: {num1} or {num2}?"
buggy_prompt = f"Q: Which is bigger: {num1} or {num2}?\nA:"

def check_correct(output: str) -> bool:
    return correct_answer in output

print(f"Testing: {num1} vs {num2}")
print(f"Correct answer: {correct_answer}")

# Get clean activation
attention_module = model.gpt_neox.layers[6].attention
saved_activation = None

def save_hook(module, input, output):
    global saved_activation
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

print(f"\nBaseline: {'✓' if baseline_correct else '✗'}")
print(f"Response: {baseline_response.strip()}")

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

print(f"\nEven heads: {'✓' if even_correct else '✗'}")
print(f"Response: {even_response.strip()}")

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

print(f"\nOdd heads: {'✓' if odd_correct else '✗'}")
print(f"Response: {odd_response.strip()}")

specialization = (1 if even_correct else 0) - (1 if odd_correct else 0)

print(f"\n🎯 SPECIALIZATION: {specialization:+.0f}")

if specialization > 0:
    print("✅ EVEN HEAD SPECIALIZATION")
elif specialization < 0:
    print("🔴 ODD HEAD SPECIALIZATION")
else:
    print("⚪ NO SPECIALIZATION")

print(f"\n🔍 COMPARISON WITH ORIGINAL:")
print(f"9.8 vs 9.11:  Even specialization (+1)")
print(f"9.11 vs 9.8:  {'Even specialization (+1)' if specialization > 0 else 'Odd specialization (-1)' if specialization < 0 else 'No specialization (0)'}")

if specialization != 1:
    print(f"\n🚨 ORDER DEPENDENCY DETECTED!")
    print(f"The pattern is order-dependent, making it even more specific.")
else:
    print(f"\n✅ ORDER INDEPENDENCE CONFIRMED!")
    print(f"The pattern works regardless of number order.")