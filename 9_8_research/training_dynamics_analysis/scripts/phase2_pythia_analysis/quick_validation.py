#!/usr/bin/env python3
"""Quick validation of 9.8 vs 9.11 with fixed methodology"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model
print("Loading Pythia-160M...")
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

# Test original case
clean_prompt = "Which is bigger: 9.8 or 9.11?"
buggy_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"

def check_bug_fixed(output: str) -> bool:
    output_lower = output.lower()
    correct_patterns = ["9.8 is bigger", "9.8 is larger", "9.8"]
    bug_patterns = ["9.11 is bigger", "9.11 is larger", "9.11"]
    has_correct = any(pattern in output_lower for pattern in correct_patterns)
    has_bug = any(pattern in output_lower for pattern in bug_patterns)
    return has_correct and not has_bug

# Test baseline
print("\nTesting baseline (no patching)...")
inputs = tokenizer(buggy_prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.pad_token_id)
response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(f"Baseline response: {response.strip()}")
print(f"Bug fixed: {check_bug_fixed(response)}")

# Get clean activation
print("\nSaving clean activation...")
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

print(f"Saved activation shape: {saved_activation.shape}")

# Test even heads
print("\nTesting even heads...")
even_heads = [0, 2, 4, 6, 8, 10]

def patch_hook(module, input, output):
    if isinstance(output, tuple):
        hidden_states = output[0]
    else:
        hidden_states = output

    batch_size, seq_len, hidden_size = hidden_states.shape
    head_dim = hidden_size // 12

    # Reshape
    hidden_reshaped = hidden_states.view(batch_size, seq_len, 12, head_dim)
    saved_reshaped = saved_activation.to(hidden_states.device).view(batch_size, -1, 12, head_dim)

    new_hidden = hidden_reshaped.clone()
    min_seq_len = min(seq_len, saved_reshaped.shape[1])

    # Patch even heads
    for head_idx in even_heads:
        new_hidden[:, :min_seq_len, head_idx, :] = saved_reshaped[:, :min_seq_len, head_idx, :]

    # Reshape back
    new_hidden = new_hidden.view(batch_size, seq_len, hidden_size)

    if isinstance(output, tuple):
        return (new_hidden,) + output[1:]
    return new_hidden

hook = attention_module.register_forward_hook(patch_hook)
inputs = tokenizer(buggy_prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.pad_token_id)
response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
hook.remove()

print(f"Even heads response: {response.strip()}")
print(f"Bug fixed: {check_bug_fixed(response)}")

print("\n" + "="*50)
print("QUICK VALIDATION COMPLETE")
print("="*50)