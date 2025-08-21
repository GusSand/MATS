#!/usr/bin/env python3
"""
Quick test to verify strawberry intervention works with updated config
"""
import torch
from transformer_lens import HookedTransformer
import json

# Load updated config
with open('experiment_config_updated.json', 'r') as f:
    config = json.load(f)

control = config['positive_controls']['counting_strawberry']
prompt = control['prompt']
correct = str(control['correct_answer'])
layers = control['fix_layers']

print(f"Testing: {prompt}")
print(f"Correct answer: {correct}")
print(f"Intervention layers: {layers}")

# Load model
model = HookedTransformer.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    device="cuda:0",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    local_files_only=True
)
model.cfg.use_attn_result = True
model.setup()

# Test without intervention
print("\n1. WITHOUT intervention:")
output_before = model.generate(prompt, max_new_tokens=30, temperature=0)
print(f"Output: {output_before}")
has_bug = "2" in output_before

# Test with intervention
print("\n2. WITH intervention (2x MLP boost):")
model.reset_hooks()

def boost_mlp(act, hook):
    return act * 2.0

for layer in layers:
    model.add_hook(f"blocks.{layer}.mlp.hook_post", boost_mlp)

output_after = model.generate(prompt, max_new_tokens=30, temperature=0)
print(f"Output: {output_after}")
is_fixed = "3" in output_after

print(f"\nResult: Bug exists: {has_bug}, Fixed by intervention: {is_fixed}")
print(f"SUCCESS: {has_bug and is_fixed}")

del model
torch.cuda.empty_cache()