#!/usr/bin/env python3
"""
Sparse Targeted Activation Editing to Fix Empty Response Bug
Goal: Fix the bug where chat format produces empty responses with constraints
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("SPARSE EDITING TO FIX EMPTY RESPONSE BUG")
print("="*70)

# Load model
model_name = "meta-llama/Llama-3.1-8B-Instruct"
print(f"\nLoading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()
device = next(model.parameters()).device

# Test prompts - pairs of (chat_format, simple_format)
TEST_PROMPTS = [
    ("What is water? One word answer only.", "What is water? One word answer only.\nAnswer:"),
    ("What color is the sky? Answer with just one word.", "What color is the sky? Answer with just one word.\nAnswer:"),
    ("Define democracy. Maximum 10 words.", "Define democracy. Maximum 10 words.\nAnswer:"),
    ("What is 2+2? Answer with just the number.", "What is 2+2? Answer with just the number.\nAnswer:"),
]

# Convert to chat format
CHAT_PROMPTS = []
SIMPLE_PROMPTS = []
for question, simple in TEST_PROMPTS:
    messages = [{"role": "user", "content": question}]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    CHAT_PROMPTS.append(chat)
    SIMPLE_PROMPTS.append(simple)

print(f"\nTest prompts prepared: {len(TEST_PROMPTS)} pairs")
print(f"Example chat prompt: {CHAT_PROMPTS[0][:50]}...")
print(f"Example simple prompt: {SIMPLE_PROMPTS[0]}")

# Step 1: Find differential neurons
print("\n" + "="*70)
print("STEP 1: FINDING DIFFERENTIAL NEURONS")
print("="*70)

# Collect activations for both formats
activations_chat = {}
activations_simple = {}

def collect_activations(prompts, storage_dict):
    """Collect MLP neuron activations for given prompts"""
    hooks = []
    
    def get_hook(layer_idx):
        def hook(module, input, output):
            # Get intermediate activations before down_proj
            x = input[0]
            gate_output = module.act_fn(module.gate_proj(x))
            up_output = module.up_proj(x)
            intermediate = gate_output * up_output
            # Store mean activation across sequence
            storage_dict[layer_idx] = intermediate.mean(dim=1).detach().cpu()
        return hook
    
    # Register hooks on all layers
    for layer_idx in range(len(model.model.layers)):
        hook = model.model.layers[layer_idx].mlp.register_forward_hook(
            get_hook(layer_idx)
        )
        hooks.append(hook)
    
    # Run forward pass
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model(**inputs)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()

print("Collecting activations for chat format...")
collect_activations(CHAT_PROMPTS[:2], activations_chat)

print("Collecting activations for simple format...")
collect_activations(SIMPLE_PROMPTS[:2], activations_simple)

# Find neurons with high differential activation
print("\nFinding differential neurons...")
differential_neurons = []

for layer_idx in range(len(model.model.layers)):
    if layer_idx in activations_chat and layer_idx in activations_simple:
        chat_act = activations_chat[layer_idx]
        simple_act = activations_simple[layer_idx]
        
        # Compute difference in activation
        diff = (chat_act - simple_act).abs().mean(dim=0)
        
        # Get top neurons in this layer
        top_k = min(100, diff.shape[0])
        values, indices = torch.topk(diff, top_k)
        
        for i in range(min(5, top_k)):  # Top 5 per layer
            neuron_idx = indices[i].item()
            diff_value = values[i].item()
            if diff_value > 0.01:  # Threshold for significance
                differential_neurons.append((layer_idx, neuron_idx, diff_value))

# Sort by differential activation
differential_neurons.sort(key=lambda x: x[2], reverse=True)

# Select candidate neurons (focus on middle to late layers for response gating)
CANDIDATE_NEURONS = []
for layer, neuron, diff in differential_neurons[:50]:
    if 10 <= layer <= 30:  # Focus on middle to late layers
        CANDIDATE_NEURONS.append((layer, neuron))
    if len(CANDIDATE_NEURONS) >= 20:
        break

print(f"\nSelected {len(CANDIDATE_NEURONS)} candidate neurons")
print(f"Examples: {CANDIDATE_NEURONS[:5]}")

# Step 2: Sparse editing module
print("\n" + "="*70)
print("STEP 2: TRAINING SPARSE EDIT")
print("="*70)

class SparseEdit(nn.Module):
    def __init__(self, neurons, device):
        super().__init__()
        self.neurons = neurons
        # Gate parameters (sigmoid will give 0-1)
        self.gates = nn.Parameter(torch.zeros(len(neurons), device=device))
        # Scale and shift for each neuron
        self.scales = nn.Parameter(torch.ones(len(neurons), device=device) * 0.1)
        self.shifts = nn.Parameter(torch.zeros(len(neurons), device=device))
    
    def get_sparse_mask(self, temperature=0.1):
        """Get binary mask using sigmoid gating"""
        return torch.sigmoid(self.gates / temperature)
    
    def get_edits(self):
        """Get the edit parameters for active neurons"""
        mask = self.get_sparse_mask()
        return mask, self.scales, self.shifts

edit_module = SparseEdit(CANDIDATE_NEURONS, device)
optimizer = optim.Adam(edit_module.parameters(), lr=0.01)

# Hook storage
edit_hooks = []

def register_edit_hooks():
    global edit_hooks
    # Remove old hooks
    for hook in edit_hooks:
        hook.remove()
    edit_hooks = []
    
    # Add new hooks
    for idx, (layer_idx, neuron_idx) in enumerate(CANDIDATE_NEURONS):
        def get_activation_hook(layer_idx, neuron_idx, edit_idx):
            def hook(module, input, output):
                if edit_module.training or hasattr(edit_module, 'apply_edits'):
                    # Get intermediate activations
                    x = input[0]
                    gate_output = module.act_fn(module.gate_proj(x))
                    up_output = module.up_proj(x)
                    intermediate = gate_output * up_output
                    
                    # Apply edit
                    mask, scales, shifts = edit_module.get_edits()
                    gate = mask[edit_idx]
                    scale = scales[edit_idx]
                    shift = shifts[edit_idx]
                    
                    # Edit: boost activation to encourage generation
                    old_val = intermediate[:, :, neuron_idx].clone()
                    new_val = old_val * (1 + gate * scale) + gate * shift
                    intermediate[:, :, neuron_idx] = new_val
                    
                    # Recompute output with edited activations
                    return module.down_proj(intermediate)
                return output
            return hook
        
        hook = model.model.layers[layer_idx].mlp.register_forward_hook(
            get_activation_hook(layer_idx, neuron_idx, idx)
        )
        edit_hooks.append(hook)

def compute_loss(chat_prompt, simple_prompt):
    """Compute loss to encourage chat format to generate like simple format"""
    
    # Get simple format output (target behavior)
    inputs_simple = tokenizer(simple_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs_simple = model(**inputs_simple)
        target_logits = outputs_simple.logits[0, -1, :].detach()
    
    # Get chat format output with edits
    inputs_chat = tokenizer(chat_prompt, return_tensors="pt").to(device)
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        outputs_chat = model(**inputs_chat)
        edited_logits = outputs_chat.logits[0, -1, :]
    
    # KL divergence loss - encourage chat to match simple
    target_probs = torch.softmax(target_logits, dim=-1)
    edited_log_probs = torch.log_softmax(edited_logits, dim=-1)
    kl_loss = torch.sum(target_probs * (torch.log(target_probs + 1e-8) - edited_log_probs))
    
    return kl_loss

# Training loop
print("\nTraining sparse edit...")
register_edit_hooks()
edit_module.train()

training_history = []
for step in range(100):
    optimizer.zero_grad()
    
    # Compute loss on each prompt pair
    total_loss = 0
    for chat_prompt, simple_prompt in zip(CHAT_PROMPTS[:2], SIMPLE_PROMPTS[:2]):
        loss = compute_loss(chat_prompt, simple_prompt)
        total_loss += loss
    
    # L0 regularization (encourage sparsity)
    mask = edit_module.get_sparse_mask()
    l0_loss = mask.sum()
    target_k = 5  # Target number of neurons
    l0_penalty = (l0_loss - target_k).abs()
    
    # Total loss
    total_loss = total_loss / len(CHAT_PROMPTS[:2]) + 0.1 * l0_penalty
    
    total_loss.backward()
    optimizer.step()
    
    # Clamp parameters
    with torch.no_grad():
        edit_module.scales.clamp_(0, 2.0)  # Only positive scaling
        edit_module.shifts.clamp_(-0.5, 0.5)
    
    training_history.append({
        'step': step,
        'loss': float(total_loss.item()),
        'active_neurons': float(mask.sum().item())
    })
    
    if step % 20 == 0:
        print(f"Step {step:3d}: loss={total_loss.item():.3f}, active={mask.sum().item():.1f}")

# Step 3: Select top neurons and evaluate
print("\n" + "="*70)
print("STEP 3: EVALUATION")
print("="*70)

edit_module.eval()
edit_module.apply_edits = True  # Flag to apply edits during evaluation

with torch.no_grad():
    mask = edit_module.get_sparse_mask(temperature=0.01)
    selected_indices = torch.topk(mask, k=min(5, len(CANDIDATE_NEURONS))).indices
    
    selected_neurons = [CANDIDATE_NEURONS[i] for i in selected_indices.cpu().numpy()]
    selected_params = {
        f"L{layer}/N{neuron}": {
            'gate': float(mask[i].item()),
            'scale': float(edit_module.scales[i].item()),
            'shift': float(edit_module.shifts[i].item())
        }
        for i, (layer, neuron) in zip(selected_indices.cpu().numpy(), selected_neurons)
    }

print(f"\nSelected neurons: {selected_neurons}")
for name, params in selected_params.items():
    print(f"  {name}: gate={params['gate']:.3f}, scale={params['scale']:.3f}, shift={params['shift']:.3f}")

# Test the fix
print("\n" + "="*70)
print("TESTING THE FIX")
print("="*70)

def test_generation(prompt, with_edit=False):
    """Generate text and check if response is produced"""
    if with_edit:
        register_edit_hooks()
        edit_module.eval()
        edit_module.apply_edits = True
    else:
        for hook in edit_hooks:
            hook.remove()
        edit_hooks.clear()
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = response[len(prompt):].strip()
    
    return len(generated) > 0, generated[:50] if generated else "[EMPTY]"

success_count = 0
for i, (question, _) in enumerate(TEST_PROMPTS):
    messages = [{"role": "user", "content": question}]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print(f"\nTest {i+1}: {question[:40]}...")
    
    has_response_before, text_before = test_generation(chat_prompt, with_edit=False)
    has_response_after, text_after = test_generation(chat_prompt, with_edit=True)
    
    print(f"  Before edit: {text_before}")
    print(f"  After edit:  {text_after}")
    
    if not has_response_before and has_response_after:
        print(f"  ✓ FIXED! Empty response now generates content")
        success_count += 1
    elif has_response_before and has_response_after:
        print(f"  ◐ Both generate responses (no bug to fix)")
    else:
        print(f"  ✗ Still empty after edit")

success_rate = (success_count / len(TEST_PROMPTS)) * 100

# Save results
results = {
    'bug_type': 'empty_response_in_chat_format',
    'selected_neurons': [(int(l), int(n)) for l, n in selected_neurons],
    'parameters': selected_params,
    'training_history': training_history[-20:],
    'evaluation': {
        'test_cases': len(TEST_PROMPTS),
        'fixed_cases': success_count,
        'success_rate': success_rate
    }
}

with open('sparse_edit_empty_response_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Success rate: {success_rate:.0f}% ({success_count}/{len(TEST_PROMPTS)} cases fixed)")
if success_rate > 0:
    print("\n✓ Sparse editing successfully fixed some empty response bugs!")
    print("This demonstrates that the method can work on format-specific bugs.")
else:
    print("\n✗ Sparse editing did not fix the empty response bug.")
    print("The bug may require different neurons or parameters.")

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Training loss
ax = axes[0]
steps = [h['step'] for h in training_history]
losses = [h['loss'] for h in training_history]
ax.plot(steps, losses, 'b-', linewidth=2)
ax.set_xlabel('Training Step', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Training Progress', fontsize=14)
ax.grid(True, alpha=0.3)

# Active neurons
ax = axes[1]
active = [h['active_neurons'] for h in training_history]
ax.plot(steps, active, 'g-', linewidth=2)
ax.axhline(y=5, color='r', linestyle='--', label='Target')
ax.set_xlabel('Training Step', fontsize=12)
ax.set_ylabel('Active Neurons', fontsize=12)
ax.set_title('Sparsity Over Time', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Selected neurons
ax = axes[2]
neuron_names = list(selected_params.keys())
gate_values = [p['gate'] for p in selected_params.values()]
colors = ['#2ecc71' if success_rate > 0 else '#e74c3c'] * len(neuron_names)
ax.barh(neuron_names, gate_values, color=colors)
ax.set_xlabel('Gate Activation', fontsize=12)
ax.set_title(f'Selected Neurons (Success: {success_rate:.0f}%)', fontsize=14)
ax.set_xlim(0, 1.1)

plt.suptitle('Sparse Editing for Empty Response Bug', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('sparse_edit_empty_response.png', dpi=300, bbox_inches='tight')
plt.savefig('sparse_edit_empty_response.pdf', format='pdf', bbox_inches='tight')

print(f"\nVisualization saved to sparse_edit_empty_response.png/pdf")
print("Results saved to sparse_edit_empty_response_results.json")

print("\n" + "="*70)
print("EXPERIMENT COMPLETE")
print("="*70)