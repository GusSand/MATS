#!/usr/bin/env python3
"""
Simplified Sparse Targeted Activation Editing
Using gradient-based optimization to find minimal neuron edits
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
print("SPARSE TARGETED ACTIVATION EDITING - SIMPLIFIED")
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

# Prompts
chat_messages = [{"role": "user", "content": "Which is bigger: 9.8 or 9.11?"}]
BAD_PROMPT = tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
GOOD_PROMPT = "Which is bigger: 9.8 or 9.11?\nAnswer:"

print(f"Bad prompt (Chat): {BAD_PROMPT[:50]}...")
print(f"Good prompt (Simple): {GOOD_PROMPT}")

# Candidate neurons from our analysis
NEURONS = [
    (7, 1978), (13, 10352), (14, 2451), (14, 12639),
    (14, 13315), (15, 421), (15, 3136), (15, 5076)
]
print(f"\nCandidate neurons: {NEURONS}")

# Learnable edit parameters
class SparseEdit(nn.Module):
    def __init__(self, neurons, device):
        super().__init__()
        self.neurons = neurons
        # Gate parameters (sigmoid will give 0-1)
        self.gates = nn.Parameter(torch.zeros(len(neurons), device=device))
        # Scale and shift for each neuron
        self.scales = nn.Parameter(torch.zeros(len(neurons), device=device))
        self.shifts = nn.Parameter(torch.zeros(len(neurons), device=device))
    
    def get_sparse_mask(self, temperature=0.1):
        """Get binary mask using Gumbel-softmax for top-k selection"""
        return torch.sigmoid(self.gates / temperature)
    
    def get_edits(self):
        """Get the edit parameters for active neurons"""
        mask = self.get_sparse_mask()
        return mask, self.scales, self.shifts

edit_module = SparseEdit(NEURONS, device)
optimizer = optim.Adam(edit_module.parameters(), lr=0.01)

# Hook storage
stored_activations = {}
edit_hooks = []

def get_activation_hook(layer_idx, neuron_idx, edit_idx):
    """Create hook to edit specific neuron activation"""
    def hook(module, input, output):
        # Get intermediate activations (before down_proj)
        x = input[0]
        gate_output = module.act_fn(module.gate_proj(x))
        up_output = module.up_proj(x)
        intermediate = gate_output * up_output
        
        # Apply edit if training
        if edit_module.training:
            mask, scales, shifts = edit_module.get_edits()
            gate = mask[edit_idx]
            scale = scales[edit_idx]
            shift = shifts[edit_idx]
            
            # Edit: new = old + gate * (scale * old + shift)
            old_val = intermediate[:, :, neuron_idx].clone()
            new_val = old_val + gate * (scale * old_val + shift)
            intermediate[:, :, neuron_idx] = new_val
            
            # Recompute output with edited activations
            return module.down_proj(intermediate)
        
        return output
    
    return hook

# Register hooks
def register_edit_hooks():
    global edit_hooks
    # Remove old hooks
    for hook in edit_hooks:
        hook.remove()
    edit_hooks = []
    
    # Add new hooks
    for idx, (layer_idx, neuron_idx) in enumerate(NEURONS):
        hook = model.model.layers[layer_idx].mlp.register_forward_hook(
            get_activation_hook(layer_idx, neuron_idx, idx)
        )
        edit_hooks.append(hook)

def compute_loss(prompt, target_next_token):
    """Compute cross-entropy loss for next token prediction"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # Last token logits
        
        # Get token IDs for "9.8" and "9.11"
        token_98 = tokenizer.encode(" 9.8", add_special_tokens=False)[0]
        token_911 = tokenizer.encode(" 9.11", add_special_tokens=False)[0]
        
        # Compute log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)
        
        if target_next_token == "9.8":
            # Want high prob for 9.8, low for 9.11
            loss = -log_probs[token_98] + 0.5 * log_probs[token_911]
        else:
            # Want high prob for 9.11, low for 9.8
            loss = -log_probs[token_911] + 0.5 * log_probs[token_98]
    
    return loss

print("\n" + "="*70)
print("TRAINING SPARSE EDIT")
print("="*70)

register_edit_hooks()
edit_module.train()

training_history = []
for step in range(100):
    optimizer.zero_grad()
    
    # Loss on bad prompt (want it to say 9.8, not 9.11)
    loss_bad = compute_loss(BAD_PROMPT, "9.8")
    
    # Loss on good prompt (preserve saying 9.8)
    loss_good = compute_loss(GOOD_PROMPT, "9.8")
    
    # L0 regularization (encourage sparsity)
    mask = edit_module.get_sparse_mask()
    l0_loss = mask.sum()
    target_k = 4  # Target number of neurons
    l0_penalty = (l0_loss - target_k).abs()
    
    # Parameter regularization
    param_reg = edit_module.scales.pow(2).mean() + edit_module.shifts.pow(2).mean()
    
    # Total loss
    total_loss = loss_bad + 0.5 * loss_good + 0.1 * l0_penalty + 0.01 * param_reg
    
    total_loss.backward()
    optimizer.step()
    
    # Clamp parameters to reasonable range
    with torch.no_grad():
        edit_module.scales.clamp_(-0.5, 0.5)
        edit_module.shifts.clamp_(-0.2, 0.2)
    
    training_history.append({
        'step': step,
        'loss': float(total_loss.item()),
        'active_neurons': float(mask.sum().item())
    })
    
    if step % 20 == 0:
        print(f"Step {step:3d}: loss={total_loss.item():.3f}, active={mask.sum().item():.1f}")

# Select top neurons
print("\n" + "="*70)
print("SELECTING TOP NEURONS")
print("="*70)

edit_module.eval()
with torch.no_grad():
    mask = edit_module.get_sparse_mask(temperature=0.01)
    selected_indices = torch.topk(mask, k=min(4, len(NEURONS))).indices
    
    selected_neurons = [NEURONS[i] for i in selected_indices.cpu().numpy()]
    selected_params = {
        f"L{layer}/N{neuron}": {
            'gate': float(mask[i].item()),
            'scale': float(edit_module.scales[i].item()),
            'shift': float(edit_module.shifts[i].item())
        }
        for i, (layer, neuron) in zip(selected_indices.cpu().numpy(), selected_neurons)
    }

print(f"Selected neurons: {selected_neurons}")
for name, params in selected_params.items():
    print(f"  {name}: gate={params['gate']:.3f}, scale={params['scale']:.3f}, shift={params['shift']:.3f}")

# Evaluation
print("\n" + "="*70)
print("EVALUATION")
print("="*70)

def evaluate_generation(prompt, with_edit=False):
    """Generate text and check what the model says"""
    if with_edit:
        register_edit_hooks()
        edit_module.eval()
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
    generated = response[len(prompt):]
    
    # Check if it says 9.8 or 9.11
    if "9.8" in generated and "bigger" in generated.lower():
        return "correct", generated[:50]
    elif "9.11" in generated and "bigger" in generated.lower():
        return "bug", generated[:50]
    else:
        return "unclear", generated[:50]

# Test on bad prompt (Chat template)
print("\nBad Prompt (Chat Template):")
result_before, text_before = evaluate_generation(BAD_PROMPT, with_edit=False)
print(f"  Before edit: {result_before} - {text_before}")

result_after, text_after = evaluate_generation(BAD_PROMPT, with_edit=True)
print(f"  After edit:  {result_after} - {text_after}")

# Test on good prompt (Simple format)
print("\nGood Prompt (Simple Format):")
result_before, text_before = evaluate_generation(GOOD_PROMPT, with_edit=False)
print(f"  Before edit: {result_before} - {text_before}")

result_after, text_after = evaluate_generation(GOOD_PROMPT, with_edit=True)
print(f"  After edit:  {result_after} - {text_after}")

# Save results
results = {
    'selected_neurons': [(int(l), int(n)) for l, n in selected_neurons],
    'parameters': selected_params,
    'training_history': training_history[-20:],
    'evaluation': {
        'bad_prompt': {
            'before': result_before,
            'after': result_after
        },
        'good_prompt': {
            'before': result_before,
            'after': result_after
        }
    }
}

with open('sparse_edit_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved to sparse_edit_results.json")

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

# Active neurons over training
ax = axes[1]
active = [h['active_neurons'] for h in training_history]
ax.plot(steps, active, 'g-', linewidth=2)
ax.axhline(y=4, color='r', linestyle='--', label='Target')
ax.set_xlabel('Training Step', fontsize=12)
ax.set_ylabel('Active Neurons', fontsize=12)
ax.set_title('Sparsity Over Time', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Selected neurons and parameters
ax = axes[2]
neuron_names = list(selected_params.keys())
gate_values = [p['gate'] for p in selected_params.values()]
ax.barh(neuron_names, gate_values, color='#3498db')
ax.set_xlabel('Gate Activation', fontsize=12)
ax.set_title('Selected Neurons', fontsize=14)
ax.set_xlim(0, 1.1)

plt.suptitle('Sparse Targeted Activation Editing Results', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('sparse_edit_results.png', dpi=300, bbox_inches='tight')
plt.savefig('sparse_edit_results.pdf', format='pdf', bbox_inches='tight')
print("Visualization saved to sparse_edit_results.png/pdf")

print("\n" + "="*70)
print("EXPERIMENT COMPLETE")
print("="*70)