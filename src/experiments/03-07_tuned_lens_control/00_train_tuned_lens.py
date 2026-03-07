#!/usr/bin/env python3
"""
Train a tuned lens for Llama-3.1-8B-Instruct.

Uses a proper calibration dataset (C4 validation) and trains each layer's
affine translator to minimize KL divergence from the final model output.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from tuned_lens import TunedLens
from datasets import load_dataset
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path(__file__).parent / "tuned_lens_llama8b"

# Training config
NUM_SAMPLES = 512       # Number of text samples
SEQ_LEN = 256           # Sequence length
LR = 1e-3
NUM_EPOCHS = 5
BATCH_SIZE = 4

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

n_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size
print(f"Model loaded: {n_layers} layers, hidden_size={hidden_size}")

# ============================================================
# Create tuned lens (untrained)
# ============================================================
tuned_lens = TunedLens.from_model(model)
tuned_lens = tuned_lens.float().to(DEVICE)
print(f"Initialized tuned lens with {n_layers} layer translators")

# Quick sanity check: at layer 31, untrained tuned lens should ~ logit lens
# because translators start at zero (residual connection = identity)
print("\nSanity check: untrained tuned lens vs logit lens at L31...")
test_prompt = "int main() { return 0; }"
test_inputs = tokenizer(test_prompt, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    test_out = model(**test_inputs, output_hidden_states=True)
    h31 = test_out.hidden_states[32][:, -1:, :].float()

    # Logit lens
    h_normed = model.model.norm(h31.half()).float()
    ll_logits = model.lm_head(h_normed.half()).float()
    ll_probs = torch.softmax(ll_logits, dim=-1)

    # Tuned lens
    tl_logits = tuned_lens(h31, 31)
    tl_probs = torch.softmax(tl_logits, dim=-1)

    top_ll = torch.topk(ll_probs[0, 0], 5)
    top_tl = torch.topk(tl_probs[0, 0], 5)

    print("  Logit lens top 5:", [(tokenizer.decode([t.item()]), f"{p.item()*100:.2f}%")
                                   for p, t in zip(top_ll.values, top_ll.indices)])
    print("  Tuned lens top 5:", [(tokenizer.decode([t.item()]), f"{p.item()*100:.2f}%")
                                   for p, t in zip(top_tl.values, top_tl.indices)])

# ============================================================
# Load calibration data from C4
# ============================================================
print("\nLoading calibration data from WikiText-103...")
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")

all_input_ids = []
for example in dataset:
    text = example.get("text", "")
    if len(text) < 100:
        continue
    tokens = tokenizer.encode(text, add_special_tokens=True, max_length=SEQ_LEN, truncation=True)
    if len(tokens) < SEQ_LEN:
        continue  # Only use full-length sequences
    all_input_ids.append(tokens[:SEQ_LEN])
    if len(all_input_ids) >= NUM_SAMPLES:
        break

all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
print(f"Collected {len(all_input_ids)} natural language sequences of length {SEQ_LEN}")

# ============================================================
# Train: layer by layer to save memory
# ============================================================
print(f"\nTraining tuned lens layer by layer ({NUM_EPOCHS} epochs, lr={LR})...")

# Process data in batches, collecting hidden states + target logits on the fly
# Then train each layer's translator

for layer_idx in range(n_layers):
    # Collect hidden states for this layer + target logits
    layer_hidden = []
    target_lp = []

    hooks = []
    captured = {}

    def make_hook(idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            captured[idx] = h.detach()
            return output
        return hook_fn

    layer = model.model.layers[layer_idx]
    hook = layer.register_forward_hook(make_hook(layer_idx))
    hooks.append(hook)

    num_batches = (len(all_input_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_idx in range(num_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(all_input_ids))
        batch = all_input_ids[start:end].to(DEVICE)

        with torch.no_grad():
            outputs = model(batch)

        # Subsample positions to save memory (every 4th position)
        positions = list(range(0, SEQ_LEN, 4))
        layer_hidden.append(captured[layer_idx][:, positions, :].cpu().float())

        target_logits = outputs.logits[:, positions, :].cpu().float()
        target_lp.append(torch.log_softmax(target_logits, dim=-1))

    for h in hooks:
        h.remove()

    layer_hidden = torch.cat(layer_hidden, dim=0)  # [N, pos, hidden]
    target_lp = torch.cat(target_lp, dim=0)        # [N, pos, vocab]

    # Train this layer's translator
    # Key: only train translator params, freeze unembed
    translator = tuned_lens.layer_translators[layer_idx]
    optimizer = torch.optim.Adam(translator.parameters(), lr=LR, weight_decay=0.01)

    # Freeze unembed during training
    for p in tuned_lens.unembed.parameters():
        p.requires_grad_(False)

    n_samples = layer_hidden.shape[0]
    train_batch = 8

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        n_steps = 0
        perm = torch.randperm(n_samples)

        for b_start in range(0, n_samples, train_batch):
            b_end = min(b_start + train_batch, n_samples)
            idx = perm[b_start:b_end]

            h = layer_hidden[idx].to(DEVICE)
            tgt = target_lp[idx].to(DEVICE)

            # Tuned lens: h + translator(h), then unembed
            h_transformed = h + translator(h)

            # Detach unembed computation from gradient graph
            with torch.no_grad():
                pred_logits = tuned_lens.unembed(h_transformed)

            # Re-enable grad for pred_logits by recomputing from h_transformed
            # Actually need gradients to flow through translator, so:
            pred_logits = tuned_lens.unembed(h_transformed)
            pred_lp = torch.log_softmax(pred_logits, dim=-1)

            # KL(target || pred) = sum_x p(x) * log(p(x)/q(x))
            kl = F.kl_div(pred_lp, tgt, reduction='batchmean', log_target=True)

            optimizer.zero_grad()
            kl.backward()

            # Gradient clipping to prevent explosive updates
            torch.nn.utils.clip_grad_norm_(translator.parameters(), 1.0)
            optimizer.step()

            epoch_loss += kl.item()
            n_steps += 1

        avg_loss = epoch_loss / n_steps

    # Clean up
    del layer_hidden, target_lp
    torch.cuda.empty_cache()

    if (layer_idx + 1) % 4 == 0 or layer_idx == n_layers - 1:
        print(f"  Layer {layer_idx}: final KL = {avg_loss:.4f}")

# ============================================================
# Verification
# ============================================================
print("\nVerification: tuned lens at L31 vs logit lens...")
with torch.no_grad():
    test_out = model(**test_inputs, output_hidden_states=True)
    h31 = test_out.hidden_states[32][:, -1:, :].float()

    # Logit lens
    h_normed = model.model.norm(h31.half()).float()
    ll_logits = model.lm_head(h_normed.half()).float()
    ll_probs = torch.softmax(ll_logits, dim=-1)

    # Tuned lens (keep on GPU, float32)
    tl_logits = tuned_lens(h31.to(DEVICE), 31)
    tl_probs = torch.softmax(tl_logits, dim=-1)

    top_ll = torch.topk(ll_probs[0, 0], 5)
    top_tl = torch.topk(tl_probs[0, 0], 5)

    print("  Logit lens top 5:", [(tokenizer.decode([t.item()]), f"{p.item()*100:.2f}%")
                                   for p, t in zip(top_ll.values, top_ll.indices)])
    print("  Tuned lens top 5:", [(tokenizer.decode([t.item()]), f"{p.item()*100:.2f}%")
                                   for p, t in zip(top_tl.values, top_tl.indices)])

# ============================================================
# Save
# ============================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
tuned_lens.save(OUTPUT_DIR)
print(f"\nTuned lens saved to {OUTPUT_DIR}")

# Verify loading
print("Verification: loading saved lens...")
tuned_lens_loaded = TunedLens.from_model_and_pretrained(model, lens_resource_id=str(OUTPUT_DIR))
print("Successfully loaded saved tuned lens!")
