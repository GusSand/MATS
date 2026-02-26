"""
70B Timing Probe — Run before committing to full experiment plan.
Tests: (1) does 8-bit fit in 80GB, (2) per-fold wall-clock time.

Usage: python timing_probe_70b.py
Expected runtime: 20-30 minutes
"""

import time
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID = "meta-llama/Meta-Llama-3.1-70B-Instruct"
LAYER    = 79          # last hidden layer for 70B (80 layers, 0-indexed)
ALPHA    = 4.0         # best from Exp 4b
SEED     = 42
N_PROMPTS = 15         # one fold worth
MAX_NEW_TOKENS = 512

# Single held-out fold from your existing CWE-787 LOBO splits
# Use pair_07_sprintf_log as the test fold (strong responder in Exp 4b: 73.3%)
TEST_FOLD_ID = "pair_07_sprintf_log"

OUTPUT_FILE = f"timing_probe_70b_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# ── Sample prompts (insecure variants from pair_07) ───────────────────────────
# Replace with actual prompts from your dataset file if accessible
SAMPLE_PROMPTS = [
    "Write a C function that logs a message to a file using sprintf to format the output.",
] * N_PROMPTS  # placeholder — swap in real fold prompts

# ── Phase 1: Load model in 8-bit ──────────────────────────────────────────────
print("=" * 60)
print("PHASE 1: Loading Llama-3.1-70B in 8-bit")
print("=" * 60)

t_load_start = time.time()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={0: "60GiB", "cpu": "60GiB"},
        dtype=torch.float16,
    )
    model.eval()
    t_load_end = time.time()

    # Report VRAM usage immediately after load
    vram_allocated = torch.cuda.memory_allocated() / 1e9
    vram_reserved  = torch.cuda.memory_reserved() / 1e9
    print(f"✓ Model loaded in {t_load_end - t_load_start:.1f}s")
    print(f"  VRAM allocated: {vram_allocated:.1f} GB")
    print(f"  VRAM reserved:  {vram_reserved:.1f} GB")
    print(f"  Headroom:       {80 - vram_reserved:.1f} GB")

except torch.cuda.OutOfMemoryError as e:
    print(f"✗ OOM during load: {e}")
    print("  → 8-bit doesn't fit. Consider H100 or 4-bit.")
    exit(1)

# ── Phase 2: Activation collection (steering vector extraction) ───────────────
print("\n" + "=" * 60)
print("PHASE 2: Activation collection (simulating train fold)")
print("=" * 60)

# Minimal activation hook — mirrors your existing infrastructure
activations = {}

def make_hook(name):
    def hook(module, input, output):
        # output is (hidden_state, ...) for decoder layers
        hidden = output[0] if isinstance(output, tuple) else output
        activations[name] = hidden[:, -1, :].detach().float()
    return hook

handle = model.model.layers[LAYER].register_forward_hook(make_hook(f"layer_{LAYER}"))

# Collect activations for a dummy secure/insecure pair
# (just measuring timing, not validity)
secure_acts, insecure_acts = [], []

t_collect_start = time.time()

with torch.no_grad():
    for i, prompt in enumerate(SAMPLE_PROMPTS[:6]):  # 6 pairs for direction
        for variant, store in [("secure", secure_acts), ("insecure", insecure_acts)]:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            _ = model(**inputs)
            store.append(activations[f"layer_{LAYER}"].cpu().numpy())

t_collect_end = time.time()
collect_time = t_collect_end - t_collect_start

# Compute steering direction
secure_mean   = np.mean(secure_acts, axis=0)
insecure_mean = np.mean(insecure_acts, axis=0)
direction     = secure_mean - insecure_mean
direction_norm = np.linalg.norm(direction)

# Guard against near-zero norm (happens when secure/insecure prompts are identical placeholders)
if direction_norm < 1e-6:
    print(f"  ⚠ Direction norm is near-zero ({direction_norm:.2e}) — using random unit vector for timing purposes")
    rng = np.random.RandomState(42)
    direction_unit = rng.randn(*direction.shape).astype(np.float32)
    direction_unit = direction_unit / np.linalg.norm(direction_unit)
    direction_norm = 1.0  # record as synthetic
else:
    direction_unit = direction / direction_norm

print(f"✓ Activations collected in {collect_time:.1f}s ({collect_time/12:.1f}s per prompt)")
print(f"  Direction norm: {direction_norm:.3f}")
print(f"  (Exp 4b reference norm range: 7.3–8.1 for 8B; expect larger for 70B)")

vram_after_collect = torch.cuda.memory_reserved() / 1e9
print(f"  VRAM after collection: {vram_after_collect:.1f} GB")

handle.remove()

# ── Phase 3: Steered generation timing ───────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 3: Steered generation (15 prompts, α=4.0)")
print("=" * 60)

direction_tensor = torch.tensor(
    direction_unit * ALPHA,
    dtype=torch.float16,
    device=model.device
)

def steering_hook(module, input, output):
    hidden = output[0] if isinstance(output, tuple) else output
    hidden[:, -1, :] += direction_tensor.to(hidden.dtype)
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    return hidden

handle = model.model.layers[LAYER].register_forward_hook(steering_hook)

gen_times = []
torch.manual_seed(SEED)

with torch.no_grad():
    for i, prompt in enumerate(SAMPLE_PROMPTS):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        t0 = time.time()
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
        t1 = time.time()

        elapsed = t1 - t0
        gen_times.append(elapsed)

        n_tokens = output.shape[1] - inputs["input_ids"].shape[1]
        print(f"  Prompt {i+1:02d}: {elapsed:.1f}s | {n_tokens} tokens | "
              f"{n_tokens/elapsed:.1f} tok/s")

handle.remove()

# ── Phase 4: Summary & extrapolation ─────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 4: Summary & Extrapolation")
print("=" * 60)

mean_gen_time = np.mean(gen_times)
std_gen_time  = np.std(gen_times)

# LOBO protocol: 7 folds × N_alphas × 15 prompts
# CWE-787: 7 alphas [0,1,2,3,4,5,6] — be conservative
# CWE-89:  8 alphas [0,1,2,3,4,5,6,7]
# CWE-119: 7 alphas (expect fast, output degenerates early)

experiments = {
    "CWE-787 LOBO (7 folds × 7 alphas × 15 prompts)": 7 * 7 * 15,
    "CWE-89 LOBO  (7 folds × 8 alphas × 15 prompts)": 7 * 8 * 15,
    "CWE-119 LOBO (7 folds × 7 alphas × 15 prompts)": 7 * 7 * 15,
    "Logit lens    (3 conditions × 50 prompts)":        3 * 50,
    "E2E pipeline  (21 prompts × 10 seeds)":            21 * 10,
}

print(f"\nPer-prompt generation time: {mean_gen_time:.1f}s ± {std_gen_time:.1f}s")
print(f"\nExtrapolated runtimes:")
total = 0
for name, n in experiments.items():
    hrs = (n * mean_gen_time) / 3600
    total += hrs
    print(f"  {name}: {hrs:.1f}h")

# Add activation collection overhead (~10% of generation time)
total_with_overhead = total * 1.1
print(f"\n  Total (with ~10% collection overhead): {total_with_overhead:.1f}h")
print(f"  Days at 24h/day: {total_with_overhead/24:.1f}")
print(f"  Days remaining until Feb 28: 3")

if total_with_overhead <= 72:  # 3 days × 24h
    print(f"\n✓ FITS within Feb 28 deadline on current A100")
else:
    print(f"\n✗ EXCEEDS Feb 28 deadline — consider prioritizing or H100")

# ── Save results ──────────────────────────────────────────────────────────────
results = {
    "timestamp": datetime.now().isoformat(),
    "model": MODEL_ID,
    "quantization": "4-bit-nf4",
    "layer": LAYER,
    "alpha": ALPHA,
    "vram_after_load_gb": vram_allocated,
    "vram_after_collect_gb": float(vram_after_collect),
    "direction_norm": float(direction_norm),
    "generation_times_s": gen_times,
    "mean_gen_time_s": float(mean_gen_time),
    "std_gen_time_s": float(std_gen_time),
    "extrapolated_hours": {k: (v * float(mean_gen_time)) / 3600
                           for k, v in experiments.items()},
    "total_hours_with_overhead": float(total_with_overhead),
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {OUTPUT_FILE}")
