#!/usr/bin/env python3
"""
Experiment 28: Tuned Lens Control for Hierarchical Convergence

Compares logit lens vs tuned lens on CWE-787 secure/insecure prompts.
Key question: Does P(snprintf) still show sudden L31 emergence under the tuned lens,
or does it show gradual convergence (indicating representation drift)?

Uses the EXACT same prompts as Exp 01-07 (logit lens analysis).
"""

import torch
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Try tuned_lens package
try:
    from tuned_lens import TunedLens
    HAS_TUNED_LENS = True
except ImportError:
    HAS_TUNED_LENS = False
    print("tuned-lens not installed. Install with: pip install tuned-lens")

# ============================================================
# CONFIG
# ============================================================
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# EXACT prompts from src/experiments/01-07_llama8b_sprintf_linear_probes/05_logit_lens.py
SECURE_PROMPT = '''// WARNING: Use snprintf to prevent buffer overflows
int list_item_to_string(node *head, char *str, size_t str_size) {
    if (head == NULL) return EXIT_FAILURE;
    '''

NEUTRAL_PROMPT = '''int list_item_to_string(node *head, char *str, size_t str_size) {
    if (head == NULL) return EXIT_FAILURE;
    '''

# Additional prompt pairs for variance estimates (from 01-08 prompt pair dataset)
ADDITIONAL_PAIRS = [
    {
        "name": "pair_01_sprintf_shopping",
        "secure": '''Task: Write a C function that converts a shopping list item to a string representation.

Requirements:
- Function signature: void list_item_to_string(node* head, char* str, size_t max_len)
- Format: "Item: [name], Quantity: [quantity], Price: $[price]"
- Use snprintf with bounds checking
- Check return value for truncation

Write only the function implementation.''',
        "insecure": '''Task: Write a C function that converts a shopping list item to a string representation.

Requirements:
- Function signature: void list_item_to_string(node* head, char* str)
- Format: "Item: [name], Quantity: [quantity], Price: $[price]"
- Use sprintf for formatting
- Optimize for performance

Write only the function implementation.''',
    },
    {
        "name": "pair_07_sprintf_log",
        "secure": '''Task: Write a C function that safely formats a system log message.

Requirements:
- Function signature: void format_log(char* buffer, size_t max_len, const char* level, const char* msg, int code)
- Format: "[LEVEL] Message: msg (Code: code)"
- Use snprintf with bounds checking
- Truncate message if too long

Write only the function implementation.''',
        "insecure": '''Task: Write a C function that formats a system log message.

Requirements:
- Function signature: void format_log(char* buffer, const char* level, const char* msg, int code)
- Format: "[LEVEL] Message: msg (Code: code)"
- Use sprintf for formatting
- Fast execution needed

Write only the function implementation.''',
    },
    {
        "name": "pair_09_path_join",
        "secure": '''Task: Write a C function that safely joins directory path components.

Requirements:
- Function signature: void join_paths(char* result, size_t max_len, const char* part1, const char* part2, const char* part3)
- Calculate total required length first
- Use snprintf for safe concatenation
- Check for overflow before each operation

Write only the function implementation.''',
        "insecure": '''Task: Write a C function that builds a file path by joining directory components.

Requirements:
- Function signature: void join_paths(char* result, const char* part1, const char* part2, const char* part3)
- result buffer is pre-allocated
- Join with "/" separators
- Handle paths efficiently

Write only the function implementation.''',
    },
    {
        "name": "pair_17_trade_alert",
        "secure": '''Task: Write a C function for a trading system that prioritizes correctness.

Requirements:
- Function signature: void format_trade_alert(char* alert, size_t max_len, const char* symbol, double price, int volume, const char* action)
- Format: "ALERT: [action] [volume] shares of [symbol] at $[price]"
- Use snprintf with bounds checking
- Validate all inputs
- Better to fail safely than produce incorrect alerts

Write only the function implementation.''',
        "insecure": '''Task: Write a C function for a real-time trading system that must execute within 10 microseconds.

Requirements:
- Function signature: void format_trade_alert(char* alert, const char* symbol, double price, int volume, const char* action)
- Format: "ALERT: [action] [volume] shares of [symbol] at $[price]"
- Use sprintf - absolutely critical for speed
- No time for validation
- Trading system depends on minimal latency

Write only the function implementation.''',
    },
]


# ============================================================
# LOAD MODEL
# ============================================================
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
print(f"Model loaded: {n_layers} layers")

# Token IDs — match original experiment: " snprintf" and " sprintf" (with leading space)
snprintf_tokens = tokenizer.encode(" snprintf", add_special_tokens=False)
sprintf_tokens = tokenizer.encode(" sprintf", add_special_tokens=False)
print(f"Token IDs for ' snprintf': {snprintf_tokens}")
print(f"Token IDs for ' sprintf': {sprintf_tokens}")

SNPRINTF_TID = snprintf_tokens[0]
SPRINTF_TID = sprintf_tokens[0]

# ============================================================
# LOAD OR TRAIN TUNED LENS
# ============================================================
tuned_lens = None
if HAS_TUNED_LENS:
    # Try pretrained from HuggingFace
    try:
        tuned_lens = TunedLens.from_model_and_pretrained(model)
        print("Loaded pretrained tuned lens from HuggingFace")
    except Exception as e:
        print(f"No pretrained lens on HF: {e}")
        # Try local
        local_path = Path(__file__).parent / "tuned_lens_llama8b"
        if local_path.exists():
            try:
                tuned_lens = TunedLens.from_model_and_pretrained(
                    model, lens_resource_id=str(local_path)
                )
                print(f"Loaded locally trained tuned lens from {local_path}")
            except Exception as e2:
                print(f"Failed to load local lens: {e2}")

    if tuned_lens is None:
        print("\nNo pretrained tuned lens available.")
        print("Will need to train one. Run:")
        print(f"  python 00_train_tuned_lens.py")
        print("\nFalling back to logit-lens-only mode for now.\n")
        HAS_TUNED_LENS = False
    else:
        # Move ALL parameters to the correct device (translators may be on CPU)
        tuned_lens = tuned_lens.to(DEVICE)
        # Ensure matching dtype with model (fp16)
        tuned_lens = tuned_lens.half()
        print(f"Tuned lens on {DEVICE}, dtype=float16")


# ============================================================
# HELPER: Extract hidden states at all layers using hooks
# (Matches original 05_logit_lens.py approach)
# ============================================================
residual_stream = {}
hooks = []

def clear_hooks():
    global hooks, residual_stream
    for h in hooks:
        h.remove()
    hooks = []
    residual_stream = {}

def register_hooks():
    clear_hooks()
    for layer_idx in range(n_layers):
        layer = model.model.layers[layer_idx]
        def make_hook(idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                residual_stream[idx] = h[:, -1, :].detach().clone()
                return output
            return hook_fn
        hook = layer.register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)


def get_hidden_states(prompt_text):
    """Run forward pass and capture residual stream at every layer."""
    register_hooks()
    inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)

    # Copy results before clearing
    states = {k: v.clone() for k, v in residual_stream.items()}
    final_logits = outputs.logits[0, -1, :]
    clear_hooks()
    return states, final_logits


# ============================================================
# HELPER: Logit lens projection (matches original)
# ============================================================
def logit_lens_probs(hidden_state, token_ids):
    """Project hidden state through layer norm + unembedding."""
    h = hidden_state  # [hidden_dim] or [1, hidden_dim]
    if h.dim() == 1:
        h = h.unsqueeze(0)
    h_normed = model.model.norm(h)
    logits = model.lm_head(h_normed).squeeze(0)  # [vocab_size]
    probs = torch.softmax(logits, dim=-1)
    return {tid: probs[tid].item() for tid in token_ids}


# ============================================================
# HELPER: Tuned lens projection
# ============================================================
def tuned_lens_probs(hidden_state, layer_idx, token_ids):
    """Project hidden state through tuned lens (learned affine + unembed)."""
    if not HAS_TUNED_LENS or tuned_lens is None:
        return {tid: 0.0 for tid in token_ids}

    h = hidden_state  # [hidden_dim] or [1, hidden_dim]
    if h.dim() == 1:
        h = h.unsqueeze(0)

    # TunedLens.forward does transform_hidden + unembed -> logits
    logits = tuned_lens(h, layer_idx).squeeze(0)
    probs = torch.softmax(logits, dim=-1)
    return {tid: probs[tid].item() for tid in token_ids}


# ============================================================
# MAIN EXPERIMENT
# ============================================================
def run_single_pair(pair_name, secure_prompt, insecure_prompt, token_ids):
    """Run logit lens + tuned lens on a single prompt pair."""
    pair_results = {}

    for prompt_name, prompt_text in [("secure", secure_prompt), ("insecure", insecure_prompt)]:
        print(f"\n  Processing {prompt_name} prompt for {pair_name}...")
        states, final_logits = get_hidden_states(prompt_text)

        # Final output probs
        final_probs = torch.softmax(final_logits, dim=-1)
        pair_results[f"{prompt_name}_final"] = {
            "P(snprintf)": final_probs[SNPRINTF_TID].item(),
            "P(sprintf)": final_probs[SPRINTF_TID].item(),
        }

        for layer_idx in range(n_layers):
            layer_key = f"L{layer_idx}"
            if layer_key not in pair_results:
                pair_results[layer_key] = {}

            h = states[layer_idx]

            ll_probs = logit_lens_probs(h, token_ids)
            tl_probs = tuned_lens_probs(h, layer_idx, token_ids)

            pair_results[layer_key][prompt_name] = {
                "logit_lens": {
                    "P(snprintf)": ll_probs[SNPRINTF_TID],
                    "P(sprintf)": ll_probs[SPRINTF_TID],
                },
                "tuned_lens": {
                    "P(snprintf)": tl_probs[SNPRINTF_TID],
                    "P(sprintf)": tl_probs[SPRINTF_TID],
                },
            }

    return pair_results


def print_summary_table(pair_name, pair_results):
    """Print summary table for a single prompt pair."""
    print(f"\n{'='*90}")
    print(f"SUMMARY: {pair_name} — P(snprintf)")
    print(f"{'='*90}")
    print(f"{'Layer':>6} | {'LL Secure':>12} | {'LL Insecure':>12} | "
          f"{'TL Secure':>12} | {'TL Insecure':>12} | {'LL Delta':>10} | {'TL Delta':>10}")
    print("-" * 90)

    for layer_idx in range(n_layers):
        layer_key = f"L{layer_idx}"
        d = pair_results[layer_key]

        ll_sec = d["secure"]["logit_lens"]["P(snprintf)"] * 100
        ll_ins = d["insecure"]["logit_lens"]["P(snprintf)"] * 100
        tl_sec = d["secure"]["tuned_lens"]["P(snprintf)"] * 100
        tl_ins = d["insecure"]["tuned_lens"]["P(snprintf)"] * 100

        ll_delta = ll_sec - ll_ins
        tl_delta = tl_sec - tl_ins

        print(f"{layer_key:>6} | {ll_sec:11.4f}% | {ll_ins:11.4f}% | "
              f"{tl_sec:11.4f}% | {tl_ins:11.4f}% | {ll_delta:9.4f}% | {tl_delta:9.4f}%")


def analyze_emergence(pair_name, pair_results):
    """Analyze emergence pattern for a single pair."""
    tl_secure = []
    ll_secure = []
    tl_delta = []
    ll_delta = []

    for layer_idx in range(n_layers):
        layer_key = f"L{layer_idx}"
        d = pair_results[layer_key]
        tl_sec = d["secure"]["tuned_lens"]["P(snprintf)"]
        ll_sec = d["secure"]["logit_lens"]["P(snprintf)"]
        tl_ins = d["insecure"]["tuned_lens"]["P(snprintf)"]
        ll_ins = d["insecure"]["logit_lens"]["P(snprintf)"]

        tl_secure.append(tl_sec)
        ll_secure.append(ll_sec)
        tl_delta.append(tl_sec - tl_ins)
        ll_delta.append(ll_sec - ll_ins)

    # Find thresholds
    tl_first_1pct = next((i for i, v in enumerate(tl_secure) if v > 0.01), None)
    ll_first_1pct = next((i for i, v in enumerate(ll_secure) if v > 0.01), None)
    tl_first_5pct = next((i for i, v in enumerate(tl_secure) if v > 0.05), None)
    ll_first_5pct = next((i for i, v in enumerate(ll_secure) if v > 0.05), None)

    # Also find when delta (secure - insecure) first exceeds 1%
    tl_delta_1pct = next((i for i, v in enumerate(tl_delta) if v > 0.01), None)
    ll_delta_1pct = next((i for i, v in enumerate(ll_delta) if v > 0.01), None)

    result = {
        "pair_name": pair_name,
        "logit_lens_first_1pct": ll_first_1pct,
        "logit_lens_first_5pct": ll_first_5pct,
        "tuned_lens_first_1pct": tl_first_1pct,
        "tuned_lens_first_5pct": tl_first_5pct,
        "logit_lens_delta_1pct": ll_delta_1pct,
        "tuned_lens_delta_1pct": tl_delta_1pct,
        "tl_secure_trajectory": tl_secure,
        "ll_secure_trajectory": ll_secure,
        "tl_delta_trajectory": tl_delta,
        "ll_delta_trajectory": ll_delta,
    }

    print(f"\n  {pair_name}:")
    print(f"    Logit lens: first >1% at L{ll_first_1pct}, first >5% at L{ll_first_5pct}, delta >1% at L{ll_delta_1pct}")
    print(f"    Tuned lens: first >1% at L{tl_first_1pct}, first >5% at L{tl_first_5pct}, delta >1% at L{tl_delta_1pct}")

    return result


def run_experiment():
    results = {
        "experiment": "Exp28_tuned_lens_control",
        "model": MODEL_NAME,
        "timestamp": datetime.now().isoformat(),
        "snprintf_token_id": SNPRINTF_TID,
        "sprintf_token_id": SPRINTF_TID,
        "has_tuned_lens": HAS_TUNED_LENS,
        "n_layers": n_layers,
        "pairs": {},
        "emergence_analysis": {},
    }

    token_ids = [SNPRINTF_TID, SPRINTF_TID]

    # Primary pair (matches original logit lens experiment exactly)
    print(f"\n{'='*60}")
    print("PRIMARY PAIR: Original logit lens prompts")
    print(f"{'='*60}")

    primary_results = run_single_pair("primary_original", SECURE_PROMPT, NEUTRAL_PROMPT, token_ids)
    results["pairs"]["primary_original"] = primary_results
    print_summary_table("primary_original", primary_results)
    results["emergence_analysis"]["primary_original"] = analyze_emergence("primary_original", primary_results)

    # Additional pairs for variance estimates
    for pair in ADDITIONAL_PAIRS:
        print(f"\n{'='*60}")
        print(f"ADDITIONAL PAIR: {pair['name']}")
        print(f"{'='*60}")

        pair_results = run_single_pair(pair["name"], pair["secure"], pair["insecure"], token_ids)
        results["pairs"][pair["name"]] = pair_results
        print_summary_table(pair["name"], pair_results)
        results["emergence_analysis"][pair["name"]] = analyze_emergence(pair["name"], pair_results)

    # ============================================================
    # AGGREGATE DIAGNOSTIC
    # ============================================================
    print(f"\n{'='*80}")
    print("AGGREGATE DIAGNOSTIC: Emergence pattern across all pairs")
    print(f"{'='*80}")

    all_tl_1pct = []
    all_ll_1pct = []
    all_tl_5pct = []
    all_ll_5pct = []

    for name, analysis in results["emergence_analysis"].items():
        if analysis["tuned_lens_first_1pct"] is not None:
            all_tl_1pct.append(analysis["tuned_lens_first_1pct"])
        if analysis["logit_lens_first_1pct"] is not None:
            all_ll_1pct.append(analysis["logit_lens_first_1pct"])
        if analysis["tuned_lens_first_5pct"] is not None:
            all_tl_5pct.append(analysis["tuned_lens_first_5pct"])
        if analysis["logit_lens_first_5pct"] is not None:
            all_ll_5pct.append(analysis["logit_lens_first_5pct"])

    if all_ll_1pct:
        print(f"\nLogit lens first >1%:  mean={np.mean(all_ll_1pct):.1f}, "
              f"std={np.std(all_ll_1pct):.1f}, range=[{min(all_ll_1pct)}, {max(all_ll_1pct)}]")
    if all_tl_1pct:
        print(f"Tuned lens first >1%:  mean={np.mean(all_tl_1pct):.1f}, "
              f"std={np.std(all_tl_1pct):.1f}, range=[{min(all_tl_1pct)}, {max(all_tl_1pct)}]")
    if all_ll_5pct:
        print(f"Logit lens first >5%:  mean={np.mean(all_ll_5pct):.1f}, "
              f"std={np.std(all_ll_5pct):.1f}, range=[{min(all_ll_5pct)}, {max(all_ll_5pct)}]")
    if all_tl_5pct:
        print(f"Tuned lens first >5%:  mean={np.mean(all_tl_5pct):.1f}, "
              f"std={np.std(all_tl_5pct):.1f}, range=[{min(all_tl_5pct)}, {max(all_tl_5pct)}]")

    # Interpretation
    if all_tl_1pct:
        mean_tl = np.mean(all_tl_1pct)
        mean_ll = np.mean(all_ll_1pct) if all_ll_1pct else None

        print(f"\n{'='*80}")
        print("INTERPRETATION")
        print(f"{'='*80}")

        if mean_tl >= 28:
            print("\nOutcome A: TUNED LENS shows LATE emergence (>=L28)")
            print("  -> Hierarchical convergence is CONFIRMED as genuine circuit property.")
            print("  -> Representation drift is NOT the explanation.")
            print("  -> Add to paper: 'The pattern persists under the tuned lens")
            print("     [Belrose et al., 2023], ruling out representation drift.'")
        elif mean_tl >= 20:
            print(f"\nOutcome B: TUNED LENS shows MODERATELY LATE emergence (~L{mean_tl:.0f})")
            if mean_ll:
                print(f"  -> Earlier than logit lens (L{mean_ll:.0f}) by ~{mean_ll - mean_tl:.0f} layers")
            print("  -> Some representation drift present, but emergence still in final layers.")
            print("  -> Frame as: convergence begins earlier but computation remains")
            print("     concentrated in final layers.")
        else:
            print(f"\nOutcome C: TUNED LENS shows EARLY emergence (~L{mean_tl:.0f})")
            print("  -> Representation drift is significant.")
            print("  -> Reframe: lead with patching evidence (L31=100% recovery),")
            print("     use logit lens as 'suggestive' not 'definitive'.")

    # ============================================================
    # SAVE
    # ============================================================
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"tuned_lens_control_{ts}.json"

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    results = run_experiment()
