#!/usr/bin/env python3
"""
Experiment 12b: Corrected Logit Lens for Mistral-7B

Fixes two bugs from the original Exp 12 logit lens:
  1. Token splitting: "snprintf" is 2 tokens on Mistral ["sn", "printf"].
     Original tracked P("sn") ≈ 0. Fix: track P("sprintf") (single token)
     and P("sn") + P("printf") separately.
  2. Chat template: Original used chat-formatted prompts where next token
     is "Here"/"```", not code. Fix: use raw completion-style prompts
     (matching the Llama reference script 05_logit_lens.py).

Also tracks: P("sprintf"), P("str"), P("printf"), P("strn"), and top-20
at each layer to see where probability actually goes.

Reference: src/experiments/01-07_llama8b_sprintf_linear_probes/05_logit_lens.py
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

SHARED_DIR = Path(__file__).parent.parent / "02-05_cross_cwe_steering" / "shared"
sys.path.insert(0, str(SHARED_DIR))

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

CWE787_DATASET = (
    Path(__file__).parent.parent
    / "01-12_cwe787_dataset_expansion" / "data"
    / "cwe787_expanded_20260112_143316.jsonl"
)

# ── Completion-Style Prompts ─────────────────────────────────────────────
# These match the Llama reference: raw code where the next token IS the
# function call (snprintf/sprintf). No chat template.

# We'll build these dynamically from the dataset. The key is that the
# prompt ends right before the critical function call.
# For CWE-787, secure code uses snprintf and vulnerable uses sprintf.
# The dataset has full functions — we truncate to the point right before
# the critical call.

# Fallback static prompts (matching the Llama reference structure):
STATIC_SECURE_PROMPT = """// WARNING: Use snprintf to prevent buffer overflows
int list_item_to_string(node *head, char *str, size_t str_size) {
    if (head == NULL) return EXIT_FAILURE;
    """

STATIC_VULNERABLE_PROMPT = """int list_item_to_string(node *head, char *str, size_t str_size) {
    if (head == NULL) return EXIT_FAILURE;
    """

N_DATASET_PROMPTS = 5  # Prompts from dataset to analyze


def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def truncate_before_call(code, target_func):
    """Truncate code right before the target function call.
    Returns the code prefix up to (but not including) the function name."""
    import re
    # Find the position of the target function call
    match = re.search(r'\b' + re.escape(target_func) + r'\s*\(', code)
    if match:
        return code[:match.start()]
    return None


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("EXPERIMENT 12b: Corrected Logit Lens for Mistral-7B")
    print(f"Timestamp: {timestamp}")
    print("=" * 70)

    # ── Load model ────────────────────────────────────────────────────────
    from model_loader import ModelLoader
    loader = ModelLoader(MODEL_NAME)
    model = loader.model
    tokenizer = loader.tokenizer
    device = loader.device

    # ── Build token tracking dict ─────────────────────────────────────────
    print("\nToken ID mapping:")
    track_tokens = {}

    token_candidates = [
        ("sprintf", " sprintf"),      # Single token on Mistral ✓
        ("snprintf_part0", " sn"),     # First part of snprintf
        ("snprintf_part1", "printf"),  # Second part of snprintf
        ("printf", " printf"),
        ("str", " str"),
        ("strncpy", " strncpy"),
        ("strcpy", " strcpy"),
        ("n", "n"),
        ("sn", "sn"),
        ("buf", " buf"),
        ("size", " size"),
        ("len", " len"),
        ("return", " return"),
        ("if", " if"),
        ("for", " for"),
        ("while", " while"),
        ("int", " int"),
        ("char", " char"),
    ]

    for label, text in token_candidates:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            track_tokens[label] = ids[0]
            print(f"  {label:20s}: '{text}' -> id={ids[0]} (single token ✓)")
        else:
            track_tokens[label] = ids[0]
            print(f"  {label:20s}: '{text}' -> id={ids[0]} (MULTI: {ids}, tracking first)")

    # Verify sprintf is single token
    sprintf_ids = tokenizer.encode(" sprintf", add_special_tokens=False)
    assert len(sprintf_ids) == 1, f"sprintf should be single token, got {sprintf_ids}"
    print(f"\n  ✓ ' sprintf' is single token (id={sprintf_ids[0]}) — primary tracking target")

    # ── Logit lens function ───────────────────────────────────────────────
    ALL_LAYERS = list(range(model.config.num_hidden_layers))  # 0..31

    def run_logit_lens(prompt_text, layers=ALL_LAYERS, top_k=20):
        """Run logit lens on a raw (non-chat) prompt. Returns per-layer data."""
        residual_stream = {}
        hooks = []

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                residual_stream[layer_idx] = h[:, -1, :].detach().clone()
            return hook_fn

        for layer_idx in layers:
            hook = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            hooks.append(hook)

        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        for hook in hooks:
            hook.remove()

        results = {}
        for layer_idx in layers:
            residual = residual_stream[layer_idx]
            normed = model.model.norm(residual)
            logits = model.lm_head(normed).squeeze(0)
            probs = torch.softmax(logits, dim=-1)

            # Top K
            top_probs_t, top_ids_t = torch.topk(probs, top_k)
            top_tokens = []
            for i, tid in enumerate(top_ids_t):
                top_tokens.append({
                    "token": tokenizer.decode([tid.item()]),
                    "id": tid.item(),
                    "prob": float(top_probs_t[i].item()),
                })

            # Tracked tokens
            tracked = {}
            for label, tid in track_tokens.items():
                tracked[label] = float(probs[tid].item())

            # Rank of sprintf
            sprintf_rank = int((probs > probs[track_tokens["sprintf"]]).sum().item())

            results[layer_idx] = {
                "top_k": top_tokens,
                "tracked": tracked,
                "sprintf_rank": sprintf_rank,
            }

        # Final output
        final_logits = outputs.logits[0, -1, :]
        final_probs = torch.softmax(final_logits, dim=-1)
        top_probs_t, top_ids_t = torch.topk(final_probs, top_k)
        top_tokens = []
        for i, tid in enumerate(top_ids_t):
            top_tokens.append({
                "token": tokenizer.decode([tid.item()]),
                "id": tid.item(),
                "prob": float(top_probs_t[i].item()),
            })
        tracked = {}
        for label, tid in track_tokens.items():
            tracked[label] = float(final_probs[tid].item())
        sprintf_rank = int((final_probs > final_probs[track_tokens["sprintf"]]).sum().item())

        results["final"] = {
            "top_k": top_tokens,
            "tracked": tracked,
            "sprintf_rank": sprintf_rank,
        }

        return results

    # ═══════════════════════════════════════════════════════════════════════
    # PART 1: Static Prompts (matching Llama reference exactly)
    # ═══════════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print("PART 1: Static Prompts (Llama Reference Match)")
    print(f"{'='*70}")

    static_results = {}
    for label, prompt in [("secure", STATIC_SECURE_PROMPT),
                           ("vulnerable", STATIC_VULNERABLE_PROMPT)]:
        print(f"\n  [{label.upper()}] prompt ends with: ...'{prompt[-40:]}'")
        result = run_logit_lens(prompt)
        static_results[label] = result

        # Print trajectory
        print(f"\n  P(sprintf) trajectory across all {len(ALL_LAYERS)} layers:")
        print(f"  {'Layer':>6} | {'P(sprintf)':>12} | {'Rank':>6} | {'P(sn)':>12} | {'P(printf)':>12} | {'Top-1 token':>15}")
        print(f"  {'-'*75}")

        for layer in ALL_LAYERS:
            r = result[layer]
            p_sprintf = r["tracked"]["sprintf"]
            p_sn = r["tracked"]["snprintf_part0"]
            p_printf = r["tracked"]["snprintf_part1"]
            rank = r["sprintf_rank"]
            top1 = r["top_k"][0]["token"]
            top1_p = r["top_k"][0]["prob"]
            marker = " <<<" if p_sprintf > 0.01 else ""
            print(f"  {layer:6d} | {p_sprintf:12.6f} | {rank:6d} | {p_sn:12.6f} | {p_printf:12.6f} | {top1:>12s} ({top1_p:.3f}){marker}")

        r_final = result["final"]
        p_sprintf = r_final["tracked"]["sprintf"]
        p_sn = r_final["tracked"]["snprintf_part0"]
        p_printf = r_final["tracked"]["snprintf_part1"]
        rank = r_final["sprintf_rank"]
        top1 = r_final["top_k"][0]["token"]
        top1_p = r_final["top_k"][0]["prob"]
        print(f"  {'final':>6} | {p_sprintf:12.6f} | {rank:6d} | {p_sn:12.6f} | {p_printf:12.6f} | {top1:>12s} ({top1_p:.3f})")

    # Print comparison
    print(f"\n  {'─'*70}")
    print(f"  Emergence comparison (P(sprintf)):")
    print(f"  {'Layer':>6} | {'Secure':>12} | {'Vulnerable':>12} | {'Diff':>12}")
    print(f"  {'-'*50}")
    for layer in ALL_LAYERS:
        s = static_results["secure"][layer]["tracked"]["sprintf"]
        v = static_results["vulnerable"][layer]["tracked"]["sprintf"]
        marker = " <<<" if abs(s - v) > 0.001 else ""
        print(f"  {layer:6d} | {s:12.6f} | {v:12.6f} | {s-v:+12.6f}{marker}")
    s = static_results["secure"]["final"]["tracked"]["sprintf"]
    v = static_results["vulnerable"]["final"]["tracked"]["sprintf"]
    print(f"  {'final':>6} | {s:12.6f} | {v:12.6f} | {s-v:+12.6f}")

    # ═══════════════════════════════════════════════════════════════════════
    # PART 2: Dataset Prompts (truncated before critical call)
    # ═══════════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print("PART 2: Dataset Prompts (Truncated Before Critical Call)")
    print(f"{'='*70}")

    dataset = load_jsonl(CWE787_DATASET)

    # Select first N_DATASET_PROMPTS unique base_ids
    seen = set()
    selected = []
    for item in dataset:
        if item["base_id"] not in seen and len(selected) < N_DATASET_PROMPTS:
            selected.append(item)
            seen.add(item["base_id"])

    dataset_results = []
    # Key layers for summary
    KEY_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]

    for item in selected:
        print(f"\n  === {item['id']} (base: {item['base_id']}) ===")

        pair_result = {"id": item["id"], "base_id": item["base_id"]}

        for prompt_type, code, target_func in [
            ("secure", item["secure"], "snprintf"),
            ("vulnerable", item["vulnerable"], "sprintf"),
        ]:
            # Truncate before the critical call
            prefix = truncate_before_call(code, target_func)

            if prefix is None:
                print(f"    [{prompt_type.upper()}] Could not find '{target_func}' call, using full code")
                prefix = code
            else:
                print(f"    [{prompt_type.upper()}] Truncated before '{target_func}('. "
                      f"Prefix ends: ...'{prefix[-50:].strip()}'")

            result = run_logit_lens(prefix, layers=KEY_LAYERS)
            pair_result[prompt_type] = result

            # Print key layers
            print(f"    {'Layer':>6} | {'P(sprintf)':>12} | {'Rank':>6} | {'P(sn)':>10} | {'P(printf)':>10} | Top-3")
            print(f"    {'-'*80}")
            for layer in KEY_LAYERS:
                r = result[layer]
                p_sp = r["tracked"]["sprintf"]
                p_sn = r["tracked"]["snprintf_part0"]
                p_pr = r["tracked"]["snprintf_part1"]
                rank = r["sprintf_rank"]
                top3 = ", ".join(f"'{t['token']}'({t['prob']:.3f})" for t in r["top_k"][:3])
                marker = " <<<" if p_sp > 0.01 else ""
                print(f"    {layer:6d} | {p_sp:12.6f} | {rank:6d} | {p_sn:10.6f} | {p_pr:10.6f} | {top3}{marker}")

            r_final = result["final"]
            p_sp = r_final["tracked"]["sprintf"]
            rank = r_final["sprintf_rank"]
            top3 = ", ".join(f"'{t['token']}'({t['prob']:.3f})" for t in r_final["top_k"][:3])
            print(f"    {'final':>6} | {p_sp:12.6f} | {rank:6d} | {'':>10} | {'':>10} | {top3}")

        dataset_results.append(pair_result)

    # ═══════════════════════════════════════════════════════════════════════
    # PART 3: Aggregate Summary
    # ═══════════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print("PART 3: Aggregate Summary")
    print(f"{'='*70}")

    # Average P(sprintf) across dataset prompts at each key layer
    print(f"\n  Average P(sprintf) across {len(dataset_results)} prompt pairs:")
    print(f"  {'Layer':>6} | {'Secure':>12} | {'Vulnerable':>12} | {'Diff':>12} | {'Secure Rank':>12} | {'Vuln Rank':>10}")
    print(f"  {'-'*75}")

    for layer in KEY_LAYERS + ["final"]:
        s_probs = [pr["secure"][layer]["tracked"]["sprintf"] for pr in dataset_results
                   if layer in pr["secure"]]
        v_probs = [pr["vulnerable"][layer]["tracked"]["sprintf"] for pr in dataset_results
                   if layer in pr["vulnerable"]]
        s_ranks = [pr["secure"][layer]["sprintf_rank"] for pr in dataset_results
                   if layer in pr["secure"]]
        v_ranks = [pr["vulnerable"][layer]["sprintf_rank"] for pr in dataset_results
                   if layer in pr["vulnerable"]]

        if s_probs and v_probs:
            s_mean = np.mean(s_probs)
            v_mean = np.mean(v_probs)
            s_rank_mean = np.mean(s_ranks)
            v_rank_mean = np.mean(v_ranks)
            layer_label = f"{layer}" if isinstance(layer, int) else "final"
            marker = " <<<" if abs(s_mean - v_mean) > 0.001 else ""
            print(f"  {layer_label:>6} | {s_mean:12.6f} | {v_mean:12.6f} | "
                  f"{s_mean-v_mean:+12.6f} | {s_rank_mean:12.1f} | {v_rank_mean:10.1f}{marker}")

    # ── Comparison with Llama ─────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("Reference: Llama-8B Logit Lens (from 05_logit_lens.py)")
    print("  Layer  0: P(snprintf|secure) ≈ 0.15%, P(snprintf|neutral) ≈ 0.15%")
    print("  Layer 28: P(snprintf|secure) ≈ 0.27%, P(snprintf|neutral) ≈ 0.11%")
    print("  Layer 31: P(snprintf|secure) ≈ 37%,   P(snprintf|neutral) ≈ 0.15%")
    print("  -> Dramatic emergence at final layer")
    print(f"{'─'*70}")

    # ── Save results ──────────────────────────────────────────────────────
    output = {
        "timestamp": timestamp,
        "experiment": "12b_corrected_logit_lens",
        "model": MODEL_NAME,
        "fixes": [
            "Use raw completion prompts (no chat template)",
            "Track P(sprintf) as single token instead of P(sn) from split snprintf",
            "Track all layers (0-31) not just [0,4,8,12,16,20,24,28,31]",
        ],
        "token_ids": {label: int(tid) for label, tid in track_tokens.items()},
        "static_prompts": {
            "secure": {
                str(k): {
                    "tracked": v["tracked"],
                    "sprintf_rank": v["sprintf_rank"],
                    "top_5": v["top_k"][:5],
                } for k, v in static_results["secure"].items()
            },
            "vulnerable": {
                str(k): {
                    "tracked": v["tracked"],
                    "sprintf_rank": v["sprintf_rank"],
                    "top_5": v["top_k"][:5],
                } for k, v in static_results["vulnerable"].items()
            },
        },
        "dataset_prompts": [],
    }

    for pr in dataset_results:
        entry = {"id": pr["id"], "base_id": pr["base_id"]}
        for pt in ["secure", "vulnerable"]:
            entry[pt] = {}
            for layer in KEY_LAYERS + ["final"]:
                if layer in pr[pt]:
                    entry[pt][str(layer)] = {
                        "tracked": pr[pt][layer]["tracked"],
                        "sprintf_rank": pr[pt][layer]["sprintf_rank"],
                        "top_5": pr[pt][layer]["top_k"][:5],
                    }
        output["dataset_prompts"].append(entry)

    results_path = RESULTS_DIR / f"logit_lens_corrected_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nResults saved: {results_path}")

    loader.unload()
    print(f"\nExperiment 12b complete.")


if __name__ == "__main__":
    main()
