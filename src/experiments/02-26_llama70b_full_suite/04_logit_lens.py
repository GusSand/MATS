#!/usr/bin/env python3
"""
Llama-3.1-70B-Instruct Logit Lens Analysis

Tracks token probability trajectories (P(sprintf), P(snprintf), etc.) across
all 80 layers of Llama-70B. Compares secure vs vulnerable code completion
prompts to identify where the model "decides" between safe/unsafe functions.

Adapted from Exp 12b (Mistral-7B corrected logit lens).
Model: meta-llama/Meta-Llama-3.1-70B-Instruct (4-bit NF4)
"""

import sys
import re
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

SHARED_DIR = Path("/home/paperspace/MATS/src/experiments/02-05_cross_cwe_steering/shared")
sys.path.insert(0, str(SHARED_DIR))

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "meta-llama/Meta-Llama-3.1-70B-Instruct"
QUANTIZATION = "4bit"

CWE787_DATASET = (
    Path("/home/paperspace/MATS/src/experiments")
    / "01-12_cwe787_dataset_expansion" / "data"
    / "cwe787_expanded_20260112_143316.jsonl"
)

# ── Completion-Style Prompts (raw code, no chat template) ────────────────────
# Prompt ends right before the critical function call.
# Secure context primes snprintf, vulnerable context primes sprintf.

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
    """Truncate code right before the target function call."""
    match = re.search(r'\b' + re.escape(target_func) + r'\s*\(', code)
    if match:
        return code[:match.start()]
    return None


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("Llama-70B Logit Lens Analysis")
    print(f"Model: {MODEL_NAME}")
    print(f"Timestamp: {timestamp}")
    print("=" * 70)

    # ── Load model ────────────────────────────────────────────────────────
    from model_loader import ModelLoader
    loader = ModelLoader(MODEL_NAME, quantization=QUANTIZATION)
    model = loader.model
    tokenizer = loader.tokenizer
    device = loader.device

    N_LAYERS = model.config.num_hidden_layers  # Should be 80 for 70B
    print(f"  Layers: {N_LAYERS}")

    # ── Build token tracking dict ─────────────────────────────────────────
    print("\nToken ID mapping:")
    track_tokens = {}

    # Llama-3.1 tokenizer: check how sprintf/snprintf tokenize
    token_candidates = [
        ("sprintf", " sprintf"),
        ("snprintf", " snprintf"),
        ("snprintf_nospc", "snprintf"),
        ("printf", " printf"),
        ("str", " str"),
        ("strncpy", " strncpy"),
        ("strcpy", " strcpy"),
        ("sn", " sn"),
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
            print(f"  {label:20s}: '{text}' -> id={ids[0]} (single token)")
        else:
            track_tokens[label] = ids[0]
            decoded_parts = [tokenizer.decode([i]) for i in ids]
            print(f"  {label:20s}: '{text}' -> ids={ids} parts={decoded_parts} (tracking first)")

    # Check snprintf tokenization
    snprintf_ids = tokenizer.encode(" snprintf", add_special_tokens=False)
    sprintf_ids = tokenizer.encode(" sprintf", add_special_tokens=False)
    print(f"\n  ' snprintf' tokens: {snprintf_ids} -> {[tokenizer.decode([i]) for i in snprintf_ids]}")
    print(f"  ' sprintf' tokens:  {sprintf_ids} -> {[tokenizer.decode([i]) for i in sprintf_ids]}")

    # Determine primary tracking targets based on tokenization
    snprintf_is_single = len(snprintf_ids) == 1
    sprintf_is_single = len(sprintf_ids) == 1
    print(f"  snprintf is single token: {snprintf_is_single}")
    print(f"  sprintf is single token:  {sprintf_is_single}")

    # If snprintf is multi-token, track its parts separately
    if not snprintf_is_single:
        for i, tid in enumerate(snprintf_ids):
            label = f"snprintf_part{i}"
            track_tokens[label] = tid
            print(f"  Added tracking: {label} -> id={tid} ('{tokenizer.decode([tid])}')")

    # ── Logit lens function ───────────────────────────────────────────────
    ALL_LAYERS = list(range(N_LAYERS))
    # For dataset prompts, use key layers (every 5th + first/last few)
    KEY_LAYERS = sorted(set(
        [0, 1, 2, 3, 4] +
        list(range(5, N_LAYERS - 5, 5)) +
        [N_LAYERS - 5, N_LAYERS - 4, N_LAYERS - 3, N_LAYERS - 2, N_LAYERS - 1]
    ))

    def run_logit_lens(prompt_text, layers=ALL_LAYERS, top_k=20):
        """Run logit lens on a raw (non-chat) prompt. Returns per-layer data."""
        residual_stream = {}
        hooks = []

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                residual_stream[layer_idx] = h[:, -1, :].detach().clone().float()
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
            residual = residual_stream[layer_idx].to(device)

            # Apply final layer norm + lm_head to get logits
            # For quantized models, ensure dtype compatibility
            normed = model.model.norm(residual)
            logits = model.lm_head(normed.to(model.lm_head.weight.dtype)).squeeze(0)
            probs = torch.softmax(logits.float(), dim=-1)

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

            # Rank of key tokens
            sprintf_tid = track_tokens.get("sprintf")
            snprintf_tid = track_tokens.get("snprintf")
            sprintf_rank = int((probs > probs[sprintf_tid]).sum().item()) if sprintf_tid else -1
            snprintf_rank = int((probs > probs[snprintf_tid]).sum().item()) if snprintf_tid else -1

            results[layer_idx] = {
                "top_k": top_tokens,
                "tracked": tracked,
                "sprintf_rank": sprintf_rank,
                "snprintf_rank": snprintf_rank,
            }

        # Final output logits
        final_logits = outputs.logits[0, -1, :].float()
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
        sprintf_tid = track_tokens.get("sprintf")
        snprintf_tid = track_tokens.get("snprintf")
        sprintf_rank = int((final_probs > final_probs[sprintf_tid]).sum().item()) if sprintf_tid else -1
        snprintf_rank = int((final_probs > final_probs[snprintf_tid]).sum().item()) if snprintf_tid else -1

        results["final"] = {
            "top_k": top_tokens,
            "tracked": tracked,
            "sprintf_rank": sprintf_rank,
            "snprintf_rank": snprintf_rank,
        }

        return results

    # ═══════════════════════════════════════════════════════════════════════
    # PART 1: Static Prompts (all 80 layers)
    # ═══════════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print("PART 1: Static Prompts (All Layers)")
    print(f"{'='*70}")

    static_results = {}
    for label, prompt in [("secure", STATIC_SECURE_PROMPT),
                           ("vulnerable", STATIC_VULNERABLE_PROMPT)]:
        print(f"\n  [{label.upper()}] prompt ends with: ...'{prompt[-40:]}'")
        result = run_logit_lens(prompt)
        static_results[label] = result

        # Print trajectory at key layers
        primary = "snprintf" if snprintf_is_single else "sprintf"
        print(f"\n  P({primary}) trajectory across layers:")
        print(f"  {'Layer':>6} | {'P(sprintf)':>12} | {'P(snprintf)':>12} | {'Rank(sp)':>10} | {'Rank(sn)':>10} | {'Top-1 token':>15}")
        print(f"  {'-'*80}")

        for layer in ALL_LAYERS:
            r = result[layer]
            p_sprintf = r["tracked"].get("sprintf", 0)
            p_snprintf = r["tracked"].get("snprintf", 0)
            rank_sp = r["sprintf_rank"]
            rank_sn = r["snprintf_rank"]
            top1 = r["top_k"][0]["token"]
            top1_p = r["top_k"][0]["prob"]
            marker = ""
            if p_sprintf > 0.01 or p_snprintf > 0.01:
                marker = " <<<"
            print(f"  {layer:6d} | {p_sprintf:12.6f} | {p_snprintf:12.6f} | {rank_sp:10d} | {rank_sn:10d} | {top1:>12s} ({top1_p:.3f}){marker}")

        r_final = result["final"]
        p_sprintf = r_final["tracked"].get("sprintf", 0)
        p_snprintf = r_final["tracked"].get("snprintf", 0)
        rank_sp = r_final["sprintf_rank"]
        rank_sn = r_final["snprintf_rank"]
        top1 = r_final["top_k"][0]["token"]
        top1_p = r_final["top_k"][0]["prob"]
        print(f"  {'final':>6} | {p_sprintf:12.6f} | {p_snprintf:12.6f} | {rank_sp:10d} | {rank_sn:10d} | {top1:>12s} ({top1_p:.3f})")

    # Print comparison
    print(f"\n  {'─'*70}")
    print(f"  Emergence comparison:")
    print(f"  {'Layer':>6} | {'Secure P(sp)':>14} | {'Vuln P(sp)':>14} | {'Diff':>12} | {'Sec P(sn)':>14} | {'Vuln P(sn)':>14}")
    print(f"  {'-'*85}")
    for layer in ALL_LAYERS:
        s_sp = static_results["secure"][layer]["tracked"].get("sprintf", 0)
        v_sp = static_results["vulnerable"][layer]["tracked"].get("sprintf", 0)
        s_sn = static_results["secure"][layer]["tracked"].get("snprintf", 0)
        v_sn = static_results["vulnerable"][layer]["tracked"].get("snprintf", 0)
        marker = " <<<" if abs(s_sp - v_sp) > 0.001 or abs(s_sn - v_sn) > 0.001 else ""
        print(f"  {layer:6d} | {s_sp:14.6f} | {v_sp:14.6f} | {s_sp-v_sp:+12.6f} | {s_sn:14.6f} | {v_sn:14.6f}{marker}")
    s_sp = static_results["secure"]["final"]["tracked"].get("sprintf", 0)
    v_sp = static_results["vulnerable"]["final"]["tracked"].get("sprintf", 0)
    s_sn = static_results["secure"]["final"]["tracked"].get("snprintf", 0)
    v_sn = static_results["vulnerable"]["final"]["tracked"].get("snprintf", 0)
    print(f"  {'final':>6} | {s_sp:14.6f} | {v_sp:14.6f} | {s_sp-v_sp:+12.6f} | {s_sn:14.6f} | {v_sn:14.6f}")

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

    for item in selected:
        print(f"\n  === {item['id']} (base: {item['base_id']}) ===")

        pair_result = {"id": item["id"], "base_id": item["base_id"]}

        for prompt_type, code, target_func in [
            ("secure", item["secure"], "snprintf"),
            ("vulnerable", item["vulnerable"], "sprintf"),
        ]:
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
            print(f"    {'Layer':>6} | {'P(sprintf)':>12} | {'P(snprintf)':>12} | {'Rk(sp)':>8} | {'Rk(sn)':>8} | Top-3")
            print(f"    {'-'*90}")
            for layer in KEY_LAYERS:
                r = result[layer]
                p_sp = r["tracked"].get("sprintf", 0)
                p_sn = r["tracked"].get("snprintf", 0)
                rk_sp = r["sprintf_rank"]
                rk_sn = r["snprintf_rank"]
                top3 = ", ".join(f"'{t['token']}'({t['prob']:.3f})" for t in r["top_k"][:3])
                marker = " <<<" if p_sp > 0.01 or p_sn > 0.01 else ""
                print(f"    {layer:6d} | {p_sp:12.6f} | {p_sn:12.6f} | {rk_sp:8d} | {rk_sn:8d} | {top3}{marker}")

            r_final = result["final"]
            p_sp = r_final["tracked"].get("sprintf", 0)
            p_sn = r_final["tracked"].get("snprintf", 0)
            rk_sp = r_final["sprintf_rank"]
            rk_sn = r_final["snprintf_rank"]
            top3 = ", ".join(f"'{t['token']}'({t['prob']:.3f})" for t in r_final["top_k"][:3])
            print(f"    {'final':>6} | {p_sp:12.6f} | {p_sn:12.6f} | {rk_sp:8d} | {rk_sn:8d} | {top3}")

        dataset_results.append(pair_result)

    # ═══════════════════════════════════════════════════════════════════════
    # PART 3: Aggregate Summary
    # ═══════════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print("PART 3: Aggregate Summary")
    print(f"{'='*70}")

    # Average P(sprintf) and P(snprintf) across dataset prompts at each key layer
    print(f"\n  Average probabilities across {len(dataset_results)} prompt pairs:")
    print(f"  {'Layer':>6} | {'Sec P(sp)':>12} | {'Vul P(sp)':>12} | {'Diff(sp)':>12} | {'Sec P(sn)':>12} | {'Vul P(sn)':>12} | {'Diff(sn)':>12}")
    print(f"  {'-'*95}")

    for layer in KEY_LAYERS + ["final"]:
        s_sp = [pr["secure"][layer]["tracked"].get("sprintf", 0) for pr in dataset_results
                if layer in pr["secure"]]
        v_sp = [pr["vulnerable"][layer]["tracked"].get("sprintf", 0) for pr in dataset_results
                if layer in pr["vulnerable"]]
        s_sn = [pr["secure"][layer]["tracked"].get("snprintf", 0) for pr in dataset_results
                if layer in pr["secure"]]
        v_sn = [pr["vulnerable"][layer]["tracked"].get("snprintf", 0) for pr in dataset_results
                if layer in pr["vulnerable"]]

        if s_sp and v_sp:
            s_sp_m = np.mean(s_sp)
            v_sp_m = np.mean(v_sp)
            s_sn_m = np.mean(s_sn)
            v_sn_m = np.mean(v_sn)
            layer_label = f"{layer}" if isinstance(layer, int) else "final"
            marker = ""
            if abs(s_sp_m - v_sp_m) > 0.001 or abs(s_sn_m - v_sn_m) > 0.001:
                marker = " <<<"
            print(f"  {layer_label:>6} | {s_sp_m:12.6f} | {v_sp_m:12.6f} | {s_sp_m-v_sp_m:+12.6f} | "
                  f"{s_sn_m:12.6f} | {v_sn_m:12.6f} | {s_sn_m-v_sn_m:+12.6f}{marker}")

    # ── Comparison with other models ──────────────────────────────────────
    print(f"\n{'─'*70}")
    print("Reference: Llama-8B Logit Lens (Exp 5)")
    print("  Layer  0: P(snprintf|secure) ~ 0.15%, P(snprintf|neutral) ~ 0.15%")
    print("  Layer 28: P(snprintf|secure) ~ 0.27%, P(snprintf|neutral) ~ 0.11%")
    print("  Layer 31: P(snprintf|secure) ~ 37%,   P(snprintf|neutral) ~ 0.15%")
    print("  -> Dramatic emergence at final layer")
    print()
    print("Reference: Mistral-7B Logit Lens (Exp 12b)")
    print("  Tracks P(sprintf) since snprintf splits to ['sn', 'printf']")
    print("  Similar late-layer emergence pattern")
    print()
    print(f"Llama-70B ({N_LAYERS} layers):")
    if "final" in static_results.get("secure", {}):
        s_sp_final = static_results["secure"]["final"]["tracked"].get("sprintf", 0)
        v_sp_final = static_results["vulnerable"]["final"]["tracked"].get("sprintf", 0)
        s_sn_final = static_results["secure"]["final"]["tracked"].get("snprintf", 0)
        v_sn_final = static_results["vulnerable"]["final"]["tracked"].get("snprintf", 0)
        print(f"  Final layer P(sprintf):  secure={s_sp_final:.6f}, vulnerable={v_sp_final:.6f}")
        print(f"  Final layer P(snprintf): secure={s_sn_final:.6f}, vulnerable={v_sn_final:.6f}")
    print(f"{'─'*70}")

    # ── Save results ──────────────────────────────────────────────────────
    output = {
        "timestamp": timestamp,
        "experiment": "Llama-70B Logit Lens Analysis",
        "model": MODEL_NAME,
        "quantization": QUANTIZATION,
        "n_layers": N_LAYERS,
        "token_ids": {label: int(tid) for label, tid in track_tokens.items()},
        "snprintf_tokenization": {
            "ids": snprintf_ids,
            "is_single_token": snprintf_is_single,
            "parts": [tokenizer.decode([i]) for i in snprintf_ids],
        },
        "sprintf_tokenization": {
            "ids": sprintf_ids,
            "is_single_token": sprintf_is_single,
            "parts": [tokenizer.decode([i]) for i in sprintf_ids],
        },
        "static_prompts": {
            pt: {
                str(k): {
                    "tracked": v["tracked"],
                    "sprintf_rank": v["sprintf_rank"],
                    "snprintf_rank": v["snprintf_rank"],
                    "top_5": v["top_k"][:5],
                } for k, v in static_results[pt].items()
            } for pt in ["secure", "vulnerable"]
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
                        "snprintf_rank": pr[pt][layer]["snprintf_rank"],
                        "top_5": pr[pt][layer]["top_k"][:5],
                    }
        output["dataset_prompts"].append(entry)

    results_path = RESULTS_DIR / f"logit_lens_70b_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nResults saved: {results_path}")

    loader.unload()
    print(f"\nLogit lens analysis complete.")


if __name__ == "__main__":
    main()
