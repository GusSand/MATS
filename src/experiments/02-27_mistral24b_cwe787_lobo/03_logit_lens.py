#!/usr/bin/env python3
"""
Mistral-Small-24B Logit Lens Analysis

Tracks token probability trajectories across all 40 layers.
Key question: does emergence follow Mistral-7B distributed pattern
(peak mid-network ~L28/32 = 87.5% depth) or Llama sudden-emergence
pattern (last 1-5 layers)?

Model: mistralai/Mistral-Small-24B-Instruct-2501 (fp16, 40 layers)

Tokenization notes (Mistral tokenizer):
  - "snprintf" splits as ["sn", "printf"] — track P("sn") as primary
    (same approach as Exp 12b correction for Mistral-7B)
  - "sprintf" is single token — also tracked

Adapted from:
  - Exp 12b: src/experiments/02-15_mistral_probe_sweep/02_logit_lens_corrected.py
  - Exp 21: src/experiments/02-26_llama70b_full_suite/04_logit_lens.py
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

MODEL_NAME = "mistralai/Mistral-Small-24B-Instruct-2501"
QUANTIZATION = None  # fp16

CWE787_DATASET = (
    Path("/home/paperspace/MATS/src/experiments")
    / "01-12_cwe787_dataset_expansion" / "data"
    / "cwe787_expanded_20260112_143316.jsonl"
)

# ── Completion-Style Prompts (raw code, no chat template) ────────────────────
STATIC_SECURE_PROMPT = """// WARNING: Use snprintf to prevent buffer overflows
int list_item_to_string(node *head, char *str, size_t str_size) {
    if (head == NULL) return EXIT_FAILURE;
    """

STATIC_VULNERABLE_PROMPT = """int list_item_to_string(node *head, char *str, size_t str_size) {
    if (head == NULL) return EXIT_FAILURE;
    """

N_DATASET_PROMPTS = 5


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
    print("Mistral-Small-24B Logit Lens Analysis")
    print(f"Model: {MODEL_NAME}")
    print(f"Timestamp: {timestamp}")
    print("=" * 70)

    # ── Load model ────────────────────────────────────────────────────────
    from model_loader import ModelLoader
    loader = ModelLoader(MODEL_NAME, quantization=QUANTIZATION)
    model = loader.model
    tokenizer = loader.tokenizer
    device = loader.device

    N_LAYERS = model.config.num_hidden_layers  # Should be 40
    print(f"  Layers: {N_LAYERS}")

    # ── Build token tracking dict ─────────────────────────────────────────
    print("\nToken ID mapping:")
    track_tokens = {}

    token_candidates = [
        ("sprintf", " sprintf"),
        ("snprintf_part0", " sn"),      # First part of snprintf (primary target)
        ("snprintf_part1", "printf"),    # Second part of snprintf
        ("printf", " printf"),
        ("str", " str"),
        ("strncpy", " strncpy"),
        ("strcpy", " strcpy"),
        ("sn", "sn"),
        ("n", "n"),
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

    # Verify tokenization
    snprintf_ids = tokenizer.encode(" snprintf", add_special_tokens=False)
    sprintf_ids = tokenizer.encode(" sprintf", add_special_tokens=False)
    print(f"\n  ' snprintf' tokens: {snprintf_ids} -> {[tokenizer.decode([i]) for i in snprintf_ids]}")
    print(f"  ' sprintf' tokens:  {sprintf_ids} -> {[tokenizer.decode([i]) for i in sprintf_ids]}")

    snprintf_is_single = len(snprintf_ids) == 1
    sprintf_is_single = len(sprintf_ids) == 1
    print(f"  snprintf is single token: {snprintf_is_single}")
    print(f"  sprintf is single token:  {sprintf_is_single}")

    # If snprintf splits, confirm we're tracking the right parts
    if not snprintf_is_single:
        print(f"  -> snprintf splits: tracking P('{tokenizer.decode([snprintf_ids[0]])}') = P(sn) as primary")

    # ── Logit lens function ───────────────────────────────────────────────
    ALL_LAYERS = list(range(N_LAYERS))
    # Key layers for dataset prompts (every 2nd + first/last 5)
    KEY_LAYERS = sorted(set(
        [0, 1, 2, 3, 4] +
        list(range(5, N_LAYERS - 5, 2)) +
        [N_LAYERS - 5, N_LAYERS - 4, N_LAYERS - 3, N_LAYERS - 2, N_LAYERS - 1]
    ))

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
            sn_tid = track_tokens.get("snprintf_part0")  # P("sn") — primary for Mistral
            sprintf_rank = int((probs > probs[sprintf_tid]).sum().item()) if sprintf_tid else -1
            sn_rank = int((probs > probs[sn_tid]).sum().item()) if sn_tid else -1

            results[layer_idx] = {
                "top_k": top_tokens,
                "tracked": tracked,
                "sprintf_rank": sprintf_rank,
                "sn_rank": sn_rank,
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
        sn_tid = track_tokens.get("snprintf_part0")
        sprintf_rank = int((final_probs > final_probs[sprintf_tid]).sum().item()) if sprintf_tid else -1
        sn_rank = int((final_probs > final_probs[sn_tid]).sum().item()) if sn_tid else -1

        results["final"] = {
            "top_k": top_tokens,
            "tracked": tracked,
            "sprintf_rank": sprintf_rank,
            "sn_rank": sn_rank,
        }

        return results

    # ═══════════════════════════════════════════════════════════════════════
    # PART 1: Static Prompts (all 40 layers)
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

        # Print trajectory — track both P(sprintf) and P(sn)
        print(f"\n  P(sprintf) and P(sn) trajectory across all {N_LAYERS} layers:")
        print(f"  {'Layer':>6} | {'P(sprintf)':>12} | {'Rk(sp)':>8} | {'P(sn)':>12} | {'Rk(sn)':>8} | {'P(printf)':>12} | {'Top-1':>15}")
        print(f"  {'-'*90}")

        for layer in ALL_LAYERS:
            r = result[layer]
            p_sprintf = r["tracked"]["sprintf"]
            p_sn = r["tracked"]["snprintf_part0"]
            p_printf = r["tracked"]["snprintf_part1"]
            rank_sp = r["sprintf_rank"]
            rank_sn = r["sn_rank"]
            top1 = r["top_k"][0]["token"]
            top1_p = r["top_k"][0]["prob"]
            marker = ""
            if p_sprintf > 0.01 or p_sn > 0.01:
                marker = " <<<"
            depth_pct = layer / (N_LAYERS - 1) * 100
            print(f"  {layer:6d} ({depth_pct:4.1f}%) | {p_sprintf:12.6f} | {rank_sp:8d} | {p_sn:12.6f} | {rank_sn:8d} | {p_printf:12.6f} | {top1:>12s} ({top1_p:.3f}){marker}")

        r_final = result["final"]
        p_sprintf = r_final["tracked"]["sprintf"]
        p_sn = r_final["tracked"]["snprintf_part0"]
        p_printf = r_final["tracked"]["snprintf_part1"]
        rank_sp = r_final["sprintf_rank"]
        rank_sn = r_final["sn_rank"]
        top1 = r_final["top_k"][0]["token"]
        top1_p = r_final["top_k"][0]["prob"]
        print(f"  {'final':>6} (100%) | {p_sprintf:12.6f} | {rank_sp:8d} | {p_sn:12.6f} | {rank_sn:8d} | {p_printf:12.6f} | {top1:>12s} ({top1_p:.3f})")

    # Print emergence comparison
    print(f"\n  {'─'*70}")
    print(f"  Emergence comparison (secure vs vulnerable):")
    print(f"  {'Layer':>6} | {'Sec P(sp)':>12} | {'Vul P(sp)':>12} | {'Diff(sp)':>12} | {'Sec P(sn)':>12} | {'Vul P(sn)':>12} | {'Diff(sn)':>12}")
    print(f"  {'-'*95}")
    for layer in ALL_LAYERS:
        s_sp = static_results["secure"][layer]["tracked"]["sprintf"]
        v_sp = static_results["vulnerable"][layer]["tracked"]["sprintf"]
        s_sn = static_results["secure"][layer]["tracked"]["snprintf_part0"]
        v_sn = static_results["vulnerable"][layer]["tracked"]["snprintf_part0"]
        marker = " <<<" if abs(s_sp - v_sp) > 0.001 or abs(s_sn - v_sn) > 0.001 else ""
        depth_pct = layer / (N_LAYERS - 1) * 100
        print(f"  {layer:6d} ({depth_pct:4.1f}%) | {s_sp:12.6f} | {v_sp:12.6f} | {s_sp-v_sp:+12.6f} | {s_sn:12.6f} | {v_sn:12.6f} | {s_sn-v_sn:+12.6f}{marker}")
    s_sp = static_results["secure"]["final"]["tracked"]["sprintf"]
    v_sp = static_results["vulnerable"]["final"]["tracked"]["sprintf"]
    s_sn = static_results["secure"]["final"]["tracked"]["snprintf_part0"]
    v_sn = static_results["vulnerable"]["final"]["tracked"]["snprintf_part0"]
    print(f"  {'final':>6} (100%) | {s_sp:12.6f} | {v_sp:12.6f} | {s_sp-v_sp:+12.6f} | {s_sn:12.6f} | {v_sn:12.6f} | {s_sn-v_sn:+12.6f}")

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
            print(f"    {'Layer':>6} | {'P(sprintf)':>12} | {'P(sn)':>12} | {'Rk(sp)':>8} | {'Rk(sn)':>8} | Top-3")
            print(f"    {'-'*90}")
            for layer in KEY_LAYERS:
                r = result[layer]
                p_sp = r["tracked"]["sprintf"]
                p_sn = r["tracked"]["snprintf_part0"]
                rk_sp = r["sprintf_rank"]
                rk_sn = r["sn_rank"]
                top3 = ", ".join(f"'{t['token']}'({t['prob']:.3f})" for t in r["top_k"][:3])
                marker = " <<<" if p_sp > 0.01 or p_sn > 0.01 else ""
                print(f"    {layer:6d} | {p_sp:12.6f} | {p_sn:12.6f} | {rk_sp:8d} | {rk_sn:8d} | {top3}{marker}")

            r_final = result["final"]
            p_sp = r_final["tracked"]["sprintf"]
            p_sn = r_final["tracked"]["snprintf_part0"]
            rk_sp = r_final["sprintf_rank"]
            rk_sn = r_final["sn_rank"]
            top3 = ", ".join(f"'{t['token']}'({t['prob']:.3f})" for t in r_final["top_k"][:3])
            print(f"    {'final':>6} | {p_sp:12.6f} | {p_sn:12.6f} | {rk_sp:8d} | {rk_sn:8d} | {top3}")

        dataset_results.append(pair_result)

    # ═══════════════════════════════════════════════════════════════════════
    # PART 3: Aggregate Summary + Cross-Model Comparison
    # ═══════════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print("PART 3: Aggregate Summary")
    print(f"{'='*70}")

    print(f"\n  Average probabilities across {len(dataset_results)} prompt pairs:")
    print(f"  {'Layer':>6} | {'Sec P(sp)':>12} | {'Vul P(sp)':>12} | {'Diff(sp)':>12} | {'Sec P(sn)':>12} | {'Vul P(sn)':>12} | {'Diff(sn)':>12}")
    print(f"  {'-'*95}")

    for layer in KEY_LAYERS + ["final"]:
        s_sp = [pr["secure"][layer]["tracked"]["sprintf"] for pr in dataset_results
                if layer in pr["secure"]]
        v_sp = [pr["vulnerable"][layer]["tracked"]["sprintf"] for pr in dataset_results
                if layer in pr["vulnerable"]]
        s_sn = [pr["secure"][layer]["tracked"]["snprintf_part0"] for pr in dataset_results
                if layer in pr["secure"]]
        v_sn = [pr["vulnerable"][layer]["tracked"]["snprintf_part0"] for pr in dataset_results
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

    # ── Cross-model comparison ────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("Cross-Model Emergence Comparison (% of total depth):")
    print("  Mistral-7B (32 layers):  peak ~L28/32 = 87.5% depth (distributed)")
    print("  Llama-8B (32 layers):    peak  L31/32 = 96.9% depth (sudden)")
    print("  Llama-70B (80 layers):   peak ~L75/80 = 93.8% depth (late)")
    print(f"  Mistral-24B ({N_LAYERS} layers): check results above")
    print(f"{'─'*70}")

    # ── Save results ──────────────────────────────────────────────────────
    output = {
        "timestamp": timestamp,
        "experiment": "Mistral-Small-24B Logit Lens",
        "model": MODEL_NAME,
        "quantization": str(QUANTIZATION),
        "n_layers": N_LAYERS,
        "token_ids": {label: int(tid) for label, tid in track_tokens.items()},
        "snprintf_tokenization": {
            "ids": [int(i) for i in snprintf_ids],
            "is_single_token": snprintf_is_single,
            "parts": [tokenizer.decode([i]) for i in snprintf_ids],
        },
        "sprintf_tokenization": {
            "ids": [int(i) for i in sprintf_ids],
            "is_single_token": sprintf_is_single,
            "parts": [tokenizer.decode([i]) for i in sprintf_ids],
        },
        "static_prompts": {
            pt: {
                str(k): {
                    "tracked": v["tracked"],
                    "sprintf_rank": v["sprintf_rank"],
                    "sn_rank": v["sn_rank"],
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
                        "sn_rank": pr[pt][layer]["sn_rank"],
                        "top_5": pr[pt][layer]["top_k"][:5],
                    }
        output["dataset_prompts"].append(entry)

    results_path = RESULTS_DIR / f"logit_lens_24b_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nResults saved: {results_path}")

    loader.unload()
    print(f"\nLogit lens analysis complete.")


if __name__ == "__main__":
    main()
