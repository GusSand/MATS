#!/usr/bin/env python3
"""LOBO α-sweep + combined-condition support.

Extends 01c_lobo_recon.py:
- Accepts a list of α values (sweep)
- Accepts a prompting condition (baseline / sys_verbose / usr_verbose)
- Re-uses the same chat-template-activation extraction across all αs
- One model load, one activation pass; iterate over (α, fold)

Usage:
    # CWE-787 baseline_steer LOBO α-sweep
    python 01d_lobo_combined_sweep.py --cwe CWE-787 --alphas 2.0 3.0 3.5 4.0 5.0 \
        --condition baseline --tag lobo_sweep

    # CWE-787 usr_verbose_steer LOBO at single α
    python 01d_lobo_combined_sweep.py --cwe CWE-787 --alphas 2.0 \
        --condition usr_verbose --tag lobo_combined

    # CWE-119 usr_verbose_steer LOBO at α=1
    python 01d_lobo_combined_sweep.py --cwe CWE-119 --alphas 1.0 \
        --condition usr_verbose --tag lobo_combined
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "config"))
sys.path.insert(0, str(ROOT / "lib"))

import phrasings  # noqa: E402
import datasets as ds  # noqa: E402
import prompt_builder  # noqa: E402
import scoring  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LAYER = 31
N_SEEDS = 10
TEMPERATURE = 0.6
TOP_P = 0.9
MAX_NEW_TOKENS = 512
BATCH_SIZE = 8
RESULTS_DIR = ROOT / "results"


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def extract_activations(model, tokenizer, prompts, batch_size=8):
    """Last-token L31 activations for a list of formatted prompts."""
    all_acts = []
    target_layer = model.model.layers[LAYER]
    captured = []
    def hook(m, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured.append(h[:, -1, :].detach().float().cpu().numpy())
    handle = target_layer.register_forward_hook(hook)
    try:
        for chunk in chunks(prompts, batch_size):
            captured.clear()
            inputs = tokenizer(chunk, return_tensors="pt", padding=True,
                               truncation=False).to(model.device)
            with torch.no_grad():
                model(**inputs)
            assert len(captured) == 1
            all_acts.append(captured[0])
    finally:
        handle.remove()
    return np.concatenate(all_acts, axis=0)


def make_steering_hook(direction_tensor, alpha):
    def hook(m, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h[:, -1, :] = h[:, -1, :] + alpha * direction_tensor
        return (h,) + out[1:] if isinstance(out, tuple) else h
    return hook


def generate_with_steering(model, tokenizer, prompts, direction, alpha, n_seeds, batch_size):
    direction_tensor = torch.tensor(direction, dtype=torch.float16, device=model.device)
    target_layer = model.model.layers[LAYER]
    completions = [[None] * n_seeds for _ in prompts]
    BIG = 1_000_000
    for seed_idx in range(n_seeds):
        for chunk_idx, chunk in enumerate(chunks(list(enumerate(prompts)), batch_size)):
            indices = [i for i, _ in chunk]
            chunk_prompts = [p for _, p in chunk]
            torch.manual_seed(seed_idx * BIG + chunk_idx)
            torch.cuda.manual_seed(seed_idx * BIG + chunk_idx)
            inputs = tokenizer(chunk_prompts, return_tensors="pt", padding=True,
                               truncation=False).to(model.device)
            padded_len = inputs.input_ids.shape[1]
            handle = target_layer.register_forward_hook(make_steering_hook(direction_tensor, alpha))
            try:
                with torch.no_grad():
                    out = model.generate(
                        **inputs, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
                        max_new_tokens=MAX_NEW_TOKENS, num_return_sequences=1,
                        pad_token_id=tokenizer.pad_token_id,
                    )
            finally:
                handle.remove()
            for i, orig in enumerate(indices):
                gen_ids = out[i, padded_len:]
                completions[orig][seed_idx] = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return completions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cwe", required=True, choices=ds.CWE_LIST)
    ap.add_argument("--alphas", type=float, nargs="+", required=True)
    ap.add_argument("--condition", required=True,
                    choices=[c["id"] for c in phrasings.CONDITIONS])
    ap.add_argument("--tag", default="lobo")
    args = ap.parse_args()

    cwe = args.cwe
    cond = phrasings.CONDITION_BY_ID[args.condition]
    print(f"[config] CWE={cwe} αs={args.alphas} cond={args.condition}", flush=True)

    print(f"[load] {MODEL}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="cuda:0").eval()
    print(f"[load] done. dtype={model.dtype}", flush=True)

    rows = [json.loads(l) for l in open(ds.ADVERSARIAL_PATHS[cwe])]
    by_base = defaultdict(list)
    for r in rows:
        by_base[r["base_id"]].append(r)
    base_ids = sorted(by_base.keys())
    print(f"[setup] {cwe}: {len(base_ids)} base_ids, {len(rows)} prompts", flush=True)

    # Activation extraction uses BASELINE condition — we want direction from
    # the natural representation, not the prompted one. The chosen `cond` is
    # only for the EVAL prompts (where we steer + generate).
    cond_baseline = phrasings.CONDITION_BY_ID["baseline"]
    print("[acts] Extracting activations (baseline-formatted, both variants)...", flush=True)
    t0 = time.time()
    vuln_prompts = [
        prompt_builder.build_prompt_string(tokenizer, r, cwe, "adversarial", cond_baseline)
        for r in rows
    ]
    sec_prompts = [
        prompt_builder.build_prompt_string(tokenizer, r, cwe, "secure", cond_baseline)
        for r in rows
    ]
    vuln_acts = extract_activations(model, tokenizer, vuln_prompts)
    sec_acts = extract_activations(model, tokenizer, sec_prompts)
    print(f"[acts] done in {time.time()-t0:.1f}s. vuln={vuln_acts.shape} sec={sec_acts.shape}", flush=True)

    base_indices = defaultdict(list)
    for i, r in enumerate(rows):
        base_indices[r["base_id"]].append(i)

    all_results = {}
    for alpha_value in args.alphas:
        print(f"\n========= α = {alpha_value} =========", flush=True)
        fold_results = {}
        agg = {"n": 0, "secure": 0, "insecure": 0, "other": 0}

        for held_out in base_ids:
            held_out_rows = by_base[held_out]
            train_indices = [
                i for bid, idxs in base_indices.items() if bid != held_out for i in idxs
            ]
            train_vuln = vuln_acts[train_indices]
            train_sec = sec_acts[train_indices]
            direction = (train_sec.mean(axis=0) - train_vuln.mean(axis=0)).astype(np.float32)
            norm = float(np.linalg.norm(direction))

            # Eval prompts use the user-specified condition (baseline / sys_verbose / usr_verbose).
            prompt_strs = [
                prompt_builder.build_prompt_string(tokenizer, r, cwe, "adversarial", cond)
                for r in held_out_rows
            ]

            t0 = time.time()
            completions = generate_with_steering(
                model, tokenizer, prompt_strs, direction, alpha_value, N_SEEDS, BATCH_SIZE
            )
            counts = {"secure": 0, "insecure": 0, "other": 0}
            n = 0
            for r, gens in zip(held_out_rows, completions):
                for gen in gens:
                    label = scoring.score_completion(gen, cwe, r, "adversarial")
                    counts[label] += 1
                    n += 1
            fold_results[held_out] = {
                "n": n, "secure": counts["secure"], "insecure": counts["insecure"],
                "other": counts["other"],
                "strict_secure_rate": counts["secure"] / n,
                "direction_norm": norm,
                "n_train_pairs": len(train_indices),
                "runtime_s": round(time.time() - t0, 1),
            }
            for k in ("n", "secure", "insecure", "other"):
                agg[k] += counts.get(k, 0) if k != "n" else n
            print(f"[α={alpha_value} {held_out}] strict={fold_results[held_out]['strict_secure_rate']:.4f} "
                  f"({counts['secure']}/{n}) norm={norm:.3f} dt={fold_results[held_out]['runtime_s']}s",
                  flush=True)

        agg["strict_secure_rate"] = agg["secure"] / agg["n"]
        all_results[str(alpha_value)] = {"fold_results": fold_results, "aggregate": agg}
        print(f"[α={alpha_value} AGG] strict_secure={agg['strict_secure_rate']:.4f}", flush=True)

    out = {
        "metadata": {
            "model": MODEL, "cwe": cwe, "condition": args.condition,
            "alphas": args.alphas, "layer": LAYER,
            "n_seeds": N_SEEDS, "temperature": TEMPERATURE, "top_p": TOP_P,
            "max_new_tokens": MAX_NEW_TOKENS,
            "method": f"true LOBO sweep: per-fold direction × condition={args.condition}",
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        },
        "results_per_alpha": all_results,
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{args.tag}_{cwe}_{args.condition}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {out_path}", flush=True)
    print("\n[summary] best-α per condition:", flush=True)
    best = max(all_results.items(), key=lambda kv: kv[1]["aggregate"]["strict_secure_rate"])
    print(f"  best α={best[0]} → strict={best[1]['aggregate']['strict_secure_rate']:.4f}", flush=True)


if __name__ == "__main__":
    main()
