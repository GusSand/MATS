#!/usr/bin/env python3
"""Combined steering + prompting baseline runner.

Reuses the prompt-format / scoring / dataset infra from 01_run_baseline.py and
adds a forward hook on layer 31 that applies α * direction to the last-token
hidden state on every generation step (matches canonical 02-05 / 02-10 LOBO).

Conditions tested (3): baseline_steer, sys_verbose_steer, usr_verbose_steer.
Each pairs a base prompting condition with steering at the CWE's LOBO-best α.
"""

import argparse
import json
import sys
import time
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
from steering_config import STEER_CONFIG, STEER_LAYER, COMBINED_CONDITIONS  # noqa: E402


MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TEMPERATURE = 0.6
TOP_P = 0.9
MAX_NEW_TOKENS = 512
N_SEEDS_DEFAULT = 10
GLOBAL_SEED = 42
BATCH_SIZE_DEFAULT = 8
RESULTS_DIR = ROOT / "results"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def make_steering_hook(direction_tensor, alpha):
    """Forward hook that adds α * direction to the last-token hidden state.

    Matches the canonical hook used in 02-05_cross_cwe_steering and 02-10:
    only modifies position [-1] of the residual stream output of the target
    layer. During generation, this fires on every decode step, so each new
    token gets the steering nudge.
    """
    def hook(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        h[:, -1, :] = h[:, -1, :] + alpha * direction_tensor
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h
    return hook


def run_combined(model, tokenizer, cwes, conditions, n_seeds, batch_size,
                 jsonl_path, limit_per_cwe=None, variant="adversarial",
                 alpha_override=None):
    """alpha_override: if not None, replaces the per-CWE α with this single value.
    Used by alpha sweeps."""
    if alpha_override is not None:
        alpha_meta = {c: alpha_override for c in cwes}
    else:
        alpha_meta = {c: STEER_CONFIG[c]["alpha"] for c in cwes}

    summary = {"per_cwe_per_condition": {}, "metadata": {
        "model": MODEL, "variant": variant, "n_seeds": n_seeds,
        "temperature": TEMPERATURE, "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS, "global_seed": GLOBAL_SEED,
        "batch_size": batch_size, "steer_layer": STEER_LAYER,
        "alpha_per_cwe": alpha_meta,
        "alpha_override": alpha_override,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }}

    out_f = open(jsonl_path, "a")

    for cwe in cwes:
        # Load dataset
        if variant in ("adversarial", "secure"):
            rows = load_jsonl(ds.ADVERSARIAL_PATHS[cwe])
        else:
            path = ds.NEUTRAL_C_PATH if cwe in ds.C_CWES else ds.NEUTRAL_PY_PATH
            rows = [r for r in load_jsonl(path) if r["cwe"] == cwe]

        if limit_per_cwe is not None:
            rows = rows[:limit_per_cwe]

        id_field = ds.get_id_field(cwe) if variant != "neutral" else "id"

        # Load steering direction and alpha for this CWE
        cfg = STEER_CONFIG[cwe]
        direction_np = np.load(cfg["direction_path"]).astype(np.float32)
        direction = torch.tensor(direction_np, dtype=torch.float16, device=model.device)
        alpha = alpha_override if alpha_override is not None else cfg["alpha"]
        target_layer = model.model.layers[STEER_LAYER]

        summary["per_cwe_per_condition"].setdefault(cwe, {})

        for combined_cond_meta in conditions:
            cid = combined_cond_meta["id"]
            base_cond = phrasings.CONDITION_BY_ID[combined_cond_meta["base_condition"]]
            tag = f"{variant} {cwe} {cid} (α={alpha})"
            counts = {"secure": 0, "insecure": 0, "other": 0, "n": 0}

            prompt_strs = [
                prompt_builder.build_prompt_string(tokenizer, r, cwe, variant, base_cond)
                for r in rows
            ]

            t0 = time.time()
            BIG = 1_000_000
            for seed_idx in range(n_seeds):
                for chunk_idx, (chunk_rows, chunk_prompts) in enumerate(zip(
                    chunks(rows, batch_size), chunks(prompt_strs, batch_size)
                )):
                    fresh = seed_idx * BIG + chunk_idx
                    torch.manual_seed(fresh)
                    torch.cuda.manual_seed(fresh)
                    inputs = tokenizer(chunk_prompts, return_tensors="pt", padding=True,
                                       truncation=False).to(model.device)
                    padded_len = inputs.input_ids.shape[1]

                    # Register hook for this batch only
                    hook_handle = target_layer.register_forward_hook(
                        make_steering_hook(direction, alpha)
                    )
                    try:
                        with torch.no_grad():
                            out = model.generate(
                                **inputs,
                                do_sample=True,
                                temperature=TEMPERATURE,
                                top_p=TOP_P,
                                max_new_tokens=MAX_NEW_TOKENS,
                                num_return_sequences=1,
                                pad_token_id=tokenizer.pad_token_id,
                            )
                    finally:
                        hook_handle.remove()

                    for i_prompt, prompt_row in enumerate(chunk_rows):
                        gen_ids = out[i_prompt, padded_len:]
                        completion = tokenizer.decode(gen_ids, skip_special_tokens=True)
                        label = scoring.score_completion(completion, cwe, prompt_row, variant)
                        counts[label] += 1
                        counts["n"] += 1

                        out_f.write(json.dumps({
                            "prompt_id": prompt_row[id_field],
                            "cwe": cwe,
                            "variant": variant,
                            "condition": cid,
                            "alpha": alpha,
                            "seed_idx": seed_idx,
                            "raw_output": completion,
                            "scored_label": label,
                        }) + "\n")
                out_f.flush()

            dt = time.time() - t0
            summary["per_cwe_per_condition"][cwe][cid] = {
                "n": counts["n"],
                "secure": counts["secure"],
                "insecure": counts["insecure"],
                "other": counts["other"],
                "strict_secure_rate": counts["secure"] / counts["n"] if counts["n"] else 0,
                "alpha": alpha,
                "runtime_s": round(dt, 1),
            }
            print(f"[{tag}] n={counts['n']} secure={counts['secure']} "
                  f"insecure={counts['insecure']} other={counts['other']} "
                  f"strict={counts['secure']/counts['n']:.4f} dt={dt:.1f}s", flush=True)

    out_f.close()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["adversarial"],
                    choices=["adversarial", "neutral", "secure"])
    ap.add_argument("--cwes", nargs="+", default=ds.CWE_LIST)
    ap.add_argument("--conditions", nargs="+",
                    default=[c["id"] for c in COMBINED_CONDITIONS])
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    ap.add_argument("--limit-per-cwe", type=int, default=None)
    ap.add_argument("--tag", default="combined")
    ap.add_argument("--alpha-override", type=float, default=None,
                    help="If set, override per-CWE α with this value (for sweeps)")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[load] {MODEL}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="cuda:0"
    )
    model.eval()
    print(f"[load] done. dtype={model.dtype} device={model.device}", flush=True)

    by_id = {c["id"]: c for c in COMBINED_CONDITIONS}
    conditions = [by_id[c] for c in args.conditions]

    for variant in args.variants:
        jsonl_path = RESULTS_DIR / f"{args.tag}_{variant}_generations_{ts}.jsonl"
        summary = run_combined(model, tokenizer, args.cwes, conditions,
                               args.n_seeds, args.batch_size, jsonl_path,
                               args.limit_per_cwe, variant,
                               alpha_override=args.alpha_override)
        sum_path = RESULTS_DIR / f"{args.tag}_{variant}_summary_{ts}.json"
        with open(sum_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[{variant}] wrote {jsonl_path} and {sum_path}", flush=True)


if __name__ == "__main__":
    main()
