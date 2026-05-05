#!/usr/bin/env python3
"""Prompt-engineering baseline runner — Llama-3.1-8B-Instruct, 6 CWEs.

Architectural reference: see ../docs/experiments/05-03_*.md (written after run).

Generation:
    HF batched generation with num_return_sequences=N_SEEDS per prompt.
    Global torch seed fixed once per (CWE, variant, condition) for reproducibility.

Output:
    results/{variant}_generations_{ts}.jsonl   — one row per generation
    results/{variant}_summary_{ts}.json        — aggregated per-CWE-per-condition
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "config"))
sys.path.insert(0, str(ROOT / "lib"))

import phrasings  # noqa: E402
import datasets as ds  # noqa: E402
import prompt_builder  # noqa: E402
import scoring  # noqa: E402


# ─── Defaults (locked in architect spec) ────────────────────────────────────

MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TEMPERATURE = 0.6
TOP_P = 0.9
MAX_NEW_TOKENS = 512
N_SEEDS_DEFAULT = 10
GLOBAL_SEED = 42

# HF batched generation: BATCH_SIZE prompts × N_SEEDS returns per call
BATCH_SIZE_DEFAULT = 4   # 4 prompts × 10 returns = 40 sequences/call. ~10-15s on A100-80GB.

RESULTS_DIR = ROOT / "results"


# ─── Helpers ────────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def truncate_python(completion: str) -> str:
    """Trim at first non-indented top-level line (mirrors SVEN truncate logic)."""
    import re
    for m in re.finditer("\n", completion):
        nxt = m.end()
        if nxt < len(completion) and not completion[nxt].isspace():
            return completion[:m.start()]
    return completion


def truncate_c(completion: str) -> str:
    """Trim at first \\n} (closing brace at column 0) or first \\n#include after start."""
    import re
    # Cut at next top-level closing brace if any
    for m in re.finditer(r"\n\}", completion):
        return completion[:m.end()]
    return completion


def truncate_completion(completion: str, cwe: str) -> str:
    return truncate_python(completion) if cwe in ds.PY_CWES else truncate_c(completion)


# ─── Main runner ────────────────────────────────────────────────────────────

def run_variant(model, tokenizer, variant: str, cwes: list, conditions: list,
                n_seeds: int, batch_size: int, jsonl_path: Path,
                limit_per_cwe: int = None) -> dict:
    """Run one variant (adversarial / neutral / secure_variant) end-to-end."""

    summary = {"per_cwe_per_condition": {}, "metadata": {
        "model": MODEL, "variant": variant, "n_seeds": n_seeds,
        "temperature": TEMPERATURE, "top_p": TOP_P, "max_new_tokens": MAX_NEW_TOKENS,
        "global_seed": GLOBAL_SEED, "batch_size": batch_size,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }}

    out_f = open(jsonl_path, "a")

    for cwe in cwes:
        # Pick dataset
        if variant in ("adversarial", "secure"):
            rows = load_jsonl(ds.ADVERSARIAL_PATHS[cwe])
        else:  # neutral
            path = ds.NEUTRAL_C_PATH if cwe in ds.C_CWES else ds.NEUTRAL_PY_PATH
            all_rows = load_jsonl(path)
            rows = [r for r in all_rows if r["cwe"] == cwe]

        if limit_per_cwe is not None:
            rows = rows[:limit_per_cwe]

        id_field = ds.get_id_field(cwe) if variant != "neutral" else "id"

        summary["per_cwe_per_condition"].setdefault(cwe, {})

        for cond in conditions:
            cid = cond["id"]
            tag = f"{variant} {cwe} {cid}"
            counts = {"secure": 0, "insecure": 0, "other": 0, "n": 0}

            # Build all prompt strings
            prompt_strs = [
                prompt_builder.build_prompt_string(tokenizer, r, cwe, variant, cond)
                for r in rows
            ]

            t0 = time.time()

            # Batched generation with per-batch seed reset.
            # No truncation: regex scorers handle markdown fences and
            # reformulated `def ...` lines correctly. The earlier "degenerate
            # batched output" symptom was caused by truncate_completion()
            # chopping `def ...` at the first non-indented line, not by any
            # actual batched-sampling bug.
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
    ap.add_argument("--conditions", nargs="+", default=phrasings.CONDITION_IDS)
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    ap.add_argument("--limit-per-cwe", type=int, default=None,
                    help="Smoke-test mode: cap prompts per CWE")
    ap.add_argument("--tag", default="run", help="Tag in output filenames")
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

    conditions = [phrasings.CONDITION_BY_ID[c] for c in args.conditions]

    for variant in args.variants:
        jsonl_path = RESULTS_DIR / f"{args.tag}_{variant}_generations_{ts}.jsonl"
        summary = run_variant(model, tokenizer, variant, args.cwes, conditions,
                              args.n_seeds, args.batch_size, jsonl_path,
                              args.limit_per_cwe)
        sum_path = RESULTS_DIR / f"{args.tag}_{variant}_summary_{ts}.json"
        with open(sum_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[{variant}] wrote {jsonl_path} and {sum_path}", flush=True)


if __name__ == "__main__":
    main()
