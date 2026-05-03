#!/usr/bin/env python3
"""
Run base CodeGen-2B-multi (no SVEN prefix) on our benchmark.

This gives us the unmodified CodeGen baseline to measure SVEN's uplift,
and also serves as the "base model" for our own steering comparison.

Usage:
    python run_baseline_codegen.py [--model-size 2b] [--n-seeds 10] [--dry-run]
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add our scorers to path
SVEN_DIR = Path(__file__).resolve().parent
MATS_ROOT = SVEN_DIR.parent.parent
DATASETS_DIR = MATS_ROOT / "src/experiments/02-05_cross_cwe_steering/datasets"
sys.path.insert(0, str(DATASETS_DIR))

from cwe89.scoring import score_cwe89
from cwe78.scoring import score_cwe78
from cwe79.scoring import score_cwe79

# ─── Config ─────────────────────────────────────────────────────────────────

ADAPTED_DIR = SVEN_DIR / "adapted_prompts"
RESULTS_DIR = SVEN_DIR / "results"

MODEL_DIRS = {
    "350m": "Salesforce/codegen-350M-multi",
    "2b": "Salesforce/codegen-2B-multi",
    "6b": "Salesforce/codegen-6B-multi",
}

CWE_CONFIG = {
    "CWE-119": {"language": "c", "file": "cwe-119_codegen.jsonl"},
    "CWE-134": {"language": "c", "file": "cwe-134_codegen.jsonl"},
    "CWE-787": {"language": "c", "file": "cwe-787_codegen.jsonl"},
    "CWE-89":  {"language": "py", "file": "cwe-89_codegen.jsonl"},
    "CWE-78":  {"language": "py", "file": "cwe-78_codegen.jsonl"},
    "CWE-79":  {"language": "py", "file": "cwe-79_codegen.jsonl"},
}

PYTHON_SCORERS = {
    "CWE-89": score_cwe89,
    "CWE-78": score_cwe78,
    "CWE-79": score_cwe79,
}

TEMPERATURE = 0.4
TOP_P = 0.95
MAX_NEW_TOKENS = 300


def score_c_completion(code, detection):
    secure_pat = detection.get("secure_pattern", "")
    insecure_pat = detection.get("insecure_pattern", "")
    has_secure = bool(re.search(secure_pat, code)) if secure_pat else False
    has_insecure = bool(re.search(insecure_pat, code)) if insecure_pat else False
    if has_secure and not has_insecure:
        return "secure"
    elif has_insecure:
        return "insecure"
    return "other"


def truncate_completion(completion, lang):
    if lang == "py":
        for match in re.finditer("\n", completion):
            cur_idx, next_idx = match.start(), match.end()
            if next_idx < len(completion) and not completion[next_idx].isspace():
                completion = completion[:cur_idx]
                break
    elif lang == "c":
        if "\n}" in completion:
            completion = completion[: completion.find("\n}") + 2]
        else:
            for s in ["\n    //", "\n    /*"]:
                if s in completion:
                    completion = completion[: completion.rfind(s)]
                    completion = completion.rstrip() + "\n}"
    return completion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", default="2b", choices=["350m", "2b", "6b"])
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cwes", nargs="+", default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_name = MODEL_DIRS[args.model_size]
    print(f"Loading base {model_name}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.to(args.device)
    model.eval()
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    cwes_to_run = args.cwes or list(CWE_CONFIG.keys())
    all_results = {}
    total_gen_time = 0.0
    total_completions = 0

    for cwe in cwes_to_run:
        cfg = CWE_CONFIG[cwe]
        lang = cfg["language"]
        data_file = ADAPTED_DIR / cfg["file"]

        print(f"\n{'='*70}")
        print(f"{cwe} ({lang.upper()}) — BASE MODEL (no SVEN)")
        print(f"{'='*70}")

        prompts = []
        with open(data_file) as f:
            for line in f:
                if line.strip():
                    prompts.append(json.loads(line))
        if args.dry_run:
            prompts = prompts[:2]

        n_seeds = args.n_seeds
        print(f"  {len(prompts)} prompts, {n_seeds} seeds = {len(prompts) * n_seeds} completions")

        cwe_results = []
        secure_count = 0
        insecure_count = 0
        other_count = 0

        for p_idx, prompt_data in enumerate(prompts):
            prompt_text = prompt_data["codegen_vulnerable"]
            prompt_id = prompt_data.get("id") or prompt_data.get("pair_id", f"prompt_{p_idx}")
            base_id = prompt_data.get("base_id", "unknown")

            input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(args.device)

            torch.manual_seed(42)
            torch.cuda.manual_seed(42)

            t_gen = time.time()
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    do_sample=True,
                    num_return_sequences=n_seeds,
                    temperature=TEMPERATURE,
                    max_new_tokens=MAX_NEW_TOKENS,
                    top_p=TOP_P,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
            gen_time = time.time() - t_gen
            total_gen_time += gen_time

            for seq_idx in range(n_seeds):
                completion_tokens = output[seq_idx, input_ids.shape[1]:]
                completion = tokenizer.decode(completion_tokens)
                if tokenizer.eos_token and tokenizer.eos_token in completion:
                    completion = completion[: completion.find(tokenizer.eos_token)]
                completion = truncate_completion(completion, lang)
                full_code = prompt_text + completion

                if lang == "py":
                    label = PYTHON_SCORERS[cwe](full_code)
                else:
                    label = score_c_completion(full_code, prompt_data["detection"])

                if label == "secure":
                    secure_count += 1
                elif label == "insecure":
                    insecure_count += 1
                else:
                    other_count += 1

                cwe_results.append({
                    "prompt_id": prompt_id,
                    "base_id": base_id,
                    "sample_idx": seq_idx,
                    "label": label,
                    "completion": completion,
                    "full_code": full_code,
                    "gen_time_s": round(gen_time / n_seeds, 2),
                })
                total_completions += 1

            if (p_idx + 1) % 10 == 0 or p_idx == len(prompts) - 1:
                total_scored = secure_count + insecure_count + other_count
                sec_rate = secure_count / total_scored if total_scored > 0 else 0
                print(f"  [{p_idx+1}/{len(prompts)}] secure={sec_rate:.1%} "
                      f"({secure_count}/{total_scored}), "
                      f"insecure={insecure_count}, other={other_count}")

        total_scored = secure_count + insecure_count + other_count
        classifiable = secure_count + insecure_count
        sec_rate = secure_count / classifiable if classifiable > 0 else 0
        sec_rate_total = secure_count / total_scored if total_scored > 0 else 0

        all_results[cwe] = {
            "results": cwe_results,
            "summary": {
                "total": total_scored,
                "secure": secure_count,
                "insecure": insecure_count,
                "other": other_count,
                "secure_rate_classifiable": round(sec_rate, 4),
                "secure_rate_total": round(sec_rate_total, 4),
            },
        }

        print(f"\n  {cwe} SUMMARY:")
        print(f"    Secure: {secure_count}/{total_scored} ({sec_rate_total:.1%} of all, "
              f"{sec_rate:.1%} of classifiable)")
        print(f"    Insecure: {insecure_count}, Other: {other_count}")

    output_data = {
        "metadata": {
            "method": "Base CodeGen (no SVEN prefix)",
            "model": model_name,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_new_tokens": MAX_NEW_TOKENS,
            "n_seeds": args.n_seeds,
            "timestamp": timestamp,
            "total_completions": total_completions,
            "total_gen_time_s": round(total_gen_time, 1),
            "model_load_time_s": round(load_time, 1),
            "gpu_hours": round((total_gen_time + load_time) / 3600, 3),
        },
        "per_cwe": {
            cwe: data["summary"] for cwe, data in all_results.items()
        },
    }

    print(f"\n{'='*70}")
    print("OVERALL: Base CodeGen (no SVEN)")
    print(f"{'='*70}")
    print(f"{'CWE':<10} {'Secure':>8} {'Insecure':>10} {'Other':>8} {'Sec Rate':>10} {'Sec/Class':>10}")
    print("-" * 58)
    for cwe, summary in output_data["per_cwe"].items():
        print(f"{cwe:<10} {summary['secure']:>8} {summary['insecure']:>10} "
              f"{summary['other']:>8} {summary['secure_rate_total']:>10.1%} "
              f"{summary['secure_rate_classifiable']:>10.1%}")

    print(f"\nTotal GPU time: {total_gen_time:.0f}s ({total_gen_time/3600:.2f} GPU-hours)")

    results_file = RESULTS_DIR / f"baseline_codegen_{args.model_size}_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to: {results_file}")

    detail_file = RESULTS_DIR / f"baseline_codegen_{args.model_size}_{timestamp}_detail.json"
    with open(detail_file, "w") as f:
        json.dump({"metadata": output_data["metadata"], "per_cwe": all_results}, f, indent=2)
    print(f"Details: {detail_file}")


if __name__ == "__main__":
    main()
