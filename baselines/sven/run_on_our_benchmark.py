#!/usr/bin/env python3
"""
Run SVEN (prefix-based secure code generation) on our 105×6 CWE benchmark.

Loads CodeGen-2B-multi + SVEN's trained secure prefix (control_id=0),
generates completions for all 630 insecure prompts across 6 CWEs,
and scores them with our existing regex-based scorers.

Usage:
    # From baselines/sven/ with the SVEN venv activated:
    python run_on_our_benchmark.py [--model-size 2b] [--n-seeds 10] [--dry-run]
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

# Add SVEN to path
SVEN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SVEN_DIR))

# Add our scorers to path
MATS_ROOT = SVEN_DIR.parent.parent
DATASETS_DIR = MATS_ROOT / "src/experiments/02-05_cross_cwe_steering/datasets"
sys.path.insert(0, str(DATASETS_DIR))

from sven.model import load_model

# Import Python CWE scorers
from cwe89.scoring import score_cwe89
from cwe78.scoring import score_cwe78
from cwe79.scoring import score_cwe79


# ─── Configuration ──────────────────────────────────────────────────────────

ADAPTED_DIR = SVEN_DIR / "adapted_prompts"
RESULTS_DIR = SVEN_DIR / "results"

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

# Match our paper's generation params where possible
# SVEN default: temp=0.4, top_p=0.95, max_new_tokens=300
# Our paper: temp=0.7 for adversarial
# We use SVEN's defaults to be fair to their method
TEMPERATURE = 0.4
TOP_P = 0.95
MAX_NEW_TOKENS = 300

# Seeds: 10 for adversarial prompts
SEEDS_ADVERSARIAL = list(range(10))  # 0..9


# ─── C scoring using detection patterns from JSON ──────────────────────────

def score_c_completion(code: str, detection: dict) -> str:
    """Score a C completion using regex detection patterns from the prompt JSON."""
    secure_pat = detection.get("secure_pattern", "")
    insecure_pat = detection.get("insecure_pattern", "")

    has_secure = bool(re.search(secure_pat, code)) if secure_pat else False
    has_insecure = bool(re.search(insecure_pat, code)) if insecure_pat else False

    if has_secure and not has_insecure:
        return "secure"
    elif has_insecure:
        return "insecure"
    return "other"


# ─── SVEN truncation (from their evaler.py) ────────────────────────────────

def truncate_completion(completion: str, lang: str) -> str:
    """Truncate completion to a single function, matching SVEN's own logic."""
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
            last_comment_strs = ["\n    //", "\n    /*"]
            for s in last_comment_strs:
                if s in completion:
                    completion = completion[: completion.rfind(s)]
                    completion = completion.rstrip() + "\n}"
    return completion


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", default="2b", choices=["350m", "2b", "6b"])
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Test on 2 prompts per CWE")
    parser.add_argument("--cwes", nargs="+", default=None,
                        help="Specific CWEs to run (e.g., CWE-89 CWE-119)")
    args = parser.parse_args()

    seeds = SEEDS_ADVERSARIAL[:args.n_seeds]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load SVEN model
    checkpoint_path = SVEN_DIR / "trained" / f"{args.model_size}-prefix" / "checkpoint-last"
    print(f"Loading SVEN ({args.model_size}) from {checkpoint_path}...")
    t0 = time.time()
    tokenizer, model, input_device = load_model("prefix", str(checkpoint_path), False, args)
    model.eval()
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s on {input_device}")

    cwes_to_run = args.cwes or list(CWE_CONFIG.keys())
    all_results = {}
    total_gen_time = 0.0
    total_completions = 0

    for cwe in cwes_to_run:
        cfg = CWE_CONFIG[cwe]
        lang = cfg["language"]
        data_file = ADAPTED_DIR / cfg["file"]

        print(f"\n{'='*70}")
        print(f"{cwe} ({lang.upper()})")
        print(f"{'='*70}")

        # Load adapted prompts
        prompts = []
        with open(data_file) as f:
            for line in f:
                if line.strip():
                    prompts.append(json.loads(line))

        if args.dry_run:
            prompts = prompts[:2]

        print(f"  {len(prompts)} prompts, {len(seeds)} seeds = {len(prompts) * len(seeds)} completions")

        cwe_results = []
        secure_count = 0
        insecure_count = 0
        other_count = 0

        for p_idx, prompt_data in enumerate(prompts):
            # Use insecure/vulnerable prompt — we want to see if SVEN's secure prefix
            # steers the model to produce secure code despite the insecure framing
            prompt_text = prompt_data["codegen_vulnerable"]
            prompt_id = prompt_data.get("id") or prompt_data.get("pair_id", f"prompt_{p_idx}")
            base_id = prompt_data.get("base_id", "unknown")

            input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(input_device)
            n_seeds = len(seeds)

            # Batch generation: all seeds in one forward pass
            torch.manual_seed(42)  # Fixed seed for batch reproducibility
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
                    control_id=0,  # SECURE prefix
                )
            gen_time = time.time() - t_gen
            total_gen_time += gen_time

            # Decode all completions from batch
            for seq_idx in range(n_seeds):
                completion_tokens = output[seq_idx, input_ids.shape[1]:]
                completion = tokenizer.decode(completion_tokens)
                if tokenizer.eos_token and tokenizer.eos_token in completion:
                    completion = completion[: completion.find(tokenizer.eos_token)]
                completion = truncate_completion(completion, lang)

                full_code = prompt_text + completion

                # Score
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

            # Progress
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

    # Save full results
    output = {
        "metadata": {
            "method": "SVEN prefix (control_id=0, secure)",
            "model": f"Salesforce/codegen-{args.model_size.upper()}-multi",
            "sven_checkpoint": str(checkpoint_path),
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_new_tokens": MAX_NEW_TOKENS,
            "n_seeds": len(seeds),
            "seeds": seeds,
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

    # Summary table
    print(f"\n{'='*70}")
    print("OVERALL RESULTS: SVEN secure prefix on our benchmark")
    print(f"{'='*70}")
    print(f"{'CWE':<10} {'Secure':>8} {'Insecure':>10} {'Other':>8} {'Sec Rate':>10} {'Sec/Class':>10}")
    print("-" * 58)
    for cwe, summary in output["per_cwe"].items():
        print(f"{cwe:<10} {summary['secure']:>8} {summary['insecure']:>10} "
              f"{summary['other']:>8} {summary['secure_rate_total']:>10.1%} "
              f"{summary['secure_rate_classifiable']:>10.1%}")

    print(f"\nTotal GPU time: {total_gen_time:.0f}s ({total_gen_time/3600:.2f} GPU-hours)")

    # Save
    results_file = RESULTS_DIR / f"sven_{args.model_size}_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSummary saved to: {results_file}")

    # Save detailed results separately (large file)
    detail_file = RESULTS_DIR / f"sven_{args.model_size}_{timestamp}_detail.json"
    detail_output = {
        "metadata": output["metadata"],
        "per_cwe": all_results,
    }
    with open(detail_file, "w") as f:
        json.dump(detail_output, f, indent=2)
    print(f"Details saved to: {detail_file}")


if __name__ == "__main__":
    main()
