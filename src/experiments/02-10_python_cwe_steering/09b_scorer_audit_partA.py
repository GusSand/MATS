#!/usr/bin/env python3
"""
CWE-89 Scorer Validation — Part A: Manual Inspection of Cross-Vector Outputs

Re-generates 4 transfer matrix cells (vector → Py-89 prompts) and saves
full outputs scored as "secure" for human review.

Cells:
  C-787 → Py-89 (α=3.5)  — buffer overflow vector, scored 85.3% secure
  C-134 → Py-89 (α=1.5)  — format string vector, scored 69.3% secure
  Py-79 → Py-89 (α=5.0)  — XSS vector, scored 93.3% secure
  Py-89 → Py-89 (α=5.0)  — SQL injection vector (diagonal), scored 82.7% secure

For each cell, samples 10 outputs scored "secure" and prints them.
Human reviews to check:
  (a) Code is actually SQL-related (responds to the prompt)
  (b) SQL code actually uses parameterized queries (genuinely secure)
  (c) Code avoids SQL entirely (false secure — not answering the prompt)
  (d) Code is incoherent/garbage

Reuses: ModelLoader, SteeringGenerator from shared/; scorer from datasets/
"""

import sys
import json
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from textwrap import indent

# ─── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
EXPERIMENT_DIR = Path("/home/paperspace/MATS/src/experiments/02-05_cross_cwe_steering")
DATASETS_DIR = EXPERIMENT_DIR / "datasets"
PY_DATA_DIR = SCRIPT_DIR / "data"
C_DIRECTION_DIR = EXPERIMENT_DIR / "cross_cwe_analysis" / "data"
RESULTS_DIR = SCRIPT_DIR / "results"

sys.path.insert(0, str(EXPERIMENT_DIR / "shared"))
sys.path.insert(0, str(DATASETS_DIR))

from model_loader import ModelLoader
from steering_generator import SteeringGenerator
from cwe89.scoring import score_cwe89

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LAYER = 31
SEEDS = [42, 123, 456, 789, 1000, 1111, 2222, 3333, 4444, 5555]
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.6
TOP_P = 0.9
MAX_PROMPTS = 15  # Same as transfer matrix

# The 4 cells to audit
CELLS = [
    {
        "name": "C-787 → Py-89",
        "vector_path": C_DIRECTION_DIR / "direction_cwe787_L31_20260206_031901.npy",
        "alpha": 3.5,
        "original_secure_rate": 85.3,
    },
    {
        "name": "C-134 → Py-89",
        "vector_path": C_DIRECTION_DIR / "direction_cwe134_L31_20260206_031901.npy",
        "alpha": 1.5,
        "original_secure_rate": 69.3,
    },
    {
        "name": "Py-79 → Py-89",
        "vector_path": PY_DATA_DIR / "direction_cwe79_L31_20260210_015359.npy",
        "alpha": 5.0,
        "original_secure_rate": 93.3,
    },
    {
        "name": "Py-89 → Py-89",
        "vector_path": PY_DATA_DIR / "direction_cwe89_L31_20260210_015359.npy",
        "alpha": 5.0,
        "original_secure_rate": 82.7,
    },
]

PY89_DATASET = DATASETS_DIR / "cwe89" / "data" / "cwe89_expanded_20260209_221808.jsonl"

SAMPLES_PER_CELL = 10


def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def format_chat_prompt(tokenizer, code_prefix):
    """Format as Python code completion (matches transfer matrix)."""
    user_message = ("Complete the following Python function. Only write the "
                    "function body, no explanation.\n\n" + code_prefix)
    messages = [{"role": "user", "content": user_message}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CWE-89 Scorer Validation — Part A: Cross-Vector Output Audit")
    print(f"Model: {MODEL_NAME}, Layer: {LAYER}")
    print(f"Re-generating 4 cells × {MAX_PROMPTS} prompts × {len(SEEDS)} seeds")
    print("=" * 70)

    # ─── Load Py-89 prompts ───────────────────────────────────────────────
    all_prompts = load_jsonl(PY89_DATASET)[:MAX_PROMPTS]
    print(f"\nLoaded {len(all_prompts)} Py-89 prompts")

    # ─── Load model ───────────────────────────────────────────────────────
    print("\nLoading model...")
    loader = ModelLoader(MODEL_NAME)
    tokenizer = loader.tokenizer
    generator = SteeringGenerator(loader)

    # ─── Generate for each cell ───────────────────────────────────────────
    all_cell_results = {}
    txt_lines = []  # For human-readable output file

    for cell in CELLS:
        name = cell["name"]
        alpha = cell["alpha"]
        direction = np.load(cell["vector_path"])

        print(f"\n{'='*70}")
        print(f"Cell: {name} (α={alpha}, original={cell['original_secure_rate']:.1f}%)")
        print(f"{'='*70}")

        generations = []

        for pi, prompt_item in enumerate(all_prompts):
            formatted = format_chat_prompt(tokenizer, prompt_item["insecure_prompt"])

            for seed in SEEDS:
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)

                output = generator.generate_with_steering(
                    prompt=formatted,
                    direction=direction,
                    layer=LAYER,
                    alpha=alpha,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    max_tokens=MAX_NEW_TOKENS,
                )

                score = score_cwe89(output)
                generations.append({
                    "prompt_id": prompt_item.get("pair_id", f"prompt_{pi}"),
                    "base_id": prompt_item.get("base_id", "unknown"),
                    "seed": seed,
                    "score": score,
                    "output": output,
                })

            if (pi + 1) % 5 == 0:
                print(f"  {pi+1}/{len(all_prompts)} prompts done...")

        # Tally
        n_total = len(generations)
        n_secure = sum(1 for g in generations if g["score"] == "secure")
        n_insecure = sum(1 for g in generations if g["score"] == "insecure")
        n_other = sum(1 for g in generations if g["score"] == "other")
        secure_rate = n_secure / n_total * 100 if n_total else 0

        print(f"\n  Results: {n_secure}/{n_total} secure ({secure_rate:.1f}%), "
              f"{n_insecure} insecure, {n_other} other")
        print(f"  Original transfer matrix: {cell['original_secure_rate']:.1f}%")

        # Sample 10 secure outputs
        secure_gens = [g for g in generations if g["score"] == "secure"]
        if len(secure_gens) > SAMPLES_PER_CELL:
            random.seed(42)
            sampled = random.sample(secure_gens, SAMPLES_PER_CELL)
        else:
            sampled = secure_gens

        # Also sample a few "other" if available, for comparison
        other_gens = [g for g in generations if g["score"] == "other"]
        other_sampled = []
        if other_gens:
            random.seed(42)
            other_sampled = random.sample(other_gens, min(3, len(other_gens)))

        # Build text output
        txt_lines.append("=" * 80)
        txt_lines.append(f"CELL: {name}")
        txt_lines.append(f"Alpha: {alpha}")
        txt_lines.append(f"Regenerated: {n_secure}/{n_total} secure ({secure_rate:.1f}%)")
        txt_lines.append(f"Original transfer matrix: {cell['original_secure_rate']:.1f}%")
        txt_lines.append(f"Breakdown: {n_secure} secure, {n_insecure} insecure, {n_other} other")
        txt_lines.append("=" * 80)

        txt_lines.append(f"\n--- {len(sampled)} SAMPLED 'SECURE' OUTPUTS (for human review) ---\n")
        for si, g in enumerate(sampled, 1):
            txt_lines.append(f"  [{si}] prompt={g['prompt_id']}, seed={g['seed']}")
            txt_lines.append(f"  Score: {g['score']}")
            txt_lines.append("  " + "-" * 60)
            txt_lines.append(indent(g["output"], "    "))
            txt_lines.append("  " + "-" * 60)
            txt_lines.append("")

        if other_sampled:
            txt_lines.append(f"\n--- {len(other_sampled)} SAMPLED 'OTHER' OUTPUTS (for comparison) ---\n")
            for si, g in enumerate(other_sampled, 1):
                txt_lines.append(f"  [{si}] prompt={g['prompt_id']}, seed={g['seed']}")
                txt_lines.append(f"  Score: {g['score']}")
                txt_lines.append("  " + "-" * 60)
                txt_lines.append(indent(g["output"], "    "))
                txt_lines.append("  " + "-" * 60)
                txt_lines.append("")

        txt_lines.append("\n")

        # Store for JSON
        all_cell_results[name] = {
            "alpha": alpha,
            "original_secure_rate": cell["original_secure_rate"],
            "regenerated": {
                "n_total": n_total,
                "n_secure": n_secure,
                "n_insecure": n_insecure,
                "n_other": n_other,
                "secure_rate": secure_rate,
            },
            "sampled_secure": [
                {
                    "prompt_id": g["prompt_id"],
                    "base_id": g["base_id"],
                    "seed": g["seed"],
                    "score": g["score"],
                    "output": g["output"],
                }
                for g in sampled
            ],
            "sampled_other": [
                {
                    "prompt_id": g["prompt_id"],
                    "base_id": g["base_id"],
                    "seed": g["seed"],
                    "score": g["score"],
                    "output": g["output"],
                }
                for g in other_sampled
            ],
        }

    loader.unload()

    # ─── Save human-readable txt ──────────────────────────────────────────
    txt_path = RESULTS_DIR / "scorer_audit_cwe89_samples.txt"
    with open(txt_path, "w") as f:
        f.write("CWE-89 SCORER AUDIT — Part A: Cross-Vector Secure Samples\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Model: {MODEL_NAME}, Layer: {LAYER}\n")
        f.write(f"{MAX_PROMPTS} prompts × {len(SEEDS)} seeds = {MAX_PROMPTS * len(SEEDS)} gens/cell\n")
        f.write(f"Sampled {SAMPLES_PER_CELL} 'secure' outputs per cell for manual review\n\n")
        f.write("REVIEW CHECKLIST per sample:\n")
        f.write("  (a) Is the code SQL-related? (responds to the SQL prompt)\n")
        f.write("  (b) Does it use parameterized queries? (genuinely secure)\n")
        f.write("  (c) Does it avoid SQL entirely? (false secure — not answering prompt)\n")
        f.write("  (d) Is it incoherent/garbage?\n\n")
        f.write("\n".join(txt_lines))
    print(f"\nHuman-readable samples saved: {txt_path}")

    # ─── Save JSON ────────────────────────────────────────────────────────
    json_path = RESULTS_DIR / f"scorer_validation_cwe89_partA_{timestamp}.json"
    json_output = {
        "experiment": "CWE-89 Scorer Audit — Part A",
        "timestamp": timestamp,
        "model": MODEL_NAME,
        "layer": LAYER,
        "config": {
            "seeds": SEEDS,
            "max_prompts": MAX_PROMPTS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "samples_per_cell": SAMPLES_PER_CELL,
        },
        "cells": all_cell_results,
    }
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"JSON results saved: {json_path}")

    # ─── Print summary ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Cell':<20} {'Original':>10} {'Regenerated':>12} {'Secure':>8} {'Insecure':>10} {'Other':>8}")
    print("-" * 70)
    for name, data in all_cell_results.items():
        r = data["regenerated"]
        print(f"{name:<20} {data['original_secure_rate']:>9.1f}% {r['secure_rate']:>11.1f}% "
              f"{r['n_secure']:>8} {r['n_insecure']:>10} {r['n_other']:>8}")
    print(f"\nReview the samples in: {txt_path}")


if __name__ == "__main__":
    main()
