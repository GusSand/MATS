#!/usr/bin/env python3
"""
Experiment 10, Step 3b: LOBO Alpha Extension for Python CWEs

Extend LOBO cross-validation to higher alphas.
Same protocol as 03_lobo_validation.py but with α ∈ {6, 7, 8, 10, 12, 15}.

We already have results for α ∈ {0, 1, 2, 3, 4, 5} — do NOT re-run those.

For each Python CWE (89, 78, 79):
  For each new alpha:
    Run full LOBO (7 folds × 15 prompts × 10 seeds)
    Record secure rate + "other" rate (coherence check)

Merges with prior LOBO results for a complete alpha curve.
Reuses: ModelLoader, SteeringGenerator from shared/; scorers from datasets/
"""
import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# ─── Paths ───────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
EXPERIMENT_DIR = Path("/home/paperspace/MATS/src/experiments/02-05_cross_cwe_steering")
DATASETS_DIR = EXPERIMENT_DIR / "datasets"
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"

sys.path.insert(0, str(EXPERIMENT_DIR / "shared"))
sys.path.insert(0, str(DATASETS_DIR))

from model_loader import ModelLoader
from steering_generator import SteeringGenerator

# Import scorers
from cwe89.scoring import score_cwe89
from cwe78.scoring import score_cwe78
from cwe79.scoring import score_cwe79

# ─── Configuration ───────────────────────────────────────────────────────────

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LAYER = 31
NEW_ALPHAS = [6.0, 7.0, 8.0, 10.0, 12.0, 15.0]
SEEDS = [42, 123, 456, 789, 1000, 1111, 2222, 3333, 4444, 5555]
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.6
TOP_P = 0.9

SCORERS = {
    "CWE-89": score_cwe89,
    "CWE-78": score_cwe78,
    "CWE-79": score_cwe79,
}

DATASET_FILES = {
    "CWE-89": DATASETS_DIR / "cwe89" / "data" / "cwe89_expanded_20260209_221808.jsonl",
    "CWE-78": DATASETS_DIR / "cwe78" / "data" / "cwe78_expanded_20260209_221808.jsonl",
    "CWE-79": DATASETS_DIR / "cwe79" / "data" / "cwe79_expanded_20260209_221808.jsonl",
}

# Prior results file to merge with
PRIOR_RESULTS_FILE = RESULTS_DIR / "lobo_full_20260210_080939.json"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def format_chat_prompt(tokenizer, code_prefix):
    """Format as Python code completion using Llama chat template."""
    user_message = ("Complete the following Python function. Only write the "
                    "function body, no explanation.\n\n" + code_prefix)
    messages = [{"role": "user", "content": user_message}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def compute_fold_direction(X, y, train_indices):
    """Compute mean-difference from training subset."""
    X_train = X[train_indices]
    y_train = y[train_indices]
    secure_mean = X_train[y_train == 1].mean(axis=0)
    insecure_mean = X_train[y_train == 0].mean(axis=0)
    return (secure_mean - insecure_mean).astype(np.float32)


def load_prior_results():
    """Load prior LOBO results (α ∈ {0..5}) for merging."""
    if not PRIOR_RESULTS_FILE.exists():
        print(f"  WARNING: Prior results not found at {PRIOR_RESULTS_FILE}")
        return None
    with open(PRIOR_RESULTS_FILE) as f:
        return json.load(f)


def print_combined_table(all_cwe_results, prior_results):
    """Print combined table with all alphas (old + new)."""
    all_alphas = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0] + NEW_ALPHAS

    print(f"\n{'='*100}")
    print("COMBINED ALPHA CURVE (prior + extension)")
    print(f"{'='*100}")

    # Header
    header = f"{'CWE':<8}"
    for a in all_alphas:
        header += f" {'α='+str(int(a)):>7}"
    header += f" {'Best α':>8} {'Other@Best':>10}"
    print(header)
    print("-" * len(header))

    for cwe in ["CWE-89", "CWE-78", "CWE-79"]:
        row = f"{cwe:<8}"
        best_alpha = None
        best_rate = -1

        for a in all_alphas:
            alpha_key = str(float(a))

            # Get from prior or new results
            if a < 6.0 and prior_results and cwe in prior_results:
                agg = prior_results[cwe]["aggregated"]
                if alpha_key in agg:
                    rate = agg[alpha_key]["secure_rate"]
                    other_rate = agg[alpha_key].get("n_other", 0) / agg[alpha_key]["n"] if agg[alpha_key]["n"] > 0 else 0
                else:
                    rate = None
                    other_rate = None
            elif cwe in all_cwe_results:
                agg = all_cwe_results[cwe]["aggregated"]
                if alpha_key in agg:
                    rate = agg[alpha_key]["secure_rate"]
                    other_rate = agg[alpha_key]["other_rate"]
                else:
                    rate = None
                    other_rate = None
            else:
                rate = None
                other_rate = None

            if rate is not None:
                row += f" {rate*100:>6.1f}%"
                if rate > best_rate:
                    best_rate = rate
                    best_alpha = a
                    best_other_rate = other_rate if other_rate is not None else 0
            else:
                row += f" {'N/A':>7}"

        row += f" {best_alpha:>8}" if best_alpha is not None else f" {'N/A':>8}"
        row += f" {best_other_rate*100:>9.1f}%" if best_alpha is not None else f" {'N/A':>10}"
        print(row)

    # Coherence table
    print(f"\n{'='*100}")
    print("COHERENCE CHECK: 'other' rate by alpha (higher = more incoherent)")
    print(f"{'='*100}")

    header = f"{'CWE':<8}"
    for a in all_alphas:
        header += f" {'α='+str(int(a)):>7}"
    print(header)
    print("-" * len(header))

    for cwe in ["CWE-89", "CWE-78", "CWE-79"]:
        row = f"{cwe:<8}"
        for a in all_alphas:
            alpha_key = str(float(a))

            if a < 6.0 and prior_results and cwe in prior_results:
                agg = prior_results[cwe]["aggregated"]
                if alpha_key in agg:
                    n_other = agg[alpha_key].get("n_other", 0)
                    n_total = agg[alpha_key]["n"]
                    rate = n_other / n_total if n_total > 0 else 0
                else:
                    rate = None
            elif cwe in all_cwe_results:
                agg = all_cwe_results[cwe]["aggregated"]
                if alpha_key in agg:
                    rate = agg[alpha_key]["other_rate"]
                else:
                    rate = None
            else:
                rate = None

            if rate is not None:
                row += f" {rate*100:>6.1f}%"
            else:
                row += f" {'N/A':>7}"
        print(row)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Experiment 10, Step 3b: LOBO Alpha Extension (Python CWEs)")
    print(f"Model: {MODEL_NAME}")
    print(f"Layer: {LAYER}")
    print(f"New Alphas: {NEW_ALPHAS}")
    print(f"Seeds: {len(SEEDS)}")
    print(f"Expected generations per CWE: {len(NEW_ALPHAS)} alphas × 7 folds × 15 prompts × {len(SEEDS)} seeds = {len(NEW_ALPHAS)*7*15*len(SEEDS)}")
    print("=" * 70)

    # Load prior results
    prior_results = load_prior_results()
    if prior_results:
        print(f"\nLoaded prior results from {PRIOR_RESULTS_FILE.name}")
        for cwe in ["CWE-89", "CWE-78", "CWE-79"]:
            if cwe in prior_results:
                print(f"  {cwe}: best α={prior_results[cwe]['best_alpha']}, "
                      f"secure={prior_results[cwe]['best_rate']*100:.1f}%")

    # ─── Load model ──────────────────────────────────────────────────────
    print("\nLoading model...")
    loader = ModelLoader(MODEL_NAME)
    tokenizer = loader.tokenizer
    generator = SteeringGenerator(loader)

    all_cwe_results = {}

    for cwe, dataset_path in DATASET_FILES.items():
        scorer = SCORERS[cwe]
        cwe_num = cwe.split("-")[1]

        print(f"\n{'='*70}")
        print(f"LOBO ALPHA EXTENSION: {cwe}")
        print(f"{'='*70}")

        # Load dataset
        dataset = load_jsonl(dataset_path)
        n = len(dataset)

        # Load pre-computed activations from Step 2
        act_files = sorted(DATA_DIR.glob(f"activations_cwe{cwe_num}_L{LAYER}_*.npz"))
        if not act_files:
            print(f"  ERROR: No activation file found for {cwe}. Run Step 2 first.")
            continue
        act_data = np.load(act_files[-1])
        X = act_data["X"]
        y = act_data["y"]
        base_ids_arr = act_data["base_ids"]
        print(f"  Loaded activations: X={X.shape}, y={y.shape}")

        # Get unique base_ids
        base_ids = sorted(set(base_ids_arr.tolist()))
        print(f"  Base IDs: {base_ids} ({len(base_ids)} folds)")

        # X layout: [insecure_0..insecure_N-1, secure_0..secure_N-1]
        fold_results = []

        for fold_idx, held_out_base in enumerate(base_ids):
            print(f"\n  --- Fold {fold_idx+1}/{len(base_ids)}: hold out base_id={held_out_base} ---")

            # Train/test split
            train_insecure = [i for i in range(n) if base_ids_arr[i] != held_out_base]
            train_secure = [i + n for i in range(n) if base_ids_arr[i] != held_out_base]
            test_indices = [i for i in range(n) if base_ids_arr[i] == held_out_base]

            train_indices = train_insecure + train_secure
            direction = compute_fold_direction(X, y, train_indices)
            print(f"    Train: {len(train_indices)} ({len(train_insecure)} insecure + {len(train_secure)} secure)")
            print(f"    Test: {len(test_indices)} prompts")
            print(f"    Direction norm: {np.linalg.norm(direction):.4f}")

            test_prompts = [dataset[i] for i in test_indices]

            fold_alpha_results = {}
            for alpha in NEW_ALPHAS:
                alpha_results = []

                for item in test_prompts:
                    formatted = format_chat_prompt(tokenizer, item["insecure_prompt"])

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

                        label = scorer(output)
                        alpha_results.append({
                            "pair_id": item["pair_id"],
                            "base_id": item["base_id"],
                            "seed": seed,
                            "label": label,
                            "output": output[:500],
                        })

                n_r = len(alpha_results)
                n_sec = sum(1 for r in alpha_results if r["label"] == "secure")
                n_ins = sum(1 for r in alpha_results if r["label"] == "insecure")
                n_oth = sum(1 for r in alpha_results if r["label"] == "other")

                fold_alpha_results[str(alpha)] = {
                    "n": n_r,
                    "n_secure": n_sec,
                    "n_insecure": n_ins,
                    "n_other": n_oth,
                    "secure_rate": n_sec / n_r if n_r > 0 else 0,
                    "insecure_rate": n_ins / n_r if n_r > 0 else 0,
                    "other_rate": n_oth / n_r if n_r > 0 else 0,
                }

                print(f"    α={alpha}: secure={n_sec}/{n_r} ({n_sec/n_r*100:.1f}%), "
                      f"insecure={n_ins}/{n_r} ({n_ins/n_r*100:.1f}%), "
                      f"other={n_oth}/{n_r} ({n_oth/n_r*100:.1f}%)")

                # Early coherence warning
                if n_oth / n_r > 0.5:
                    print(f"    ⚠ WARNING: >50% 'other' at α={alpha} — possible coherence degradation!")

            fold_results.append({
                "fold_id": held_out_base,
                "n_test": len(test_indices),
                "direction_norm": float(np.linalg.norm(direction)),
                "alpha_results": fold_alpha_results,
            })

        # ─── Aggregate across folds ──────────────────────────────────────
        print(f"\n  {'='*60}")
        print(f"  {cwe} ALPHA EXTENSION AGGREGATE")
        print(f"  {'='*60}")

        agg = {}
        for alpha in NEW_ALPHAS:
            alpha_key = str(alpha)
            total_n = sum(f["alpha_results"][alpha_key]["n"] for f in fold_results)
            total_sec = sum(f["alpha_results"][alpha_key]["n_secure"] for f in fold_results)
            total_ins = sum(f["alpha_results"][alpha_key]["n_insecure"] for f in fold_results)
            total_oth = sum(f["alpha_results"][alpha_key]["n_other"] for f in fold_results)

            agg[alpha_key] = {
                "n": total_n,
                "n_secure": total_sec,
                "n_insecure": total_ins,
                "n_other": total_oth,
                "secure_rate": total_sec / total_n if total_n > 0 else 0,
                "insecure_rate": total_ins / total_n if total_n > 0 else 0,
                "other_rate": total_oth / total_n if total_n > 0 else 0,
            }

            print(f"  α={alpha}: secure={total_sec}/{total_n} ({total_sec/total_n*100:.1f}%), "
                  f"insecure={total_ins}/{total_n} ({total_ins/total_n*100:.1f}%), "
                  f"other={total_oth}/{total_n} ({total_oth/total_n*100:.1f}%)")

        # Find best alpha across old + new
        best_alpha_new = max(NEW_ALPHAS, key=lambda a: agg[str(a)]["secure_rate"])
        best_rate_new = agg[str(best_alpha_new)]["secure_rate"]

        # Compare with prior best
        prior_best_rate = 0
        prior_best_alpha = None
        if prior_results and cwe in prior_results:
            prior_best_rate = prior_results[cwe]["best_rate"]
            prior_best_alpha = prior_results[cwe]["best_alpha"]

        if best_rate_new > prior_best_rate:
            print(f"\n  NEW BEST: α={best_alpha_new} ({best_rate_new*100:.1f}%) beats prior α={prior_best_alpha} ({prior_best_rate*100:.1f}%)")
        else:
            print(f"\n  Prior α={prior_best_alpha} ({prior_best_rate*100:.1f}%) still best. "
                  f"New best: α={best_alpha_new} ({best_rate_new*100:.1f}%)")

        # Check monotonicity
        rates = [agg[str(a)]["secure_rate"] for a in NEW_ALPHAS]
        if all(rates[i] <= rates[i+1] for i in range(len(rates)-1)):
            print(f"  NOTE: Still monotonically increasing at α={NEW_ALPHAS[-1]} — may need even higher alphas")
        elif all(rates[i] >= rates[i+1] for i in range(len(rates)-1)):
            print(f"  NOTE: Monotonically decreasing from α={NEW_ALPHAS[0]} — sweet spot is at or below α=5")

        all_cwe_results[cwe] = {
            "fold_results": fold_results,
            "aggregated": agg,
            "best_alpha_new": best_alpha_new,
            "best_rate_new": best_rate_new,
        }

    # ─── Combined table ──────────────────────────────────────────────────
    print_combined_table(all_cwe_results, prior_results)

    # ─── Save new results ────────────────────────────────────────────────
    output_data = {
        "timestamp": timestamp,
        "experiment": "03b_lobo_alpha_extension",
        "model": MODEL_NAME,
        "layer": LAYER,
        "new_alpha_grid": NEW_ALPHAS,
        "prior_alpha_grid": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
        "generation_config": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": True,
        },
        "cwe_results": {
            cwe: {
                "aggregated": r["aggregated"],
                "best_alpha_new": r["best_alpha_new"],
                "best_rate_new": r["best_rate_new"],
            }
            for cwe, r in all_cwe_results.items()
        },
    }

    results_path = RESULTS_DIR / f"alpha_extension_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nNew alpha results saved: {results_path}")

    # Save full fold data
    full_path = RESULTS_DIR / f"alpha_extension_full_{timestamp}.json"
    with open(full_path, "w") as f:
        json.dump(all_cwe_results, f, indent=2, default=str)
    print(f"Full results saved: {full_path}")

    # ─── Save merged results (all alphas combined) ───────────────────────
    merged = {}
    for cwe in ["CWE-89", "CWE-78", "CWE-79"]:
        merged_agg = {}

        # Prior alphas
        if prior_results and cwe in prior_results:
            for alpha_key, data in prior_results[cwe]["aggregated"].items():
                # Add other_rate if missing from prior
                if "other_rate" not in data:
                    data["other_rate"] = data.get("n_other", 0) / data["n"] if data["n"] > 0 else 0
                merged_agg[alpha_key] = data

        # New alphas
        if cwe in all_cwe_results:
            for alpha_key, data in all_cwe_results[cwe]["aggregated"].items():
                merged_agg[alpha_key] = data

        # Find overall best
        all_alphas_for_cwe = sorted([float(k) for k in merged_agg.keys()])
        best_alpha = max(all_alphas_for_cwe, key=lambda a: merged_agg[str(a)]["secure_rate"])

        merged[cwe] = {
            "aggregated": merged_agg,
            "all_alphas": all_alphas_for_cwe,
            "best_alpha": best_alpha,
            "best_rate": merged_agg[str(best_alpha)]["secure_rate"],
            "best_other_rate": merged_agg[str(best_alpha)]["other_rate"],
        }

    merged_path = RESULTS_DIR / f"alpha_curve_merged_{timestamp}.json"
    with open(merged_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Merged alpha curve saved: {merged_path}")

    loader.unload()
    print("\nAlpha extension complete.")


if __name__ == "__main__":
    main()
