#!/usr/bin/env python3
"""
Experiment 10, Step 3: LOBO Validation for Python CWEs

Leave-One-Base-Out cross-validation for each Python CWE:
  - For each CWE: hold out 1 of 7 base_ids (15 prompts), train on remaining 6 (90 prompts)
  - Compute mean-difference vector from training set
  - Generate steered completions on held-out test prompts
  - Alpha sweep: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
  - Score with per-CWE regex scorers
  - 10 seeds per prompt

Reports per-CWE best alpha and secure rate.
Reuses: ModelLoader, SteeringGenerator from shared/; scorers from datasets/
"""
import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

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
ALPHA_GRID = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
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


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Experiment 10, Step 3: LOBO Validation (Python CWEs)")
    print(f"Model: {MODEL_NAME}")
    print(f"Layer: {LAYER}")
    print(f"Alphas: {ALPHA_GRID}")
    print(f"Seeds: {len(SEEDS)}")
    print("=" * 70)

    # ─── Load model ──────────────────────────────────────────────────────
    print("\nLoading model...")
    loader = ModelLoader(MODEL_NAME)
    model = loader.model
    tokenizer = loader.tokenizer
    generator = SteeringGenerator(loader)

    all_cwe_results = {}

    for cwe, dataset_path in DATASET_FILES.items():
        scorer = SCORERS[cwe]
        cwe_num = cwe.split("-")[1]

        print(f"\n{'='*70}")
        print(f"LOBO: {cwe}")
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

        # Map base_id to dataset indices
        # X layout: [insecure_0..insecure_N-1, secure_0..secure_N-1]
        # base_ids_arr has length N (one per pair)
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
            for alpha in ALPHA_GRID:
                alpha_results = []

                for item in test_prompts:
                    formatted = format_chat_prompt(tokenizer, item["insecure_prompt"])

                    for seed in SEEDS:
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed(seed)

                        if alpha == 0.0:
                            output = generator.generate_baseline(
                                prompt=formatted,
                                temperature=TEMPERATURE,
                                top_p=TOP_P,
                                max_tokens=MAX_NEW_TOKENS,
                            )
                        else:
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
                }

                print(f"    α={alpha}: secure={n_sec}/{n_r} ({n_sec/n_r*100:.1f}%), "
                      f"insecure={n_ins}/{n_r} ({n_ins/n_r*100:.1f}%)")

            fold_results.append({
                "fold_id": held_out_base,
                "n_test": len(test_indices),
                "direction_norm": float(np.linalg.norm(direction)),
                "alpha_results": fold_alpha_results,
            })

        # ─── Aggregate across folds ──────────────────────────────────────
        print(f"\n  {'='*60}")
        print(f"  {cwe} LOBO AGGREGATE")
        print(f"  {'='*60}")

        agg = {}
        for alpha in ALPHA_GRID:
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
            }

            print(f"  α={alpha}: secure={total_sec}/{total_n} ({total_sec/total_n*100:.1f}%), "
                  f"insecure={total_ins}/{total_n} ({total_ins/total_n*100:.1f}%)")

        baseline_rate = agg["0.0"]["secure_rate"]
        best_alpha = max(ALPHA_GRID, key=lambda a: agg[str(a)]["secure_rate"])
        best_rate = agg[str(best_alpha)]["secure_rate"]
        improvement = best_rate - baseline_rate

        print(f"\n  Baseline: {baseline_rate*100:.1f}%")
        print(f"  Best: {best_rate*100:.1f}% at α={best_alpha}")
        print(f"  Improvement: {improvement*100:.1f}pp")

        if improvement > 10:
            print(f"  *** GOOD TARGET: {improvement*100:.1f}pp improvement ***")

        all_cwe_results[cwe] = {
            "fold_results": fold_results,
            "aggregated": agg,
            "baseline_rate": baseline_rate,
            "best_alpha": best_alpha,
            "best_rate": best_rate,
            "improvement_pp": improvement * 100,
        }

    # ─── Overall summary ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("LOBO RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'CWE':<10} {'Baseline':>10} {'Best α':>8} {'Best Rate':>10} {'Δpp':>8}")
    print("-" * 48)

    for cwe, r in all_cwe_results.items():
        print(f"{cwe:<10} {r['baseline_rate']*100:>9.1f}% {r['best_alpha']:>8} "
              f"{r['best_rate']*100:>9.1f}% {r['improvement_pp']:>+7.1f}")

    # ─── Save ────────────────────────────────────────────────────────────
    output_data = {
        "timestamp": timestamp,
        "model": MODEL_NAME,
        "layer": LAYER,
        "alpha_grid": ALPHA_GRID,
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
                "baseline_rate": r["baseline_rate"],
                "best_alpha": r["best_alpha"],
                "best_rate": r["best_rate"],
                "improvement_pp": r["improvement_pp"],
            }
            for cwe, r in all_cwe_results.items()
        },
    }

    results_path = RESULTS_DIR / f"lobo_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # Save full fold data
    full_path = RESULTS_DIR / f"lobo_full_{timestamp}.json"
    with open(full_path, "w") as f:
        json.dump(all_cwe_results, f, indent=2, default=str)
    print(f"Full results saved: {full_path}")

    loader.unload()
    print("\nLOBO complete.")


if __name__ == "__main__":
    main()
