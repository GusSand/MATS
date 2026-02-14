#!/usr/bin/env python3
"""
Experiment 11: CWE-134 Full LOBO with Extended Alpha Sweep

The pilot LOBO (Exp 5) only ran 2 folds with alpha up to 5.0 and 1 gen per prompt.
This runs the full 7-fold LOBO with alpha up to 10.0 and 10 seeds per prompt
to determine whether higher alpha can push CWE-134 steering above 0%.

Reuses: ModelLoader, SteeringGenerator, ActivationCollector, scoring from shared/
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
RESULTS_DIR = SCRIPT_DIR / "results"
DATA_DIR = SCRIPT_DIR / "data"

sys.path.insert(0, str(EXPERIMENT_DIR / "shared"))
sys.path.insert(0, str(EXPERIMENT_DIR / "experiment_cwe134_llama8b"))

from model_loader import ModelLoader
from steering_generator import SteeringGenerator
from activation_collector import ActivationCollector
from scoring import score_completion, detect_refusal
import experiment_config as config_134

# ─── Configuration ───────────────────────────────────────────────────────────

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LAYER = 31
ALPHA_GRID = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
SEEDS = [42, 123, 456]  # 3 seeds initially; follow up with more if alpha curve is interesting
MAX_NEW_TOKENS = 256  # Printf functions are ~20 tokens; 256 is more than enough
TEMPERATURE = 0.6
TOP_P = 0.9

DATASET_PATH = DATASETS_DIR / "cwe134" / "data" / "cwe134_expanded_20260205_151207.jsonl"

BASE_IDS = config_134.BASE_IDS  # 7 base_ids


# ─── Helpers ─────────────────────────────────────────────────────────────────

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


def score_output(output, vuln_type):
    """Score a CWE-134 output."""
    result = score_completion(
        output, vuln_type,
        strict_patterns=config_134.STRICT_PATTERNS,
        expanded_secure_additions=config_134.EXPANDED_SECURE_ADDITIONS,
        bounds_check_patterns=config_134.BOUNDS_CHECK_PATTERNS,
    )
    is_refusal, _ = detect_refusal(
        output,
        c_code_indicators=config_134.C_CODE_INDICATORS,
        refusal_patterns=config_134.REFUSAL_PATTERNS,
    )
    return result.strict_label, result.expanded_label, is_refusal


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Experiment 11: CWE-134 Full LOBO with Extended Alpha Sweep")
    print(f"Model: {MODEL_NAME}")
    print(f"Layer: {LAYER}")
    print(f"Alphas: {ALPHA_GRID}")
    print(f"Seeds: {len(SEEDS)} per prompt")
    print(f"Base IDs: {len(BASE_IDS)} folds")
    print("=" * 70)

    # ─── Load dataset ─────────────────────────────────────────────────────
    dataset = load_jsonl(DATASET_PATH)
    n = len(dataset)
    print(f"\nDataset: {n} pairs from {DATASET_PATH.name}")

    # Build base_id index
    base_ids_arr = np.array([item["base_id"] for item in dataset])
    print(f"Base IDs: {sorted(set(base_ids_arr))}")

    # ─── Load model ───────────────────────────────────────────────────────
    print("\nLoading model...")
    loader = ModelLoader(MODEL_NAME)
    tokenizer = loader.tokenizer
    generator = SteeringGenerator(loader)

    # ─── Step 1: Collect Layer-31 activations ─────────────────────────────
    # Check if activations already exist
    act_files = sorted(DATA_DIR.glob("activations_cwe134_L31_*.npz"))
    if act_files:
        print(f"\nLoading existing activations: {act_files[-1]}")
        act_data = np.load(act_files[-1])
        X = act_data["X"]
        y = act_data["y"]
    else:
        print("\nCollecting Layer 31 activations...")
        collector = ActivationCollector(loader)

        # Collect at Layer 31 only
        X_all = np.zeros((2 * n, loader.hidden_size), dtype=np.float16)
        y = np.array([0] * n + [1] * n)

        # Hook only layer 31
        def make_hook(storage, idx):
            def hook_fn(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                storage[idx] = h[:, -1, :].detach().cpu().numpy().astype(np.float16)
            return hook_fn

        layer_module = loader.model.model.layers[LAYER]

        # Collect insecure activations
        print("  Collecting insecure prompt activations...")
        for i, item in enumerate(tqdm(dataset, desc="Insecure")):
            storage = {}
            hook = layer_module.register_forward_hook(make_hook(storage, 0))
            inputs = tokenizer(item["vulnerable"], return_tensors="pt").to(loader.device)
            with torch.no_grad():
                _ = loader.model(**inputs)
            hook.remove()
            X_all[i] = storage[0]

        # Collect secure activations
        print("  Collecting secure prompt activations...")
        for i, item in enumerate(tqdm(dataset, desc="Secure")):
            storage = {}
            hook = layer_module.register_forward_hook(make_hook(storage, 0))
            inputs = tokenizer(item["secure"], return_tensors="pt").to(loader.device)
            with torch.no_grad():
                _ = loader.model(**inputs)
            hook.remove()
            X_all[n + i] = storage[0]

        X = X_all.astype(np.float32)

        # Save activations for reuse
        act_path = DATA_DIR / f"activations_cwe134_L31_{timestamp}.npz"
        np.savez_compressed(act_path, X=X, y=y, base_ids=base_ids_arr)
        print(f"  Saved activations: {act_path}")

    print(f"  X shape: {X.shape}, y shape: {y.shape}")

    # ─── Step 2: Full 7-Fold LOBO ────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FULL 7-FOLD LOBO")
    print(f"{'='*70}")

    all_fold_results = []

    for fold_idx, held_out_base in enumerate(BASE_IDS):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx+1}/{len(BASE_IDS)}: hold out {held_out_base}")
        print(f"{'='*60}")

        # Train/test split
        train_insecure = [i for i in range(n) if base_ids_arr[i] != held_out_base]
        train_secure = [i + n for i in range(n) if base_ids_arr[i] != held_out_base]
        test_indices = [i for i in range(n) if base_ids_arr[i] == held_out_base]

        train_indices = train_insecure + train_secure
        direction = compute_fold_direction(X, y, train_indices)

        test_prompts = [dataset[i] for i in test_indices]
        print(f"  Train: {len(train_indices)} ({len(train_insecure)} insec + {len(train_secure)} sec)")
        print(f"  Test: {len(test_indices)} prompts × {len(SEEDS)} seeds = {len(test_indices)*len(SEEDS)} gens/alpha")
        print(f"  Direction norm: {np.linalg.norm(direction):.4f}")

        fold_alpha_results = {}

        for alpha in ALPHA_GRID:
            alpha_results = []

            desc = f"Fold {fold_idx+1} α={alpha}"
            for item in tqdm(test_prompts, desc=desc, leave=False):
                prompt = item["vulnerable"]
                vuln_type = item["vulnerability_type"]

                for seed in SEEDS:
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)

                    if alpha == 0.0:
                        output = generator.generate_baseline(
                            prompt=prompt,
                            temperature=TEMPERATURE,
                            top_p=TOP_P,
                            max_tokens=MAX_NEW_TOKENS,
                        )
                    else:
                        output = generator.generate_with_steering(
                            prompt=prompt,
                            direction=direction,
                            layer=LAYER,
                            alpha=alpha,
                            temperature=TEMPERATURE,
                            top_p=TOP_P,
                            max_tokens=MAX_NEW_TOKENS,
                        )

                    strict_label, expanded_label, is_refusal = score_output(output, vuln_type)

                    alpha_results.append({
                        "id": item["id"],
                        "base_id": item["base_id"],
                        "seed": seed,
                        "strict_label": strict_label,
                        "expanded_label": expanded_label,
                        "is_refusal": is_refusal,
                        "output": output[:500],
                    })

            n_r = len(alpha_results)
            n_sec = sum(1 for r in alpha_results if r["strict_label"] == "secure")
            n_ins = sum(1 for r in alpha_results if r["strict_label"] == "insecure")
            n_oth = n_r - n_sec - n_ins
            n_ref = sum(1 for r in alpha_results if r["is_refusal"])
            n_exp = sum(1 for r in alpha_results if r["expanded_label"] == "secure")

            fold_alpha_results[str(alpha)] = {
                "n": n_r,
                "n_secure": n_sec,
                "n_insecure": n_ins,
                "n_other": n_oth,
                "n_refusal": n_ref,
                "n_expanded_secure": n_exp,
                "secure_rate": n_sec / n_r if n_r > 0 else 0,
                "insecure_rate": n_ins / n_r if n_r > 0 else 0,
                "refusal_rate": n_ref / n_r if n_r > 0 else 0,
                "results": alpha_results,
            }

            print(f"  α={alpha:>5.1f}: secure={n_sec:>3}/{n_r} ({n_sec/n_r*100:>5.1f}%), "
                  f"insecure={n_ins:>3}/{n_r} ({n_ins/n_r*100:>5.1f}%), "
                  f"other={n_oth:>3}, refusal={n_ref:>3}")

        all_fold_results.append({
            "fold_id": held_out_base,
            "fold_idx": fold_idx,
            "n_test": len(test_indices),
            "direction_norm": float(np.linalg.norm(direction)),
            "alpha_results": fold_alpha_results,
        })

        # Save fold results incrementally
        fold_path = RESULTS_DIR / f"fold_{held_out_base}_{timestamp}.json"
        # Save without per-generation outputs for the fold file (save space)
        fold_save = {
            "fold_id": held_out_base,
            "direction_norm": float(np.linalg.norm(direction)),
            "n_test": len(test_indices),
            "alpha_summary": {
                ak: {k: v for k, v in av.items() if k != "results"}
                for ak, av in fold_alpha_results.items()
            },
        }
        with open(fold_path, "w") as f:
            json.dump(fold_save, f, indent=2)

    # ─── Aggregate across folds ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("AGGREGATED RESULTS")
    print(f"{'='*70}")

    aggregated = {}
    for alpha in ALPHA_GRID:
        alpha_key = str(alpha)
        total_n = sum(f["alpha_results"][alpha_key]["n"] for f in all_fold_results)
        total_sec = sum(f["alpha_results"][alpha_key]["n_secure"] for f in all_fold_results)
        total_ins = sum(f["alpha_results"][alpha_key]["n_insecure"] for f in all_fold_results)
        total_oth = sum(f["alpha_results"][alpha_key]["n_other"] for f in all_fold_results)
        total_ref = sum(f["alpha_results"][alpha_key]["n_refusal"] for f in all_fold_results)
        total_exp = sum(f["alpha_results"][alpha_key]["n_expanded_secure"] for f in all_fold_results)

        aggregated[alpha_key] = {
            "n": total_n,
            "n_secure": total_sec,
            "n_insecure": total_ins,
            "n_other": total_oth,
            "n_refusal": total_ref,
            "n_expanded_secure": total_exp,
            "strict_secure_rate": total_sec / total_n if total_n > 0 else 0,
            "strict_insecure_rate": total_ins / total_n if total_n > 0 else 0,
            "expanded_secure_rate": total_exp / total_n if total_n > 0 else 0,
            "refusal_rate": total_ref / total_n if total_n > 0 else 0,
        }

    # Print summary table
    print(f"\n{'Alpha':<8} {'Secure%':>10} {'Insecure%':>10} {'Other%':>10} {'Refusal%':>10} {'N':>6}")
    print("-" * 58)

    baseline_rate = 0.0
    best_rate = 0.0
    best_alpha = 0.0

    for alpha in ALPHA_GRID:
        r = aggregated[str(alpha)]
        rate = r["strict_secure_rate"]
        if alpha == 0.0:
            baseline_rate = rate
        if rate > best_rate:
            best_rate = rate
            best_alpha = alpha
        other_rate = r["n_other"] / r["n"] if r["n"] > 0 else 0
        print(f"{alpha:<8} {rate*100:>9.1f}% {r['strict_insecure_rate']*100:>9.1f}% "
              f"{other_rate*100:>9.1f}% {r['refusal_rate']*100:>9.1f}% {r['n']:>6}")

    improvement = best_rate - baseline_rate
    print(f"\nBaseline (α=0): {baseline_rate*100:.1f}%")
    print(f"Best: {best_rate*100:.1f}% at α={best_alpha}")
    print(f"Improvement: {improvement*100:+.1f}pp")

    # ─── Per-fold breakdown at best alpha ────────────────────────────────
    print(f"\nPer-fold at best α={best_alpha}:")
    for fr in all_fold_results:
        r = fr["alpha_results"][str(best_alpha)]
        print(f"  {fr['fold_id']}: {r['n_secure']}/{r['n']} "
              f"({r['secure_rate']*100:.1f}%) secure")

    # ─── Save results ────────────────────────────────────────────────────
    output_data = {
        "timestamp": timestamp,
        "experiment": "Experiment 11: CWE-134 Full LOBO Extended Alpha",
        "model": MODEL_NAME,
        "layer": LAYER,
        "alpha_grid": ALPHA_GRID,
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
        "n_folds": len(BASE_IDS),
        "base_ids": BASE_IDS,
        "generation_config": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": True,
        },
        "aggregated": aggregated,
        "baseline_rate": baseline_rate,
        "best_rate": best_rate,
        "best_alpha": best_alpha,
        "improvement_pp": improvement * 100,
        "per_fold_summary": [
            {
                "fold_id": fr["fold_id"],
                "direction_norm": fr["direction_norm"],
                "n_test": fr["n_test"],
                "best_alpha_results": {
                    k: v for k, v in fr["alpha_results"][str(best_alpha)].items()
                    if k != "results"
                },
            }
            for fr in all_fold_results
        ],
    }

    results_path = RESULTS_DIR / f"c134_full_lobo_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # Save full data with outputs (for detailed analysis)
    full_path = RESULTS_DIR / f"c134_full_lobo_full_{timestamp}.json"
    full_data = {
        "timestamp": timestamp,
        "config": output_data,
        "fold_results": [
            {
                "fold_id": fr["fold_id"],
                "direction_norm": fr["direction_norm"],
                "alpha_results": {
                    ak: {k: v for k, v in av.items() if k != "results"}
                    for ak, av in fr["alpha_results"].items()
                },
            }
            for fr in all_fold_results
        ],
    }
    with open(full_path, "w") as f:
        json.dump(full_data, f, indent=2)
    print(f"Full results saved: {full_path}")

    # Also save to Exp 10 results dir for the transfer matrix update
    exp10_path = Path("/home/paperspace/MATS/src/experiments/02-10_python_cwe_steering/results")
    exp10_results = exp10_path / f"c134_full_lobo_{timestamp}.json"
    with open(exp10_results, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Also saved to Exp 10: {exp10_results}")

    loader.unload()
    print("\nDone.")


if __name__ == "__main__":
    main()
