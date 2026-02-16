#!/usr/bin/env python3
"""
Experiment 16 (OPTIONAL): Qwen2.5-14B-Instruct CWE-89 LOBO Cross-Validation

Goal: Add a third architecture to cross-language steering validation.
We already have Qwen-14B CWE-787 LOBO (77.1% at alpha=4.0) from Exp 4.

Model: Qwen/Qwen2.5-14B-Instruct (fp16)
Steering layer: 47 (last hidden layer, confirmed in Exp 4)
Dataset: CWE-89 expanded (105 pairs, 7 base_ids)
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
from cwe89.scoring import score_cwe89

# ─── Configuration ───────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
LAYER = 47  # Last hidden layer for Qwen-14B (48 layers total)
ALPHA_GRID = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
SEEDS = [42, 123, 456]
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.6
TOP_P = 0.9

DATASET_PATH = DATASETS_DIR / "cwe89" / "data" / "cwe89_expanded_20260209_221808.jsonl"

BASE_IDS = [
    "admin_delete",
    "log_entry",
    "order_history",
    "product_search",
    "report_filter",
    "user_login",
    "user_profile_update",
]

# Reference results
LLAMA_REFERENCE = {
    "baseline_rate": 0.570,
    "best_alpha": 5.0,
    "best_rate": 0.703,
    "improvement_pp": 13.3,
}
# Mistral reference will be filled from Exp 13 if available
MISTRAL_REFERENCE = None


# ─── Helpers ─────────────────────────────────────────────────────────────────

def format_chat_prompt(tokenizer, code_prefix):
    """Format as Python code completion using Qwen chat template."""
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
    X_train = X[train_indices]
    y_train = y[train_indices]
    secure_mean = X_train[y_train == 1].mean(axis=0)
    insecure_mean = X_train[y_train == 0].mean(axis=0)
    return (secure_mean - insecure_mean).astype(np.float32)


@torch.no_grad()
def collect_activations(model, tokenizer, prompts, layer):
    activations = []
    captured = {}

    def hook_fn(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["act"] = h[:, -1, :].detach().cpu()

    hook = model.model.layers[layer].register_forward_hook(hook_fn)

    for prompt in tqdm(prompts, desc="Collecting activations"):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        _ = model(**inputs)
        activations.append(captured["act"].numpy().astype(np.float32))

    hook.remove()
    return np.vstack(activations)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 16: Qwen2.5-14B-Instruct CWE-89 LOBO (Third Architecture)")
    print(f"Model: {MODEL_NAME}")
    print(f"Layer: {LAYER}")
    print(f"Alphas: {ALPHA_GRID}")
    print(f"Seeds: {SEEDS}")
    print("=" * 70)

    # Try to load Mistral reference from Exp 13
    global MISTRAL_REFERENCE
    exp13_results = sorted(Path("/home/paperspace/MATS/src/experiments/02-16_mistral_cwe89_lobo/results/").glob("lobo_results_*.json"))
    if exp13_results:
        with open(exp13_results[-1]) as f:
            exp13 = json.load(f)
        MISTRAL_REFERENCE = {
            "baseline_rate": exp13.get("baseline_rate", None),
            "best_alpha": exp13.get("best_alpha", None),
            "best_rate": exp13.get("best_rate", None),
            "improvement_pp": exp13.get("improvement_pp", None),
        }
        print(f"  Loaded Mistral reference: baseline={MISTRAL_REFERENCE['baseline_rate']:.1%}, "
              f"best={MISTRAL_REFERENCE['best_rate']:.1%}")

    # ─── Load dataset ─────────────────────────────────────────────────────
    dataset = load_jsonl(DATASET_PATH)
    n = len(dataset)
    print(f"\nLoaded {n} prompt pairs")

    # ─── Load model ───────────────────────────────────────────────────────
    print("\nLoading model...")
    loader = ModelLoader(MODEL_NAME)
    model = loader.model
    tokenizer = loader.tokenizer
    device = loader.device

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1: Activation Extraction
    # ═══════════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print("PHASE 1: Activation Extraction")
    print(f"{'='*70}")

    insecure_prompts = [format_chat_prompt(tokenizer, item["insecure_prompt"])
                        for item in dataset]
    secure_prompts = [format_chat_prompt(tokenizer, item["secure_prompt"])
                      for item in dataset]

    print(f"\nCollecting insecure prompt activations...")
    X_insecure = collect_activations(model, tokenizer, insecure_prompts, LAYER)

    print(f"Collecting secure prompt activations...")
    X_secure = collect_activations(model, tokenizer, secure_prompts, LAYER)

    X = np.vstack([X_insecure, X_secure])
    y = np.array([0] * n + [1] * n)
    base_ids_arr = np.array([item["base_id"] for item in dataset])

    npz_path = DATA_DIR / f"activations_qwen14b_cwe89_L{LAYER}.npz"
    np.savez_compressed(npz_path, X=X, y=y, base_ids=base_ids_arr)
    print(f"\nActivations saved: {npz_path} ({npz_path.stat().st_size / 1e6:.1f} MB)")

    overall_direction = (X_secure.mean(axis=0) - X_insecure.mean(axis=0)).astype(np.float32)
    print(f"Overall direction norm: {np.linalg.norm(overall_direction):.4f}")

    npy_path = DATA_DIR / f"direction_qwen14b_cwe89_L{LAYER}_{timestamp}.npy"
    np.save(npy_path, overall_direction)

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2: LOBO Cross-Validation
    # ═══════════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print("PHASE 2: LOBO Cross-Validation (7 folds)")
    print(f"{'='*70}")

    total_gens = len(BASE_IDS) * 15 * len(ALPHA_GRID) * len(SEEDS)
    print(f"Total generations: {total_gens}")

    fold_results = []

    for fold_idx, held_out_base in enumerate(BASE_IDS):
        print(f"\n  --- Fold {fold_idx+1}/{len(BASE_IDS)}: hold out {held_out_base} ---")

        train_insecure = [i for i in range(n) if base_ids_arr[i] != held_out_base]
        train_secure = [i + n for i in range(n) if base_ids_arr[i] != held_out_base]
        test_indices = [i for i in range(n) if base_ids_arr[i] == held_out_base]

        train_indices = train_insecure + train_secure
        direction = compute_fold_direction(X, y, train_indices)
        direction_tensor = torch.tensor(direction, dtype=torch.float16).to(device)

        print(f"    Train: {len(train_indices)}, Test: {len(test_indices)}")
        print(f"    Direction norm: {np.linalg.norm(direction):.4f}")

        test_prompts = [dataset[i] for i in test_indices]

        fold_alpha_results = {}
        for alpha in ALPHA_GRID:
            alpha_results = []

            for item in test_prompts:
                formatted = format_chat_prompt(tokenizer, item["insecure_prompt"])
                input_ids = tokenizer(formatted, return_tensors="pt").to(device)
                input_len = input_ids.input_ids.shape[1]

                for seed in SEEDS:
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)

                    if alpha == 0.0:
                        with torch.no_grad():
                            outputs = model.generate(
                                **input_ids,
                                max_new_tokens=MAX_NEW_TOKENS,
                                temperature=TEMPERATURE,
                                do_sample=True,
                                top_p=TOP_P,
                                pad_token_id=tokenizer.pad_token_id,
                            )
                    else:
                        def make_hook(a, v):
                            def fn(module, input, output):
                                h = output[0] if isinstance(output, tuple) else output
                                h[:, -1, :] = h[:, -1, :] + a * v
                                if isinstance(output, tuple):
                                    return (h,) + output[1:]
                                return h
                            return fn

                        hook = model.model.layers[LAYER].register_forward_hook(
                            make_hook(alpha, direction_tensor)
                        )
                        with torch.no_grad():
                            outputs = model.generate(
                                **input_ids,
                                max_new_tokens=MAX_NEW_TOKENS,
                                temperature=TEMPERATURE,
                                do_sample=True,
                                top_p=TOP_P,
                                pad_token_id=tokenizer.pad_token_id,
                            )
                        hook.remove()

                    new_tokens = outputs[0][input_len:]
                    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

                    label = score_cwe89(output_text)
                    alpha_results.append({
                        "pair_id": item["pair_id"],
                        "base_id": item["base_id"],
                        "seed": seed,
                        "label": label,
                        "output": output_text[:500],
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
                "details": alpha_results,
            }

            print(f"    alpha={alpha}: secure={n_sec}/{n_r} ({n_sec/n_r*100:.1f}%), "
                  f"insecure={n_ins}/{n_r} ({n_ins/n_r*100:.1f}%)")

        fold_results.append({
            "fold_id": held_out_base,
            "n_test": len(test_indices),
            "direction_norm": float(np.linalg.norm(direction)),
            "alpha_results": fold_alpha_results,
        })

        fold_path = RESULTS_DIR / f"fold_{held_out_base}_{timestamp}.json"
        with open(fold_path, "w") as f:
            json.dump(fold_results[-1], f, indent=2)

    # ─── Aggregate ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("LOBO AGGREGATE RESULTS")
    print(f"{'='*70}")

    agg = {}
    for alpha in ALPHA_GRID:
        ak = str(alpha)
        total_n = sum(f["alpha_results"][ak]["n"] for f in fold_results)
        total_sec = sum(f["alpha_results"][ak]["n_secure"] for f in fold_results)
        total_ins = sum(f["alpha_results"][ak]["n_insecure"] for f in fold_results)
        total_oth = sum(f["alpha_results"][ak]["n_other"] for f in fold_results)

        agg[ak] = {
            "n": total_n,
            "n_secure": total_sec,
            "n_insecure": total_ins,
            "n_other": total_oth,
            "secure_rate": total_sec / total_n if total_n > 0 else 0,
            "insecure_rate": total_ins / total_n if total_n > 0 else 0,
        }

        print(f"  alpha={alpha}: secure={total_sec}/{total_n} ({total_sec/total_n*100:.1f}%), "
              f"insecure={total_ins}/{total_n} ({total_ins/total_n*100:.1f}%)")

    baseline_rate = agg["0.0"]["secure_rate"]
    best_alpha = max(ALPHA_GRID, key=lambda a: agg[str(a)]["secure_rate"])
    best_rate = agg[str(best_alpha)]["secure_rate"]
    improvement = best_rate - baseline_rate

    print(f"\n  Baseline: {baseline_rate*100:.1f}%")
    print(f"  Best: {best_rate*100:.1f}% at alpha={best_alpha}")
    print(f"  Improvement: {improvement*100:.1f}pp")

    # ═══════════════════════════════════════════════════════════════════════
    # 3-Way Comparison
    # ═══════════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print("3-WAY ARCHITECTURE COMPARISON: CWE-89")
    print(f"{'='*70}")

    print(f"\n{'Model':<20} {'Baseline':>10} {'Best alpha':>12} {'Best Rate':>12} {'Delta':>8}")
    print("-" * 65)
    print(f"{'Llama-8B':<20} {LLAMA_REFERENCE['baseline_rate']*100:>9.1f}% "
          f"{LLAMA_REFERENCE['best_alpha']:>12} {LLAMA_REFERENCE['best_rate']*100:>11.1f}% "
          f"{LLAMA_REFERENCE['improvement_pp']:>+7.1f}")
    if MISTRAL_REFERENCE and MISTRAL_REFERENCE['baseline_rate'] is not None:
        print(f"{'Mistral-7B':<20} {MISTRAL_REFERENCE['baseline_rate']*100:>9.1f}% "
              f"{MISTRAL_REFERENCE['best_alpha']:>12} {MISTRAL_REFERENCE['best_rate']*100:>11.1f}% "
              f"{MISTRAL_REFERENCE['improvement_pp']:>+7.1f}")
    print(f"{'Qwen-14B':<20} {baseline_rate*100:>9.1f}% "
          f"{best_alpha:>12} {best_rate*100:>11.1f}% "
          f"{improvement*100:>+7.1f}")

    # ─── Save results ─────────────────────────────────────────────────────
    output_data = {
        "timestamp": timestamp,
        "experiment": "16_qwen14b_cwe89_lobo",
        "model": MODEL_NAME,
        "layer": LAYER,
        "alpha_grid": ALPHA_GRID,
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
        "n_folds": len(BASE_IDS),
        "base_ids": BASE_IDS,
        "aggregated": agg,
        "baseline_rate": baseline_rate,
        "best_alpha": best_alpha,
        "best_rate": best_rate,
        "improvement_pp": improvement * 100,
        "llama_reference": LLAMA_REFERENCE,
        "mistral_reference": MISTRAL_REFERENCE,
        "fold_summaries": [
            {
                "fold_id": f["fold_id"],
                "n_test": f["n_test"],
                "direction_norm": f["direction_norm"],
                "alpha_results": {
                    k: {key: val for key, val in v.items() if key != "details"}
                    for k, v in f["alpha_results"].items()
                },
            }
            for f in fold_results
        ],
    }

    results_path = RESULTS_DIR / f"lobo_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved: {results_path}")

    full_path = RESULTS_DIR / f"lobo_full_{timestamp}.json"
    with open(full_path, "w") as f:
        json.dump(fold_results, f, indent=2, default=str)
    print(f"Full results saved: {full_path}")

    loader.unload()
    print("\nExperiment 16 complete.")


if __name__ == "__main__":
    main()
