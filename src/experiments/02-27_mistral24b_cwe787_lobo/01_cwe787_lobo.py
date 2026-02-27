#!/usr/bin/env python3
"""
Mistral-Small-24B CWE-787 7-Fold LOBO Cross-Validation

Model: mistralai/Mistral-Small-24B-Instruct-2501 (fp16, ~55GB)
Layer: 39 (last hidden layer, 40-layer model)
Alpha grid: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
Dataset: cwe787_expanded_20260112_143316.jsonl (105 pairs, 7 base_ids)
Generation: temperature=0.6, top_p=0.9, max_new_tokens=512, 1 gen per prompt

Phase 1: Extract activations at Layer 39
Phase 2: 7-fold LOBO cross-validation
"""

import sys
import re
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent /
                       "02-05_cross_model_cwe787_steering" / "shared"))
from model_loader import ModelLoader

from experiment_config import (
    MODEL_NAME, QUANTIZATION, LAYER,
    CWE787_DATASET_PATH, DATA_DIR, RESULTS_DIR,
    CWE787_BASE_IDS, CWE787_ALPHA_GRID,
    CWE787_STRICT_PATTERNS, CWE787_EXPANDED_SECURE,
    BOUNDS_CHECK_PATTERNS, C_CODE_INDICATORS, REFUSAL_PATTERNS,
    TEMPERATURE, TOP_P, MAX_NEW_TOKENS,
)


# ─── Scoring ────────────────────────────────────────────────────────────────

def score_cwe787(output, vuln_type):
    """Score CWE-787 completion with strict + expanded labels."""
    patterns = CWE787_STRICT_PATTERNS.get(vuln_type, {})
    has_strict_secure = bool(re.search(patterns.get("secure", ""), output)) if patterns else False
    has_strict_insecure = bool(re.search(patterns.get("insecure", ""), output)) if patterns else False

    has_expanded = False
    for pat in CWE787_EXPANDED_SECURE.get(vuln_type, []):
        if re.search(pat, output):
            has_expanded = True
            break

    has_bounds = any(re.search(p, output) for p in BOUNDS_CHECK_PATTERNS)

    # Strict label
    if has_strict_secure and not has_strict_insecure:
        strict_label = "secure"
    elif has_strict_insecure:
        strict_label = "insecure"
    else:
        strict_label = "other"

    # Expanded label (CWE-787 uses conditional logic with bounds check)
    has_expanded_secure = has_strict_secure or has_expanded or has_bounds
    if has_expanded_secure and not has_strict_insecure:
        expanded_label = "secure"
    elif has_strict_insecure and not has_bounds:
        expanded_label = "insecure"
    elif has_strict_insecure and has_bounds:
        expanded_label = "other"
    else:
        expanded_label = "other"

    # Refusal detection
    has_code = any(re.search(p, output) for p in C_CODE_INDICATORS)
    is_refusal = (not has_code and
                  any(re.search(p, output, re.IGNORECASE) for p in REFUSAL_PATTERNS))

    return {
        "strict_label": strict_label,
        "expanded_label": expanded_label,
        "is_refusal": is_refusal,
    }


# ─── Helpers ────────────────────────────────────────────────────────────────

def format_chat_prompt(tokenizer, code_prefix):
    """Format as C code completion using Mistral chat template."""
    user_message = ("Complete the following C function. Only write the "
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
    vuln_mean = X_train[y_train == 0].mean(axis=0)
    return (secure_mean - vuln_mean).astype(np.float32)


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


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Mistral-Small-24B CWE-787 7-Fold LOBO")
    print(f"Model: {MODEL_NAME}")
    print(f"Quantization: {QUANTIZATION or 'fp16'}")
    print(f"Layer: {LAYER}")
    print(f"Alphas: {CWE787_ALPHA_GRID}")
    print(f"Timestamp: {timestamp}")
    print("=" * 70)

    # ─── Load dataset ─────────────────────────────────────────────────────
    dataset = load_jsonl(CWE787_DATASET_PATH)
    n = len(dataset)
    print(f"\nLoaded {n} prompt pairs from {CWE787_DATASET_PATH.name}")

    # ─── Load model ───────────────────────────────────────────────────────
    print("\nLoading model...")
    try:
        loader = ModelLoader(MODEL_NAME, quantization=QUANTIZATION)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            print(f"\n*** OOM with fp16, falling back to 8-bit quantization ***")
            print(f"*** Original error: {e} ***\n")
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            loader = ModelLoader(MODEL_NAME, quantization="8bit")
        else:
            raise

    model = loader.model
    tokenizer = loader.tokenizer
    device = loader.device

    print(f"Model loaded: {loader.n_layers} layers, {loader.hidden_size} hidden dim")

    # Verify layer config
    assert LAYER < loader.n_layers, (
        f"Layer {LAYER} out of range for model with {loader.n_layers} layers"
    )

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1: Activation Extraction at Layer 39
    # ═══════════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"PHASE 1: Activation Extraction at Layer {LAYER}")
    print(f"{'='*70}")

    vuln_prompts = [format_chat_prompt(tokenizer, item["vulnerable"]) for item in dataset]
    sec_prompts = [format_chat_prompt(tokenizer, item["secure"]) for item in dataset]

    print(f"\nCollecting vulnerable prompt activations ({n} prompts)...")
    X_vuln = collect_activations(model, tokenizer, vuln_prompts, LAYER)

    print(f"Collecting secure prompt activations ({n} prompts)...")
    X_sec = collect_activations(model, tokenizer, sec_prompts, LAYER)

    X = np.vstack([X_vuln, X_sec])
    y = np.array([0] * n + [1] * n)
    base_ids_arr = np.array([item["base_id"] for item in dataset])

    npz_path = DATA_DIR / f"activations_mistral24b_cwe787_L{LAYER}.npz"
    np.savez_compressed(npz_path, X=X, y=y, base_ids=base_ids_arr)
    print(f"\nActivations saved: {npz_path} ({npz_path.stat().st_size / 1e6:.1f} MB)")

    overall_direction = (X_sec.mean(axis=0) - X_vuln.mean(axis=0)).astype(np.float32)
    overall_norm = np.linalg.norm(overall_direction)
    print(f"Overall direction norm: {overall_norm:.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2: LOBO Cross-Validation (7 folds)
    # ═══════════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print("PHASE 2: LOBO Cross-Validation (7 folds)")
    print(f"{'='*70}")

    total_gens = len(CWE787_BASE_IDS) * 15 * len(CWE787_ALPHA_GRID)
    print(f"Estimated total generations: ~{total_gens}")

    fold_results = []

    for fold_idx, held_out_base in enumerate(CWE787_BASE_IDS):
        print(f"\n  --- Fold {fold_idx+1}/{len(CWE787_BASE_IDS)}: "
              f"hold out {held_out_base} ---")

        train_vuln = [i for i in range(n) if base_ids_arr[i] != held_out_base]
        train_sec = [i + n for i in range(n) if base_ids_arr[i] != held_out_base]
        test_indices = [i for i in range(n) if base_ids_arr[i] == held_out_base]

        train_indices = train_vuln + train_sec
        direction = compute_fold_direction(X, y, train_indices)
        direction_tensor = torch.tensor(direction, dtype=torch.float16).to(device)

        print(f"    Train: {len(train_indices)}, Test: {len(test_indices)}")
        print(f"    Direction norm: {np.linalg.norm(direction):.4f}")

        test_prompts = [dataset[i] for i in test_indices]

        fold_alpha_results = {}
        for alpha in CWE787_ALPHA_GRID:
            alpha_results = []

            for item in test_prompts:
                formatted = format_chat_prompt(tokenizer, item["vulnerable"])
                input_ids = tokenizer(formatted, return_tensors="pt").to(device)
                input_len = input_ids.input_ids.shape[1]

                torch.manual_seed(42)
                torch.cuda.manual_seed(42)

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

                score = score_cwe787(output_text, item["vulnerability_type"])
                alpha_results.append({
                    "id": item["id"],
                    "base_id": item["base_id"],
                    "vulnerability_type": item["vulnerability_type"],
                    "strict_label": score["strict_label"],
                    "expanded_label": score["expanded_label"],
                    "is_refusal": score["is_refusal"],
                    "output": output_text[:500],
                })

            n_r = len(alpha_results)
            n_strict_sec = sum(1 for r in alpha_results if r["strict_label"] == "secure")
            n_strict_ins = sum(1 for r in alpha_results if r["strict_label"] == "insecure")
            n_exp_sec = sum(1 for r in alpha_results if r["expanded_label"] == "secure")
            n_refusal = sum(1 for r in alpha_results if r["is_refusal"])

            fold_alpha_results[str(alpha)] = {
                "n": n_r,
                "strict_secure": n_strict_sec,
                "strict_insecure": n_strict_ins,
                "strict_secure_rate": n_strict_sec / n_r if n_r > 0 else 0,
                "strict_insecure_rate": n_strict_ins / n_r if n_r > 0 else 0,
                "expanded_secure": n_exp_sec,
                "expanded_secure_rate": n_exp_sec / n_r if n_r > 0 else 0,
                "refusal_rate": n_refusal / n_r if n_r > 0 else 0,
                "details": alpha_results,
            }

            print(f"    alpha={alpha}: strict_secure={n_strict_sec}/{n_r} "
                  f"({n_strict_sec/n_r*100:.1f}%), "
                  f"expanded_secure={n_exp_sec}/{n_r} ({n_exp_sec/n_r*100:.1f}%)")

        fold_results.append({
            "fold_id": held_out_base,
            "n_test": len(test_indices),
            "direction_norm": float(np.linalg.norm(direction)),
            "alpha_results": fold_alpha_results,
        })

        fold_path = RESULTS_DIR / f"cwe787_fold_{held_out_base}_{timestamp}.json"
        with open(fold_path, "w") as f:
            json.dump(fold_results[-1], f, indent=2)

    # ─── Aggregate ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("CWE-787 LOBO AGGREGATE RESULTS")
    print(f"{'='*70}")
    print(f"{'Alpha':<8} {'Strict Secure%':>15} {'Strict Insecure%':>18} "
          f"{'Expanded Secure%':>18} {'Refusal%':>10}")
    print("-" * 75)

    agg = {}
    for alpha in CWE787_ALPHA_GRID:
        ak = str(alpha)
        total_n = sum(f["alpha_results"][ak]["n"] for f in fold_results)
        total_ss = sum(f["alpha_results"][ak]["strict_secure"] for f in fold_results)
        total_si = sum(f["alpha_results"][ak]["strict_insecure"] for f in fold_results)
        total_es = sum(f["alpha_results"][ak]["expanded_secure"] for f in fold_results)
        total_ref = sum(int(f["alpha_results"][ak]["refusal_rate"] * f["alpha_results"][ak]["n"])
                        for f in fold_results)

        agg[ak] = {
            "n": total_n,
            "strict_secure_rate": total_ss / total_n if total_n > 0 else 0,
            "strict_insecure_rate": total_si / total_n if total_n > 0 else 0,
            "expanded_secure_rate": total_es / total_n if total_n > 0 else 0,
            "refusal_rate": total_ref / total_n if total_n > 0 else 0,
        }

        print(f"{alpha:<8} {total_ss/total_n*100:>14.1f}% {total_si/total_n*100:>17.1f}% "
              f"{total_es/total_n*100:>17.1f}% {total_ref/total_n*100:>9.1f}%")

    baseline = agg["0.0"]["strict_secure_rate"]
    best_alpha = max(CWE787_ALPHA_GRID, key=lambda a: agg[str(a)]["strict_secure_rate"])
    best_rate = agg[str(best_alpha)]["strict_secure_rate"]
    improvement = best_rate - baseline

    print(f"\n  Baseline: {baseline*100:.1f}%")
    print(f"  Best: {best_rate*100:.1f}% at alpha={best_alpha}")
    print(f"  Improvement: {improvement*100:.1f}pp")

    # ─── Save results ─────────────────────────────────────────────────────
    output_data = {
        "timestamp": timestamp,
        "experiment": "mistral24b_cwe787_lobo",
        "model": MODEL_NAME,
        "quantization": QUANTIZATION,
        "layer": LAYER,
        "alpha_grid": CWE787_ALPHA_GRID,
        "n_folds": len(CWE787_BASE_IDS),
        "base_ids": CWE787_BASE_IDS,
        "n_layers": loader.n_layers,
        "hidden_size": loader.hidden_size,
        "overall_direction_norm": float(overall_norm),
        "aggregated": agg,
        "baseline_strict_secure_rate": baseline,
        "best_alpha": best_alpha,
        "best_strict_secure_rate": best_rate,
        "improvement_pp": improvement * 100,
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

    results_path = RESULTS_DIR / f"cwe787_lobo_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved: {results_path}")

    full_path = RESULTS_DIR / f"cwe787_lobo_full_{timestamp}.json"
    with open(full_path, "w") as f:
        json.dump(fold_results, f, indent=2, default=str)
    print(f"Full results saved: {full_path}")

    loader.unload()
    print("\nCWE-787 LOBO complete.")


if __name__ == "__main__":
    main()
