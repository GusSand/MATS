#!/usr/bin/env python3
"""
Run our mean-difference steering on CodeGen-2B-multi for same-model comparison.

Steps:
  1. Collect last-token activations at all 32 layers for each prompt pair
  2. Find emergence layer via logit lens (vocabulary projection)
  3. Compute per-CWE mean-difference steering vectors at emergence layer
  4. LOBO cross-validation with alpha sweep
  5. Score with our regex scorers

Usage:
    python run_steering_codegen.py [--model-size 2b] [--n-seeds 10] [--dry-run]
"""

import argparse
import json
import os
import re
import sys
import time
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Scorers
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

ALPHA_GRID = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
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


def load_prompts(cwe):
    """Load adapted prompts for a CWE."""
    cfg = CWE_CONFIG[cwe]
    path = ADAPTED_DIR / cfg["file"]
    prompts = []
    with open(path) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    return prompts


# ─── Step 1: Activation Collection ─────────────────────────────────────────

def collect_activations(model, tokenizer, prompts, cwe, device, n_layers):
    """Collect last-token activations at all layers for vulnerable and secure prompts."""
    print(f"\n  Collecting activations for {cwe} ({len(prompts)} pairs × 2 conditions)...")

    all_activations = []  # List of (activations_dict, label, base_id)
    activations_store = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            activations_store[layer_idx] = h[:, -1, :].detach().cpu()
        return hook_fn

    for p_idx, prompt_data in enumerate(prompts):
        for condition in ["codegen_vulnerable", "codegen_secure"]:
            prompt_text = prompt_data[condition]
            label = 0 if condition == "codegen_vulnerable" else 1  # 0=insecure, 1=secure
            base_id = prompt_data.get("base_id", "unknown")

            # Register hooks
            hooks = []
            activations_store.clear()
            for layer_idx in range(n_layers):
                layer = model.transformer.h[layer_idx]
                hook = layer.register_forward_hook(make_hook(layer_idx))
                hooks.append(hook)

            # Forward pass
            input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                model(input_ids)

            # Remove hooks
            for hook in hooks:
                hook.remove()

            # Store activations (all layers)
            act_dict = {l: activations_store[l].numpy() for l in range(n_layers)}
            all_activations.append((act_dict, label, base_id))

        if (p_idx + 1) % 20 == 0:
            print(f"    [{p_idx+1}/{len(prompts)}]")

    return all_activations


# ─── Step 2: Logit Lens (emergence layer) ──────────────────────────────────

def find_emergence_layer(model, tokenizer, activations_data, cwe, n_layers, device):
    """Find the layer where secure/insecure tokens diverge most in vocab projection."""
    print(f"\n  Running logit lens for {cwe}...")

    # Get the output embedding (lm_head)
    lm_head = model.lm_head.weight.data  # (vocab_size, hidden_size)
    ln_f = model.transformer.ln_f  # Final layer norm

    # For each layer, compute mean difference in logit-space between secure and insecure
    layer_divergences = []

    for layer_idx in range(n_layers):
        sec_logits_list = []
        insec_logits_list = []

        for act_dict, label, base_id in activations_data:
            h = torch.tensor(act_dict[layer_idx], dtype=torch.float16).to(device)
            # Apply final layer norm before projection
            h_normed = ln_f(h)
            logits = h_normed @ lm_head.T  # (1, vocab_size)

            if label == 1:
                sec_logits_list.append(logits.cpu().float())
            else:
                insec_logits_list.append(logits.cpu().float())

        sec_mean = torch.stack(sec_logits_list).mean(0)
        insec_mean = torch.stack(insec_logits_list).mean(0)
        divergence = (sec_mean - insec_mean).norm().item()
        layer_divergences.append(divergence)

    # Find layer with max divergence
    best_layer = int(np.argmax(layer_divergences))
    print(f"    Emergence layer: {best_layer} (divergence: {layer_divergences[best_layer]:.2f})")
    print(f"    Top 5 layers: {sorted(range(n_layers), key=lambda i: -layer_divergences[i])[:5]}")

    return best_layer, layer_divergences


# ─── Step 3+4: LOBO Cross-Validation ───────────────────────────────────────

def run_lobo(model, tokenizer, prompts, activations_data, cwe, layer, device, n_seeds, dry_run=False):
    """Run leave-one-base-out cross-validation with alpha sweep."""
    lang = CWE_CONFIG[cwe]["language"]
    scorer = PYTHON_SCORERS.get(cwe)

    # Organize by base_id
    base_id_to_indices = defaultdict(list)
    for idx, (act_dict, label, base_id) in enumerate(activations_data):
        base_id_to_indices[base_id].append(idx)

    base_ids = sorted(base_id_to_indices.keys())
    n_folds = len(base_ids)
    print(f"\n  LOBO for {cwe}: {n_folds} folds, layer {layer}")

    # Extract layer activations into arrays
    n = len(activations_data)
    hidden_size = activations_data[0][0][layer].shape[-1]
    X = np.zeros((n, hidden_size), dtype=np.float32)
    y = np.zeros(n, dtype=np.int32)
    base_ids_arr = []
    for idx, (act_dict, label, base_id) in enumerate(activations_data):
        X[idx] = act_dict[layer].flatten()
        y[idx] = label
        base_ids_arr.append(base_id)

    fold_results = {}

    for fold_idx, held_out_base in enumerate(base_ids):
        print(f"\n    Fold {fold_idx+1}/{n_folds}: hold out {held_out_base}")

        # Train/test split
        train_mask = np.array([b != held_out_base for b in base_ids_arr])
        test_indices = [i for i, b in enumerate(base_ids_arr) if b == held_out_base and y[i] == 0]
        # test_indices: only insecure prompts for generation

        # Compute mean-difference direction from training set
        X_train = X[train_mask]
        y_train = y[train_mask]
        sec_mean = X_train[y_train == 1].mean(axis=0)
        insec_mean = X_train[y_train == 0].mean(axis=0)
        direction = (sec_mean - insec_mean).astype(np.float32)
        direction_tensor = torch.tensor(direction, dtype=torch.float16).to(device)

        # Get test prompts
        test_prompts = []
        for idx in test_indices:
            # Map back to original prompt — activations_data alternates vuln/sec per prompt
            prompt_idx = idx // 2
            test_prompts.append(prompts[prompt_idx])

        if dry_run:
            test_prompts = test_prompts[:1]

        alpha_results = {}
        for alpha in ALPHA_GRID:
            secure_count = 0
            insecure_count = 0
            other_count = 0
            results_list = []

            for prompt_data in test_prompts:
                prompt_text = prompt_data["codegen_vulnerable"]
                input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

                # Register steering hook
                hooks = []
                if alpha > 0:
                    def make_steering_hook(dir_t, a):
                        def hook_fn(module, input, output):
                            if isinstance(output, tuple):
                                h = output[0]
                            else:
                                h = output
                            h[:, -1, :] = h[:, -1, :] + a * dir_t
                            if isinstance(output, tuple):
                                return (h,) + output[1:]
                            return h
                        return hook_fn

                    target = model.transformer.h[layer]
                    hook = target.register_forward_hook(
                        make_steering_hook(direction_tensor, alpha)
                    )
                    hooks.append(hook)

                torch.manual_seed(42)
                torch.cuda.manual_seed(42)

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

                for hook in hooks:
                    hook.remove()

                for seq_idx in range(n_seeds):
                    completion_tokens = output[seq_idx, input_ids.shape[1]:]
                    completion = tokenizer.decode(completion_tokens)
                    if tokenizer.eos_token and tokenizer.eos_token in completion:
                        completion = completion[: completion.find(tokenizer.eos_token)]
                    completion = truncate_completion(completion, lang)
                    full_code = prompt_text + completion

                    if lang == "py":
                        label = scorer(full_code)
                    else:
                        label = score_c_completion(full_code, prompt_data["detection"])

                    if label == "secure":
                        secure_count += 1
                    elif label == "insecure":
                        insecure_count += 1
                    else:
                        other_count += 1

                    results_list.append(label)

            total = secure_count + insecure_count + other_count
            classifiable = secure_count + insecure_count
            alpha_results[alpha] = {
                "secure": secure_count,
                "insecure": insecure_count,
                "other": other_count,
                "total": total,
                "sec_rate_total": round(secure_count / total, 4) if total > 0 else 0,
                "sec_rate_classifiable": round(secure_count / classifiable, 4) if classifiable > 0 else 0,
            }
            print(f"      alpha={alpha:.1f}: sec={secure_count}/{total} "
                  f"({alpha_results[alpha]['sec_rate_total']:.1%})")

        fold_results[held_out_base] = alpha_results

    return fold_results


# ─── Main ───────────────────────────────────────────────────────────────────

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
    print(f"Loading {model_name}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.to(args.device)
    model.eval()
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    n_layers = model.config.n_layer
    hidden_size = model.config.n_embd
    print(f"  {n_layers} layers, hidden_size={hidden_size}")

    cwes_to_run = args.cwes or list(CWE_CONFIG.keys())
    all_cwe_results = {}
    total_gpu_time = load_time

    for cwe in cwes_to_run:
        print(f"\n{'='*70}")
        print(f"{cwe}")
        print(f"{'='*70}")

        prompts = load_prompts(cwe)
        if args.dry_run:
            prompts = prompts[:6]  # 6 prompts from 2 base_ids minimum for LOBO

        # Step 1: Collect activations
        t1 = time.time()
        act_data = collect_activations(model, tokenizer, prompts, cwe, args.device, n_layers)
        act_time = time.time() - t1
        total_gpu_time += act_time
        print(f"  Activations collected in {act_time:.1f}s")

        # Step 2: Find emergence layer
        t2 = time.time()
        emergence_layer, divergences = find_emergence_layer(
            model, tokenizer, act_data, cwe, n_layers, args.device
        )
        lens_time = time.time() - t2
        total_gpu_time += lens_time

        # Step 3+4: LOBO
        t3 = time.time()
        fold_results = run_lobo(
            model, tokenizer, prompts, act_data, cwe, emergence_layer,
            args.device, args.n_seeds, args.dry_run
        )
        lobo_time = time.time() - t3
        total_gpu_time += lobo_time

        # Aggregate: find best alpha across folds
        alpha_aggregates = defaultdict(lambda: {"secure": 0, "insecure": 0, "other": 0, "total": 0})
        for fold_base, alpha_dict in fold_results.items():
            for alpha, stats in alpha_dict.items():
                alpha_aggregates[alpha]["secure"] += stats["secure"]
                alpha_aggregates[alpha]["insecure"] += stats["insecure"]
                alpha_aggregates[alpha]["other"] += stats["other"]
                alpha_aggregates[alpha]["total"] += stats["total"]

        # Find best alpha by secure_rate_total
        best_alpha = 0.0
        best_rate = 0.0
        for alpha, agg in alpha_aggregates.items():
            rate = agg["secure"] / agg["total"] if agg["total"] > 0 else 0
            classifiable = agg["secure"] + agg["insecure"]
            agg["sec_rate_total"] = round(rate, 4)
            agg["sec_rate_classifiable"] = round(agg["secure"] / classifiable, 4) if classifiable > 0 else 0
            if rate > best_rate:
                best_rate = rate
                best_alpha = alpha

        print(f"\n  Best alpha: {best_alpha:.1f} (sec_rate={best_rate:.1%})")
        print(f"  Emergence layer: {emergence_layer}")

        all_cwe_results[cwe] = {
            "emergence_layer": emergence_layer,
            "layer_divergences": divergences,
            "best_alpha": best_alpha,
            "alpha_aggregates": {str(a): dict(v) for a, v in alpha_aggregates.items()},
            "fold_results": {
                base: {str(a): v for a, v in alpha_dict.items()}
                for base, alpha_dict in fold_results.items()
            },
            "timing": {
                "activation_collection_s": round(act_time, 1),
                "logit_lens_s": round(lens_time, 1),
                "lobo_s": round(lobo_time, 1),
            },
        }

    # Summary
    output = {
        "metadata": {
            "method": "Mean-difference steering (ours) on CodeGen",
            "model": model_name,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_new_tokens": MAX_NEW_TOKENS,
            "n_seeds": args.n_seeds,
            "alpha_grid": ALPHA_GRID,
            "timestamp": timestamp,
            "total_gpu_time_s": round(total_gpu_time, 1),
            "gpu_hours": round(total_gpu_time / 3600, 3),
        },
        "per_cwe": {},
    }

    print(f"\n{'='*70}")
    print("OVERALL: Our steering on CodeGen-2B-multi")
    print(f"{'='*70}")
    print(f"{'CWE':<10} {'Layer':>6} {'Alpha':>6} {'Secure':>8} {'Total':>8} {'Sec Rate':>10} {'Sec/Class':>10}")
    print("-" * 62)

    for cwe, data in all_cwe_results.items():
        best = data["alpha_aggregates"][str(data["best_alpha"])]
        output["per_cwe"][cwe] = {
            "emergence_layer": data["emergence_layer"],
            "best_alpha": data["best_alpha"],
            "secure": best["secure"],
            "insecure": best["insecure"],
            "other": best["other"],
            "total": best["total"],
            "secure_rate_total": best["sec_rate_total"],
            "secure_rate_classifiable": best["sec_rate_classifiable"],
        }
        print(f"{cwe:<10} {data['emergence_layer']:>6} {data['best_alpha']:>6.1f} "
              f"{best['secure']:>8} {best['total']:>8} "
              f"{best['sec_rate_total']:>10.1%} {best['sec_rate_classifiable']:>10.1%}")

    print(f"\nTotal GPU time: {total_gpu_time:.0f}s ({total_gpu_time/3600:.2f} GPU-hours)")

    results_file = RESULTS_DIR / f"steering_codegen_{args.model_size}_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {results_file}")

    detail_file = RESULTS_DIR / f"steering_codegen_{args.model_size}_{timestamp}_detail.json"
    with open(detail_file, "w") as f:
        json.dump({"metadata": output["metadata"], "per_cwe": all_cwe_results}, f, indent=2)
    print(f"Details: {detail_file}")


if __name__ == "__main__":
    main()
