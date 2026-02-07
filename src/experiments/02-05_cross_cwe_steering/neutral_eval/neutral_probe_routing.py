#!/usr/bin/env python3
"""
Experiment 8 — Phase 4: Probe-Gated Routing Simulation

Demonstrates the full deployment pipeline: probe classifies CWE type,
routes to the correct per-CWE steering vector.

Approach:
1. Collect last-token activations for adversarial training data (3 CWEs)
2. Collect last-token activations for neutral evaluation prompts
3. Train 3-class logistic regression probe at Layer 0 and Layer 31
4. Test routing accuracy on neutral prompts
5. Also test direction-based routing (dot product with steering vectors)

Expected: 21 neutral prompts x 1 forward pass = ~1 minute.
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

SCRIPT_DIR = Path(__file__).parent
EXPERIMENT_DIR = SCRIPT_DIR.parent  # 02-05_cross_cwe_steering/
sys.path.insert(0, str(EXPERIMENT_DIR / "shared"))

from model_loader import ModelLoader

# ─── Configuration ──────────────────────────────────────────────────────────

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
PROBE_LAYERS = [0, 31]  # Layer 0 (early) and Layer 31 (steering layer)

PROMPTS_PATH = SCRIPT_DIR / "data" / "neutral_eval_prompts.jsonl"
RESULTS_DIR = SCRIPT_DIR / "results"
VECTORS_DIR = EXPERIMENT_DIR / "cross_cwe_analysis" / "data"

# Adversarial datasets (for training the CWE-type probe)
ADVERSARIAL_DATASETS = {
    "CWE-787": Path("src/experiments/01-12_cwe787_dataset_expansion/data/cwe787_expanded_20260112_143316.jsonl"),
    "CWE-119": EXPERIMENT_DIR / "datasets" / "cwe119" / "data" / "cwe119_expanded_20260207_024627.jsonl",
    "CWE-134": EXPERIMENT_DIR / "datasets" / "cwe134" / "data" / "cwe134_expanded_20260207_024627.jsonl",
}

# Steering vectors at Layer 31 (for direction-based routing)
VECTOR_FILES = {
    "CWE-787": "direction_cwe787_L31_20260206_031901.npy",
    "CWE-119": "direction_cwe119_L31_20260206_031901.npy",
    "CWE-134": "direction_cwe134_L31_20260206_031901.npy",
}

CWE_TO_IDX = {"CWE-787": 0, "CWE-119": 1, "CWE-134": 2}
IDX_TO_CWE = {0: "CWE-787", 1: "CWE-119", 2: "CWE-134"}


# ─── Chat Prompt Formatting ─────────────────────────────────────────────────

def format_chat_prompt(tokenizer, code_prefix):
    user_message = f"Complete the following C function. Only write the function body, no explanation.\n\n{code_prefix}"
    messages = [{"role": "user", "content": user_message}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ─── Activation Collection ──────────────────────────────────────────────────

def collect_activations(model, tokenizer, device, prompts, layers):
    """
    Collect last-token activations at specified layers for a list of prompts.

    Args:
        model: loaded model
        tokenizer: tokenizer
        device: torch device
        prompts: list of formatted prompt strings
        layers: list of layer indices

    Returns:
        dict: {layer_idx: np.array of shape (n_prompts, hidden_size)}
    """
    activations = {layer: [] for layer in layers}
    hooks = []
    current_acts = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            current_acts[layer_idx] = h[:, -1, :].detach().cpu().numpy().astype(np.float32)
        return hook_fn

    # Register hooks
    for layer_idx in layers:
        layer = model.model.layers[layer_idx]
        hook = layer.register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)

    # Forward pass each prompt
    for prompt in tqdm(prompts, desc="Collecting activations"):
        current_acts.clear()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model(**inputs)
        for layer_idx in layers:
            activations[layer_idx].append(current_acts[layer_idx].squeeze(0))

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Stack into arrays
    return {layer: np.stack(acts) for layer, acts in activations.items()}


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Experiment 8 — Phase 4: Probe-Gated Routing Simulation")
    print(f"Model: {MODEL_NAME}")
    print(f"Probe layers: {PROBE_LAYERS}")
    print("=" * 70)

    # ─── Load neutral prompts ────────────────────────────────────────────
    neutral_prompts = []
    with open(PROMPTS_PATH) as f:
        for line in f:
            if line.strip():
                neutral_prompts.append(json.loads(line))

    neutral_by_cwe = {}
    for p in neutral_prompts:
        neutral_by_cwe.setdefault(p["cwe"], []).append(p)
    for cwe, ps in sorted(neutral_by_cwe.items()):
        print(f"  Neutral {cwe}: {len(ps)} prompts")

    # ─── Load adversarial datasets ───────────────────────────────────────
    adv_prompts_by_cwe = {}
    for cwe, path in ADVERSARIAL_DATASETS.items():
        # Handle relative paths
        if not path.is_absolute():
            path = Path("/home/paperspace/MATS") / path
        with open(path) as f:
            data = [json.loads(line) for line in f]
        adv_prompts_by_cwe[cwe] = data
        print(f"  Adversarial {cwe}: {len(data)} pairs")

    # ─── Load model ──────────────────────────────────────────────────────
    loader = ModelLoader(MODEL_NAME)
    model = loader.model
    tokenizer = loader.tokenizer
    device = loader.device

    # ─── Format all prompts through chat template ────────────────────────
    # Adversarial: use VULNERABLE prompts (realistic: model sees insecure-context code)
    print("\nFormatting adversarial prompts...")
    adv_formatted = []
    adv_labels = []  # CWE index labels
    adv_cwe_names = []
    for cwe in ["CWE-787", "CWE-119", "CWE-134"]:
        for pair in adv_prompts_by_cwe[cwe]:
            adv_formatted.append(format_chat_prompt(tokenizer, pair["vulnerable"]))
            adv_labels.append(CWE_TO_IDX[cwe])
            adv_cwe_names.append(cwe)
    adv_labels = np.array(adv_labels)
    print(f"  Total adversarial prompts: {len(adv_formatted)}")
    for cwe_idx, cwe_name in IDX_TO_CWE.items():
        print(f"    {cwe_name}: {(adv_labels == cwe_idx).sum()}")

    # Neutral: format through chat template
    print("\nFormatting neutral prompts...")
    neutral_formatted = []
    neutral_labels = []
    neutral_ids = []
    for p in neutral_prompts:
        neutral_formatted.append(format_chat_prompt(tokenizer, p["prompt"]))
        neutral_labels.append(CWE_TO_IDX[p["cwe"]])
        neutral_ids.append(p["id"])
    neutral_labels = np.array(neutral_labels)
    print(f"  Total neutral prompts: {len(neutral_formatted)}")

    # ─── Collect activations ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Collecting adversarial prompt activations...")
    print(f"{'='*70}")
    adv_activations = collect_activations(model, tokenizer, device, adv_formatted, PROBE_LAYERS)
    for layer in PROBE_LAYERS:
        print(f"  Layer {layer}: shape={adv_activations[layer].shape}")

    print(f"\n{'='*70}")
    print("Collecting neutral prompt activations...")
    print(f"{'='*70}")
    neutral_activations = collect_activations(model, tokenizer, device, neutral_formatted, PROBE_LAYERS)
    for layer in PROBE_LAYERS:
        print(f"  Layer {layer}: shape={neutral_activations[layer].shape}")

    # Unload model to free GPU
    loader.unload()

    # ─── Method 1: Logistic Regression Probe ─────────────────────────────
    print(f"\n{'='*70}")
    print("METHOD 1: 3-Class Logistic Regression Probe")
    print(f"{'='*70}")

    probe_results = {}
    for layer in PROBE_LAYERS:
        X_train = adv_activations[layer]
        y_train = adv_labels
        X_test = neutral_activations[layer]
        y_test = neutral_labels

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train probe
        clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', multi_class='multinomial')
        clf.fit(X_train_scaled, y_train)

        # Cross-validation on training data
        cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5, scoring='accuracy')
        train_acc = clf.score(X_train_scaled, y_train)

        # Test on neutral prompts
        predictions = clf.predict(X_test_scaled)
        probabilities = clf.predict_proba(X_test_scaled)
        test_acc = (predictions == y_test).mean()

        print(f"\n  Layer {layer}:")
        print(f"    Training accuracy: {train_acc*100:.1f}%")
        print(f"    5-fold CV accuracy: {cv_scores.mean()*100:.1f}% (+/- {cv_scores.std()*100:.1f}%)")
        print(f"    Neutral routing accuracy: {test_acc*100:.1f}% ({int(test_acc*len(y_test))}/{len(y_test)})")

        # Per-CWE breakdown
        print(f"\n    Per-CWE routing accuracy:")
        per_cwe = {}
        for cwe_idx, cwe_name in IDX_TO_CWE.items():
            mask = y_test == cwe_idx
            if mask.sum() > 0:
                cwe_acc = (predictions[mask] == y_test[mask]).mean()
                correct = (predictions[mask] == y_test[mask]).sum()
                total = mask.sum()
                print(f"      {cwe_name}: {cwe_acc*100:.1f}% ({correct}/{total})")
                per_cwe[cwe_name] = {"accuracy": float(cwe_acc), "correct": int(correct), "total": int(total)}

        # Confusion details for misrouted prompts
        misrouted = []
        for i in range(len(y_test)):
            if predictions[i] != y_test[i]:
                true_cwe = IDX_TO_CWE[y_test[i]]
                pred_cwe = IDX_TO_CWE[predictions[i]]
                probs = {IDX_TO_CWE[j]: float(probabilities[i][j]) for j in range(3)}
                misrouted.append({
                    "id": neutral_ids[i],
                    "true_cwe": true_cwe,
                    "predicted_cwe": pred_cwe,
                    "probabilities": probs,
                })
                print(f"\n    MISROUTED: {neutral_ids[i]} — true={true_cwe}, predicted={pred_cwe}")
                print(f"      Probabilities: {probs}")

        probe_results[f"layer_{layer}"] = {
            "train_accuracy": float(train_acc),
            "cv_accuracy": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "neutral_routing_accuracy": float(test_acc),
            "neutral_correct": int(test_acc * len(y_test)),
            "neutral_total": len(y_test),
            "per_cwe": per_cwe,
            "misrouted": misrouted,
        }

    # ─── Method 2: Direction-Based Routing ───────────────────────────────
    print(f"\n{'='*70}")
    print("METHOD 2: Direction-Based Routing (dot product with steering vectors)")
    print(f"{'='*70}")

    # Load steering vectors
    directions = {}
    for cwe, fname in VECTOR_FILES.items():
        vec = np.load(VECTORS_DIR / fname)
        directions[cwe] = vec.astype(np.float32)
        print(f"  Loaded {cwe} direction: norm={np.linalg.norm(vec):.4f}")

    # Only works at Layer 31 (where steering vectors were computed)
    X_neutral_L31 = neutral_activations[31]

    # Compute dot product with each direction, normalized by direction norm
    dot_scores = {}
    for cwe in ["CWE-787", "CWE-119", "CWE-134"]:
        d = directions[cwe]
        d_normed = d / np.linalg.norm(d)
        dot_scores[cwe] = X_neutral_L31 @ d_normed

    # Route to the CWE with highest dot product
    score_matrix = np.stack([dot_scores["CWE-787"], dot_scores["CWE-119"], dot_scores["CWE-134"]], axis=1)
    direction_predictions = np.argmax(score_matrix, axis=1)
    direction_acc = (direction_predictions == neutral_labels).mean()

    print(f"\n  Direction-based routing accuracy: {direction_acc*100:.1f}% ({int(direction_acc*len(neutral_labels))}/{len(neutral_labels)})")

    # Per-CWE breakdown
    dir_per_cwe = {}
    print(f"\n  Per-CWE routing accuracy:")
    for cwe_idx, cwe_name in IDX_TO_CWE.items():
        mask = neutral_labels == cwe_idx
        if mask.sum() > 0:
            cwe_acc = (direction_predictions[mask] == neutral_labels[mask]).mean()
            correct = (direction_predictions[mask] == neutral_labels[mask]).sum()
            total = mask.sum()
            print(f"    {cwe_name}: {cwe_acc*100:.1f}% ({correct}/{total})")
            dir_per_cwe[cwe_name] = {"accuracy": float(cwe_acc), "correct": int(correct), "total": int(total)}

    # Confusion details
    dir_misrouted = []
    for i in range(len(neutral_labels)):
        if direction_predictions[i] != neutral_labels[i]:
            true_cwe = IDX_TO_CWE[neutral_labels[i]]
            pred_cwe = IDX_TO_CWE[direction_predictions[i]]
            scores = {IDX_TO_CWE[j]: float(score_matrix[i][j]) for j in range(3)}
            dir_misrouted.append({
                "id": neutral_ids[i],
                "true_cwe": true_cwe,
                "predicted_cwe": pred_cwe,
                "scores": scores,
            })
            print(f"\n  MISROUTED: {neutral_ids[i]} — true={true_cwe}, predicted={pred_cwe}")
            print(f"    Scores: {scores}")

    direction_results = {
        "layer": 31,
        "routing_accuracy": float(direction_acc),
        "correct": int(direction_acc * len(neutral_labels)),
        "total": len(neutral_labels),
        "per_cwe": dir_per_cwe,
        "misrouted": dir_misrouted,
    }

    # ─── Method 3: Cosine Similarity Routing ─────────────────────────────
    print(f"\n{'='*70}")
    print("METHOD 3: Cosine Similarity Routing")
    print(f"{'='*70}")

    cos_scores = {}
    for cwe in ["CWE-787", "CWE-119", "CWE-134"]:
        d = directions[cwe]
        d_normed = d / np.linalg.norm(d)
        # Normalize each activation too
        norms = np.linalg.norm(X_neutral_L31, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        X_normed = X_neutral_L31 / norms
        cos_scores[cwe] = X_normed @ d_normed

    cos_matrix = np.stack([cos_scores["CWE-787"], cos_scores["CWE-119"], cos_scores["CWE-134"]], axis=1)
    cos_predictions = np.argmax(cos_matrix, axis=1)
    cos_acc = (cos_predictions == neutral_labels).mean()

    print(f"\n  Cosine routing accuracy: {cos_acc*100:.1f}% ({int(cos_acc*len(neutral_labels))}/{len(neutral_labels)})")

    cos_per_cwe = {}
    print(f"\n  Per-CWE routing accuracy:")
    for cwe_idx, cwe_name in IDX_TO_CWE.items():
        mask = neutral_labels == cwe_idx
        if mask.sum() > 0:
            cwe_acc = (cos_predictions[mask] == neutral_labels[mask]).mean()
            correct = (cos_predictions[mask] == neutral_labels[mask]).sum()
            total = mask.sum()
            print(f"    {cwe_name}: {cwe_acc*100:.1f}% ({correct}/{total})")
            cos_per_cwe[cwe_name] = {"accuracy": float(cwe_acc), "correct": int(correct), "total": int(total)}

    cosine_results = {
        "layer": 31,
        "routing_accuracy": float(cos_acc),
        "correct": int(cos_acc * len(neutral_labels)),
        "total": len(neutral_labels),
        "per_cwe": cos_per_cwe,
    }

    # ─── Summary Table ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("ROUTING ACCURACY SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':<40} {'Overall':>10} {'CWE-787':>10} {'CWE-119':>10} {'CWE-134':>10}")
    print("-" * 80)

    for layer in PROBE_LAYERS:
        r = probe_results[f"layer_{layer}"]
        cwe787_acc = r["per_cwe"].get("CWE-787", {}).get("accuracy", 0)
        cwe119_acc = r["per_cwe"].get("CWE-119", {}).get("accuracy", 0)
        cwe134_acc = r["per_cwe"].get("CWE-134", {}).get("accuracy", 0)
        print(f"{'LogReg probe L' + str(layer):<40} {r['neutral_routing_accuracy']*100:>9.1f}% {cwe787_acc*100:>9.1f}% {cwe119_acc*100:>9.1f}% {cwe134_acc*100:>9.1f}%")

    # Direction-based
    cwe787_acc = direction_results["per_cwe"].get("CWE-787", {}).get("accuracy", 0)
    cwe119_acc = direction_results["per_cwe"].get("CWE-119", {}).get("accuracy", 0)
    cwe134_acc = direction_results["per_cwe"].get("CWE-134", {}).get("accuracy", 0)
    print(f"{'Direction dot-product (L31)':<40} {direction_results['routing_accuracy']*100:>9.1f}% {cwe787_acc*100:>9.1f}% {cwe119_acc*100:>9.1f}% {cwe134_acc*100:>9.1f}%")

    # Cosine
    cwe787_acc = cosine_results["per_cwe"].get("CWE-787", {}).get("accuracy", 0)
    cwe119_acc = cosine_results["per_cwe"].get("CWE-119", {}).get("accuracy", 0)
    cwe134_acc = cosine_results["per_cwe"].get("CWE-134", {}).get("accuracy", 0)
    print(f"{'Cosine similarity (L31)':<40} {cosine_results['routing_accuracy']*100:>9.1f}% {cwe787_acc*100:>9.1f}% {cwe119_acc*100:>9.1f}% {cwe134_acc*100:>9.1f}%")

    # ─── Save Results ────────────────────────────────────────────────────
    output = {
        "timestamp": timestamp,
        "model": MODEL_NAME,
        "probe_layers": PROBE_LAYERS,
        "n_adversarial_train": len(adv_formatted),
        "n_neutral_test": len(neutral_formatted),
        "probe_results": probe_results,
        "direction_routing": direction_results,
        "cosine_routing": cosine_results,
    }

    results_path = RESULTS_DIR / f"neutral_probe_routing_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {results_path}")

    print("\nPhase 4 complete.")


if __name__ == "__main__":
    main()
