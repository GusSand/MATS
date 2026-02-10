#!/usr/bin/env python3
"""
Experiment 10, Step 5: Probe-Based CWE Routing for Python

Train a probe on Layer 31 activations to classify which Python CWE a prompt
belongs to (CWE-89 vs CWE-78 vs CWE-79), then route to the correct
steering vector.

Steps:
  1. Load activation data from Step 2 (insecure prompts only)
  2. Train LogisticRegression probe (3-class)
  3. Evaluate on neutral Python prompts
  4. Report routing accuracy

Reuses: ModelLoader from shared/
"""
import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# ─── Paths ───────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
EXPERIMENT_DIR = Path("/home/paperspace/MATS/src/experiments/02-05_cross_cwe_steering")
DATASETS_DIR = EXPERIMENT_DIR / "datasets"
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"

sys.path.insert(0, str(EXPERIMENT_DIR / "shared"))
from model_loader import ModelLoader

# ─── Configuration ───────────────────────────────────────────────────────────

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LAYER = 31

NEUTRAL_FILE = DATASETS_DIR / "neutral_eval" / "data" / "neutral_python_prompts.jsonl"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def format_chat_prompt(tokenizer, code_prefix):
    """Format as Python code completion."""
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


@torch.no_grad()
def collect_l31_activations(model, tokenizer, prompts, layer=31):
    """Collect last-token activations at specified layer."""
    activations = []
    captured = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        captured["act"] = h[:, -1, :].detach().cpu()

    target_layer = model.model.layers[layer]
    hook = target_layer.register_forward_hook(hook_fn)

    for prompt in tqdm(prompts, desc="Collecting activations"):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        _ = model(**inputs)
        activations.append(captured["act"].numpy().astype(np.float32))

    hook.remove()
    return np.vstack(activations)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Experiment 10, Step 5: Probe-Based CWE Routing (Python)")
    print(f"Model: {MODEL_NAME}")
    print(f"Layer: {LAYER}")
    print("=" * 70)

    # ─── Load training data from Step 2 activations ──────────────────────
    print("\nLoading activation data...")

    X_train_parts = []
    y_train_parts = []
    cwe_labels = {"CWE-89": 0, "CWE-78": 1, "CWE-79": 2}
    cwe_names = {0: "CWE-89", 1: "CWE-78", 2: "CWE-79"}

    for cwe, label_idx in cwe_labels.items():
        cwe_num = cwe.split("-")[1]
        act_files = sorted(DATA_DIR.glob(f"activations_cwe{cwe_num}_L{LAYER}_*.npz"))
        if not act_files:
            print(f"  ERROR: No activation file for {cwe}. Run Step 2 first.")
            return
        data = np.load(act_files[-1])
        X = data["X"]
        y = data["y"]
        n_pairs = len(y) // 2

        # Use insecure prompts only (first n_pairs rows) for training
        X_insecure = X[:n_pairs]
        X_train_parts.append(X_insecure)
        y_train_parts.append(np.full(n_pairs, label_idx))
        print(f"  {cwe}: {n_pairs} insecure activations loaded")

    X_train = np.vstack(X_train_parts)
    y_train = np.concatenate(y_train_parts)
    print(f"\nTraining data: X={X_train.shape}, y={y_train.shape}")
    print(f"  Class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    # ─── Train probe ─────────────────────────────────────────────────────
    print("\nTraining 3-class LogisticRegression probe...")
    probe = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="multinomial")
    probe.fit(X_train, y_train)
    train_acc = probe.score(X_train, y_train)
    print(f"  Training accuracy: {train_acc*100:.1f}%")

    # Cross-validation
    cv_scores = cross_val_score(probe, X_train, y_train, cv=5, scoring="accuracy")
    print(f"  5-fold CV accuracy: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

    # ─── Evaluate on neutral prompts ─────────────────────────────────────
    print("\nEvaluating on neutral Python prompts...")

    loader = ModelLoader(MODEL_NAME)
    model = loader.model
    tokenizer = loader.tokenizer

    neutral_prompts = load_jsonl(NEUTRAL_FILE)
    print(f"Loaded {len(neutral_prompts)} neutral prompts")

    # Collect activations for neutral prompts
    formatted_neutral = [format_chat_prompt(tokenizer, item["prompt"])
                        for item in neutral_prompts]
    X_neutral = collect_l31_activations(model, tokenizer, formatted_neutral, LAYER)

    loader.unload()

    # Predict
    predictions = probe.predict(X_neutral)
    probabilities = probe.predict_proba(X_neutral)

    # Evaluate routing accuracy
    print(f"\n{'='*70}")
    print("ROUTING RESULTS")
    print(f"{'='*70}")

    correct = 0
    total = 0
    results = []

    for i, item in enumerate(neutral_prompts):
        true_cwe = item["cwe"]
        true_label = cwe_labels.get(true_cwe)
        pred_label = predictions[i]
        pred_cwe = cwe_names[pred_label]
        confidence = probabilities[i][pred_label]
        is_correct = bool(true_label == pred_label) if true_label is not None else None

        if is_correct is not None:
            total += 1
            if is_correct:
                correct += 1

        status = "OK" if is_correct else "MISS" if is_correct is not None else "?"
        print(f"  {item['id']:<20} true={true_cwe:<8} pred={pred_cwe:<8} "
              f"conf={confidence:.3f} {status}")

        results.append({
            "id": item["id"],
            "true_cwe": true_cwe,
            "predicted_cwe": pred_cwe,
            "confidence": float(confidence),
            "correct": is_correct,
            "probabilities": {cwe_names[j]: float(probabilities[i][j]) for j in range(3)},
        })

    accuracy = correct / total if total > 0 else 0
    print(f"\nRouting accuracy: {correct}/{total} ({accuracy*100:.1f}%)")

    # ─── Save ────────────────────────────────────────────────────────────
    output_data = {
        "timestamp": timestamp,
        "model": MODEL_NAME,
        "layer": LAYER,
        "n_train": len(X_train),
        "train_accuracy": train_acc,
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "n_neutral": len(neutral_prompts),
        "routing_accuracy": accuracy,
        "routing_correct": correct,
        "routing_total": total,
        "results": results,
    }

    # Save probe weights first (needed by Step 6 E2E pipeline)
    probe_path = DATA_DIR / f"cwe_probe_weights_{timestamp}.npz"
    np.savez(probe_path,
             coef=probe.coef_,
             intercept=probe.intercept_,
             classes=probe.classes_)
    print(f"Probe weights saved: {probe_path}")

    results_path = RESULTS_DIR / f"probe_routing_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved: {results_path}")
    print("\nProbe routing complete.")


if __name__ == "__main__":
    main()
