#!/usr/bin/env python3
"""
Experiment 10, Step 2: Extract Steering Vectors for Python CWEs

For each CWE (89, 78, 79):
  1. Load 105 prompt pairs (insecure_prompt, secure_prompt)
  2. Format with chat template (same as baseline generation)
  3. Forward pass through model, collect Layer 31 last-token activations
  4. Compute mean-difference vector: secure_mean - insecure_mean
  5. Save as .npy file

Also computes cosine similarity matrix between all 6 vectors (3 C + 3 Python)
to preview cross-language transfer potential.

Reuses: ModelLoader from shared/
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

sys.path.insert(0, str(EXPERIMENT_DIR / "shared"))
from model_loader import ModelLoader

# ─── Configuration ───────────────────────────────────────────────────────────

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LAYER = 31

DATASET_FILES = {
    "CWE-89": DATASETS_DIR / "cwe89" / "data" / "cwe89_expanded_20260209_221808.jsonl",
    "CWE-78": DATASETS_DIR / "cwe78" / "data" / "cwe78_expanded_20260209_221808.jsonl",
    "CWE-79": DATASETS_DIR / "cwe79" / "data" / "cwe79_expanded_20260209_221808.jsonl",
}

# C-language directions for cross-language similarity
C_DIRECTION_DIR = EXPERIMENT_DIR / "cross_cwe_analysis" / "data"
C_DIRECTIONS = {
    "CWE-787": "direction_cwe787_L31_20260206_031901.npy",
    "CWE-119": "direction_cwe119_L31_20260206_031901.npy",
    "CWE-134": "direction_cwe134_L31_20260206_031901.npy",
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


@torch.no_grad()
def collect_l31_activations(model, tokenizer, prompts, layer=31):
    """Collect last-token activations at specified layer for a list of prompts.

    Returns: numpy array of shape (n_prompts, hidden_size) in float32.
    """
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


def compute_direction(X_insecure, X_secure):
    """Compute mean-difference direction: secure_mean - insecure_mean."""
    secure_mean = X_secure.mean(axis=0)
    insecure_mean = X_insecure.mean(axis=0)
    return (secure_mean - insecure_mean).astype(np.float32)


def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Experiment 10, Step 2: Extract Steering Vectors (Python CWEs)")
    print(f"Model: {MODEL_NAME}")
    print(f"Layer: {LAYER}")
    print("=" * 70)

    # ─── Load model ──────────────────────────────────────────────────────
    print("\nLoading model...")
    loader = ModelLoader(MODEL_NAME)
    model = loader.model
    tokenizer = loader.tokenizer

    # ─── Extract vectors per CWE ─────────────────────────────────────────
    python_directions = {}

    for cwe, dataset_path in DATASET_FILES.items():
        dataset = load_jsonl(dataset_path)
        n = len(dataset)
        print(f"\n{'='*60}")
        print(f"{cwe}: {n} prompt pairs")
        print(f"{'='*60}")

        # Format prompts with chat template
        insecure_prompts = [format_chat_prompt(tokenizer, item["insecure_prompt"])
                           for item in dataset]
        secure_prompts = [format_chat_prompt(tokenizer, item["secure_prompt"])
                         for item in dataset]

        # Collect activations
        print(f"Collecting insecure prompt activations...")
        X_insecure = collect_l31_activations(model, tokenizer, insecure_prompts, LAYER)
        print(f"  Shape: {X_insecure.shape}")

        print(f"Collecting secure prompt activations...")
        X_secure = collect_l31_activations(model, tokenizer, secure_prompts, LAYER)
        print(f"  Shape: {X_secure.shape}")

        # Compute direction
        direction = compute_direction(X_insecure, X_secure)
        norm = np.linalg.norm(direction)
        print(f"Direction norm: {norm:.4f}")

        # Save
        cwe_num = cwe.split("-")[1]
        npy_path = DATA_DIR / f"direction_cwe{cwe_num}_L{LAYER}_{timestamp}.npy"
        np.save(npy_path, direction)
        print(f"Saved: {npy_path}")

        # Also save full activations for potential LOBO use
        npz_path = DATA_DIR / f"activations_cwe{cwe_num}_L{LAYER}_{timestamp}.npz"
        # y: 0=insecure, 1=secure (matching C convention)
        X_all = np.vstack([X_insecure, X_secure])
        y_all = np.array([0] * n + [1] * n)
        base_ids = [item["base_id"] for item in dataset]

        np.savez_compressed(npz_path,
                           X=X_all, y=y_all,
                           base_ids=np.array(base_ids))
        print(f"Activations saved: {npz_path} ({npz_path.stat().st_size / 1e6:.1f} MB)")

        python_directions[cwe] = direction

    loader.unload()

    # ─── Cross-language similarity analysis ──────────────────────────────
    print(f"\n{'='*70}")
    print("CROSS-LANGUAGE SIMILARITY (Cosine)")
    print(f"{'='*70}")

    # Load C directions
    c_directions = {}
    for cwe, fname in C_DIRECTIONS.items():
        path = C_DIRECTION_DIR / fname
        c_directions[cwe] = np.load(path).astype(np.float64)

    # All 6 vectors
    all_labels = list(c_directions.keys()) + list(python_directions.keys())
    c_labels = list(c_directions.keys())
    py_labels = list(python_directions.keys())

    all_vecs = ([c_directions[k].astype(np.float64) for k in c_labels] +
                [python_directions[k].astype(np.float64) for k in py_labels])

    n_vecs = len(all_vecs)
    sim_matrix = np.zeros((n_vecs, n_vecs))
    for i in range(n_vecs):
        for j in range(n_vecs):
            sim_matrix[i, j] = cosine_similarity(all_vecs[i], all_vecs[j])

    # Print similarity matrix
    header = [f"C-{l.split('-')[1]}" for l in c_labels] + [f"Py-{l.split('-')[1]}" for l in py_labels]
    print(f"\n{'':>12}", end="")
    for h in header:
        print(f"{h:>10}", end="")
    print()
    print("-" * (12 + 10 * n_vecs))

    for i, label in enumerate(header):
        print(f"{label:>12}", end="")
        for j in range(n_vecs):
            val = sim_matrix[i, j]
            print(f"{val:>10.3f}", end="")
        print()

    # Key cross-language pairs
    print(f"\nKey cross-language correlations:")
    # For each Python CWE, show similarity to each C CWE
    for i_py, py_cwe in enumerate(py_labels):
        py_idx = len(c_labels) + i_py
        sims = [(c_labels[j], sim_matrix[py_idx, j]) for j in range(len(c_labels))]
        sims.sort(key=lambda x: -abs(x[1]))
        sims_str = ", ".join(f"{c}={s:.3f}" for c, s in sims)
        print(f"  Py-{py_cwe}: {sims_str}")

    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "model": MODEL_NAME,
        "layer": LAYER,
        "python_cwe_directions": {
            cwe: f"direction_cwe{cwe.split('-')[1]}_L{LAYER}_{timestamp}.npy"
            for cwe in python_directions
        },
        "c_cwe_directions": {
            cwe: str(C_DIRECTION_DIR / fname) for cwe, fname in C_DIRECTIONS.items()
        },
        "direction_norms": {
            f"C-{cwe}": float(np.linalg.norm(c_directions[cwe]))
            for cwe in c_directions
        } | {
            f"Py-{cwe}": float(np.linalg.norm(python_directions[cwe]))
            for cwe in python_directions
        },
        "similarity_matrix": sim_matrix.tolist(),
        "similarity_labels": header,
    }

    meta_path = DATA_DIR / f"vector_metadata_{timestamp}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved: {meta_path}")
    print("\nVector extraction complete.")


if __name__ == "__main__":
    main()
