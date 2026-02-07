#!/usr/bin/env python3
"""
Experiment 9 — Phase 1: Extract Per-CWE Steering Vectors

For each model (Mistral-7B, Qwen-14B):
  - CWE-787: Extract direction from stored activations NPZ (no model load needed)
  - CWE-119/134: Load model, collect last-token activations at target layer, compute direction

Direction = mean(secure_activations) - mean(vulnerable_activations)
Uses same methodology as vector_correlation_analysis.py (last token position).

Reuses: shared/model_loader.py from 02-05_cross_model_cwe787_steering
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# ─── Path Setup ─────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent  # src/experiments/
CROSS_MODEL_DIR = EXPERIMENTS_DIR / "02-05_cross_model_cwe787_steering"
CROSS_CWE_DIR = EXPERIMENTS_DIR / "02-05_cross_cwe_steering"

sys.path.insert(0, str(CROSS_MODEL_DIR / "shared"))
from model_loader import ModelLoader

# ─── Configuration ──────────────────────────────────────────────────────────

MODELS = {
    "mistral7b": {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "layer": 31,
        "activations_npz": CROSS_MODEL_DIR / "experiment_4a_mistral7b" / "data" / "activations_20260205_042810.npz",
        "output_dir": SCRIPT_DIR / "mistral7b" / "data",
    },
    "qwen14b": {
        "name": "Qwen/Qwen2.5-14B-Instruct",
        "layer": 47,
        "activations_npz": CROSS_MODEL_DIR / "experiment_4c_qwen14b" / "data" / "activations_20260205_214033.npz",
        "output_dir": SCRIPT_DIR / "qwen14b" / "data",
    },
}

DATASETS = {
    "CWE-119": CROSS_CWE_DIR / "datasets" / "cwe119" / "data" / "cwe119_expanded_20260207_024627.jsonl",
    "CWE-134": CROSS_CWE_DIR / "datasets" / "cwe134" / "data" / "cwe134_expanded_20260207_024627.jsonl",
}


# ─── Utilities ──────────────────────────────────────────────────────────────

def load_dataset(path):
    """Load JSONL dataset."""
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def compute_direction(X, y):
    """Compute steering direction: mean(secure) - mean(vulnerable)."""
    secure_mean = X[y == 1].astype(np.float32).mean(axis=0)
    vuln_mean = X[y == 0].astype(np.float32).mean(axis=0)
    return (secure_mean - vuln_mean).astype(np.float32)


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ─── Phase 1a: CWE-787 from stored activations ─────────────────────────────

def extract_787_from_npz(model_key, config):
    """Extract CWE-787 direction from stored activations (no model load needed)."""
    npz_path = config["activations_npz"]
    layer = config["layer"]

    print(f"\n  Loading stored activations: {npz_path.name}")
    data = np.load(npz_path)

    layer_key = f"X_layer_{layer}"
    y_key = f"y_layer_{layer}"

    if layer_key not in data:
        available = [k for k in data.keys() if k.startswith("X_layer_")]
        raise KeyError(f"Layer {layer} not found in NPZ. Available: {available}")

    X = data[layer_key]
    y = data[y_key]

    print(f"  Shape: X={X.shape}, y={y.shape}")
    print(f"  Vulnerable: {sum(y == 0)}, Secure: {sum(y == 1)}")

    direction = compute_direction(X, y)
    norm = float(np.linalg.norm(direction))
    print(f"  CWE-787 direction norm: {norm:.4f}")

    return direction


# ─── Phase 1b: CWE-119/134 from model forward passes ───────────────────────

def extract_vectors_from_model(model_key, config, datasets):
    """Load model and extract CWE-119/134 vectors via forward passes."""
    layer = config["layer"]

    loader = ModelLoader(config["name"], quantization=None)
    model = loader.model
    tokenizer = loader.tokenizer
    device = loader.device
    hidden_size = loader.hidden_size

    directions = {}

    for cwe, dataset_path in datasets.items():
        print(f"\n  Extracting {cwe} direction at layer {layer}...")
        dataset = load_dataset(dataset_path)
        n_pairs = len(dataset)
        print(f"  Dataset: {n_pairs} pairs from {dataset_path.name}")

        # Register hook at target layer only (memory efficient)
        activations = {}

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                # Last token position, move to CPU immediately
                activations[layer_idx] = h[:, -1, :].detach().cpu()
            return hook_fn

        hook_handle = model.model.layers[layer].register_forward_hook(make_hook(layer))

        X = np.zeros((2 * n_pairs, hidden_size), dtype=np.float16)
        y = np.array([0] * n_pairs + [1] * n_pairs)

        # Collect vulnerable prompt activations (indices 0..N-1)
        for i, pair in enumerate(tqdm(dataset, desc=f"  {cwe} vulnerable")):
            inputs = tokenizer(pair['vulnerable'], return_tensors="pt").to(device)
            with torch.no_grad():
                _ = model(**inputs)
            X[i] = activations[layer].numpy().astype(np.float16)

        # Collect secure prompt activations (indices N..2N-1)
        for i, pair in enumerate(tqdm(dataset, desc=f"  {cwe} secure")):
            idx = n_pairs + i
            inputs = tokenizer(pair['secure'], return_tensors="pt").to(device)
            with torch.no_grad():
                _ = model(**inputs)
            X[idx] = activations[layer].numpy().astype(np.float16)

        hook_handle.remove()

        # Validate
        assert not np.any(np.isnan(X)), f"{cwe}: NaN detected in activations"
        assert not np.any(np.isinf(X)), f"{cwe}: Inf detected in activations"

        direction = compute_direction(X, y)
        norm = float(np.linalg.norm(direction))
        print(f"  {cwe} direction norm: {norm:.4f}, shape: {direction.shape}")

        directions[cwe] = direction

    loader.unload()
    return directions


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("Experiment 9 — Phase 1: Extract Per-CWE Steering Vectors")
    print(f"Models: {', '.join(MODELS.keys())}")
    print(f"CWEs: 787 (from NPZ), 119/134 (from model forward passes)")
    print(f"Timestamp: {timestamp}")
    print("=" * 70)

    all_directions = {}

    for model_key, config in MODELS.items():
        print(f"\n{'=' * 60}")
        print(f"MODEL: {model_key} ({config['name']})")
        print(f"Target layer: {config['layer']}")
        print(f"{'=' * 60}")

        output_dir = config["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        layer = config["layer"]

        # Phase 1a: CWE-787 from stored NPZ
        print("\n--- CWE-787 (from stored activations) ---")
        dir_787 = extract_787_from_npz(model_key, config)

        # Phase 1b: CWE-119/134 from model
        print(f"\n--- CWE-119/134 (loading {config['name']}) ---")
        new_dirs = extract_vectors_from_model(model_key, config, DATASETS)

        # Combine all directions
        directions = {"CWE-787": dir_787, **new_dirs}
        all_directions[model_key] = directions

        # Save direction vectors
        print(f"\n  Saving direction vectors to {output_dir}/")
        for cwe, direction in directions.items():
            cwe_num = cwe.split("-")[1]
            filename = f"direction_cwe{cwe_num}_L{layer}_{timestamp}.npy"
            save_path = output_dir / filename
            np.save(save_path, direction)
            print(f"    {filename} (norm={np.linalg.norm(direction):.4f})")

        # Cross-CWE cosine similarities
        print(f"\n  Cross-CWE Cosine Similarities ({model_key}):")
        cwes = sorted(directions.keys())
        for i, cwe_a in enumerate(cwes):
            for cwe_b in cwes[i + 1:]:
                sim = cosine_sim(directions[cwe_a], directions[cwe_b])
                print(f"    {cwe_a} <-> {cwe_b}: {sim:.4f}")

        # Save metadata
        meta = {
            "timestamp": timestamp,
            "model_name": config["name"],
            "layer": layer,
            "directions": {},
            "cosine_similarities": {},
        }
        for cwe, direction in directions.items():
            cwe_num = cwe.split("-")[1]
            meta["directions"][cwe] = {
                "filename": f"direction_cwe{cwe_num}_L{layer}_{timestamp}.npy",
                "norm": float(np.linalg.norm(direction)),
                "shape": list(direction.shape),
            }
        for i, cwe_a in enumerate(cwes):
            for cwe_b in cwes[i + 1:]:
                key = f"{cwe_a}_vs_{cwe_b}"
                meta["cosine_similarities"][key] = cosine_sim(
                    directions[cwe_a], directions[cwe_b]
                )

        meta_path = output_dir / f"vector_metadata_{timestamp}.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"  Metadata saved: {meta_path.name}")

    # ─── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SUMMARY: All Direction Vectors")
    print(f"{'=' * 70}")

    for model_key in MODELS:
        layer = MODELS[model_key]["layer"]
        print(f"\n  {model_key} (layer {layer}):")
        for cwe in sorted(all_directions[model_key].keys()):
            d = all_directions[model_key][cwe]
            print(f"    {cwe}: norm={np.linalg.norm(d):.4f}, dim={d.shape[0]}")

    print(f"\n  Cross-model note: Mistral (4096-dim) and Qwen (5120-dim) vectors")
    print(f"  are NOT directly comparable due to different hidden dimensions.")

    print("\nPhase 1 complete.")


if __name__ == "__main__":
    main()
