#!/usr/bin/env python3
"""
Experiment 12: Mistral-7B Linear Probe Layer Sweep (Mechanistic Replication)

Replicates the "hierarchical convergence" finding from Llama-3.1-8B-Instruct
on Mistral-7B-Instruct-v0.3:
  - Linear probes at layers [0, 4, 8, 12, 16, 20, 24, 28, 31]
  - LOBO cross-validation (7 folds, leave-one-base-id-out)
  - Steering vector norms per layer
  - Logit lens analysis for secure token probability trajectory

Datasets: CWE-787 (C, sprintf/snprintf) + CWE-89 (Python, SQL injection)
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Add shared utilities to path
SHARED_DIR = Path(__file__).parent.parent / "02-05_cross_cwe_steering" / "shared"
sys.path.insert(0, str(SHARED_DIR))

# ── Configuration ──────────────────────────────────────────────────────────
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
TARGET_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]

CWE787_DATASET = (
    Path(__file__).parent.parent
    / "01-12_cwe787_dataset_expansion" / "data"
    / "cwe787_expanded_20260112_143316.jsonl"
)
CWE89_DATASET = (
    Path(__file__).parent.parent
    / "02-05_cross_cwe_steering" / "datasets" / "cwe89" / "data"
    / "cwe89_expanded_20260209_221808.jsonl"
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Logit lens: number of representative prompts per CWE
N_LOGIT_LENS_PROMPTS = 5


# ── Data Loading ───────────────────────────────────────────────────────────
def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def normalize_dataset(items, cwe_type):
    """Normalize both dataset formats to have 'vulnerable' and 'secure' keys."""
    normalized = []
    for item in items:
        if cwe_type == "CWE-787":
            normalized.append({
                "id": item["id"],
                "base_id": item["base_id"],
                "vulnerable": item["vulnerable"],
                "secure": item["secure"],
            })
        elif cwe_type == "CWE-89":
            normalized.append({
                "id": item["pair_id"],
                "base_id": item["base_id"],
                "vulnerable": item["insecure_prompt"],
                "secure": item["secure_prompt"],
            })
    return normalized


# ── Activation Collection ──────────────────────────────────────────────────
def collect_activations(model, tokenizer, dataset, target_layers, device,
                        use_chat_template=True):
    """
    Collect last-token activations at target layers for all prompts.

    Returns:
        activations: dict[layer] -> np.ndarray of shape (2*N, hidden_size) in float16
        y: np.ndarray of shape (2*N,) where 0=vulnerable, 1=secure
        metadata: list of dicts with id, base_id, prompt_type
    """
    n_pairs = len(dataset)
    hidden_size = model.config.hidden_size
    n_total = 2 * n_pairs

    activations = {layer: np.zeros((n_total, hidden_size), dtype=np.float16)
                   for layer in target_layers}
    metadata = []

    hooks = []
    hook_storage = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            hook_storage[layer_idx] = h[:, -1, :].detach().cpu()
        return hook_fn

    # Register hooks
    for layer_idx in target_layers:
        layer = model.model.layers[layer_idx]
        hook = layer.register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)

    def format_prompt(text):
        if use_chat_template:
            messages = [{"role": "user", "content": text}]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return text

    # Collect vulnerable prompts (indices 0..N-1)
    print(f"  Collecting vulnerable activations ({n_pairs} prompts)...")
    for i, pair in enumerate(dataset):
        prompt = format_prompt(pair["vulnerable"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model(**inputs)
        for layer in target_layers:
            activations[layer][i] = hook_storage[layer].numpy().astype(np.float16)
        metadata.append({"index": i, "id": pair["id"], "base_id": pair["base_id"],
                         "prompt_type": "vulnerable"})
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{n_pairs}")

    # Collect secure prompts (indices N..2N-1)
    print(f"  Collecting secure activations ({n_pairs} prompts)...")
    for i, pair in enumerate(dataset):
        idx = n_pairs + i
        prompt = format_prompt(pair["secure"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model(**inputs)
        for layer in target_layers:
            activations[layer][idx] = hook_storage[layer].numpy().astype(np.float16)
        metadata.append({"index": idx, "id": pair["id"], "base_id": pair["base_id"],
                         "prompt_type": "secure"})
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{n_pairs}")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    y = np.array([0] * n_pairs + [1] * n_pairs)
    return activations, y, metadata, n_pairs


# ── LOBO Probe Training ───────────────────────────────────────────────────
def train_lobo_probes(activations, y, metadata, n_pairs, target_layers, base_ids):
    """
    Train LogisticRegression probes with Leave-One-Base-ID-Out CV.

    Returns:
        results: list of dicts per layer with accuracy, std, vector_norm
    """
    # Build base_id index
    vuln_base_ids = [m["base_id"] for m in metadata[:n_pairs]]
    sec_base_ids = [m["base_id"] for m in metadata[n_pairs:]]

    results = []

    for layer in target_layers:
        X = activations[layer].astype(np.float32)
        fold_accs = []
        fold_norms = []

        for held_out in base_ids:
            # Train indices: all pairs NOT matching held_out base_id
            train_vuln = [i for i, bid in enumerate(vuln_base_ids) if bid != held_out]
            train_sec = [i + n_pairs for i, bid in enumerate(sec_base_ids) if bid != held_out]
            train_idx = train_vuln + train_sec

            # Test indices: pairs matching held_out base_id
            test_vuln = [i for i, bid in enumerate(vuln_base_ids) if bid == held_out]
            test_sec = [i + n_pairs for i, bid in enumerate(sec_base_ids) if bid == held_out]
            test_idx = test_vuln + test_sec

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Standardize
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # Train probe
            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(X_train_s, y_train)
            acc = clf.score(X_test_s, y_test)
            fold_accs.append(acc)

            # Steering vector norm (mean_secure - mean_vulnerable)
            secure_mean = X_train[y_train == 1].mean(axis=0)
            vuln_mean = X_train[y_train == 0].mean(axis=0)
            direction = secure_mean - vuln_mean
            fold_norms.append(float(np.linalg.norm(direction)))

        results.append({
            "layer": layer,
            "accuracy_mean": float(np.mean(fold_accs)),
            "accuracy_std": float(np.std(fold_accs)),
            "accuracy_per_fold": [float(a) for a in fold_accs],
            "vector_norm_mean": float(np.mean(fold_norms)),
            "vector_norm_std": float(np.std(fold_norms)),
            "vector_norm_per_fold": [float(n) for n in fold_norms],
        })

        print(f"    Layer {layer:2d}: acc={np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f}  "
              f"norm={np.mean(fold_norms):.2f}")

    return results


# ── Logit Lens Analysis ───────────────────────────────────────────────────
def run_logit_lens(model, tokenizer, dataset, target_layers, device,
                   secure_tokens, cwe_type, n_prompts=5):
    """
    Project residual stream through unembedding at each layer.

    Args:
        secure_tokens: dict mapping token_label -> token_id
    Returns:
        results: dict with per-prompt, per-layer probabilities
    """
    lm_head = model.lm_head
    final_norm = model.model.norm

    # Select representative prompts (first n_prompts base_ids, first variation each)
    base_ids_seen = set()
    selected = []
    for pair in dataset:
        if pair["base_id"] not in base_ids_seen and len(selected) < n_prompts:
            selected.append(pair)
            base_ids_seen.add(pair["base_id"])

    results = {"secure_tokens": {k: int(v) for k, v in secure_tokens.items()},
               "prompts": []}

    for pair in selected:
        prompt_result = {"base_id": pair["base_id"], "id": pair["id"],
                         "secure": {}, "vulnerable": {}}

        for prompt_type in ["secure", "vulnerable"]:
            text = pair[prompt_type]
            messages = [{"role": "user", "content": text}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Collect residual stream at target layers
            residual_stream = {}
            hooks = []

            def make_hook(layer_idx):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                    else:
                        h = output
                    residual_stream[layer_idx] = h[:, -1, :].detach().clone()
                return hook_fn

            for layer_idx in target_layers:
                layer = model.model.layers[layer_idx]
                hook = layer.register_forward_hook(make_hook(layer_idx))
                hooks.append(hook)

            inputs = tokenizer(formatted, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            for hook in hooks:
                hook.remove()

            # Project each layer through unembedding
            layer_probs = {}
            for layer_idx in target_layers:
                residual = residual_stream[layer_idx]
                normed = final_norm(residual)
                logits = lm_head(normed).squeeze(0)
                probs = torch.softmax(logits, dim=-1)

                token_probs = {}
                for token_label, token_id in secure_tokens.items():
                    token_probs[token_label] = float(probs[token_id].item())

                layer_probs[str(layer_idx)] = token_probs

            # Also get final output probs
            final_logits = outputs.logits[0, -1, :]
            final_probs = torch.softmax(final_logits, dim=-1)
            final_token_probs = {}
            for token_label, token_id in secure_tokens.items():
                final_token_probs[token_label] = float(final_probs[token_id].item())
            layer_probs["final"] = final_token_probs

            prompt_result[prompt_type] = layer_probs

        results["prompts"].append(prompt_result)

    return results


def find_best_secure_tokens(tokenizer, cwe_type):
    """Find token IDs for secure-code tokens to track in logit lens."""
    if cwe_type == "CWE-787":
        # C buffer overflow: track snprintf
        candidates = {
            "snprintf": " snprintf",
            "snprintf_no_space": "snprintf",
        }
    elif cwe_type == "CWE-89":
        # SQL injection: track parameterized query tokens
        candidates = {
            "question_mark": "?",
            "execute": "execute",
            "parameterized": "parameterized",
            "placeholder": "placeholder",
        }

    tokens = {}
    for label, text in candidates.items():
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            tokens[label] = ids[0]
            decoded = tokenizer.decode([ids[0]])
            print(f"    {label}: '{text}' -> token_id={ids[0]} (decodes to '{decoded}')")

    return tokens


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 70)
    print("EXPERIMENT 12: Mistral-7B Linear Probe Layer Sweep")
    print(f"Timestamp: {timestamp}")
    print("=" * 70)

    # Load model using shared ModelLoader
    from model_loader import ModelLoader
    loader = ModelLoader(MODEL_NAME)
    model = loader.model
    tokenizer = loader.tokenizer
    device = loader.device

    all_results = {
        "experiment": "12_mistral_probe_sweep",
        "model": MODEL_NAME,
        "target_layers": TARGET_LAYERS,
        "timestamp": timestamp,
        "cwe_results": {},
    }

    # Process each CWE
    for cwe_type, dataset_path in [("CWE-787", CWE787_DATASET),
                                     ("CWE-89", CWE89_DATASET)]:
        print(f"\n{'=' * 70}")
        print(f"Processing {cwe_type}")
        print(f"Dataset: {dataset_path}")
        print("=" * 70)

        # Load and normalize dataset
        raw_items = load_jsonl(dataset_path)
        dataset = normalize_dataset(raw_items, cwe_type)
        base_ids = sorted(set(item["base_id"] for item in dataset))
        print(f"  {len(dataset)} pairs, {len(base_ids)} base_ids: {base_ids}")

        # Step 1: Collect activations
        print(f"\n--- Step 1: Collect Activations at {len(TARGET_LAYERS)} layers ---")
        activations, y, metadata, n_pairs = collect_activations(
            model, tokenizer, dataset, TARGET_LAYERS, device, use_chat_template=True
        )

        # Save activations as NPZ
        npz_path = RESULTS_DIR / f"activations_{cwe_type}_{timestamp}.npz"
        npz_data = {}
        for layer in TARGET_LAYERS:
            npz_data[f"X_layer_{layer}"] = activations[layer]
            npz_data[f"y_layer_{layer}"] = y
        np.savez_compressed(npz_path, **npz_data)
        print(f"  Saved activations: {npz_path} ({npz_path.stat().st_size / 1e6:.1f} MB)")

        # Save metadata
        meta_path = RESULTS_DIR / f"metadata_{cwe_type}_{timestamp}.json"
        with open(meta_path, "w") as f:
            json.dump({
                "cwe": cwe_type,
                "n_pairs": n_pairs,
                "n_total": 2 * n_pairs,
                "layers": TARGET_LAYERS,
                "base_ids": base_ids,
                "metadata": metadata,
            }, f, indent=2)

        # Step 2: Train LOBO probes
        print(f"\n--- Step 2: LOBO Probe Training ({len(base_ids)} folds) ---")
        probe_results = train_lobo_probes(
            activations, y, metadata, n_pairs, TARGET_LAYERS, base_ids
        )

        # Step 3: Logit lens
        print(f"\n--- Step 3: Logit Lens Analysis ---")
        print(f"  Finding secure tokens for {cwe_type}...")
        secure_tokens = find_best_secure_tokens(tokenizer, cwe_type)
        logit_lens_results = run_logit_lens(
            model, tokenizer, dataset, TARGET_LAYERS, device,
            secure_tokens, cwe_type, n_prompts=N_LOGIT_LENS_PROMPTS
        )

        # Store results
        all_results["cwe_results"][cwe_type] = {
            "n_pairs": n_pairs,
            "n_base_ids": len(base_ids),
            "base_ids": base_ids,
            "probe_results": probe_results,
            "logit_lens": logit_lens_results,
            "activations_file": str(npz_path),
            "metadata_file": str(meta_path),
        }

    # ── Summary Table ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: Probe Accuracy by Layer")
    print("=" * 70)
    print(f"{'Layer':>6} | {'CWE-787 Acc':>14} | {'CWE-787 Norm':>13} | "
          f"{'CWE-89 Acc':>13} | {'CWE-89 Norm':>12}")
    print("-" * 70)

    for i, layer in enumerate(TARGET_LAYERS):
        r787 = all_results["cwe_results"]["CWE-787"]["probe_results"][i]
        r89 = all_results["cwe_results"]["CWE-89"]["probe_results"][i]
        print(f"{layer:6d} | {r787['accuracy_mean']:.4f} +/- {r787['accuracy_std']:.4f} | "
              f"{r787['vector_norm_mean']:12.2f} | "
              f"{r89['accuracy_mean']:.4f} +/- {r89['accuracy_std']:.4f} | "
              f"{r89['vector_norm_mean']:11.2f}")

    # ── Logit Lens Summary ─────────────────────────────────────────────────
    for cwe_type in ["CWE-787", "CWE-89"]:
        print(f"\nLogit Lens Summary ({cwe_type}):")
        ll = all_results["cwe_results"][cwe_type]["logit_lens"]
        token_labels = list(ll["secure_tokens"].keys())
        primary_token = token_labels[0]  # Use first token for summary

        print(f"  Tracking P({primary_token}) across layers:")
        print(f"  {'Layer':>6} | {'Secure P':>12} | {'Vulnerable P':>14} | {'Diff':>10}")
        print(f"  {'-'*50}")

        for prompt_data in ll["prompts"][:1]:  # Show first prompt as example
            for layer in TARGET_LAYERS:
                s_prob = prompt_data["secure"][str(layer)][primary_token]
                v_prob = prompt_data["vulnerable"][str(layer)][primary_token]
                diff = s_prob - v_prob
                print(f"  {layer:6d} | {s_prob:12.6f} | {v_prob:14.6f} | {diff:+10.6f}")

            # Final
            s_final = prompt_data["secure"]["final"][primary_token]
            v_final = prompt_data["vulnerable"]["final"][primary_token]
            print(f"  {'final':>6} | {s_final:12.6f} | {v_final:14.6f} | {s_final-v_final:+10.6f}")

    # ── Save full results ──────────────────────────────────────────────────
    results_path = RESULTS_DIR / f"probe_sweep_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nResults saved: {results_path}")

    # ── Key Finding ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("KEY FINDING: Does Mistral show hierarchical convergence?")
    print("=" * 70)

    for cwe_type in ["CWE-787", "CWE-89"]:
        probes = all_results["cwe_results"][cwe_type]["probe_results"]
        early_acc = probes[0]["accuracy_mean"]  # Layer 0
        mid_acc = probes[4]["accuracy_mean"]    # Layer 16
        late_acc = probes[-1]["accuracy_mean"]  # Layer 31
        print(f"\n  {cwe_type}:")
        print(f"    Layer  0 probe accuracy: {early_acc:.4f}")
        print(f"    Layer 16 probe accuracy: {mid_acc:.4f}")
        print(f"    Layer 31 probe accuracy: {late_acc:.4f}")
        if early_acc > 0.85:
            print(f"    -> High early-layer accuracy ({early_acc:.2%}): consistent with Llama pattern")
        else:
            print(f"    -> Low early-layer accuracy ({early_acc:.2%}): DIFFERS from Llama pattern")

    print(f"\nDone! Total results saved to: {results_path}")
    return all_results


if __name__ == "__main__":
    results = main()
