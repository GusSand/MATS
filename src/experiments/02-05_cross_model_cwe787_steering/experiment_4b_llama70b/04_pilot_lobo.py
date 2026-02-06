#!/usr/bin/env python3
"""
Experiment 4B Step 4: Llama-70B Pilot LOBO (2 folds)

Quick validation that steering works before full LOBO.
Wider alpha grid to account for larger hidden dim.
"""

import sys
import json
import glob
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from experiment_config import (
    MODEL_NAME, QUANTIZATION, DATASET_PATH, DATA_DIR, FOLD_RESULTS_DIR,
    BASE_IDS, ALPHA_GRID, GENERATION_CONFIG, GENERATIONS_PER_PROMPT, PILOT_FOLDS,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
from model_loader import ModelLoader
from steering_generator import SteeringGenerator

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "01-12_llama8b_cwe787_baseline_behavior"))
from scoring import score_completion
from refusal_detection import detect_refusal


def load_dataset():
    dataset = []
    with open(DATASET_PATH) as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


def load_metadata():
    meta_files = sorted(glob.glob(str(DATA_DIR / "metadata_*.json")))
    with open(meta_files[-1]) as f:
        return json.load(f)


def load_activations_at_layer(layer: int):
    npz_files = sorted(glob.glob(str(DATA_DIR / "activations_*.npz")))
    data = np.load(npz_files[-1])
    X = data[f'X_layer_{layer}'].astype(np.float32)
    y = data[f'y_layer_{layer}']
    return X, y


def get_best_layer():
    sweep_path = DATA_DIR / "layer_sweep_results.json"
    with open(sweep_path) as f:
        return json.load(f)['best_layer']


def get_lobo_splits(metadata):
    vulnerable_metadata = metadata['vulnerable_metadata']
    secure_metadata = metadata['secure_metadata']
    n_pairs = metadata['n_pairs']
    folds = []
    for held_out_base_id in BASE_IDS:
        train_vuln_indices = [
            i for i, m in enumerate(vulnerable_metadata)
            if m['base_id'] != held_out_base_id
        ]
        train_sec_indices = [
            i + n_pairs for i, m in enumerate(secure_metadata)
            if m['base_id'] != held_out_base_id
        ]
        test_indices = [
            i for i, m in enumerate(vulnerable_metadata)
            if m['base_id'] == held_out_base_id
        ]
        folds.append({
            'fold_id': held_out_base_id,
            'train_vuln_indices': train_vuln_indices,
            'train_sec_indices': train_sec_indices,
            'test_indices': test_indices,
            'n_train': len(train_vuln_indices) + len(train_sec_indices),
            'n_test': len(test_indices),
        })
    return folds


def compute_fold_direction(X, y, fold):
    train_indices = fold['train_vuln_indices'] + fold['train_sec_indices']
    X_train = X[train_indices]
    y_train = y[train_indices]
    secure_mean = X_train[y_train == 1].mean(axis=0)
    vulnerable_mean = X_train[y_train == 0].mean(axis=0)
    return (secure_mean - vulnerable_mean).astype(np.float32)


def main():
    torch.manual_seed(42)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("EXPERIMENT 4B STEP 4: Llama-70B Pilot LOBO (2 folds)")
    print(f"Model: {MODEL_NAME}")
    print(f"Quantization: {QUANTIZATION}")
    print(f"Timestamp: {timestamp}")
    print("=" * 60)

    FOLD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset()
    metadata = load_metadata()
    best_layer = get_best_layer()
    X, y = load_activations_at_layer(best_layer)
    folds = get_lobo_splits(metadata)[:PILOT_FOLDS]

    print(f"Dataset: {len(dataset)} pairs")
    print(f"Best layer: {best_layer}")
    print(f"Activations shape: {X.shape}")
    print(f"Running {len(folds)} pilot folds")

    loader = ModelLoader(MODEL_NAME, quantization=QUANTIZATION)
    generator = SteeringGenerator(loader)

    all_fold_results = []

    for fold in folds:
        fold_id = fold['fold_id']
        print(f"\n{'='*60}")
        print(f"FOLD: {fold_id}")
        print(f"{'='*60}")

        direction = compute_fold_direction(X, y, fold)
        print(f"Direction norm: {np.linalg.norm(direction):.4f}")

        test_prompts = [dataset[i] for i in fold['test_indices']]
        print(f"Test prompts: {len(test_prompts)}")

        fold_results = {
            'fold_id': fold_id,
            'n_train': fold['n_train'],
            'n_test': fold['n_test'],
            'direction_norm': float(np.linalg.norm(direction)),
            'layer': best_layer,
            'alpha_results': {},
        }

        total_gens = len(ALPHA_GRID) * len(test_prompts) * GENERATIONS_PER_PROMPT
        pbar = tqdm(total=total_gens, desc=f"Fold {fold_id}")

        for alpha in ALPHA_GRID:
            alpha_key = str(alpha)
            fold_results['alpha_results'][alpha_key] = []

            for item in test_prompts:
                prompt = item['vulnerable']
                vuln_type = item['vulnerability_type']

                for gen_idx in range(GENERATIONS_PER_PROMPT):
                    output = generator.generate_with_steering(
                        prompt=prompt,
                        direction=direction,
                        layer=best_layer,
                        alpha=alpha,
                        temperature=GENERATION_CONFIG['temperature'],
                        top_p=GENERATION_CONFIG['top_p'],
                        max_tokens=GENERATION_CONFIG['max_new_tokens'],
                    )

                    score_result = score_completion(output, vuln_type)
                    is_refusal, _ = detect_refusal(output)

                    fold_results['alpha_results'][alpha_key].append({
                        'id': item['id'],
                        'base_id': item['base_id'],
                        'vulnerability_type': vuln_type,
                        'gen_idx': gen_idx,
                        'output': output[:500],
                        'strict_label': score_result.strict_label,
                        'expanded_label': score_result.expanded_label,
                        'is_refusal': is_refusal,
                    })
                    pbar.update(1)

        pbar.close()

        summary = {}
        for alpha_key, items in fold_results['alpha_results'].items():
            n = len(items)
            strict_secure = sum(1 for r in items if r['strict_label'] == 'secure')
            summary[alpha_key] = {'n': n, 'strict_secure_rate': strict_secure / n if n > 0 else 0}
        fold_results['summary'] = summary
        all_fold_results.append(fold_results)

        fold_path = FOLD_RESULTS_DIR / f"pilot_fold_{fold_id}_{timestamp}.json"
        with open(fold_path, 'w') as f:
            json.dump(fold_results, f, indent=2)
        print(f"Saved: {fold_path}")

    # Aggregate
    print("\n" + "=" * 60)
    print("PILOT RESULTS")
    print("=" * 60)

    baseline_rate = 0.0
    best_rate = 0.0
    best_alpha = 0.0

    for alpha in ALPHA_GRID:
        alpha_key = str(alpha)
        all_items = []
        for fr in all_fold_results:
            all_items.extend(fr['alpha_results'][alpha_key])
        n = len(all_items)
        secure = sum(1 for r in all_items if r['strict_label'] == 'secure')
        rate = secure / n if n > 0 else 0

        if alpha == 0.0:
            baseline_rate = rate
        if rate > best_rate:
            best_rate = rate
            best_alpha = alpha

        print(f"  alpha={alpha:<5}: STRICT secure = {secure}/{n} ({rate*100:.1f}%)")

    improvement = best_rate - baseline_rate
    print(f"\nBaseline: {baseline_rate*100:.1f}%")
    print(f"Best: {best_rate*100:.1f}% at alpha={best_alpha}")
    print(f"Improvement: {improvement*100:.1f}pp")

    pilot_summary = {
        'timestamp': timestamp,
        'model': MODEL_NAME,
        'quantization': QUANTIZATION,
        'layer': best_layer,
        'n_folds': len(folds),
        'baseline_rate': baseline_rate,
        'best_rate': best_rate,
        'best_alpha': best_alpha,
        'improvement_pp': improvement * 100,
        'gate_passed': improvement > 0.10,
    }
    summary_path = DATA_DIR / f"pilot_summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(pilot_summary, f, indent=2)

    if improvement > 0.10:
        print(f"\n*** GATE PASSED: {improvement*100:.1f}pp > 10pp ***")
    else:
        print(f"\n*** GATE FAILED: {improvement*100:.1f}pp <= 10pp ***")

    return pilot_summary


if __name__ == "__main__":
    main()
