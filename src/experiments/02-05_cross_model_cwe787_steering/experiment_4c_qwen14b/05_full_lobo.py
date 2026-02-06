#!/usr/bin/env python3
"""
Experiment 4C Step 5: Qwen2.5-14B Full 7-Fold LOBO

Full cross-validation with all 7 LOBO folds.
Only run if pilot gate passed (>10pp improvement).
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
    BASE_IDS, ALPHA_GRID, GENERATION_CONFIG, GENERATIONS_PER_PROMPT,
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


def summarize_fold(fold_results):
    summaries = {}
    for alpha_key, items in fold_results['alpha_results'].items():
        n = len(items)
        strict_secure = sum(1 for r in items if r['strict_label'] == 'secure')
        strict_insecure = sum(1 for r in items if r['strict_label'] == 'insecure')
        expanded_secure = sum(1 for r in items if r['expanded_label'] == 'secure')
        refusals = sum(1 for r in items if r['is_refusal'])
        summaries[alpha_key] = {
            'n': n,
            'strict': {
                'secure': strict_secure, 'insecure': strict_insecure,
                'other': n - strict_secure - strict_insecure,
                'secure_rate': strict_secure / n if n > 0 else 0,
                'insecure_rate': strict_insecure / n if n > 0 else 0,
            },
            'expanded': {
                'secure': expanded_secure,
                'secure_rate': expanded_secure / n if n > 0 else 0,
            },
            'refusal_rate': refusals / n if n > 0 else 0,
        }
    return summaries


def aggregate_all_folds(all_fold_results):
    alpha_keys = list(all_fold_results[0]['alpha_results'].keys())
    aggregated = {}
    for alpha_key in alpha_keys:
        all_items = []
        for fr in all_fold_results:
            all_items.extend(fr['alpha_results'][alpha_key])
        n = len(all_items)
        strict_secure = sum(1 for r in all_items if r['strict_label'] == 'secure')
        strict_insecure = sum(1 for r in all_items if r['strict_label'] == 'insecure')
        expanded_secure = sum(1 for r in all_items if r['expanded_label'] == 'secure')
        refusals = sum(1 for r in all_items if r['is_refusal'])
        aggregated[alpha_key] = {
            'n': n,
            'strict_secure_rate': strict_secure / n if n > 0 else 0,
            'strict_insecure_rate': strict_insecure / n if n > 0 else 0,
            'expanded_secure_rate': expanded_secure / n if n > 0 else 0,
            'refusal_rate': refusals / n if n > 0 else 0,
        }
    return aggregated


def main():
    torch.manual_seed(42)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("EXPERIMENT 4C STEP 5: Qwen2.5-14B Full 7-Fold LOBO")
    print(f"Model: {MODEL_NAME}")
    print(f"Timestamp: {timestamp}")
    print("=" * 60)

    FOLD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset()
    metadata = load_metadata()
    best_layer = get_best_layer()
    X, y = load_activations_at_layer(best_layer)
    folds = get_lobo_splits(metadata)

    print(f"Dataset: {len(dataset)} pairs")
    print(f"Layer: {best_layer}")
    print(f"Folds: {len(folds)}")
    print(f"Alpha grid: {ALPHA_GRID}")

    loader = ModelLoader(MODEL_NAME, quantization=QUANTIZATION)
    generator = SteeringGenerator(loader)

    all_fold_results = []
    all_fold_summaries = []

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

        fold_summary = summarize_fold(fold_results)
        fold_results['summary'] = fold_summary
        all_fold_results.append(fold_results)
        all_fold_summaries.append({'fold_id': fold_id, 'summary': fold_summary})

        fold_path = FOLD_RESULTS_DIR / f"fold_{fold_id}_{timestamp}.json"
        with open(fold_path, 'w') as f:
            json.dump(fold_results, f, indent=2)
        print(f"Saved: {fold_path}")

    # Aggregate
    aggregated = aggregate_all_folds(all_fold_results)

    output = {
        'timestamp': timestamp,
        'config': {
            'model': MODEL_NAME,
            'quantization': QUANTIZATION,
            'layer': best_layer,
            'alpha_grid': ALPHA_GRID,
            'generations_per_prompt': GENERATIONS_PER_PROMPT,
            'generation_config': GENERATION_CONFIG,
            'n_folds': len(folds),
            'base_ids': BASE_IDS,
        },
        'fold_summaries': all_fold_summaries,
        'aggregated': aggregated,
    }

    output_path = DATA_DIR / f"lobo_results_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("FULL LOBO RESULTS (STRICT)")
    print("=" * 60)
    print(f"{'Alpha':<8} {'Secure%':<12} {'Insecure%':<12} {'Refusal%':<10}")
    print("-" * 42)
    for alpha in ALPHA_GRID:
        r = aggregated[str(alpha)]
        print(f"{alpha:<8} {r['strict_secure_rate']*100:>10.1f}% "
              f"{r['strict_insecure_rate']*100:>10.1f}% "
              f"{r['refusal_rate']*100:>8.1f}%")

    print(f"\nResults saved: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    main()
