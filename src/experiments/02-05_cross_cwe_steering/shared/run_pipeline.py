#!/usr/bin/env python3
"""
Unified Pipeline for CWE Steering Experiments

Runs all 5 steps: baseline -> activations -> layer sweep -> pilot LOBO -> full LOBO
Can be called from individual experiment directories.
"""

import sys
import json
import glob as glob_module
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


def load_dataset(dataset_path):
    dataset = []
    with open(dataset_path) as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


# =============================================================================
# STEP 1: BASELINE
# =============================================================================

def run_baseline(config):
    """Generate baseline completions (no steering) for vulnerable prompts."""
    from model_loader import ModelLoader
    from steering_generator import SteeringGenerator
    from scoring import score_completion, detect_refusal

    torch.manual_seed(42)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print(f"STEP 1: Baseline Behavior")
    print(f"Model: {config.MODEL_NAME}")
    print("=" * 60)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(config.DATASET_PATH)
    print(f"Dataset: {len(dataset)} pairs")

    loader = ModelLoader(config.MODEL_NAME, quantization=config.QUANTIZATION)
    generator = SteeringGenerator(loader)

    results = []
    for pair in tqdm(dataset, desc="Baseline generation"):
        prompt = pair['vulnerable']
        vuln_type = pair['vulnerability_type']

        output = generator.generate_baseline(
            prompt=prompt,
            temperature=config.GENERATION_CONFIG['temperature'],
            top_p=config.GENERATION_CONFIG['top_p'],
            max_tokens=config.GENERATION_CONFIG['max_new_tokens'],
        )

        score_result = score_completion(
            output, vuln_type,
            strict_patterns=config.STRICT_PATTERNS,
            expanded_secure_additions=config.EXPANDED_SECURE_ADDITIONS,
            bounds_check_patterns=config.BOUNDS_CHECK_PATTERNS,
        )
        is_refusal, _ = detect_refusal(
            output,
            c_code_indicators=config.C_CODE_INDICATORS,
            refusal_patterns=config.REFUSAL_PATTERNS,
        )

        results.append({
            'id': pair['id'],
            'base_id': pair['base_id'],
            'vulnerability_type': vuln_type,
            'output': output[:500],
            'strict_label': score_result.strict_label,
            'expanded_label': score_result.expanded_label,
            'is_refusal': is_refusal,
        })

    n = len(results)
    strict_secure = sum(1 for r in results if r['strict_label'] == 'secure')
    strict_insecure = sum(1 for r in results if r['strict_label'] == 'insecure')
    strict_other = sum(1 for r in results if r['strict_label'] == 'other')
    expanded_secure = sum(1 for r in results if r['expanded_label'] == 'secure')
    refusals = sum(1 for r in results if r['is_refusal'])

    summary = {
        'n': n,
        'strict': {
            'secure': strict_secure, 'insecure': strict_insecure, 'other': strict_other,
            'secure_rate': strict_secure / n, 'insecure_rate': strict_insecure / n,
        },
        'expanded': {'secure': expanded_secure, 'secure_rate': expanded_secure / n},
        'refusal_rate': refusals / n,
    }

    output_data = {
        'timestamp': timestamp, 'model': config.MODEL_NAME,
        'quantization': config.QUANTIZATION,
        'generation_config': config.GENERATION_CONFIG,
        'n_prompts': n, 'summary': summary, 'results': results,
    }

    output_path = config.DATA_DIR / f"baseline_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSTRICT:   secure={strict_secure}/{n} ({strict_secure/n*100:.1f}%), "
          f"insecure={strict_insecure}/{n} ({strict_insecure/n*100:.1f}%), "
          f"other={strict_other}/{n} ({strict_other/n*100:.1f}%)")
    print(f"EXPANDED: secure={expanded_secure}/{n} ({expanded_secure/n*100:.1f}%)")
    print(f"Refusals: {refusals}/{n} ({refusals/n*100:.1f}%)")
    print(f"Results saved: {output_path}")

    loader.unload()
    return str(output_path)


# =============================================================================
# STEP 2: COLLECT ACTIVATIONS
# =============================================================================

def run_collect_activations(config):
    """Collect activations at all layers."""
    from model_loader import ModelLoader
    from activation_collector import ActivationCollector

    torch.manual_seed(42)

    print("=" * 60)
    print(f"STEP 2: Collect Activations")
    print(f"Model: {config.MODEL_NAME}")
    print("=" * 60)

    dataset = load_dataset(config.DATASET_PATH)
    print(f"Dataset: {len(dataset)} pairs ({len(dataset)*2} prompts)")

    loader = ModelLoader(config.MODEL_NAME, quantization=config.QUANTIZATION)
    collector = ActivationCollector(loader)

    npz_path, metadata_path = collector.collect_dataset(dataset, config.DATA_DIR)
    print(f"Activations saved: {npz_path}")
    print(f"Metadata saved: {metadata_path}")

    loader.unload()
    return str(npz_path), str(metadata_path)


# =============================================================================
# STEP 3: LAYER SWEEP
# =============================================================================

def run_layer_sweep(config):
    """Find optimal steering layer via probe sweep."""
    from layer_sweep import run_layer_sweep as sweep, save_sweep_results

    print("=" * 60)
    print("STEP 3: Layer Sweep")
    print("=" * 60)

    npz_files = sorted(glob_module.glob(str(config.DATA_DIR / "activations_*.npz")))
    if not npz_files:
        raise FileNotFoundError("No activation files found. Run step 2 first.")
    npz_path = Path(npz_files[-1])
    print(f"Using activations: {npz_path}")

    results = sweep(npz_path)
    output_path = save_sweep_results(results, config.DATA_DIR)

    print(f"\nBest layer: {results[0]['layer']} (accuracy={results[0]['probe_accuracy']:.4f})")
    print(f"Results saved: {output_path}")
    return results


# =============================================================================
# STEP 4: PILOT LOBO
# =============================================================================

def get_lobo_splits(metadata, base_ids):
    """Generate LOBO splits."""
    vulnerable_metadata = metadata['vulnerable_metadata']
    secure_metadata = metadata['secure_metadata']
    n_pairs = metadata['n_pairs']

    folds = []
    for held_out_base_id in base_ids:
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


def run_lobo(config, n_folds=None, is_pilot=False):
    """Run LOBO experiment (pilot or full)."""
    from model_loader import ModelLoader
    from steering_generator import SteeringGenerator
    from scoring import score_completion, detect_refusal

    torch.manual_seed(42)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = "Pilot LOBO" if is_pilot else "Full LOBO"

    print("=" * 60)
    print(f"STEP {'4' if is_pilot else '5'}: {label}")
    print(f"Model: {config.MODEL_NAME}")
    print("=" * 60)

    config.FOLD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(config.DATASET_PATH)

    # Load metadata
    meta_files = sorted(glob_module.glob(str(config.DATA_DIR / "metadata_*.json")))
    with open(meta_files[-1]) as f:
        metadata = json.load(f)

    # Get best layer
    sweep_path = config.DATA_DIR / "layer_sweep_results.json"
    with open(sweep_path) as f:
        best_layer = json.load(f)['best_layer']

    # Load activations
    npz_files = sorted(glob_module.glob(str(config.DATA_DIR / "activations_*.npz")))
    data = np.load(npz_files[-1])
    X = data[f'X_layer_{best_layer}'].astype(np.float32)
    y = data[f'y_layer_{best_layer}']

    folds = get_lobo_splits(metadata, config.BASE_IDS)
    if n_folds:
        folds = folds[:n_folds]

    print(f"Dataset: {len(dataset)} pairs")
    print(f"Layer: {best_layer}")
    print(f"Folds: {len(folds)}")
    print(f"Alpha grid: {config.ALPHA_GRID}")

    loader = ModelLoader(config.MODEL_NAME, quantization=config.QUANTIZATION)
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

        total_gens = len(config.ALPHA_GRID) * len(test_prompts) * config.GENERATIONS_PER_PROMPT
        pbar = tqdm(total=total_gens, desc=f"Fold {fold_id}")

        for alpha in config.ALPHA_GRID:
            alpha_key = str(alpha)
            fold_results['alpha_results'][alpha_key] = []

            for item in test_prompts:
                prompt = item['vulnerable']
                vuln_type = item['vulnerability_type']

                for gen_idx in range(config.GENERATIONS_PER_PROMPT):
                    output = generator.generate_with_steering(
                        prompt=prompt,
                        direction=direction,
                        layer=best_layer,
                        alpha=alpha,
                        temperature=config.GENERATION_CONFIG['temperature'],
                        top_p=config.GENERATION_CONFIG['top_p'],
                        max_tokens=config.GENERATION_CONFIG['max_new_tokens'],
                    )

                    score_result = score_completion(
                        output, vuln_type,
                        strict_patterns=config.STRICT_PATTERNS,
                        expanded_secure_additions=config.EXPANDED_SECURE_ADDITIONS,
                        bounds_check_patterns=config.BOUNDS_CHECK_PATTERNS,
                    )
                    is_refusal, _ = detect_refusal(
                        output,
                        c_code_indicators=config.C_CODE_INDICATORS,
                        refusal_patterns=config.REFUSAL_PATTERNS,
                    )

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

        # Compute fold summary
        summary = {}
        for alpha_key, items in fold_results['alpha_results'].items():
            n = len(items)
            strict_secure = sum(1 for r in items if r['strict_label'] == 'secure')
            strict_insecure = sum(1 for r in items if r['strict_label'] == 'insecure')
            expanded_secure = sum(1 for r in items if r['expanded_label'] == 'secure')
            refusals = sum(1 for r in items if r['is_refusal'])
            summary[alpha_key] = {
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
        fold_results['summary'] = summary
        all_fold_results.append(fold_results)

        prefix = "pilot_fold" if is_pilot else "fold"
        fold_path = config.FOLD_RESULTS_DIR / f"{prefix}_{fold_id}_{timestamp}.json"
        with open(fold_path, 'w') as f:
            json.dump(fold_results, f, indent=2)
        print(f"Saved: {fold_path}")

    # Aggregate results
    aggregated = {}
    for alpha in config.ALPHA_GRID:
        alpha_key = str(alpha)
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

    # Print summary
    print(f"\n{'='*60}")
    print(f"{label.upper()} RESULTS (STRICT)")
    print("=" * 60)
    print(f"{'Alpha':<8} {'Secure%':<12} {'Insecure%':<12} {'Refusal%':<10}")
    print("-" * 42)

    baseline_rate = 0.0
    best_rate = 0.0
    best_alpha = 0.0

    for alpha in config.ALPHA_GRID:
        r = aggregated[str(alpha)]
        rate = r['strict_secure_rate']
        if alpha == 0.0:
            baseline_rate = rate
        if rate > best_rate:
            best_rate = rate
            best_alpha = alpha
        print(f"{alpha:<8} {rate*100:>10.1f}% "
              f"{r['strict_insecure_rate']*100:>10.1f}% "
              f"{r['refusal_rate']*100:>8.1f}%")

    improvement = best_rate - baseline_rate
    print(f"\nBaseline: {baseline_rate*100:.1f}%")
    print(f"Best: {best_rate*100:.1f}% at alpha={best_alpha}")
    print(f"Improvement: {improvement*100:.1f}pp")

    # Save aggregated results
    output = {
        'timestamp': timestamp,
        'config': {
            'model': config.MODEL_NAME,
            'layer': best_layer,
            'alpha_grid': config.ALPHA_GRID,
            'generations_per_prompt': config.GENERATIONS_PER_PROMPT,
            'generation_config': config.GENERATION_CONFIG,
            'n_folds': len(folds),
            'base_ids': config.BASE_IDS,
            'is_pilot': is_pilot,
        },
        'aggregated': aggregated,
        'baseline_rate': baseline_rate,
        'best_rate': best_rate,
        'best_alpha': best_alpha,
        'improvement_pp': improvement * 100,
    }

    prefix = "pilot" if is_pilot else "lobo"
    output_path = config.DATA_DIR / f"{prefix}_results_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved: {output_path}")

    if is_pilot:
        if improvement > 0.10:
            print(f"\n*** GATE PASSED: {improvement*100:.1f}pp > 10pp -- proceed to full LOBO ***")
        else:
            print(f"\n*** GATE FAILED: {improvement*100:.1f}pp <= 10pp ***")

    loader.unload()
    return output
