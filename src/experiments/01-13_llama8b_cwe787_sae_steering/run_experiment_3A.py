#!/home/paperspace/dev/MATS/9_8_research/experiments/sae_env/bin/python
"""
Experiment 3A: Precision Steering Head-to-Head (LOBO)

Compares Mean-diff vs SAE-based steering methods:
- M1: Mean-diff at L31 (baseline from Experiment 2)
- M2a: Single SAE feature L31:1895
- M2b: Single SAE feature L30:10391
- M3a: Top-5 SAE features at L31
- M3b: Top-10 SAE features at L31

Uses LOBO (Leave-One-Base-ID-Out) cross-validation.
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Optional
import pandas as pd

from experiment_config import (
    DATA_DIR, FOLD_RESULTS_DIR,
    ACTIVATION_CACHE, METADATA_CACHE, DATASET_PATH,
    BASE_IDS, METHODS, MEAN_DIFF_ALPHA_GRID, TARGET_SHIFTS_SIGMA,
    MODEL_NAME, GENERATION_CONFIG, GENERATIONS_PER_PROMPT,
)
from sae_loader import SAEManager
from sae_calibration import (
    calibrate_single_feature, calibrate_top_k_features,
    calibration_result_to_dict,
)
from steering_generator import MultiMethodSteeringGenerator

# Import scoring from baseline experiment
sys.path.insert(0, str(Path(__file__).parent.parent / "01-12_llama8b_cwe787_baseline_behavior"))
from scoring import score_completion
from refusal_detection import detect_refusal

# Import LOBO splits from LOBO experiment
sys.path.insert(0, str(Path(__file__).parent.parent / "01-12_llama8b_cwe787_lobo_steering"))
from lobo_splits import load_metadata, load_dataset, get_lobo_splits


def load_activations_all_layers() -> Dict[int, np.ndarray]:
    """Load activations for all layers needed."""
    data = np.load(ACTIVATION_CACHE)
    activations = {}

    # Load layers 30 and 31 (needed for SAE features)
    for layer in [30, 31]:
        activations[layer] = data[f'X_layer_{layer}']

    # Also load labels from layer 31
    labels = data['y_layer_31']

    return activations, labels


def compute_mean_diff_direction(X: np.ndarray, y: np.ndarray, fold: Dict) -> np.ndarray:
    """
    Compute mean-difference direction from train activations only.

    Replicates logic from LOBO experiment's compute_fold_direction.
    """
    train_indices = fold['train_vuln_indices'] + fold['train_sec_indices']
    X_train = X[train_indices]
    y_train = y[train_indices]

    secure_mean = X_train[y_train == 1].mean(axis=0)
    vulnerable_mean = X_train[y_train == 0].mean(axis=0)

    direction = secure_mean - vulnerable_mean
    return direction.astype(np.float32)


def get_test_prompts(dataset: List[Dict], test_indices: List[int]) -> List[Dict]:
    """Get test prompts for generation (vulnerable prompts only)."""
    return [dataset[i] for i in test_indices]


def run_fold_all_methods(
    generator: MultiMethodSteeringGenerator,
    sae_manager: SAEManager,
    fold: Dict,
    activations: Dict[int, np.ndarray],
    labels: np.ndarray,
    dataset: List[Dict],
    n_gens: int,
    gen_config: Dict,
) -> Dict:
    """
    Run all steering methods for a single LOBO fold.

    Returns:
        Dict with results for each method at each alpha/sigma
    """
    fold_id = fold['fold_id']
    print(f"\n{'='*60}")
    print(f"FOLD: {fold_id}")
    print(f"{'='*60}")

    # Get train/test data
    train_indices = fold['train_vuln_indices'] + fold['train_sec_indices']
    test_prompts = get_test_prompts(dataset, fold['test_indices'])

    print(f"Train samples: {len(train_indices)}")
    print(f"Test prompts: {len(test_prompts)}")

    # Prepare train data for calibration
    X_train_31 = activations[31][train_indices]
    X_train_30 = activations[30][train_indices]
    y_train = labels[train_indices]

    # Initialize results
    results = {
        'fold_id': fold_id,
        'n_train': len(train_indices),
        'n_test': len(test_prompts),
        'method_results': {},
        'calibration_info': {},
    }

    # ==========================================================================
    # M1: Mean-diff steering
    # ==========================================================================
    print("\n--- M1: Mean-diff ---")
    direction_m1 = compute_mean_diff_direction(activations[31], labels, fold)
    direction_norm = float(np.linalg.norm(direction_m1))
    print(f"Direction norm: {direction_norm:.4f}")

    results['method_results']['M1_mean_diff'] = {}
    results['calibration_info']['M1_mean_diff'] = {'direction_norm': direction_norm}

    for alpha in MEAN_DIFF_ALPHA_GRID:
        alpha_key = str(alpha)
        results['method_results']['M1_mean_diff'][alpha_key] = []

        for item in tqdm(test_prompts, desc=f"M1 α={alpha}", leave=False):
            for gen_idx in range(n_gens):
                output = generator.generate_with_steering(
                    prompt=item['vulnerable'],
                    direction=direction_m1,
                    layer=31,
                    alpha=alpha,
                    **gen_config,
                )

                score_result = score_completion(output, item['vulnerability_type'])
                is_refusal, _ = detect_refusal(output)

                results['method_results']['M1_mean_diff'][alpha_key].append({
                    'id': item['id'],
                    'base_id': item['base_id'],
                    'vulnerability_type': item['vulnerability_type'],
                    'gen_idx': gen_idx,
                    'output': output[:500],
                    'strict_label': score_result.strict_label,
                    'expanded_label': score_result.expanded_label,
                    'is_refusal': is_refusal,
                })

    # ==========================================================================
    # M2a: Single SAE L31:1895
    # ==========================================================================
    print("\n--- M2a: SAE L31:1895 ---")
    results['method_results']['M2a_sae_L31_1895'] = {}
    results['calibration_info']['M2a_sae_L31_1895'] = {}

    for target_sigma in TARGET_SHIFTS_SIGMA:
        calib = calibrate_single_feature(
            sae_manager, layer=31, feature_idx=1895,
            X_train=X_train_31, target_sigma=target_sigma
        )
        sigma_key = f"+{target_sigma}σ"
        print(f"  {sigma_key}: α={calib.alpha:.4f}")

        results['calibration_info']['M2a_sae_L31_1895'][sigma_key] = calibration_result_to_dict(calib)
        results['method_results']['M2a_sae_L31_1895'][sigma_key] = []

        for item in tqdm(test_prompts, desc=f"M2a {sigma_key}", leave=False):
            for gen_idx in range(n_gens):
                output = generator.generate_with_steering(
                    prompt=item['vulnerable'],
                    direction=calib.direction,
                    layer=31,
                    alpha=calib.alpha,
                    **gen_config,
                )

                score_result = score_completion(output, item['vulnerability_type'])
                is_refusal, _ = detect_refusal(output)

                results['method_results']['M2a_sae_L31_1895'][sigma_key].append({
                    'id': item['id'],
                    'base_id': item['base_id'],
                    'vulnerability_type': item['vulnerability_type'],
                    'gen_idx': gen_idx,
                    'output': output[:500],
                    'strict_label': score_result.strict_label,
                    'expanded_label': score_result.expanded_label,
                    'is_refusal': is_refusal,
                })

    # ==========================================================================
    # M2b: Single SAE L30:10391
    # ==========================================================================
    print("\n--- M2b: SAE L30:10391 ---")
    results['method_results']['M2b_sae_L30_10391'] = {}
    results['calibration_info']['M2b_sae_L30_10391'] = {}

    for target_sigma in TARGET_SHIFTS_SIGMA:
        calib = calibrate_single_feature(
            sae_manager, layer=30, feature_idx=10391,
            X_train=X_train_30, target_sigma=target_sigma
        )
        sigma_key = f"+{target_sigma}σ"
        print(f"  {sigma_key}: α={calib.alpha:.4f}")

        results['calibration_info']['M2b_sae_L30_10391'][sigma_key] = calibration_result_to_dict(calib)
        results['method_results']['M2b_sae_L30_10391'][sigma_key] = []

        for item in tqdm(test_prompts, desc=f"M2b {sigma_key}", leave=False):
            for gen_idx in range(n_gens):
                output = generator.generate_with_steering(
                    prompt=item['vulnerable'],
                    direction=calib.direction,
                    layer=30,  # NOTE: Steering at L30!
                    alpha=calib.alpha,
                    **gen_config,
                )

                score_result = score_completion(output, item['vulnerability_type'])
                is_refusal, _ = detect_refusal(output)

                results['method_results']['M2b_sae_L30_10391'][sigma_key].append({
                    'id': item['id'],
                    'base_id': item['base_id'],
                    'vulnerability_type': item['vulnerability_type'],
                    'gen_idx': gen_idx,
                    'output': output[:500],
                    'strict_label': score_result.strict_label,
                    'expanded_label': score_result.expanded_label,
                    'is_refusal': is_refusal,
                })

    # ==========================================================================
    # M3a: Top-5 SAE features
    # ==========================================================================
    print("\n--- M3a: Top-5 SAE features ---")
    results['method_results']['M3a_sae_top5'] = {}
    results['calibration_info']['M3a_sae_top5'] = {}

    for target_sigma in TARGET_SHIFTS_SIGMA:
        calib = calibrate_top_k_features(
            sae_manager, layer=31,
            X_train=X_train_31, y_train=y_train,
            k=5, target_sigma=target_sigma
        )
        sigma_key = f"+{target_sigma}σ"
        print(f"  {sigma_key}: α={calib.alpha:.4f}")

        results['calibration_info']['M3a_sae_top5'][sigma_key] = calibration_result_to_dict(calib)
        results['method_results']['M3a_sae_top5'][sigma_key] = []

        for item in tqdm(test_prompts, desc=f"M3a {sigma_key}", leave=False):
            for gen_idx in range(n_gens):
                output = generator.generate_with_steering(
                    prompt=item['vulnerable'],
                    direction=calib.direction,
                    layer=31,
                    alpha=calib.alpha,
                    **gen_config,
                )

                score_result = score_completion(output, item['vulnerability_type'])
                is_refusal, _ = detect_refusal(output)

                results['method_results']['M3a_sae_top5'][sigma_key].append({
                    'id': item['id'],
                    'base_id': item['base_id'],
                    'vulnerability_type': item['vulnerability_type'],
                    'gen_idx': gen_idx,
                    'output': output[:500],
                    'strict_label': score_result.strict_label,
                    'expanded_label': score_result.expanded_label,
                    'is_refusal': is_refusal,
                })

    # ==========================================================================
    # M3b: Top-10 SAE features
    # ==========================================================================
    print("\n--- M3b: Top-10 SAE features ---")
    results['method_results']['M3b_sae_top10'] = {}
    results['calibration_info']['M3b_sae_top10'] = {}

    for target_sigma in TARGET_SHIFTS_SIGMA:
        calib = calibrate_top_k_features(
            sae_manager, layer=31,
            X_train=X_train_31, y_train=y_train,
            k=10, target_sigma=target_sigma
        )
        sigma_key = f"+{target_sigma}σ"
        print(f"  {sigma_key}: α={calib.alpha:.4f}")

        results['calibration_info']['M3b_sae_top10'][sigma_key] = calibration_result_to_dict(calib)
        results['method_results']['M3b_sae_top10'][sigma_key] = []

        for item in tqdm(test_prompts, desc=f"M3b {sigma_key}", leave=False):
            for gen_idx in range(n_gens):
                output = generator.generate_with_steering(
                    prompt=item['vulnerable'],
                    direction=calib.direction,
                    layer=31,
                    alpha=calib.alpha,
                    **gen_config,
                )

                score_result = score_completion(output, item['vulnerability_type'])
                is_refusal, _ = detect_refusal(output)

                results['method_results']['M3b_sae_top10'][sigma_key].append({
                    'id': item['id'],
                    'base_id': item['base_id'],
                    'vulnerability_type': item['vulnerability_type'],
                    'gen_idx': gen_idx,
                    'output': output[:500],
                    'strict_label': score_result.strict_label,
                    'expanded_label': score_result.expanded_label,
                    'is_refusal': is_refusal,
                })

    return results


def summarize_method_results(method_results: Dict) -> Dict:
    """Compute summary statistics for a method."""
    summaries = {}

    for alpha_key, items in method_results.items():
        n = len(items)
        if n == 0:
            continue

        strict_secure = sum(1 for r in items if r['strict_label'] == 'secure')
        strict_insecure = sum(1 for r in items if r['strict_label'] == 'insecure')
        strict_other = sum(1 for r in items if r['strict_label'] == 'other')

        expanded_secure = sum(1 for r in items if r['expanded_label'] == 'secure')
        expanded_insecure = sum(1 for r in items if r['expanded_label'] == 'insecure')
        expanded_other = sum(1 for r in items if r['expanded_label'] == 'other')

        refusals = sum(1 for r in items if r['is_refusal'])

        summaries[alpha_key] = {
            'n': n,
            'strict': {
                'secure': strict_secure,
                'insecure': strict_insecure,
                'other': strict_other,
                'secure_rate': strict_secure / n,
                'insecure_rate': strict_insecure / n,
                'other_rate': strict_other / n,
            },
            'expanded': {
                'secure': expanded_secure,
                'insecure': expanded_insecure,
                'other': expanded_other,
                'secure_rate': expanded_secure / n,
                'insecure_rate': expanded_insecure / n,
                'other_rate': expanded_other / n,
            },
            'refusal_rate': refusals / n,
        }

    return summaries


def main():
    """Run the full Part 3A experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("EXPERIMENT 3A: Precision Steering Head-to-Head (LOBO)")
    print(f"Timestamp: {timestamp}")
    print("=" * 60)

    # Create output directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FOLD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n--- Loading Data ---")
    metadata = load_metadata()
    activations, labels = load_activations_all_layers()
    dataset = load_dataset()

    print(f"Activations L31: {activations[31].shape}")
    print(f"Activations L30: {activations[30].shape}")
    print(f"Labels: {labels.shape}")
    print(f"Dataset: {len(dataset)} pairs")

    # Generate LOBO splits
    print("\n--- Generating LOBO Splits ---")
    folds = get_lobo_splits(metadata)
    print(f"Generated {len(folds)} folds")

    # Initialize models
    print("\n--- Initializing Models ---")
    generator = MultiMethodSteeringGenerator()
    sae_manager = SAEManager()

    # Pre-load SAEs
    print("Loading SAEs for layers 30 and 31...")
    sae_manager.load_sae(30)
    sae_manager.load_sae(31)

    # Run experiment
    print("\n--- Running Experiment ---")
    print(f"Methods: M1 (mean-diff), M2a (L31:1895), M2b (L30:10391), M3a (top-5), M3b (top-10)")
    print(f"Generations per prompt: {GENERATIONS_PER_PROMPT}")

    all_fold_results = []

    for fold in folds:
        fold_results = run_fold_all_methods(
            generator=generator,
            sae_manager=sae_manager,
            fold=fold,
            activations=activations,
            labels=labels,
            dataset=dataset,
            n_gens=GENERATIONS_PER_PROMPT,
            gen_config={
                'temperature': GENERATION_CONFIG['temperature'],
                'top_p': GENERATION_CONFIG['top_p'],
                'max_tokens': GENERATION_CONFIG['max_new_tokens'],
            },
        )

        # Compute summaries for each method
        fold_results['summaries'] = {}
        for method_name, method_results in fold_results['method_results'].items():
            fold_results['summaries'][method_name] = summarize_method_results(method_results)

        all_fold_results.append(fold_results)

        # Save fold results
        fold_path = FOLD_RESULTS_DIR / f"fold_{fold['fold_id']}_{timestamp}.json"
        with open(fold_path, 'w') as f:
            json.dump(fold_results, f, indent=2)
        print(f"\nSaved fold results: {fold_path}")

    # Aggregate results across folds
    print("\n--- Aggregating Results ---")
    aggregated = aggregate_all_folds(all_fold_results)

    # Save full results
    output = {
        'timestamp': timestamp,
        'config': {
            'model': MODEL_NAME,
            'methods': list(METHODS.keys()),
            'mean_diff_alpha_grid': MEAN_DIFF_ALPHA_GRID,
            'sae_target_shifts': TARGET_SHIFTS_SIGMA,
            'generations_per_prompt': GENERATIONS_PER_PROMPT,
            'generation_config': GENERATION_CONFIG,
            'n_folds': len(folds),
            'base_ids': BASE_IDS,
        },
        'aggregated': aggregated,
    }

    output_path = DATA_DIR / f"results_3A_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved aggregated results: {output_path}")

    # Print summary
    print_summary(aggregated)

    return str(output_path)


def aggregate_all_folds(all_fold_results: List[Dict]) -> Dict:
    """Aggregate results across all folds."""
    aggregated = {}

    # Get all methods
    methods = list(all_fold_results[0]['method_results'].keys())

    for method in methods:
        aggregated[method] = {}

        # Get all alpha/sigma keys for this method
        all_keys = set()
        for fold_results in all_fold_results:
            all_keys.update(fold_results['method_results'][method].keys())

        for key in sorted(all_keys):
            # Collect all items across folds
            all_items = []
            for fold_results in all_fold_results:
                if key in fold_results['method_results'][method]:
                    all_items.extend(fold_results['method_results'][method][key])

            n = len(all_items)
            if n == 0:
                continue

            strict_secure = sum(1 for r in all_items if r['strict_label'] == 'secure')
            strict_insecure = sum(1 for r in all_items if r['strict_label'] == 'insecure')
            strict_other = sum(1 for r in all_items if r['strict_label'] == 'other')

            expanded_secure = sum(1 for r in all_items if r['expanded_label'] == 'secure')
            expanded_insecure = sum(1 for r in all_items if r['expanded_label'] == 'insecure')
            expanded_other = sum(1 for r in all_items if r['expanded_label'] == 'other')

            refusals = sum(1 for r in all_items if r['is_refusal'])

            aggregated[method][key] = {
                'n': n,
                'strict_secure_rate': strict_secure / n,
                'strict_insecure_rate': strict_insecure / n,
                'strict_other_rate': strict_other / n,
                'expanded_secure_rate': expanded_secure / n,
                'expanded_insecure_rate': expanded_insecure / n,
                'expanded_other_rate': expanded_other / n,
                'refusal_rate': refusals / n,
            }

    return aggregated


def print_summary(aggregated: Dict):
    """Print summary table."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 3A SUMMARY")
    print("=" * 80)

    for method in sorted(aggregated.keys()):
        print(f"\n{method}:")
        print(f"{'Setting':<12} {'Secure%':<12} {'Insecure%':<12} {'Other%':<12} {'Refusal%':<10}")
        print("-" * 58)

        for key, stats in sorted(aggregated[method].items()):
            print(f"{key:<12} "
                  f"{stats['strict_secure_rate']*100:>10.1f}% "
                  f"{stats['strict_insecure_rate']*100:>10.1f}% "
                  f"{stats['strict_other_rate']*100:>10.1f}% "
                  f"{stats['refusal_rate']*100:>8.1f}%")


if __name__ == "__main__":
    output_path = main()
    print(f"\nResults saved to: {output_path}")
