#!/home/paperspace/dev/MATS/9_8_research/experiments/sae_env/bin/python
"""
Experiment 3B: Mechanistic Validation via Forced-Choice Logit Gap

Demonstrates that steering shifts the local decision boundary at the API choice token:
Δ = logit(safe_api) - logit(unsafe_api)

Compares three settings:
- S0: α=0 baseline (no steering)
- S1: Best mean-diff operating point from 3A
- S2: Best SAE operating point from 3A
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
    DATA_DIR, FORCED_CHOICE_DIR, ACTIVATION_CACHE,
    MODEL_NAME, GENERATION_CONFIG, GENERATIONS_PER_PROMPT,
    HIDDEN_SIZE,
)
from sae_loader import SAEManager
from sae_calibration import calibrate_single_feature, calibrate_top_k_features
from steering_generator import MultiMethodSteeringGenerator
from forced_choice import (
    create_forced_choice_prompts, load_forced_choice_prompts,
    ForcedChoicePrompt, get_forced_choice_stats,
)

# Import scoring from baseline experiment
sys.path.insert(0, str(Path(__file__).parent.parent / "01-12_llama8b_cwe787_baseline_behavior"))
from scoring import score_completion
from refusal_detection import detect_refusal


def load_best_operating_points(results_3A_path: Path = None) -> Dict:
    """
    Load best operating points from Part 3A results.

    Returns dict with S1 (mean-diff) and S2 (best SAE) configurations.
    """
    if results_3A_path is None:
        # Try to find most recent results
        results_files = list(DATA_DIR.glob("results_3A_*.json"))
        if not results_files:
            print("WARNING: No 3A results found, using defaults")
            return get_default_operating_points()
        results_3A_path = sorted(results_files)[-1]

    print(f"Loading operating points from: {results_3A_path}")

    with open(results_3A_path) as f:
        results_3A = json.load(f)

    aggregated = results_3A['aggregated']

    # Find best mean-diff operating point (highest secure rate with Other% <= 10%)
    best_m1_alpha = None
    best_m1_secure = 0

    for alpha_key, stats in aggregated.get('M1_mean_diff', {}).items():
        other_rate = stats['expanded_other_rate']
        secure_rate = stats['expanded_secure_rate']

        if other_rate <= 0.10 and secure_rate > best_m1_secure:
            best_m1_secure = secure_rate
            best_m1_alpha = float(alpha_key)

    # If no point meets threshold, use highest alpha
    if best_m1_alpha is None:
        alphas = [float(k) for k in aggregated.get('M1_mean_diff', {}).keys()]
        best_m1_alpha = max(alphas) if alphas else 3.0

    # Find best SAE operating point across all SAE methods
    best_sae_method = None
    best_sae_setting = None
    best_sae_secure = 0

    for method in ['M2a_sae_L31_1895', 'M2b_sae_L30_10391', 'M3a_sae_top5', 'M3b_sae_top10']:
        for sigma_key, stats in aggregated.get(method, {}).items():
            other_rate = stats['expanded_other_rate']
            secure_rate = stats['expanded_secure_rate']

            if other_rate <= 0.10 and secure_rate > best_sae_secure:
                best_sae_secure = secure_rate
                best_sae_method = method
                best_sae_setting = sigma_key

    # If no SAE method meets threshold, use M2a +3σ
    if best_sae_method is None:
        best_sae_method = 'M2a_sae_L31_1895'
        best_sae_setting = '+3.0σ'

    return {
        'S1': {
            'method': 'mean_diff',
            'alpha': best_m1_alpha,
            'layer': 31,
            'secure_rate': best_m1_secure,
        },
        'S2': {
            'method': best_sae_method,
            'setting': best_sae_setting,
            'secure_rate': best_sae_secure,
        },
    }


def get_default_operating_points() -> Dict:
    """Default operating points if 3A results not available."""
    return {
        'S1': {
            'method': 'mean_diff',
            'alpha': 3.0,
            'layer': 31,
            'secure_rate': 0.0,
        },
        'S2': {
            'method': 'M2a_sae_L31_1895',
            'setting': '+3.0σ',
            'secure_rate': 0.0,
        },
    }


def compute_mean_diff_direction() -> np.ndarray:
    """Compute mean-diff direction from cached activations."""
    data = np.load(ACTIVATION_CACHE)
    X = data['X_layer_31']
    y = data['y_layer_31']

    secure_mean = X[y == 1].mean(axis=0)
    vulnerable_mean = X[y == 0].mean(axis=0)

    direction = secure_mean - vulnerable_mean
    return direction.astype(np.float32)


def get_sae_direction(
    sae_manager: SAEManager,
    method: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    target_sigma: float,
) -> tuple:
    """
    Get SAE steering direction and calibrated alpha.

    Returns:
        (direction, alpha, layer)
    """
    if method == 'M2a_sae_L31_1895':
        calib = calibrate_single_feature(
            sae_manager, layer=31, feature_idx=1895,
            X_train=X_train, target_sigma=target_sigma
        )
        return calib.direction, calib.alpha, 31

    elif method == 'M2b_sae_L30_10391':
        X_train_30 = np.load(ACTIVATION_CACHE)['X_layer_30']
        # Need to get train indices - use all data for simplicity
        calib = calibrate_single_feature(
            sae_manager, layer=30, feature_idx=10391,
            X_train=X_train_30, target_sigma=target_sigma
        )
        return calib.direction, calib.alpha, 30

    elif method == 'M3a_sae_top5':
        calib = calibrate_top_k_features(
            sae_manager, layer=31,
            X_train=X_train, y_train=y_train,
            k=5, target_sigma=target_sigma
        )
        return calib.direction, calib.alpha, 31

    elif method == 'M3b_sae_top10':
        calib = calibrate_top_k_features(
            sae_manager, layer=31,
            X_train=X_train, y_train=y_train,
            k=10, target_sigma=target_sigma
        )
        return calib.direction, calib.alpha, 31

    else:
        raise ValueError(f"Unknown method: {method}")


def run_experiment_3B(
    generator: MultiMethodSteeringGenerator,
    sae_manager: SAEManager,
    forced_choice_prompts: List[ForcedChoicePrompt],
    operating_points: Dict,
    n_free_gens: int = 3,
) -> Dict:
    """
    Run Part 3B: Forced-choice logit gap measurements.

    For each prompt, measures Δlogit under S0, S1, S2 settings.
    Also generates free completions for comparison.
    """
    print("\n--- Setting up steering directions ---")

    # Load activations for direction computation
    data = np.load(ACTIVATION_CACHE)
    X_31 = data['X_layer_31']
    y = data['y_layer_31']

    # S0: No steering (alpha=0)
    direction_s0 = np.zeros(HIDDEN_SIZE, dtype=np.float32)
    alpha_s0 = 0.0
    layer_s0 = 31

    # S1: Mean-diff
    direction_s1 = compute_mean_diff_direction()
    alpha_s1 = operating_points['S1']['alpha']
    layer_s1 = operating_points['S1']['layer']
    print(f"S1 (mean-diff): α={alpha_s1:.2f}, L{layer_s1}")

    # S2: Best SAE method
    sae_method = operating_points['S2']['method']
    sae_setting = operating_points['S2']['setting']
    target_sigma = float(sae_setting.replace('+', '').replace('σ', ''))

    direction_s2, alpha_s2, layer_s2 = get_sae_direction(
        sae_manager, sae_method, X_31, y, target_sigma
    )
    print(f"S2 ({sae_method} {sae_setting}): α={alpha_s2:.4f}, L{layer_s2}")

    # Settings configuration
    settings = {
        'S0': {'direction': direction_s0, 'alpha': alpha_s0, 'layer': layer_s0},
        'S1': {'direction': direction_s1, 'alpha': alpha_s1, 'layer': layer_s1},
        'S2': {'direction': direction_s2, 'alpha': alpha_s2, 'layer': layer_s2},
    }

    # ==========================================================================
    # Logit gap measurements
    # ==========================================================================
    print("\n--- Measuring logit gaps ---")
    logit_results = []

    for fc in tqdm(forced_choice_prompts, desc="Logit gaps"):
        for setting_name, setting_config in settings.items():
            result = generator.compute_logit_gap_for_vuln_type(
                prompt=fc.forced_choice_prompt,
                vuln_type=fc.vuln_type,
                direction=setting_config['direction'],
                layer=setting_config['layer'],
                alpha=setting_config['alpha'],
            )

            logit_results.append({
                'prompt_id': fc.id,
                'base_id': fc.base_id,
                'vuln_type': fc.vuln_type,
                'setting': setting_name,
                'logit_safe': result['logit_safe'],
                'logit_unsafe': result['logit_unsafe'],
                'gap': result['gap'],
                'prob_safe': result['prob_safe'],
                'prob_unsafe': result['prob_unsafe'],
            })

    # ==========================================================================
    # Free generations for correlation analysis
    # ==========================================================================
    print("\n--- Generating free completions ---")
    freegen_results = []

    for fc in tqdm(forced_choice_prompts, desc="Free gens"):
        for setting_name, setting_config in settings.items():
            for gen_idx in range(n_free_gens):
                output = generator.generate_with_steering(
                    prompt=fc.original_prompt,  # Use original prompt for free gen
                    direction=setting_config['direction'],
                    layer=setting_config['layer'],
                    alpha=setting_config['alpha'],
                    max_tokens=GENERATION_CONFIG['max_new_tokens'],
                    temperature=GENERATION_CONFIG['temperature'],
                    top_p=GENERATION_CONFIG['top_p'],
                )

                score_result = score_completion(output, fc.vuln_type)
                is_refusal, _ = detect_refusal(output)

                freegen_results.append({
                    'prompt_id': fc.id,
                    'base_id': fc.base_id,
                    'vuln_type': fc.vuln_type,
                    'setting': setting_name,
                    'gen_idx': gen_idx,
                    'output': output[:500],
                    'strict_label': score_result.strict_label,
                    'expanded_label': score_result.expanded_label,
                    'is_refusal': is_refusal,
                })

    return {
        'logit_results': logit_results,
        'freegen_results': freegen_results,
        'settings_config': {
            'S0': {'method': 'none', 'alpha': 0.0, 'layer': 31},
            'S1': operating_points['S1'],
            'S2': operating_points['S2'],
        },
    }


def compute_3B_statistics(results: Dict) -> Dict:
    """Compute summary statistics for Part 3B."""
    logit_df = pd.DataFrame(results['logit_results'])
    freegen_df = pd.DataFrame(results['freegen_results'])

    stats = {}

    # Logit gap statistics by setting
    for setting in ['S0', 'S1', 'S2']:
        setting_data = logit_df[logit_df['setting'] == setting]
        gaps = setting_data['gap'].values

        stats[f'{setting}_gap'] = {
            'mean': float(np.mean(gaps)),
            'std': float(np.std(gaps)),
            'median': float(np.median(gaps)),
            'min': float(np.min(gaps)),
            'max': float(np.max(gaps)),
            'n': len(gaps),
        }

    # Gap shifts
    s0_gaps = logit_df[logit_df['setting'] == 'S0'].set_index('prompt_id')['gap']
    s1_gaps = logit_df[logit_df['setting'] == 'S1'].set_index('prompt_id')['gap']
    s2_gaps = logit_df[logit_df['setting'] == 'S2'].set_index('prompt_id')['gap']

    common_ids = s0_gaps.index.intersection(s1_gaps.index).intersection(s2_gaps.index)

    s1_shift = s1_gaps[common_ids] - s0_gaps[common_ids]
    s2_shift = s2_gaps[common_ids] - s0_gaps[common_ids]

    stats['S1_shift'] = {
        'mean': float(np.mean(s1_shift)),
        'std': float(np.std(s1_shift)),
        'positive_frac': float((s1_shift > 0).mean()),
    }

    stats['S2_shift'] = {
        'mean': float(np.mean(s2_shift)),
        'std': float(np.std(s2_shift)),
        'positive_frac': float((s2_shift > 0).mean()),
    }

    # Free generation statistics
    for setting in ['S0', 'S1', 'S2']:
        setting_data = freegen_df[freegen_df['setting'] == setting]
        n = len(setting_data)

        stats[f'{setting}_freegen'] = {
            'n': n,
            'secure_rate': float((setting_data['expanded_label'] == 'secure').mean()),
            'insecure_rate': float((setting_data['expanded_label'] == 'insecure').mean()),
            'other_rate': float((setting_data['expanded_label'] == 'other').mean()),
            'refusal_rate': float(setting_data['is_refusal'].mean()),
        }

    # Correlation between gap and secure label
    # Merge logit and freegen data
    merged = pd.merge(
        logit_df[['prompt_id', 'setting', 'gap']],
        freegen_df[['prompt_id', 'setting', 'expanded_label']].groupby(['prompt_id', 'setting']).agg(
            lambda x: (x == 'secure').mean()
        ).reset_index().rename(columns={'expanded_label': 'secure_rate'}),
        on=['prompt_id', 'setting'],
    )

    if len(merged) > 10:
        correlation = merged['gap'].corr(merged['secure_rate'])
        stats['gap_secure_correlation'] = float(correlation)
    else:
        stats['gap_secure_correlation'] = None

    return stats


def main():
    """Run the full Part 3B experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("EXPERIMENT 3B: Forced-Choice Logit Gap Validation")
    print(f"Timestamp: {timestamp}")
    print("=" * 60)

    # Create output directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FORCED_CHOICE_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize models
    print("\n--- Initializing Models ---")
    generator = MultiMethodSteeringGenerator()
    sae_manager = SAEManager()

    # Pre-load SAEs
    sae_manager.load_sae(30)
    sae_manager.load_sae(31)

    # Create or load forced-choice prompts
    fc_path = FORCED_CHOICE_DIR / "forced_choice_prompts.jsonl"

    if fc_path.exists():
        print(f"\n--- Loading existing forced-choice prompts ---")
        forced_choice_prompts = load_forced_choice_prompts(fc_path)
    else:
        print(f"\n--- Creating forced-choice prompts ---")
        forced_choice_prompts = create_forced_choice_prompts(generator)

    print(f"Using {len(forced_choice_prompts)} forced-choice prompts")

    # Get stats
    fc_stats = get_forced_choice_stats(forced_choice_prompts)
    print(f"By vuln_type: {fc_stats['by_vuln_type']}")
    print(f"Baseline gap: mean={fc_stats['gap_stats']['mean']:.2f}, std={fc_stats['gap_stats']['std']:.2f}")

    # Load operating points from 3A
    print("\n--- Loading operating points from 3A ---")
    operating_points = load_best_operating_points()
    print(f"S1: {operating_points['S1']}")
    print(f"S2: {operating_points['S2']}")

    # Run experiment
    print("\n--- Running Experiment 3B ---")
    results = run_experiment_3B(
        generator=generator,
        sae_manager=sae_manager,
        forced_choice_prompts=forced_choice_prompts,
        operating_points=operating_points,
        n_free_gens=3,
    )

    # Compute statistics
    print("\n--- Computing Statistics ---")
    stats = compute_3B_statistics(results)

    # Save results
    output = {
        'timestamp': timestamp,
        'config': {
            'model': MODEL_NAME,
            'n_forced_choice_prompts': len(forced_choice_prompts),
            'n_free_gens_per_setting': 3,
            'operating_points': operating_points,
        },
        'forced_choice_stats': fc_stats,
        'results_stats': stats,
    }

    # Save full output
    output_path = DATA_DIR / f"results_3B_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    # Save detailed CSVs
    logit_csv_path = DATA_DIR / f"results_3B_logits_{timestamp}.csv"
    pd.DataFrame(results['logit_results']).to_csv(logit_csv_path, index=False)

    freegen_csv_path = DATA_DIR / f"results_3B_freegen_{timestamp}.csv"
    pd.DataFrame(results['freegen_results']).to_csv(freegen_csv_path, index=False)

    print(f"\nSaved results to:")
    print(f"  {output_path}")
    print(f"  {logit_csv_path}")
    print(f"  {freegen_csv_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT 3B SUMMARY")
    print("=" * 60)

    print("\nLogit Gap by Setting:")
    for setting in ['S0', 'S1', 'S2']:
        s = stats[f'{setting}_gap']
        print(f"  {setting}: mean={s['mean']:.2f}, std={s['std']:.2f}, n={s['n']}")

    print("\nGap Shifts (relative to S0):")
    print(f"  S1 shift: mean={stats['S1_shift']['mean']:.2f}, {stats['S1_shift']['positive_frac']*100:.1f}% positive")
    print(f"  S2 shift: mean={stats['S2_shift']['mean']:.2f}, {stats['S2_shift']['positive_frac']*100:.1f}% positive")

    print("\nFree Generation Secure Rates:")
    for setting in ['S0', 'S1', 'S2']:
        s = stats[f'{setting}_freegen']
        print(f"  {setting}: {s['secure_rate']*100:.1f}% secure, {s['insecure_rate']*100:.1f}% insecure")

    if stats['gap_secure_correlation'] is not None:
        print(f"\nCorrelation (gap vs secure): {stats['gap_secure_correlation']:.3f}")

    return str(output_path)


if __name__ == "__main__":
    output_path = main()
    print(f"\nResults saved to: {output_path}")
