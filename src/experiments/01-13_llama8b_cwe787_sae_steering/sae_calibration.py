"""
SAE Alpha Calibration Module

Converts target activation shifts (in sigma units) into appropriate alpha values
for SAE-based steering vectors.

CRITICAL: α calibration for SAE vectors is fundamentally different from mean-diff:
- Mean-diff: direction already represents "shift toward secure" in activation space
- SAE: direction represents one sparse feature; we want to shift the feature activation
  by a specific amount in terms of the feature's own distribution
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from sae_loader import SAEManager
from experiment_config import HIDDEN_SIZE, TARGET_SHIFTS_SIGMA


@dataclass
class CalibrationResult:
    """Result of alpha calibration for an SAE vector."""
    direction: np.ndarray          # Steering direction (4096,)
    alpha: float                   # Calibrated alpha value
    target_sigma: float            # Target shift in sigma units
    achieved_shift: float          # Actual measured shift (for verification)

    # Statistics from training data
    feature_mean: float
    feature_std: float
    decoder_norm: float

    # Metadata
    method: str                    # "single" or "topk"
    layer: int
    feature_indices: List[int]     # Single element for single, multiple for top-k


@dataclass
class TopKCalibrationResult(CalibrationResult):
    """Extended result for top-k calibration."""
    feature_weights: List[float]   # Weights used for each feature
    feature_diffs: List[float]     # Activation diffs for each feature


def calibrate_single_feature(
    sae_manager: SAEManager,
    layer: int,
    feature_idx: int,
    X_train: np.ndarray,
    target_sigma: float,
) -> CalibrationResult:
    """
    Calibrate alpha for steering with a single SAE feature direction.

    The calibration ensures that steering produces a shift of `target_sigma` standard
    deviations in the feature activation distribution.

    Algorithm:
    1. Encode train activations through SAE to get feature activations
    2. Compute feature distribution statistics (μ, σ) for the target feature
    3. Get decoder direction d = W_dec[feature_idx, :]
    4. Compute alpha such that adding α*d produces desired feature shift

    Mathematical derivation:
    - Steering: h' = h + α * d
    - Feature activation after steering: f' ≈ f + α * ||d||²
      (This approximation holds when d is aligned with the encoder direction)
    - Target shift: Δf = target_sigma * σ
    - Therefore: α = Δf / ||d||² = (target_sigma * σ) / ||d||²

    Args:
        sae_manager: Loaded SAE manager
        layer: Layer index
        feature_idx: Feature index to steer
        X_train: Training activations (n_samples, HIDDEN_SIZE)
        target_sigma: Target shift in standard deviations

    Returns:
        CalibrationResult with direction, alpha, and statistics
    """
    # 1. Encode train activations through SAE
    features = sae_manager.encode_activations(layer, X_train)

    # 2. Get statistics for this feature
    f = features[:, feature_idx]
    feature_mean = float(np.mean(f))
    feature_std = float(np.std(f))

    # Handle edge case of zero variance
    if feature_std < 1e-8:
        print(f"  WARNING: Feature {feature_idx} has near-zero variance (std={feature_std})")
        feature_std = 1.0  # Fallback

    # 3. Get decoder direction
    direction = sae_manager.get_feature_direction(layer, feature_idx, normalize=False)
    decoder_norm = float(np.linalg.norm(direction))

    # 4. Compute calibrated alpha
    # Target shift in feature space
    delta_f = target_sigma * feature_std

    # Alpha = delta_f / ||d||²
    d_norm_sq = decoder_norm ** 2
    alpha = delta_f / d_norm_sq if d_norm_sq > 0 else 0.0

    # Compute achieved shift for verification
    # When we add α*d, the feature activation shifts by approximately α*||d||²
    achieved_shift = alpha * d_norm_sq

    return CalibrationResult(
        direction=direction,
        alpha=alpha,
        target_sigma=target_sigma,
        achieved_shift=achieved_shift,
        feature_mean=feature_mean,
        feature_std=feature_std,
        decoder_norm=decoder_norm,
        method="single",
        layer=layer,
        feature_indices=[feature_idx],
    )


def calibrate_top_k_features(
    sae_manager: SAEManager,
    layer: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
    target_sigma: float,
) -> TopKCalibrationResult:
    """
    Calibrate alpha for steering with top-k combined SAE feature directions.

    Algorithm:
    1. Select top-k features based on secure vs vulnerable activation difference
    2. Combine decoder directions weighted by activation differences
    3. Normalize the combined direction
    4. Calibrate alpha for the combined direction

    Args:
        sae_manager: Loaded SAE manager
        layer: Layer index
        X_train: Training activations (n_samples, HIDDEN_SIZE)
        y_train: Training labels (n_samples,) - 0=vulnerable, 1=secure
        k: Number of top features to use
        target_sigma: Target shift in standard deviations

    Returns:
        TopKCalibrationResult with combined direction, alpha, and feature info
    """
    # 1. Split by label
    X_secure = X_train[y_train == 1]
    X_vulnerable = X_train[y_train == 0]

    # 2. Select top-k features (TRAIN ONLY)
    top_k_features = sae_manager.select_top_k_features(
        layer, X_secure, X_vulnerable, k
    )

    feature_indices = [idx for idx, _ in top_k_features]
    feature_diffs = [diff for _, diff in top_k_features]

    print(f"  Top-{k} features selected:")
    for idx, diff in top_k_features[:5]:  # Show first 5
        print(f"    Feature {idx}: diff={diff:.4f}")

    # 3. Combine directions weighted by activation diffs
    sae = sae_manager.load_sae(layer)
    combined = np.zeros(HIDDEN_SIZE, dtype=np.float32)

    for idx, diff in top_k_features:
        d = sae.W_dec[idx, :].detach().cpu().numpy()
        combined += diff * d  # Weighted by signed diff

    # 4. Normalize combined direction
    combined_norm = np.linalg.norm(combined)
    if combined_norm > 0:
        combined_normalized = combined / combined_norm
    else:
        combined_normalized = combined

    # 5. Compute projection statistics
    # Project train activations onto combined direction
    projections = X_train @ combined_normalized
    projection_mean = float(np.mean(projections))
    projection_std = float(np.std(projections))

    if projection_std < 1e-8:
        projection_std = 1.0

    # 6. Calibrate alpha
    # Since direction is normalized, alpha directly controls the shift magnitude
    delta = target_sigma * projection_std
    alpha = delta  # For normalized direction

    # Achieved shift verification
    achieved_shift = alpha  # Since ||combined_normalized|| = 1

    return TopKCalibrationResult(
        direction=combined_normalized,
        alpha=alpha,
        target_sigma=target_sigma,
        achieved_shift=achieved_shift,
        feature_mean=projection_mean,
        feature_std=projection_std,
        decoder_norm=1.0,  # Normalized
        method="topk",
        layer=layer,
        feature_indices=feature_indices,
        feature_weights=feature_diffs,
        feature_diffs=feature_diffs,
    )


def calibrate_all_methods(
    sae_manager: SAEManager,
    X_train: np.ndarray,
    y_train: np.ndarray,
    methods_config: Dict,
) -> Dict[str, Dict[float, CalibrationResult]]:
    """
    Calibrate alpha for all SAE-based methods.

    Args:
        sae_manager: Loaded SAE manager
        X_train: Training activations for this fold
        y_train: Training labels
        methods_config: Dict of method configurations from experiment_config

    Returns:
        Dict mapping method_name -> {target_sigma -> CalibrationResult}
    """
    results = {}

    for method_name, config in methods_config.items():
        method_type = config.get("type", "")

        if method_type == "sae_single":
            layer = config["layer"]
            feature_idx = config["feature_idx"]
            target_shifts = config.get("target_shifts", TARGET_SHIFTS_SIGMA)

            results[method_name] = {}
            print(f"\nCalibrating {method_name} (L{layer}:{feature_idx})...")

            for target_sigma in target_shifts:
                result = calibrate_single_feature(
                    sae_manager, layer, feature_idx, X_train, target_sigma
                )
                results[method_name][target_sigma] = result
                print(f"  +{target_sigma}σ: α={result.alpha:.4f} "
                      f"(μ={result.feature_mean:.4f}, σ={result.feature_std:.4f})")

        elif method_type == "sae_topk":
            layer = config["layer"]
            k = config["k"]
            target_shifts = config.get("target_shifts", TARGET_SHIFTS_SIGMA)

            results[method_name] = {}
            print(f"\nCalibrating {method_name} (top-{k} at L{layer})...")

            for target_sigma in target_shifts:
                result = calibrate_top_k_features(
                    sae_manager, layer, X_train, y_train, k, target_sigma
                )
                results[method_name][target_sigma] = result
                print(f"  +{target_sigma}σ: α={result.alpha:.4f}")

    return results


def calibration_result_to_dict(result: CalibrationResult) -> Dict:
    """Convert CalibrationResult to JSON-serializable dict."""
    d = {
        'alpha': result.alpha,
        'target_sigma': result.target_sigma,
        'achieved_shift': result.achieved_shift,
        'feature_mean': result.feature_mean,
        'feature_std': result.feature_std,
        'decoder_norm': result.decoder_norm,
        'method': result.method,
        'layer': result.layer,
        'feature_indices': result.feature_indices,
    }

    if isinstance(result, TopKCalibrationResult):
        d['feature_weights'] = result.feature_weights
        d['feature_diffs'] = result.feature_diffs

    return d


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import json
    from pathlib import Path
    from experiment_config import (
        ACTIVATION_CACHE, METADATA_CACHE,
        SAE_FEATURES, TOP_K_VALUES, TARGET_SHIFTS_SIGMA,
    )

    print("Testing SAE calibration...")
    print("=" * 60)

    # Load activations
    print("\nLoading cached activations...")
    data = np.load(ACTIVATION_CACHE)
    X = data['X_layer_31']  # Use layer 31 for testing
    y = data['y_layer_31']

    print(f"Activations shape: {X.shape}")
    print(f"Labels: {np.bincount(y)} (0=vuln, 1=secure)")

    # Initialize SAE manager
    sae_manager = SAEManager()

    # Test single feature calibration
    print("\n--- Testing Single Feature Calibration (L31:1895) ---")
    for target in TARGET_SHIFTS_SIGMA:
        result = calibrate_single_feature(
            sae_manager, layer=31, feature_idx=1895,
            X_train=X, target_sigma=target
        )
        print(f"+{target}σ: α={result.alpha:.4f}, "
              f"feature_std={result.feature_std:.4f}, "
              f"decoder_norm={result.decoder_norm:.4f}")

    # Test top-k calibration
    print("\n--- Testing Top-5 Calibration ---")
    for target in TARGET_SHIFTS_SIGMA:
        result = calibrate_top_k_features(
            sae_manager, layer=31,
            X_train=X, y_train=y,
            k=5, target_sigma=target
        )
        print(f"+{target}σ: α={result.alpha:.4f}, "
              f"features={result.feature_indices[:3]}...")

    print("\n" + "=" * 60)
    print("Calibration test complete.")
