#!/usr/bin/env python3
"""
Steering Mechanism Verification - Metric Computation

Computes:
1. Probe projections at each layer (how far along the secure-vulnerable axis)
2. SAE feature activations (security-promoting and suppressing features)
3. Steering vector alignment (how much of the change is in the steering direction)
"""

import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import sys

# Add parent experiments to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_config import (
    DATA_DIR, RESULTS_DIR, ACTIVATION_CACHE,
    LAYERS_TO_EXTRACT, STEERING_LAYER,
    SECURITY_PROMOTING_FEATURES, SECURITY_SUPPRESSING_FEATURES,
    SAE_RELEASE, HIDDEN_SIZE, N_SAMPLES,
)

# Import SAEManager from prior experiment
try:
    from importlib.util import spec_from_file_location, module_from_spec
    sae_loader_path = Path(__file__).parent.parent / "01-13_llama8b_cwe787_sae_steering" / "sae_loader.py"
    spec = spec_from_file_location("sae_loader", sae_loader_path)
    sae_loader = module_from_spec(spec)
    spec.loader.exec_module(sae_loader)
    SAEManager = sae_loader.SAEManager
    HAS_SAE = True
    print("SAEManager imported successfully")
except Exception as e:
    print(f"Warning: Could not import SAEManager: {e}")
    print("SAE analysis will be skipped.")
    HAS_SAE = False
    SAEManager = None


# =============================================================================
# PROBE DIRECTION COMPUTATION
# =============================================================================

def load_probe_directions():
    """
    Load cached activations and compute probe directions at each layer.
    Probe direction = mean(secure) - mean(vulnerable)
    """
    print("Loading cached activations for probe directions...")
    data = np.load(ACTIVATION_CACHE)

    probe_directions = {}

    for layer in LAYERS_TO_EXTRACT:
        X = data[f'X_layer_{layer}']
        y = data[f'y_layer_{layer}']

        # Compute mean-difference direction
        secure_mean = X[y == 1].mean(axis=0)
        vulnerable_mean = X[y == 0].mean(axis=0)
        direction = secure_mean - vulnerable_mean

        # Normalize
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm

        probe_directions[layer] = direction.astype(np.float32)
        print(f"  Layer {layer}: direction norm = {np.linalg.norm(direction):.4f}")

    return probe_directions


def compute_probe_projection(activation, probe_direction):
    """
    Project activation onto probe direction.

    Args:
        activation: (hidden_dim,) activation vector
        probe_direction: (hidden_dim,) normalized probe direction

    Returns:
        Scalar projection (higher = more "secure")
    """
    # Normalize activation for cosine similarity interpretation
    act_norm = np.linalg.norm(activation)
    if act_norm > 0:
        activation = activation / act_norm

    # Dot product gives projection (cosine similarity since both normalized)
    return float(np.dot(activation, probe_direction))


# =============================================================================
# STEERING ALIGNMENT
# =============================================================================

def compute_steering_alignment(act_baseline, act_steered, steering_vector):
    """
    Compute how much the activation change aligns with the steering direction.

    Args:
        act_baseline: (hidden_dim,) activation without steering
        act_steered: (hidden_dim,) activation with steering
        steering_vector: (hidden_dim,) steering direction (unnormalized)

    Returns:
        dict with alignment metrics
    """
    # Actual change from steering
    delta = act_steered - act_baseline

    # Normalize steering vector for projection
    steering_norm = np.linalg.norm(steering_vector)
    steering_unit = steering_vector / (steering_norm + 1e-8)

    # Projection onto steering direction
    alignment = np.dot(delta, steering_unit)

    # Decompose into parallel and orthogonal components
    parallel = alignment * steering_unit
    orthogonal = delta - parallel

    parallel_mag = np.linalg.norm(parallel)
    orthogonal_mag = np.linalg.norm(orthogonal)
    delta_norm = np.linalg.norm(delta)

    return {
        "alignment": float(alignment),  # Signed magnitude in steering direction
        "parallel_magnitude": float(parallel_mag),
        "orthogonal_magnitude": float(orthogonal_mag),
        "alignment_ratio": float(parallel_mag / (orthogonal_mag + 1e-8)),  # >1 means mostly aligned
        "delta_norm": float(delta_norm),
        "steering_vector_norm": float(steering_norm),
    }


# =============================================================================
# SAE FEATURE EXTRACTION
# =============================================================================

class SAEFeatureExtractor:
    """Extract activations for specific SAE features."""

    def __init__(self, device="cuda"):
        self.device = device
        self.sae_manager = None
        if HAS_SAE:
            print("Initializing SAEManager...")
            try:
                self.sae_manager = SAEManager(device=device)
            except Exception as e:
                print(f"Warning: Could not initialize SAEManager: {e}")
                self.sae_manager = None

    def get_feature_activations(self, activations_dict, layer, feature_ids):
        """
        Get activations for specific SAE features.

        Args:
            activations_dict: dict mapping layer -> activation array (hidden_dim,)
            layer: Layer index
            feature_ids: List of feature indices

        Returns:
            dict mapping feature_id -> activation value
        """
        if self.sae_manager is None:
            return {fid: None for fid in feature_ids}

        if layer not in activations_dict:
            return {fid: None for fid in feature_ids}

        try:
            # Get activation for this layer
            activation = activations_dict[layer]
            if isinstance(activation, list):
                activation = np.array(activation)

            # Encode through SAE (expects shape (n_samples, hidden_dim))
            activation = activation.reshape(1, -1)
            features = self.sae_manager.encode_activations(layer, activation)

            # Extract specific features
            return {fid: float(features[0, fid]) for fid in feature_ids}
        except Exception as e:
            print(f"Warning: SAE encoding failed for layer {layer}: {e}")
            return {fid: None for fid in feature_ids}


# =============================================================================
# MAIN
# =============================================================================

def find_latest_file(directory, pattern):
    """Find the most recent file matching pattern."""
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}")
    return files[-1]


def main():
    print("=" * 60)
    print("Steering Mechanism Verification - Metric Computation")
    print("=" * 60)

    # Find most recent activation files
    try:
        latest_npz = find_latest_file(DATA_DIR, "activations_*.npz")
        latest_json = latest_npz.with_suffix('.json')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run 01_collect_activations.py first.")
        return

    print(f"\nLoading activations from: {latest_npz}")
    act_data = dict(np.load(latest_npz))

    print(f"Loading metadata from: {latest_json}")
    with open(latest_json, 'r') as f:
        metadata = json.load(f)

    n_samples = metadata["config"]["n_samples"]
    print(f"Processing {n_samples} samples per condition")

    # Load probe directions
    probe_directions = load_probe_directions()

    # Load steering vector
    steering_vector_path = DATA_DIR / "steering_direction.npy"
    if steering_vector_path.exists():
        steering_vector = np.load(steering_vector_path)
        print(f"\nSteering vector loaded, norm = {np.linalg.norm(steering_vector):.4f}")
    else:
        # Compute from cached activations
        data = np.load(ACTIVATION_CACHE)
        X = data['X_layer_31']
        y = data['y_layer_31']
        steering_vector = X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)
        print(f"\nSteering vector computed, norm = {np.linalg.norm(steering_vector):.4f}")

    # Initialize SAE extractor
    sae_extractor = SAEFeatureExtractor() if HAS_SAE else None

    # Results structure
    results = {
        "probe_projections": {cond: {str(layer): [] for layer in LAYERS_TO_EXTRACT}
                             for cond in ["A", "B", "C"]},
        "sae_features": {cond: {"promoting": [], "suppressing": []}
                        for cond in ["A", "B", "C"]},
        "steering_alignment": [],
        "classifications": {cond: [] for cond in ["A", "B", "C"]},
        "metadata": {
            "n_samples": n_samples,
            "layers_extracted": LAYERS_TO_EXTRACT,
            "steering_layer": STEERING_LAYER,
            "source_activation_file": str(latest_npz),
        }
    }

    # Process each condition
    condition_map = {"A": "condition_A", "B": "condition_B", "C": "condition_C"}

    for cond_letter, cond_key in condition_map.items():
        print(f"\nProcessing Condition {cond_letter}...")

        # Store classifications
        results["classifications"][cond_letter] = [
            r["classification"] for r in metadata[cond_key]
        ]

        # Probe projections at each layer
        for layer in LAYERS_TO_EXTRACT:
            layer_key = f"{cond_key}_L{layer}"
            if layer_key not in act_data:
                print(f"  Warning: Missing {layer_key}")
                continue

            acts = act_data[layer_key]  # Shape: (n_samples, hidden_dim)
            probe_dir = probe_directions[layer]

            projections = []
            for i in range(n_samples):
                proj = compute_probe_projection(acts[i], probe_dir)
                projections.append(proj)

            results["probe_projections"][cond_letter][str(layer)] = projections

        # SAE features
        if sae_extractor and sae_extractor.sae_manager:
            print(f"  Extracting SAE features...")

            for i in tqdm(range(n_samples), desc=f"  SAE {cond_letter}"):
                # Build activation dict for this sample
                sample_acts = {}
                for layer in set(list(SECURITY_PROMOTING_FEATURES.keys()) +
                                list(SECURITY_SUPPRESSING_FEATURES.keys())):
                    layer_key = f"{cond_key}_L{layer}"
                    if layer_key in act_data:
                        sample_acts[layer] = act_data[layer_key][i]

                # Get promoting features
                promoting_acts = {}
                for layer, features in SECURITY_PROMOTING_FEATURES.items():
                    feat_acts = sae_extractor.get_feature_activations(sample_acts, layer, features)
                    promoting_acts[str(layer)] = feat_acts

                # Get suppressing features
                suppressing_acts = {}
                for layer, features in SECURITY_SUPPRESSING_FEATURES.items():
                    feat_acts = sae_extractor.get_feature_activations(sample_acts, layer, features)
                    suppressing_acts[str(layer)] = feat_acts

                results["sae_features"][cond_letter]["promoting"].append(promoting_acts)
                results["sae_features"][cond_letter]["suppressing"].append(suppressing_acts)

    # Steering alignment (comparing A and B at steering layer)
    print("\nComputing steering alignment (A vs B)...")
    layer_key_A = f"condition_A_L{STEERING_LAYER}"
    layer_key_B = f"condition_B_L{STEERING_LAYER}"

    if layer_key_A in act_data and layer_key_B in act_data:
        acts_A = act_data[layer_key_A]
        acts_B = act_data[layer_key_B]

        for i in range(n_samples):
            alignment = compute_steering_alignment(acts_A[i], acts_B[i], steering_vector)
            results["steering_alignment"].append(alignment)

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    # Probe projections at steering layer
    print(f"\nProbe Projections at Layer {STEERING_LAYER}:")
    for cond in ["A", "B", "C"]:
        projs = results["probe_projections"][cond].get(str(STEERING_LAYER), [])
        if projs:
            print(f"  Condition {cond}: mean={np.mean(projs):.4f}, std={np.std(projs):.4f}")

    # Steering alignment
    if results["steering_alignment"]:
        print(f"\nSteering Alignment (A -> B):")
        alignments = [a["alignment"] for a in results["steering_alignment"]]
        ratios = [a["alignment_ratio"] for a in results["steering_alignment"]]
        print(f"  Alignment: mean={np.mean(alignments):.4f}, std={np.std(alignments):.4f}")
        print(f"  Ratio (parallel/orthogonal): mean={np.mean(ratios):.2f}")

    # Classification rates
    print(f"\nClassification Rates:")
    for cond in ["A", "B", "C"]:
        classes = results["classifications"][cond]
        if classes:
            secure = sum(1 for c in classes if c == "secure") / len(classes)
            insecure = sum(1 for c in classes if c == "insecure") / len(classes)
            other = sum(1 for c in classes if c == "other") / len(classes)
            print(f"  Condition {cond}: secure={secure:.1%}, insecure={insecure:.1%}, other={other:.1%}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"metrics_{timestamp}.json"

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nMetrics saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
