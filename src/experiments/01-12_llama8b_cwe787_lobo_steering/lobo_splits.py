"""
LOBO (Leave-One-Base-ID-Out) Cross-Validation Splits

Generates train/test splits for the LOBO experiment:
- For each fold: hold out 1 base_id (all its variants) as test
- Use remaining 6 base_ids as train to compute direction
"""

import json
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

from experiment_config import BASE_IDS, METADATA_CACHE, ACTIVATION_CACHE, DATASET_PATH


def load_metadata() -> Dict:
    """Load metadata with base_id mappings."""
    with open(METADATA_CACHE) as f:
        return json.load(f)


def load_activations() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load cached activations for layer 31.

    Returns:
        X: activations (210, 4096)
        y: labels (210,) - 0=vulnerable, 1=secure
    """
    data = np.load(ACTIVATION_CACHE)
    X = data['X_layer_31']
    y = data['y_layer_31']
    return X, y


def load_dataset() -> List[Dict]:
    """Load the expanded dataset."""
    dataset = []
    with open(DATASET_PATH) as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


def get_lobo_splits(metadata: Dict) -> List[Dict]:
    """
    Generate LOBO splits.

    The activations are ordered as:
    - Indices 0-104: Vulnerable prompts (order matches dataset)
    - Indices 105-209: Secure prompts (order matches dataset)

    Returns:
        List of 7 fold dicts, each with:
        - fold_id: held-out base_id
        - train_vuln_indices: indices of train vulnerable prompts
        - train_sec_indices: indices of train secure prompts
        - test_indices: indices of test vulnerable prompts (for generation)
    """
    vulnerable_metadata = metadata['vulnerable_metadata']
    secure_metadata = metadata['secure_metadata']

    folds = []

    for held_out_base_id in BASE_IDS:
        # Train indices: all base_ids except held_out
        train_vuln_indices = [
            i for i, m in enumerate(vulnerable_metadata)
            if m['base_id'] != held_out_base_id
        ]
        train_sec_indices = [
            i + 105 for i, m in enumerate(secure_metadata)
            if m['base_id'] != held_out_base_id
        ]

        # Test indices: only held_out base_id (vulnerable prompts only for generation)
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


def compute_fold_direction(X: np.ndarray, y: np.ndarray, fold: Dict) -> np.ndarray:
    """
    Compute mean-difference direction from train activations only.

    Direction = mean(secure_activations) - mean(vulnerable_activations)

    Args:
        X: all activations (210, 4096)
        y: all labels (210,)
        fold: fold dict with train indices

    Returns:
        direction: (4096,) numpy array
    """
    # Combine train indices
    train_indices = fold['train_vuln_indices'] + fold['train_sec_indices']

    # Extract train data
    X_train = X[train_indices]
    y_train = y[train_indices]

    # Compute mean-difference direction
    secure_mean = X_train[y_train == 1].mean(axis=0)
    vulnerable_mean = X_train[y_train == 0].mean(axis=0)

    direction = secure_mean - vulnerable_mean

    return direction.astype(np.float32)


def get_test_prompts(dataset: List[Dict], test_indices: List[int]) -> List[Dict]:
    """
    Get test prompts for generation.

    Args:
        dataset: full dataset (105 pairs)
        test_indices: indices of test pairs

    Returns:
        List of test pair dicts (with vulnerable prompt, detection, etc.)
    """
    return [dataset[i] for i in test_indices]


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing LOBO splits...")
    print("=" * 60)

    # Load data
    metadata = load_metadata()
    X, y = load_activations()
    dataset = load_dataset()

    print(f"Activations shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Dataset size: {len(dataset)}")

    # Generate splits
    folds = get_lobo_splits(metadata)

    print(f"\nGenerated {len(folds)} LOBO folds:")
    print("-" * 60)

    for fold in folds:
        print(f"Fold: {fold['fold_id']}")
        print(f"  Train: {fold['n_train']} (vuln: {len(fold['train_vuln_indices'])}, sec: {len(fold['train_sec_indices'])})")
        print(f"  Test: {fold['n_test']} vulnerable prompts")

        # Compute direction for this fold
        direction = compute_fold_direction(X, y, fold)
        print(f"  Direction norm: {np.linalg.norm(direction):.4f}")

        # Get test prompts
        test_prompts = get_test_prompts(dataset, fold['test_indices'])
        print(f"  Test prompt example: {test_prompts[0]['id']}")
        print()

    # Verify no overlap
    print("Verifying no data leakage:")
    for fold in folds:
        train_set = set(fold['train_vuln_indices'])
        test_set = set(fold['test_indices'])
        overlap = train_set.intersection(test_set)
        status = "PASS" if len(overlap) == 0 else "FAIL"
        print(f"  {fold['fold_id']}: {status} (overlap: {len(overlap)})")

    print("\n" + "=" * 60)
    print("LOBO splits test complete.")
