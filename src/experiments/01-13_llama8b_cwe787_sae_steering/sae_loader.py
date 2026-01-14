"""
SAE Loader Module

Handles loading Llama-Scope SAEs and extracting decoder directions for steering.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sae_lens import SAE

from experiment_config import SAE_RELEASE, HIDDEN_SIZE


class SAEManager:
    """Manages SAE loading and feature extraction for multiple layers."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.saes: Dict[int, SAE] = {}
        self.loaded_layers: List[int] = []

    def load_sae(self, layer: int) -> SAE:
        """
        Load pretrained Llama-Scope SAE for a given layer.

        Args:
            layer: Layer index (0-31 for Llama-8B)

        Returns:
            Loaded SAE model
        """
        if layer in self.saes:
            return self.saes[layer]

        sae_id = f"l{layer}r_8x"  # residual stream, 8x expansion

        print(f"Loading SAE for layer {layer} (sae_id={sae_id})...")
        sae = SAE.from_pretrained(
            release=SAE_RELEASE,
            sae_id=sae_id,
            device=self.device
        )

        self.saes[layer] = sae
        self.loaded_layers.append(layer)

        # Log SAE configuration
        d_in = getattr(sae.cfg, 'd_in', HIDDEN_SIZE)
        d_sae = getattr(sae.cfg, 'd_sae', 'unknown')
        print(f"  Loaded: d_in={d_in}, d_sae={d_sae}")

        return sae

    def get_feature_direction(
        self,
        layer: int,
        feature_idx: int,
        normalize: bool = False,
    ) -> np.ndarray:
        """
        Extract decoder direction for a single SAE feature.

        Args:
            layer: Layer index
            feature_idx: Feature index within SAE
            normalize: If True, L2-normalize the direction

        Returns:
            Direction vector (HIDDEN_SIZE,) = (4096,)
        """
        sae = self.load_sae(layer)

        # Get decoder direction (W_dec has shape [d_sae, d_in])
        direction = sae.W_dec[feature_idx, :].detach().cpu().numpy()

        if normalize:
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm

        return direction.astype(np.float32)

    def get_decoder_norm(self, layer: int, feature_idx: int) -> float:
        """Get the L2 norm of a decoder direction."""
        sae = self.load_sae(layer)
        direction = sae.W_dec[feature_idx, :]
        return torch.norm(direction).item()

    def encode_activations(
        self,
        layer: int,
        activations: np.ndarray,
    ) -> np.ndarray:
        """
        Encode activations through SAE to get sparse feature activations.

        Args:
            layer: Layer index
            activations: Shape (n_samples, HIDDEN_SIZE)

        Returns:
            Feature activations: Shape (n_samples, d_sae)
        """
        sae = self.load_sae(layer)

        # Convert to tensor and move to device
        x = torch.tensor(activations, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            features = sae.encode(x)

        return features.cpu().numpy()

    def get_feature_statistics(
        self,
        layer: int,
        activations: np.ndarray,
        feature_idx: int,
    ) -> Dict[str, float]:
        """
        Compute statistics for a specific feature across activations.

        Args:
            layer: Layer index
            activations: Shape (n_samples, HIDDEN_SIZE)
            feature_idx: Feature index

        Returns:
            Dict with mean, std, min, max, sparsity
        """
        features = self.encode_activations(layer, activations)
        f = features[:, feature_idx]

        return {
            'mean': float(np.mean(f)),
            'std': float(np.std(f)),
            'min': float(np.min(f)),
            'max': float(np.max(f)),
            'sparsity': float(np.mean(f == 0)),  # Fraction of zeros
            'n_samples': len(f),
        }

    def select_top_k_features(
        self,
        layer: int,
        X_secure: np.ndarray,
        X_vulnerable: np.ndarray,
        k: int,
    ) -> List[Tuple[int, float]]:
        """
        Select top-k security-associated features based on activation difference.

        Features are ranked by |mean(secure) - mean(vulnerable)|.
        Must be called on TRAIN split only to avoid data leakage.

        Args:
            layer: Layer index
            X_secure: Secure prompt activations (n_secure, HIDDEN_SIZE)
            X_vulnerable: Vulnerable prompt activations (n_vuln, HIDDEN_SIZE)
            k: Number of top features to select

        Returns:
            List of (feature_idx, activation_diff) tuples, sorted by |diff|
        """
        # Encode both sets through SAE
        features_secure = self.encode_activations(layer, X_secure)
        features_vuln = self.encode_activations(layer, X_vulnerable)

        # Compute mean activation per feature
        mean_secure = np.mean(features_secure, axis=0)
        mean_vuln = np.mean(features_vuln, axis=0)

        # Compute difference (positive = more active in secure)
        diffs = mean_secure - mean_vuln

        # Rank by absolute difference
        ranked_indices = np.argsort(np.abs(diffs))[::-1]  # Descending

        # Return top-k
        top_k = []
        for i in range(k):
            idx = ranked_indices[i]
            top_k.append((int(idx), float(diffs[idx])))

        return top_k

    def get_combined_direction(
        self,
        layer: int,
        feature_indices: List[int],
        weights: Optional[List[float]] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Combine multiple SAE feature directions into a single steering direction.

        Args:
            layer: Layer index
            feature_indices: List of feature indices to combine
            weights: Optional weights for each feature (default: uniform)
            normalize: If True, L2-normalize the combined direction

        Returns:
            Combined direction vector (HIDDEN_SIZE,)
        """
        sae = self.load_sae(layer)

        if weights is None:
            weights = [1.0] * len(feature_indices)

        combined = np.zeros(HIDDEN_SIZE, dtype=np.float32)

        for idx, w in zip(feature_indices, weights):
            d = sae.W_dec[idx, :].detach().cpu().numpy()
            combined += w * d

        if normalize:
            norm = np.linalg.norm(combined)
            if norm > 0:
                combined = combined / norm

        return combined


def load_sae_for_layer(layer: int, device: str = "cuda") -> SAE:
    """
    Convenience function to load a single SAE.

    Args:
        layer: Layer index
        device: Device to load SAE on

    Returns:
        Loaded SAE model
    """
    sae_id = f"l{layer}r_8x"
    return SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=sae_id,
        device=device
    )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing SAE loader...")
    print("=" * 60)

    # Test loading SAEs
    manager = SAEManager(device="cuda" if torch.cuda.is_available() else "cpu")

    # Test L31:1895
    print("\n--- Testing L31:1895 ---")
    direction = manager.get_feature_direction(31, 1895)
    norm = manager.get_decoder_norm(31, 1895)
    print(f"Direction shape: {direction.shape}")
    print(f"Direction norm: {norm:.4f}")
    print(f"Direction dtype: {direction.dtype}")

    # Test L30:10391
    print("\n--- Testing L30:10391 ---")
    direction = manager.get_feature_direction(30, 10391)
    norm = manager.get_decoder_norm(30, 10391)
    print(f"Direction shape: {direction.shape}")
    print(f"Direction norm: {norm:.4f}")

    # Test encoding (with dummy data)
    print("\n--- Testing encoding ---")
    dummy_activations = np.random.randn(10, 4096).astype(np.float32)
    features = manager.encode_activations(31, dummy_activations)
    print(f"Features shape: {features.shape}")
    print(f"Active features per sample: {(features > 0).sum(axis=1).mean():.1f}")

    print("\n" + "=" * 60)
    print("SAE loader test complete.")
