"""
Model-Agnostic Activation Collector

Collects last-token activations at all layers for a dataset of prompt pairs.
Adapted from 01-08_llama8b_sr_scg_separation/utils/activation_collector.py
"""

import torch
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm


class ActivationCollector:
    """Collect last-token activations at all layers."""

    def __init__(self, model_loader):
        """
        Args:
            model_loader: ModelLoader instance with .model, .tokenizer, .n_layers, .hidden_size
        """
        self.model = model_loader.model
        self.tokenizer = model_loader.tokenizer
        self.device = model_loader.device
        self.n_layers = model_loader.n_layers
        self.hidden_size = model_loader.hidden_size

        self.activations = {}
        self.hooks = []

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}

    def _make_layer_hook(self, layer_idx: int):
        """Create a hook that saves last-token activation."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            # Move to CPU immediately to save GPU memory
            self.activations[layer_idx] = h[:, -1, :].detach().cpu()
        return hook_fn

    def register_all_layer_hooks(self):
        """Register hooks at all layers."""
        self.clear_hooks()
        for layer_idx in range(self.n_layers):
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(self._make_layer_hook(layer_idx))
            self.hooks.append(hook)

    def get_activations(self, prompt: str) -> dict:
        """Get last-token activations at all layers for a prompt."""
        self.register_all_layer_hooks()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            _ = self.model(**inputs)

        result = {k: v.numpy() for k, v in self.activations.items()}
        self.clear_hooks()
        return result

    def collect_dataset(self, dataset: list, output_dir: Path, layers: list = None):
        """
        Collect activations for all prompts in dataset.

        Layout:
        - Indices 0 to N-1: vulnerable prompts
        - Indices N to 2N-1: secure prompts

        Args:
            dataset: List of pair dicts with 'vulnerable' and 'secure' keys
            output_dir: Directory to save NPZ + metadata
            layers: Optional list of layer indices (default: all layers)

        Returns:
            (npz_path, metadata_path)
        """
        if layers is None:
            layers = list(range(self.n_layers))

        n_pairs = len(dataset)
        n_total = 2 * n_pairs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"Collecting activations for {n_pairs} pairs ({n_total} prompts) at {len(layers)} layers")

        # Initialize storage
        all_activations = {layer: np.zeros((n_total, self.hidden_size), dtype=np.float16)
                          for layer in layers}
        vulnerable_metadata = []
        secure_metadata = []

        self.register_all_layer_hooks()

        # Collect vulnerable prompts (indices 0..N-1)
        print("Collecting vulnerable prompt activations...")
        for i, pair in enumerate(tqdm(dataset, desc="Vulnerable")):
            prompt = pair['vulnerable']
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                _ = self.model(**inputs)

            for layer in layers:
                all_activations[layer][i] = self.activations[layer].numpy().astype(np.float16)

            vulnerable_metadata.append({
                'index': i,
                'id': pair['id'],
                'base_id': pair['base_id'],
                'vulnerability_type': pair['vulnerability_type'],
                'prompt_type': 'vulnerable',
            })

        # Collect secure prompts (indices N..2N-1)
        print("Collecting secure prompt activations...")
        for i, pair in enumerate(tqdm(dataset, desc="Secure")):
            idx = n_pairs + i
            prompt = pair['secure']
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                _ = self.model(**inputs)

            for layer in layers:
                all_activations[layer][idx] = self.activations[layer].numpy().astype(np.float16)

            secure_metadata.append({
                'index': idx,
                'id': pair['id'],
                'base_id': pair['base_id'],
                'vulnerability_type': pair['vulnerability_type'],
                'prompt_type': 'secure',
            })

        self.clear_hooks()

        # Save NPZ
        npz_data = {}
        for layer in layers:
            npz_data[f'X_layer_{layer}'] = all_activations[layer]
            y = np.array([0] * n_pairs + [1] * n_pairs)  # 0=vulnerable, 1=secure
            npz_data[f'y_layer_{layer}'] = y

        output_dir.mkdir(parents=True, exist_ok=True)
        npz_path = output_dir / f"activations_{timestamp}.npz"
        np.savez_compressed(npz_path, **npz_data)
        print(f"Saved activations: {npz_path} ({npz_path.stat().st_size / 1e6:.1f} MB)")

        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'n_pairs': n_pairs,
            'n_total': n_total,
            'n_layers': len(layers),
            'layers': layers,
            'hidden_size': self.hidden_size,
            'vulnerable_metadata': vulnerable_metadata,
            'secure_metadata': secure_metadata,
        }
        metadata_path = output_dir / f"metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata: {metadata_path}")

        # Validate
        self._validate(npz_path, metadata, layers, n_total)

        return npz_path, metadata_path

    def _validate(self, npz_path, metadata, layers, n_total):
        """Validate saved data integrity."""
        data = np.load(npz_path)
        for layer in layers:
            X = data[f'X_layer_{layer}']
            y = data[f'y_layer_{layer}']
            assert X.shape == (n_total, self.hidden_size), \
                f"Layer {layer}: expected ({n_total}, {self.hidden_size}), got {X.shape}"
            assert y.shape == (n_total,), f"Layer {layer}: expected ({n_total},), got {y.shape}"
            assert not np.any(np.isnan(X)), f"Layer {layer}: NaN detected"
            assert not np.any(np.isinf(X)), f"Layer {layer}: Inf detected"
            assert sum(y == 0) == metadata['n_pairs'], f"Layer {layer}: wrong vulnerable count"
            assert sum(y == 1) == metadata['n_pairs'], f"Layer {layer}: wrong secure count"

        print(f"Validation passed: {len(layers)} layers, {n_total} prompts, no NaN/Inf")
