"""
Probe-based CWE router.
Single forward pass on prompt tokens only (no generation).
Extracts Layer 31 activations, runs binary probe, returns CWE classification.

Reuses: BinaryProbe pattern from Exp 8.5 (02-08_probe_routing_v2/03_e2e_pipeline.py)
Probe weights: trained on adversarial L31 activations (95.2% neutral LOO accuracy)
"""
import numpy as np
import torch
from typing import Tuple, Literal


class BinaryProbe:
    """Lightweight binary probe — loads weights from numpy files (Exp 8.5 format)."""

    def __init__(self, data_dir):
        from pathlib import Path
        data_dir = Path(data_dir)
        self.weights = np.load(data_dir / "binary_probe_weights.npy")       # [1, 4096]
        self.bias = np.load(data_dir / "binary_probe_bias.npy")             # [1]
        self.scaler_mean = np.load(data_dir / "binary_probe_scaler_mean.npy")
        self.scaler_scale = np.load(data_dir / "binary_probe_scaler_scale.npy")

    def predict(self, activation) -> Tuple[str, float]:
        """
        Classify a single L31 activation vector.
        Returns: ("format_string", prob) or ("buffer", prob)
        """
        x = (activation - self.scaler_mean) / self.scaler_scale
        logit = x @ self.weights.T + self.bias
        prob = 1.0 / (1.0 + np.exp(-logit.item()))
        if prob > 0.5:
            return "format_string", prob
        else:
            return "buffer", 1.0 - prob


class ProbeRouter:
    """
    Probe-based router that classifies prompts via a single forward pass.

    Key design: the extraction hook is registered and removed within the probe
    pass only. By the time model.generate() is called, there are zero hooks.
    """

    def __init__(self, model, tokenizer, probe_path: str, probe_layer: int = 31):
        """
        Args:
            model: Loaded Llama-3.1-8B-Instruct
            tokenizer: Corresponding tokenizer
            probe_path: Path to directory with binary probe numpy files (from Exp 8.5)
            probe_layer: Layer to extract activations from (31 = trained probe layer)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.probe_layer = probe_layer
        self.probe = BinaryProbe(probe_path)
        self._activation_cache = {}

    def _hook_fn(self, module, input, output):
        """Temporary hook — only used during the single probe pass, NOT during generation."""
        if isinstance(output, tuple):
            self._activation_cache['activations'] = output[0][:, -1, :].detach().cpu().numpy().astype(np.float32).squeeze(0)
        else:
            self._activation_cache['activations'] = output[:, -1, :].detach().cpu().numpy().astype(np.float32).squeeze(0)

    @torch.no_grad()
    def classify(self, prompt: str) -> Tuple[Literal["buffer", "format_string"], float]:
        """
        Run a single forward pass (no generation) to classify prompt CWE type.
        Cost: ~54ms on A100.

        Returns: (cwe_type, confidence)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Register hook ONLY for this single forward pass
        layer = self.model.model.layers[self.probe_layer]
        handle = layer.register_forward_hook(self._hook_fn)

        try:
            self.model(**inputs)
            activations = self._activation_cache['activations']
            cwe_type, confidence = self.probe.predict(activations)
        finally:
            handle.remove()  # Critical: remove hook before generation
            self._activation_cache.clear()

        return cwe_type, confidence
