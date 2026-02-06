"""
Model-Agnostic Steering Generator

Applies activation steering during generation via forward hooks.
Adapted from 01-12_llama8b_cwe787_lobo_steering/run_experiment.py
"""

import torch
import numpy as np


class SteeringGenerator:
    """Generator with activation steering support."""

    def __init__(self, model_loader):
        """
        Args:
            model_loader: ModelLoader instance
        """
        self.model = model_loader.model
        self.tokenizer = model_loader.tokenizer
        self.device = model_loader.device
        self.n_layers = model_loader.n_layers
        self.hooks = []

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def generate_with_steering(
        self,
        prompt: str,
        direction: np.ndarray,
        layer: int,
        alpha: float,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_tokens: int = 512,
    ) -> str:
        """
        Generate with steering applied at specified layer.

        Args:
            prompt: Input prompt
            direction: Steering direction vector (hidden_size,)
            layer: Layer index to apply steering
            alpha: Steering strength
            temperature: Sampling temperature
            top_p: Top-p nucleus sampling
            max_tokens: Maximum new tokens

        Returns:
            Generated text (prompt stripped)
        """
        # Activations are float16 even in 8-bit quantized models
        direction_tensor = torch.tensor(direction, dtype=torch.float16).to(self.device)

        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            h[:, -1, :] = h[:, -1, :] + alpha * direction_tensor
            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h

        self.clear_hooks()
        target_layer = self.model.model.layers[layer]
        hook = target_layer.register_forward_hook(steering_hook)
        self.hooks.append(hook)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id
            )

        self.clear_hooks()

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated[len(prompt):]

    def generate_baseline(
        self,
        prompt: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_tokens: int = 512,
    ) -> str:
        """Generate without steering (baseline)."""
        self.clear_hooks()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated[len(prompt):]
