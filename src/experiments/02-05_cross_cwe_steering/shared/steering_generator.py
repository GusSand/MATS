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
        input_len = inputs.input_ids.shape[1]

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

        new_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def generate_with_multi_steering(
        self,
        prompt: str,
        directions: list,
        layer: int,
        alphas: list,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_tokens: int = 512,
    ) -> str:
        """
        Generate with multiple steering vectors applied simultaneously.

        Args:
            prompt: Input prompt
            directions: List of steering direction vectors, each (hidden_size,)
            layer: Layer index to apply steering
            alphas: List of steering strengths (same length as directions)
            temperature: Sampling temperature
            top_p: Top-p nucleus sampling
            max_tokens: Maximum new tokens

        Returns:
            Generated text (prompt stripped)
        """
        direction_tensors = [
            torch.tensor(d, dtype=torch.float16).to(self.device)
            for d in directions
        ]
        alpha_values = list(alphas)

        def multi_steering_hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            for alpha, d_tensor in zip(alpha_values, direction_tensors):
                h[:, -1, :] = h[:, -1, :] + alpha * d_tensor
            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h

        self.clear_hooks()
        target_layer = self.model.model.layers[layer]
        hook = target_layer.register_forward_hook(multi_steering_hook)
        self.hooks.append(hook)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]

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

        new_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

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
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id
            )

        new_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
