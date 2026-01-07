"""
Security Patcher - Activation patching framework for sprintf/snprintf localization.

Adapted from the 9_8_research BidirectionalPatcher for security code analysis.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple
import warnings
import os

from .classification import classify_security, get_classification_symbol

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'


class SecurityPatcher:
    """
    Activation patching framework for analyzing sprintf vs snprintf decisions.

    Supports:
    - Layer-level patching (attention outputs)
    - Head-level selective patching
    - Bidirectional causality testing
    """

    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
                 device: str = "cuda"):
        print("Loading model...")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.model.eval()

        # Model config
        self.n_layers = self.model.config.num_hidden_layers
        self.n_heads = self.model.config.num_attention_heads
        self.hidden_size = self.model.config.hidden_size
        self.head_dim = self.hidden_size // self.n_heads

        print(f"Model loaded: {self.n_layers} layers, {self.n_heads} heads, {self.hidden_size} hidden dim")

        # Storage for activations and hooks
        self.saved_activations = {}
        self.hooks = []

    def get_attention_module(self, layer_idx: int) -> nn.Module:
        """Get the attention module for a specific layer."""
        layer = self.model.model.layers[layer_idx]
        if hasattr(layer, 'self_attn'):
            return layer.self_attn
        else:
            raise AttributeError(f"Layer {layer_idx} does not have self_attn module")

    def get_mlp_module(self, layer_idx: int) -> nn.Module:
        """Get the MLP module for a specific layer."""
        layer = self.model.model.layers[layer_idx]
        if hasattr(layer, 'mlp'):
            return layer.mlp
        else:
            raise AttributeError(f"Layer {layer_idx} does not have mlp module")

    def save_activation_hook(self, key: str):
        """Create a hook that saves the activation."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self.saved_activations[key] = hidden_states.detach().clone()
            return output
        return hook_fn

    def patch_activation_hook(self, saved_activation: torch.Tensor):
        """Create a hook that patches in a saved activation (full layer)."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            batch_size, seq_len, hidden_dim = hidden_states.shape
            saved_seq_len = saved_activation.shape[1]

            # Patch overlapping sequence positions
            min_seq_len = min(seq_len, saved_seq_len)
            new_hidden = hidden_states.clone()
            new_hidden[:, :min_seq_len, :] = saved_activation[:, :min_seq_len, :]

            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden
        return hook_fn

    def selective_head_patch_hook(self, saved_activation: torch.Tensor,
                                   head_indices: List[int]):
        """Create a hook that patches only specific attention heads."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            batch_size, seq_len, hidden_size = hidden_states.shape

            # Reshape to separate heads
            hidden_reshaped = hidden_states.view(batch_size, seq_len, self.n_heads, self.head_dim)
            saved_reshaped = saved_activation.view(batch_size, -1, self.n_heads, self.head_dim)

            new_hidden = hidden_reshaped.clone()
            min_seq_len = min(seq_len, saved_reshaped.shape[1])

            # Patch only specified heads
            for head_idx in head_indices:
                new_hidden[:, :min_seq_len, head_idx, :] = saved_reshaped[:, :min_seq_len, head_idx, :]

            new_hidden = new_hidden.view(batch_size, seq_len, hidden_size)

            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden
        return hook_fn

    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.saved_activations = {}

    @contextmanager
    def save_attention_context(self, prompt: str, layer_idx: int):
        """Context manager to save attention output at a specific layer."""
        try:
            module = self.get_attention_module(layer_idx)
            key = f"layer_{layer_idx}"
            hook = module.register_forward_hook(self.save_activation_hook(key))
            self.hooks.append(hook)

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                _ = self.model(**inputs)

            yield self.saved_activations.copy()

        finally:
            self.clear_hooks()

    @contextmanager
    def patch_attention_context(self, saved_activation: torch.Tensor, layer_idx: int):
        """Context manager to patch attention during generation."""
        try:
            module = self.get_attention_module(layer_idx)
            hook = module.register_forward_hook(
                self.patch_activation_hook(saved_activation)
            )
            self.hooks.append(hook)
            yield
        finally:
            self.clear_hooks()

    @contextmanager
    def patch_heads_context(self, saved_activation: torch.Tensor, layer_idx: int,
                            head_indices: List[int]):
        """Context manager to patch specific heads during generation."""
        try:
            module = self.get_attention_module(layer_idx)
            hook = module.register_forward_hook(
                self.selective_head_patch_hook(saved_activation, head_indices)
            )
            self.hooks.append(hook)
            yield
        finally:
            self.clear_hooks()

    def generate(self, prompt: str, max_new_tokens: int = 150,
                 temperature: float = 0.0, do_sample: bool = False) -> str:
        """Generate text from prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            if do_sample and temperature > 0:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated[len(prompt):]  # Return only generated part

    def generate_and_classify(self, prompt: str, max_new_tokens: int = 150,
                              temperature: float = 0.0, do_sample: bool = False) -> Dict:
        """Generate and classify the output."""
        output = self.generate(prompt, max_new_tokens, temperature, do_sample)
        classification = classify_security(output)
        classification['generated_text'] = output
        return classification

    def test_layer_patch(self, secure_prompt: str, insecure_prompt: str,
                         layer_idx: int, n_trials: int = 5) -> Dict:
        """
        Test if patching a specific layer can flip security behavior.

        Forward: insecure_prompt + secure_activation → should become secure
        Reverse: secure_prompt + insecure_activation → should become insecure
        """
        results = {
            'layer': layer_idx,
            'forward_patch': [],
            'reverse_patch': []
        }

        # Save activations from both prompts
        with self.save_attention_context(secure_prompt, layer_idx) as saved:
            secure_activation = saved[f"layer_{layer_idx}"].clone()

        with self.save_attention_context(insecure_prompt, layer_idx) as saved:
            insecure_activation = saved[f"layer_{layer_idx}"].clone()

        # Forward patch: Apply secure attention to insecure prompt
        for _ in range(n_trials):
            with self.patch_attention_context(secure_activation, layer_idx):
                output = self.generate(insecure_prompt)
                classification = classify_security(output)
                results['forward_patch'].append(classification)

        # Reverse patch: Apply insecure attention to secure prompt
        for _ in range(n_trials):
            with self.patch_attention_context(insecure_activation, layer_idx):
                output = self.generate(secure_prompt)
                classification = classify_security(output)
                results['reverse_patch'].append(classification)

        # Calculate success rates
        forward_secure = sum(1 for r in results['forward_patch'] if r['is_secure'])
        reverse_insecure = sum(1 for r in results['reverse_patch'] if r['is_insecure'])

        results['forward_secure_rate'] = forward_secure / n_trials
        results['reverse_insecure_rate'] = reverse_insecure / n_trials
        results['bidirectional_success'] = (
            results['forward_secure_rate'] >= 0.8 and
            results['reverse_insecure_rate'] >= 0.8
        )

        return results
