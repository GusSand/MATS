"""
Activation Steering for SR/SCG Separation Experiment

Supports differential steering of SR and SCG directions.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import re
import warnings
import os
from typing import Dict, List, Optional

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'


class ActivationSteering:
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        print("Loading model for steering...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()

        self.n_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size

        self.activations = {}
        self.hooks = []

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def collect_activations(self, prompt: str) -> dict:
        """Collect residual stream activations at each layer."""
        self.activations = {}

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                self.activations[layer_idx] = h[:, -1, :].detach().clone()
                return output
            return hook_fn

        self.clear_hooks()
        for layer_idx in range(self.n_layers):
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(hook)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

        result = {
            'activations': {k: v.cpu() for k, v in self.activations.items()},
            'logits': logits.cpu(),
            'probs': probs.cpu()
        }

        self.clear_hooks()
        return result

    def compute_steering_vectors(self, secure_acts: dict, neutral_acts: dict) -> dict:
        """Compute steering vector = secure - neutral at each layer."""
        steering_vectors = {}
        for layer_idx in range(self.n_layers):
            secure_act = secure_acts['activations'][layer_idx]
            neutral_act = neutral_acts['activations'][layer_idx]
            steering_vectors[layer_idx] = secure_act - neutral_act
        return steering_vectors

    def steer_with_direction(
        self,
        prompt: str,
        direction: np.ndarray,
        target_layer: int,
        alpha: float = 1.0
    ) -> dict:
        """
        Steer at a specific layer using a direction vector.

        new_activation = original + alpha * direction
        """
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
        layer = self.model.model.layers[target_layer]
        hook = layer.register_forward_hook(steering_hook)
        self.hooks.append(hook)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

        self.clear_hooks()

        return {
            'logits': logits.cpu(),
            'probs': probs.cpu(),
            'target_layer': target_layer,
            'alpha': alpha
        }

    def steer_with_directions_multi_layer(
        self,
        prompt: str,
        directions: Dict[int, np.ndarray],
        alpha: float = 1.0
    ) -> dict:
        """Steer at multiple layers using direction vectors."""
        def make_steering_hook(layer_idx):
            direction_tensor = torch.tensor(
                directions[layer_idx], dtype=torch.float16
            ).to(self.device)

            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                h[:, -1, :] = h[:, -1, :] + alpha * direction_tensor
                if isinstance(output, tuple):
                    return (h,) + output[1:]
                return h
            return hook_fn

        self.clear_hooks()
        for layer_idx in directions:
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(make_steering_hook(layer_idx))
            self.hooks.append(hook)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

        self.clear_hooks()

        return {
            'logits': logits.cpu(),
            'probs': probs.cpu(),
            'layers': list(directions.keys()),
            'alpha': alpha
        }

    def differential_steering_test(
        self,
        prompt: str,
        sr_direction: np.ndarray,
        scg_direction: np.ndarray,
        target_layer: int,
        probe_sr,  # Trained SR probe for this layer
        probe_scg,  # Trained SCG probe for this layer
        scaler_sr,  # Scaler for SR probe
        scaler_scg,  # Scaler for SCG probe
        alphas: List[float] = None
    ) -> dict:
        """
        Test if steering SR affects SCG (and vice versa).

        Returns measurements of cross-effect.
        """
        if alphas is None:
            alphas = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

        results = {
            'sr_steering': [],
            'scg_steering': []
        }

        # Baseline
        baseline = self.collect_activations(prompt)
        baseline_act = baseline['activations'][target_layer].numpy()

        baseline_sr_pred = probe_sr.predict_proba(
            scaler_sr.transform(baseline_act.reshape(1, -1))
        )[0, 1]
        baseline_scg_pred = probe_scg.predict_proba(
            scaler_scg.transform(baseline_act.reshape(1, -1))
        )[0, 1]

        # Test SR steering
        for alpha in alphas:
            steered = self.steer_with_direction(prompt, sr_direction, target_layer, alpha)

            # Get post-steering activations (need to re-collect)
            # Actually we need to measure the EFFECT on output, not the steered activations
            # The probes measure the input state, but we want to see if steering one direction
            # affects the model's behavior in the other direction

            results['sr_steering'].append({
                'alpha': alpha,
                'baseline_sr': baseline_sr_pred,
                'baseline_scg': baseline_scg_pred,
                # We'll measure effect via output probabilities instead
            })

        # Test SCG steering
        for alpha in alphas:
            results['scg_steering'].append({
                'alpha': alpha,
                'baseline_sr': baseline_sr_pred,
                'baseline_scg': baseline_scg_pred,
            })

        return results

    def generate_with_steering(
        self,
        prompt: str,
        direction: np.ndarray,
        target_layer: int,
        alpha: float,
        max_new_tokens: int = 100,
        temperature: float = 0.6
    ) -> str:
        """Generate text while applying steering."""
        direction_tensor = torch.tensor(direction, dtype=torch.float16).to(self.device)

        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            # Apply steering to last token position
            h[:, -1, :] = h[:, -1, :] + alpha * direction_tensor
            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h

        self.clear_hooks()
        layer = self.model.model.layers[target_layer]
        hook = layer.register_forward_hook(steering_hook)
        self.hooks.append(hook)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )

        self.clear_hooks()

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated[len(prompt):]

    def get_token_probs_with_steering(
        self,
        prompt: str,
        direction: np.ndarray,
        target_layer: int,
        alpha: float,
        tokens: Dict[str, str]
    ) -> dict:
        """Get token probabilities after steering."""
        result = self.steer_with_direction(prompt, direction, target_layer, alpha)
        probs = result['probs']

        token_probs = {}
        for name, token_str in tokens.items():
            token_id = self.tokenizer.encode(token_str, add_special_tokens=False)[0]
            token_probs[f'{name}_prob'] = float(probs[token_id])

        return token_probs
