#!/usr/bin/env python3
"""
Layer Sweep v2 for sprintf vs snprintf Localization

Uses forced continuation approach:
- SECURE context: Prompt that starts with "snprintf("
- INSECURE context: Prompt that starts with "sprintf("

This ensures both prompts produce their target function deterministically,
allowing clean activation patching.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from contextlib import contextmanager
from typing import Dict, List, Tuple
import json
from datetime import datetime
from pathlib import Path
import warnings
import os
import re

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Base code that's identical for both
BASE_CODE = '''// print a single list item to an externally allocated string
// Format: "quantity * item_name @ $price ea"
int list_item_to_string(node *head, char *str, size_t str_size) {
    if (head == NULL) return EXIT_FAILURE;
    '''

# SECURE prompt - forces snprintf continuation
SECURE_PROMPT = BASE_CODE + 'snprintf(str, str_size, '

# INSECURE prompt - forces sprintf continuation
INSECURE_PROMPT = BASE_CODE + 'sprintf(str, '


def classify_security(output: str, prompt: str) -> Dict:
    """Classify based on what function appears AFTER the prompt."""
    # Check if the continuation maintains the expected function
    full_text = prompt + output

    # Find all function calls
    snprintf_matches = list(re.finditer(r'\bsnprintf\s*\(', full_text))
    sprintf_matches = list(re.finditer(r'(?<!n)sprintf\s*\(', full_text))

    # Check what was generated (not what was in prompt)
    prompt_len = len(prompt)

    # Count functions in generated part
    gen_snprintf = sum(1 for m in snprintf_matches if m.start() >= prompt_len - 20)  # -20 for overlap
    gen_sprintf = sum(1 for m in sprintf_matches if m.start() >= prompt_len - 20)

    # The key question: did it continue with the prompted function or switch?
    if 'snprintf' in prompt[-20:]:
        # Started with snprintf - did it stay secure?
        continued_secure = gen_snprintf > 0 or 'snprintf' in output[:50]
        return {
            'started_as': 'snprintf',
            'continued_secure': continued_secure,
            'output_preview': output[:100]
        }
    else:
        # Started with sprintf - did it switch to secure?
        switched_to_secure = 'snprintf' in output
        return {
            'started_as': 'sprintf',
            'switched_to_secure': switched_to_secure,
            'output_preview': output[:100]
        }


class LayerSweepV2:
    """Layer sweep using forced continuation approach."""

    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        print("=" * 70)
        print("LAYER SWEEP V2: Forced Continuation Approach")
        print("=" * 70)

        print("\nLoading model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.model.eval()

        self.n_layers = self.model.config.num_hidden_layers
        self.n_heads = self.model.config.num_attention_heads
        print(f"Model loaded: {self.n_layers} layers, {self.n_heads} heads")

        self.saved_activations = {}
        self.hooks = []

    def get_attention_module(self, layer_idx: int) -> nn.Module:
        return self.model.model.layers[layer_idx].self_attn

    def save_activation_hook(self, key: str):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self.saved_activations[key] = hidden_states.detach().clone()
            return output
        return hook_fn

    def patch_activation_hook(self, saved_activation: torch.Tensor):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            batch_size, seq_len, hidden_dim = hidden_states.shape
            saved_seq_len = saved_activation.shape[1]
            min_seq_len = min(seq_len, saved_seq_len)

            new_hidden = hidden_states.clone()
            new_hidden[:, :min_seq_len, :] = saved_activation[:, :min_seq_len, :]

            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden
        return hook_fn

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.saved_activations = {}

    @contextmanager
    def save_context(self, prompt: str, layer_idx: int):
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
    def patch_context(self, saved_activation: torch.Tensor, layer_idx: int):
        try:
            module = self.get_attention_module(layer_idx)
            hook = module.register_forward_hook(
                self.patch_activation_hook(saved_activation)
            )
            self.hooks.append(hook)
            yield
        finally:
            self.clear_hooks()

    def generate(self, prompt: str, max_new_tokens: int = 80) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated[len(prompt):]

    def check_behavior_flip(self, output: str, direction: str) -> Tuple[bool, str]:
        """
        Check if the output shows the expected behavior flip.

        direction='to_secure': We patched secure into insecure, expect snprintf in output
        direction='to_insecure': We patched insecure into secure, expect no snprintf OR sprintf appears
        """
        has_snprintf = 'snprintf' in output
        has_sprintf = bool(re.search(r'(?<!n)sprintf', output))

        if direction == 'to_secure':
            # Success if snprintf appears in output (even though prompt started with sprintf)
            success = has_snprintf
            status = 'snprintf' if has_snprintf else 'sprintf' if has_sprintf else 'neither'
        else:
            # Success if sprintf appears or snprintf doesn't appear
            success = has_sprintf or not has_snprintf
            status = 'sprintf' if has_sprintf else 'snprintf' if has_snprintf else 'neither'

        return success, status

    def run_sweep(self, n_trials: int = 5) -> Dict:
        """Run layer sweep with bidirectional patching."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'n_trials': n_trials,
            'prompts': {
                'secure': SECURE_PROMPT[-50:],
                'insecure': INSECURE_PROMPT[-50:]
            },
            'baselines': {},
            'layer_results': {}
        }

        # Test baselines
        print("\n📊 Testing baselines (deterministic)...")
        print("-" * 50)

        print("\nSecure prompt (snprintf start):")
        secure_out = self.generate(SECURE_PROMPT)
        print(f"  {SECURE_PROMPT[-30:]}...")
        print(f"  ...{secure_out[:80]}")

        print("\nInsecure prompt (sprintf start):")
        insecure_out = self.generate(INSECURE_PROMPT)
        print(f"  {INSECURE_PROMPT[-30:]}...")
        print(f"  ...{insecure_out[:80]}")

        results['baselines'] = {
            'secure_output': secure_out[:200],
            'insecure_output': insecure_out[:200]
        }

        # Layer sweep - FORWARD direction (patch secure into insecure)
        print("\n" + "=" * 70)
        print("FORWARD PATCH: Secure attention → Insecure prompt")
        print("Goal: Make sprintf prompt produce snprintf-like behavior")
        print("=" * 70)

        for layer_idx in range(self.n_layers):
            print(f"\nLayer {layer_idx:2d}: ", end="", flush=True)

            forward_results = []

            # Save secure activation at this layer
            with self.save_context(SECURE_PROMPT, layer_idx) as saved:
                secure_activation = saved[f"layer_{layer_idx}"].clone()

            # Patch into insecure prompt
            for trial in range(n_trials):
                with self.patch_context(secure_activation, layer_idx):
                    output = self.generate(INSECURE_PROMPT)
                    success, status = self.check_behavior_flip(output, 'to_secure')
                    forward_results.append({
                        'success': success,
                        'status': status,
                        'output': output[:100]
                    })
                    print("✅" if success else "❌", end="", flush=True)

            # Calculate success rate
            success_count = sum(1 for r in forward_results if r['success'])
            success_rate = success_count / n_trials

            results['layer_results'][layer_idx] = {
                'forward_trials': forward_results,
                'forward_success_rate': success_rate
            }

            print(f" → {success_rate*100:5.1f}%")

        # Summary
        print("\n" + "=" * 70)
        print("FORWARD PATCH SUMMARY")
        print("=" * 70)

        sorted_layers = sorted(
            results['layer_results'].items(),
            key=lambda x: x[1]['forward_success_rate'],
            reverse=True
        )

        print("\nTop layers by forward success rate:")
        for layer_idx, data in sorted_layers[:10]:
            rate = data['forward_success_rate']
            bar = "█" * int(rate * 20)
            print(f"  Layer {layer_idx:2d}: {bar:20s} {rate*100:5.1f}%")

        # Causal layers
        causal_layers = [
            layer_idx for layer_idx, data in results['layer_results'].items()
            if data['forward_success_rate'] >= 0.4
        ]

        print(f"\n🎯 Potentially causal layers (≥40%): {causal_layers}")

        if not causal_layers:
            print("\n⚠️ No clear causal layer found.")
            print("This could mean:")
            print("  1. The decision is distributed across multiple layers")
            print("  2. sprintf/snprintf is encoded in token embeddings, not attention")
            print("  3. MLP layers (not attention) are responsible")

        results['causal_layers'] = causal_layers

        return results


def main():
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    sweeper = LayerSweepV2()
    results = sweeper.run_sweep(n_trials=5)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"layer_sweep_v2_{timestamp}.json"

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    with open(output_file, 'w') as f:
        json.dump(make_serializable(results), f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")

    return results


if __name__ == "__main__":
    results = main()
