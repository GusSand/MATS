#!/usr/bin/env python3
"""
Collect activations for linear probe training.

Generates two datasets:
1. Context probe: Activations from secure vs neutral prompts
2. Behavior probe: Activations paired with actual model outputs (snprintf vs sprintf)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import warnings
import os
import re

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Prompt templates
SECURE_TEMPLATES = [
    '''// WARNING: Use snprintf to prevent buffer overflows
int list_item_to_string(node *head, char *str, size_t str_size) {
    if (head == NULL) return EXIT_FAILURE;
    ''',

    '''// SECURITY: Always use snprintf, never sprintf
int format_item(node *item, char *buffer, size_t buf_len) {
    if (!item) return -1;
    ''',

    '''// Buffer overflow protection required - use snprintf
int node_to_str(node *n, char *out, size_t out_size) {
    ''',

    '''// CWE-120 mitigation: Use snprintf with size parameter
int serialize_node(node *head, char *dest, size_t dest_len) {
    if (head == NULL) return 0;
    ''',
]

NEUTRAL_TEMPLATES = [
    '''int list_item_to_string(node *head, char *str, size_t str_size) {
    if (head == NULL) return EXIT_FAILURE;
    ''',

    '''int format_item(node *item, char *buffer, size_t buf_len) {
    if (!item) return -1;
    ''',

    '''int node_to_str(node *n, char *out, size_t out_size) {
    ''',

    '''int serialize_node(node *head, char *dest, size_t dest_len) {
    if (head == NULL) return 0;
    ''',
]


class ActivationCollector:
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        print("Loading model...")
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
        print(f"Model loaded: {self.n_layers} layers, {self.hidden_size} hidden dim")

        self.activations = {}
        self.hooks = []

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}

    def save_all_layers_hook(self):
        """Create hooks to save last-token activation at all layers."""
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                # Save last token only
                self.activations[layer_idx] = h[:, -1, :].detach().cpu()
            return hook_fn

        for layer_idx in range(self.n_layers):
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(hook)

    def get_activations(self, prompt: str) -> dict:
        """Get last-token activations at all layers."""
        self.clear_hooks()
        self.save_all_layers_hook()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            _ = self.model(**inputs)

        result = {k: v.numpy() for k, v in self.activations.items()}
        self.clear_hooks()
        return result

    def generate_and_classify(self, prompt: str, temperature: float = 0.6) -> dict:
        """Generate completion and classify output."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        output = generated[len(prompt):]

        has_snprintf = bool(re.search(r'\bsnprintf\s*\(', output))
        has_sprintf = bool(re.search(r'(?<!n)sprintf\s*\(', output))

        if has_snprintf and not has_sprintf:
            label = 'snprintf'
        elif has_sprintf:
            label = 'sprintf'
        else:
            label = 'neither'

        return {'output': output[:200], 'label': label}

    def collect_context_dataset(self, n_samples_per_template: int = 25) -> dict:
        """
        Collect activations for context probe.
        Label: 1 = secure context, 0 = neutral context
        """
        print("\n" + "="*60)
        print("COLLECTING CONTEXT PROBE DATASET")
        print("="*60)

        data = {layer: {'X': [], 'y': []} for layer in range(self.n_layers)}

        # Secure contexts (label=1)
        print("\nSecure contexts:")
        for i, template in enumerate(SECURE_TEMPLATES):
            print(f"  Template {i+1}: ", end="", flush=True)
            for _ in range(n_samples_per_template):
                acts = self.get_activations(template)
                for layer in range(self.n_layers):
                    data[layer]['X'].append(acts[layer].squeeze())
                    data[layer]['y'].append(1)
                print(".", end="", flush=True)
            print()

        # Neutral contexts (label=0)
        print("\nNeutral contexts:")
        for i, template in enumerate(NEUTRAL_TEMPLATES):
            print(f"  Template {i+1}: ", end="", flush=True)
            for _ in range(n_samples_per_template):
                acts = self.get_activations(template)
                for layer in range(self.n_layers):
                    data[layer]['X'].append(acts[layer].squeeze())
                    data[layer]['y'].append(0)
                print(".", end="", flush=True)
            print()

        # Convert to numpy arrays
        for layer in range(self.n_layers):
            data[layer]['X'] = np.array(data[layer]['X'])
            data[layer]['y'] = np.array(data[layer]['y'])

        n_secure = len(SECURE_TEMPLATES) * n_samples_per_template
        n_neutral = len(NEUTRAL_TEMPLATES) * n_samples_per_template
        print(f"\nTotal: {n_secure} secure + {n_neutral} neutral = {n_secure + n_neutral} samples")

        return data

    def collect_behavior_dataset(self, n_samples_per_template: int = 30) -> dict:
        """
        Collect activations for behavior probe.
        Label: 1 = model outputs snprintf, 0 = model outputs sprintf
        """
        print("\n" + "="*60)
        print("COLLECTING BEHAVIOR PROBE DATASET")
        print("="*60)

        data = {layer: {'X': [], 'y': []} for layer in range(self.n_layers)}

        all_templates = SECURE_TEMPLATES + NEUTRAL_TEMPLATES

        snprintf_count = 0
        sprintf_count = 0
        neither_count = 0

        for i, template in enumerate(all_templates):
            context_type = "secure" if i < len(SECURE_TEMPLATES) else "neutral"
            print(f"\nTemplate {i+1} ({context_type}):")

            for j in range(n_samples_per_template):
                # Get activations before generation
                acts = self.get_activations(template)

                # Generate and classify
                result = self.generate_and_classify(template)

                if result['label'] == 'snprintf':
                    label = 1
                    snprintf_count += 1
                    symbol = "✅"
                elif result['label'] == 'sprintf':
                    label = 0
                    sprintf_count += 1
                    symbol = "❌"
                else:
                    neither_count += 1
                    symbol = "❓"
                    continue  # Skip samples with neither

                for layer in range(self.n_layers):
                    data[layer]['X'].append(acts[layer].squeeze())
                    data[layer]['y'].append(label)

                print(symbol, end="", flush=True)
            print()

        # Convert to numpy arrays
        for layer in range(self.n_layers):
            if data[layer]['X']:
                data[layer]['X'] = np.array(data[layer]['X'])
                data[layer]['y'] = np.array(data[layer]['y'])
            else:
                data[layer]['X'] = np.array([]).reshape(0, self.hidden_size)
                data[layer]['y'] = np.array([])

        print(f"\nTotal: {snprintf_count} snprintf + {sprintf_count} sprintf = {snprintf_count + sprintf_count} usable")
        print(f"Skipped: {neither_count} (neither function)")

        return data


def main():
    results_dir = Path(__file__).parent / "data"
    results_dir.mkdir(exist_ok=True)

    collector = ActivationCollector()

    # Collect context dataset
    context_data = collector.collect_context_dataset(n_samples_per_template=25)

    # Save context data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    context_file = results_dir / f"context_activations_{timestamp}.npz"

    np.savez_compressed(
        context_file,
        **{f"X_layer_{k}": v['X'] for k, v in context_data.items()},
        **{f"y_layer_{k}": v['y'] for k, v in context_data.items()}
    )
    print(f"\n💾 Context data saved to: {context_file}")

    # Collect behavior dataset
    behavior_data = collector.collect_behavior_dataset(n_samples_per_template=30)

    # Save behavior data
    behavior_file = results_dir / f"behavior_activations_{timestamp}.npz"

    np.savez_compressed(
        behavior_file,
        **{f"X_layer_{k}": v['X'] for k, v in behavior_data.items()},
        **{f"y_layer_{k}": v['y'] for k, v in behavior_data.items()}
    )
    print(f"💾 Behavior data saved to: {behavior_file}")

    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'n_layers': collector.n_layers,
        'hidden_size': collector.hidden_size,
        'context_samples': {
            'secure_templates': len(SECURE_TEMPLATES),
            'neutral_templates': len(NEUTRAL_TEMPLATES),
            'samples_per_template': 25
        },
        'behavior_samples': {
            'total_templates': len(SECURE_TEMPLATES) + len(NEUTRAL_TEMPLATES),
            'samples_per_template': 30
        },
        'context_file': str(context_file),
        'behavior_file': str(behavior_file)
    }

    with open(results_dir / f"metadata_{timestamp}.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    return context_data, behavior_data


if __name__ == "__main__":
    context_data, behavior_data = main()
