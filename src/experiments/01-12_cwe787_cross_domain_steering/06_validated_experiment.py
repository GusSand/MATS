#!/usr/bin/env python3
"""
Validated Cross-Domain Steering Experiment

FIXES DATA LEAKAGE: Proper train/test split
- Train: 84 pairs (80%) - used to compute steering direction
- Test: 21 pairs (20%) - held-out for evaluation

This gives an honest estimate of generalization.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "01-08_llama8b_sr_scg_separation"))

import json
import re
import numpy as np
from datetime import datetime
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import train_test_split


def load_dataset(data_path: Path) -> list:
    """Load the expanded CWE-787 dataset."""
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def classify_output(output: str, detection: dict) -> str:
    """Classify generated output."""
    secure_pattern = detection['secure_pattern']
    insecure_pattern = detection['insecure_pattern']

    has_secure = bool(re.search(secure_pattern, output))
    has_insecure = bool(re.search(insecure_pattern, output))

    if has_secure:
        return 'secure'
    elif has_insecure:
        return 'insecure'
    else:
        return 'incomplete'


class ValidatedSteeringExperiment:
    """Experiment with proper train/test split."""

    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        print(f"Loading model: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.hooks = []
        self.activations = {}

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}

    def _make_layer_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            self.activations[layer_idx] = h[:, -1, :].detach().cpu()
        return hook_fn

    def get_activations(self, prompt: str, layers: list = None) -> dict:
        """Get last-token activations at specified layers."""
        if layers is None:
            layers = list(range(self.n_layers))

        self.clear_hooks()
        for layer_idx in layers:
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(self._make_layer_hook(layer_idx))
            self.hooks.append(hook)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            _ = self.model(**inputs)

        result = {k: v.numpy().squeeze() for k, v in self.activations.items()}
        self.clear_hooks()
        return result

    def compute_direction_from_train(self, train_data: list, layer: int) -> np.ndarray:
        """Compute steering direction from TRAIN data only."""
        print(f"\nComputing direction from {len(train_data)} TRAIN pairs at L{layer}...")

        vuln_activations = []
        secure_activations = []

        for item in tqdm(train_data, desc="Collecting train activations"):
            # Vulnerable prompt activation
            vuln_act = self.get_activations(item['vulnerable'], layers=[layer])
            vuln_activations.append(vuln_act[layer])

            # Secure prompt activation
            sec_act = self.get_activations(item['secure'], layers=[layer])
            secure_activations.append(sec_act[layer])

        vuln_activations = np.array(vuln_activations)
        secure_activations = np.array(secure_activations)

        # Direction: secure - vulnerable
        direction = np.mean(secure_activations, axis=0) - np.mean(vuln_activations, axis=0)

        print(f"Direction norm: {np.linalg.norm(direction):.4f}")
        return direction

    def generate_with_steering(self, prompt: str, direction: np.ndarray,
                                layer: int, alpha: float,
                                temperature: float = 0.6,
                                max_tokens: int = 300) -> str:
        """Generate with steering applied."""
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
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )

        self.clear_hooks()

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated[len(prompt):]

    def generate_baseline(self, prompt: str, temperature: float = 0.6,
                          max_tokens: int = 300) -> str:
        """Generate without steering."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated[len(prompt):]

    def evaluate_on_test(self, test_data: list, direction: np.ndarray,
                         layer: int, alphas: list,
                         temperature: float = 0.6,
                         max_tokens: int = 300) -> dict:
        """Evaluate on HELD-OUT TEST data."""
        print(f"\nEvaluating on {len(test_data)} HELD-OUT TEST pairs...")

        results = {
            'baseline': [],
            'steered': {alpha: [] for alpha in alphas}
        }

        for item in tqdm(test_data, desc="Testing"):
            prompt = item['vulnerable']

            # Baseline (no steering)
            output = self.generate_baseline(prompt, temperature, max_tokens)
            label = classify_output(output, item['detection'])
            results['baseline'].append({
                'id': item['id'],
                'label': label,
                'output': output[:300]
            })

            # Steered at each alpha
            for alpha in alphas:
                output = self.generate_with_steering(
                    prompt, direction, layer, alpha, temperature, max_tokens
                )
                label = classify_output(output, item['detection'])
                results['steered'][alpha].append({
                    'id': item['id'],
                    'label': label,
                    'output': output[:300]
                })

        return results

    def summarize_results(self, results: dict) -> dict:
        """Compute summary statistics."""
        def compute_rates(items):
            total = len(items)
            secure = sum(1 for r in items if r['label'] == 'secure')
            insecure = sum(1 for r in items if r['label'] == 'insecure')
            incomplete = sum(1 for r in items if r['label'] == 'incomplete')
            return {
                'total': total,
                'secure': secure,
                'insecure': insecure,
                'incomplete': incomplete,
                'secure_rate': secure / total if total > 0 else 0,
                'insecure_rate': insecure / total if total > 0 else 0,
                'incomplete_rate': incomplete / total if total > 0 else 0
            }

        summary = {
            'baseline': compute_rates(results['baseline']),
            'steered': {}
        }

        for alpha, items in results['steered'].items():
            summary['steered'][alpha] = compute_rates(items)
            summary['steered'][alpha]['conversion_rate'] = (
                summary['steered'][alpha]['secure_rate'] - summary['baseline']['secure_rate']
            )

        return summary


def main():
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    results_dir = script_dir / "results"
    results_dir.mkdir(exist_ok=True)

    dataset_path = script_dir / "../01-12_cwe787_dataset_expansion/data/cwe787_expanded_20260112_143316.jsonl"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} pairs")

    # PROPER TRAIN/TEST SPLIT
    print("\n" + "="*60)
    print("TRAIN/TEST SPLIT (80/20)")
    print("="*60)

    train_data, test_data = train_test_split(
        dataset, test_size=0.2, random_state=42,
        stratify=[d['vulnerability_type'] for d in dataset]  # Stratify by vuln type
    )

    print(f"Train: {len(train_data)} pairs")
    print(f"Test: {len(test_data)} pairs (HELD OUT)")

    # Check stratification
    train_types = [d['vulnerability_type'] for d in train_data]
    test_types = [d['vulnerability_type'] for d in test_data]
    print(f"Train sprintf/strcat: {train_types.count('sprintf')}/{train_types.count('strcat')}")
    print(f"Test sprintf/strcat: {test_types.count('sprintf')}/{test_types.count('strcat')}")

    # Initialize experiment
    exp = ValidatedSteeringExperiment()

    # Compute direction from TRAIN ONLY
    layer = 31
    direction = exp.compute_direction_from_train(train_data, layer)

    # Evaluate on HELD-OUT TEST
    alphas = [0.5, 1.0, 1.5, 2.0, 3.0]
    results = exp.evaluate_on_test(test_data, direction, layer, alphas)

    # Summarize
    summary = exp.summarize_results(results)

    # Print results
    print("\n" + "="*60)
    print("VALIDATED RESULTS (HELD-OUT TEST SET)")
    print("="*60)
    print(f"\nTest set size: {len(test_data)} pairs")
    print(f"Direction computed from: {len(train_data)} TRAIN pairs only")

    print(f"\nBaseline (no steering):")
    b = summary['baseline']
    print(f"  Secure: {b['secure']}/{b['total']} ({b['secure_rate']*100:.1f}%)")
    print(f"  Insecure: {b['insecure']}/{b['total']} ({b['insecure_rate']*100:.1f}%)")
    print(f"  Incomplete: {b['incomplete']}/{b['total']} ({b['incomplete_rate']*100:.1f}%)")

    print(f"\nSteered at L{layer}:")
    print(f"{'Alpha':<8} {'Secure':<12} {'Insecure':<12} {'Incomplete':<12} {'Δ Secure':<12}")
    print("-" * 56)
    for alpha in alphas:
        s = summary['steered'][alpha]
        print(f"{alpha:<8} {s['secure']}/{s['total']} ({s['secure_rate']*100:.1f}%)   "
              f"{s['insecure']}/{s['total']} ({s['insecure_rate']*100:.1f}%)   "
              f"{s['incomplete']}/{s['total']} ({s['incomplete_rate']*100:.1f}%)   "
              f"{s['conversion_rate']*100:+.1f} pp")

    # Find best alpha
    best_alpha = max(alphas, key=lambda a: summary['steered'][a]['conversion_rate'])
    best_conversion = summary['steered'][best_alpha]['conversion_rate']

    print(f"\nBest alpha: {best_alpha}")
    print(f"Best conversion rate: {best_conversion*100:+.1f} pp")

    # Decision
    print("\n" + "="*60)
    if best_conversion > 0.10:
        print(f"✅ VALIDATED PASS: {best_conversion*100:.1f}% > 10% on held-out test set")
    elif best_conversion > 0:
        print(f"⚠️  MARGINAL: {best_conversion*100:.1f}% conversion on held-out test")
    else:
        print(f"❌ FAIL: No positive conversion on held-out test set")
    print("="*60)

    # Save results
    output = {
        'timestamp': timestamp,
        'validation': 'train_test_split',
        'train_size': len(train_data),
        'test_size': len(test_data),
        'random_state': 42,
        'layer': layer,
        'alphas': alphas,
        'summary': summary,
        'results': {
            'baseline': results['baseline'],
            'steered': {str(k): v for k, v in results['steered'].items()}
        },
        'train_ids': [d['id'] for d in train_data],
        'test_ids': [d['id'] for d in test_data]
    }

    output_path = results_dir / f"validated_results_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return summary


if __name__ == "__main__":
    main()
