#!/usr/bin/env python3
"""
Step 4: Generate outputs with steering applied.

Applies steering vector at specified layer(s) with various alpha values.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "01-08_llama8b_sr_scg_separation"))

import json
import re
import numpy as np
from datetime import datetime
from tqdm import tqdm
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_dataset(data_path: Path) -> list:
    """Load the expanded CWE-787 dataset."""
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_directions(dir_path: Path) -> dict:
    """Load steering directions."""
    data = np.load(dir_path)
    directions = {}

    for key in data.keys():
        if key.startswith('direction_layer_') and 'normalized' not in key:
            layer = int(key.replace('direction_layer_', ''))
            directions[layer] = data[key]

    return directions


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


class SteeringGenerator:
    """Generator with steering support."""

    def __init__(self, model_name: str):
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
        self.hooks = []

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def generate_with_steering(self, prompt: str, direction: np.ndarray,
                                layer: int, alpha: float,
                                temperature: float = 0.6,
                                max_tokens: int = 300) -> str:
        """Generate with steering applied at specified layer."""
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


def run_alpha_sweep(generator: SteeringGenerator, dataset: list,
                    directions: dict, layer: int, alphas: list,
                    temperature: float, max_tokens: int) -> dict:
    """Run steering with multiple alpha values at a single layer."""
    results = {alpha: [] for alpha in alphas}
    direction = directions[layer]

    print(f"\nRunning alpha sweep at L{layer}")
    print(f"Alphas: {alphas}")
    print(f"Prompts: {len(dataset)}")

    total_iterations = len(alphas) * len(dataset)
    pbar = tqdm(total=total_iterations, desc=f"L{layer} sweep")

    for alpha in alphas:
        for item in dataset:
            prompt = item['vulnerable']

            output = generator.generate_with_steering(
                prompt, direction, layer, alpha,
                temperature=temperature, max_tokens=max_tokens
            )

            label = classify_output(output, item['detection'])

            results[alpha].append({
                'id': item['id'],
                'base_id': item['base_id'],
                'vulnerability_type': item['vulnerability_type'],
                'output': output[:500],
                'label': label
            })

            pbar.update(1)

    pbar.close()
    return results


def run_layer_sweep(generator: SteeringGenerator, dataset: list,
                    directions: dict, layers: list, alpha: float,
                    temperature: float, max_tokens: int) -> dict:
    """Run steering across multiple layers at a single alpha."""
    results = {layer: [] for layer in layers}

    print(f"\nRunning layer sweep with alpha={alpha}")
    print(f"Layers: {layers}")
    print(f"Prompts: {len(dataset)}")

    total_iterations = len(layers) * len(dataset)
    pbar = tqdm(total=total_iterations, desc=f"Layer sweep (α={alpha})")

    for layer in layers:
        direction = directions[layer]

        for item in dataset:
            prompt = item['vulnerable']

            output = generator.generate_with_steering(
                prompt, direction, layer, alpha,
                temperature=temperature, max_tokens=max_tokens
            )

            label = classify_output(output, item['detection'])

            results[layer].append({
                'id': item['id'],
                'base_id': item['base_id'],
                'vulnerability_type': item['vulnerability_type'],
                'output': output[:500],
                'label': label
            })

            pbar.update(1)

    pbar.close()
    return results


def summarize_results(results: dict, key_type: str = 'alpha') -> dict:
    """Compute summary statistics for sweep results."""
    summaries = {}

    for key, items in results.items():
        total = len(items)
        secure = sum(1 for r in items if r['label'] == 'secure')
        insecure = sum(1 for r in items if r['label'] == 'insecure')
        incomplete = sum(1 for r in items if r['label'] == 'incomplete')

        summaries[key] = {
            'total': total,
            'secure': secure,
            'insecure': insecure,
            'incomplete': incomplete,
            'secure_rate': secure / total if total > 0 else 0,
            'insecure_rate': insecure / total if total > 0 else 0,
            'incomplete_rate': incomplete / total if total > 0 else 0
        }

    return summaries


def main():
    parser = argparse.ArgumentParser(description="Generate with steering")
    parser.add_argument("--dataset", type=str,
                        default="../01-12_cwe787_dataset_expansion/data/cwe787_expanded_20260112_143316.jsonl")
    parser.add_argument("--directions", type=str, required=True,
                        help="Path to directions NPZ file")
    parser.add_argument("--model", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--layer", type=int, default=31,
                        help="Layer for alpha sweep")
    parser.add_argument("--alphas", type=str, default="0.5,1.0,1.5,2.0,3.0",
                        help="Comma-separated alpha values")
    parser.add_argument("--mode", type=str, choices=['alpha_sweep', 'layer_sweep'],
                        default='alpha_sweep')
    parser.add_argument("--sweep-layers", type=str, default=None,
                        help="Layers for layer sweep (e.g., '16,20,24,28,31')")
    parser.add_argument("--sweep-alpha", type=float, default=1.0,
                        help="Alpha for layer sweep")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"

    dataset_path = script_dir / args.dataset
    if not dataset_path.exists():
        dataset_path = Path(args.dataset)

    dir_path = Path(args.directions)
    if not dir_path.is_absolute():
        dir_path = data_dir / args.directions

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load data
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} pairs")

    print(f"Loading directions from: {dir_path}")
    directions = load_directions(dir_path)
    print(f"Loaded directions for {len(directions)} layers")

    # Initialize generator
    generator = SteeringGenerator(args.model)

    # Run sweep
    if args.mode == 'alpha_sweep':
        alphas = [float(a) for a in args.alphas.split(',')]

        results = run_alpha_sweep(
            generator, dataset, directions, args.layer, alphas,
            args.temperature, args.max_tokens
        )

        summaries = summarize_results(results, 'alpha')

        output = {
            'timestamp': timestamp,
            'mode': 'alpha_sweep',
            'config': {
                'model': args.model,
                'temperature': args.temperature,
                'max_tokens': args.max_tokens,
                'layer': args.layer,
                'alphas': alphas,
                'n_prompts': len(dataset)
            },
            'summaries': {str(k): v for k, v in summaries.items()},
            'results': {str(k): v for k, v in results.items()}
        }

        output_path = data_dir / f"steered_L{args.layer}_alpha_sweep_{timestamp}.json"

    else:  # layer_sweep
        if args.sweep_layers:
            layers = [int(l) for l in args.sweep_layers.split(',')]
        else:
            layers = list(range(32))

        results = run_layer_sweep(
            generator, dataset, directions, layers, args.sweep_alpha,
            args.temperature, args.max_tokens
        )

        summaries = summarize_results(results, 'layer')

        output = {
            'timestamp': timestamp,
            'mode': 'layer_sweep',
            'config': {
                'model': args.model,
                'temperature': args.temperature,
                'max_tokens': args.max_tokens,
                'alpha': args.sweep_alpha,
                'layers': layers,
                'n_prompts': len(dataset)
            },
            'summaries': {str(k): v for k, v in summaries.items()},
            'results': {str(k): v for k, v in results.items()}
        }

        output_path = data_dir / f"steered_layer_sweep_a{args.sweep_alpha}_{timestamp}.json"

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    # Print summary
    print("\n" + "="*50)
    print("STEERED GENERATION COMPLETE")
    print("="*50)

    if args.mode == 'alpha_sweep':
        print(f"\nAlpha sweep at L{args.layer}:")
        print(f"{'Alpha':<8} {'Secure':<10} {'Insecure':<10} {'Incomplete':<10} {'Secure %':<10}")
        print("-" * 48)
        for alpha in alphas:
            s = summaries[alpha]
            print(f"{alpha:<8} {s['secure']:<10} {s['insecure']:<10} {s['incomplete']:<10} {s['secure_rate']*100:.1f}%")
    else:
        print(f"\nLayer sweep with α={args.sweep_alpha}:")
        print(f"{'Layer':<8} {'Secure':<10} {'Insecure':<10} {'Incomplete':<10} {'Secure %':<10}")
        print("-" * 48)
        for layer in layers:
            s = summaries[layer]
            print(f"L{layer:<7} {s['secure']:<10} {s['insecure']:<10} {s['incomplete']:<10} {s['secure_rate']*100:.1f}%")

    print(f"\nOutput: {output_path}")

    return str(output_path)


if __name__ == "__main__":
    main()
