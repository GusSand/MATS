#!/usr/bin/env python3
"""
Quick test: Run 1 fold (pair_12_xml) with 800 tokens to check if higher limit helps.

Compares to 512-token results to determine if full rerun is worthwhile.
"""

import sys
from pathlib import Path

import json
import numpy as np
from datetime import datetime
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiment_config import (
    DATA_DIR, FOLD_RESULTS_DIR,
    STEERING_LAYER, ALPHA_GRID,
    MODEL_NAME, GENERATIONS_PER_PROMPT,
)
from lobo_splits import (
    load_metadata, load_activations, load_dataset,
    get_lobo_splits, compute_fold_direction, get_test_prompts,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "01-12_llama8b_cwe787_baseline_behavior"))
from scoring import score_completion
from refusal_detection import detect_refusal


# Test configuration
TEST_FOLD = "pair_12_xml"
MAX_NEW_TOKENS = 800  # Testing higher limit

GENERATION_CONFIG = {
    "temperature": 0.6,
    "top_p": 0.9,
    "max_new_tokens": MAX_NEW_TOKENS,
    "do_sample": True,
}


class SteeringGenerator:
    """Generator with steering support."""

    def __init__(self, model_name: str):
        print(f"Loading model: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        self.model.eval()
        self.hooks = []

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def generate_with_steering(self, prompt, direction, layer, alpha,
                               temperature=0.6, top_p=0.9, max_tokens=800):
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
                **inputs, max_new_tokens=max_tokens, temperature=temperature,
                do_sample=True, top_p=top_p, pad_token_id=self.tokenizer.pad_token_id
            )
        self.clear_hooks()
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated[len(prompt):]


def run_fold(generator, fold, X, y, dataset, alpha_grid, layer, n_gens, gen_config):
    """Run α-sweep for a single fold."""
    fold_id = fold['fold_id']
    print(f"\n{'='*60}")
    print(f"FOLD: {fold_id}")
    print(f"{'='*60}")

    direction = compute_fold_direction(X, y, fold)
    print(f"Direction computed from {fold['n_train']} train samples")
    print(f"Direction norm: {np.linalg.norm(direction):.4f}")

    test_prompts = get_test_prompts(dataset, fold['test_indices'])
    print(f"Test prompts: {len(test_prompts)}")

    results = {
        'fold_id': fold_id,
        'n_train': fold['n_train'],
        'n_test': fold['n_test'],
        'direction_norm': float(np.linalg.norm(direction)),
        'alpha_results': {},
    }

    total_gens = len(alpha_grid) * len(test_prompts) * n_gens
    pbar = tqdm(total=total_gens, desc=f"Fold {fold_id}")

    for alpha in alpha_grid:
        alpha_key = str(alpha)
        results['alpha_results'][alpha_key] = []

        for item in test_prompts:
            prompt = item['vulnerable']
            vuln_type = item['vulnerability_type']

            for gen_idx in range(n_gens):
                output = generator.generate_with_steering(
                    prompt=prompt, direction=direction, layer=layer, alpha=alpha,
                    temperature=gen_config['temperature'], top_p=gen_config['top_p'],
                    max_tokens=gen_config['max_new_tokens'],
                )
                score_result = score_completion(output, vuln_type)
                is_refusal, _ = detect_refusal(output)

                results['alpha_results'][alpha_key].append({
                    'id': item['id'],
                    'base_id': item['base_id'],
                    'vulnerability_type': vuln_type,
                    'gen_idx': gen_idx,
                    'output': output[:800],  # Store more for analysis
                    'output_len': len(output),
                    'strict_label': score_result.strict_label,
                    'expanded_label': score_result.expanded_label,
                    'is_refusal': is_refusal,
                })
                pbar.update(1)

    pbar.close()
    return results


def summarize_results(fold_results):
    """Compute summary statistics."""
    print("\n" + "="*60)
    print(f"RESULTS: {TEST_FOLD} @ {MAX_NEW_TOKENS} tokens")
    print("="*60)

    print(f"\n{'Alpha':<8} {'Secure%':<12} {'Insecure%':<12} {'Other%':<10} {'Avg Len':<10}")
    print("-" * 52)

    for alpha in ALPHA_GRID:
        items = fold_results['alpha_results'][str(alpha)]
        n = len(items)
        secure = sum(1 for r in items if r['strict_label'] == 'secure')
        insecure = sum(1 for r in items if r['strict_label'] == 'insecure')
        other = n - secure - insecure
        avg_len = sum(r['output_len'] for r in items) / n

        print(f"{alpha:<8} {secure/n*100:>10.1f}% {insecure/n*100:>10.1f}% {other/n*100:>8.1f}% {avg_len:>8.0f}")

    # Compare to 512-token results
    print("\n" + "="*60)
    print("COMPARISON: 512 vs 800 tokens (pair_12_xml)")
    print("="*60)

    # Load 512-token results for this fold
    result_512 = FOLD_RESULTS_DIR / f"fold_{TEST_FOLD}_20260113_171820.json"
    if result_512.exists():
        with open(result_512) as f:
            data_512 = json.load(f)

        print(f"\n{'Alpha':<8} {'512 Secure%':<14} {'800 Secure%':<14} {'Δ':<8}")
        print("-" * 44)

        for alpha in ALPHA_GRID:
            items_512 = data_512['alpha_results'][str(alpha)]
            items_800 = fold_results['alpha_results'][str(alpha)]

            sec_512 = sum(1 for r in items_512 if r['strict_label'] == 'secure') / len(items_512) * 100
            sec_800 = sum(1 for r in items_800 if r['strict_label'] == 'secure') / len(items_800) * 100
            delta = sec_800 - sec_512

            print(f"{alpha:<8} {sec_512:>12.1f}% {sec_800:>12.1f}% {delta:>+7.1f}")
    else:
        print("512-token results not found for comparison")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("="*60)
    print(f"TEST: 800 Tokens on {TEST_FOLD}")
    print(f"Timestamp: {timestamp}")
    print("="*60)

    # Load data
    print("\n--- Loading Data ---")
    metadata = load_metadata()
    X, y = load_activations()
    dataset = load_dataset()

    # Get the specific fold
    folds = get_lobo_splits(metadata)
    test_fold = next(f for f in folds if f['fold_id'] == TEST_FOLD)

    print(f"Test fold: {test_fold['fold_id']}")
    print(f"Train: {test_fold['n_train']}, Test: {test_fold['n_test']}")

    # Initialize generator
    print("\n--- Initializing Model ---")
    generator = SteeringGenerator(MODEL_NAME)

    # Run fold
    print("\n--- Running α-Sweep ---")
    print(f"max_new_tokens: {MAX_NEW_TOKENS}")

    fold_results = run_fold(
        generator=generator, fold=test_fold, X=X, y=y, dataset=dataset,
        alpha_grid=ALPHA_GRID, layer=STEERING_LAYER,
        n_gens=GENERATIONS_PER_PROMPT, gen_config=GENERATION_CONFIG,
    )

    # Save results
    output_path = FOLD_RESULTS_DIR / f"fold_{TEST_FOLD}_800tok_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(fold_results, f, indent=2)
    print(f"\nSaved: {output_path}")

    # Print comparison
    summarize_results(fold_results)

    return str(output_path)


if __name__ == "__main__":
    output_path = main()
    print(f"\nResults saved to: {output_path}")
