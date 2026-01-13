#!/usr/bin/env python3
"""
Experiment 2: LOBO Steering α-Sweep

Leave-One-Base-ID-Out cross-validation to prove steering generalizes
across scenario families (not just paraphrases).

Flow:
1. Load cached activations and metadata
2. For each of 7 LOBO folds:
   - Compute direction from train split only
   - Run α-sweep generation on test prompts
   - Score with STRICT + EXPANDED + refusal
3. Save per-fold and aggregated results
"""

import sys
from pathlib import Path

import json
import numpy as np
from datetime import datetime
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import local config first (before adding other paths)
from experiment_config import (
    DATA_DIR, FOLD_RESULTS_DIR,
    ACTIVATION_CACHE, METADATA_CACHE, DATASET_PATH,
    BASE_IDS, STEERING_LAYER, ALPHA_GRID,
    MODEL_NAME, GENERATION_CONFIG, GENERATIONS_PER_PROMPT,
)
from lobo_splits import (
    load_metadata, load_activations, load_dataset,
    get_lobo_splits, compute_fold_direction, get_test_prompts,
)

# Now add path for scoring/refusal_detection from baseline experiment
sys.path.insert(0, str(Path(__file__).parent.parent / "01-12_llama8b_cwe787_baseline_behavior"))
from scoring import score_completion
from refusal_detection import detect_refusal


class SteeringGenerator:
    """Generator with steering support (adapted from prior experiment)."""

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

    def generate_with_steering(
        self,
        prompt: str,
        direction: np.ndarray,
        layer: int,
        alpha: float,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_tokens: int = 300,
    ) -> str:
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
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id
            )

        self.clear_hooks()

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated[len(prompt):]


def run_fold(
    generator: SteeringGenerator,
    fold: dict,
    X: np.ndarray,
    y: np.ndarray,
    dataset: list,
    alpha_grid: list,
    layer: int,
    n_gens: int,
    gen_config: dict,
) -> dict:
    """
    Run α-sweep for a single LOBO fold.

    Args:
        generator: SteeringGenerator instance
        fold: Fold dict with train/test indices
        X: All activations (210, 4096)
        y: All labels (210,)
        dataset: Full dataset
        alpha_grid: List of alpha values
        layer: Steering layer
        n_gens: Generations per prompt
        gen_config: Generation configuration

    Returns:
        Fold results dict
    """
    fold_id = fold['fold_id']
    print(f"\n{'='*60}")
    print(f"FOLD: {fold_id}")
    print(f"{'='*60}")

    # Compute direction from train split only
    direction = compute_fold_direction(X, y, fold)
    print(f"Direction computed from {fold['n_train']} train samples")
    print(f"Direction norm: {np.linalg.norm(direction):.4f}")

    # Get test prompts
    test_prompts = get_test_prompts(dataset, fold['test_indices'])
    print(f"Test prompts: {len(test_prompts)}")

    # Initialize results structure
    results = {
        'fold_id': fold_id,
        'n_train': fold['n_train'],
        'n_test': fold['n_test'],
        'direction_norm': float(np.linalg.norm(direction)),
        'alpha_results': {},
    }

    # Run α-sweep
    total_gens = len(alpha_grid) * len(test_prompts) * n_gens
    pbar = tqdm(total=total_gens, desc=f"Fold {fold_id}")

    for alpha in alpha_grid:
        alpha_key = str(alpha)
        results['alpha_results'][alpha_key] = []

        for item in test_prompts:
            prompt = item['vulnerable']
            vuln_type = item['vulnerability_type']

            for gen_idx in range(n_gens):
                # Generate with steering
                output = generator.generate_with_steering(
                    prompt=prompt,
                    direction=direction,
                    layer=layer,
                    alpha=alpha,
                    temperature=gen_config['temperature'],
                    top_p=gen_config['top_p'],
                    max_tokens=gen_config['max_new_tokens'],
                )

                # Score
                score_result = score_completion(output, vuln_type)
                is_refusal, refusal_details = detect_refusal(output)

                results['alpha_results'][alpha_key].append({
                    'id': item['id'],
                    'base_id': item['base_id'],
                    'vulnerability_type': vuln_type,
                    'gen_idx': gen_idx,
                    'output': output[:500],  # Truncate for storage
                    'strict_label': score_result.strict_label,
                    'expanded_label': score_result.expanded_label,
                    'is_refusal': is_refusal,
                })

                pbar.update(1)

    pbar.close()
    return results


def summarize_fold_results(fold_results: dict) -> dict:
    """Compute summary statistics for a fold."""
    summaries = {}

    for alpha_key, items in fold_results['alpha_results'].items():
        n = len(items)

        # STRICT scoring
        strict_secure = sum(1 for r in items if r['strict_label'] == 'secure')
        strict_insecure = sum(1 for r in items if r['strict_label'] == 'insecure')
        strict_other = sum(1 for r in items if r['strict_label'] == 'other')

        # EXPANDED scoring
        expanded_secure = sum(1 for r in items if r['expanded_label'] == 'secure')
        expanded_insecure = sum(1 for r in items if r['expanded_label'] == 'insecure')
        expanded_other = sum(1 for r in items if r['expanded_label'] == 'other')

        # Refusals
        refusals = sum(1 for r in items if r['is_refusal'])

        summaries[alpha_key] = {
            'n': n,
            'strict': {
                'secure': strict_secure,
                'insecure': strict_insecure,
                'other': strict_other,
                'secure_rate': strict_secure / n if n > 0 else 0,
                'insecure_rate': strict_insecure / n if n > 0 else 0,
            },
            'expanded': {
                'secure': expanded_secure,
                'insecure': expanded_insecure,
                'other': expanded_other,
                'secure_rate': expanded_secure / n if n > 0 else 0,
                'insecure_rate': expanded_insecure / n if n > 0 else 0,
            },
            'refusal_rate': refusals / n if n > 0 else 0,
        }

    return summaries


def aggregate_all_folds(all_fold_results: list) -> dict:
    """Aggregate results across all folds for each alpha."""
    alpha_keys = list(all_fold_results[0]['alpha_results'].keys())
    aggregated = {}

    for alpha_key in alpha_keys:
        # Collect all items for this alpha across folds
        all_items = []
        for fold_results in all_fold_results:
            all_items.extend(fold_results['alpha_results'][alpha_key])

        n = len(all_items)

        # STRICT scoring
        strict_secure = sum(1 for r in all_items if r['strict_label'] == 'secure')
        strict_insecure = sum(1 for r in all_items if r['strict_label'] == 'insecure')

        # EXPANDED scoring
        expanded_secure = sum(1 for r in all_items if r['expanded_label'] == 'secure')
        expanded_insecure = sum(1 for r in all_items if r['expanded_label'] == 'insecure')

        # Refusals
        refusals = sum(1 for r in all_items if r['is_refusal'])

        aggregated[alpha_key] = {
            'n': n,
            'strict_secure_rate': strict_secure / n if n > 0 else 0,
            'strict_insecure_rate': strict_insecure / n if n > 0 else 0,
            'expanded_secure_rate': expanded_secure / n if n > 0 else 0,
            'expanded_insecure_rate': expanded_insecure / n if n > 0 else 0,
            'refusal_rate': refusals / n if n > 0 else 0,
        }

    return aggregated


def main():
    """Run the full LOBO experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("="*60)
    print("EXPERIMENT 2: LOBO Steering α-Sweep")
    print(f"Timestamp: {timestamp}")
    print("="*60)

    # Create output directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FOLD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n--- Loading Data ---")
    metadata = load_metadata()
    X, y = load_activations()
    dataset = load_dataset()

    print(f"Activations: {X.shape}")
    print(f"Labels: {y.shape} (0=vulnerable, 1=secure)")
    print(f"Dataset: {len(dataset)} pairs")

    # Generate LOBO splits
    print("\n--- Generating LOBO Splits ---")
    folds = get_lobo_splits(metadata)
    print(f"Generated {len(folds)} folds")

    for fold in folds:
        print(f"  {fold['fold_id']}: train={fold['n_train']}, test={fold['n_test']}")

    # Initialize generator
    print("\n--- Initializing Model ---")
    generator = SteeringGenerator(MODEL_NAME)

    # Run experiment
    print("\n--- Running α-Sweep ---")
    print(f"Layer: L{STEERING_LAYER}")
    print(f"Alphas: {ALPHA_GRID}")
    print(f"Generations per prompt: {GENERATIONS_PER_PROMPT}")

    all_fold_results = []
    all_fold_summaries = []

    for fold in folds:
        # Run fold
        fold_results = run_fold(
            generator=generator,
            fold=fold,
            X=X,
            y=y,
            dataset=dataset,
            alpha_grid=ALPHA_GRID,
            layer=STEERING_LAYER,
            n_gens=GENERATIONS_PER_PROMPT,
            gen_config=GENERATION_CONFIG,
        )

        # Compute fold summary
        fold_summary = summarize_fold_results(fold_results)
        fold_results['summary'] = fold_summary

        all_fold_results.append(fold_results)
        all_fold_summaries.append({
            'fold_id': fold_results['fold_id'],
            'summary': fold_summary,
        })

        # Save fold results
        fold_path = FOLD_RESULTS_DIR / f"fold_{fold['fold_id']}_{timestamp}.json"
        with open(fold_path, 'w') as f:
            json.dump(fold_results, f, indent=2)
        print(f"\nSaved fold results: {fold_path}")

    # Aggregate across folds
    print("\n--- Aggregating Results ---")
    aggregated = aggregate_all_folds(all_fold_results)

    # Save aggregated results
    output = {
        'timestamp': timestamp,
        'config': {
            'model': MODEL_NAME,
            'layer': STEERING_LAYER,
            'alpha_grid': ALPHA_GRID,
            'generations_per_prompt': GENERATIONS_PER_PROMPT,
            'generation_config': GENERATION_CONFIG,
            'n_folds': len(folds),
            'base_ids': BASE_IDS,
        },
        'fold_summaries': all_fold_summaries,
        'aggregated': aggregated,
    }

    output_path = DATA_DIR / f"lobo_results_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved aggregated results: {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("LOBO EXPERIMENT COMPLETE")
    print("="*60)

    print("\nAggregated Results (STRICT scoring):")
    print(f"{'Alpha':<8} {'Secure%':<12} {'Insecure%':<12} {'Refusal%':<10}")
    print("-" * 42)

    for alpha in ALPHA_GRID:
        alpha_key = str(alpha)
        r = aggregated[alpha_key]
        print(f"{alpha:<8} {r['strict_secure_rate']*100:>10.1f}% {r['strict_insecure_rate']*100:>10.1f}% {r['refusal_rate']*100:>8.1f}%")

    print("\n" + "="*60)

    # Return path for downstream use
    return str(output_path)


if __name__ == "__main__":
    output_path = main()
    print(f"\nResults saved to: {output_path}")
