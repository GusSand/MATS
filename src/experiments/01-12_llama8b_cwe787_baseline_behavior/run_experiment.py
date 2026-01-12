#!/usr/bin/env python3
"""
Experiment 1: Baseline Behavior (Base vs Expanded)

Goal: Show the unsteered model's security behavior and why the Expanded dataset
is necessary (stability + diversity).

Usage:
    python run_experiment.py                    # Run full experiment
    python run_experiment.py --dataset base     # Run only Base
    python run_experiment.py --dataset expanded # Run only Expanded
    python run_experiment.py --dry-run          # Test without GPU
"""

import sys
from pathlib import Path

# Add experiment directory to path
sys.path.insert(0, str(Path(__file__).parent))

import json
import argparse
from datetime import datetime
from typing import List, Dict, Any
from tqdm import tqdm

from experiment_config import (
    MODEL_NAME,
    GENERATION_CONFIG,
    GENERATIONS_PER_PROMPT,
    DATA_DIR,
    BASE_PAIRS_MODULE,
    EXPANDED_DATASET_PATH,
)
from scoring import score_completion, ScoringResult
from refusal_detection import detect_refusal
from analysis import (
    aggregate_overall,
    aggregate_by_base_id,
    aggregate_by_vuln_type,
    format_comparison_table,
    format_by_base_id_table,
    format_by_vuln_type_table,
    results_to_serializable,
)


def load_base_dataset() -> List[Dict]:
    """Load the 7 validated base pairs."""
    sys.path.insert(0, str(BASE_PAIRS_MODULE))
    from validated_pairs import get_all_pairs

    pairs = get_all_pairs()
    print(f"Loaded {len(pairs)} base pairs")

    # Convert to standard format
    dataset = []
    for pair in pairs:
        dataset.append({
            'id': pair['id'],
            'base_id': pair['id'],  # For base, id == base_id
            'name': pair['name'],
            'vulnerable': pair['vulnerable'],
            'vulnerability_type': pair['vulnerability_type'],
            'detection': pair['detection'],
        })

    return dataset


def load_expanded_dataset() -> List[Dict]:
    """Load the 105 expanded pairs."""
    dataset = []
    with open(EXPANDED_DATASET_PATH) as f:
        for line in f:
            dataset.append(json.loads(line))

    print(f"Loaded {len(dataset)} expanded pairs")
    return dataset


def load_model():
    """Load the model and tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    return model, tokenizer


def generate_completion(
    model,
    tokenizer,
    prompt: str,
    config: Dict,
) -> str:
    """Generate a single completion."""
    import torch

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config['max_new_tokens'],
            temperature=config['temperature'],
            do_sample=config['do_sample'],
            top_p=config['top_p'],
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the output
    completion = generated[len(prompt):]

    return completion


def run_generations(
    model,
    tokenizer,
    dataset: List[Dict],
    n_generations: int,
    dataset_name: str,
) -> List[Dict]:
    """
    Run multiple generations per prompt and score each.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        dataset: List of prompt dicts
        n_generations: Number of generations per prompt
        dataset_name: 'base' or 'expanded' (for logging)

    Returns:
        List of result dicts
    """
    results = []
    total = len(dataset) * n_generations

    print(f"\nGenerating {total} completions ({len(dataset)} prompts x {n_generations} gens)...")

    pbar = tqdm(total=total, desc=f"Generating ({dataset_name})")

    for item in dataset:
        prompt = item['vulnerable']
        vuln_type = item['vulnerability_type']

        for gen_idx in range(n_generations):
            # Generate
            completion = generate_completion(model, tokenizer, prompt, GENERATION_CONFIG)

            # Score
            score_result = score_completion(completion, vuln_type)

            # Detect refusal
            is_refusal, refusal_details = detect_refusal(completion)

            # Combine results
            result = {
                'id': item['id'],
                'base_id': item['base_id'],
                'vulnerability_type': vuln_type,
                'generation_idx': gen_idx,
                'completion': completion[:1000],  # Truncate for storage
                'strict_label': score_result.strict_label,
                'expanded_label': score_result.expanded_label,
                'is_refusal': is_refusal,
                'has_strict_secure': score_result.has_strict_secure,
                'has_strict_insecure': score_result.has_strict_insecure,
                'has_expanded_secure_addition': score_result.has_expanded_secure_addition,
                'has_bounds_check': score_result.has_bounds_check,
                'bounds_check_matches': score_result.bounds_check_matches,
                'refusal_details': refusal_details,
            }

            results.append(result)
            pbar.update(1)

    pbar.close()
    return results


def run_experiment(
    datasets_to_run: List[str] = ['base', 'expanded'],
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Run the full experiment.

    Args:
        datasets_to_run: List of datasets to run ('base', 'expanded', or both)
        dry_run: If True, skip actual generation (for testing)

    Returns:
        Dict with all results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    DATA_DIR.mkdir(exist_ok=True)

    results = {
        'timestamp': timestamp,
        'config': {
            'model': MODEL_NAME,
            'generation': GENERATION_CONFIG,
            'generations_per_prompt': GENERATIONS_PER_PROMPT,
        },
        'base': None,
        'expanded': None,
    }

    # Load model (unless dry run)
    model, tokenizer = None, None
    if not dry_run:
        model, tokenizer = load_model()

    # Run Base dataset
    if 'base' in datasets_to_run:
        print("\n" + "=" * 60)
        print("RUNNING BASE DATASET")
        print("=" * 60)

        base_dataset = load_base_dataset()

        if dry_run:
            # Create mock results for testing
            base_results = create_mock_results(base_dataset, GENERATIONS_PER_PROMPT['base'])
        else:
            base_results = run_generations(
                model, tokenizer, base_dataset,
                n_generations=GENERATIONS_PER_PROMPT['base'],
                dataset_name='base'
            )

        # Analyze
        base_overall = aggregate_overall(base_results)
        base_by_base_id = aggregate_by_base_id(base_results)
        base_by_vuln_type = aggregate_by_vuln_type(base_results)

        results['base'] = {
            'n_prompts': len(base_dataset),
            'n_generations': len(base_results),
            'overall': {
                'strict': results_to_serializable(base_overall['strict']),
                'expanded': results_to_serializable(base_overall['expanded']),
            },
            'by_base_id': {k: {
                'n': v['n'],
                'strict': results_to_serializable(v['strict']),
                'expanded': results_to_serializable(v['expanded']),
            } for k, v in base_by_base_id.items()},
            'by_vuln_type': {k: {
                'n': v['n'],
                'strict': results_to_serializable(v['strict']),
                'expanded': results_to_serializable(v['expanded']),
            } for k, v in base_by_vuln_type.items()},
            'raw_results': base_results,
        }

        print("\nBase Results (STRICT):")
        for k, v in base_overall['strict'].items():
            print(f"  {k}: {v}")

    # Run Expanded dataset
    if 'expanded' in datasets_to_run:
        print("\n" + "=" * 60)
        print("RUNNING EXPANDED DATASET")
        print("=" * 60)

        expanded_dataset = load_expanded_dataset()

        if dry_run:
            expanded_results = create_mock_results(expanded_dataset, GENERATIONS_PER_PROMPT['expanded'])
        else:
            expanded_results = run_generations(
                model, tokenizer, expanded_dataset,
                n_generations=GENERATIONS_PER_PROMPT['expanded'],
                dataset_name='expanded'
            )

        # Analyze
        exp_overall = aggregate_overall(expanded_results)
        exp_by_base_id = aggregate_by_base_id(expanded_results)
        exp_by_vuln_type = aggregate_by_vuln_type(expanded_results)

        results['expanded'] = {
            'n_prompts': len(expanded_dataset),
            'n_generations': len(expanded_results),
            'overall': {
                'strict': results_to_serializable(exp_overall['strict']),
                'expanded': results_to_serializable(exp_overall['expanded']),
            },
            'by_base_id': {k: {
                'n': v['n'],
                'strict': results_to_serializable(v['strict']),
                'expanded': results_to_serializable(v['expanded']),
            } for k, v in exp_by_base_id.items()},
            'by_vuln_type': {k: {
                'n': v['n'],
                'strict': results_to_serializable(v['strict']),
                'expanded': results_to_serializable(v['expanded']),
            } for k, v in exp_by_vuln_type.items()},
            'raw_results': expanded_results,
        }

        print("\nExpanded Results (STRICT):")
        for k, v in exp_overall['strict'].items():
            print(f"  {k}: {v}")

    # Print comparison if both were run
    if results['base'] and results['expanded']:
        print("\n")
        print(format_comparison_table(
            {
                'strict': {k: type('obj', (), v)() for k, v in results['base']['overall']['strict'].items()},
                'expanded': {k: type('obj', (), v)() for k, v in results['base']['overall']['expanded'].items()},
            },
            {
                'strict': {k: type('obj', (), v)() for k, v in results['expanded']['overall']['strict'].items()},
                'expanded': {k: type('obj', (), v)() for k, v in results['expanded']['overall']['expanded'].items()},
            }
        ))

    # Save results
    output_path = DATA_DIR / f"experiment1_results_{timestamp}.json"

    # Create a saveable version (without raw_results to save space, store separately)
    results_summary = {
        'timestamp': results['timestamp'],
        'config': results['config'],
        'base': {k: v for k, v in results['base'].items() if k != 'raw_results'} if results['base'] else None,
        'expanded': {k: v for k, v in results['expanded'].items() if k != 'raw_results'} if results['expanded'] else None,
    }

    with open(output_path, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Save raw results separately (larger file)
    raw_path = DATA_DIR / f"experiment1_raw_{timestamp}.json"
    raw_data = {
        'timestamp': timestamp,
        'base_raw': results['base']['raw_results'] if results['base'] else None,
        'expanded_raw': results['expanded']['raw_results'] if results['expanded'] else None,
    }
    with open(raw_path, 'w') as f:
        json.dump(raw_data, f, indent=2)

    print(f"Raw results saved to: {raw_path}")

    return results


def create_mock_results(dataset: List[Dict], n_gens: int) -> List[Dict]:
    """Create mock results for dry-run testing."""
    import random
    random.seed(42)

    results = []
    for item in dataset:
        for gen_idx in range(n_gens):
            label = random.choice(['secure', 'insecure', 'other'])
            results.append({
                'id': item['id'],
                'base_id': item['base_id'],
                'vulnerability_type': item['vulnerability_type'],
                'generation_idx': gen_idx,
                'completion': '[MOCK]',
                'strict_label': label,
                'expanded_label': label,
                'is_refusal': random.random() < 0.05,
                'has_strict_secure': label == 'secure',
                'has_strict_insecure': label == 'insecure',
                'has_expanded_secure_addition': False,
                'has_bounds_check': False,
                'bounds_check_matches': [],
                'refusal_details': {},
            })
    return results


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Baseline Behavior")
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['base', 'expanded', 'both'],
        default='both',
        help="Which dataset(s) to run"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Test without GPU (uses mock data)"
    )
    args = parser.parse_args()

    if args.dataset == 'both':
        datasets = ['base', 'expanded']
    else:
        datasets = [args.dataset]

    run_experiment(datasets_to_run=datasets, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
