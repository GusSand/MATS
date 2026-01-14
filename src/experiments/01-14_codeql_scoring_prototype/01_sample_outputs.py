#!/usr/bin/env python3
"""
Sample outputs from LOBO experiment for CodeQL testing.

Samples 10 outputs from each category (secure/insecure/other) at α=3.5.
"""

import json
from pathlib import Path
from collections import defaultdict
import random

from experiment_config import (
    LOBO_FOLD_RESULTS, DATA_DIR,
    SAMPLES_PER_CATEGORY, ALPHA_TO_SAMPLE,
)


def load_all_fold_results():
    """Load all fold results from the 512-token run."""
    all_results = []

    for fold_file in LOBO_FOLD_RESULTS.glob("*_20260113_171820.json"):
        with open(fold_file) as f:
            fold_data = json.load(f)

        fold_id = fold_data['fold_id']
        alpha_results = fold_data['alpha_results'].get(ALPHA_TO_SAMPLE, [])

        for item in alpha_results:
            item['fold_id'] = fold_id
            all_results.append(item)

    return all_results


def categorize_by_label(results):
    """Group results by strict_label."""
    categories = defaultdict(list)
    for item in results:
        label = item['strict_label']
        categories[label].append(item)
    return categories


def sample_from_categories(categories, n_per_category):
    """Sample n items from each category."""
    samples = {}

    for label, items in categories.items():
        if len(items) <= n_per_category:
            samples[label] = items
        else:
            samples[label] = random.sample(items, n_per_category)

    return samples


def main():
    random.seed(42)  # Reproducibility

    print("="*60)
    print("Sampling LOBO Outputs for CodeQL Prototype")
    print("="*60)

    # Load all results
    results = load_all_fold_results()
    print(f"\nLoaded {len(results)} outputs at α={ALPHA_TO_SAMPLE}")

    # Categorize
    categories = categorize_by_label(results)
    print("\nCategory breakdown:")
    for label, items in sorted(categories.items()):
        print(f"  {label}: {len(items)}")

    # Sample
    samples = sample_from_categories(categories, SAMPLES_PER_CATEGORY)
    print(f"\nSampled {SAMPLES_PER_CATEGORY} per category:")
    for label, items in sorted(samples.items()):
        print(f"  {label}: {len(items)}")

    # Flatten and save
    all_samples = []
    for label, items in samples.items():
        for i, item in enumerate(items):
            sample = {
                'sample_id': f"{label}_{i:02d}",
                'regex_label': label,
                'fold_id': item['fold_id'],
                'base_id': item['base_id'],
                'vulnerability_type': item['vulnerability_type'],
                'output': item['output'],
            }
            all_samples.append(sample)

    # Save samples
    output_path = DATA_DIR / "sampled_outputs.json"
    with open(output_path, 'w') as f:
        json.dump(all_samples, f, indent=2)

    print(f"\nSaved {len(all_samples)} samples to: {output_path}")

    # Print a few examples
    print("\n" + "="*60)
    print("Sample Examples:")
    print("="*60)

    for label in ['secure', 'insecure', 'other']:
        sample = next((s for s in all_samples if s['regex_label'] == label), None)
        if sample:
            print(f"\n--- {label.upper()} example ({sample['sample_id']}) ---")
            print(sample['output'][:500] + "..." if len(sample['output']) > 500 else sample['output'])

    return output_path


if __name__ == "__main__":
    main()
