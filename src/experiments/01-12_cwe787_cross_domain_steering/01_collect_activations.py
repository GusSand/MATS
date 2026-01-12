#!/usr/bin/env python3
"""
Step 1: Collect activations from expanded CWE-787 dataset.

Collects last-token activations at all 32 layers for:
- 105 vulnerable prompts (label=0)
- 105 secure prompts (label=1)

Output: activations_TIMESTAMP.npz with X_layer_N and y arrays
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "01-08_llama8b_sr_scg_separation"))

import json
import numpy as np
from datetime import datetime
from tqdm import tqdm
import argparse

from utils.activation_collector import ActivationCollector


def load_expanded_dataset(data_path: Path) -> list:
    """Load the expanded CWE-787 dataset."""
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def collect_activations(collector: ActivationCollector, dataset: list) -> dict:
    """Collect activations for all prompts in dataset."""
    n_layers = collector.n_layers

    # Initialize storage
    data = {
        'vulnerable': {layer: [] for layer in range(n_layers)},
        'secure': {layer: [] for layer in range(n_layers)}
    }
    metadata = {
        'vulnerable': [],
        'secure': []
    }

    print(f"\nCollecting activations for {len(dataset)} pairs ({len(dataset)*2} prompts)...")

    for item in tqdm(dataset, desc="Processing pairs"):
        # Vulnerable prompt (label=0)
        vuln_acts = collector.get_activations(item['vulnerable'])
        for layer in range(n_layers):
            data['vulnerable'][layer].append(vuln_acts[layer].squeeze())
        metadata['vulnerable'].append({
            'id': item['id'],
            'base_id': item['base_id'],
            'vulnerability_type': item['vulnerability_type'],
            'detection': item['detection']
        })

        # Secure prompt (label=1)
        sec_acts = collector.get_activations(item['secure'])
        for layer in range(n_layers):
            data['secure'][layer].append(sec_acts[layer].squeeze())
        metadata['secure'].append({
            'id': item['id'],
            'base_id': item['base_id'],
            'vulnerability_type': item['vulnerability_type'],
            'detection': item['detection']
        })

    return data, metadata


def save_activations(data: dict, metadata: dict, output_dir: Path, timestamp: str):
    """Save activations to NPZ file."""
    n_layers = len(data['vulnerable'])

    # Prepare arrays for saving
    save_dict = {}

    for layer in range(n_layers):
        # Combine vulnerable (label=0) and secure (label=1) activations
        X_vuln = np.array(data['vulnerable'][layer])
        X_sec = np.array(data['secure'][layer])
        X = np.vstack([X_vuln, X_sec])
        y = np.array([0] * len(X_vuln) + [1] * len(X_sec))

        save_dict[f'X_layer_{layer}'] = X
        save_dict[f'y_layer_{layer}'] = y

    # Save activations
    output_path = output_dir / f"activations_{timestamp}.npz"
    np.savez_compressed(output_path, **save_dict)
    print(f"\nActivations saved to: {output_path}")

    # Save metadata
    meta_path = output_dir / f"metadata_{timestamp}.json"
    with open(meta_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'n_pairs': len(metadata['vulnerable']),
            'n_prompts': len(metadata['vulnerable']) * 2,
            'n_layers': n_layers,
            'vulnerable_metadata': metadata['vulnerable'],
            'secure_metadata': metadata['secure']
        }, f, indent=2)
    print(f"Metadata saved to: {meta_path}")

    return output_path, meta_path


def main():
    parser = argparse.ArgumentParser(description="Collect activations from expanded dataset")
    parser.add_argument("--dataset", type=str,
                        default="../01-12_cwe787_dataset_expansion/data/cwe787_expanded_20260112_143316.jsonl",
                        help="Path to expanded dataset")
    parser.add_argument("--model", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="Model name")
    args = parser.parse_args()

    # Setup paths
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    data_dir.mkdir(exist_ok=True)

    dataset_path = script_dir / args.dataset
    if not dataset_path.exists():
        # Try absolute path
        dataset_path = Path(args.dataset)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_expanded_dataset(dataset_path)
    print(f"Loaded {len(dataset)} pairs")

    # Initialize collector
    collector = ActivationCollector(args.model)

    # Collect activations
    data, metadata = collect_activations(collector, dataset)

    # Save
    act_path, meta_path = save_activations(data, metadata, data_dir, timestamp)

    # Summary
    print("\n" + "="*50)
    print("ACTIVATION COLLECTION COMPLETE")
    print("="*50)
    print(f"Pairs processed: {len(dataset)}")
    print(f"Total prompts: {len(dataset) * 2}")
    print(f"Layers: {collector.n_layers}")
    print(f"Hidden dim: {collector.hidden_size}")
    print(f"Output: {act_path}")

    return str(act_path)


if __name__ == "__main__":
    main()
