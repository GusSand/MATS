#!/usr/bin/env python3
"""
Collect SR (Security Recognition) and SCG (Secure Code Generation) activations.

This collects TWO separate datasets:
1. SR dataset: Label = 1 if context has security warning, 0 otherwise
   - Measures: "Does the model recognize security-relevant context?"

2. SCG dataset: Label = 1 if model outputs secure function, 0 if insecure
   - Measures: "Will the model generate secure code?"

The key hypothesis is that these are DIFFERENT directions (like harmfulness vs refusal).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import json
from datetime import datetime
import argparse

from config.security_pairs import SECURITY_PAIRS, CORE_PAIRS, ALL_PAIRS
from utils.activation_collector import ActivationCollector


def collect_for_pair(collector: ActivationCollector, pair_name: str,
                     n_sr_samples: int = 15, n_scg_samples: int = 20,
                     verbose: bool = True) -> dict:
    """Collect SR and SCG data for a single security pair."""
    config = SECURITY_PAIRS[pair_name]

    if verbose:
        print(f"\n{'='*60}")
        print(f"PAIR: {pair_name}")
        print(f"  {config['insecure']} -> {config['secure']}")
        print(f"={'='*60}")

    # Collect SR data
    if verbose:
        print(f"\nCollecting SR data (n={n_sr_samples} per template)...")
    sr_data, sr_metadata = collector.collect_sr_dataset(config, n_sr_samples)

    n_secure_ctx = sum(1 for m in sr_metadata if m['context'] == 'secure')
    n_neutral_ctx = sum(1 for m in sr_metadata if m['context'] == 'neutral')
    if verbose:
        print(f"  SR samples: {n_secure_ctx} secure + {n_neutral_ctx} neutral = {len(sr_metadata)}")

    # Collect SCG data
    if verbose:
        print(f"\nCollecting SCG data (n={n_scg_samples} per template)...")
    scg_data, scg_metadata, scg_stats = collector.collect_scg_dataset(config, n_scg_samples)

    if verbose:
        print(f"  SCG samples: {scg_stats['secure']} secure + {scg_stats['insecure']} insecure")
        print(f"  Skipped: {scg_stats['neither']} (neither function detected)")

    return {
        'pair_name': pair_name,
        'config': {
            'insecure': config['insecure'],
            'secure': config['secure'],
            'vulnerability_class': config['vulnerability_class']
        },
        'sr_data': sr_data,
        'sr_metadata': sr_metadata,
        'scg_data': scg_data,
        'scg_metadata': scg_metadata,
        'scg_stats': scg_stats
    }


def collect_all_pairs(collector: ActivationCollector, pair_names: list,
                      n_sr_samples: int = 15, n_scg_samples: int = 20) -> dict:
    """Collect data for multiple security pairs."""
    all_data = {}

    for i, pair_name in enumerate(pair_names):
        print(f"\n[{i+1}/{len(pair_names)}] Processing {pair_name}...")
        pair_data = collect_for_pair(
            collector, pair_name, n_sr_samples, n_scg_samples
        )
        all_data[pair_name] = pair_data

    return all_data


def merge_data_across_pairs(all_data: dict, n_layers: int) -> dict:
    """
    Merge SR and SCG data across all pairs into combined datasets.

    This allows training probes that generalize across vulnerability types.
    """
    # Initialize merged data structures
    sr_merged = {layer: {'X': [], 'y': []} for layer in range(n_layers)}
    scg_merged = {layer: {'X': [], 'y': []} for layer in range(n_layers)}

    sr_pair_labels = []  # Track which pair each sample came from
    scg_pair_labels = []

    for pair_name, pair_data in all_data.items():
        sr_data = pair_data['sr_data']
        scg_data = pair_data['scg_data']

        # Merge SR data
        for layer in range(n_layers):
            if len(sr_data[layer]['X']) > 0:
                sr_merged[layer]['X'].extend(sr_data[layer]['X'])
                sr_merged[layer]['y'].extend(sr_data[layer]['y'])

        sr_pair_labels.extend([pair_name] * len(pair_data['sr_metadata']))

        # Merge SCG data
        for layer in range(n_layers):
            if len(scg_data[layer]['X']) > 0:
                scg_merged[layer]['X'].extend(scg_data[layer]['X'])
                scg_merged[layer]['y'].extend(scg_data[layer]['y'])

        scg_pair_labels.extend([pair_name] * len(pair_data['scg_metadata']))

    # Convert to numpy arrays
    for layer in range(n_layers):
        sr_merged[layer]['X'] = np.array(sr_merged[layer]['X'])
        sr_merged[layer]['y'] = np.array(sr_merged[layer]['y'])
        scg_merged[layer]['X'] = np.array(scg_merged[layer]['X']) if sr_merged[layer]['X'].size > 0 else np.array([])
        scg_merged[layer]['y'] = np.array(scg_merged[layer]['y']) if sr_merged[layer]['y'].size > 0 else np.array([])

    return {
        'sr_merged': sr_merged,
        'scg_merged': scg_merged,
        'sr_pair_labels': sr_pair_labels,
        'scg_pair_labels': scg_pair_labels
    }


def save_data(all_data: dict, merged_data: dict, output_dir: Path, timestamp: str):
    """Save collected data to disk."""
    # Save per-pair data as NPZ files
    for pair_name, pair_data in all_data.items():
        # SR data
        sr_file = output_dir / f"sr_{pair_name}_{timestamp}.npz"
        np.savez_compressed(
            sr_file,
            **{f"X_layer_{k}": v['X'] for k, v in pair_data['sr_data'].items()},
            **{f"y_layer_{k}": v['y'] for k, v in pair_data['sr_data'].items()}
        )

        # SCG data
        scg_file = output_dir / f"scg_{pair_name}_{timestamp}.npz"
        np.savez_compressed(
            scg_file,
            **{f"X_layer_{k}": v['X'] for k, v in pair_data['scg_data'].items()},
            **{f"y_layer_{k}": v['y'] for k, v in pair_data['scg_data'].items()}
        )

    # Save merged data
    sr_merged_file = output_dir / f"sr_merged_{timestamp}.npz"
    np.savez_compressed(
        sr_merged_file,
        **{f"X_layer_{k}": v['X'] for k, v in merged_data['sr_merged'].items()},
        **{f"y_layer_{k}": v['y'] for k, v in merged_data['sr_merged'].items()}
    )

    scg_merged_file = output_dir / f"scg_merged_{timestamp}.npz"
    np.savez_compressed(
        scg_merged_file,
        **{f"X_layer_{k}": v['X'] for k, v in merged_data['scg_merged'].items()},
        **{f"y_layer_{k}": v['y'] for k, v in merged_data['scg_merged'].items()}
    )

    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'pairs': list(all_data.keys()),
        'n_pairs': len(all_data),
        'per_pair_stats': {},
        'merged_stats': {
            'sr_total_samples': len(merged_data['sr_pair_labels']),
            'scg_total_samples': len(merged_data['scg_pair_labels']),
        }
    }

    for pair_name, pair_data in all_data.items():
        metadata['per_pair_stats'][pair_name] = {
            'sr_samples': len(pair_data['sr_metadata']),
            'scg_stats': pair_data['scg_stats']
        }

    with open(output_dir / f"metadata_{timestamp}.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nData saved to: {output_dir}")
    print(f"  - Per-pair files: sr_<pair>_{timestamp}.npz, scg_<pair>_{timestamp}.npz")
    print(f"  - Merged files: sr_merged_{timestamp}.npz, scg_merged_{timestamp}.npz")
    print(f"  - Metadata: metadata_{timestamp}.json")


def main():
    parser = argparse.ArgumentParser(description="Collect SR and SCG activations")
    parser.add_argument("--pairs", choices=["core", "all"], default="core",
                        help="Which pairs to collect: core (5 pairs) or all (14 pairs)")
    parser.add_argument("--n-sr", type=int, default=15,
                        help="Samples per template for SR dataset")
    parser.add_argument("--n-scg", type=int, default=20,
                        help="Samples per template for SCG dataset")
    args = parser.parse_args()

    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    # Select pairs
    if args.pairs == "core":
        pair_names = CORE_PAIRS
        print(f"Using CORE pairs: {pair_names}")
    else:
        pair_names = ALL_PAIRS
        print(f"Using ALL pairs: {len(pair_names)} pairs")

    # Initialize collector
    collector = ActivationCollector()
    n_layers = collector.n_layers

    # Collect data
    print(f"\nCollecting data for {len(pair_names)} pairs...")
    print(f"  SR samples per template: {args.n_sr}")
    print(f"  SCG samples per template: {args.n_scg}")

    all_data = collect_all_pairs(collector, pair_names, args.n_sr, args.n_scg)

    # Merge across pairs
    print("\nMerging data across pairs...")
    merged_data = merge_data_across_pairs(all_data, n_layers)

    print(f"\nMerged dataset sizes:")
    print(f"  SR: {len(merged_data['sr_pair_labels'])} samples")
    print(f"  SCG: {len(merged_data['scg_pair_labels'])} samples")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_data(all_data, merged_data, output_dir, timestamp)

    # Summary
    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)

    total_sr = sum(len(d['sr_metadata']) for d in all_data.values())
    total_scg = sum(d['scg_stats']['total_usable'] for d in all_data.values())

    print(f"\nTotal samples collected:")
    print(f"  SR (Security Recognition): {total_sr}")
    print(f"  SCG (Secure Code Generation): {total_scg}")

    return all_data, merged_data


if __name__ == "__main__":
    all_data, merged_data = main()
