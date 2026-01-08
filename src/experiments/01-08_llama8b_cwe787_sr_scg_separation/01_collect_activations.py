#!/usr/bin/env python3
"""
Collect SR (Security Recognition) and SCG (Secure Code Generation) activations
using validated CWE-787 prompt pairs.

This experiment tests whether SR and SCG are separately encoded using
prompt pairs that have 100% separation in validation testing.

Labeling Strategy:
- SR label: Based on prompt type (secure prompt = 1, vulnerable prompt = 0)
- SCG label: Based on actual output (snprintf/strncat = 1, sprintf/strcat = 0)

Data source: 7 validated CWE-787 pairs from 01-08_llama8b_cwe787_prompt_pairs
"""

import sys
from pathlib import Path

# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import json
import re
from datetime import datetime
import argparse
from tqdm import tqdm

from validated_pairs import get_all_pairs, get_prompts
from utils.activation_collector import ActivationCollector


def collect_sr_data(collector: ActivationCollector, n_samples_per_prompt: int = 50) -> dict:
    """
    Collect Security Recognition (SR) dataset.

    SR Label: 1 = secure prompt, 0 = vulnerable prompt
    This measures whether the model recognizes the prompt is security-relevant.
    """
    pairs = get_all_pairs()
    n_layers = collector.n_layers

    data = {layer: {'X': [], 'y': []} for layer in range(n_layers)}
    pair_indices = []  # Track which pair each sample belongs to
    metadata = []

    print(f"\nCollecting SR data: {len(pairs)} pairs × 2 prompts × {n_samples_per_prompt} samples")
    print("=" * 60)

    total_prompts = len(pairs) * 2

    with tqdm(total=total_prompts * n_samples_per_prompt, desc="SR Collection") as pbar:
        for pair_idx, pair in enumerate(pairs):
            pair_id = pair['id']

            # Vulnerable prompts (label = 0)
            for _ in range(n_samples_per_prompt):
                acts = collector.get_activations(pair['vulnerable'])
                for layer in range(n_layers):
                    data[layer]['X'].append(acts[layer].squeeze())
                    data[layer]['y'].append(0)
                pair_indices.append(pair_idx)
                metadata.append({
                    'pair_id': pair_id,
                    'pair_idx': pair_idx,
                    'prompt_type': 'vulnerable',
                    'sr_label': 0
                })
                pbar.update(1)

            # Secure prompts (label = 1)
            for _ in range(n_samples_per_prompt):
                acts = collector.get_activations(pair['secure'])
                for layer in range(n_layers):
                    data[layer]['X'].append(acts[layer].squeeze())
                    data[layer]['y'].append(1)
                pair_indices.append(pair_idx)
                metadata.append({
                    'pair_id': pair_id,
                    'pair_idx': pair_idx,
                    'prompt_type': 'secure',
                    'sr_label': 1
                })
                pbar.update(1)

    # Convert to numpy arrays
    for layer in range(n_layers):
        data[layer]['X'] = np.array(data[layer]['X'])
        data[layer]['y'] = np.array(data[layer]['y'])

    # Add pair indices to data for proper cross-validation
    data['pair_indices'] = np.array(pair_indices)

    n_secure = sum(1 for m in metadata if m['sr_label'] == 1)
    n_vuln = sum(1 for m in metadata if m['sr_label'] == 0)
    print(f"\nSR data collected: {n_vuln} vulnerable + {n_secure} secure = {len(metadata)} total")
    print(f"  Unique pairs: {len(pairs)}")

    return data, metadata


def collect_scg_data(collector: ActivationCollector, n_samples_per_prompt: int = 50,
                     temperature: float = 0.7) -> dict:
    """
    Collect Secure Code Generation (SCG) dataset.

    SCG Label: 1 = model outputs secure code, 0 = model outputs insecure code
    This measures whether the model will generate secure code.

    We generate from BOTH vulnerable and secure prompts and label based on OUTPUT.
    """
    pairs = get_all_pairs()
    n_layers = collector.n_layers

    data = {layer: {'X': [], 'y': []} for layer in range(n_layers)}
    pair_indices = []  # Track which pair each sample belongs to
    metadata = []

    stats = {
        'secure_outputs': 0,
        'insecure_outputs': 0,
        'neither_outputs': 0,
        'by_pair': {}
    }

    print(f"\nCollecting SCG data: {len(pairs)} pairs × 2 prompts × {n_samples_per_prompt} samples")
    print("(Generating outputs and classifying based on actual code)")
    print("=" * 60)

    total_prompts = len(pairs) * 2

    with tqdm(total=total_prompts * n_samples_per_prompt, desc="SCG Collection") as pbar:
        for pair_idx, pair in enumerate(pairs):
            pair_id = pair['id']
            # Map detection pattern keys to match activation_collector format
            raw_detection = pair['detection']
            detection = {
                'secure': raw_detection['secure_pattern'],
                'insecure': raw_detection['insecure_pattern']
            }

            pair_stats = {'secure': 0, 'insecure': 0, 'neither': 0}

            # Collect from both vulnerable and secure prompts
            for prompt_type in ['vulnerable', 'secure']:
                prompt = pair[prompt_type]

                for _ in range(n_samples_per_prompt):
                    # Get activations BEFORE generation
                    acts = collector.get_activations(prompt)

                    # Generate output and classify
                    result = collector.generate_and_classify(
                        prompt, detection, temperature=temperature
                    )

                    output_label = result['label']

                    if output_label == 'secure':
                        scg_label = 1
                        stats['secure_outputs'] += 1
                        pair_stats['secure'] += 1
                    elif output_label == 'insecure':
                        scg_label = 0
                        stats['insecure_outputs'] += 1
                        pair_stats['insecure'] += 1
                    else:
                        # Skip 'neither' outputs
                        stats['neither_outputs'] += 1
                        pair_stats['neither'] += 1
                        pbar.update(1)
                        continue

                    for layer in range(n_layers):
                        data[layer]['X'].append(acts[layer].squeeze())
                        data[layer]['y'].append(scg_label)

                    pair_indices.append(pair_idx)
                    metadata.append({
                        'pair_id': pair_id,
                        'pair_idx': pair_idx,
                        'prompt_type': prompt_type,
                        'scg_label': scg_label,
                        'output_snippet': result['output'][:100]
                    })
                    pbar.update(1)

            stats['by_pair'][pair_id] = pair_stats

    # Convert to numpy arrays
    for layer in range(n_layers):
        if data[layer]['X']:
            data[layer]['X'] = np.array(data[layer]['X'])
            data[layer]['y'] = np.array(data[layer]['y'])
        else:
            data[layer]['X'] = np.array([]).reshape(0, collector.hidden_size)
            data[layer]['y'] = np.array([])

    # Add pair indices to data for proper cross-validation
    data['pair_indices'] = np.array(pair_indices)

    print(f"\nSCG data collected:")
    print(f"  Secure outputs: {stats['secure_outputs']}")
    print(f"  Insecure outputs: {stats['insecure_outputs']}")
    print(f"  Neither (skipped): {stats['neither_outputs']}")
    print(f"  Total usable: {stats['secure_outputs'] + stats['insecure_outputs']}")
    print(f"  Unique pairs: {len(pairs)}")

    return data, metadata, stats


def save_data(sr_data: dict, sr_metadata: list, scg_data: dict, scg_metadata: list,
              scg_stats: dict, output_dir: Path, timestamp: str, n_layers: int):
    """Save collected data to disk."""

    # Save SR data (including pair_indices for proper cross-validation)
    sr_file = output_dir / f"sr_data_{timestamp}.npz"
    sr_arrays = {f"X_layer_{k}": v['X'] for k, v in sr_data.items() if isinstance(k, int)}
    sr_arrays.update({f"y_layer_{k}": v['y'] for k, v in sr_data.items() if isinstance(k, int)})
    sr_arrays['pair_indices'] = sr_data['pair_indices']
    np.savez_compressed(sr_file, **sr_arrays)

    # Save SCG data (including pair_indices for proper cross-validation)
    scg_file = output_dir / f"scg_data_{timestamp}.npz"
    scg_arrays = {f"X_layer_{k}": v['X'] for k, v in scg_data.items() if isinstance(k, int)}
    scg_arrays.update({f"y_layer_{k}": v['y'] for k, v in scg_data.items() if isinstance(k, int)})
    scg_arrays['pair_indices'] = scg_data['pair_indices']
    np.savez_compressed(scg_file, **scg_arrays)

    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'experiment': 'CWE-787 SR vs SCG Separation',
        'data_source': '7 validated CWE-787 prompt pairs',
        'sr_samples': len(sr_metadata),
        'scg_samples': len(scg_metadata),
        'scg_stats': scg_stats,
        'n_layers': n_layers,
        'sr_labeling': 'secure_prompt=1, vulnerable_prompt=0',
        'scg_labeling': 'secure_output=1, insecure_output=0'
    }

    with open(output_dir / f"metadata_{timestamp}.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nData saved to: {output_dir}")
    print(f"  - SR data: sr_data_{timestamp}.npz")
    print(f"  - SCG data: scg_data_{timestamp}.npz")
    print(f"  - Metadata: metadata_{timestamp}.json")


def main():
    parser = argparse.ArgumentParser(description="Collect SR and SCG activations from CWE-787 pairs")
    parser.add_argument("--n-samples", type=int, default=50,
                        help="Samples per prompt (default: 50)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for SCG generation (default: 0.7)")
    args = parser.parse_args()

    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("SR vs SCG SEPARATION: CWE-787 PROMPT PAIRS")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Samples per prompt: {args.n_samples}")
    print(f"  Prompt pairs: 7 validated CWE-787")
    print(f"  Total prompts: 14")
    print(f"  Expected SR samples: {args.n_samples * 14}")
    print(f"  Expected SCG samples: ~{args.n_samples * 14} (minus 'neither' outputs)")

    # Initialize collector
    print("\nInitializing model...")
    collector = ActivationCollector()
    n_layers = collector.n_layers

    # Collect SR data
    print("\n" + "-" * 60)
    print("PHASE 1: COLLECTING SR DATA")
    print("-" * 60)
    sr_data, sr_metadata = collect_sr_data(collector, args.n_samples)

    # Collect SCG data
    print("\n" + "-" * 60)
    print("PHASE 2: COLLECTING SCG DATA")
    print("-" * 60)
    scg_data, scg_metadata, scg_stats = collect_scg_data(
        collector, args.n_samples, args.temperature
    )

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_data(sr_data, sr_metadata, scg_data, scg_metadata, scg_stats,
              output_dir, timestamp, n_layers)

    # Summary
    print("\n" + "=" * 70)
    print("COLLECTION COMPLETE")
    print("=" * 70)
    print(f"\nSR samples: {len(sr_metadata)}")
    print(f"SCG samples: {len(scg_metadata)}")
    print(f"\nNext step: python 02_train_probes.py")

    return sr_data, scg_data


if __name__ == "__main__":
    sr_data, scg_data = main()
