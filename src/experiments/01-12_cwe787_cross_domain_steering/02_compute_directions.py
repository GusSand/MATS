#!/usr/bin/env python3
"""
Step 2: Compute steering directions from activations.

Direction = mean(secure_activations) - mean(vulnerable_activations)

This gives us a vector pointing from "vulnerable code mindset" to "secure code mindset".
"""

import numpy as np
from pathlib import Path
from datetime import datetime
import json
import argparse


def load_activations(act_path: Path) -> tuple:
    """Load activations and return as dict."""
    data = np.load(act_path)

    # Find number of layers
    n_layers = sum(1 for k in data.keys() if k.startswith('X_layer_'))

    activations = {}
    labels = {}

    for layer in range(n_layers):
        activations[layer] = data[f'X_layer_{layer}']
        labels[layer] = data[f'y_layer_{layer}']

    return activations, labels, n_layers


def compute_directions(activations: dict, labels: dict, n_layers: int) -> dict:
    """Compute mean-difference directions at each layer."""
    directions = {}

    print(f"\nComputing directions for {n_layers} layers...")

    for layer in range(n_layers):
        X = activations[layer]
        y = labels[layer]

        # Split by label
        X_secure = X[y == 1]  # secure prompts
        X_vulnerable = X[y == 0]  # vulnerable prompts

        # Compute means
        mean_secure = np.mean(X_secure, axis=0)
        mean_vulnerable = np.mean(X_vulnerable, axis=0)

        # Direction: secure - vulnerable (steering toward secure)
        direction = mean_secure - mean_vulnerable
        direction_norm = np.linalg.norm(direction)
        direction_normalized = direction / direction_norm if direction_norm > 0 else direction

        directions[layer] = {
            'raw': direction,
            'normalized': direction_normalized,
            'norm': direction_norm,
            'n_secure': len(X_secure),
            'n_vulnerable': len(X_vulnerable)
        }

        if layer == 31:  # Print L31 stats
            print(f"\nL31 Direction Stats:")
            print(f"  Secure samples: {len(X_secure)}")
            print(f"  Vulnerable samples: {len(X_vulnerable)}")
            print(f"  Direction norm: {direction_norm:.4f}")

    return directions


def save_directions(directions: dict, output_dir: Path, timestamp: str) -> Path:
    """Save directions to NPZ file."""
    save_dict = {}

    for layer, d in directions.items():
        save_dict[f'direction_layer_{layer}'] = d['raw']
        save_dict[f'direction_normalized_layer_{layer}'] = d['normalized']

    output_path = output_dir / f"directions_{timestamp}.npz"
    np.savez_compressed(output_path, **save_dict)

    # Also save summary
    summary = {
        'timestamp': timestamp,
        'n_layers': len(directions),
        'layer_stats': {
            layer: {
                'norm': float(d['norm']),
                'n_secure': d['n_secure'],
                'n_vulnerable': d['n_vulnerable']
            }
            for layer, d in directions.items()
        }
    }

    summary_path = output_dir / f"directions_summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDirections saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Compute steering directions")
    parser.add_argument("--activations", type=str, required=True,
                        help="Path to activations NPZ file")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"

    act_path = Path(args.activations)
    if not act_path.is_absolute():
        act_path = data_dir / args.activations

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load activations
    print(f"Loading activations from: {act_path}")
    activations, labels, n_layers = load_activations(act_path)

    # Compute directions
    directions = compute_directions(activations, labels, n_layers)

    # Save
    dir_path = save_directions(directions, data_dir, timestamp)

    # Print summary
    print("\n" + "="*50)
    print("DIRECTION COMPUTATION COMPLETE")
    print("="*50)
    print(f"Layers: {n_layers}")
    print(f"Output: {dir_path}")

    # Show norm distribution
    norms = [directions[l]['norm'] for l in range(n_layers)]
    print(f"\nDirection norms:")
    print(f"  Min: {min(norms):.4f} (L{norms.index(min(norms))})")
    print(f"  Max: {max(norms):.4f} (L{norms.index(max(norms))})")
    print(f"  L31: {directions[31]['norm']:.4f}")

    return str(dir_path)


if __name__ == "__main__":
    main()
