#!/usr/bin/env python3
"""
Experiment 4A Step 2: Collect Mistral-7B Activations

All 32 layers × 210 prompts × 4096 dim.
Output: activations_TIMESTAMP.npz + metadata_TIMESTAMP.json
"""

import sys
import json
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

from experiment_config import MODEL_NAME, QUANTIZATION, DATASET_PATH, DATA_DIR
from model_loader import ModelLoader
from activation_collector import ActivationCollector


def load_dataset():
    dataset = []
    with open(DATASET_PATH) as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


def main():
    torch.manual_seed(42)

    print("=" * 60)
    print("EXPERIMENT 4A STEP 2: Collect Mistral-7B Activations")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    dataset = load_dataset()
    print(f"Dataset: {len(dataset)} pairs ({len(dataset)*2} prompts)")

    loader = ModelLoader(MODEL_NAME, quantization=QUANTIZATION)
    collector = ActivationCollector(loader)

    npz_path, metadata_path = collector.collect_dataset(dataset, DATA_DIR)

    print(f"\nActivations saved: {npz_path}")
    print(f"Metadata saved: {metadata_path}")

    return str(npz_path), str(metadata_path)


if __name__ == "__main__":
    main()
