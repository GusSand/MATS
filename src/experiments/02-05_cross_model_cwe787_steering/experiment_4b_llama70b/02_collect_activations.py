#!/usr/bin/env python3
"""
Experiment 4B Step 2: Collect Llama-70B Activations

All 80 layers x 210 prompts x 8192 dim.
Memory: activations moved to CPU immediately via hooks.
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
    print("EXPERIMENT 4B STEP 2: Collect Llama-70B Activations")
    print(f"Model: {MODEL_NAME}")
    print(f"Quantization: {QUANTIZATION}")
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
