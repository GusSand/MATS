#!/usr/bin/env python3
"""
Experiment 4A Step 1: Mistral-7B Baseline Behavior

Generate 1 completion per 105 vulnerable prompts (no steering).
Score with STRICT + EXPANDED + refusal detection.
Decision gate: If baseline secure rate > 50%, report as finding.
"""

import sys
import json
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Import local config FIRST (before modifying sys.path)
from experiment_config import (
    MODEL_NAME, QUANTIZATION, DATASET_PATH, DATA_DIR, GENERATION_CONFIG,
    STRICT_PATTERNS, EXPANDED_SECURE_ADDITIONS, BOUNDS_CHECK_PATTERNS,
    C_CODE_INDICATORS, REFUSAL_PATTERNS,
)

# Now add shared and baseline paths
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
from model_loader import ModelLoader
from steering_generator import SteeringGenerator

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "01-12_llama8b_cwe787_baseline_behavior"))
from scoring import score_completion
from refusal_detection import detect_refusal


def load_dataset():
    dataset = []
    with open(DATASET_PATH) as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


def main():
    torch.manual_seed(42)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("EXPERIMENT 4A STEP 1: Mistral-7B Baseline Behavior")
    print(f"Model: {MODEL_NAME}")
    print(f"Timestamp: {timestamp}")
    print("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = load_dataset()
    print(f"Dataset: {len(dataset)} pairs")

    # Load model
    loader = ModelLoader(MODEL_NAME, quantization=QUANTIZATION)
    generator = SteeringGenerator(loader)

    # Generate baselines for vulnerable prompts
    results = []
    for pair in tqdm(dataset, desc="Baseline generation"):
        prompt = pair['vulnerable']
        vuln_type = pair['vulnerability_type']

        output = generator.generate_baseline(
            prompt=prompt,
            temperature=GENERATION_CONFIG['temperature'],
            top_p=GENERATION_CONFIG['top_p'],
            max_tokens=GENERATION_CONFIG['max_new_tokens'],
        )

        score_result = score_completion(output, vuln_type)
        is_refusal, refusal_details = detect_refusal(output)

        results.append({
            'id': pair['id'],
            'base_id': pair['base_id'],
            'vulnerability_type': vuln_type,
            'output': output[:500],
            'strict_label': score_result.strict_label,
            'expanded_label': score_result.expanded_label,
            'is_refusal': is_refusal,
        })

    # Compute summary
    n = len(results)
    strict_secure = sum(1 for r in results if r['strict_label'] == 'secure')
    strict_insecure = sum(1 for r in results if r['strict_label'] == 'insecure')
    strict_other = sum(1 for r in results if r['strict_label'] == 'other')
    expanded_secure = sum(1 for r in results if r['expanded_label'] == 'secure')
    refusals = sum(1 for r in results if r['is_refusal'])

    summary = {
        'n': n,
        'strict': {
            'secure': strict_secure,
            'insecure': strict_insecure,
            'other': strict_other,
            'secure_rate': strict_secure / n,
            'insecure_rate': strict_insecure / n,
        },
        'expanded': {
            'secure': expanded_secure,
            'secure_rate': expanded_secure / n,
        },
        'refusal_rate': refusals / n,
    }

    output_data = {
        'timestamp': timestamp,
        'model': MODEL_NAME,
        'quantization': QUANTIZATION,
        'generation_config': GENERATION_CONFIG,
        'n_prompts': n,
        'summary': summary,
        'results': results,
    }

    output_path = DATA_DIR / f"baseline_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    print(f"STRICT:   secure={strict_secure}/{n} ({strict_secure/n*100:.1f}%), "
          f"insecure={strict_insecure}/{n} ({strict_insecure/n*100:.1f}%), "
          f"other={strict_other}/{n} ({strict_other/n*100:.1f}%)")
    print(f"EXPANDED: secure={expanded_secure}/{n} ({expanded_secure/n*100:.1f}%)")
    print(f"Refusals: {refusals}/{n} ({refusals/n*100:.1f}%)")
    print(f"\nResults saved: {output_path}")

    # Decision gate
    if summary['strict']['secure_rate'] > 0.50:
        print("\n*** GATE: Baseline secure rate > 50% — steering has limited room ***")
        print("*** This is a finding: model may already be secure for CWE-787 ***")
    else:
        print(f"\n*** GATE PASSED: Baseline secure rate = {summary['strict']['secure_rate']*100:.1f}% ***")
        print("*** Proceed to activation collection ***")

    return str(output_path)


if __name__ == "__main__":
    main()
