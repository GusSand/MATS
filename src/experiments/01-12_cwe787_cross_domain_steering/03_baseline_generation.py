#!/usr/bin/env python3
"""
Step 3: Generate baseline outputs from vulnerable prompts (no steering).

Generates code from all 105 vulnerable prompts and classifies using regex patterns.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "01-08_llama8b_sr_scg_separation"))

import json
import re
from datetime import datetime
from tqdm import tqdm
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_dataset(data_path: Path) -> list:
    """Load the expanded CWE-787 dataset."""
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def classify_output(output: str, detection: dict) -> str:
    """Classify generated output as secure/insecure/incomplete."""
    secure_pattern = detection['secure_pattern']
    insecure_pattern = detection['insecure_pattern']

    has_secure = bool(re.search(secure_pattern, output))
    has_insecure = bool(re.search(insecure_pattern, output))

    if has_secure:
        return 'secure'
    elif has_insecure:
        return 'insecure'
    else:
        return 'incomplete'


def generate_baseline(model, tokenizer, dataset: list,
                      temperature: float = 0.6, max_tokens: int = 300) -> list:
    """Generate outputs from vulnerable prompts without steering."""
    device = next(model.parameters()).device
    results = []

    print(f"\nGenerating baseline outputs for {len(dataset)} vulnerable prompts...")
    print(f"Temperature: {temperature}, Max tokens: {max_tokens}")

    for item in tqdm(dataset, desc="Generating"):
        prompt = item['vulnerable']

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output = generated[len(prompt):]

        label = classify_output(output, item['detection'])

        results.append({
            'id': item['id'],
            'base_id': item['base_id'],
            'vulnerability_type': item['vulnerability_type'],
            'output': output[:500],  # Truncate for storage
            'label': label,
            'prompt_snippet': prompt[:100]
        })

    return results


def summarize_results(results: list) -> dict:
    """Compute summary statistics."""
    total = len(results)
    secure = sum(1 for r in results if r['label'] == 'secure')
    insecure = sum(1 for r in results if r['label'] == 'insecure')
    incomplete = sum(1 for r in results if r['label'] == 'incomplete')

    # By vulnerability type
    by_vuln_type = {}
    for vtype in set(r['vulnerability_type'] for r in results):
        vtype_results = [r for r in results if r['vulnerability_type'] == vtype]
        by_vuln_type[vtype] = {
            'total': len(vtype_results),
            'secure': sum(1 for r in vtype_results if r['label'] == 'secure'),
            'insecure': sum(1 for r in vtype_results if r['label'] == 'insecure'),
            'incomplete': sum(1 for r in vtype_results if r['label'] == 'incomplete')
        }

    return {
        'total': total,
        'secure': secure,
        'insecure': insecure,
        'incomplete': incomplete,
        'secure_rate': secure / total if total > 0 else 0,
        'insecure_rate': insecure / total if total > 0 else 0,
        'incomplete_rate': incomplete / total if total > 0 else 0,
        'by_vulnerability_type': by_vuln_type
    }


def main():
    parser = argparse.ArgumentParser(description="Generate baseline outputs")
    parser.add_argument("--dataset", type=str,
                        default="../01-12_cwe787_dataset_expansion/data/cwe787_expanded_20260112_143316.jsonl",
                        help="Path to expanded dataset")
    parser.add_argument("--model", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="Model name")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=300)
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    data_dir.mkdir(exist_ok=True)

    dataset_path = script_dir / args.dataset
    if not dataset_path.exists():
        dataset_path = Path(args.dataset)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} pairs")

    # Load model
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    # Generate
    results = generate_baseline(
        model, tokenizer, dataset,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )

    # Summarize
    summary = summarize_results(results)

    # Save
    output = {
        'timestamp': timestamp,
        'config': {
            'model': args.model,
            'temperature': args.temperature,
            'max_tokens': args.max_tokens,
            'n_prompts': len(dataset)
        },
        'summary': summary,
        'results': results
    }

    output_path = data_dir / f"baseline_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "="*50)
    print("BASELINE GENERATION COMPLETE")
    print("="*50)
    print(f"\nResults:")
    print(f"  Total: {summary['total']}")
    print(f"  Secure: {summary['secure']} ({summary['secure_rate']*100:.1f}%)")
    print(f"  Insecure: {summary['insecure']} ({summary['insecure_rate']*100:.1f}%)")
    print(f"  Incomplete: {summary['incomplete']} ({summary['incomplete_rate']*100:.1f}%)")
    print(f"\nBy vulnerability type:")
    for vtype, stats in summary['by_vulnerability_type'].items():
        print(f"  {vtype}: {stats['secure']}/{stats['total']} secure ({stats['secure']/stats['total']*100:.1f}%)")
    print(f"\nOutput: {output_path}")

    return str(output_path)


if __name__ == "__main__":
    main()
