#!/usr/bin/env python3
"""
Validate Multi-CWE Prompt Pairs

Tests prompt pairs across multiple CWEs:
- CWE-476: NULL Pointer Dereference
- CWE-252: Unchecked Return Value
- CWE-401: Memory Leak
- CWE-772: Resource Leak
- CWE-681: Integer Overflow
- CWE-787: Out-of-bounds Write (from original pairs)

Usage:
    python 02_validate_multi_cwe.py [--samples-per-prompt N] [--cwe CWE-XXX]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datetime import datetime
import argparse
import warnings
import os

from config.multi_cwe_prompt_pairs import (
    ALL_NEW_CWE_PAIRS, PAIRS_BY_CWE,
    CWE476_PAIRS, CWE252_PAIRS, CWE401_PAIRS, CWE772_PAIRS, CWE681_PAIRS
)
from config.cwe787_prompt_pairs import (
    PAIR_07_SPRINTF_LOG, PAIR_09_PATH_JOIN, PAIR_11_JSON, PAIR_12_XML,
    PAIR_16_HIGH_COMPLEXITY, PAIR_17_TIME_PRESSURE, PAIR_19_GRAPHICS
)
from utils.cwe787_classification import (
    classify_output, classify_with_enhanced_patterns, get_classification_symbol,
    summarize_results, print_summary, analyze_pair_separation, ENHANCED_PATTERNS
)

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Validated CWE-787 pairs
CWE787_VALIDATED = [
    PAIR_07_SPRINTF_LOG, PAIR_09_PATH_JOIN, PAIR_11_JSON, PAIR_12_XML,
    PAIR_16_HIGH_COMPLEXITY, PAIR_17_TIME_PRESSURE, PAIR_19_GRAPHICS
]

# Add CWE info to 787 pairs
for p in CWE787_VALIDATED:
    p['cwe'] = 'CWE-787'
    p['cwe_name'] = 'Out-of-bounds Write'


class MultiCWEValidator:
    """Validate prompt pairs across multiple CWEs."""

    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        print("=" * 70)
        print("Multi-CWE Prompt Pair Validation")
        print("=" * 70)

        print("\nLoading model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.model.eval()
        print("Model loaded successfully!")

    def generate(self, prompt: str, max_new_tokens: int = 400,
                 temperature: float = 0.7) -> str:
        """Generate code completion."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated[len(prompt):]

    def validate_pair(self, pair: dict, n_samples: int = 1) -> dict:
        """Validate a single prompt pair."""
        pair_id = pair['id']
        vulnerability_type = pair['vulnerability_type']
        cwe = pair.get('cwe', 'Unknown')

        use_enhanced = vulnerability_type in ENHANCED_PATTERNS

        vulnerable_results = []
        secure_results = []

        # Generate from vulnerable prompt
        for i in range(n_samples):
            output = self.generate(pair['vulnerable'])
            if use_enhanced:
                result = classify_with_enhanced_patterns(output, vulnerability_type)
            else:
                result = classify_output(output, pair.get('detection', {}))
            result['sample_idx'] = i
            result['prompt_type'] = 'vulnerable'
            vulnerable_results.append(result)

        # Generate from secure prompt
        for i in range(n_samples):
            output = self.generate(pair['secure'])
            if use_enhanced:
                result = classify_with_enhanced_patterns(output, vulnerability_type)
            else:
                result = classify_output(output, pair.get('detection', {}))
            result['sample_idx'] = i
            result['prompt_type'] = 'secure'
            secure_results.append(result)

        return {
            'pair_id': pair_id,
            'pair_name': pair['name'],
            'cwe': cwe,
            'cwe_name': pair.get('cwe_name', ''),
            'vulnerability_type': vulnerability_type,
            'category': pair.get('category', ''),
            'vulnerable_results': vulnerable_results,
            'secure_results': secure_results
        }

    def validate_cwe(self, cwe: str, pairs: list, n_samples: int = 1) -> dict:
        """Validate all pairs for a specific CWE."""
        print(f"\n{'='*70}")
        print(f"{cwe}: {pairs[0].get('cwe_name', '')}")
        print(f"{'='*70}")

        results = {
            'cwe': cwe,
            'cwe_name': pairs[0].get('cwe_name', ''),
            'n_pairs': len(pairs),
            'pairs': {}
        }

        all_vuln = []
        all_secure = []

        for i, pair in enumerate(pairs):
            print(f"\n[{i+1}/{len(pairs)}] {pair['id']}: {pair['name']}")

            pair_results = self.validate_pair(pair, n_samples)

            # Print results
            print(f"    Vulnerable: ", end="")
            for r in pair_results['vulnerable_results']:
                print(f"[{get_classification_symbol(r)}] ", end="")
            print()

            print(f"    Secure:     ", end="")
            for r in pair_results['secure_results']:
                print(f"[{get_classification_symbol(r)}] ", end="")
            print()

            results['pairs'][pair['id']] = pair_results
            all_vuln.extend(pair_results['vulnerable_results'])
            all_secure.extend(pair_results['secure_results'])

        # CWE summary
        vuln_summary = summarize_results(all_vuln)
        secure_summary = summarize_results(all_secure)
        separation = analyze_pair_separation(all_vuln, all_secure)

        print(f"\n{cwe} Summary:")
        print(f"  Vulnerable prompts: {vuln_summary['insecure_rate']*100:.0f}% vuln, {vuln_summary['secure_rate']*100:.0f}% secure")
        print(f"  Secure prompts: {secure_summary['secure_rate']*100:.0f}% secure, {secure_summary['insecure_rate']*100:.0f}% vuln")
        print(f"  Separation: {separation['separation_percentage_points']:.0f}pp")

        results['summary'] = {
            'vulnerable': vuln_summary,
            'secure': secure_summary,
            'separation': separation
        }

        return results

    def validate_all(self, n_samples: int = 1) -> dict:
        """Validate all CWEs."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'n_samples_per_prompt': n_samples,
            'cwes': {}
        }

        # CWE-787 (validated pairs)
        results['cwes']['CWE-787'] = self.validate_cwe(
            'CWE-787', CWE787_VALIDATED, n_samples
        )

        # New CWEs
        for cwe, pairs in PAIRS_BY_CWE.items():
            results['cwes'][cwe] = self.validate_cwe(cwe, pairs, n_samples)

        # Overall summary
        print("\n" + "=" * 70)
        print("OVERALL SUMMARY BY CWE")
        print("=" * 70)

        total_vuln = []
        total_secure = []

        for cwe, cwe_results in results['cwes'].items():
            sep = cwe_results['summary']['separation']['separation_percentage_points']
            vuln_rate = cwe_results['summary']['vulnerable']['insecure_rate'] * 100
            secure_rate = cwe_results['summary']['secure']['secure_rate'] * 100
            status = "GOOD" if sep >= 40 else "WEAK" if sep >= 20 else "POOR"

            print(f"{cwe}: {vuln_rate:.0f}% vuln / {secure_rate:.0f}% secure = {sep:.0f}pp [{status}]")

            # Aggregate
            for pair_results in cwe_results['pairs'].values():
                total_vuln.extend(pair_results['vulnerable_results'])
                total_secure.extend(pair_results['secure_results'])

        # Overall
        overall_sep = analyze_pair_separation(total_vuln, total_secure)
        print(f"\nOVERALL: {overall_sep['separation_percentage_points']:.0f}pp separation")

        results['overall'] = {
            'separation': overall_sep,
            'total_samples': len(total_vuln) + len(total_secure)
        }

        return results


def main():
    parser = argparse.ArgumentParser(description="Validate multi-CWE prompt pairs")
    parser.add_argument("--samples-per-prompt", type=int, default=2,
                        help="Samples per prompt type (default: 2)")
    parser.add_argument("--cwe", type=str, default=None,
                        help="Test specific CWE only (e.g., CWE-476)")
    args = parser.parse_args()

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    validator = MultiCWEValidator()

    if args.cwe:
        # Single CWE
        if args.cwe == "CWE-787":
            pairs = CWE787_VALIDATED
        elif args.cwe in PAIRS_BY_CWE:
            pairs = PAIRS_BY_CWE[args.cwe]
        else:
            print(f"Unknown CWE: {args.cwe}")
            print(f"Available: CWE-787, {', '.join(PAIRS_BY_CWE.keys())}")
            return

        results = {'cwes': {args.cwe: validator.validate_cwe(args.cwe, pairs, args.samples_per_prompt)}}
    else:
        # All CWEs
        results = validator.validate_all(args.samples_per_prompt)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"multi_cwe_validation_{timestamp}.json"

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    with open(output_file, 'w') as f:
        json.dump(make_serializable(results), f, indent=2)

    print(f"\nResults saved to: {output_file}")
    return results


if __name__ == "__main__":
    results = main()
