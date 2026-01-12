#!/usr/bin/env python3
"""
Validate CWE-787 Prompt Pairs

This script:
1. Loads all 20 prompt pairs
2. Generates 1-2 samples per prompt (20-40 total)
3. Classifies outputs using regex
4. Reports separation between vulnerable and secure prompts

Usage:
    python 01_validate_prompts.py [--samples-per-prompt N] [--pairs N]
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datetime import datetime
import argparse
import warnings
import os

from config.cwe787_prompt_pairs import ALL_PAIRS, CORE_PAIRS
from utils.cwe787_classification import (
    classify_output, classify_with_enhanced_patterns, get_classification_symbol,
    summarize_results, print_summary, analyze_pair_separation, ENHANCED_PATTERNS
)

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'


class PromptValidator:
    """Validate prompt pairs by generating and classifying outputs."""

    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        print("=" * 70)
        print("CWE-787 Prompt Pair Validation")
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

    def generate(self, prompt: str, max_new_tokens: int = 350,
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
        # Return only the generated part
        return generated[len(prompt):]

    def validate_pair(self, pair: dict, n_samples: int = 1) -> dict:
        """
        Validate a single prompt pair.

        Args:
            pair: Prompt pair dictionary
            n_samples: Number of samples to generate per prompt type

        Returns:
            Results dictionary with classifications
        """
        pair_id = pair['id']
        vulnerability_type = pair['vulnerability_type']

        # Use enhanced patterns if available, otherwise fall back to pair-specific detection
        use_enhanced = vulnerability_type in ENHANCED_PATTERNS

        vulnerable_results = []
        secure_results = []

        # Generate from vulnerable prompt
        for i in range(n_samples):
            output = self.generate(pair['vulnerable'])
            if use_enhanced:
                result = classify_with_enhanced_patterns(output, vulnerability_type)
            else:
                result = classify_output(output, pair['detection'])
            result['sample_idx'] = i
            result['prompt_type'] = 'vulnerable'
            vulnerable_results.append(result)

        # Generate from secure prompt
        for i in range(n_samples):
            output = self.generate(pair['secure'])
            if use_enhanced:
                result = classify_with_enhanced_patterns(output, vulnerability_type)
            else:
                result = classify_output(output, pair['detection'])
            result['sample_idx'] = i
            result['prompt_type'] = 'secure'
            secure_results.append(result)

        return {
            'pair_id': pair_id,
            'pair_name': pair['name'],
            'vulnerability_type': vulnerability_type,
            'category': pair['category'],
            'vulnerable_results': vulnerable_results,
            'secure_results': secure_results
        }

    def validate_all(self, pairs: list, n_samples: int = 1) -> dict:
        """
        Validate all prompt pairs.

        Args:
            pairs: List of prompt pair dictionaries
            n_samples: Number of samples per prompt type

        Returns:
            Full results dictionary
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'n_pairs': len(pairs),
            'n_samples_per_prompt': n_samples,
            'total_samples': len(pairs) * n_samples * 2,
            'pairs': {}
        }

        all_vulnerable_results = []
        all_secure_results = []

        print(f"\nValidating {len(pairs)} pairs with {n_samples} sample(s) each...")
        print(f"Total generations: {len(pairs) * n_samples * 2}")
        print("-" * 70)

        for i, pair in enumerate(pairs):
            print(f"\n[{i+1}/{len(pairs)}] {pair['id']}: {pair['name']}")
            print(f"    Type: {pair['vulnerability_type']} | Category: {pair['category']}")

            pair_results = self.validate_pair(pair, n_samples)

            # Print per-sample results
            print(f"    Vulnerable prompt results: ", end="")
            for r in pair_results['vulnerable_results']:
                symbol = get_classification_symbol(r)
                print(f"[{symbol}] ", end="")
            print()

            print(f"    Secure prompt results:     ", end="")
            for r in pair_results['secure_results']:
                symbol = get_classification_symbol(r)
                print(f"[{symbol}] ", end="")
            print()

            results['pairs'][pair['id']] = pair_results
            all_vulnerable_results.extend(pair_results['vulnerable_results'])
            all_secure_results.extend(pair_results['secure_results'])

        # Overall analysis
        print("\n" + "=" * 70)
        print("OVERALL RESULTS")
        print("=" * 70)

        vuln_summary = summarize_results(all_vulnerable_results)
        secure_summary = summarize_results(all_secure_results)

        print_summary(vuln_summary, "Vulnerable Prompts")
        print_summary(secure_summary, "Secure Prompts")

        # Separation analysis
        separation = analyze_pair_separation(all_vulnerable_results, all_secure_results)

        print("\n" + "-" * 70)
        print("SEPARATION ANALYSIS")
        print("-" * 70)
        print(f"  Insecure rate (vulnerable prompts): {vuln_summary['insecure_rate']*100:.1f}%")
        print(f"  Insecure rate (secure prompts):     {secure_summary['insecure_rate']*100:.1f}%")
        print(f"  Separation: {separation['separation_percentage_points']:.1f} percentage points")
        print(f"  Target (>=60pp): {'MET' if separation['meets_threshold'] else 'NOT MET'}")

        results['overall'] = {
            'vulnerable_summary': vuln_summary,
            'secure_summary': secure_summary,
            'separation': separation
        }

        # Success check
        print("\n" + "=" * 70)
        if separation['separation_percentage_points'] > 0:
            print("SUCCESS: Vulnerable prompts produce more insecure code than secure prompts")
        else:
            print("WARNING: No separation detected - prompts may need refinement")
        print("=" * 70)

        return results


def main():
    parser = argparse.ArgumentParser(description="Validate CWE-787 prompt pairs")
    parser.add_argument("--samples-per-prompt", type=int, default=1,
                        help="Number of samples to generate per prompt type (default: 1)")
    parser.add_argument("--pairs", type=int, default=None,
                        help="Number of pairs to test (default: all 20)")
    parser.add_argument("--core-only", action="store_true",
                        help="Only test core pairs (1-10)")
    args = parser.parse_args()

    # Select pairs
    if args.core_only:
        pairs = CORE_PAIRS
        print(f"Using CORE pairs only: {len(pairs)} pairs")
    elif args.pairs:
        pairs = ALL_PAIRS[:args.pairs]
        print(f"Using first {len(pairs)} pairs")
    else:
        pairs = ALL_PAIRS
        print(f"Using ALL {len(pairs)} pairs")

    # Create results directory
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Run validation
    validator = PromptValidator()
    results = validator.validate_all(pairs, args.samples_per_prompt)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"validation_{timestamp}.json"

    # Make JSON serializable
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
