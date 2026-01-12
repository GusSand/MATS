#!/usr/bin/env python3
"""
Validate Expanded CWE-787 Prompts

Tests whether the GPT-4o augmented prompts still elicit the correct behavior:
- Vulnerable prompts → insecure code (sprintf, strcat, etc.)
- Secure prompts → secure code (snprintf, strncat, etc.)

Usage:
    python 03_validate_expanded.py [--samples N] [--subset N]
"""

import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "01-08_llama8b_cwe787_prompt_pairs"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datetime import datetime
import argparse
import warnings
import os
import random
from tqdm import tqdm

from utils.cwe787_classification import (
    classify_with_enhanced_patterns, get_classification_symbol,
    ENHANCED_PATTERNS
)

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'


def load_expanded_dataset(data_dir: Path) -> list:
    """Load the most recent expanded dataset."""
    jsonl_files = list(data_dir.glob("cwe787_expanded_*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError("No expanded dataset found!")

    latest_file = sorted(jsonl_files)[-1]
    print(f"Loading: {latest_file.name}")

    pairs = []
    with open(latest_file) as f:
        for line in f:
            pairs.append(json.loads(line))

    return pairs, latest_file.name


class ExpandedValidator:
    """Validate expanded prompt pairs."""

    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        print("=" * 70)
        print("Expanded CWE-787 Prompt Validation")
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
        return generated[len(prompt):]

    def validate_pair(self, pair: dict, n_samples: int = 1) -> dict:
        """Validate a single expanded pair."""
        vulnerability_type = pair.get('vulnerability_type', 'sprintf')

        # Map vulnerability types to enhanced pattern types
        vuln_type_map = {
            'sprintf': 'sprintf',
            'strcat': 'strcat',
            'strcpy': 'strcpy',
        }
        pattern_type = vuln_type_map.get(vulnerability_type, 'sprintf')

        vulnerable_results = []
        secure_results = []

        # Generate from vulnerable prompt
        for i in range(n_samples):
            output = self.generate(pair['vulnerable'])
            result = classify_with_enhanced_patterns(output, pattern_type)
            result['sample_idx'] = i
            result['prompt_type'] = 'vulnerable'
            result['output_snippet'] = output[:200]
            vulnerable_results.append(result)

        # Generate from secure prompt
        for i in range(n_samples):
            output = self.generate(pair['secure'])
            result = classify_with_enhanced_patterns(output, pattern_type)
            result['sample_idx'] = i
            result['prompt_type'] = 'secure'
            result['output_snippet'] = output[:200]
            secure_results.append(result)

        return {
            'pair_id': pair['id'],
            'base_id': pair['base_id'],
            'name': pair['name'],
            'category': pair['category'],
            'vulnerability_type': vulnerability_type,
            'vulnerable_results': vulnerable_results,
            'secure_results': secure_results
        }

    def validate_dataset(self, pairs: list, n_samples: int = 1) -> dict:
        """Validate all pairs in the dataset."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'n_pairs': len(pairs),
            'n_samples_per_prompt': n_samples,
            'total_generations': len(pairs) * n_samples * 2,
            'pairs': {}
        }

        all_vulnerable = []
        all_secure = []

        # Track by category (original vs expanded)
        original_vuln = []
        original_sec = []
        expanded_vuln = []
        expanded_sec = []

        print(f"\nValidating {len(pairs)} pairs with {n_samples} sample(s) each...")
        print(f"Total generations: {len(pairs) * n_samples * 2}")
        print("-" * 70)

        for pair in tqdm(pairs, desc="Validating"):
            pair_results = self.validate_pair(pair, n_samples)
            results['pairs'][pair['id']] = pair_results

            # Collect results
            all_vulnerable.extend(pair_results['vulnerable_results'])
            all_secure.extend(pair_results['secure_results'])

            if pair['category'] == 'original':
                original_vuln.extend(pair_results['vulnerable_results'])
                original_sec.extend(pair_results['secure_results'])
            else:
                expanded_vuln.extend(pair_results['vulnerable_results'])
                expanded_sec.extend(pair_results['secure_results'])

        # Compute summaries
        def compute_summary(results_list):
            total = len(results_list)
            if total == 0:
                return {'total': 0, 'insecure': 0, 'secure': 0, 'neither': 0}

            insecure = sum(1 for r in results_list if r['classification'] == 'insecure')
            secure = sum(1 for r in results_list if r['classification'] == 'secure')
            neither = sum(1 for r in results_list if r['classification'] == 'neither')

            return {
                'total': total,
                'insecure': insecure,
                'secure': secure,
                'neither': neither,
                'insecure_rate': insecure / total,
                'secure_rate': secure / total,
                'neither_rate': neither / total
            }

        def compute_separation(vuln_summary, sec_summary):
            vuln_insecure = vuln_summary['insecure_rate'] if vuln_summary['total'] > 0 else 0
            sec_insecure = sec_summary['insecure_rate'] if sec_summary['total'] > 0 else 0
            return {
                'vuln_insecure_rate': vuln_insecure,
                'sec_insecure_rate': sec_insecure,
                'separation_pp': (vuln_insecure - sec_insecure) * 100
            }

        # Overall results
        vuln_summary = compute_summary(all_vulnerable)
        sec_summary = compute_summary(all_secure)
        overall_sep = compute_separation(vuln_summary, sec_summary)

        # By category
        orig_vuln_sum = compute_summary(original_vuln)
        orig_sec_sum = compute_summary(original_sec)
        orig_sep = compute_separation(orig_vuln_sum, orig_sec_sum)

        exp_vuln_sum = compute_summary(expanded_vuln)
        exp_sec_sum = compute_summary(expanded_sec)
        exp_sep = compute_separation(exp_vuln_sum, exp_sec_sum)

        results['summary'] = {
            'overall': {
                'vulnerable_prompts': vuln_summary,
                'secure_prompts': sec_summary,
                'separation': overall_sep
            },
            'original_pairs': {
                'n_pairs': len([p for p in pairs if p['category'] == 'original']),
                'vulnerable_prompts': orig_vuln_sum,
                'secure_prompts': orig_sec_sum,
                'separation': orig_sep
            },
            'expanded_pairs': {
                'n_pairs': len([p for p in pairs if p['category'] == 'expanded']),
                'vulnerable_prompts': exp_vuln_sum,
                'secure_prompts': exp_sec_sum,
                'separation': exp_sep
            }
        }

        # Print results
        self._print_results(results['summary'])

        return results

    def _print_results(self, summary):
        """Print formatted results."""
        print("\n" + "=" * 70)
        print("VALIDATION RESULTS")
        print("=" * 70)

        for category, label in [('overall', 'OVERALL'),
                                ('original_pairs', 'ORIGINAL (7 pairs)'),
                                ('expanded_pairs', 'EXPANDED (98 pairs)')]:
            data = summary[category]
            sep = data['separation']

            print(f"\n{label}:")
            print(f"  Vulnerable prompts → insecure: {data['vulnerable_prompts']['insecure_rate']*100:.1f}%")
            print(f"  Secure prompts → insecure:     {data['secure_prompts']['insecure_rate']*100:.1f}%")
            print(f"  Separation: {sep['separation_pp']:.1f} pp", end="")

            if sep['separation_pp'] >= 60:
                print(" ✓ GOOD")
            elif sep['separation_pp'] >= 30:
                print(" ~ ACCEPTABLE")
            else:
                print(" ✗ LOW")

        print("\n" + "=" * 70)
        overall_sep = summary['overall']['separation']['separation_pp']
        if overall_sep >= 50:
            print("SUCCESS: Expanded prompts maintain good separation")
        elif overall_sep >= 30:
            print("PARTIAL: Expanded prompts have reduced but acceptable separation")
        else:
            print("WARNING: Expanded prompts may need review")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Validate expanded CWE-787 prompts")
    parser.add_argument("--samples", type=int, default=1,
                        help="Samples per prompt (default: 1)")
    parser.add_argument("--subset", type=int, default=None,
                        help="Random subset of pairs to test (default: all)")
    parser.add_argument("--originals-only", action="store_true",
                        help="Only test original pairs (sanity check)")
    args = parser.parse_args()

    # Load dataset
    data_dir = Path(__file__).parent / "data"
    pairs, dataset_file = load_expanded_dataset(data_dir)

    print(f"Loaded {len(pairs)} pairs from {dataset_file}")

    # Filter if requested
    if args.originals_only:
        pairs = [p for p in pairs if p['category'] == 'original']
        print(f"Testing originals only: {len(pairs)} pairs")
    elif args.subset:
        # Ensure we include some originals and some expanded
        originals = [p for p in pairs if p['category'] == 'original']
        expanded = [p for p in pairs if p['category'] == 'expanded']

        n_orig = min(len(originals), max(2, args.subset // 5))
        n_exp = args.subset - n_orig

        pairs = random.sample(originals, n_orig) + random.sample(expanded, min(n_exp, len(expanded)))
        random.shuffle(pairs)
        print(f"Testing subset: {len(pairs)} pairs ({n_orig} original, {len(pairs)-n_orig} expanded)")

    # Create results directory
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Run validation
    validator = ExpandedValidator()
    results = validator.validate_dataset(pairs, args.samples)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"validation_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    results = main()
