#!/usr/bin/env python3
"""
Generate and validate prompts for all 14 security pairs.

This script:
1. Loads all security pair configurations
2. Validates that prompts are well-formed
3. Optionally tests a sample to verify model behavior
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config.security_pairs import SECURITY_PAIRS, ALL_PAIRS, CORE_PAIRS
import json
from datetime import datetime


def validate_pair_config(pair_name: str, config: dict) -> list:
    """Validate a security pair configuration."""
    errors = []

    required_keys = ['insecure', 'secure', 'vulnerability_class',
                     'secure_templates', 'neutral_templates', 'detection_patterns']

    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required key: {key}")

    if 'secure_templates' in config:
        if len(config['secure_templates']) < 2:
            errors.append(f"Need at least 2 secure templates, got {len(config['secure_templates'])}")

    if 'neutral_templates' in config:
        if len(config['neutral_templates']) < 2:
            errors.append(f"Need at least 2 neutral templates, got {len(config['neutral_templates'])}")

    if 'detection_patterns' in config:
        if 'secure' not in config['detection_patterns']:
            errors.append("Missing 'secure' detection pattern")
        if 'insecure' not in config['detection_patterns']:
            errors.append("Missing 'insecure' detection pattern")

    return errors


def summarize_pairs():
    """Print summary of all security pairs."""
    print("=" * 70)
    print("SECURITY PAIRS SUMMARY")
    print("=" * 70)

    by_class = {}
    for pair_name, config in SECURITY_PAIRS.items():
        vuln_class = config['vulnerability_class']
        if vuln_class not in by_class:
            by_class[vuln_class] = []
        by_class[vuln_class].append(pair_name)

    for vuln_class, pairs in by_class.items():
        print(f"\n{vuln_class.upper()} ({len(pairs)} pairs):")
        for pair_name in pairs:
            config = SECURITY_PAIRS[pair_name]
            print(f"  - {pair_name}: {config['insecure']} -> {config['secure']}")
            print(f"    Templates: {len(config['secure_templates'])} secure, {len(config['neutral_templates'])} neutral")


def validate_all_pairs():
    """Validate all security pair configurations."""
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    all_valid = True
    for pair_name, config in SECURITY_PAIRS.items():
        errors = validate_pair_config(pair_name, config)
        if errors:
            print(f"\n{pair_name}: FAILED")
            for error in errors:
                print(f"  - {error}")
            all_valid = False
        else:
            print(f"{pair_name}: OK")

    return all_valid


def export_prompts(output_path: Path):
    """Export all prompts to a JSON file for inspection."""
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'n_pairs': len(SECURITY_PAIRS),
        'pairs': {}
    }

    total_secure = 0
    total_neutral = 0

    for pair_name, config in SECURITY_PAIRS.items():
        export_data['pairs'][pair_name] = {
            'insecure_fn': config['insecure'],
            'secure_fn': config['secure'],
            'vulnerability_class': config['vulnerability_class'],
            'secure_templates': config['secure_templates'],
            'neutral_templates': config['neutral_templates'],
            'detection_patterns': config['detection_patterns'],
            'n_secure': len(config['secure_templates']),
            'n_neutral': len(config['neutral_templates'])
        }
        total_secure += len(config['secure_templates'])
        total_neutral += len(config['neutral_templates'])

    export_data['totals'] = {
        'secure_templates': total_secure,
        'neutral_templates': total_neutral,
        'total_templates': total_secure + total_neutral
    }

    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"\nExported to: {output_path}")
    print(f"Total templates: {total_secure} secure + {total_neutral} neutral = {total_secure + total_neutral}")


def main():
    results_dir = Path(__file__).parent / "data"
    results_dir.mkdir(exist_ok=True)

    # Summarize
    summarize_pairs()

    # Validate
    all_valid = validate_all_pairs()

    if not all_valid:
        print("\nWARNING: Some pairs have validation errors!")
        return False

    # Export
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = results_dir / f"prompts_{timestamp}.json"
    export_prompts(export_path)

    print("\n" + "=" * 70)
    print("READY FOR EXPERIMENTS")
    print("=" * 70)
    print(f"\nTotal pairs: {len(ALL_PAIRS)}")
    print(f"Core pairs (for quick testing): {len(CORE_PAIRS)}")
    print(f"  {CORE_PAIRS}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
