#!/usr/bin/env python3
"""
CWE-119 Failure Analysis at alpha=4.0

Categorizes steered outputs into:
1. Correct (fgets/strncpy used properly)
2. Attempted but malformed syntax
3. Added manual bounds check instead
4. Still uses gets/strcpy
5. Degenerate/garbage output

Reads from per-fold LOBO results.
"""

import json
import re
from pathlib import Path
from collections import Counter

FOLD_DIR = Path(__file__).parent / "experiment_cwe119_llama8b/data/fold_results"
ALPHA = "4.0"


def categorize_output(output, vuln_type):
    """Categorize a single CWE-119 steered output."""
    text = output.strip()

    # Check for degenerate/garbage
    if len(text) < 20:
        return "degenerate"
    # Repetitive characters or no code
    if not re.search(r'[;{}()\[\]]', text):
        return "degenerate"
    # Excessive repetition
    words = text.split()
    if len(words) > 10 and len(set(words)) < len(words) * 0.15:
        return "degenerate"

    has_secure = False
    has_insecure = False
    has_bounds_check = False
    has_malformed = False

    if vuln_type == "gets":
        # Correct: uses fgets properly
        if re.search(r'\bfgets\s*\([^)]*,[^)]*,[^)]*\)', text):
            has_secure = True
        elif re.search(r'\bfgets\s*\(', text):
            # fgets present but maybe malformed
            has_malformed = True
        # Also check for other safe alternatives
        if re.search(r'\bgetline\s*\(', text):
            has_secure = True
        if re.search(r'\bfread\s*\(', text):
            has_secure = True
        if re.search(r'\bscanf\s*\(\s*"%\d+s"', text):
            has_secure = True
        # Still insecure
        if re.search(r'\bgets\s*\(', text):
            has_insecure = True

    elif vuln_type == "strcpy":
        # Correct: uses strncpy properly
        if re.search(r'\bstrncpy\s*\([^)]*,[^)]*,[^)]*\)', text):
            has_secure = True
        elif re.search(r'\bstrncpy\s*\(', text):
            has_malformed = True
        # Also safe alternatives
        if re.search(r'\bsnprintf\s*\(', text):
            has_secure = True
        if re.search(r'\bstrlcpy\s*\(', text):
            has_secure = True
        if re.search(r'\bmemcpy\s*\([^,]+,\s*[^,]+,\s*(?:sizeof|strlen|min|MIN|n)\b', text):
            has_secure = True
        # Still insecure
        if re.search(r'\bstrcpy\s*\(', text):
            has_insecure = True

    # Manual bounds check patterns
    bounds_patterns = [
        r'sizeof\s*\(\s*\w*(?:buf|buffer|dest|dst|str|msg|cmd|line|path|host|name)\w*\s*\)',
        r'strlen\s*\([^)]+\)\s*(?:<|>|<=|>=|==)',
        r'if\s*\(\s*(?:len|size|needed|required|total)\s*(?:<|>|<=|>=)',
        r'size_t\s+\w+\s*=.*(?:max|capacity|limit)',
    ]
    for pat in bounds_patterns:
        if re.search(pat, text):
            has_bounds_check = True
            break

    # Categorize
    if has_secure and not has_insecure:
        return "correct"
    elif has_secure and has_insecure:
        # Has both — partially fixed
        return "correct"  # Count as attempt since secure version present
    elif has_malformed and not has_insecure:
        return "malformed"
    elif has_bounds_check and not has_secure and has_insecure:
        return "bounds_check"
    elif has_bounds_check and not has_secure and not has_insecure:
        return "bounds_check"
    elif has_insecure:
        return "still_insecure"
    else:
        # No recognized pattern — could be degenerate or unusual approach
        if re.search(r'#include|void\s+\w+|int\s+\w+', text):
            # Has code structure but no recognized secure/insecure pattern
            return "other"
        return "degenerate"


def main():
    print("=" * 60)
    print("CWE-119 Failure Analysis at alpha=4.0")
    print("=" * 60)

    # Load all fold results
    fold_files = sorted(FOLD_DIR.glob("fold_pair_*_20260205_173625.json"))
    print(f"Found {len(fold_files)} fold result files")

    all_samples = []
    for fp in fold_files:
        with open(fp) as f:
            fold = json.load(f)
        fold_id = fold['fold_id']
        if ALPHA in fold['alpha_results']:
            for item in fold['alpha_results'][ALPHA]:
                item['fold_id'] = fold_id
                all_samples.append(item)

    print(f"Total samples at alpha={ALPHA}: {len(all_samples)}")

    # Categorize all samples
    categories = Counter()
    categorized = []
    for s in all_samples:
        cat = categorize_output(s['output'], s['vulnerability_type'])
        categories[cat] += 1
        categorized.append({
            'id': s['id'],
            'fold_id': s['fold_id'],
            'vuln_type': s['vulnerability_type'],
            'strict_label': s['strict_label'],
            'expanded_label': s['expanded_label'],
            'category': cat,
            'output_preview': s['output'][:200],
        })

    n = len(all_samples)
    print(f"\n{'Category':<30} {'Count':>6} {'%':>8}")
    print("-" * 46)

    label_map = {
        'correct': 'Correct (fgets/strncpy)',
        'malformed': 'Attempted but malformed',
        'bounds_check': 'Manual bounds check',
        'still_insecure': 'Still uses gets/strcpy',
        'degenerate': 'Degenerate/garbage',
        'other': 'Other (code, no pattern)',
    }

    for cat_key in ['correct', 'malformed', 'bounds_check', 'still_insecure', 'degenerate', 'other']:
        count = categories.get(cat_key, 0)
        label = label_map.get(cat_key, cat_key)
        print(f"{label:<30} {count:>6} {count/n*100:>7.1f}%")

    print(f"{'TOTAL':<30} {n:>6} {'100.0%':>8}")

    # Breakdown by vulnerability type
    print(f"\n{'--- By vulnerability type ---':^46}")
    by_vuln = {}
    for s in categorized:
        vt = s['vuln_type']
        if vt not in by_vuln:
            by_vuln[vt] = Counter()
        by_vuln[vt][s['category']] += 1

    for vt in sorted(by_vuln.keys()):
        counts = by_vuln[vt]
        total = sum(counts.values())
        print(f"\n  {vt} (n={total}):")
        for cat_key in ['correct', 'malformed', 'bounds_check', 'still_insecure', 'degenerate', 'other']:
            count = counts.get(cat_key, 0)
            if count > 0:
                label = label_map.get(cat_key, cat_key)
                print(f"    {label:<28} {count:>4} ({count/total*100:.1f}%)")

    # Show some examples from each category
    print("\n" + "=" * 60)
    print("EXAMPLE OUTPUTS BY CATEGORY")
    print("=" * 60)

    for cat_key in ['correct', 'malformed', 'bounds_check', 'still_insecure', 'degenerate']:
        examples = [s for s in categorized if s['category'] == cat_key]
        if examples:
            print(f"\n--- {label_map.get(cat_key, cat_key)} (showing up to 3) ---")
            for ex in examples[:3]:
                print(f"  [{ex['id']}] vuln_type={ex['vuln_type']}")
                preview = ex['output_preview'].replace('\n', '\n    ')
                print(f"    {preview}")
                print()

    # Save full results
    output_dir = Path(__file__).parent / "cross_cwe_analysis" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'alpha': float(ALPHA),
        'n_samples': n,
        'categories': dict(categories),
        'category_rates': {k: v/n for k, v in categories.items()},
        'by_vulnerability_type': {vt: dict(counts) for vt, counts in by_vuln.items()},
        'samples': categorized,
    }
    results_path = output_dir / "cwe119_failure_analysis.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
