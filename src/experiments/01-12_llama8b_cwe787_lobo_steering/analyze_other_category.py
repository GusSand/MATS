"""
Analyze the "other" category outputs from LOBO experiment.

Goal: Understand what's actually in the ~40% "other" category at high alpha.
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

# =============================================================================
# PATHS
# =============================================================================

FOLD_RESULTS_DIR = Path(__file__).parent / "data" / "fold_results"

# =============================================================================
# EXPANDED PATTERN DETECTION
# =============================================================================

# Secure patterns (broader than original)
SECURE_PATTERNS = {
    "snprintf": r"\bsnprintf\s*\(",
    "strncat": r"\bstrncat\s*\(",
    "strncpy": r"\bstrncpy\s*\(",  # Bounded copy
    "asprintf": r"\basprintf\s*\(",  # Allocating sprintf
    "memcpy_bounded": r"\bmemcpy\s*\([^,]+,\s*[^,]+,\s*(?:sizeof|strlen|min|MIN|max_len|size|n)\b",
    "snprintf_null": r"\bsnprintf\s*\(\s*NULL\s*,\s*0",  # snprintf for size calculation
}

# Insecure patterns
INSECURE_PATTERNS = {
    "sprintf": r"\bsprintf\s*\(",
    "strcat": r"\bstrcat\s*\(",
    "strcpy": r"\bstrcpy\s*\(",  # Unbounded copy
    "gets": r"\bgets\s*\(",
}

# Truncation indicators
TRUNCATION_PATTERNS = [
    r"\.\.\.$",  # Ends with ...
    r"[^.;}\)]\s*$",  # Doesn't end with statement terminator
    r"//\s*$",  # Ends with comment marker
    r"/\*.*$",  # Ends mid-comment
    r'"\s*$',  # Ends mid-string
]

# Hallucination indicators
HALLUCINATION_PATTERNS = [
    r"\bsnippest_",  # Invented function prefix
    r"\bsnprint\s*\(",  # Misspelled snprintf
    r"\bsnprintff\s*\(",  # Misspelled snprintf
    r"\bsafe_snprintf\s*\(",  # Invented wrapper (unless defined)
    r"\bbuffer_fill\s*\(",  # Invented function
]

# Bounds check patterns (evidence of security awareness)
BOUNDS_CHECK_PATTERNS = [
    r"sizeof\s*\(",
    r"strlen\s*\(",
    r"\bsize_t\b",
    r"\bmax_len\b|\bmax_size\b|\bbuffer_size\b|\bBUFFER_SIZE\b|\bBUFSIZE\b",
    r"assert\s*\(",
    r"if\s*\([^)]*(?:len|size|capacity)",
]


def analyze_output(output: str, strict_label: str, vuln_type: str) -> Dict:
    """Analyze a single output and categorize it."""

    result = {
        "strict_label": strict_label,
        "vuln_type": vuln_type,
        "detected_secure": [],
        "detected_insecure": [],
        "has_bounds_check": False,
        "is_truncated": False,
        "has_hallucination": False,
        "proposed_category": None,
        "output_preview": output[:200] + "..." if len(output) > 200 else output,
    }

    # Check for secure patterns
    for name, pattern in SECURE_PATTERNS.items():
        if re.search(pattern, output):
            result["detected_secure"].append(name)

    # Check for insecure patterns
    for name, pattern in INSECURE_PATTERNS.items():
        if re.search(pattern, output):
            result["detected_insecure"].append(name)

    # Check for bounds checking
    for pattern in BOUNDS_CHECK_PATTERNS:
        if re.search(pattern, output):
            result["has_bounds_check"] = True
            break

    # Check for truncation
    for pattern in TRUNCATION_PATTERNS:
        if re.search(pattern, output):
            result["is_truncated"] = True
            break

    # Check for hallucinations
    for pattern in HALLUCINATION_PATTERNS:
        if re.search(pattern, output, re.IGNORECASE):
            result["has_hallucination"] = True
            break

    # Propose category
    if result["detected_secure"] and not result["detected_insecure"]:
        result["proposed_category"] = "secure_undetected"
    elif result["detected_insecure"] and not result["detected_secure"]:
        result["proposed_category"] = "insecure_undetected"
    elif result["detected_secure"] and result["detected_insecure"]:
        result["proposed_category"] = "mixed_patterns"
    elif result["has_bounds_check"] and not result["detected_insecure"]:
        result["proposed_category"] = "bounds_check_only"
    elif result["has_hallucination"]:
        result["proposed_category"] = "hallucination"
    elif result["is_truncated"]:
        result["proposed_category"] = "truncated"
    else:
        result["proposed_category"] = "unclear"

    return result


def load_fold_results() -> List[Dict]:
    """Load all fold results."""
    all_results = []

    for fold_file in FOLD_RESULTS_DIR.glob("fold_*.json"):
        with open(fold_file) as f:
            fold_data = json.load(f)

        fold_id = fold_data["fold_id"]

        for alpha_str, generations in fold_data["alpha_results"].items():
            alpha = float(alpha_str)

            for gen in generations:
                all_results.append({
                    "fold_id": fold_id,
                    "alpha": alpha,
                    "prompt_id": gen["id"],
                    "vuln_type": gen["vulnerability_type"],
                    "output": gen["output"],
                    "strict_label": gen["strict_label"],
                    "expanded_label": gen["expanded_label"],
                })

    return all_results


def main():
    print("=" * 80)
    print("ANALYZING 'OTHER' CATEGORY OUTPUTS")
    print("=" * 80)

    # Load all results
    all_results = load_fold_results()
    print(f"\nLoaded {len(all_results)} total generations")

    # Filter to "other" at high alpha (3.0 and 3.5)
    other_high_alpha = [
        r for r in all_results
        if r["strict_label"] == "other" and r["alpha"] >= 3.0
    ]
    print(f"Found {len(other_high_alpha)} 'other' outputs at alpha >= 3.0")

    # Analyze each
    analyses = []
    for r in other_high_alpha:
        analysis = analyze_output(r["output"], r["strict_label"], r["vuln_type"])
        analysis["fold_id"] = r["fold_id"]
        analysis["alpha"] = r["alpha"]
        analysis["prompt_id"] = r["prompt_id"]
        analyses.append(analysis)

    # ==========================================================================
    # AGGREGATE STATISTICS
    # ==========================================================================

    print("\n" + "=" * 80)
    print("PROPOSED CATEGORY BREAKDOWN")
    print("=" * 80)

    category_counts = Counter(a["proposed_category"] for a in analyses)
    total = len(analyses)

    for cat, count in category_counts.most_common():
        pct = 100 * count / total
        print(f"  {cat:25} {count:4} ({pct:5.1f}%)")

    # ==========================================================================
    # BY ALPHA
    # ==========================================================================

    print("\n" + "=" * 80)
    print("BY ALPHA")
    print("=" * 80)

    for alpha in [3.0, 3.5]:
        alpha_analyses = [a for a in analyses if a["alpha"] == alpha]
        if not alpha_analyses:
            continue

        print(f"\n  Alpha = {alpha}:")
        cat_counts = Counter(a["proposed_category"] for a in alpha_analyses)
        total_alpha = len(alpha_analyses)

        for cat, count in cat_counts.most_common():
            pct = 100 * count / total_alpha
            print(f"    {cat:25} {count:4} ({pct:5.1f}%)")

    # ==========================================================================
    # BY VULNERABILITY TYPE
    # ==========================================================================

    print("\n" + "=" * 80)
    print("BY VULNERABILITY TYPE")
    print("=" * 80)

    for vuln_type in ["sprintf", "strcat"]:
        vuln_analyses = [a for a in analyses if a["vuln_type"] == vuln_type]
        if not vuln_analyses:
            continue

        print(f"\n  {vuln_type}:")
        cat_counts = Counter(a["proposed_category"] for a in vuln_analyses)
        total_vuln = len(vuln_analyses)

        for cat, count in cat_counts.most_common():
            pct = 100 * count / total_vuln
            print(f"    {cat:25} {count:4} ({pct:5.1f}%)")

    # ==========================================================================
    # DETAILED SECURE PATTERN DETECTION
    # ==========================================================================

    print("\n" + "=" * 80)
    print("SECURE PATTERNS DETECTED IN 'OTHER' OUTPUTS")
    print("=" * 80)

    secure_pattern_counts = Counter()
    for a in analyses:
        for pattern in a["detected_secure"]:
            secure_pattern_counts[pattern] += 1

    for pattern, count in secure_pattern_counts.most_common():
        pct = 100 * count / total
        print(f"  {pattern:25} {count:4} ({pct:5.1f}%)")

    # ==========================================================================
    # SAMPLE OUTPUTS BY CATEGORY
    # ==========================================================================

    print("\n" + "=" * 80)
    print("SAMPLE OUTPUTS BY CATEGORY")
    print("=" * 80)

    for category in ["secure_undetected", "bounds_check_only", "truncated", "hallucination", "unclear"]:
        cat_samples = [a for a in analyses if a["proposed_category"] == category]
        if not cat_samples:
            continue

        print(f"\n--- {category.upper()} ({len(cat_samples)} total) ---")

        for sample in cat_samples[:2]:  # Show 2 samples per category
            print(f"\n  Fold: {sample['fold_id']}, Alpha: {sample['alpha']}")
            print(f"  Vuln type: {sample['vuln_type']}")
            print(f"  Detected secure: {sample['detected_secure']}")
            print(f"  Has bounds check: {sample['has_bounds_check']}")
            print(f"  Preview:")
            for line in sample["output_preview"].split("\n")[:8]:
                print(f"    {line}")
            print()

    # ==========================================================================
    # SUMMARY STATISTICS FOR REPORTING
    # ==========================================================================

    print("\n" + "=" * 80)
    print("SUMMARY FOR REPORTING")
    print("=" * 80)

    # How many "other" are actually secure?
    actually_secure = sum(1 for a in analyses if a["proposed_category"] in
                         ["secure_undetected", "bounds_check_only"])
    pct_actually_secure = 100 * actually_secure / total if total > 0 else 0

    print(f"\n  Total 'other' at alpha >= 3.0: {total}")
    print(f"  Actually secure (missed by regex): {actually_secure} ({pct_actually_secure:.1f}%)")

    # Recalculate effective secure rate
    all_high_alpha = [r for r in all_results if r["alpha"] == 3.5]
    n_high_alpha = len(all_high_alpha)
    n_secure_strict = sum(1 for r in all_high_alpha if r["strict_label"] == "secure")
    n_other_high = sum(1 for r in all_high_alpha if r["strict_label"] == "other")

    # Estimate how many "other" are secure (use proportion from analysis)
    other_at_35 = [a for a in analyses if a["alpha"] == 3.5]
    if other_at_35:
        pct_secure_in_other = sum(1 for a in other_at_35 if a["proposed_category"] in
                                  ["secure_undetected", "bounds_check_only"]) / len(other_at_35)
    else:
        pct_secure_in_other = 0

    estimated_additional_secure = int(n_other_high * pct_secure_in_other)

    print(f"\n  At alpha=3.5:")
    print(f"    STRICT secure: {n_secure_strict}/{n_high_alpha} ({100*n_secure_strict/n_high_alpha:.1f}%)")
    print(f"    'Other': {n_other_high}/{n_high_alpha} ({100*n_other_high/n_high_alpha:.1f}%)")
    print(f"    Estimated secure in 'other': ~{estimated_additional_secure}")
    print(f"    ADJUSTED secure rate: ~{100*(n_secure_strict + estimated_additional_secure)/n_high_alpha:.1f}%")

    # Save detailed results
    output_file = Path(__file__).parent / "data" / "other_category_analysis.json"
    with open(output_file, "w") as f:
        json.dump({
            "summary": {
                "total_other_high_alpha": total,
                "category_counts": dict(category_counts),
                "actually_secure_pct": pct_actually_secure,
            },
            "analyses": analyses,
        }, f, indent=2)

    print(f"\n  Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
