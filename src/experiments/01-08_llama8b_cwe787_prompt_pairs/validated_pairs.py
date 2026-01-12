"""
Validated CWE-787 Prompt Pairs

This module provides easy access to the 7 validated prompt pairs that achieve
100% separation between vulnerable and secure prompts.

Usage:
    from validated_pairs import get_all_pairs, get_pair, iter_prompts

    # Get all 7 validated pairs
    pairs = get_all_pairs()

    # Get a specific pair
    pair = get_pair('time_pressure')

    # Iterate over all prompts for activation collection
    for prompt, label, pair_id in iter_prompts():
        activations = your_model.get_activations(prompt)
        # label is 'vulnerable' or 'secure'
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Iterator, Optional

# Ensure config is importable
sys.path.insert(0, str(Path(__file__).parent))

from config.cwe787_prompt_pairs import (
    PAIR_07_SPRINTF_LOG,
    PAIR_09_PATH_JOIN,
    PAIR_11_JSON,
    PAIR_12_XML,
    PAIR_16_HIGH_COMPLEXITY,
    PAIR_17_TIME_PRESSURE,
    PAIR_19_GRAPHICS,
)

# =============================================================================
# VALIDATED PAIRS (100% separation in testing)
# =============================================================================

VALIDATED_PAIRS = [
    PAIR_07_SPRINTF_LOG,
    PAIR_09_PATH_JOIN,
    PAIR_11_JSON,
    PAIR_12_XML,
    PAIR_16_HIGH_COMPLEXITY,
    PAIR_17_TIME_PRESSURE,
    PAIR_19_GRAPHICS,
]

# Short names for convenience
PAIRS_BY_SHORT_NAME = {
    'sprintf_log': PAIR_07_SPRINTF_LOG,
    'path_join': PAIR_09_PATH_JOIN,
    'json': PAIR_11_JSON,
    'xml': PAIR_12_XML,
    'high_complexity': PAIR_16_HIGH_COMPLEXITY,
    'time_pressure': PAIR_17_TIME_PRESSURE,
    'graphics': PAIR_19_GRAPHICS,
}

PAIRS_BY_ID = {p['id']: p for p in VALIDATED_PAIRS}


# =============================================================================
# PUBLIC API
# =============================================================================

def get_all_pairs() -> List[Dict]:
    """
    Get all 7 validated prompt pairs.

    Returns:
        List of pair dictionaries, each containing:
        - id: Unique identifier
        - name: Human-readable name
        - vulnerable: Prompt that elicits insecure code (sprintf)
        - secure: Prompt that elicits secure code (snprintf)
        - vulnerability_type: 'sprintf' or 'strcat'
        - detection: Regex patterns for classification
    """
    return VALIDATED_PAIRS.copy()


def get_pair(name: str) -> Dict:
    """
    Get a specific pair by short name or full ID.

    Args:
        name: Short name ('time_pressure') or full ID ('pair_17_time_pressure')

    Returns:
        Pair dictionary

    Example:
        pair = get_pair('time_pressure')
        pair = get_pair('pair_17_time_pressure')
    """
    if name in PAIRS_BY_SHORT_NAME:
        return PAIRS_BY_SHORT_NAME[name]
    if name in PAIRS_BY_ID:
        return PAIRS_BY_ID[name]

    available = list(PAIRS_BY_SHORT_NAME.keys())
    raise ValueError(f"Unknown pair: {name}. Available: {available}")


def get_prompts(pair_name: Optional[str] = None) -> List[Tuple[str, str, str]]:
    """
    Get all prompts as a flat list for batch processing.

    Args:
        pair_name: Optional - get prompts for specific pair only

    Returns:
        List of (prompt, label, pair_id) tuples
        - prompt: The prompt string
        - label: 'vulnerable' or 'secure'
        - pair_id: ID of the pair this prompt belongs to

    Example:
        prompts = get_prompts()
        for prompt, label, pair_id in prompts:
            acts = model.get_activations(prompt)
    """
    if pair_name:
        pairs = [get_pair(pair_name)]
    else:
        pairs = VALIDATED_PAIRS

    result = []
    for pair in pairs:
        result.append((pair['vulnerable'], 'vulnerable', pair['id']))
        result.append((pair['secure'], 'secure', pair['id']))

    return result


def iter_prompts(pair_name: Optional[str] = None) -> Iterator[Tuple[str, str, str]]:
    """
    Iterate over all prompts (generator version of get_prompts).

    Yields:
        (prompt, label, pair_id) tuples
    """
    yield from get_prompts(pair_name)


def get_vulnerable_prompts() -> List[Tuple[str, str]]:
    """
    Get only vulnerable prompts.

    Returns:
        List of (prompt, pair_id) tuples
    """
    return [(p['vulnerable'], p['id']) for p in VALIDATED_PAIRS]


def get_secure_prompts() -> List[Tuple[str, str]]:
    """
    Get only secure prompts.

    Returns:
        List of (prompt, pair_id) tuples
    """
    return [(p['secure'], p['id']) for p in VALIDATED_PAIRS]


def get_pair_info() -> List[Dict]:
    """
    Get metadata about all pairs (without full prompt text).

    Returns:
        List of dicts with id, name, vulnerability_type
    """
    return [
        {
            'id': p['id'],
            'name': p['name'],
            'vulnerability_type': p['vulnerability_type'],
            'category': p.get('category', ''),
        }
        for p in VALIDATED_PAIRS
    ]


# =============================================================================
# SUMMARY INFO
# =============================================================================

SUMMARY = """
Validated CWE-787 Prompt Pairs
==============================

7 pairs with 100% separation (vulnerable → insecure, secure → safe):

| Short Name      | Vulnerability | Context                    |
|-----------------|---------------|----------------------------|
| sprintf_log     | sprintf       | System logging             |
| path_join       | strcat        | File path building         |
| json            | sprintf       | JSON format + speed        |
| xml             | strcat        | XML format + simplicity    |
| high_complexity | sprintf       | Complex task + speed       |
| time_pressure   | sprintf       | "10 microseconds" framing  |
| graphics        | sprintf       | Graphics pipeline + speed  |

All pairs produce CWE-787: Out-of-bounds Write
- Vulnerable prompt → sprintf/strcat (no bounds check)
- Secure prompt → snprintf/strncat (with bounds check)
"""


def print_summary():
    """Print summary of available pairs."""
    print(SUMMARY)


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print_summary()

    print("\nExample usage:")
    print("-" * 40)

    pair = get_pair('time_pressure')
    print(f"Pair: {pair['name']}")
    print(f"Vulnerable prompt (first 100 chars):")
    print(f"  {pair['vulnerable'][:100]}...")
    print(f"\nSecure prompt (first 100 chars):")
    print(f"  {pair['secure'][:100]}...")

    print(f"\nTotal prompts: {len(get_prompts())}")
