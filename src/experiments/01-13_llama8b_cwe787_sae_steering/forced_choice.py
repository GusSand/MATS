"""
Forced-Choice Prompt Construction for Part 3B

Creates forced-choice prompts where the next token is the API decision point
(snprintf vs sprintf, or strncat vs strcat).

The prompts are constructed by:
1. Taking vulnerable prompts from the expanded dataset
2. Generating baseline completions to find the API decision point
3. Truncating just before the API call
4. Validating that both tokens have non-trivial probability
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm

from experiment_config import (
    DATASET_PATH, FORCED_CHOICE_DIR,
    FORCED_CHOICE_TOKENS, N_FORCED_CHOICE_PROMPTS, MIN_TOKEN_PROB,
    BASE_IDS,
)


@dataclass
class ForcedChoicePrompt:
    """A validated forced-choice prompt."""
    id: str
    base_id: str
    vuln_type: str
    original_prompt: str
    forced_choice_prompt: str
    safe_token: str
    unsafe_token: str
    baseline_logit_safe: float
    baseline_logit_unsafe: float
    baseline_gap: float
    baseline_prob_safe: float
    baseline_prob_unsafe: float
    is_valid: bool
    validation_note: str


def load_dataset() -> List[Dict]:
    """Load the expanded CWE-787 dataset."""
    dataset = []
    with open(DATASET_PATH) as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


def find_api_call_position(completion: str, vuln_type: str) -> Optional[int]:
    """
    Find the position just before the API call in a completion.

    Returns the index in the completion where we should truncate,
    or None if no suitable position found.
    """
    if vuln_type == "sprintf":
        # Look for sprintf or snprintf call
        patterns = [
            r'\bs(?=printf\s*\()',      # Just before 'sprintf' or 'snprintf'
            r'\bsn(?=printf\s*\()',     # After 'sn' in 'snprintf'
        ]
    elif vuln_type == "strcat":
        # Look for strcat or strncat call
        patterns = [
            r'\bstr(?=n?cat\s*\()',     # Just before 'strcat' or 'strncat'
        ]
    else:
        return None

    for pattern in patterns:
        match = re.search(pattern, completion)
        if match:
            return match.start()

    return None


def construct_forced_choice_prompt(
    original_prompt: str,
    completion: str,
    vuln_type: str,
) -> Optional[str]:
    """
    Construct a forced-choice prompt by concatenating original + completion prefix.

    The goal is to end the prompt just before the API decision point,
    so the next token is either 'printf' or 'nprintf' (for sprintf family)
    or 'cat' or 'ncat' (for strcat family).
    """
    # Find where the API call starts
    pos = find_api_call_position(completion, vuln_type)

    if pos is None:
        return None

    # Get the prefix up to the decision point
    prefix = completion[:pos]

    # For sprintf family, we want to end at 's' so next token is 'printf' or 'nprintf'
    # For strcat family, we want to end at 'str' so next token is 'cat' or 'ncat'
    if vuln_type == "sprintf":
        # Include the 's' so next is 'printf' or 'nprintf'
        prefix = completion[:pos + 1]
    elif vuln_type == "strcat":
        # Include 'str' so next is 'cat' or 'ncat'
        prefix = completion[:pos + 3]

    # Construct full forced-choice prompt
    forced_choice = original_prompt + prefix

    return forced_choice


def select_diverse_prompts(
    dataset: List[Dict],
    n_prompts: int,
) -> List[Dict]:
    """
    Select diverse prompts covering all base_ids and vuln_types.
    """
    # Group by base_id and vuln_type
    by_base_id = {base_id: [] for base_id in BASE_IDS}
    for item in dataset:
        if item['base_id'] in by_base_id:
            by_base_id[item['base_id']].append(item)

    # Calculate prompts per base_id
    prompts_per_base = n_prompts // len(BASE_IDS)
    remainder = n_prompts % len(BASE_IDS)

    selected = []
    for i, (base_id, items) in enumerate(by_base_id.items()):
        n = prompts_per_base + (1 if i < remainder else 0)
        # Randomly sample
        np.random.seed(42 + i)  # Reproducible
        indices = np.random.choice(len(items), min(n, len(items)), replace=False)
        for idx in indices:
            selected.append(items[idx])

    return selected


def create_forced_choice_prompts(
    generator,  # MultiMethodSteeringGenerator
    dataset: List[Dict] = None,
    n_prompts: int = N_FORCED_CHOICE_PROMPTS,
    output_path: Path = None,
) -> List[ForcedChoicePrompt]:
    """
    Create forced-choice prompts from the dataset.

    This function:
    1. Selects diverse prompts
    2. Generates baseline completions
    3. Finds API decision points
    4. Validates logit gaps
    5. Saves valid prompts

    Args:
        generator: MultiMethodSteeringGenerator instance
        dataset: Dataset (loaded if None)
        n_prompts: Target number of prompts
        output_path: Path to save JSONL (default: FORCED_CHOICE_DIR)

    Returns:
        List of ForcedChoicePrompt objects
    """
    if dataset is None:
        dataset = load_dataset()

    if output_path is None:
        output_path = FORCED_CHOICE_DIR / "forced_choice_prompts.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Select diverse prompts (vulnerable only)
    selected = select_diverse_prompts(dataset, n_prompts * 2)  # Select 2x to have fallbacks
    print(f"Selected {len(selected)} candidate prompts")

    # Token IDs for validation
    sprintf_tokens = FORCED_CHOICE_TOKENS["sprintf"]
    strcat_tokens = FORCED_CHOICE_TOKENS["strcat"]

    valid_prompts = []
    invalid_count = 0

    for item in tqdm(selected, desc="Creating forced-choice prompts"):
        vuln_type = item['vulnerability_type']
        original_prompt = item['vulnerable']

        # Generate baseline completion
        completion = generator.generate_with_steering(
            prompt=original_prompt,
            direction=np.zeros(4096, dtype=np.float32),
            layer=31,
            alpha=0.0,
            max_tokens=200,
        )

        # Try to find decision point
        forced_choice = construct_forced_choice_prompt(
            original_prompt, completion, vuln_type
        )

        if forced_choice is None:
            invalid_count += 1
            continue

        # Validate with logit gap
        result = generator.compute_logit_gap_for_vuln_type(
            prompt=forced_choice,
            vuln_type=vuln_type,
        )

        # Check validity
        prob_safe = result['prob_safe']
        prob_unsafe = result['prob_unsafe']
        is_valid = (prob_safe >= MIN_TOKEN_PROB and prob_unsafe >= MIN_TOKEN_PROB)

        validation_note = ""
        if not is_valid:
            if prob_safe < MIN_TOKEN_PROB:
                validation_note = f"P(safe)={prob_safe:.4f} < {MIN_TOKEN_PROB}"
            else:
                validation_note = f"P(unsafe)={prob_unsafe:.4f} < {MIN_TOKEN_PROB}"

        fc_prompt = ForcedChoicePrompt(
            id=item['id'],
            base_id=item['base_id'],
            vuln_type=vuln_type,
            original_prompt=original_prompt,
            forced_choice_prompt=forced_choice,
            safe_token=result['safe_token'],
            unsafe_token=result['unsafe_token'],
            baseline_logit_safe=result['logit_safe'],
            baseline_logit_unsafe=result['logit_unsafe'],
            baseline_gap=result['gap'],
            baseline_prob_safe=prob_safe,
            baseline_prob_unsafe=prob_unsafe,
            is_valid=is_valid,
            validation_note=validation_note,
        )

        if is_valid:
            valid_prompts.append(fc_prompt)

        # Stop when we have enough valid prompts
        if len(valid_prompts) >= n_prompts:
            break

    print(f"\nCreated {len(valid_prompts)} valid forced-choice prompts")
    print(f"Invalid/skipped: {invalid_count + (len(selected) - len(valid_prompts) - invalid_count)}")

    # Save to JSONL
    with open(output_path, 'w') as f:
        for fc in valid_prompts:
            f.write(json.dumps(asdict(fc)) + '\n')

    print(f"Saved to: {output_path}")

    return valid_prompts


def load_forced_choice_prompts(path: Path = None) -> List[ForcedChoicePrompt]:
    """Load previously created forced-choice prompts."""
    if path is None:
        path = FORCED_CHOICE_DIR / "forced_choice_prompts.jsonl"

    prompts = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            prompts.append(ForcedChoicePrompt(**data))

    return prompts


def get_forced_choice_stats(prompts: List[ForcedChoicePrompt]) -> Dict:
    """Get statistics about the forced-choice prompt set."""
    valid = [p for p in prompts if p.is_valid]

    by_vuln_type = {}
    by_base_id = {}

    for p in valid:
        by_vuln_type[p.vuln_type] = by_vuln_type.get(p.vuln_type, 0) + 1
        by_base_id[p.base_id] = by_base_id.get(p.base_id, 0) + 1

    gaps = [p.baseline_gap for p in valid]

    return {
        'n_total': len(prompts),
        'n_valid': len(valid),
        'by_vuln_type': by_vuln_type,
        'by_base_id': by_base_id,
        'gap_stats': {
            'mean': float(np.mean(gaps)),
            'std': float(np.std(gaps)),
            'min': float(np.min(gaps)),
            'max': float(np.max(gaps)),
        }
    }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing forced-choice prompt construction...")
    print("=" * 60)

    # Load dataset
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} pairs")

    # Test selection
    selected = select_diverse_prompts(dataset, 20)
    print(f"\nSelected {len(selected)} diverse prompts:")

    by_base = {}
    by_vuln = {}
    for item in selected:
        by_base[item['base_id']] = by_base.get(item['base_id'], 0) + 1
        by_vuln[item['vulnerability_type']] = by_vuln.get(item['vulnerability_type'], 0) + 1

    print(f"By base_id: {by_base}")
    print(f"By vuln_type: {by_vuln}")

    # Test find_api_call_position
    print("\n--- Testing API call detection ---")
    test_completions = [
        ("void f() { sprintf(buf, fmt); }", "sprintf", 11),
        ("void f() { snprintf(buf, sz, fmt); }", "sprintf", 11),
        ("void f() { strcat(dest, src); }", "strcat", 11),
        ("void f() { strncat(dest, src, n); }", "strcat", 11),
    ]

    for comp, vuln_type, expected in test_completions:
        pos = find_api_call_position(comp, vuln_type)
        status = "PASS" if pos == expected else f"FAIL (got {pos})"
        print(f"  {status}: {comp[:40]}... -> pos={pos}")

    print("\n" + "=" * 60)
    print("To create forced-choice prompts, run with generator:")
    print("  from steering_generator import MultiMethodSteeringGenerator")
    print("  generator = MultiMethodSteeringGenerator()")
    print("  prompts = create_forced_choice_prompts(generator)")
