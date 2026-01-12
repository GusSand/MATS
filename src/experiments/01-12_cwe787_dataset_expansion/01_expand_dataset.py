"""
CWE-787 Dataset Expansion via LLM-based Prompt Augmentation

Takes the 7 validated high-quality prompt pairs and generates variations
using GPT-4o to create ~100 total pairs for robust probe training.
"""

import os
import json
import sys
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm
from openai import OpenAI

# Add the previous experiment to path to import validated pairs
sys.path.append(os.path.abspath("../01-08_llama8b_cwe787_prompt_pairs"))
try:
    from validated_pairs import VALIDATED_PAIRS, get_pair_info
except ImportError:
    print("Error: Could not import VALIDATED_PAIRS. Check your path.")
    sys.exit(1)

# Configuration
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = f"data/cwe787_expanded_{TIMESTAMP}.jsonl"
NUM_VARIATIONS = 14  # 7 templates * 14 vars = 98 pairs + 7 originals = 105 pairs

# Initialize Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_variation_prompt(original_prompt: str, context: str) -> str:
    """
    Constructs a meta-prompt asking the LLM to rewrite the C coding task.
    Preserves the specific constraints that trigger the behavior while changing surface details.
    """
    return f"""You are a dataset augmentor for machine learning security research.

ORIGINAL C CODING PROMPT:
"{original_prompt}"

CONTEXT/INTENT:
{context}

TASK:
Generate a variation of this prompt that is semantically equivalent but syntactically different.
1. KEEP the core constraints (e.g., if it asks for "high performance" or "legacy code", keep that).
2. KEEP the functional goal (e.g., if it joins paths, the new one must join paths).
3. CHANGE variable names, specific string values, function names (if defined in prompt), and sentence structure.
4. The output must be a single string containing ONLY the new prompt text, no quotes or explanation."""


def generate_variations(pair: Dict, n: int) -> List[Dict]:
    """Generates n variations for a single prompt pair."""
    variations = []

    pair_id = pair['id']
    pair_name = pair['name']

    print(f"\nGenerating {n} variations for {pair_id} ({pair_name})...")

    for i in tqdm(range(n), desc=f"  {pair_name[:20]}"):
        try:
            # 1. Vary the Vulnerable Prompt
            vuln_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": get_variation_prompt(
                        pair['vulnerable'],
                        "This prompt is designed to elicit INSECURE code (e.g. sprintf) via framing like performance or legacy constraints."
                    )
                }],
                temperature=0.8,
                max_tokens=1000
            )
            new_vuln = vuln_response.choices[0].message.content.strip()
            # Remove surrounding quotes if present
            if new_vuln.startswith('"') and new_vuln.endswith('"'):
                new_vuln = new_vuln[1:-1]

            # 2. Vary the Secure Prompt
            sec_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": get_variation_prompt(
                        pair['secure'],
                        "This prompt is designed to elicit SECURE code (e.g. snprintf) via framing like security warnings or modern standards."
                    )
                }],
                temperature=0.8,
                max_tokens=1000
            )
            new_sec = sec_response.choices[0].message.content.strip()
            if new_sec.startswith('"') and new_sec.endswith('"'):
                new_sec = new_sec[1:-1]

            variations.append({
                "id": f"{pair_id}_var_{i+1:02d}",
                "base_id": pair_id,
                "name": f"{pair_name}_var_{i+1:02d}",
                "vulnerable": new_vuln,
                "secure": new_sec,
                "vulnerability_type": pair.get('vulnerability_type', 'sprintf'),
                "category": "expanded",
                "detection": pair.get('detection', {})
            })

        except Exception as e:
            print(f"\n  Warning: Failed to generate variation {i+1}: {e}")
            continue

    return variations


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    all_pairs = []

    print(f"Found {len(VALIDATED_PAIRS)} validated base templates.")
    print(f"Generating {NUM_VARIATIONS} variations each = ~{len(VALIDATED_PAIRS) * NUM_VARIATIONS + len(VALIDATED_PAIRS)} total pairs")
    print("-" * 60)

    for pair in VALIDATED_PAIRS:
        # Add the original with consistent structure
        original = {
            "id": f"{pair['id']}_original",
            "base_id": pair['id'],
            "name": f"{pair['name']}_original",
            "vulnerable": pair['vulnerable'],
            "secure": pair['secure'],
            "vulnerability_type": pair.get('vulnerability_type', 'sprintf'),
            "category": "original",
            "detection": pair.get('detection', {})
        }
        all_pairs.append(original)

        # Add variations
        variations = generate_variations(pair, NUM_VARIATIONS)
        all_pairs.extend(variations)

    # Save as JSONL
    with open(OUTPUT_FILE, 'w') as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")

    # Also save summary
    summary = {
        "timestamp": TIMESTAMP,
        "num_base_templates": len(VALIDATED_PAIRS),
        "num_variations_per_template": NUM_VARIATIONS,
        "total_pairs": len(all_pairs),
        "total_prompts": len(all_pairs) * 2,
        "output_file": OUTPUT_FILE,
        "base_templates": [p['id'] for p in VALIDATED_PAIRS]
    }

    summary_file = f"data/expansion_summary_{TIMESTAMP}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Success! Generated {len(all_pairs)} total pairs ({len(all_pairs) * 2} prompts)")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Summary:  {summary_file}")


if __name__ == "__main__":
    main()
