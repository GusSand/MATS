"""
Display sample comparisons between original and expanded prompts.
"""

import json
import random
from pathlib import Path

# Find the most recent expanded dataset
data_dir = Path(__file__).parent / "data"
jsonl_files = list(data_dir.glob("cwe787_expanded_*.jsonl"))
if not jsonl_files:
    print("No expanded dataset found!")
    exit(1)

latest_file = sorted(jsonl_files)[-1]
print(f"Reading: {latest_file.name}\n")

# Load all pairs
pairs = []
with open(latest_file) as f:
    for line in f:
        pairs.append(json.loads(line))

# Group by base_id
by_base = {}
for p in pairs:
    base = p['base_id']
    if base not in by_base:
        by_base[base] = {'original': None, 'variations': []}
    if p['category'] == 'original':
        by_base[base]['original'] = p
    else:
        by_base[base]['variations'].append(p)

# Show 5 samples (one from each of 5 different base templates)
print("=" * 80)
print("SAMPLE COMPARISONS: ORIGINAL vs EXPANDED PROMPTS")
print("=" * 80)

base_ids = list(by_base.keys())[:5]  # First 5 templates

for i, base_id in enumerate(base_ids, 1):
    data = by_base[base_id]
    original = data['original']
    variation = random.choice(data['variations'])

    print(f"\n{'='*80}")
    print(f"SAMPLE {i}: {original['name'].replace('_original', '')}")
    print(f"{'='*80}")

    print(f"\n--- ORIGINAL VULNERABLE PROMPT ---")
    print(original['vulnerable'][:500] + "..." if len(original['vulnerable']) > 500 else original['vulnerable'])

    print(f"\n--- EXPANDED VULNERABLE PROMPT ({variation['id']}) ---")
    print(variation['vulnerable'][:500] + "..." if len(variation['vulnerable']) > 500 else variation['vulnerable'])

    print(f"\n--- ORIGINAL SECURE PROMPT ---")
    print(original['secure'][:500] + "..." if len(original['secure']) > 500 else original['secure'])

    print(f"\n--- EXPANDED SECURE PROMPT ({variation['id']}) ---")
    print(variation['secure'][:500] + "..." if len(variation['secure']) > 500 else variation['secure'])

# Summary stats
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Total pairs: {len(pairs)}")
print(f"Original templates: {len([p for p in pairs if p['category'] == 'original'])}")
print(f"Generated variations: {len([p for p in pairs if p['category'] == 'expanded'])}")
print(f"Total prompts: {len(pairs) * 2}")
