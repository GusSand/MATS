#!/usr/bin/env python3
"""Run ONLY Section 3 of R12: Test All GPT-2 Variants."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_experiment import load_checkpoint, save_checkpoint, section3_all_variants

results = load_checkpoint()
assert results and "section1" in results.get("completed_sections", []), "Sections 1-2 must be done first"

completed = set(results["completed_sections"])
if "section3" in completed:
    print("Section 3 already completed, skipping.")
    sys.exit(0)

print("Running Section 3: Test All GPT-2 Variants")
section3_all_variants(results)
results["completed_sections"].append("section3")
save_checkpoint(results)
print("\n  ✓ Section 3 completed and checkpointed.")
