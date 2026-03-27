#!/usr/bin/env python3
"""Run ONLY Section 4 of R12: Evaluate the Evaluation."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_experiment import load_checkpoint, save_checkpoint, section4_evaluation_comparison

results = load_checkpoint()
assert results and "section3" in results.get("completed_sections", []), "Section 3 must be done first"

completed = set(results["completed_sections"])
if "section4" in completed:
    print("Section 4 already completed, skipping.")
    sys.exit(0)

print("Running Section 4: Evaluate the Evaluation")
section4_evaluation_comparison(results)
results["completed_sections"].append("section4")
save_checkpoint(results)
print("\n  ✓ Section 4 completed and checkpointed.")
