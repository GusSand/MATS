#!/usr/bin/env python3
"""Run ONLY Section 5 of R12: Test Hoang's 5-Head Circuit on GPT-2-Small."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_experiment import load_checkpoint, save_checkpoint, section5_hoang_circuit

results = load_checkpoint()
assert results and "section4" in results.get("completed_sections", []), "Section 4 must be done first"

completed = set(results["completed_sections"])
if "section5" in completed:
    print("Section 5 already completed, skipping.")
    sys.exit(0)

print("Running Section 5: Test Hoang's 5-Head Circuit on GPT-2-Small")
section5_hoang_circuit(results)
results["completed_sections"].append("section5")
save_checkpoint(results)
print("\n  ✓ Section 5 completed and checkpointed.")
