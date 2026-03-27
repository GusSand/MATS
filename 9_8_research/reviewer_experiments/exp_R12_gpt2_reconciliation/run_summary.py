#!/usr/bin/env python3
"""Run ONLY the final summary of R12."""
import sys, os, json
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_experiment import load_checkpoint, generate_summary, RESULTS_DIR
import numpy as np

results = load_checkpoint()
assert results and "section6" in results.get("completed_sections", []), "All sections must be done first"

summary = generate_summary(results)
print("\n" + summary)

results["summary"] = summary
results["completed"] = datetime.now().isoformat()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_path = os.path.join(RESULTS_DIR, f"results_{timestamp}.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2, default=lambda o:
              int(o) if isinstance(o, np.integer)
              else float(o) if isinstance(o, np.floating)
              else o.tolist() if isinstance(o, np.ndarray) else str(o))
print(f"\nResults saved to: {results_path}")
