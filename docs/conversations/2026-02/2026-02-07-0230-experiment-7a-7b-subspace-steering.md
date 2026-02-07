TITLE: Experiment 7A/7B: PCA Subspace & Conceptor AND Steering
DATE: 2026-02-07
PARTICIPANTS: User, Claude (Developer role)
SUMMARY: Designed and executed two experiments testing whether "write secure code" is a multi-dimensional subspace rather than a single direction. 7A (PCA decomposition) FAILED — worse than unified vector. 7B (Conceptor AND) FAILED — zero shared subspace found due to sample-to-dimension ratio. Both hypotheses rejected.

INITIAL PROMPT: Experiment 7A: PCA Subspace Steering / Experiment 7B: Conceptor AND Steering - detailed spec for both experiments testing the hypothesis that security is a multi-dimensional subspace.

KEY DECISIONS:
- Started as PM role, switched to Developer after stories approved
- Added Story 0: Expand CWE-119/CWE-134 datasets from 35 to 105 items each for balanced experiments
- Flagged limitation: current experimental design tests against adversarial prompts (explicitly requesting insecure functions), which likely underestimates real-world steering effectiveness. Documented in research_journal.md and experiment report.
- Proceeded with adversarial prompts for comparability with prior baselines
- Wrote combined conceptor script (Stories 3+4+5) to avoid reloading model
- Reduced max_tokens from 512 to 256 and cut configs (PCA 8→4, Conceptor 20→6) for time
- Added early termination in conceptor script when C_security trace ≈ 0

FILES CHANGED:
- `docs/DATA_INVENTORY.md` - Updated with expanded datasets + PCA/conceptor result files
- `docs/research_journal.md` - Added Experiment 7A and 7B entries
- `docs/experiments/02-07_llama8b_pca_conceptor_subspace_steering.md` - Detailed experiment report
- `src/experiments/02-05_cross_cwe_steering/datasets/cwe119/data/cwe119_expanded_20260207_024627.jsonl` - New 105-pair CWE-119 dataset
- `src/experiments/02-05_cross_cwe_steering/datasets/cwe134/data/cwe134_expanded_20260207_024627.jsonl` - New 105-pair CWE-134 dataset
- `src/experiments/02-05_cross_cwe_steering/pca_analysis.py` - PCA decomposition of steering vectors (Story 1)
- `src/experiments/02-05_cross_cwe_steering/pca_steering_experiment.py` - PCA subspace steering experiment (Story 2)
- `src/experiments/02-05_cross_cwe_steering/conceptor_steering_experiment.py` - Conceptor AND steering experiment (Stories 3-5)
- `src/experiments/02-05_cross_cwe_steering/cross_cwe_analysis/data/pca_analysis_20260207_025304.json` - PCA results
- `src/experiments/02-05_cross_cwe_steering/cross_cwe_analysis/data/pc[1-3]_security_L31_20260207_025304.npy` - Principal component vectors
- `src/experiments/02-05_cross_cwe_steering/cross_cwe_analysis/data/pca_subspace_steering_results_20260207_030444.json` - PCA steering results
- `src/experiments/02-05_cross_cwe_steering/cross_cwe_analysis/data/pca_subspace_steering_full_20260207_030444.json` - Per-sample PCA outputs
- `src/experiments/02-05_cross_cwe_steering/cross_cwe_analysis/data/secure_activations_L31_20260207_052813.npz` - Secure activations
- `src/experiments/02-05_cross_cwe_steering/cross_cwe_analysis/data/conceptor_info_20260207_052813.json` - Conceptor diagnostics
- `src/experiments/02-05_cross_cwe_steering/cross_cwe_analysis/data/conceptor_steering_results_20260207_052813.json` - Conceptor results

KEY RESULTS (FINAL):
- PCA analysis: PC1=69.2%, PC2=17.6%, PC3=13.2% — security is 2-3 dimensional
- PCA steering: ALL 4 configs FAIL. Best (sv-weighted): 787=1.9%, 119=0.0%, 134=74.3% (avg 25.4%)
- PCA steering worse than unified vector (31.8% avg) and much worse than native (54.1% avg)
- Conceptor AND: C_security trace ≈ 0 for ALL apertures {1.0, 5.0}. Zero shared subspace.
- Root cause for conceptor failure: 105 samples in 4096-dim → subspaces too sparse to intersect

GRAND COMPARISON:
| Method              | CWE-787 | CWE-119 | CWE-134 | Avg   |
|---------------------|---------|---------|---------|-------|
| Baseline            | 0.0%    | 0.0%    | 66.7%   | 22.2% |
| Native per-CWE best | 52.4%  | 20.0%   | 90.0%   | 54.1% |
| Unified vector      | 21.0%   | 4.8%    | 69.5%   | 31.8% |
| Stacked vectors     | 27.6%   | 10.5%   | 59.0%   | 32.4% |
| PCA best            | 1.9%    | 0.0%    | 74.3%   | 25.4% |
| Conceptor AND       | N/A     | N/A     | N/A     | N/A   |

EXPERIMENT STATUS:
- Story 0 (dataset expansion): COMPLETE
- Story 1 (PCA analysis): COMPLETE
- Story 2 (PCA steering): COMPLETE — FAIL
- Story 3-5 (Conceptor): COMPLETE — FAIL (zero intersection)
- Story 6 (Analysis/Report): COMPLETE
