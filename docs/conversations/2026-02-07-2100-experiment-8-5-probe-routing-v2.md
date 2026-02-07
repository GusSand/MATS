TITLE: Experiment 8.5 — Neutral-Trained CWE Router & 2-Tier Deployment
DATE: 2026-02-07
PARTICIPANTS: User, Claude (Developer role)
SUMMARY: Implemented and ran Experiment 8.5 with three parts: (A) retrained CWE-type probes on neutral/mixed data fixing distribution shift from 66.7% → 95.2%, (B) validated 2-tier binary routing architecture (88.1% avg), (C) ran full E2E pipeline (88.6% secure, 101.8% overhead). Found data leakage bug in mixed+augmented method.

INITIAL PROMPT: We are doing a new experiment: Experiment 8.5: Neutral-Trained CWE Router & 2-Tier Deployment Architecture [full multi-part specification with Parts A, B, C]

KEY DECISIONS:
- Assigned Developer role
- Reused code patterns from Phase 3/4 scripts (scoring classifiers, model loader, activation collection)
- Used 5 prefix instruction variants for data augmentation
- Layers [0, 8, 16, 24, 31] for probe layer sweep
- Binary probe saved with adv-trained weights (not mixed, due to data leakage)
- Used LOO for 21-sample neutral set, LOBO for augmented (group by base prompt)

FILES CHANGED:
- `src/experiments/02-08_probe_routing_v2/01_probe_retraining.py` - Created: Part A probe retraining script (fixed data leakage bug in mixed+augmented method)
- `src/experiments/02-08_probe_routing_v2/02_two_tier_analysis.py` - Created: Part B 2-tier strategy analysis
- `src/experiments/02-08_probe_routing_v2/03_e2e_pipeline.py` - Created: Part C E2E pipeline + timing
- `src/experiments/02-08_probe_routing_v2/data/` - Created: activation NPYs, probe weights, labels
- `src/experiments/02-08_probe_routing_v2/results/` - Created: all result JSONs
- `docs/research_journal.md` - Updated: added Experiment 8.5 summary
- `docs/experiments/02-07_llama8b_neutral_probe_routing_v2.md` - Created: detailed experiment report
- `docs/DATA_INVENTORY.md` - Updated: added Experiment 8.5 datasets
- `docs/conversations/2026-02-07-2100-experiment-8-5-probe-routing-v2.md` - Created: this conversation

KEY RESULTS:
- Part A: Neutral LOO at L16 achieves 95.2% 3-way routing (vs 66.7% from adv-trained at L31)
- Part A: Binary LOO at L16 achieves 100% (format-string vs buffer)
- Part A: Layer 16 beats Layer 31 for probing across all methods
- Part A: Mixed+Augmented method had data leakage (100% at all layers = bug, not breakthrough)
- Part B: 2-Tier costs 5.7pp avg vs perfect 3-way (88.1% vs 93.8%)
- Part B: CWE-119 takes 17.1pp hit (gets CWE-787 vector instead of native)
- Part C: E2E pipeline achieves 88.6% overall, 95.2% routing accuracy
- Part C: 1 misrouted prompt: neutral_787_05 → format_string (conf=84.7%)
- Part C: Overhead is 101.8% (2x slowdown from hook mechanism), not expected <5%
