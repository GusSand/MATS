TITLE: Probe-Then-Steer Architecture (Experiment 9b)
DATE: 2026-02-09
PARTICIPANTS: User, Claude (Developer role)
SUMMARY: Implemented and ran Experiment 9b — a probe-then-steer architecture for activation steering. Discovered that the ~100% overhead from Exp 8.5 was a benchmarking artifact from unequal token counts (baseline EOS at 32 tokens vs steered at 64). With equalized token counts, all steering methods have <5% overhead. E2E validation matched Exp 8.5 exactly (88.6% secure rate, 95.2% routing accuracy).

INITIAL PROMPT: Ok. we have a new experiment. Here it is: [Full Experiment 9 instructions for Probe-Then-Steer Architecture]

KEY DECISIONS:
- Used Developer role
- Directory: 02-09b_probe_then_steer
- Checked existing Exp 8.5 code for reusable artifacts (probe weights, steering vectors, scoring functions)
- Implemented 4 steering options: monkey-patch (A), torch.compile (B), layernorm bias (C), weight bias (D)
- Added persistent variants (no per-iteration teardown) to isolate overhead source
- Discovered and fixed token count confound: baseline hits EOS at ~32 tokens
- Used min_new_tokens=64 to equalize token counts across all benchmark conditions

FILES CHANGED:
- src/experiments/02-09b_probe_then_steer/probe_router.py — Binary probe router (BinaryProbe + ProbeRouter classes)
- src/experiments/02-09b_probe_then_steer/steered_generator.py — Hook-free steered generation (4 options)
- src/experiments/02-09b_probe_then_steer/benchmark.py — Timing benchmark (8 conditions × 50 iterations)
- src/experiments/02-09b_probe_then_steer/e2e_pipeline.py — E2E security validation (21 prompts × 10 seeds)
- src/experiments/02-09b_probe_then_steer/results/ — Benchmark and E2E result JSON files
- docs/research_journal.md — Added Experiment 9b entry
- docs/experiments/02-09_llama_neutral_probe_then_steer.md — Detailed experiment report
- docs/conversations/2026-02/2026-02-09-2300-probe-then-steer-experiment.md — This conversation log
