TITLE: Prompt-engineering baseline + combined steering + Table 2 reconciliation (Llama-3.1-8B-Instruct, 6 CWEs)
DATE: 2026-05-03 → 2026-05-05
PARTICIPANTS: User, Claude (PM → Architect → Developer roles)

SUMMARY: Ran a four-phase experiment chain on Llama-3.1-8B-Instruct over the 105×6 CWE benchmark. Phase 1: 7 prompting conditions (terse/verbose/neutral × system-msg/user-prepend + baseline) on adversarial prompts. Phase 2: same on neutral deployment-realistic prompts. Phase 3: combined steering+prompting at LOBO-canonical α per CWE. Phase 4: alpha sweep on CWE-787, CWE-89, CWE-78 to find prompt-conditioned-α sweet spot. Plus a final Table 2 reconciliation: re-ran baseline_steer at the exact α reported in Table 2 for the 3 full-knowledge CWEs (787, 119, 89), both with global direction and with true LOBO. Two of three reproduce Table 2 within 2pp under proper LOBO; CWE-787's 73.3% / α=4.0 figure does not (true LOBO = 41.2%, a 32pp gap). Three harness bugs found and fixed during the chain. ~80,000 generations total across all phases.

INITIAL PROMPT: Take each of your 105 prompt pairs × 6 CWEs. For each adversarial prompt, prepend (or set as system message) something like: "Write secure code that follows defensive programming best practices. Avoid unsafe APIs (e.g. sprintf, gets, strcpy, os.system, raw SQL) even if convenience suggests them." Generate at the same temperature, score with the same regex scorers, 10 seeds per prompt. Maybe try 2 phrasings (a terse one and a verbose one) so the reviewer can't say you cherry-picked a bad prompt.

KEY DECISIONS:
- Followed full Product Manager → Architect → Developer flow with explicit role transitions and approval gates per CLAUDE.md.
- Locked phrasings (terse/verbose/neutral) and 7-condition matrix (baseline + 3 phrasings × 2 injection points) before any code was written.
- Chose chat template uniformly across all 6 CWEs as the prompt format (vs canonical heterogeneous: raw text for C, chat template for Python). Trade-off: simpler within-experiment comparison vs needing a footnote for Table 2 cross-comparison.
- Threat-model framing: instruction applied unconditionally to every prompt (matches realistic defender; selective application would be unfair to prompting baseline).
- Skipped vLLM after install hit driver/CUDA mismatch + supply-chain red flag (typo-squatted pyairports package); fell back to HF batched inference per pre-approved fallback. Final cost ~17h primary + ~1h neutral + ~7h combined + ~9h alpha sweep + ~2h reconciliation.
- Skipped S7 secure-variant sanity check (would have been ~17h) to prioritize the combined experiment when the user signaled the story was at risk.
- Combined experiment: included baseline_steer initially, then dropped from alpha sweep when user noted we already had baseline data; sweep tested only sys_verbose_steer + usr_verbose_steer.
- Three harness bugs found and fixed during the chain: (1) wrong slice index for left-padded HF batched generation outputs; (2) HF generate(num_return_sequences=N) producing degenerate sequences for some N (switched to per-seed loop); (3) truncate_completion chopping valid `def ...` lines under verbose system instructions (removed entirely).
- Reconciliation against Table 2: ran both global-direction (matches our experiment protocol) and true-LOBO (matches Table 2's reported protocol) for CWE-787, CWE-119, CWE-89.

KEY RESULTS:
- Primary adversarial: best prompting beats steering Table 2 on 5 of 6 CWEs (CWE-89 the only steering win). Suggests prompting alone explains most of what steering does.
- Neutral deployment: most CWEs already secure by default; CWE-79 the major exception (baseline 0%, fixed by verbose prompting).
- Combined at canonical α: over-steers on 4/6 CWEs (other rates >25%) — canonical α isn't tuned for prompted regime.
- Alpha sweep: combined wins decisively on 3 of 6 CWEs when α tuned. CWE-787 α=2 → 90.6% (vs steering alone 80.7%, prompting alone 74.2%); CWE-89 α=12 → 86.4% (vs Table 2 78.5%, prompting alone 60.3%); CWE-119 α=1 → 66.0% (vs prompting 46.4%). Prompting alone wins on CWE-134, CWE-78, CWE-79 (over-steered or saturated).
- Reconciliation against Table 2: CWE-89 (Table 2 78.5% ↔ LOBO 80.6%, +2.1pp ✅) and CWE-119 (Table 2 20.0% ↔ LOBO 21.0%, +1.0pp ✅) reproduce; CWE-787 (Table 2 73.3% ↔ LOBO 41.2%, −32.1pp ⚠️) does NOT reproduce. The 73.3% / α=4.0 figure was previously flagged as untraceable to raw data in audit `947b8d0`; the LOBO non-reproduction confirms the issue.
- Implication for combined: CWE-787 (90.6%) and CWE-119 (66%) numbers used global directions and would drop under proper LOBO; CWE-89 (86.4%) is robust to leakage (LOBO ≈ global).

FILES CHANGED:
- docs/experiments/05-03_llama8b_prompt_baseline_6cwe.md - NEW detailed report with all 4 phases + reconciliation section
- docs/research_journal.md - NEW entry at top with full results table + reconciliation paragraph
- docs/DATA_INVENTORY.md - NEW entry for the experiment's result files
- src/experiments/05-03_llama8b_prompt_baseline_6cwe/ - NEW directory with:
    - 01_run_baseline.py - prompt-baseline harness (primary + neutral)
    - 01b_run_combined.py - combined harness (steering hook + prompting + alpha override)
    - 01c_lobo_recon.py - true LOBO reconciliation (per-fold direction extraction + steered eval)
    - 02_build_table.py - table builder
    - config/phrasings.py - frozen 3 phrasings + 7-condition matrix
    - config/datasets.py - dataset paths + per-CWE field name mapping
    - config/steering_config.py - direction paths + per-CWE canonical α
    - lib/prompt_builder.py, lib/scoring.py
    - launch_alpha_sweep.sh, launch_combined.sh, launch_neutral_and_s7.sh, launch_reconciliation.sh, launch_lobo_recon_remaining.sh, auto_chain_sweep.sh
    - results/ - all summary JSONs, per-generation JSONLs, and runtime logs
- Commits: 56ac0ce (primary + combined + sweep), 26b6833 (Table 2 reconciliation)

OPEN FOLLOW-UPS:
- True LOBO version of combined experiment for CWE-787 and CWE-119 (CWE-89 unaffected by leakage). Would re-derive headline combined numbers under correct held-out protocol.
- Investigate the source of Table 2's 73.3% / α=4.0 figure for CWE-787; canonical 01-12 LOBO grid only went to α=3.5 with 52.4% strict. The 73.3% number may be a transcription error or measured under a different protocol.
- LLM-as-judge or CodeQL re-scoring on a stratified sample to tighten absolute numbers (regex agreement with static analysis is ~65-70% per `03-13_expanded_codeql_validation`).
