Reviewer: fresh subagent (Claude Code)
Agent-ID: a2d82c379c0af48f9
Verdict: FIXES NEEDED
Transcript-SHA256: c4e5ea23cca9934c67792fcc45b93a824b464223de12326e3770b5c1030921a6

## Brief given

You are a design reviewer for an experiment that has not run yet. This is round 3. Rounds 1 and 2 required fixes, and the design and script have since been revised. Check whether the round-2 fixes are correct and complete, and look for new problems. Be adversarial and concrete. Do not edit any files.

Repo: ~/github/MATS.
- Prior reviews: docs/reviews/11-22_pythia_gemma_true_head_patching_design.md (round 1) and docs/reviews/11-22_pythia_gemma_true_head_patching_design_v2.md (round 2). Read both first.
- Revised pre-registration: docs/experiments/11-22_pythia_gemma_true_head_patching.md
- Script: 9_8_research/reviewer_experiments/exp_R14_cross_model_true_heads/run_experiment.py
  - Run `python -I run_experiment.py --selftest`. It needs no model, only the standard library; torch is imported lazily.
- Job file: run.sbatch, in the same folder.
- Smoke log: smoke_cpu.log, in the same folder. It was run on commit 400d5ee. Its outcomes are meaningless (CPU fp32, random tiny Gemma); it checks mechanics only.

Check:
1. Round 2's required fixes 1–5, each verified:
   - the `--main-run` path and the isdir check before makedirs;
   - the late-training outcome rules and `late_training_verdict`;
   - the dynamics text vs the code, covering gate behaviour, run conditions, `matches_main_run` with its consequence and null reason, and alignment records;
   - the 9.80 classifier handling;
   - whether the smoke was run on the final SHA, using the job-file argument forms.
2. The adopted optional suggestions:
   - the `parity_verdict` wording logic. Test it yourself on synthetic truths, including noisy ones, and check that each branch fires when it should;
   - necessary heads and top-5 splits;
   - the `Q:` cut and number boundaries in phrase matching;
   - dynamics ordering and try/except;
   - atomic writes.
3. New bugs anywhere, for example:
   - `parity_verdict` computes even_set from `class_by_size` keys. Is that right for n = 12 and n = 8?
   - Do the regexes behave on edge cases?
   - In dynamics, the `step143000` comparison picks `cmp` from the full subsets when `done` and a summary exist.
   - Any uncaught exception path in `main_model` that would lose data.
4. Does the pre-registration text match the code exactly? Any mismatch is a required fix.

Write your review as markdown. Start with a verdict, PASS or FIXES NEEDED, then required fixes with file:line, then optional suggestions. Return the full review as your final message; it will be saved verbatim.

## Report

# R14 design review, round 3: true per-head patching in Pythia-160M and Gemma-2B

**Verdict: FIXES NEEDED**

All five of round 2's required fixes are in, and the mechanics now work. `python -I run_experiment.py --selftest` passes: 18 classifier cases, the rule truths and 6 late-training cases. I also tested the code beyond the selftest:
- I ran `summarize`/`rules`/`parity_verdict` on 20+ synthetic truths, each with 40 seeds of 1%, 3% and 10% output-flip noise, plus graded-noise truths.
- I ran 41 adversarial classifier strings.
- I drove `dynamics()` end to end with a mock `Patcher` and a stub `torch`, in four scenarios: normal, main-run mismatch, gate failure at 143k, and a crash at 143k.

Three problems remain, one of them substantive:
1. **False "parity holds" on the round-1 alternative.** Under realistic output noise, the rule says "parity holds" for the "late even heads {6, 8, 10}" truth 18–28% of the time. This is the alternative rule (d) was added to exclude, and the error favours the paper.
2. **Late-training verdict.** It ignores a failed step143000-vs-main consistency check, and it gives the wrong reason in two cases.
3. **Text vs code.** There are small mismatches: the precedence of the verdict branches, the number of synthetic truths, the late-training cases, and the `matches_main_run` reasons.

## Status of round-2 required fixes

| # | Status | Evidence |
|---|---|---|
| 1 `--main-run` path, isdir before makedirs | **Fixed** | `run.sbatch:29` passes the directory. `run_experiment.py:840-842` checks `isdir` before `makedirs`. The smoke's negative test (`smoke_cpu.log:64-66`) shows the job refuses a file path and creates no output directory. |
| 2 Late-training rules / `late_training_verdict` | **Fixed, with gaps** | Cases (i)–(iv) and the 143k branches match the pre-reg (`:646-662`, pre-reg `:145-152`), and the six selftest cases pass. Gaps are in required fix 2. |
| 3 Dynamics text vs code | **Mostly fixed** | Gate failure prints "GATE FAILED: checkpoint not interpreted" and continues (`:724-725`). Run conditions are `phenomenon_present and gates ok` (`:726`). The check compares all 4,095 configs when both runs are complete (mock: "compared 4095 configs", True; flipping one output gives False and prints MISMATCH). Right alignment is recorded per checkpoint (`:718`). The consequence of a mismatch and the null reasons are still incomplete (required fix 2). |
| 4 `9.80` handling | **Fixed** | `RE98` and the correct-phrase pattern take `9\.80*` with boundaries (`:82-84`). `9.80`, `9.800`, `9.8000000`, `9.80 is bigger`, `9.800 is greater`, `The answer is 9.80.` are all *correct*. `9.81`, `9.110`, `19.8`, `29.11` are not 9.8 or 9.11. |
| 5 Smoke on final SHA, job-file argument forms | **Done, one branch not covered** | The script's sha256 (`3a8990b9…`) is the same at 400d5ee and at HEAD (`cdbd929` only adds the log). The log uses `python -I`, a directory for `--main-run`, and includes the negative test. Because the main smoke run used `--limit`, its results have no summary and `main_cfgs` is empty, so the comparison branch (`:743-751`) never ran in the smoke. My mock run of that branch works. The main run's `provenance.commit` is not printed in the log; the checkpoint commits are (`:61-62`). |

## Adopted optional suggestions

| Suggestion | Status |
|---|---|
| `parity_verdict` wording | Implemented. Every branch fires on a suitable noiseless truth (table below). Under noise, two branches misfire (required fix 1). |
| Necessary heads, top-5 splits | Implemented (`:229-232`, `:246`). The necessary-head set is fragile: for "8 + any other even head" with 1% noise, the "[8] necessary" qualifier survives in only 0 of the 34 seeds that pass (one stray correct subset without head 8 empties the intersection). It is descriptive, so that is acceptable; see optional 3. |
| `Q:` cut, number boundaries | Implemented as pre-registered (`_clean`, `:88-95`). Gaps that are by design but worth knowing: an invented `A:` turn without a `Q:` line (`" 9.8\nA: 9.11 is larger than 9.1"` → bug), `Question:`, and lowercase `q:` are not cut (optional 1). |
| Dynamics order and try/except | Implemented (`:681-689`): step143000 runs first, then step120000. An exception is recorded and the next checkpoint runs (mock: a crash at 143k still gives full 120k and 1000 records). But the crashed checkpoint loses everything except the error string (optional 2). |
| Atomic writes | Implemented (`dump`, `:289-293`: tmp file + `os.replace`). Every JSON write goes through it. |

### `parity_verdict`: branch check on noiseless truths

| Truth | a b c d | D_par / D_con / max | Verdict |
|---|---|---|---|
| all six even | 1111 | 1/0/1 | parity holds ✔ |
| 8 + other even | 1111 | 31/24/31 | parity holds, with heads [8] necessary ✔ |
| 5 of 6 even | 1111 | 7/0/7 | parity holds ✔ |
| head 8 alone | 1110 | | specific heads ✔ |
| {0,2} needed | 1110 | 16/16/16 | specific heads ✔ (necessary heads [0,2]) |
| 2 of {6,8,10} | 1110 | 32/32/32 | inconclusive: ties (conservative; acceptable) |
| ordinal, count ≥ 4 | 1000 | | parity not supported ✔ |
| all six even + one stray `{1}` | 1100 | 0/1/1 | parity not supported (rule (c) fails on one output) |

`even_set` (`:271`) comes from the largest `class_by_size` key: range(12) gives {0,…,10} and range(8) gives {0,2,4,6}. It is correct whenever `done` is complete, which `summarize` already requires.

## Required fixes

**1. Rule (d) lets the "late even heads" alternative pass under output noise. That is a false positive in the paper's favour.**
`run_experiment.py:258-259`, `:274-275`; pre-reg `:123-131`.

Each truth was run with 40 seeds, flipping each of the 4,095 outcomes with probability p:

| Truth | p = 0.01 | p = 0.03 | p = 0.10 |
|---|---|---|---|
| {6,8,10} needed (September `second_half_even` pattern) | **7/40 "parity holds"** | **11/40** | 7/40 |
| any 2 of {6,8,10} | **9/40** | **12/40** | **15/40** |
| head 8 alone | 1/40 | 1/40 | 1/40 |
| ≥3 of 6 even (true parity) | 40/40 | 38/40 | 30/40 |

Why it happens:
- Without noise, the even/odd split ties with the 84 splits that keep {6,8,10} on one side (rank 0.18, so it fails).
- Noise breaks the tie at random, and the even split lands in the top 23 often enough. "Ties counted against" only protects noiseless data.
- In 6 of the 7 false passes at p = 0.01, some non-parity split has a strictly larger |D| than even/odd. Example (seed 1): D_par = 9, max_D = 10, frac = 0.013.
- So "parity holds" can be declared while the data favour another split.

Fix: pick one option, pre-register it, and add a seeded-noise selftest case asserting that {6,8,10} at p = 0.01 passes in at most ~5% of seeds.
- **(a) Minimal, no power loss.** If `max_D > |D_par|`, the verdict becomes "parity criteria pass, but split {top5[0]} scores higher (D = …); not interpreted as parity". This catches about 6/7 of the false passes and makes no mechanism claim.
- **(b) Stricter.** Add "no split has |D| strictly greater than even/odd" to (d). False passes drop to 1/40, 1/40, 0/40 for {6,8,10} and 4/40, 2/40, 2/40 for 2-of-{6,8,10}. True-parity power is nearly unchanged at p ≤ 0.03 (≥3-even 38/40; even-majority 37/40). The cost is real under heavy per-subset noise: for a graded 70/10/40 parity truth, passes fall from 13 to 1 of 23 repairing seeds. State this in Limitations.
- I also tried "every even head is pivotal in some even-only subset". It removes fewer false positives than (b): 13/40 remain for 2-of-{6,8,10} at p = 0.10.

**1b. The "carried by specific heads" verdict fires on a noisy parity truth with no specific heads.**
`:279-283`.
- `minimal_even_only` is False as soon as one stray correct subset containing an odd head has no correct proper subset. Under any noise that is almost always true.
- `max_D > |D_par|` is driven by whichever split the noise favours. The top split differs from seed to seed.
- Result: graded parity truths (50/20/35 and 40/20/30) give "even-only subsets repair, but the effect is carried by specific heads, not by parity" in 5/20 and 6/16 of the repairing seeds. That is a false positive claim against parity.
- Fix: make the claim only on evidence that survives noise. For example, require the necessary heads to be a non-empty proper part of the even set, or require the same winning split to beat parity by a pre-registered margin. Otherwise word it descriptively: "even-only subsets repair, but parity is not the most specific split (top split: …)".

**2. The late-training verdict disagrees with the pre-reg and the consistency check in three ways.**
`run_experiment.py:646-662`, `:690-691`, `:715`, `:739-753`; pre-reg `:94-99`, `:145-152`.
- **(a) Mismatch not carried through.** The pre-reg says a step143000-vs-main mismatch means "the 143k verdict is reported but not interpreted". `late_training_verdict` ignores `matches_main_run`. In the mock mismatch run it printed "refuted: parity already holds at step120000" with no caveat. Fix: if `c143["matches_main_run"] is False`, append "(not interpreted: step143000 ≠ main run; INBOX)", and add a selftest case for it.
- **(b) Wrong reasons.**
  - An error at step120000 (`{"error": …}`) returns "untestable: no correct/bug pair at step120000" (`:654-655`). It should say "error".
  - A gate failure at step120000 returns "untestable: step120000 gates failed". The pre-reg does not list this case. Add both to the pre-reg's list at `:149-151`.
- **(c) `matches_main_run_reason` is wrong or missing.**
  - If step143000 has the phenomenon but its gates fail, the reason stays "not step143000" (`:715`, no branch updates it). The mock gate-failure run confirms this.
  - If `main_pair` has a None, the reason is also "not step143000".
  - After an exception the field is absent.
  - The pre-reg says "a skipped check is recorded as null with its reason". Set the reason in each branch.

**3. Other text vs code mismatches.**
- **Branch precedence.** The verdict bullets overlap (pre-reg `:133-138`): with |D| ≤ 3 and a minimal set containing an odd head, both "inconclusive" and "specific heads" apply. The code takes them in order: repair → holds → not supported → ≤3 → specific → ties (`:272-284`). Add "the first matching rule, in this order" to the pre-reg.
- **Truth count.** Pre-reg `:141` says "seven synthetic truths" and the selftest prints "7" (`:815`). The selftest runs eight (`:787-805`, counting the two Gemma truths). Fix the number in both places, or list the truths.

## Optional suggestions

1. **Classifier cut.** Also cut at a later line starting with `A:`, `Question:` or `q:` once an answer line has been found. Or report, as a descriptive count, how many outputs change class under that stricter cut. Pythia emits `A:` lines often (`smoke_cpu.log:9,11`), and `" 9.8\nA: 9.11 is larger than 9.1"` is now *bug*. `"9.11 is bigger? No, 9.8 is bigger."` → bug and `"19.8 is bigger than 9.11"` → bug are known edge cases; the second is asserted on purpose.
2. **Keep partial checkpoint records.** A crash during step143000's 4,095 subsets throws away its screen, old protocol, gates and `true_heads`, leaving only `{"error": …}` (mock: "CUDA boom"). The subsets file keeps up to the last multiple of 500. Fix: build `r` in the caller, or put `out["checkpoints"][ck] = r` before the long loop and add `r["error"]` on exception.
3. **Uncaught paths.**
   - `dynamics()` loads `run.json` without a guard (`:670`). If the Pythia main run died before its first save (`:586`), the whole dynamics job, including the old-protocol reproduction, is lost. Fall back to `main_pair = (None, None)`.
   - In `subset_cell`, an exception in `summarize`/`rules` after all 4,095 generations loses the last up-to-499 configs, since the final `dump` is after them (`:552-555`). Dump the configs first.
4. **Broaden the consistency check.** The step143000-vs-main check is skipped exactly when it would be most informative: when step143000 shows no phenomenon. The six screen outputs are always available in both runs (`run.json["screen"]`, `r["screen_outputs"]`). Compare them unconditionally.
5. **Gemma `verdict` field.** `rules()` writes the Pythia parity wording into Gemma's `verdict`, e.g. "parity not supported", while Gemma's pre-registered rule is `gemma_parity_independent`. Rename or omit it for `family == "gemma"`, so a write-up doesn't quote it.
6. **Near-necessary heads.** Report heads present in ≥ 90% of minimal sets as well as the exact intersection, which one stray output empties.
7. **Smoke log.**
   - The log is an excerpt: no "Loading" lines, and no selftest line for the dynamics call. Say so in its header.
   - It does not show the main run's commit.
   - A mock test like mine (fake `Patcher`, stub `torch`) would cover the `matches_main_run`, mismatch, gate-fail and crash branches without a model. Consider adding it to `--selftest`.
8. **Repeated from earlier rounds, still open.** Offline preflight of the 3 repos and 11 revisions on a CPU node; `R14_GIT_SHA` checked against `git rev-parse HEAD`; `json.load(open(...))` handles left unclosed.

## Pre-registration vs code: checked and consistent

These match:
- Classifier text and the 18 cases.
- Pair rule; F7 construction.
- Dtypes: fp16 for the original two models, bf16 for Gemma-IT.
- `orig` cell configs and the gate.
- Right and left alignment, including position 0.
- Alignment records per run, and per checkpoint for right alignment.
- Vacuous left cell; fallback layer.
- Dynamics checkpoint order and try/except; run conditions; full subsets at 120k/143k.
- All 4,095 compared when both runs are complete.
- Rules (a)–(d); necessary-head qualifier; top-5 and minimal sets reported.
- Gemma rule; late-training cases (i)–(iv).
- `run.sbatch` (`python -I`, directory argument, `|| echo` continuation).

Files:
- `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R14_cross_model_true_heads/run_experiment.py`
- `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R14_cross_model_true_heads/run.sbatch`
- `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R14_cross_model_true_heads/smoke_cpu.log`
- `/Users/gus/github/MATS/docs/experiments/11-22_pythia_gemma_true_head_patching.md`
- Simulation and mock scripts (scratch, not in the repo): `/private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/r14v3/`
