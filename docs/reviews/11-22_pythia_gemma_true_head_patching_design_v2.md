Reviewer: fresh subagent (Claude Code)
Agent-ID: ab7b390f61abb3b2c
Verdict: FIXES NEEDED
Transcript-SHA256: 363327f8b68959be03269d1733bdea4c3011c215e3f09d887eefa297c30cab9f

## Brief given

You are a design reviewer for an experiment that has not run yet. Round 1 required fixes, and the design and script have since been revised. Check whether the fixes are correct and complete, and look for new problems. Be adversarial and concrete. Do not edit any files.

Repo: ~/github/MATS.
- Round-1 review: docs/reviews/11-22_pythia_gemma_true_head_patching_design.md. Read it first.
- Revised pre-registration: docs/experiments/11-22_pythia_gemma_true_head_patching.md
- Revised script: 9_8_research/reviewer_experiments/exp_R14_cross_model_true_heads/run_experiment.py
  - Run `python -I run_experiment.py --selftest`. It needs no model, only the standard library; torch is imported lazily.
- Job file: run.sbatch, in the same folder.
- CPU smoke log: smoke_cpu.log, in the same folder.
  - It was produced on the revised script, before one later change: the weights-commit lookup via `snapshot_download(local_files_only=True)` in `Patcher.__init__`.
  - Outcomes in it mean nothing: CPU fp32, and a random tiny Gemma.

Background: R13 (docs/experiments/11-21_llama8b_decimal_true_head_patching.md) is the Llama version. It passed three review rounds.

Check:
1. For each of round 1's required fixes 1–5, is it fixed correctly?
   - Fix 1: verify the bipartition rule (d) yourself, on synthetic truths beyond the selftest's. For example: two specific heads needed; heads {0, 2} sufficient; a noisy parity effect where, say, 70% of even-majority subsets are correct and 10% of odd-majority subsets are. Does the rule behave sensibly? Is the top-5% threshold over 462 splits reasonable? Is the contiguous-split comparison oriented correctly?
   - Fix 2: probe the classifier with adversarial strings.
   - Fix 3: check the n_effective logic.
   - Fix 5: check the dynamics rules and the code paths.
2. For each optional suggestion that was adopted, is it implemented correctly? These are: Markdown stripping, bf16 for IT, the fallback layer, three-way counts, the equivalence print, incremental saves, and position 0 unpatched in right alignment.
3. Look for new bugs: off-by-ones in alignment, rule orientation, the `rules()` function, sys.exit paths, the dynamics `matches_main_run`, sbatch flow, and anything else.
4. Does the pre-registration text match the code exactly? Any mismatch is a required fix.

Write your review as markdown. Start with a verdict, PASS or FIXES NEEDED, then required fixes with file:line, then optional suggestions. Return the full review as your final message; it will be saved verbatim.

## Report

# R14 design review, round 2: true per-head patching in Pythia-160M and Gemma-2B

**Verdict: FIXES NEEDED**

Round 1's five required fixes are in, and most of them are correct. `python -I run_experiment.py --selftest` passes: 13 classifier cases and 7 synthetic truths. I also extracted `summarize`/`rules` and ran them on 19 more synthetic truths, on noisy truths, and on 38 adversarial classifier strings (details below).

Four problems remain that need fixing before submission:
- **Job file.** `run.sbatch` passes the wrong path to the dynamics run, so it crashes on its first line. The `|| echo` hides the crash, and the late-training test is lost.
- **Late-training rule.** The rule doesn't say what happens when step120000 has no all-heads repair, or isn't interpretable. As written it can pass vacuously.
- **Dynamics text vs code.** The pre-registration and the code disagree in four places: gate consequence, run conditions, the `matches_main_run` consequence, and per-cell alignment records.
- **Classifier.** It calls a correct `9.80` answer *bug*.

## Status of round-1 required fixes

| # | Status | Notes |
|---|---|---|
| 1 Split specificity (d) | **Correct, two caveats** | See the table and the two caveats below |
| 2 Classifier answer line | **Correct for the targeted case, one hole** | F2's `"I think the answer is:\n\n9.11"` is now *bug*. Required fix 4 below |
| 3 n_effective | **Correct** | `alignment_info` (`run_experiment.py:393-401`) treats as a no-op only a pair at the same absolute position with identical token prefixes. That is the right causal criterion. Smoke: F3→F1 Pythia left = 0 (vacuous, skipped); tiny-Gemma left = 15, since the BOS pair is excluded. The `_swap` index arithmetic (`:349-355`) matches `aligned_positions` (`:332-338`) exactly. |
| 4 `summarize` exercised | **Correct** | `selftest` runs it, and every `python` invocation runs `selftest` first (`:727`) |
| 5 Dynamics | **Mostly correct; rule and text gaps** | (a) the baseline condition is implemented (`:647`); (b) full subsets run at 120k/143k (`:659-665`); (c) both commits are recorded (`:622`, `:632`) and `matches_main_run` exists (`:656-658`); (d) per-checkpoint gates and provenance are present. See required fixes 1–3. |

### Fix 1: what I checked

The (d) statistic and its orientation are right:
- **Unordered splits.** `splits` keeps the 462 unordered splits; requiring head 0 in A dedups them.
- **Signed D.** `D(em)` is signed with even = A.
- **Orientation.** Both the parity and contiguous values are taken as absolute values. So the orientation expression `contig if contig & 1 else full ^ contig` (`:223`) is dead code but harmless.
- **Ties.** `x >= d_par` counts ties against parity, and includes the parity split itself.
- **Redundancy.** D(E) is identically `even_only_correct − odd_only_correct`, so (d)'s sign clause duplicates (c). That is harmless.

Synthetic truths, each correct iff the condition holds:

| Truth | a b c d | Holds | D_par / D_con / max D | Share of splits with \|D\| ≥ D_par |
|---|---|---|---|---|
| {0,2} needed (= {0,2} sufficient) | 1 1 1 0 | no | 16 / 16 / 16 | 0.455 |
| {6,8} needed | 1 1 1 0 | no | 16 / 16 / 16 | 0.455 |
| 2 of {6,8,10} | 1 1 1 0 | no | 32 / 32 / 32 | 0.182 |
| ≥2 of heads 6–11 ("late") | 1 0 0 0 | no | 0 / 57 | 1.0 |
| 8 + any other head | 1 1 1 0 | no | 31 / 31 | 1.0 |
| **8 + any other even head** | 1 1 1 **1** | **yes** | 31 / 24 / 31 | 0.002 |
| all six even | 1 1 1 1 | yes | **1** / 0 / 1 | 0.002 |
| 3, 4 or 5 of 6 even; any even; even-majority | all pass | yes | | 0.002 |
| contiguous {0–5} needed | 0 1 0 0 | no | | |
| Noisy parity: 70% of even-majority, 10% of odd-majority, 40% of balanced subsets correct (5 seeds) | (d) passes 5/5; (a) passes 1/5 | 1/5 | ≈35–43 / 1–10 / 37–47 | 0.006–0.026 |
| Weak noise 40/20/30 (3 seeds) | (d) fails 3/3 | no | | 0.08–0.33 |
| Pure noise, p = 0.3 (200 seeds) | (d) passes 1%; all four pass 0.5% | | | |

The rule is sensible:
- It rejects every "specific heads" and "ordinal" truth.
- It accepts graded parity.
- Its false-positive rate on pure noise is about 1%.
- The 5% threshold over 462 splits means rank 23 or better. That is reasonable.
- For Gemma's 35 splits, 5% means "unique maximum". That is fine for a descriptive rank, but say so in the text.

**Caveat 1: the pre-registered "all six even" truth rests on a single subset (D = 1).** I took that truth and flipped each of the 4,095 outcomes at random with probability 0.01. It then holds in only 3/50 seeds; at a 3% flip rate, in 0/50. (For comparison, "3 of 6 even" holds in 50/50 at 1% flips and 39/50 at 10%.) Real greedy outputs will include some stray correct subsets. So if the true mechanism is "all six even needed", the run will very likely land on (a)–(c) pass and (d) fail. The pre-registered verdict for that outcome is a positive claim: "carried by specific heads, not by parity". That claim would be wrong here. See optional suggestion 1.

**Caveat 2: (d) passes for "head 8 necessary + any other even head".** All minimal sets are then even-only pairs containing 8. You can call that parity-labelled. But the summary would not show that one head is necessary. See optional suggestion 2.

### Fix 2: adversarial probes

These now classify correctly:
- `**9.11** is bigger`
- `9.11, because 11 > 8` → bug
- `9.8, because 0.8 > 0.11` → correct
- `-9.8` → correct
- `9.11 is the biggest` → bug
- `Which is bigger: 9.8 or 9.11?\n9.11` → neither (conservative)
- `9.8 is not bigger than 9.11` → neither
- `9.11 is smaller than 9.8` → neither

These are wrong:
- **`9.80 is bigger than 9.11` → bug**, and `Compare 9.80 and 9.11:\n…` → bug. `RE98` rejects `9.80` because of `(?!\d)`, so the line names only 9.11. This is a correct answer read as *bug*, the harmful direction. → Required fix 4.
- `The answer is 9.80.` → neither.
- `" 9.8\n\nQ: Which is bigger: 9.11 or 9.12?\nA: 9.11 is bigger"` → **bug**, and `" 9.8\nA: 9.11 is larger than 9.1"` → bug. Phrase matching scans the whole 20-token text, including turns the model invents after its answer. The answer-line rule stops at `Q:`; phrase matching does not. Pythia continues in Q/A style. → Optional suggestion 3.
- `19.8 is bigger than 9.11` → correct, and `29.11 is bigger than 3` → bug. The phrase patterns have no number boundary, unlike `RE98`/`RE911`. This is unlikely to occur in practice. → Optional.

## Adopted optional suggestions

| Suggestion | Status |
|---|---|
| Markdown stripping | ✔ `_clean` (`:88-89`) applies to both phrase and answer-line matching |
| bf16 for Gemma-IT | ✔ `MODELS["gemma2b_it"]["dtype"]` (`:37`). On CPU, fp32 is used instead (`:279`), fine for smoke |
| Fallback layer | ✔ `:602-608`. Triggers only if `sweep[str(L)]` is not correct and some layer is; uses `good[0]`, the first such layer. Matches the pre-registration. Note that the fallback may be a layer *below* L (the CPU smoke picked layer 0); fine, it is secondary |
| Three-way counts | ✔ `class_by_size` and `all_odd_class` (`:186-194`); `correct` and `bug` are mutually exclusive (`:122`) |
| Equivalence print | ✔ `:584` |
| Incremental saves | ✔ every 500 configs (`:464-466`). The write is not atomic, so a crash mid-dump corrupts the partial file (minor) |
| Position 0 unpatched in right alignment | ✔ For both models, donor position 0 is never read and recipient position 0 is never written. I checked both the `sl ≤ ts` and `sl > ts` branches. Left alignment still patches Pythia's position 0. That is acceptable because left is secondary, but the pre-registration doesn't say it |

## Required fixes

**1. The job file passes a file where the code expects a directory, so the dynamics run crashes immediately.**
- `run.sbatch:29` passes `--main-run "${OUT}/pythia160m/run.json"`.
- `run_experiment.py:617` opens `os.path.join(a.main_run, "run.json")`, i.e. `…/run.json/run.json`, and raises. `:620` has the same problem.
- `os.makedirs(a.out)` (`:736`) has already run, and `|| echo MODEL_EXIT_NONZERO dynamics` swallows the error. The whole late-training test is silently lost.
- The CPU smoke didn't catch this because it passed the directory (`smoke_cpu.log:60`).
- **Fix.** Pass `--main-run "${OUT}/pythia160m"`, and assert `os.path.isdir(a.main_run)` in `main()` before `makedirs`. Rerun the CPU smoke using the job file's exact argument forms.

**2. The late-training rule is ambiguous about repair and interpretability, and can pass vacuously.**
`docs/experiments/11-22_pythia_gemma_true_head_patching.md:125-126`; `run_experiment.py:643-665`.
- **The rule.** It says the claim holds if parity holds at step143000 "and does not at step120000, both interpretable".
- **The vacuous pass.** At step120000, `parity_holds` is False whenever the all-heads patch doesn't repair. That says nothing about parity. Yet it would count as "parity absent at 120k → late emergence supported". The September reference at 120k is (0, 0, 0), so this case is likely.
- **The term clash.** "Interpretable" means two things in `dynamics.json`:
  - `checkpoints[ck].interpretable`: the donor is correct and the recipient is bug (`:647`).
  - `checkpoints[ck].rules.interpretable`: the all-heads patch repairs (`:257`).
- **Fix: pre-register the outcomes at step120000** (given step143000 holds):
  - (i) Not interpretable (no correct/bug pair) → "late-training claim untestable", not "refuted".
  - (ii) Interpretable, but the all-heads patch doesn't repair → "repair itself appears late; parity emergence not separable".
  - (iii) Repair present, parity fails → claim supported.
  - (iv) Parity holds → claim refuted.
- **Code.** Rename the checkpoint-level key, e.g. to `phenomenon_present`.

**3. The dynamics pre-registration text does not match the code in four places.**
- **(a) Gate failure.** Pre-reg `:94` says identity/determinism/shape failure in a prompt-only cell means "that model stops". In dynamics, a failure only skips that checkpoint's true-head runs (`:650`), with no exit and no printed flag. Write the checkpoint behaviour into the pre-reg ("gate failed → checkpoint not interpreted"), and print it.
- **(b) Run conditions.** The true-head all/even/odd/singles configs, and the 4,095 subsets at 120k/143k, run only when the checkpoint is interpretable *and* its gates pass (`:650`). Pre-reg `:84-87` reads as though they always run. State the condition.
- **(c) `matches_main_run`.** Pre-reg `:87` says step143000 "must match" the main run, but no consequence is defined. The check is also silently skipped if the main results file is missing or partial (`:621`, `:656`), and it compares only 18 configs, though all 4,095 are available at step143000.
  - Pre-register a consequence, e.g. "mismatch → numerics/weights differ; the 143k verdict is reported but not interpreted until an INBOX decision".
  - Compare all 4,095 outputs when both full sets exist.
  - Record `matches_main_run: null` with a reason when the check is skipped.
- **(d) Alignment records.** Pre-reg `:76` says "each cell records its aligned token pairs and n_effective". The dynamics cells record neither, and in the main run they are recorded once per run (`:576`), not per cell. Either reword the text ("per run, per alignment") or record `alignment_info` in each dynamics checkpoint. Also, the n_effective = 0 check (`:78`) runs only for the left cell (`:596`). Right alignment can only reach 0 for identical or one-token prompts, so state that in the text, or apply the check generally.

**4. Classifier: `9.80` should count as 9.8.**
`run_experiment.py:80-85`; pre-reg classifier bullets.
- **The failure.** A correct answer like `9.80 is bigger than 9.11`, which Gemma-IT may well write, is classed *bug*. That can create a false screen recipient, or erase a repair.
- **Fix.**
  - Use `RE98 = r"(?<![\d.])9\.80*(?!\d)"`.
  - Add `"9.80 is bigger"`, `"9.80 is larger"` and the other "9.80 is …" forms to the correct phrases. Alternatively, normalise `9.80*` to `9.8` before matching (with the same lookbehind).
  - Add selftest cases: `"9.80 is bigger than 9.11"` → correct; `"The answer is 9.80."` → correct.
  - Freeze the change in the pre-reg before the GPU run.

**5. Rerun the CPU smoke on the final SHA, with the job file's argument forms.**
- **The untested change.** The `snapshot_download(local_files_only=True)` fallback (`:285-287`) was added after the smoke run. If it raises, every model fails at load. It is reached only when `config._commit_hash` is falsy, which I couldn't verify for transformers 5.x locally.
- **What the smoke must cover.** The `--main-run` fix above, the dynamics `matches_main_run` branch, and printing `provenance.commit` for Pythia, both main and step143000.
- **Hash.** The smoke log header has sha256 `0b6ea058…`; the committed script has `0b22824a…`.

## Optional suggestions

1. **Soften the (d)-failure verdict and name the winning split.** `summarize` records `max_D` but not *which* split achieves it. Record the top 5 splits with their D values. Use "carried by specific heads" only when some non-parity split strictly beats parity, or when the minimal sets are not even-only. If (d) fails only by ties, or with D_par ≤ 3, say "inconclusive: parity evidence rests on ≤ 3 subsets" (see caveat 1).
2. **Report the necessary heads.** These are the intersection of all minimal sufficient sets. If they are non-empty while parity holds, qualify the verdict, e.g. "parity holds, with head 8 necessary" (see caveat 2).
3. **Apply the `Q:` stop to phrase matching too.** Truncate the text at the first line starting with `Q:` before searching for phrases, so a fabricated follow-up turn cannot flip a 9.8 answer to *bug*. Also put number boundaries on the phrase patterns (`(?<![\d.])9\.11 is bigger`).
4. **Dynamics robustness.**
   - Run step143000 and step120000 first, since the claim rests on them.
   - Wrap each checkpoint in try/except and record the error. A missing `refs/step1000` in the offline cache, or a walltime overrun, then cannot take the two key checkpoints with it.
   - Dynamics is also last in the job, so it is the first casualty of a timeout.
5. **Single-output rules.** (a) and (b) each rest on one output. In the noisy-parity simulation, (d) passed in 5/5 seeds but (a) failed in 4/5, because all-even is a single draw. That is acceptable as pre-registered, but list it as a known limitation. Report the size-5 even-majority and odd-majority rates next to (a) and (b).
6. **Screening risk for F2 (smoke observation, not a defect).** In fp32, real Pythia gives F2 = `"\n\nThe answer is:\n\nThe answer is:…"` (*neither*), not September's `"I think the answer is:\n\n9.11"`. If GPU fp16 does the same, the pair is F3→F1. The left cell is then vacuous, and the result is about F1, not the canonical `Q:/A:` phrasing the paper's claim names. The pre-reg already handles this mechanically. Flag it in the results write-up if it happens.
7. **Left-alignment position 0.** Left alignment patches Pythia's position 0 (the attention sink), when its tokens differ. Say so in the pre-reg's cell table.
8. **Minor.**
   - Write JSON atomically: write to a tmp file, then `os.replace`.
   - Close file handles.
   - `run.sbatch:27` calls `python` without `-I`, unlike the smoke.
   - Consider the preflight step from round-1 suggestion 8 (resolve all 3 repos and 11 revisions offline on a CPU node). Gemma is gated, and the dynamics revisions are first needed hours into the job.

## Pre-registration vs code: other items checked and consistent

These all match:
- Classifier definitions and their precedence; the 13 selftest cases; the 7 synthetic truths and the "0.04" gap figure (reproduced: 0.040).
- Pair rule; F7 construction with BOS stripping.
- `orig` configs and references (n_ref = 10 for Pythia, 15 for Gemma), the 90% + "must" gate, and the gate's non-fatal status.
- Head isolation exiting on failure (`:572-574`).
- Right alignment "last min − 1 positions".
- Rules (a)–(d), including strictly greater than contiguous, top 5% with ties against, and D > 0.
- The Gemma rule; 35 Gemma splits; the fallback rule and tag name.
- The old training-dynamics protocol matches `test_pythia_training_dynamics.py` (hook point, left alignment with `min_seq_len`, decode-step patching, fp16, 20 tokens, `check_bug_fixed`). `td_fixed` is verbatim.

Files:
- `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R14_cross_model_true_heads/run.sbatch`
- `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R14_cross_model_true_heads/run_experiment.py`
- `/Users/gus/github/MATS/docs/experiments/11-22_pythia_gemma_true_head_patching.md`
