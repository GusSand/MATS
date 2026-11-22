Reviewer: fresh subagent (Claude Code)
Agent-ID: ad51f2bf3bbf6d590
Verdict: FIXES NEEDED
Transcript-SHA256: 139637b7605e81ce5599c2754f7315d968b8d186a4a3422ce3988c2cdca4e429

## Brief given

You are a design reviewer for an experiment that has not run yet. Judge whether it will answer its question. Be adversarial and concrete. Do not edit any files.

Repo: ~/github/MATS.
- Pre-registration: docs/experiments/11-22_pythia_gemma_true_head_patching.md
- Script: 9_8_research/reviewer_experiments/exp_R14_cross_model_true_heads/run_experiment.py
- Job file: 9_8_research/reviewer_experiments/exp_R14_cross_model_true_heads/run.sbatch
- CPU smoke log: 9_8_research/reviewer_experiments/exp_R14_cross_model_true_heads/smoke_cpu.log. It was produced before the last change, which added `matched_pairs` to `summarize`.

Context:
- R14 extends R13: docs/experiments/11-21_llama8b_decimal_true_head_patching.md, with script exp_R13_true_head_patching/run_experiment.py. Its design passed three review rounds; the reviews are in docs/reviews/.
- The audit motivating R14 is docs/reviews/11-22_cross_model_head_scripts_audit.md.
- The original scripts are 9_8_research/cross_model_validation/test_pythia_160m.py and test_gemma_2b.py, and 9_8_research/training_dynamics_analysis/scripts/phase1_training_dynamics/test_pythia_training_dynamics.py. Their result JSONs are next to them, or under training_dynamics_analysis/results/.
- The paper claims at stake are in ~/github/phd_thesis/preamble/surgeon_dilemma/sections/02_study.tex, around lines 285–320.

Check at least the following:
1. Does pre-o_proj / pre-dense hooking isolate heads correctly in GPT-NeoX (Pythia) and Gemma under transformers 5.x? Consider the dense bias, Gemma's MQA, and head_dim vs hidden. Check the attribute names against the installed library if you can. The library is not installed locally, so read the transformers source from GitHub or pip-download it into a temp dir if needed.
2. Does the `orig` cell faithfully reproduce the original protocol, including hook point, alignment, block width, decode-step behaviour, prompts, layers, classifier and dtype? Are the reproduction references and the gate correct, given the original JSONs?
3. Are the phenomenon screen and its mechanical pair rule sound? Is there any way the rule picks a degenerate pair, or depends on outcomes in a way that biases the verdict? Is F7 built correctly for gemma-2b-it, with the BOS handling right?
4. Classifier: is the extended `is_bug` sensible? Is "neither" handled correctly in the rules?
5. Do the decision rules test the paper's claims? Is the shift-matched parity rule reasonable? Look at the Gemma rule too.
6. Gates and failure paths: does a failure in one model stop or skip correctly? Is anything not persisted? Are run dirs single-use? Is provenance recorded?
7. The training-dynamics mode: correctness, and whether it is meaningful.
8. Anything that would make a result uninterpretable, or a cheap addition that would make it much more informative.

Write your review as markdown. Start with a verdict, PASS or FIXES NEEDED. Then list required fixes, numbered and each with file:line, then optional suggestions. Return the full review as your final message. It will be saved verbatim.

## Report

# R14 design review: true per-head patching in Pythia-160M and Gemma-2B

**Verdict: FIXES NEEDED**

Most of the machinery is correct. I read the transformers 5.15.1 source, the version R13 ran on, and checked these points against it:

- **Hook points.** `GPTNeoXAttention` reshapes the attention output to `(B, T, H·64)` in head-major order and then calls `self.dense(attn_output)` with it as a positional argument (`modeling_gpt_neox.py:233-234`). Gemma does the same with `self.o_proj` (`modeling_gemma.py:290-291`). So `args[0]` in a pre-hook on `dense`/`o_proj` holds the per-head z.
- **Dense bias.** Pythia's `dense` has a bias, and pre-dense patching leaves it alone.
- **Gemma MQA.** Gemma's 8 query-head slices of width 256 are true per-query-head outputs that share one K/V head.
- **BOS.** The v5 GPT-NeoX tokenizer defaults to `add_bos_token=False`, so `skip0` is False for Pythia and True for Gemma, as intended.
- **`orig` cell.** It faithfully reproduces the old hook point, left alignment, block width (`hidden/n_heads`), decode-step writes of donor position 0, prompts, layers, fp16 and classifier.
- **References.** `ORIG_REFS` and `TD_REF` match `pythia_160m_validation_20250926_174151.json`, `gemma_2b_validation_20250926_174510.json` and `pythia_training_dynamics_20250926_191344.json`.

The design still has four problems that would make the Pythia verdict uninterpretable, or a false positive in the paper's favour. The decision rules cannot tell parity from a single head. The classifier drops Pythia's own bug answer, which moves the pair off F2. The left-aligned cell is likely a no-op. And `summarize` has never run.

## Required fixes

**1. The Pythia parity rule passes for a single-head effect. Add a parity-specificity criterion.**
`docs/experiments/11-22_pythia_gemma_true_head_patching.md:96-100`; `run_experiment.py:333-340`.

I ran `summarize` on synthetic outcomes (script extracted verbatim, CPU):

| Synthetic truth | all_even | all_odd | matched pairs E¬O / O¬E | Rule (a)(b)(c) |
|---|---|---|---|---|
| Head 8 alone suffices | ✔ | ✘ | 32 / 0 | **"parity holds"** |
| Needs all 6 even heads | ✔ | ✘ | 1 / 0 | "parity holds" |
| Needs {6, 8, 10} (the old `second_half_even` result) | ✔ | ✘ | >0 / 0 | **"parity holds"** |

- **Rule (c) is not a matched comparison.** S ↦ S+1 is a bijection between even-only and odd-only subsets of each size. So E¬O − O¬E always equals #correct even-only − #correct odd-only, whatever the pairing. The "shift-matched" wording adds nothing; say plainly what the rule counts.
- **What the rule tests.** In practice it asks whether some even head is necessary or sufficient and no odd head is. The paper's claim is "parity-labeled head specialization" (`02_study.tex:289-293`).
- **The live alternative is ordinal.** The September data already has `second_half_even` = 1 and `first_half_even` = 0. That is what a "late heads" effect looks like.
- **Fix: add rule (d), computed on the 4,095 outputs already collected (no extra compute).**
  - For every balanced bipartition (A, B) of the 12 heads (462 of them), compute D(A, B) = #correct subsets ⊆ A − #correct subsets ⊆ B, oriented so that D ≥ 0.
  - Parity holds only if the even/odd split has D strictly greater than the contiguous split {0–5} | {6–11}, and ranks in the top 5% of the 462 splits, counting ties against parity.
  - If it fails, the verdict is: "even-only subsets repair, but the effect is carried by specific heads, not by parity."
  - Also report the minimal sufficient sets (correct subsets none of whose proper subsets is correct).
- **Gemma.** Report the same rank there as a descriptive number.

**2. The classifier labels Pythia's canonical bug answer "neither". This moves the screen pair off F2.**
`run_experiment.py:88-110`; pre-reg `:45-48`.

- **What happens to F2.** The September F2 output is `"\n\nI think the answer is:\n\n9.11\n\n…"`. The first stripped line is `"I think the answer is:"`, so both `is_correct` and the extended `is_bug` return False, and F2 is "neither". The CPU smoke already shows `F2=neither` on the real Pythia weights (`smoke_cpu.log:4`).
- **What happens to the pair.** It becomes F3 → F1, which is not the prompt the claim is about. The paper says the even-head intervention "works only on the exact canonical phrasing `Q: …\nA:`" (`02_study.tex:300-302`).
- **The same blind spot hits correct answers.** The old `second_half_even` output `"I think the answer is:\n\n9.8"` is also "neither".
- **Fix.** For both `is_correct` and `is_bug`, decide from the first non-empty line that mentions exactly one of {9.8, 9.11}. Stop scanning at a line that mentions both, or that starts with `Q:`, so a repeated question cannot count. Freeze this in the pre-registration before any GPU run.
- **Effect on the pair.** With this fix F2 is "bug", and the pair becomes F3 → F2 if F3 stays correct on GPU. That pair is ideal. Tokenized with Pythia's tokenizer, right alignment maps F3's 15 tokens onto the last 15 of F2's 17. Only `Which`/`ĠWhich` and `Answer`/`A` differ, and both prompts end in `Ċ … :`.

**3. The left-aligned cell is an identity patch whenever one format is a token prefix of the other. Detect it and report it.**
`run_experiment.py:213-215`, `:437`; pre-reg `:55-57`, `:68-69`.

- **Pythia tokenization (checked with the HF `tokenizer.json`).** F1 is `['Which','Ġis','Ġbigger',':','Ġ9','.','8','Ġor','Ġ9','.','11','?']`. F3 is exactly F1 followed by `['Ċ','Answer',':']`.
- **Consequence.** With the smoke pair F3 → F1, left alignment copies donor positions 0–11, which are F1's own causal activations. The `pre_wo/prefill` cell then returns "bug" for all 4,095 subsets. That would read as a failed replication, but it is a no-op. Gemma's SentencePiece very likely has the same F1 ⊂ F3 prefix.
- **Fix, part 1.** For each cell, record the aligned donor/recipient token pairs. Also record n_effective: the number of patched positions not inside a shared prefix at the same absolute position (left alignment), or simply all patched positions (right alignment).
- **Fix, part 2.** Pre-register that a cell with n_effective = 0 is "vacuous, not reported as a replication".
- **Fix, part 3.** Optionally, make the pair rule skip donor/recipient combinations where one is a token prefix of the other.

**4. `summarize` has never run; test it before submitting.**
`run_experiment.py:310-343`, `:445`.

- **Why it never ran.** Under `--limit`, `summ = … if not a.limit else None`, so all three smoke runs printed `summary: null`. The full `run_subsets` path (`:346-355`) was never exercised either. The smoke log also predates `matched_pairs`.
- **What I checked.** My synthetic run (fix 1) shows the function executes and the key format is consistent for n=12 and n=8.
- **Fix.** Add a `--selftest` (or a test file) that runs `summarize`, and the new rule (d), on synthetic `done` dicts: all-six-even, head-8-sufficient, {6,8,10}-needed, any-4 for Gemma. Assert the verdicts. Then redo the CPU smoke on the committed SHA.

**5. The training-dynamics run has no rule for which checkpoints are interpretable, and it uses a different revision from the original.**
`run_experiment.py:481-516`, `:41-42`, `:487`; pre-reg `:70-73`, `:102-103`.

- **Fix (a): baseline condition.** At most checkpoints the main pair's donor will not be correct, or its recipient will not be buggy. The `main_pair_baseline` check is recorded but has no consequence. Pre-register: a checkpoint's true-head results count only if, at that checkpoint, the donor is *correct* and the recipient is *bug*. Otherwise it is "no phenomenon at this checkpoint".
- **Fix (b): rule mismatch.** Only all/even/odd/singles run per checkpoint, so rules (c) and (d) cannot be evaluated there.
  - Either run all 4,095 subsets at step120000 and step143000. That is about 14 min, and those are the two checkpoints the "emerges in the final 15%" claim (`02_study.tex:295-299`) rests on.
  - Or state that the checkpoint verdict uses only (a) and (b), and is weaker for that reason.
- **Fix (c): revision mismatch.** The original script mapped step143000 to revision `"main"` (`test_pythia_training_dynamics.py:44`); R14 loads `"step143000"`. Record both commit hashes. Use the overlap as a free check: step143000's all/even/odd/singles must equal the corresponding configs in the main run, or the weights or numerics differ.
- **Fix (d): gates.** Add the identity gate (own-activation patch equals unpatched) for each checkpoint. Add provenance (transformers/torch versions, GPU UUID, hostname) to `dynamics.json`; currently it holds only the SHAs.

## Things I checked that are fine

- **Head isolation.** `head_isolation` (`:277-284`) only tests that `_swap` writes slice 1. Given the source check above, that is enough, but the gate is close to tautological. A stronger check would recompute head 1's z from eager attention weights (`softmax(QKᵀ)V`, query head 1, the shared KV head) and compare it with the captured pre-o_proj slice.
- **Reproduction gate.** The references are correct. `sweep_L6` duplicates `all_even` and `sweep_L8` duplicates `all_even_4`, so each "must" config is counted twice. That only makes the gate stricter. Gemma's `random_4` equals `all_odd_4` and is correctly omitted.
- **The smoke-run gate failure is expected.** In the smoke, the Pythia gate failed on `all_even` under CPU fp32. That says nothing about fp16 on GPU. It does show that the gate is sensitive to numerics, so record `model.config._attn_implementation` in provenance: sdpa vs eager is the most likely source of drift from September.
- **Failure paths.** `|| echo MODEL_EXIT_NONZERO` lets the job continue past a failed model. Run directories are single-use (`exist_ok=False` under a job-ID directory). Screen, gates and orig rows are saved before any fatal exit.
- **"Neither" in the rules.** Counting "neither" as "not correct" is acceptable for repair claims. See optional suggestion 6 for reporting the three-way split.
- **Gemma rule.** "even-4 and odd-4 both correct" tests the paper's sentence as written (`02_study.tex:310-314`). Fine as primary; add the rank from fix 1 as descriptive.
- **F7 construction.** Stripping the `<bos>` string and letting the tokenizer re-add it is right, and `skip0` then protects BOS.

## Optional suggestions

1. **If all heads fail at the original layer, rerun at the first layer that works.** Layers 6 and 8 were picked under the buggy protocol. If the all-heads patch fails there but the sweep finds a layer where it succeeds, rerun the subset cells at the first such layer, chosen by a pre-registered rule. This costs about 7 min for Pythia, needs the sweep moved before the subset cells, and should be reported as secondary. Without it, "Layer-L repair fails" does not answer whether parity exists at the layer where repair does happen.
2. **Strip Markdown before matching.** Remove `*`, `` ` `` and `_` before pattern matching. Gemma-IT tends to write `**9.11** is bigger than 9.8`: no phrase matches, the first line has both numbers, so it is classed "neither". This can silently turn the IT screen into "no phenomenon".
3. **Run Gemma-IT in bf16.** It has no reproduction constraint, and bf16 is its native dtype; fp16 Gemma activations can overflow. At minimum, print the screen outputs to the log, not just the classes.
4. **Print the all-heads equivalence result (`:452-456`) to the log.** It is the gate that says "escalate", but it is only written to JSON.
5. **Save subset outputs incrementally.** For example, flush `results_{tag}.json` every 500 configs, so a crash at config 4,000 does not lose 15 minutes of outputs.
6. **Add a three-way class count.** In `summarize`, count correct/bug/neither per size and for all-odd. "All-odd fails" means something different when the output is garbage ("neither") than when it is "9.11".
7. **Treat Pythia position 0 like a BOS.** With right alignment and T_recipient ≤ T_donor, Pythia's position 0 (its de facto attention sink) gets a mid-prompt donor activation. Skipping position 0 for Pythia too would make the two models comparable.
8. **Add a preflight step.** Have it resolve all 3 repos and 11 revisions with `HF_HUB_OFFLINE=1` on a CPU node before submitting (the project CLAUDE.md asks for this). Gemma is gated, and a missing `refs/stepN` would only show up hours into the job.
9. **Check `R14_GIT_SHA`.** If a git checkout is present, compare it with `git rev-parse HEAD` and refuse a dirty tree. Otherwise the env var is unchecked, and only `script_sha256` ties the run to a commit.
10. **Paper text, independent of R14.** `02_study.tex:291` says "all 16 even heads"; Pythia-160M has 6. And `02_study.tex:311` attributes "~5% on the canonical prompt" to Gemma-2B, while the head result is from base Gemma, whose Q&A bug rate is 100%.

Files referenced:
- `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R14_cross_model_true_heads/run_experiment.py`
- `/Users/gus/github/MATS/docs/experiments/11-22_pythia_gemma_true_head_patching.md`
- `/Users/gus/github/MATS/9_8_research/cross_model_validation/test_pythia_160m.py`
- `/Users/gus/github/MATS/9_8_research/training_dynamics_analysis/scripts/phase1_training_dynamics/test_pythia_training_dynamics.py`
- `/Users/gus/github/phd_thesis/preamble/surgeon_dilemma/sections/02_study.tex`
