Reviewer: fresh subagent (Claude Code)
Agent-ID: a75ebbcd07fc23a2b
Verdict: VERIFIED-WITH-FLAGS
Transcript-SHA256: a03b7df4a2b3b821b3c9d05d344c9a4567083988e49194b9aae8ab19df8c173f

## Brief given

You are an independent verifier for a finished experiment. Recompute everything from the raw files, and trust no summary. Do not edit repo files.

Repo: ~/github/MATS.
- Read the pre-registration first: docs/experiments/11-22_mistral7b_true_head_patching.md. It has the claims S and M1–M7 and the decision rules. The paper claims under test are in ~/github/phd_thesis/preamble/surgeon_dilemma/sections/02_study.tex, lines 224–285.
- Script: 9_8_research/reviewer_experiments/exp_R15_mistral_true_heads/run_experiment.py. It imports R14's script. `decide()` and `make_classifier` are pure; import the module (numpy, torch lazy) and recompute.
- Raw results (Slurm job 24092912, A100): 9_8_research/reviewer_experiments/exp_R15_mistral_true_heads/results/r15_24092912/ — run.json, results.json and the job log.

Tasks:
1. **Provenance.** Check both script sha256s against run.json and the expected values, the snapshot commit, the GPU, and that git_sha e1c139e exists.
2. **Classifications.** Reclassify every output from raw text: the screen, the layer sweep, the 1,024 single heads, the 524 parity configs, the controls, reverse, and all 100×4 generalization outputs plus the repairs.
   - Flag every output where the classifier's label is wrong in substance. Watch for:
     - "9.8 is smaller than 9.11" (says_less) counted as *neither*;
     - long reasoning truncated at 64 tokens;
     - answers that state both numbers.
   - Report how many outputs hit max_new and how many say_less, by section.
   - In particular, check the screen. Chat is *bug*. Comparative is *neither*, which is what fails S. chat_comparative and chat_simple say "9.8 is smaller than 9.11", the bug in substance, yet are labelled *neither*.
3. **Decisions.** Recompute `decide()` and confirm every value. Check the Δ-testability condition: chat baseline Δ = +3.75 although chat generates the bug. Explain why. The reply starts "The number 9.8 is smaller…", not " 9.8"/" 9.11". Assess whether Δ as defined is meaningful for chat at all.
4. **Every claim.** For each of S and M1–M7, state the pre-registered verdict, give the supporting numbers, and add any caveat from task 2. Key points:
   - M1a: only layer 7 repairs. The draft named layers 2, 17, 13, 10 and 11.
   - M5: the five-layer transplant repairs, but 7 of 10 random 5-layer sets also repair, so it is not specific.
   - M2/M3/M4: no single head and no even subset repairs at layers 2 or 17. H22 is not special.
   - M6: the reverse transplant leaves simple correct.
   - M7: bug counts 6/2/0/2, and 3 of 4 eligible pairs repaired.
5. **Gates and sanity.** Check the gates and the sanity transplant, and anything odd.

Report as markdown. Put the verdict first: VERIFIED, VERIFIED-WITH-FLAGS, or DISCREPANCIES. Return the full report as your final message.

## Report

# R15 verification report (Mistral-7B true-head patching, Slurm 24092912)

## Verdict: **VERIFIED-WITH-FLAGS**

I recomputed every stored label and decision from the raw files and found no mismatches: all 2,073 outputs reclassify to the stored label, `says_less` matches on every output, and `decide()` on `results.json` equals `run.json["decisions"]` exactly. `decisions_before_M7` also matches, as does the `[decisions]` line in the job log. Every pre-registered verdict stands as computed.

The flags are about substance:
- The classifier labels 187 patched outputs *neither* when they say "9.8 is smaller than 9.11", which is the bug.
- One reverse output is a false *bug*.
- In M7, one chat pair is a false *bug* (8.5 vs 8.12) and one real bug (9.7 vs 9.12) is labelled *neither*. M7 is therefore really "3 of 3 genuine pairs repaired, 1 genuine pair not tested", not "3 of 4".
- The comparative format is mostly unresolved at 64 tokens.

None of this changes S or M1–M6.

Scratch scripts (recomputation): `/private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/verify.py` and `.../scratchpad/subst.py`.

---

## 1. Provenance: OK

| Item | Expected | Found |
|---|---|---|
| R15 sha256 | `1fba89a4…` (exp file) | run.json `r15_script_sha256` = `1fba89a41bec…8064a39`. The local file and `git show e1c139e:` both hash to this. |
| R14 sha256 | `20819e06…` | run.json `r14_script_sha256` = `20819e0620464…20fc2`. Local file and `git show e1c139e:` both match. |
| git_sha | e1c139e | Exists: `e1c139eadae8… "R15: design review PASS (round 4)"`, 2026-11-22 11:03. Job started at 11:07. |
| Snapshot | `63a8b081…` | `revision` and `commit` both equal `63a8b081895390a26e140280378bc85ec8bce07a` |
| GPU | A100 | `NVIDIA A100-SXM4-80GB`, host ga006, bf16, sdpa, transformers 5.15.1 |
| Chat template | — | String path and `tokenize=True` give identical ids (23 tokens) |

Minor issue: run.json's generic `provenance.script_sha256` field holds **R14's** hash, because `r14.provenance` hashes its own file. The explicit `r15_script_sha256` field is correct. It's a label hazard only.

## 2. Classifications

### Counts by section (recomputed)

| Section | n | correct | bug | neither | hit max_new | says_less | says_less & *neither* | substantive mislabels |
|---|---|---|---|---|---|---|---|---|
| screen | 7 | 3 | 1 | 3 | 6 | 3 | 2 | 2 (chat_comparative, chat_simple → bug) |
| sanity (all layers) | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| layer sweep | 32 | 1 (L7) | 28 | 3 | 28 | 31 | 3 | 3 (L2, L6, L14 → bug) |
| single heads | 1024 | 0 | 968 | 56 | 1015 | **1024** | 56 | 56 → bug |
| parity | 524 | 0 | 416 | 108 | 495 | **524** | 108 | 108 → bug |
| five-layer | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| random 5-layer | 10 | 7 | 3 | 0 | 9 | 3 | 0 | 0 |
| head × five layers | 32 | 0 | 22 | 10 | 31 | 32 | 10 | 10 → bug |
| head × all layers | 32 | 0 | 22 | 10 | 31 | 32 | 10 | 10 → bug |
| reverse | 6 | 5 | 1 | 0 | 3 | 0 | 0 | 1 (L13 bug → correct) |
| gen chat | 100 | 93 | 6 | 1 | 54 | 1 | 1 | 2 (8.5/8.12 bug → correct; 9.7/9.12 neither → bug) |
| gen comparative | 100 | 36 | 2 | 62 | 99 | 4 | 2 | ~29 (see below) |
| gen qa | 100 | 100 | 0 | 0 | 0 | 0 | 0 | 0 (all bare "X.D") |
| gen simple | 100 | 98 | 2 | 0 | 47 | 1 | 0 | 0 |
| gen repairs | 4 | 3 | 1 | 0 | 3 | 1 | 0 | 1 (8.5/8.12 bug → correct) |

### Specific findings

- **"9.8 is smaller than (the number) 9.11" counted as *neither*.** The classifier's `answer_line` returns None when a line names both numbers, and these texts contain no "9.11 is greater/larger" to rescue them. Across the patching sections, 187 outputs are bug in substance but labelled *neither*: 3 in the layer sweep, 56 single heads, 108 parity, 10 head×five and 10 head×all.
  - Every non-*correct* patched output (all 1024 single heads, all 524 parity) begins "The number 9.8 is smaller…".
  - Single-head outputs take only 2 distinct texts. Parity outputs take 4.
  - Effect on decisions: none, because every M1–M6 success criterion is *correct*.
- **Screen.**
  - chat = *bug*, correct in substance ("9.8 is smaller than 9.11. … 9.11 is greater than 9.8").
  - **comparative = *neither*.** It hit max_new and gives no answer in 64 tokens ("To compare two numbers… First, let's compare the digits in the tens place"). This is what fails S. It is *unresolved*, not evidence that comparative is clean.
  - **chat_comparative and chat_simple are labelled *neither* but are the bug in substance** ("The number 9.8 is smaller than 9.11"). chat_qa is correct.
  - So 3 of the 4 wordings err inside the chat template. The bug follows the template more than the wording.
- **False *bug*: answers that state both numbers.** The comparative regex fires on a later "Y is larger" inside a "it may seem that…" or "misconception" clause:
  - reverse L13: "The number 9.8 is bigger. The misconception … 9.11 is larger…". Correct in substance. This is not the five-layer reverse, so M6 is unaffected.
  - gen chat 8.5 vs 8.12: "The number 8.5 is bigger than 8.12. Although at first glance it may seem that 8.12 is larger…". Correct in substance.
  - The 8.5 vs 8.12 five-layer "repair" is labelled *bug* by the same mechanism and is also correct in substance.
- **Comparative format in M7.** 99/100 outputs hit max_new.
  - 59 of the 62 *neither* state no comparison at all.
  - 25 of the 36 *correct* contain no explicit correct statement. `answer_line` picks the first line that mentions only the big number, for example "1.4 = 1 + 0.4 = 1 + (1/3)".
  - 2.9 vs 2.14 is labelled *correct* from the fragment "2 in 2.9 is smaller than 2".
  - Bug in substance but labelled *neither*:
    - 1.9 vs 1.31: "1.9 is smaller than 1.31." as the final answer.
    - 3.6 vs 3.13: "3.6 < 3.13".
    - 2.3 vs 2.28: "2.3 is smaller than 2.28", as an intermediate step.
  - The 2 labelled bugs are 8.2 vs 8.13 (real) and 4.2 vs 4.18 (a "software version numbers" reading).
  - Bottom line: the comparative error rate cannot be measured at 64 tokens. In substance it is at least 5/100, with about 60 unresolved.
- **Simple bugs** (5.2 vs 5.12 and 5.5 vs 5.14) are real statements of "5.12 is bigger than 5.2". The model justifies them by reframing: rock-climbing grades, and "different measurements".
- **Long reasoning truncated at 64 tokens.** This decides labels only in comparative. In the patched sections the answer is always in the first sentence.

## 3. Decisions and Δ

`decide()` recomputed and identical on every key. Supporting checks:
- `decide_selftest` and the R14 classifier cases pass locally.
- `gen_pairs()`, `parity_subsets()` (262 per layer, order identical at L2 and L17) and `random_layer_sets()` all regenerate exactly what was run.
- `fisher_greater(0,60,0,60)` = 1.0.

**Δ testability.** Chat's baseline Δ is +3.75, which is greater than 0, so `delta_testable = False`. M1b, M2-Δ and M3b are correctly "untestable".

**Why Δ is positive while chat generates the bug.** Δ = log P(`▁ 9 . 8`) − log P(`▁ 9 . 1 1`), teacher-forced as the first reply tokens. The token ids in the template check confirm `▁`=28705, `9`=28774, `.`=28723, `8`=28783, `1`=28740.
- The shared prefix `▁ 9 .` cancels. Δ reduces to log P(8 | "9.") − [log P(1 | "9.") + log P(1 | "9.1")]. The second-digit term in " 9.11" is ≤ 0, so Δ has a built-in positive bias.
- Chat's actual reply begins "The number 9.8 is smaller…". Forcing a leading " 9." is off-distribution.
- Even on-distribution, the first number named is 9.8 in both the bug ("9.8 is smaller") and the fix ("9.8 is bigger"). Δ measures which number gets named first, not which one is judged bigger.
- The data show this directly:
  - Δ > 0 at all 32 layers, although only L7 repairs.
  - Bug-producing random 5-layer sets have Δ = 5.5, 7.0 and 5.4.
  - chat_comparative and chat_simple (both bug in substance) have Δ = 1.38 and 1.63.

**Conclusion:** for the chat format, Δ as defined is not meaningful. The "untestable" ruling is correct, and the draft's Δ values (−1.19, +2.78, …) cannot be tested with this metric.

## 4. Claim by claim

| Claim | Pre-registered verdict | Numbers | Caveats |
|---|---|---|---|
| **S** | **False** | chat bug; comparative *neither*; qa correct; simple correct | Comparative is unresolved (hit max_new), not clean. chat_comparative and chat_simple are bug in substance, so the template carries the bug. S fails on "comparative errs" only because there is no answer in 64 tokens. |
| **M1a** (no single layer repairs) | **Refuted** (`False`) | Only **L7** is correct: "The number 9.8 is bigger than the number 9.11…". The sanity transplant is correct, so the rule is interpretable. | The draft's layers 2, 17, 13, 10 and 11 each produce the bug when patched alone. L2 is labelled *neither* but says "smaller". |
| **M1b** (five layers fix individually, Δ) | **Untestable** | Layer Δ ranges from 1.88 to 10.87 and is > 0 everywhere. L7 has the largest Δ. | Δ is not meaningful (task 3). |
| **M2** at ℓ = 2, 17 | **Not supported** (generation); Δ untestable | Even k=16 is bug at both layers. Every subset kind and k gives 0 correct: even, odd, mixed, even_noH22, each 0/20, and k=16 0/1. c: 0/60 vs 0/60, p = 1.0. With H22 vs without: 0/30 vs 0/60. "≈10 even heads for >50%": 0/20 at k = 4, 8, 12. | odd k16 at L2 is *neither*, which says "smaller", so it is a bug. No subset moves the output at all. |
| **M3** (H22 the only fixing head) | **False** | 0 correct heads at L2 and L17, and 0 of 1024 across all layers. H22 is bug at L2 and L17. M3b untestable. | 56 of the single heads are *neither*, all of them "smaller", so all are bugs. |
| **M4** (H22 across layers fixes) | **False**; H22 special False | H22 at the five layers: bug. H22 at all 32 layers: bug. No head is correct in either config (0/32 each). | 10 of each are *neither*, all "smaller". H22 is not special. |
| **M5** (five-layer transplant repairs) | **True**; **specific False** | Five-layer all-heads output is correct (Δ 8.0). **7 of 10** random 5-layer sets are also correct, above the ≤ 2 threshold. | Not specific. The failing sets are [5,10,17,24,31], [12,22,24,25,31] and [13,19,20,30,31]. Only one of the 7 passing sets contains L7. |
| **M6** (reverse doesn't induce bug) | **True** | Five-layer reverse leaves simple correct: "The number 9.8 is bigger than 9.11…". Single-layer reverses are all correct in substance (L13 is mislabelled bug). | The paper says reverse Δ "stays positive and even increases". Here the five-layer reverse Δ *drops* from 12.37 to 3.00 (still positive). Single-layer Δ ranges 6.13–13.5. The output takes on chat style ("The number…"). |
| **M7** (generalization) | Bug counts **6 / 2 / 0 / 2** (draft 2/19/0/0). Eligible 4, repaired 3. **Repair transfers = False ("3 of 4")** | Eligible pairs: 1.8 vs 1.18 ✓, 3.6 vs 3.13 ✓, 8.2 vs 8.13 ✓, 8.5 vs 8.12 ✗ | The ✗ is a classifier artifact: 8.5 vs 8.12 chat *and* its repair both say "8.5 is bigger than 8.12". 9.7 vs 9.12 ("9.7 is smaller than 9.12") is a real chat bug, labelled *neither* and never patched. In substance: chat bugs = 6 (a different set), 4 genuine eligible pairs, **3 of 3 tested repaired, 1 untested**. Comparative count is unreliable (see §2). Simple 2/100 contradicts the draft's 0/100. |

Against the paper text (`02_study.tex:224–285`):
- Not supported or refuted on our prompts:
  - "no single layer fully repairs" (L7 does);
  - the five named layers fix individually;
  - the even/odd parity result and "≈10 even heads";
  - H22 dominance (single head and across 7 layers);
  - "reverse Δ increases";
  - chat 2/100 and simple 0/100.
- Holds: the clean→buggy five-layer repair, but it is non-specific.
- Holds directionally: the asymmetry (M6), and transfer of the repair to the genuine extra buggy pairs (3/3 tested).
- Untestable: every Δ number.

## 5. Gates, sanity, oddities

- **Gates all pass.**
  - Head isolation changed only head 1.
  - Identity: own-activation patch output equals the unpatched output.
  - Determinism: 3 identical outputs, also identical to `five_layer_all_heads`.
  - Hook calls: per layer, 23 (prompt), then 63 × 1 (decode). The patch acts on the prompt pass only.
- **Sanity transplant** (all 32 layers, all heads) is *correct*, Δ 13.1. Its output is "9.8 is bigger. [\n\nThat's correct!…", which is simple-style text with a stray "[". That suggests the transplant imports the donor prompt's tail.
- **Alignment oddity.** Right alignment is exact on the question tokens (positions 1–15 → 4–18). But the last 4 recipient positions, ` [/INST]`, receive the donor's `\n Answer :` activations (pairs 16→19 … 19→22). Those are the positions where the reply is decided.
  - The L7, five-layer and random-set repairs may therefore be carried by transplanted "Answer:"-position state rather than by question processing.
  - This could explain why 7/10 random 5-layer sets repair.
  - The pre-registration says "exactly token-aligned on the question", which is true but omits this tail mismatch. Chat positions 0–3 (BOS, `[INST]`) are never patched.
- **Near-null patch effects.** Single-head and parity patches barely change the output: 2 and 4 distinct texts respectively. The nulls in M2–M4 are "no effect", not "small effects".
- **max_new = 64** is enough everywhere except comparative (99/100 truncated) and the screen's comparative prompt. Chat hits the cap 54/100 times in M7, but the answer is always in the first sentence.
- Paths verified:
  - `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R15_mistral_true_heads/run_experiment.py`
  - `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R15_mistral_true_heads/results/r15_24092912/{run.json,results.json,24092912_r15_mistral.log}`
  - `/Users/gus/github/MATS/docs/experiments/11-22_mistral7b_true_head_patching.md`

No repo files were edited.
