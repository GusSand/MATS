Reviewer: fresh subagent (Claude Code)
Agent-ID: aba4d2442bf4a4630
Verdict: FIXES NEEDED
Transcript-SHA256: e75185da3abed0a55f5a83ee5a13cb55dba0da215d03ee548ddc29a208415206

## Brief given

You are a fresh, skeptical design reviewer for a pre-registered mechanistic-interpretability experiment (R19) in ~/github/MATS. Read-only: do NOT edit files, do NOT ssh or run GPU jobs. You may run small local checks (`uv run -q --no-project --with torch --with transformers --with numpy python -I ...`; a tiny random Llama with the real Llama-3 tokenizer is at /private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/tiny_llama3; real tokenizer.json at .../scratchpad/tok/tokenizer.json).

Read:
- Pre-registration: docs/experiments/11-23_llama8b_last_fraction.md
- Script: 9_8_research/reviewer_experiments/exp_R19_last_fraction/run_experiment.py, run.sbatch, smoke_cpu.log
- Background: R17 docs/experiments/11-23_llama8b_localization_map.md and R18 docs/experiments/11-23_llama8b_eleven_token.md (results sections and their verification reports in docs/reviews/*_verify.md). R19 imports R18's script (9_8_research/reviewer_experiments/exp_R18_eleven_token/run_experiment.py) for classify_pair, gen_ids, capture_resid, and R17's Model.

R19 tests, pre-registered, a post hoc R18 observation: in Llama-3.1-8B's format-dependent decimal questions (simple format correct, Q&A wrong), the flip is carried by the residual stream at the question's LAST fractional token, in both number orders, versus the other fractional token and every other prompt position.

Review for: code correctness (question construction, natural tokenization alignment check, LFT/OFT lookup, eligibility, batched left-padded screening vs unbatched recheck, site analysis, margin/competitors, controls, reverse), whether the decision rule is unambiguous, implemented exactly, and discriminates H from the boring hypotheses (e.g., 'any late position works' — is the margin rule adequate, given the LFT is adjacent to '?'; is choosing the best layer within the band biased toward the LFT vs the OFT/competitors, which are evaluated only at the LFT's/OFT's best layer?), gates that can fail spuriously or pass vacuously, sample-size realism (how many big-first format-dependent questions to expect), and anything a referee would demand. Don't relitigate R17/R18.

Output: first line exactly one of PASS, PASS WITH MINOR FIXES, FIXES NEEDED. Then numbered findings with file:line, severity (blocking / should-fix / minor), and a concrete fix. Be concise.

## Report

FIXES NEEDED

The code mechanics hold up. I checked the Llama-3 tokenizer on all 9,600 questions: the alignment check passes every one with mismatches exactly at [1, len−2], the LFT sits directly before "?\n", the LFT/OFT positions are 13/8 for W1/W2 and 14/9 for W3, and every Y in 2..9 and Z in 10..19 is a single token. The R17 p13 reference is tracked in git, has layers 0–31, and its baselines match the hard-coded 1.171875 / −0.90625. The R18 sha prefix `1c5e555b` matches. The decide() labels follow the written rule order. The problems are in what the rule can discriminate, not in the arithmetic.

1. **Competitors are scored only at the site's own best layer, so the margin rule does not test "any late position works"** (`run_experiment.py:238,250-253,260`; doc lines 23-24, 53). Severity: **blocking** (the doc claims a test the code doesn't run).
   - The LFT gets the max over 6 band layers. "?\n", "Answer"/"A" and ":" are each scored at that one layer only.
   - Example: the LFT peaks at L10 (r 0.9) and "?\n" reaches r 0.9 at L14 but only 0.3 at L10. The LFT gets a margin of 0.6 and counts as a repair, yet a late position also works. That is exactly the boring hypothesis the doc says this rule rules out.
   - Fix, before the run: score every competitor position at every band layer. Primary margin = r_site(Lb) − max over (p ≠ site, L ∈ band) of r(p, L). Report the same-layer margin as secondary.
   - Cost: about 16 positions × 6 layers per site, about 100 extra passes per site. Under 30 min for 240 questions on one A100.
   - Alternative: keep the same-layer margin, but reword the claim to "at the layer where the LFT peaks, no other position carries it", and drop the "any late position" sentence.

2. **H cannot be separated from "the token right before '?'"** (doc lines 18-24). Severity: **should-fix** (claim wording).
   - In every prompt the LFT is the token right before "?\n". So "last fractional token" and "the position before the question mark" are the same thing here, and no competitor separates them.
   - Fix: say this in "What this can and cannot show". Or, if you want to test it, add a small variant with a non-number token between the second number and "?" (e.g. "{n2} exactly?"), with the LFT site defined as before.

3. **A run can land in the "exactly one holds" bucket when the other stratum is only too small** (`run_experiment.py:98-99`, selftest line 120; doc rule 2). Severity: **should-fix**.
   - If big-first has n < 10 and small-first holds, the label reads "holds for small_first only". Readers will take that as a negative big-first result, and big-first is the stratum most likely to be small.
   - Fix: add an explicit label, e.g. "holds for small_first; big_first descriptive (n < 10)". Also make rule 4's label name any descriptive strata.

4. **The confirmatory sample includes the questions that generated the hypothesis** (grid at `run_experiment.py:304-305`). Severity: **should-fix**.
   - W1 with Y 6–9 and Z 10–13 is R18's grid. Its 39 format-dependent questions, including the 6 big-first ones behind the post hoc claim, can be sampled again.
   - Fix: pre-register that the decision is computed excluding (W1, Y∈6..9, Z∈10..13), reporting the overlap separately. Or drop those questions from the pool.

5. **Competitors are computed only when the site's r ≥ 0.5** (`run_experiment.py:242`). Severity: **should-fix**.
   - In failing questions you lose the "where does it live instead" map. That map is how R18 found this hypothesis, and the doc says "keep designing until we find smth".
   - In R18 small-first, Y's r was only 0.4–0.56 on average, so many LFT rows will fall below 0.5 and have no competitor data.
   - Fix: always compute the band-wide competitor map for the LFT (it's needed for item 1 anyway). Keep generation and controls gated if you want to save time.

6. **Sample-size realism isn't stated** (doc line 91). Severity: **minor**.
   - R18's rates suggest small-first will easily hit the 120 cap (33/320 ≈ 10%, so about 480 in the screen).
   - Big-first was 6/320 ≈ 1.9% on R18's W1 subgrid. Projected over 4,800 big-first questions that is about 90, but the rate may differ for Y 2–5, Z 14–19 and wordings W2/W3, and item 4's exclusion removes some. A realistic range is about 20–90, so n ≥ 10 is likely but not certain.
   - Fix: write this expectation into the doc. Also report Wilson 95% CIs for both rates; with n = 10, a rate of 0.5 has a CI of about 0.24–0.76, which a referee will ask about.

7. **The batch-agreement gate is untargeted and can kill the run without cause** (`run_experiment.py:323-335`). Severity: **minor**.
   - The 64 sampled questions are mostly bug/neither, so the gate says little about format-dependent recall. A failure halts the run even though every analysed question is re-checked unbatched anyway.
   - Fix: also report agreement on the format-dependent candidates (from the unbatched-eligibility skip counts), and consider making the gate report-only. Either way, pre-register the choice.

8. **The "LFT − OFT ≥ 0.3" criterion is close to redundant** (`run_experiment.py:92`). Severity: **minor**.
   - Each site's repair needs margin ≥ 0.3 over all positions, including the other site. So LFT and OFT can rarely both repair, and LFT ≥ 0.5 almost implies the difference.
   - Fix: note this in the doc. No code change needed.

9. **Small items** (all minor):
   - `load_r18` exits via `sys.exit(msg)` with code 1, not 2, and before `run.json` exists (`run_experiment.py:66-67`). The doc says any failure exits 2. Fix: call it inside the try, or exit 2 explicitly.
   - The answer-prefix positions (" ", "X", ".") aren't competitors (`run_experiment.py:251`). Say in the doc that "every other prompt position" excludes the answer prefix.
   - The R19 directory is untracked. `R19_GIT_SHA` must name a commit that contains this script, and the sbatch should check that `script_sha256` matches it.
   - Doc line 96 says "~17" competitors; the real count is 15 (W1/W2) or 16 (W3).

Files reviewed:
- `/Users/gus/github/MATS/docs/experiments/11-23_llama8b_last_fraction.md`
- `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R19_last_fraction/run_experiment.py`
- `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R18_eleven_token/run_experiment.py`
- `/Users/gus/github/MATS/docs/reviews/11-23_llama8b_eleven_token_verify.md`

The tokenizer check script is at `/private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/r19tok.py`.
