Reviewer: fresh subagent (Claude Code)
Agent-ID: afacdd4027cf8269b
Verdict: FIXES NEEDED
Transcript-SHA256: 2d413bac1357bc75c14040724aec62140c5a036326cc54a35a748904ac9ef225

## Brief given

You are a design reviewer for an experiment that has not run yet. Judge whether it will answer its question. Be adversarial and concrete. Do not edit any files.

Repo: ~/github/MATS.
- Pre-registration: docs/experiments/11-22_mistral7b_true_head_patching.md
- Script: 9_8_research/reviewer_experiments/exp_R15_mistral_true_heads/run_experiment.py. It imports R14's script by path: 9_8_research/reviewer_experiments/exp_R14_cross_model_true_heads/run_experiment.py, whose design is in its own review rounds, docs/reviews/11-22_pythia_gemma_true_head_patching_design*.md.
- Job file: run.sbatch, in the same folder.
- CPU smoke log: smoke_cpu.log, in the same folder. It used a random tiny Mistral, so outcomes are meaningless.

Context:
- The claims at stake are in ~/github/phd_thesis/preamble/surgeon_dilemma/sections/02_study.tex, lines 224–285. They came from a collaborator by email, with no code, prompts or result files.
- So this run tests the claims rather than reproducing them.
- For background on the two bugs found in the Llama head experiments, see docs/experiments/11-21_llama8b_decimal_true_head_patching.md (R13).

Check at least:
1. Is true per-head patching correct for MistralForCausalLM under transformers 5.x? Consider `o_proj` input, GQA with 32 query heads and 8 KV heads, head_dim 128, and sliding window, if relevant. Is the multi-layer `Multi` class right? It registers one hook per layer and patches the prompt pass only, right-aligned, end index.
2. Does the teacher-forced Δ metric compute what it claims, given SentencePiece tokenization of " 9.8" vs " 9.11" after the chat template's `[/INST]`, or after "Answer:"? Check the prefix-consistency check. Check the patch positions during the teacher-forced forward. Since the end index is the prompt length, continuation positions must stay unpatched.
3. Do the formats and the pair rule test the draft's claims fairly, given that the draft's prompts are unknown? Check the chat template handling and BOS stripping.
4. For each claim S and M1–M7: does the decision rule test it? Look for ambiguities, and for rules that pass or fail vacuously. The draft's M1 sentences ("no single layer fully repairs" and "five layers fix individually") look contradictory under generation; is the handling sensible?
5. Parity subset sampling: are the subsets distinct, are the counts right, and is k=16 handled? Does the GQA structure confound even/odd? Heads 4g..4g+3 share a KV head.
6. The generalization pairs and the pair-generic classifier, including its equivalence with R14's on 9.8/9.11.
7. Gates and failure paths: what is persisted when, and what happens if the screen finds no pair. Is provenance complete, including both script hashes and the snapshot? Is the R14 import robust on Torch?
8. Compute, walltime, and anything that would make a result uninterpretable. Also, is there a cheap addition that would make the result much more informative?

Write your review as markdown. Start with a verdict, PASS or FIXES NEEDED, then required fixes with file:line, then optional suggestions. Return the full review as your final message; it will be saved verbatim.

## Report

# Design review: R15, testing the Mistral-7B head claims with true per-head patching

**Verdict: FIXES NEEDED**

The patching mechanics are correct for Mistral under transformers 5.x. That covers the `o_proj` input, the prompt-only multi-layer hooks, and the teacher-forced Δ with continuation positions left unpatched. The problem is the decision rules. Some can pass vacuously, and some are guaranteed to fail. Most of them don't read the claims the way the draft states them, and the draft's own numbers are Δ-based. The worst case is M2. H22 is an even head, so the M2 rule passes if H22 alone does the work (M3). The "≈10 even heads for > 50%" curve is also exactly what H22 alone predicts. As written, M2 cannot tell parity from H22. Every fix below is cheap and needs no new infrastructure.

## What I checked and found correct

- **Head layout.** In `MistralAttention`, `o_proj` takes `attn_output.reshape(bs, sl, 32*128)` after the transpose, so the layout is head-major. Each query head h owns columns `128h..128h+127`. GQA only shares K/V across heads `4g..4g+3`. It does not change the `o_proj` input layout, so patching query heads there is well defined. Patcher asserts `in_features == n_heads*head_dim` (R14 `:333-334`).
- **Sliding window.** v0.2's config has `sliding_window: null`, and the prompts have about 20 tokens anyway. It doesn't matter here.
- **`Multi._swap`** (`run_experiment.py:130-140`). When `end == sl` (generation), it matches R14's right alignment exactly. Recipient position 0 is never written, because `end-n ≥ 1`.
- **Teacher-forced Δ** (`:168-188`). The prompt positions `[end-n, end)` are patched, and positions `end..` are not. That matches generation: the prefill patch changes the residual, and that flows into later layers' K/V, which the decode steps then read. The first continuation logit comes from the last prompt position, which is patched, as in the first generated token.
- **Prefix check, on the real Mistral tokenizer.** I checked with the v0.1 tokenizer, which has the same SentencePiece vocab as v0.2. The prefix check passes for all four formats.
  - ` 9.8` → `▁ 9 . 8`, and ` 9.11` → `▁ 9 . 1 1`.
  - So Δ = log P(`8`|`▁9.`) − log P(`1`|`▁9.`) − log P(`1`|`▁9.1`). The shared `▁9.` cancels.
- **Token counts.** chat 23, comparative 18, qa 21, simple **20** (`An`,`swer`). The smoke log's 19 came from the toy tokenizer.
- **Hook gates.** The per-layer call count (`:146-148`) and the shape gate (`:259`) check prompt-pass-only behaviour for all five layers.
- **Parity subsets.** I reran `parity_subsets()`. It gives 202 per layer: even and odd get 20/20/20 at k = 4/8/12 plus 1 at k = 16; mixed gets 20 at each k. Subsets are distinct within each (kind, k). Total is 404, not the pre-registration's "~470".
- **GQA and even/odd.** Every KV group `4g..4g+3` holds exactly 2 even and 2 odd heads. So a parity effect cannot be a disguised KV-group effect. Parity is the same as within-group slot {0,2} vs {1,3}, which is worth one sentence.
- **Classifier.** `make_classifier("9.8","9.11")` is the same regex logic as R14's `classify`: `re.escape("9.8")+"0*"` equals `9\.80*`, with the same lookbehind/lookahead and the same order. In `gen_pairs`, `and` binds before `or`, so the exclusion is right. The shorter-larger constraint `dd < 10d` is right, there are 2,916 possible pairs, and 9.8/9.11 is excluded.

## Required fixes

### 1. M2 cannot tell parity from H22 (pre-reg `:76-78`, script `:291-300`)

- H22 is even. If M3 holds (H22 alone fixes), M2 (a)–(c) all pass automatically: odd-only subsets never contain H22.
- The share of even-only subsets that contain H22 is k/16: 25 / 50 / 75 / 100% at k = 4/8/12/16. So "≈10 even heads needed for > 50%" is exactly what H22 alone predicts.
- In the seeded subsets, H22 appears in 5, 11, 14 and 1 of the even subsets at k = 4, 8, 12, 16.
- (c) is redundant given (a) and (b): if no odd subset is correct, (c) only needs one correct even subset, and (a) already requires that.
- **Fix:** pre-register an H22-excluded reading. Parity holds at ℓ only if even-only subsets **without H22** are correct more often than odd-only subsets at matched k. Otherwise the verdict is "even-only repair is carried by H22". Report every rate split by H22 ∈ S. If you want power on this reading, draw the extra even subsets without H22 (≤ 60 more generations per layer). Also report a mixed set containing H22 as the specificity control. Mixed subsets currently include 3 pure-even sets at k = 4; label them or redraw.

### 2. Pre-register a Δ reading where the draft uses Δ (pre-reg `:70-85`)

The draft's "fixes" are Δ claims: "five layers fix … with non-trivial Δ", "H22 across 7 layers gives Δ = +2.16, a full fix", "Δ flips −1.19 → +5.88". Under generation, the pre-registered M1 makes its two halves mutually exclusive (`:75`), so M1 cannot pass as a whole. M2 and M3 have the same tension with "no single layer fully repairs": a subset fixing a layer whose full patch fails is implausible under generation.

**Fix:**
- **M1.** Split it into two parts:
  - M1a, generation: no layer gives *correct*.
  - M1b, Δ: at least 4 of {2, 10, 11, 13, 17} have Δ_patched > 0.
- **M2 and M3.** Give each a generation reading and a Δ reading (Δ > 0 = success). Pre-register M3b: H22 has the largest Σ_ℓ (Δ_patched − Δ_base) over the single-head sweep. The draft claims +3.07 vs +2.14 for H29, and the data is already collected.
- **Precondition.** The Δ readings can be tested only if chat's baseline Δ ≤ 0 (the draft has −1.19). Δ is conditioned on a reply that starts with a number. Mistral may reply "The larger number is 9.11" while Δ > 0. In that case every Δ "fix" would pass vacuously. State this as an "untestable" branch.
- Also give the *not None* rate of Δ, so a tokenizer mismatch can't pass silently.

### 3. "not bug" passes on broken output (pre-reg `:72`, `:83`)

- **M6** passes if the reverse transplant leaves simple "not bug". A transplant that garbles simple into "neither" would count as "does not induce the bug". The draft says simple stays correct and Δ "stays positive and even increases". Require reverse = *correct* (or not *bug* and Δ > 0), and report "neither" separately.
- **S.** "qa not bug, simple not bug" is satisfied by junk too. Require at least one of them to be *correct*. The pair rule needs that anyway.

### 4. Boring hypothesis 2 has no layer-level test (pre-reg `:32`, `:58`)

M5 (five-layer repair) and M4 (H22 at five or all 32 layers) have no count-matched controls.

- **M4.** Add each single head h at {2,10,11,13,17} and each h at all 32 layers: 64 generations. M4 then becomes "is H22 special?" rather than "does writing one head at many layers ever work?".
- **M5.** Add about 10 seeded random 5-layer sets, all heads: 10 generations.
- **The OR rule.** M4 currently passes if *either* of two different interventions passes (`:80`). Keep that, but report the rank of H22 among the 32 heads for each.

### 5. Pin the snapshot (pre-reg `:36`; R14 `:306-318`; sbatch)

The pre-registration names snapshot `63a8b081…`, but the script loads `revision="main"`, which is whatever `refs/main` points to in the offline cache. Nothing checks it. Assert `p.commit.startswith("63a8b081")`, or pass that revision explicitly, before any generation.

### 6. Possibly stale R14 git SHA (`run_experiment.py:218`)

`os.environ.setdefault("R14_GIT_SHA", ...)` combined with `sbatch --export=ALL` keeps any `R14_GIT_SHA` left in the submitter's shell from the R14 jobs today. `provenance["git_sha"]` would then name the wrong commit. Assign it directly.

### 7. Record alignment per pair (`:245-247`)

`alignment_info` from R14 isn't called. With the real tokenizer, simple(20) → chat(23) right-aligned is exactly token-aligned:
- recipient 4..18 ↔ donor 1..15 (question);
- recipient 19..22 `▁[ / INST ]` ↔ donor `\n An swer :`;
- recipient `[INST]` (1–3) is unpatched.

If simple is not *correct* and the rule falls back to **qa**, qa(21) → chat(23) puts every question token one position off: the recipient's `8` gets the donor's `.`, and so on. Record `alignment_info(p, dp, rp, "right")` in `run.json`. Pre-register that a qa donor is reported as "off-by-one aligned", or use a span-aligned mapping for it.

### 8. Rerun the smoke on the committed code

`smoke_cpu.log:3` prints "rules (7 synthetic truths), late-training verdict (6 cases)". That is the **400d5ee** R14 selftest string. The committed R14 (d330a52) prints "noise test …, (9 cases)". So the smoke ran against an older R14 than the one the job imports. `Patcher` is unchanged, so this is about provenance and the start-up path, not mechanics.

The rerun should:
- run on the committed pair, without `--force-pair`, so the **no-pair path** runs at least once (it never has);
- time R14's new noise selftest: 80 calls to `summarize` on 4,095 subsets in pure Python at every start-up, unmeasured.

## Optional suggestions, roughly in order of value per cost

1. **Separate template from wording (cheap, informative).** Also screen the comparative, qa and simple wordings *inside* the chat template: 3 generations plus Δ.
   - The draft's chat-vs-other contrast mixes "template" with "wording". Your four formats do too.
   - If a chat-wrapped qa or simple is *correct*, it is a donor that differs only in wording, which is a much cleaner transplant. Report it descriptively, or as a pre-registered secondary pair.
2. **Truncation.** At 40 tokens Mistral-Instruct often opens "To compare 9.8 and 9.11, we…". That line names both numbers, so `answer_line` returns None. If the conclusion is cut off, the output is "neither", and "neither" can kill the screen (no pair).
   - Record `hit_max_new` (no EOS) for every generation.
   - Consider `MAX_NEW = 64–80`. At about 1.5 s per 40 tokens, 2,000 generations at 80 tokens is about 1.7 h, which fits the 3 h walltime.
   - Also record outputs that state the comparison as smaller/less ("9.8 is less than 9.11"), which the R14 classifier labels "neither", without changing the classifier.
3. **Save during M7** (`:313-333`). `results.json` is written only after all 400 generations. On the no-pair path, nothing is written before then. Save after each pair.
4. **Chat template check.** Compare `p.ids(fm["chat"])` against `tok.apply_chat_template(..., tokenize=True)` and record both. The BOS-stripped string gives `<s> ▁[ INST ]`, which matches mistral-common. The HF template path may give `[` without the `▁`, and it's better to know which one the draft's collaborator likely used.
5. **R14 import.** Record the R14 file's sha256 and also check it against an expected value (an env var or a constant). The scratch copy is synced by hand (sbatch comment), so a mismatch should stop the run, not just be logged.
6. **Classifier equivalence.** Test it on all 18 of R14's selftest cases, not 8. Move R14's `cases` list to module level and loop `make_classifier("9.8","9.11")` over it.
7. **Fix the compute table.** Parity is 404 generations, not "~470". Add the gate and sanity runs (about 6).
8. **Rank the draft's top five.** Report H22, H29, H31, H10 and H21 by summed ΔΔ from the single-head sweep. That tests the draft's "top-5 mixed parity" sentence for free.
9. **Wording.** The draft says "attention pattern" and "z-pattern" transplant. Your intervention is z, the `o_proj` input. A softmax-pattern transplant across prompts of different lengths is a different intervention. Say so in the limitations: a mismatch can come from that alone.

## Key file references

- `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R15_mistral_true_heads/run_experiment.py`: `:105-118` parity subsets, `:168-188` Δ, `:218` env setdefault, `:236-237` pair rule, `:245-267` gates, `:304-306` reverse, `:313-333` M7.
- `/Users/gus/github/MATS/docs/experiments/11-22_mistral7b_true_head_patching.md`: `:36` snapshot, `:72-87` decision rules, `:91` compute.
- `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R14_cross_model_true_heads/run_experiment.py`: `:306-318` Patcher revision, `:430-438` `alignment_info`, which R15 does not call.
- `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R15_mistral_true_heads/smoke_cpu.log`: `:3` is the stale-R14 selftest line.
