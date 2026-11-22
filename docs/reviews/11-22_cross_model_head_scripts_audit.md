Reviewer: fresh subagent (Claude Code)
Agent-ID: ad10326287d26deda
Verdict: BOTH
Transcript-SHA256: 1ec3d743a88ce53222ad1a62a18adede3fbe63138538f9ef85076010a7d59f52

## Brief given

Read-only code audit. Do not edit any files.

Repo: ~/github/MATS. Paper draft whose claims are at stake: ~/github/phd_thesis/preamble/surgeon_dilemma/ (sections/*.tex, tables/*.tex). It makes head-level and parity claims for Mistral, Pythia, Gemma and possibly other models beyond Llama.

Background: we found two bugs in the Llama head-patching scripts, and a corrected experiment (R13) showed that the Llama head-level claims do not survive the fix.
- Bug A (hook point): a forward hook on `model.layers[L].self_attn` (or the equivalent attention module) returns the output AFTER the output projection (o_proj / W_O / dense). Reshaping that to [n_heads, head_dim] and patching slices swaps blocks of residual-stream coordinates, not heads. True per-head patching acts on the input of the output projection (e.g. `o_proj.register_forward_pre_hook`), or uses hooks such as TransformerLens `hook_z`.
- Bug B (decode steps): the hook remains registered during `model.generate()`. On decode steps (seq_len 1) it overwrites the new token's activations with the saved clean activations at position 0, or with whatever positions the slice selects.

Tasks:
1. Find every script that produced head-level or head-subset results for non-Llama models. Start at `9_8_research/cross_model_validation/` (test_*.py) and also grep the rest of `9_8_research/` for Mistral, Pythia, Gemma, Qwen, GPT-2 and Phi head patching.
2. Map each paper claim about non-Llama heads, parity or head counts (cite the .tex file and line) to the script and results file that produced it. Use the result filenames, docs and READMEs to make the link.
3. For each script, report with `file:line` evidence:
   - (a) Which module is hooked, and is it pre- or post-output-projection? Be careful with architectures: Pythia/GPT-NeoX uses `attention.dense`, Gemma has o_proj and possibly attention-output norms, and some models' attention outputs include bias.
   - (b) Is the hook active during generate's decode steps, and what does it write there?
   - (c) How are prompts of different lengths aligned?
   - (d) Is the outcome measured by generation or by a single forward pass/logits? A logits-only measurement avoids bug B.
   - (e) Anything else that would invalidate a head-level claim, e.g. GQA/MQA head counts mis-set, or a wrong head_dim.
4. Verdict per claim: AFFECTED BY A, AFFECTED BY B, BOTH, CLEAN, or CAN'T TELL. Give one line of reasoning for each.

Return a compact table and then the details. Be precise; don't speculate beyond what the code shows.

## Report

## Non-Llama head-level claims: audit of the code behind them (read-only)

**Bottom line:** Every non-Llama head or parity claim that has code in `~/github/MATS/9_8_research` comes from the same copy-pasted hook, and it has both Bug A and Bug B. That covers all Pythia and Gemma claims. The Mistral claims have no code or results anywhere under `~/github`. They came from the collaborator (Hoang) by email, so I can't tell. The Pythia "0% on 50 decimal pairs" result has a third problem: the scoring makes 0% certain whatever the patch does.

Note that the paper's own method paragraph describes Bug A as the method. `sections/02_study.tex:53-58` says: hook `self_attn` output "post-$W_O$", reshape to `[1,T,H,d_head]`, and replace head indices.

### Verdict table

| # | Paper claim (file:line) | Script → results | Verdict | One-line reason |
|---|---|---|---|---|
| 1 | Pythia: even-head subsets repair, all-odd fail, "all 16 even heads → 100%". `02_study.tex:289-293`, `01_intro.tex:66-67`, fig caption `02_study.tex:31`, `tables/cross_model_rates.tex:20` (R_patch ~1.0) | `cross_model_validation/test_pythia_160m.py` → `pythia_160m_validation_20250926_174151.json` (all_even [0,2,4,6,8,10] = 1.0, all_odd = 0.0) | **BOTH** (plus a head-count error) | Hooks `attention` output, which is after `dense`. The hook stays on during `generate`. Pythia-160M has 12 heads, so there are 6 even heads, not 16. |
| 2 | Pythia parity appears only at about 120k/143k steps, in the final 15% of training. `02_study.tex:295-299`, `03_discussion.tex:17-19`, `02_study.tex:168-169`, `cross_model_rates.tex` footnote † | `training_dynamics_analysis/scripts/phase1_training_dynamics/test_pythia_training_dynamics.py` → `training_dynamics_analysis/results/pythia_training_dynamics_20250926_191344.json` | **BOTH** | Same hook. Also, at 32k and 64k the unpatched baseline is already 100% correct, so there is no bug to repair at those checkpoints. |
| 3 | Pythia: "On 50 different decimal pairs we get 0% accuracy from the intervention". `02_study.tex:302-303`, `cross_model_rates.tex` footnote * | `phase2_pythia_analysis/test_biblical_interference_hypothesis.py` → `training_dynamics_analysis/scripts/biblical_interference_analysis_20250926_201257.json` (also `COMPREHENSIVE_FINDINGS_REPORT.md:49`, "0.0% (0/50)") | **BOTH, and invalid even without them** | A response counts as correct only if it contains "9.8" and not "9.11", but the 50 prompts are 1.x and 5.x pairs. All 50 baseline/even/odd answers are "unclear". The donor is always the 9.8/9.11 prompt. |
| 4 | Pythia: synonyms, word-order reversals and rephrasings break the intervention. `02_study.tex:300-305`, `03_discussion.tex:72-74` | `phase2_pythia_analysis/test_memorization_hypothesis.py` → `training_dynamics_analysis/results/memorization_analysis_20250926_195511.json` | **BOTH** | Same hook. The donor is fixed at "Which is bigger: 9.8 or 9.11?" for every recipient variant (line 136). |
| 5 | Gemma-2B: both even-only and odd-only subsets fix the bug ("parity-independent"). `02_study.tex:310-314`, `cross_model_rates.tex:21,32-34`, `01_intro.tex:67-68`, `03_discussion.tex:9`, fig caption `02_study.tex:31` | `cross_model_validation/test_gemma_2b.py` → `gemma_2b_validation_20250926_174510.json` (layer 8: even4 = odd4 = mixed4 = 1.0, even2 = even3 = 0.0) | **BOTH** | Hooks `self_attn` output, which is after `o_proj`. The hook stays on during `generate`. Results track how many 256-wide residual blocks are overwritten (4 succeed, 2-3 fail, mixed succeeds), not head identity. |
| 6 | Mistral: H22 is the only single head that fixes the bug (+3.07); top-5 heads mixed parity; even-only fixes at ℓ=2,17, odd-only fails at k∈{4,8,12,16}; about 10 even heads needed; H22 across 7 layers gives Δ=+2.16. `02_study.tex:238-252`, fig caption `02_study.tex:285`, `01_intro.tex:58-61`, `03_discussion.tex:8` | No script or results in the repo. `todo.md:19-20,198-199` say the numbers came from the collaborator by email. | **CAN'T TELL** | No code to inspect. The text says "$z$-pattern transplant" (`02_study.tex:234`), which could mean `hook_z` (pre-W_O), and outcomes are given as Δ logit-diff. Both would suggest the method is clean, but neither can be verified. |
| 7 | Mistral layer-level claims: 5-layer fix, asymmetric causality, decimal-pair generalization. `02_study.tex:232-237, 254-280` | None in repo (collaborator) | **CAN'T TELL** | These are layer-level, so Bug A does not apply. Bug B depends on code we don't have. |
| 8 | GPT-2: no logit-level bug. `02_study.tex:321-332`, `05_appendix.tex:4-28` | `reviewer_experiments/exp_R12_gpt2_reconciliation/run_experiment.py` | **CLEAN for the claim as stated** | This claim is baseline logits only, with no patching. The head patching in R12 (section 5, below) has Bug A but is not cited. |

### Details for each script

**A. `cross_model_validation/test_pythia_160m.py` (claim 1)**
- **(a) Hook point:** `get_attention_module` returns `model.gpt_neox.layers[L].attention` (`:54-56`), with `register_forward_hook` (`:100`, `:118`). GPTNeoXAttention's output is after `dense`, and `dense` has a bias, so the bias is patched along with everything else. The hook reshapes `hidden_size // n_heads` (768/12 = 64) and swaps slices (`:74-85`). This is Bug A: it swaps residual-coordinate blocks.
- **(b) Decode steps:** The patch hook is registered around `self.generate()` (`:221-222`, `generate` at `:132`). At decode steps `seq_len = 1`, so `min_seq_len = 1` (`:81`). Every new token gets the donor's position-0 slice in the selected blocks.
- **(c) Alignment:** Positions `[0, min(T_donor, T_recip))` (`:81-85`). The donor is "Which is bigger: 9.8 or 9.11?" and the recipient is "Q: Which is bigger: 9.8 or 9.11?\nA:" (`:210-211`). The "Q:" prefix shifts the tokens, so donor position i is matched to a different token in the recipient. The donor is shorter, so the recipient's last positions ("\nA:"), where the answer is predicted, are never patched during the prompt pass.
- **(d) Measurement:** Generation of 20 tokens, scored by string matching (`check_bug_fixed`, `:145-168`). Any output containing "9.8" counts as correct.
- **(e) Other problems:**
  - Pythia has 6 even heads. The paper's "all 16 even heads" doesn't match this or any other Pythia script.
  - The donor prompt is itself 100% buggy at baseline in the results JSON ("Which is bigger…" bug_rate 1.0).
  - `layer_results` in the JSON is empty.
  - `second_half_even` [6,8,10] also gives 1.0 and `first_half_even` gives 0.0.

**B. `training_dynamics_analysis/scripts/phase1_training_dynamics/test_pythia_training_dynamics.py` (claims 2 and 3 in the table)**
- **(a)** Same hook: `.attention` post-dense (`:60-62`, `:106`, `:125`, reshape at `:80-91`). Bug A.
- **(b)** The hook context wraps `model.generate` (`:220-229`). Bug B, as in A.
- **(c)** Same donor/recipient pair and prefix alignment as A (`:159-160`, `:87`).
- **(d)** Generation, scored by substring: "9.8" present and "9.11" absent (`:176-184`).
- **(e) Other problems:**
  - The layer is fixed at 6 for every checkpoint (`:50`).
  - Exceptions inside trials are silently counted as failures (`:201-202`, `:239-240`).
  - Results: 32k and 64k have baseline = even = odd = 1.0, because the model is already correct there. 80k-120k are all 0. 143k has even 1.0 and odd 0.0.
- The duplicate script `cross_model_validation/test_pythia_training_dynamics.py` loads invalid model names (`:54-59`). Its run in `results/…191136.json` shows steps 256k and 512k failed with errors. Neither is the source of the paper's numbers.

**C. `phase2_pythia_analysis/test_biblical_interference_hypothesis.py` (the 0/50 result)**
- **(a)** Hooks `gpt_neox.layers[6].attention`, post-dense (`:158`, `:170`, `:206`, `:238`).
- **(b)** Hook active during `generate` (`:209`, `:241`).
- **(c)** The donor is always `"Which is bigger: 9.8 or 9.11?"` (`:139`), while recipients are `"Q: Which is bigger: {X}.{Y} or {X}.{Z}?\nA:"` (`:132-135`).
- **(d)** Scored on generation, but a response is correct only if `answer == "9.8"` (`:247-249`).
- **(e) Why 0/50 is certain:** `cases[:50]` (`:135`) covers only X=1 (32 cases) and X=5 (18 cases), so no prompt contains 9.8. In the JSON all 50 cases are (`unclear`, `unclear`, `unclear`). The 0% would be the same with no intervention.

**D. `phase2_pythia_analysis/test_memorization_hypothesis.py` (claim 4)** and **`test_comprehensive_decimal_patterns.py`**
- Both use the same post-dense hook, keep it on through `generate`, and use prefix alignment:
  - memorization: `:143`, `:155`, `:191-194`
  - comprehensive: `:101`, `:212`, `:225-232`
- The memorization script uses the fixed 9.8/9.11 donor for every variant (`:136`).
- The comprehensive script uses matched donors (`:154-155`) but summarizes as 25 cases with specialization 0.0 everywhere. It is not the source of "50 pairs".
- The other phase2 scripts (`quick_validation`, `test_order_dependency`, `test_pattern_boundaries`, `test_reversed_order`, `test_type1_only`, `validate_original_case`) use the identical pattern. The paper does not cite them directly.

**E. `cross_model_validation/test_gemma_2b.py` (claim 5)**
- **(a)** Hooks `model.model.layers[L].self_attn` with a forward hook (`:56`, `:100`, `:118`). GemmaAttention's output is after `o_proj`. Gemma-1 has no norm on the attention output inside `self_attn`, and `o_proj` has no bias. The reshape uses `hidden_size // 8` = 256 (`:75-77`). That equals Gemma-2B's true head_dim of 256 only because 8 × 256 = 2048, but the tensor is still post-o_proj. Bug A.
- **(b)** Hook active during `generate` (`:221-222`). Bug B: each decode token gets donor position 0, which is BOS for Gemma.
- **(c)** Same misaligned pair and prefix slicing (`:81`, `:210-211`).
- **(d)** Generation, scored by string match (`:145-168`).
- **(e) Other problems:**
  - The model tested is base `google/gemma-2b` (`:30`), and the JSON shows a 100% baseline bug rate on the Q&A prompt. The paper's "~5% on the canonical prompt" (`02_study.tex:311`, `cross_model_rates.tex:21`) instead matches the IT number in `research_journal_even_heads.md:210`. So the head claim and the bug-rate claim describe different checkpoints.
  - The donor prompt is also 100% buggy at baseline.
  - MQA (1 KV head) does not affect the 8 query-head count used here.

**F. Mistral (claims 6 and 7)**
- `grep` across `~/github` turns up no 9.8/9.11 Mistral patching code. The only Mistral 9.8/9.11 hit is `cross_model_validation/test_gqa_hypothesis.py:46`, where the model is commented out.
- Provenance is "collaborator email" (`todo.md:198-199`). `paper_outline_v4.md:181` lists "Mistral code" as still to be added to the anonymized repo.

**G. Scripts with head patching that the paper does not cite**
- `exp_R12_gpt2_reconciliation/run_experiment.py:617-720` ("Hoang 5-head circuit"):
  - Hooks `transformer.h[L].attn` output, which is after `c_proj` and includes its bias (`:673`, `:700`), then slices `head_idx*64:(head_idx+1)*64` (`:691-696`). This is **Bug A**.
  - Measured with a single forward pass on logits (`:707-709`), so no Bug B.
  - The paper does not use this result.
- `cross_model_validation/test_kv_sharing_hypothesis.py`: `self_attn`/`.attention` with forward hooks (`:107-112`, `:160`, `:179`) and `generate` (`:263`). BOTH. Not in the paper.
- `cross_model_validation/test_gqa_hypothesis.py:180-206`: the "even/odd" runs apply no patch at all; they are two identical baseline generations. Meaningless, and not in the paper.
- `pythia_clustering_dynamics/scripts/*` (Exp 14, head swapping and activation patching): same `.attention` post-dense forward hook plus `generate` (e.g. `head_swapping_experiment_fixed.py:94-154`, `verify_activation_patching_corrected.py:120-177`). Not cited in the paper.

### Points that change the paper even before rerunning
1. "All 16 even heads" for Pythia is wrong on head count. Pythia-160M has 12 heads (`test_pythia_160m.py:44-46`).
2. "0% on 50 decimal pairs" is a scoring artifact (point C above), not a measurement.
3. The Gemma head result comes from the base model with a 100% Q&A bug rate. The paper presents Gemma's bug rate as about 5%.
4. The method description at `02_study.tex:53-58` describes the post-W_O reshape and needs to change along with the results.
