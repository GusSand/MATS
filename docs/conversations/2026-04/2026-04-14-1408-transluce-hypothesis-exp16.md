TITLE: Experiment 16 + 16b — Transluce Hypothesis Test (Llama / Pythia / GPT-2)
DATE: 2026-04-14
PARTICIPANTS: Gustavo, Claude (Developer role)
SUMMARY: Ran two experiments testing Transluce's (Oct 2024) claim that the Llama 9.11 > 9.8 bug is caused by spurious MLP neurons from Sept-11 / Bible-verse / gravity contexts.

**Experiment 16** (`run_experiment.py`): originally framed as a Transluce test but used activation contrast between chat/simple formats instead of Transluce's actual gradient-attribution method. Found modest (14pp) bug-rate reduction from targeted ablation in Llama, concentrated at L31, only when mixing positive- and negative-scoring neurons. Pythia no effect. GPT-2 logit error rate 11% (matches R12). Recognized mid-writeup that this was not a Transluce test — the differential-activation metric differs fundamentally from Transluce's `e · ∂z/∂e` gradient attribution.

**Experiment 16b** (`run_proper_transluce.py` + followups): proper replication with gradient attribution on a single format, plus semantic probe on Sept-11/Bible/gravity/neutral probe sets. Headline findings:
- Pythia-160M on `answer_prompt` format: **72.3% → 7.1% bug rate** from top-50 positive-attribution ablation (vs 55% from layer-matched random). Clean replication of Transluce's direction.
- Llama-3.1-8B on Transluce prompt: 40.6% → 27.3% (13pp), but n_valid drops to 22/100 (heavy destabilization).
- GPT-2-Small: 0% baseline error under R12 logit eval → hypothesis untestable.
- **Pythia format sweep: bug rate spans 1.1% to 68.4% across near-equivalent phrasings** — format-dominance is much stronger in Pythia than Experiment 10 had shown.
- Semantic probe: **gravity is the strongest theme across all three models** (Pythia +5.65, GPT-2 +1.48, Llama sept11 +1.30). Transluce's sept-11/bible narrative barely replicates in our probes.

INITIAL PROMPT: we need to run another experiment on the 9_8_research directory and using the research_journal_even_heads.md. Here are the details follow all the guidelines in Claude.md. [followed by a full Python script spec for the Transluce hypothesis test on Llama-3.1-8B, Pythia-160M, and GPT-2-Small]

KEY DECISIONS:
- Role: Developer
- Directory: `9_8_research/experimental/16_transluce_hypothesis/` (Experiment 16, continuing 1–15 main + R1–R12 reviewer numbering)
- Experiment 16 fixes (all user-approved):
  - Target positions → last 4 tokens of prompt (tokenizer-agnostic), replacing fragile "9.11" substring finder
  - Llama neuron hook at `down_proj.register_forward_pre_hook` (canonical SwiGLU `act_fn(gate) * up_proj`)
  - Pythia bug fixed: `compute_differential_neurons` takes buggy/correct prompts as params (was silently reusing Llama prompts)
  - GPT-2 logit-diff reused from exp_R12 via sys.path import
  - `wilson_ci` reused from R3/R12
  - Per-section checkpointing for crash safety
  - Bug-rate measurement switched to stochastic sampling (temp=0.6, N=100) — user explicitly chose 0.6 over my suggested 0.7 and 0.0 alternatives
- After Transluce article (https://transluce.org/observability-interface) fetch revealed single-format gradient-attribution method: reframed Exp 16 as format-differential (not Transluce test), and ran Exp 16b for actual Transluce replication. User explicitly wanted both written up.
- CPU probe fallback for Pythia/GPT-2 after CUBLAS errors during semantic probe — not debugged, treated as workaround
- Pythia format sweep after Transluce prompt gave near-zero baseline bug; user explicitly wanted to "play with the pythia formatting"
- Pythia rerun on `answer_prompt` format after initial auto-selection picked `compare_two` (bug=100% but n_valid=2)
- GPT-2 ablation switched to logit-based eval using R12's discriminating-token method after text-match eval gave n_valid=2-17

FILES CHANGED:
- 9_8_research/experimental/16_transluce_hypothesis/run_experiment.py — Exp 16 main script (~520 lines)
- 9_8_research/experimental/16_transluce_hypothesis/run_l31_control.py — L31 random control (follow-up to Exp 16)
- 9_8_research/experimental/16_transluce_hypothesis/run_proper_transluce.py — Exp 16b main (gradient attribution + probe)
- 9_8_research/experimental/16_transluce_hypothesis/run_proper_followup.py — Pythia format sweep + GPT-2 logit ablation + CPU probes
- 9_8_research/experimental/16_transluce_hypothesis/run_pythia_answerprompt.py — Pythia rerun on `answer_prompt`
- 9_8_research/experimental/16_transluce_hypothesis/probe_texts.py — 80 probe texts (20 per theme × 4 themes)
- 9_8_research/research_journal_even_heads.md — added Experiment 16 and 16b entries after R12
- docs/experiments/04-14_multi_9_8_transluce_hypothesis.md — detailed report for Exp 16
- docs/experiments/04-14_multi_9_8_transluce_replication.md — detailed report for Exp 16b

RESULTS:
- Exp 16 (format-differential): real but small selective ablation effect on Llama (100% → 86.3%) that doesn't support Transluce's direction; Pythia no effect; GPT-2 11% baseline error
- Exp 16b (proper Transluce): clean replication in Pythia (72.3% → 7.1%), weak in Llama (40.6% → 27.3% with n_valid=22), untestable on GPT-2 (0% baseline). Gravity is strongest theme across all three models in semantic probe; sept11/bible barely replicate.
