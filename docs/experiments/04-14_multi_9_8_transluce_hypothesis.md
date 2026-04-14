# Experiment 16: Format-Differential MLP Neurons and the 9.11 > 9.8 Bug

**Date**: 2026-04-14
**Models**: Llama-3.1-8B-Instruct, Pythia-160M, GPT-2-Small
**Dataset**: Synthetic decimal-comparison prompts (9.8 vs 9.11, X.8 vs X.11 for X=1..9)
**Location**: `9_8_research/experimental/16_transluce_hypothesis/`
**Status**: Complete. Follow-up experiment 16b (proper Transluce replication) runs separately.

---

## 1. Purpose and Framing

Transluce (October 2024) reported that Llama's 9.11 > 9.8 decimal-comparison bug is caused by MLP neurons tied to three spurious concept clusters — September 11 (dates), Bible verses (chapter:verse), and 9.8 m/s² (gravity) — that pull the model toward treating "9.11" as a sequential label rather than a decimal.

This experiment tested a **related but distinct hypothesis**: that neurons which activate differentially between the *chat* format (where the bug is strong) and a *simple* Q&A format (where it is weaker) are causally responsible for the bug, and that such neurons exist in Pythia-160M (bug present) but are weaker or absent in GPT-2-Small (bug absent in most evaluations).

### Methodological caveat
After running and analyzing the results, we recognized that this is **not** a direct replication of Transluce's methodology. Transluce used a single format and gradient-based attribution (`e · ∂z/∂e`). We used a format contrast (activation-mean difference). These measure different things. This report honestly reports what we measured; Experiment 16b replicates Transluce's actual method.

---

## 2. Methods

### 2.1 Bug-rate measurement
- Stochastic decoding: `do_sample=True`, `temperature=0.6`, `max_new_tokens=10`
- N=100 trials per condition
- Wilson 95% CIs reported
- Bug detection: output contains "9.11" → bug; contains "9.8" (and not "9.11") → correct; else ambiguous (excluded from rate, counted in n_valid)

### 2.2 Differential attribution
- Forward pass on each format (buggy and correct), capturing MLP neuron activations
- "Neuron activation" definitions:
  - Llama (SwiGLU): `act_fn(gate_proj(x)) * up_proj(x)` — captured via `down_proj.register_forward_pre_hook`
  - Pythia / GPT-2: output of `mlp.act` — captured via post-hook
- Target positions: last 4 tokens of the prompt (tokenizer-agnostic; the original spec's "9.11"-substring finder was fragile)
- Score per neuron: `chat_mean − simple_mean` averaged over 50 forward passes per prompt
- Top-200 retained by |score|; top-50 and top-100 selected for ablation

### 2.3 Ablation
- Llama: pre-hook on `down_proj` zeroing specified channels of its input (zeroes the neuron pre-down-projection)
- Pythia / GPT-2: post-hook on `mlp.act` zeroing specified channels of the activation output

### 2.4 Prompts
| Model | "Buggy" format | "Correct" format |
|---|---|---|
| Llama | Llama-3.1 chat template with user turn | `Q: Which is bigger: 9.8 or 9.11?\nA:` |
| Pythia | `Q: Which is bigger: 9.8 or 9.11?\nA:` | `Which is bigger: 9.8 or 9.11? Answer:` |
| GPT-2 | `Q: Which is bigger: 9.8 or 9.11?\nA:` | `Which is bigger: 9.8 or 9.11? The larger number is` |

### 2.5 GPT-2 bug evaluation
GPT-2 outputs text-completion garbage, so a text-match bug-rate is unreliable. We used logit-based evaluation with R12's discriminating-token method (max-per-side logit, filter shared token IDs). Swept X=1..9.

### 2.6 Controls
- Main script: 50 uniformly-random neurons over all layers
- Follow-up (`run_l31_control.py`): 50 random at layer 31, 50 random at layers 28–31, 100 random at layer 31

### 2.7 Code reuse
- `wilson_ci`: imported from `reviewer_experiments/exp_R12_gpt2_reconciliation/run_experiment.py`
- `GPT2Experiment.get_logit_difference`: imported from the same file (battle-tested discriminating-token approach)
- Hook patterns adapted from `reviewer_experiments/exp_R3_mlp_analysis/run_experiment.py`

---

## 3. Results

### 3.1 Llama-3.1-8B-Instruct

**Baselines and targeted ablations (N=100, temp=0.6):**

| Condition | Bug rate | 95% CI | n_valid |
|---|---|---|---|
| Baseline chat | 100.0% | [96.3, 100] | 100/100 |
| Baseline simple | 78.8% | [69.7, 85.7] | 99/100 |
| Top-50 by \|s\| (mixed sign) | **86.3%** | [78.0, 91.8] | 95/100 |
| Top-100 by \|s\| (mixed sign) | **81.0%** | [72.2, 87.5] | 100/100 |
| Mid cluster (L10–21, s>0, 15 neurons) | 100.0% | [96.3, 100] | 100/100 |
| Late cluster (L22–31, s>0, 50 neurons) | 100.0% | [96.3, 100] | 100/100 |
| Random 50 (uniform over all layers) | 100.0% | [96.3, 100] | 100/100 |

**Layer-matched random controls:**

| Condition | Bug rate | 95% CI |
|---|---|---|
| Random 50 at L31 only | 100.0% | [96.3, 100] |
| Random 50 at L28–31 | 100.0% | [96.3, 100] |
| Random 100 at L31 only | 100.0% | [96.3, 100] |

**Top neuron distribution:** 16/20 top-|s| neurons are at layer 31; scores are roughly balanced in sign (top values: +10.22, −10.27, +9.48, +8.12, +7.62, −7.29, −6.24, …). Cluster composition of top-200: early (<L10) = 3, mid (L10–21) = 15, late (L22–31) = 83 (all with s>0).

### 3.2 Pythia-160M

| Condition | Bug rate | 95% CI | n_valid |
|---|---|---|---|
| Baseline Q&A | 82.1% | [64.4, 92.1] | 28/100 |
| Baseline simple | 73.7% | [62.8, 82.3] | 76/100 |
| Top-50 by \|s\| ablated | 83.5% | [73.9, 90.1] | 79/100 |
| Random 50 ablated | 83.3% | [68.1, 92.1] | 36/100 |

Top neurons concentrate at layers 9–11; scores all positive in top-10 (+6.45, +4.03, +3.63, +3.59, +3.22, +2.86, +2.60, +2.15, +2.08, +1.99).

### 3.3 GPT-2-Small (logit-based)

- Correct on 8/9 X-values; error rate 11.1% (matches R12 exactly)
- X=8 returned −Infinity due to the R12 shared-token edge case
- Llama top-50 mean |s| = 3.56; GPT-2 top-50 mean |s| = 1.65; ratio ≈ **2.16×**

### 3.4 Timing
Total wall-clock: 4.4 min for main run (attribution uses forward passes not `generate`; stochastic sampling at max_new_tokens=10 is fast). Follow-up L31 controls: ~3 min. Original spec estimated ~45 min on the same hardware.

---

## 4. Interpretation (flagged as author's, not direct data)

### 4.1 What the data supports
A specific, non-random subset of late-layer (L28–31) MLP neurons in Llama is causally implicated in the bug. The evidence is that layer-matched random controls (50 and 100 neurons at L31, 50 at L28–31) leave the bug rate at 100%, while the targeted top-50/100 move it to 86.3% / 81.0%.

### 4.2 What the data does NOT support
The Transluce hypothesis **as stated**. Transluce describes positive-firing bug-promoting concept neurons. We observed:
- Positive-s-only ablations had zero effect (mid and late clusters both 100%)
- The effect only appears once negative-s neurons are included (top-K by |s|)
- Pythia showed no ablation effect at all under this metric

The data's most natural reading is that our **differential-activation metric does not select the same neurons Transluce's attribution method would**, and consequently cannot confirm or refute their specific mechanistic claims.

### 4.3 Methodological caveats
- **Layer bias**: residual-stream norms grow through Llama's depth, so raw activation differences naturally grow at late layers. Without per-layer normalization, "top differential neurons" is partially just "neurons at the layer with the biggest activations". This could explain the layer-31 dominance independent of any causal story.
- **Destabilization**: top-50 ablation reduced n_valid from 100 to 95 — 5 responses contained neither "9.8" nor "9.11". Some of the apparent bug-rate improvement is the model going off-script rather than cleanly flipping to the correct answer.
- **Metric mismatch**: Transluce's `e · ∂z/∂e` attribution is a linearized causal metric. Our `chat − simple` activation contrast is correlational. These cannot be compared directly.

### 4.4 The "negative-score neurons" finding
The most interesting single observation is that ablating *only positive-s* neurons (neurons that fire more in chat) does nothing, but ablating a mix of positive and negative-s neurons does. This suggests that neurons which fire *less* in the buggy format than the correct format — and are thus "absent" from the bug circuit by the differential metric — are nonetheless carrying a nonzero causal contribution to the bug. This would require activation patching (not just zeroing) to confirm, since zeroing a neuron that is already low in the buggy format is an unusual intervention.

---

## 5. Relationship to Other Experiments

- **Experiment 10** (cross-model): found Pythia-160M has the bug, GPT-2 does not. Experiment 16 examined whether format-differential MLP neurons explain this cross-architecture pattern; the answer is "this metric does not support that hypothesis for Pythia".
- **Experiment R3** (MLP vs attention): found that attention patching at Layer 10 fixes the bug while MLP patching does not, in the Llama chat → simple direction. Experiment 16 ablates individual MLP neurons rather than whole-layer MLP outputs; the 14pp effect from top-50 ablation is consistent with MLP contributing a modest signal on top of the dominant attention mechanism.
- **Experiment R12** (GPT-2 reconciliation): produced the logit-difference method with shared-token filtering that this experiment reuses.
- **Experiment 16b** (in-progress at time of writing): proper gradient-attribution replication of Transluce's method, plus semantic probe on curated Sept-11 / Bible / gravity / neutral text sets.

---

## 6. Files

### Code
- [run_experiment.py](../../9_8_research/experimental/16_transluce_hypothesis/run_experiment.py) — main experiment
- [run_l31_control.py](../../9_8_research/experimental/16_transluce_hypothesis/run_l31_control.py) — layer-31 random controls
- [probe_texts.py](../../9_8_research/experimental/16_transluce_hypothesis/probe_texts.py) — semantic probe texts (used in 16b)

### Data
- `9_8_research/experimental/16_transluce_hypothesis/results_20260414_140942.json`
- `9_8_research/experimental/16_transluce_hypothesis/l31_control_20260414_144929.json`
- `9_8_research/experimental/16_transluce_hypothesis/checkpoint.json`
- `9_8_research/experimental/16_transluce_hypothesis/output.log`
