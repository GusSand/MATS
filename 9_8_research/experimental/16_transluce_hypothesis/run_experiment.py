#!/usr/bin/env python3
"""
Experiment 16: Test Transluce Hypothesis on Llama-3.1-8B-Instruct, Pythia-160M, GPT-2
======================================================================================

Hypothesis (Transluce, Oct 2024): The 9.11 > 9.8 bug in Llama is caused by
spurious MLP neurons that fire on "9.11" in contexts like:
  - September 11th / dates  (9/11 attacks)
  - Bible chapter:verse     (9:8, 9:11)
  - Software/gravity        (9.8 m/s^2, version 9.11)
These neurons pull the model toward treating 9.11 as a sequential label
(date, verse, version) rather than a decimal magnitude.

Contribution: test whether these neurons are also present in Pythia-160M
(which has the bug) but absent/weaker in GPT-2-Small (which doesn't).
This would explain the cross-architecture pattern in Experiment 10.

Run on Paperspace A100-80GB:
  cd /home/paperspace/MATS/
  python3 9_8_research/experimental/16_transluce_hypothesis/run_experiment.py

Expected runtime: ~45 minutes total.

---- Design decisions (vs. original spec) ----

1. Target positions: last 4 tokens of the prompt (decision-forming positions),
   tokenizer-agnostic. Replaces fragile "9.11"-substring finder.

2. Llama neuron activations: hook down_proj pre-hook (input to down_proj
   = act_fn(gate) * up_proj). This is the canonical "neuron activation"
   semantically consistent with Transluce. For Pythia/GPT-2 (GELU/ReLU),
   hook act_fn output which is structurally equivalent.

3. Llama ablation: pre-hook on down_proj that zeros specified channels of its
   input. For Pythia/GPT-2, post-forward hook on act_fn zeroing channels.

4. GPT-2 logit-difference: reuses validated discriminating-token logic from
   exp_R12 (max-per-side, filter shared token IDs).

5. Wilson 95% CIs on all rates (reused from R3/R12).

6. Per-section checkpoint so crashes don't lose the 30-min Llama run.
"""

import sys
import os
import json
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np
import torch

# ── Reuse helpers from sibling reviewer experiments ───────────────────
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "9_8_research" / "reviewer_experiments" / "exp_R3_mlp_analysis"))
sys.path.insert(0, str(_REPO_ROOT / "9_8_research" / "reviewer_experiments" / "exp_R12_gpt2_reconciliation"))

# wilson_ci is defined identically in R3 and R12; import once.
from run_experiment import wilson_ci  # noqa: E402  (from R12 via sys.path order; R3 also has it)
# Import R12's GPT2Experiment for its battle-tested logit-diff method.
# We import lazily inside section 3 to avoid loading GPT2 modules unnecessarily.

# ── Config ────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

LLAMA_CHAT = "<|start_header_id|>user<|end_header_id|>\nWhich is bigger: 9.8 or 9.11?\n<|start_header_id|>assistant<|end_header_id|>\n"
LLAMA_SIMPLE = "Q: Which is bigger: 9.8 or 9.11?\nA:"
PYTHIA_CHAT = "Q: Which is bigger: 9.8 or 9.11?\nA:"
PYTHIA_SIMPLE = "Which is bigger: 9.8 or 9.11? Answer:"
GPT2_CHAT = "Q: Which is bigger: 9.8 or 9.11?\nA:"
GPT2_SIMPLE = "Which is bigger: 9.8 or 9.11? The larger number is"

N_TRIALS = 100          # ablation accuracy tests
N_ATTRIBUTION = 50      # trials for finding top neurons (deterministic; repeats for stability noise)
N_TOP = 50              # top-K differential neurons to ablate
RANDOM_SEED = 20260414

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ── Checkpointing ─────────────────────────────────────────────────────

def _json_default(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)

def save_checkpoint(data: dict):
    tmp = str(CHECKPOINT_PATH) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)
    os.replace(tmp, CHECKPOINT_PATH)

def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {}


# ── Generation + bug detection ────────────────────────────────────────

SAMPLING_TEMPERATURE = 0.6

def get_answer(model, tokenizer, prompt: str, max_new_tokens: int = 10,
               do_sample: bool = True, temperature: float = SAMPLING_TEMPERATURE) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id,
        )
    new_tokens = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def check_bug(answer: str) -> Optional[bool]:
    """True = bug (says 9.11 bigger). False = correct (says 9.8). None = unclear.

    Priority check for 9.11 first (its prefix '9.' is a substring of '9.8', so
    naive substring check on '9.8' would match '9.11 ...').
    """
    a = answer.lower()
    if "9.11" in a:
        return True
    if "9.8" in a:
        return False
    return None


def measure_bug_rate(model, tokenizer, prompt: str, n: int = N_TRIALS) -> Dict:
    """Stochastic sampling (temp=0.6) to measure bug rate as a true frequency."""
    bugs = 0
    valid = 0
    samples = []
    for i in range(n):
        ans = get_answer(model, tokenizer, prompt)
        r = check_bug(ans)
        if r is not None:
            valid += 1
            if r:
                bugs += 1
        if i < 2:
            samples.append(ans[:120])
    bug_rate = bugs / valid if valid > 0 else None
    ci = list(wilson_ci(bugs, valid)) if valid > 0 else [0.0, 0.0]
    return {
        "bug_rate": bug_rate,
        "n_valid": valid, "n_total": n,
        "n_bugs": bugs,
        "samples": samples,
        "ci": ci,
    }


# ── MLP neuron activation capture ─────────────────────────────────────

def _detect_arch(model) -> str:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return "llama"
    if hasattr(model, "gpt_neox"):
        return "pythia"
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return "gpt2"
    raise ValueError("Unknown architecture")


def _get_layers(model):
    arch = _detect_arch(model)
    if arch == "llama": return model.model.layers, arch
    if arch == "pythia": return model.gpt_neox.layers, arch
    if arch == "gpt2": return model.transformer.h, arch
    raise ValueError


def get_target_positions(seq_len: int, n: int = 4) -> List[int]:
    """Last n tokens — tokenizer-agnostic decision-forming positions."""
    return list(range(max(0, seq_len - n), seq_len))


def get_mlp_neuron_activations(model, tokenizer, prompt: str) -> Dict[int, np.ndarray]:
    """Capture per-neuron activations at target positions, averaged over positions.

    Returns {layer_idx: np.ndarray of shape [intermediate_size]}.

    For Llama: input to down_proj (i.e. act_fn(gate_proj(x)) * up_proj(x)) —
      canonical SwiGLU "neuron activation".
    For Pythia (GPT-NeoX): output of mlp.act (GELU) — its output enters
      mlp.dense_4h_to_h equivalently.
    For GPT-2: output of mlp.act (NewGELU) — enters mlp.c_proj.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    seq_len = inputs.input_ids.shape[1]
    target_positions = get_target_positions(seq_len)

    layers, arch = _get_layers(model)
    activations: Dict[int, torch.Tensor] = {}
    hooks = []

    def make_post_hook(layer_idx):
        def fn(module, inp, out):
            activations[layer_idx] = out.detach().cpu()
        return fn

    def make_pre_hook(layer_idx):
        def fn(module, inp):
            # input to down_proj: either tuple (x,) or tensor
            x = inp[0] if isinstance(inp, tuple) else inp
            activations[layer_idx] = x.detach().cpu()
        return fn

    for i, layer in enumerate(layers):
        if arch == "llama":
            h = layer.mlp.down_proj.register_forward_pre_hook(make_pre_hook(i))
        elif arch == "pythia":
            h = layer.mlp.act.register_forward_hook(make_post_hook(i))
        else:  # gpt2
            h = layer.mlp.act.register_forward_hook(make_post_hook(i))
        hooks.append(h)

    try:
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        for h in hooks:
            h.remove()

    result: Dict[int, np.ndarray] = {}
    for layer_idx, acts in activations.items():
        # acts: [1, seq_len, intermediate_size]
        if acts.dim() == 3:
            sel = acts[0, target_positions, :].float().mean(0)
        else:
            sel = acts[0].float()
        result[layer_idx] = sel.numpy()
    return result


# ── Differential neuron attribution ───────────────────────────────────

def compute_differential_neurons(
    model, tokenizer, buggy_prompt: str, correct_prompt: str,
    n: int = N_ATTRIBUTION, per_layer_k: int = 20, total_k: int = 200,
) -> List[Tuple[int, int, float]]:
    """Find neurons that fire more on buggy format than correct format.

    With greedy (deterministic) forward passes, one capture per prompt suffices
    — but we follow the original design and average n captures to be robust
    across any internal nondeterminism.
    """
    print(f"  Computing differential neuron activations (n={n}) ...")
    chat_list, simple_list = [], []
    for t in range(n):
        if t % 10 == 0:
            print(f"    Trial {t}/{n}")
        chat_list.append(get_mlp_neuron_activations(model, tokenizer, buggy_prompt))
        simple_list.append(get_mlp_neuron_activations(model, tokenizer, correct_prompt))

    layers = list(chat_list[0].keys())
    differential = {}
    for L in layers:
        chat_mean = np.mean([a[L] for a in chat_list], axis=0)
        simple_mean = np.mean([a[L] for a in simple_list], axis=0)
        differential[L] = chat_mean - simple_mean

    top_neurons: List[Tuple[int, int, float]] = []
    for L, diff in differential.items():
        idx = np.argsort(np.abs(diff))[-per_layer_k:]
        for ni in idx:
            top_neurons.append((int(L), int(ni), float(diff[ni])))
    top_neurons.sort(key=lambda x: abs(x[2]), reverse=True)
    return top_neurons[:total_k]


# ── Neuron ablation ───────────────────────────────────────────────────

def ablate_neurons_and_measure(
    model, tokenizer, neuron_list: List[Tuple[int, int, float]],
    prompt: str,
) -> Dict:
    """Zero the specified (layer, neuron_idx) channels and measure bug rate."""
    ablation_dict: Dict[int, List[int]] = {}
    for L, ni, _ in neuron_list:
        ablation_dict.setdefault(L, []).append(ni)

    layers, arch = _get_layers(model)
    hooks = []

    def make_pre_ablation_hook(indices):
        idx_t = torch.tensor(indices, dtype=torch.long)
        def fn(module, inp):
            x = inp[0] if isinstance(inp, tuple) else inp
            x = x.clone()
            x[..., idx_t.to(x.device)] = 0.0
            return (x,) if isinstance(inp, tuple) else x
        return fn

    def make_post_ablation_hook(indices):
        idx_t = torch.tensor(indices, dtype=torch.long)
        def fn(module, inp, out):
            out = out.clone()
            out[..., idx_t.to(out.device)] = 0.0
            return out
        return fn

    for i, layer in enumerate(layers):
        if i not in ablation_dict:
            continue
        idx = ablation_dict[i]
        if arch == "llama":
            h = layer.mlp.down_proj.register_forward_pre_hook(make_pre_ablation_hook(idx))
        elif arch == "pythia":
            h = layer.mlp.act.register_forward_hook(make_post_ablation_hook(idx))
        else:  # gpt2
            h = layer.mlp.act.register_forward_hook(make_post_ablation_hook(idx))
        hooks.append(h)

    try:
        result = measure_bug_rate(model, tokenizer, prompt)
    finally:
        for h in hooks:
            h.remove()
    return result


# ══════════════════════════════════════════════════════════════════════
# SECTION 1: LLAMA-3.1-8B-INSTRUCT
# ══════════════════════════════════════════════════════════════════════

def run_llama(ckpt: dict) -> dict:
    if "llama" in ckpt:
        print("\n[checkpoint] Restoring Llama section")
        return ckpt["llama"]

    print("\n" + "=" * 60)
    print("SECTION 1: Llama-3.1-8B-Instruct")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    print(f"Loading {model_id} ...")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    out: dict = {}

    print("\n1a. Baseline bug rates ...")
    chat = measure_bug_rate(model, tok, LLAMA_CHAT)
    simple = measure_bug_rate(model, tok, LLAMA_SIMPLE)
    print(f"  Chat format: bug_rate={chat['bug_rate']}  sample={chat.get('samples')!r}")
    print(f"  Simple format: bug_rate={simple['bug_rate']}  sample={simple.get('samples')!r}")
    out["baseline_chat"] = chat
    out["baseline_simple"] = simple

    print("\n1b. Differential neuron attribution ...")
    top_neurons = compute_differential_neurons(
        model, tok, buggy_prompt=LLAMA_CHAT, correct_prompt=LLAMA_SIMPLE)
    print(f"  Top differential neurons: {len(top_neurons)} (top-10):")
    for L, ni, s in top_neurons[:10]:
        print(f"    Layer {L:2d}, Neuron {ni:5d}: diff={s:+.3f}")
    out["top_differential_neurons"] = [
        {"layer": L, "neuron": n, "score": s} for L, n, s in top_neurons[:50]
    ]

    # Layer clusters
    n_layers = len(model.model.layers)
    split_1 = n_layers // 3
    split_2 = 2 * n_layers // 3
    mid = [t for t in top_neurons if split_1 <= t[0] <= split_2 and t[2] > 0]
    late = [t for t in top_neurons if t[0] > split_2 and t[2] > 0]
    early = [t for t in top_neurons if t[0] < split_1 and t[2] > 0]
    print(f"\n  Clusters: early(<{split_1})={len(early)}  "
          f"mid({split_1}-{split_2})={len(mid)}  late(>{split_2})={len(late)}")

    print("\n1c. Ablation tests ...")
    print("  Ablating top-50 ...")
    r_top50 = ablate_neurons_and_measure(model, tok, top_neurons[:50], LLAMA_CHAT)
    print(f"    bug_rate={r_top50['bug_rate']}  sample={r_top50.get('samples')!r}")
    print("  Ablating top-100 ...")
    r_top100 = ablate_neurons_and_measure(model, tok, top_neurons[:100], LLAMA_CHAT)
    print(f"    bug_rate={r_top100['bug_rate']}  sample={r_top100.get('samples')!r}")

    print("  Ablating mid-layer cluster (top 50) ...")
    r_mid = ablate_neurons_and_measure(model, tok, mid[:50], LLAMA_CHAT)
    print(f"    bug_rate={r_mid['bug_rate']}  sample={r_mid.get('samples')!r}")

    print("  Ablating late-layer cluster (top 50) ...")
    r_late = ablate_neurons_and_measure(model, tok, late[:50], LLAMA_CHAT)
    print(f"    bug_rate={r_late['bug_rate']}  sample={r_late.get('samples')!r}")

    intermediate_size = model.model.layers[0].mlp.down_proj.in_features
    rnd = [(random.randint(0, n_layers - 1), random.randint(0, intermediate_size - 1), 0.0)
           for _ in range(50)]
    print("  Ablating 50 random neurons (control) ...")
    r_rand = ablate_neurons_and_measure(model, tok, rnd, LLAMA_CHAT)
    print(f"    bug_rate={r_rand['bug_rate']}  sample={r_rand.get('samples')!r}")

    out["ablation"] = {
        "top50": r_top50, "top100": r_top100,
        "mid_cluster": r_mid, "late_cluster": r_late,
        "random_control": r_rand,
    }
    out["n_layers"] = n_layers
    out["intermediate_size"] = intermediate_size

    # Persist before unloading
    ckpt["llama"] = out
    save_checkpoint(ckpt)

    del model
    torch.cuda.empty_cache()
    return out


# ══════════════════════════════════════════════════════════════════════
# SECTION 2: PYTHIA-160M
# ══════════════════════════════════════════════════════════════════════

def run_pythia(ckpt: dict) -> dict:
    if "pythia" in ckpt:
        print("\n[checkpoint] Restoring Pythia section")
        return ckpt["pythia"]

    print("\n" + "=" * 60)
    print("SECTION 2: Pythia-160M")
    print("=" * 60)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "EleutherAI/pythia-160m"
    print(f"Loading {model_id} ...")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, device_map="auto")
    model.eval()

    out: dict = {}
    print("\n2a. Baseline bug rates ...")
    chat = measure_bug_rate(model, tok, PYTHIA_CHAT)
    simple = measure_bug_rate(model, tok, PYTHIA_SIMPLE)
    print(f"  Q&A format: bug_rate={chat['bug_rate']}  sample={chat.get('samples')!r}")
    print(f"  Simple format: bug_rate={simple['bug_rate']}  sample={simple.get('samples')!r}")
    out["baseline_qa"] = chat
    out["baseline_simple"] = simple

    print("\n2b. Differential neuron attribution ...")
    top_neurons = compute_differential_neurons(
        model, tok, buggy_prompt=PYTHIA_CHAT, correct_prompt=PYTHIA_SIMPLE)
    print(f"  Top differential neurons (top-10):")
    for L, ni, s in top_neurons[:10]:
        print(f"    Layer {L:2d}, Neuron {ni:5d}: diff={s:+.3f}")
    out["top_differential_neurons"] = [
        {"layer": L, "neuron": n, "score": s} for L, n, s in top_neurons[:30]
    ]

    n_layers = len(model.gpt_neox.layers)
    intermediate_size = model.gpt_neox.layers[0].mlp.dense_h_to_4h.out_features

    print("\n2c. Ablation tests ...")
    r_top50 = ablate_neurons_and_measure(model, tok, top_neurons[:50], PYTHIA_CHAT)
    print(f"  Top-50 ablation: bug_rate={r_top50['bug_rate']}  sample={r_top50.get('samples')!r}")
    rnd = [(random.randint(0, n_layers - 1), random.randint(0, intermediate_size - 1), 0.0)
           for _ in range(50)]
    r_rand = ablate_neurons_and_measure(model, tok, rnd, PYTHIA_CHAT)
    print(f"  Random-50 ablation: bug_rate={r_rand['bug_rate']}  sample={r_rand.get('samples')!r}")

    out["ablation"] = {"top50": r_top50, "random_control": r_rand}
    out["n_layers"] = n_layers
    out["intermediate_size"] = intermediate_size

    ckpt["pythia"] = out
    save_checkpoint(ckpt)

    del model
    torch.cuda.empty_cache()
    return out


# ══════════════════════════════════════════════════════════════════════
# SECTION 3: GPT-2-SMALL (control)
# ══════════════════════════════════════════════════════════════════════

def run_gpt2(ckpt: dict, llama_out: dict) -> dict:
    if "gpt2" in ckpt:
        print("\n[checkpoint] Restoring GPT-2 section")
        return ckpt["gpt2"]

    print("\n" + "=" * 60)
    print("SECTION 3: GPT-2-Small (control — no bug)")
    print("=" * 60)

    # Reuse R12's validated GPT2Experiment for logit-diff
    from run_experiment import GPT2Experiment  # from R12 via sys.path

    exp = GPT2Experiment("gpt2")
    model, tok = exp.model, exp.tokenizer

    out: dict = {}

    print("\n3a. Logit-based evaluation across X=1..9 ...")
    logit_diffs = []
    correct_flags = []
    for x in range(1, 10):
        prompt = f"Q: Which is bigger: {x}.8 or {x}.11?\nA:"
        r = exp.get_logit_difference(prompt, f"{x}.8", f"{x}.11")
        logit_diffs.append(r["logit_diff"])
        correct_flags.append(bool(r["correct_preferred"]))
    correct_count = sum(correct_flags)
    total = len(correct_flags)
    error_rate = 1 - correct_count / total
    print(f"  Correct: {correct_count}/{total}  error_rate={error_rate:.1%}")
    print(f"  Mean logit diff: {np.mean(logit_diffs):.3f}")
    out["logit_eval"] = {
        "correct": correct_count, "total": total,
        "error_rate": float(error_rate),
        "mean_logit_diff": float(np.mean(logit_diffs)),
        "per_x_logit_diff": [float(d) for d in logit_diffs],
    }

    print("\n3b. Differential neuron attribution ...")
    top_neurons = compute_differential_neurons(
        model, tok, buggy_prompt=GPT2_CHAT, correct_prompt=GPT2_SIMPLE,
        n=20,  # fewer trials for speed, per original design
        per_layer_k=10, total_k=100,
    )
    print(f"  GPT-2 top differential neurons (top-10):")
    for L, ni, s in top_neurons[:10]:
        print(f"    Layer {L:2d}, Neuron {ni:5d}: diff={s:+.3f}")

    llama_scores = [abs(d["score"]) for d in llama_out.get("top_differential_neurons", [])[:50]]
    gpt2_scores = [abs(s) for _, _, s in top_neurons[:50]]
    ratio = None
    if llama_scores and gpt2_scores and np.mean(gpt2_scores) > 0:
        ratio = float(np.mean(llama_scores) / np.mean(gpt2_scores))
    print(f"\n  Differential score comparison:")
    print(f"    Llama top-50 mean |score|: {np.mean(llama_scores) if llama_scores else 'N/A'}")
    print(f"    GPT-2 top-50 mean |score|: {np.mean(gpt2_scores) if gpt2_scores else 'N/A'}")
    print(f"    Ratio (Llama/GPT-2): {ratio}")

    out["top_differential_neurons"] = [
        {"layer": L, "neuron": n, "score": s} for L, n, s in top_neurons[:20]
    ]
    out["score_comparison"] = {
        "llama_mean_abs_score": float(np.mean(llama_scores)) if llama_scores else None,
        "gpt2_mean_abs_score": float(np.mean(gpt2_scores)) if gpt2_scores else None,
        "ratio_llama_over_gpt2": ratio,
    }

    ckpt["gpt2"] = out
    save_checkpoint(ckpt)

    exp.cleanup()
    torch.cuda.empty_cache()
    return out


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Timestamp: {TIMESTAMP}")

    t_start = time.time()
    ckpt = load_checkpoint()

    llama_out = run_llama(ckpt)
    pythia_out = run_pythia(ckpt)
    gpt2_out = run_gpt2(ckpt, llama_out)

    total_time = time.time() - t_start

    results = {
        "experiment": "16_transluce_hypothesis",
        "timestamp": TIMESTAMP,
        "hypothesis": "Transluce: spurious Sept11/Bible/gravity neurons cause 9.11 bug",
        "config": {
            "n_trials": N_TRIALS, "n_attribution": N_ATTRIBUTION,
            "n_top": N_TOP, "seed": RANDOM_SEED,
            "prompts": {
                "llama_chat": LLAMA_CHAT, "llama_simple": LLAMA_SIMPLE,
                "pythia_chat": PYTHIA_CHAT, "pythia_simple": PYTHIA_SIMPLE,
                "gpt2_chat": GPT2_CHAT, "gpt2_simple": GPT2_SIMPLE,
            },
        },
        "total_time_seconds": total_time,
        "llama": llama_out,
        "pythia": pythia_out,
        "gpt2": gpt2_out,
    }

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY: Transluce Hypothesis Test Results")
    print("=" * 60)

    l_base = llama_out["baseline_chat"].get("bug_rate")
    l_top50 = llama_out["ablation"]["top50"].get("bug_rate")
    l_rand = llama_out["ablation"]["random_control"].get("bug_rate")
    print(f"\nQ1: Llama ablation effect")
    print(f"  Baseline chat: {l_base}")
    print(f"  Top-50 ablated: {l_top50}")
    print(f"  Random control: {l_rand}")
    verdict_q1 = "inconclusive"
    if l_base is not None and l_top50 is not None:
        delta = l_top50 - l_base
        if delta < -0.10: verdict_q1 = "YES — strong evidence for Transluce hypothesis"
        elif delta < -0.03: verdict_q1 = "PARTIAL — modest evidence"
        else: verdict_q1 = "WEAK — minimal effect"
    print(f"  VERDICT: {verdict_q1}")
    results["verdict_q1"] = verdict_q1

    p_base = pythia_out["baseline_qa"].get("bug_rate")
    p_top50 = pythia_out["ablation"]["top50"].get("bug_rate")
    print(f"\nQ2: Pythia ablation effect")
    print(f"  Baseline: {p_base}  Top-50 ablated: {p_top50}")
    verdict_q2 = "inconclusive"
    if p_base is not None and p_top50 is not None:
        pdel = p_top50 - p_base
        verdict_q2 = "same_effect" if pdel < -0.05 else "different_effect"
    print(f"  VERDICT: {verdict_q2}")
    results["verdict_q2"] = verdict_q2

    g_err = gpt2_out["logit_eval"]["error_rate"]
    ratio = gpt2_out["score_comparison"].get("ratio_llama_over_gpt2")
    print(f"\nQ3: GPT-2 comparison")
    print(f"  GPT-2 error rate (logit-based): {g_err:.1%}")
    print(f"  Llama/GPT-2 differential score ratio: {ratio}")
    verdict_q3 = "inconclusive"
    if ratio is not None:
        verdict_q3 = "stronger_in_llama" if ratio > 2.0 else "similar_strength"
    print(f"  VERDICT: {verdict_q3}")
    results["verdict_q3"] = verdict_q3

    outfile = RESULTS_DIR / f"results_{TIMESTAMP}.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"\n✓ Results saved to: {outfile}")
    print(f"Total time: {total_time / 60:.1f} min")


if __name__ == "__main__":
    main()
