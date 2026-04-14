#!/usr/bin/env python3
"""
Experiment 16b follow-up:
 - Pythia format sweep + attribution/ablation/probe on the buggiest format
 - GPT-2 with LOGIT-BASED bug-rate eval during ablation (reusing R12's method)
 - Probes run on CPU (Llama probe worked fine on GPU; Pythia/GPT-2 hit CUBLAS
   errors after heavy hook traffic. CPU is trivially fast for these small models.)

Leaves the Llama section from 16b intact (already in checkpoint_proper.json).
Writes fresh results to results_followup_<ts>.json and updates the main
checkpoint with corrected pythia/gpt2 sections.
"""

import sys, os, json, time, random, importlib.util
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Reuse main-script helpers
_spec = importlib.util.spec_from_file_location("exp16b_main", _HERE / "run_proper_transluce.py")
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
compute_attribution = _mod.compute_attribution
top_k_from_attribution = _mod.top_k_from_attribution
semantic_probe = _mod.semantic_probe
make_attribution_prompts = _mod.make_attribution_prompts
ablate_neurons_and_measure = _mod.ablate_neurons_and_measure
measure_bug_rate = _mod.measure_bug_rate
_detect_arch = _mod._detect_arch
_get_layers = _mod._get_layers
_json_default = _mod._json_default
CHECKPOINT_PATH = _mod.CHECKPOINT_PATH
N_TRIALS = _mod.N_TRIALS
N_TOP = _mod.N_TOP

# Reuse R12 logit-diff
sys.path.insert(0, str(_REPO_ROOT / "9_8_research" / "reviewer_experiments" / "exp_R12_gpt2_reconciliation"))
from run_experiment import GPT2Experiment  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
random.seed(20260414); np.random.seed(20260414); torch.manual_seed(20260414)


# ── Pythia format sweep ───────────────────────────────────────────────

PYTHIA_FORMATS = {
    "transluce":      "Which is bigger, 9.11 or 9.8?",
    "qa_8_first":     "Q: Which is bigger: 9.8 or 9.11?\nA:",
    "qa_11_first":    "Q: Which is bigger: 9.11 or 9.8?\nA:",
    "answer_prompt":  "Which is bigger: 9.8 or 9.11? Answer:",
    "answer_prompt_rev": "Which is bigger: 9.11 or 9.8? Answer:",
    "larger_is":      "Between 9.8 and 9.11, the larger number is",
    "compare_two":    "Compare two numbers: 9.11 and 9.8. Which one is bigger?",
    "is_bigger":      "Is 9.11 bigger than 9.8? Answer:",
}

# Attribution prompts per X for Pythia — parameterized by format
def pythia_attribution_prompts(fmt_template: str):
    """Build attribution prompts by substituting varying X.A / X.B decimal pairs
    into the chosen format. We don't have direct next-token discrimination in
    the generic template, so fall back to a committed-answer completion style:
    append ' 9.' or equivalent so next token discriminates."""
    prompts = []
    for X in range(1, 12):
        for A, B in [("8", "11"), ("9", "10"), ("8", "13")]:
            p_q = fmt_template.replace("9.8", f"{X}.{A}").replace("9.11", f"{X}.{B}")
            # Commit to numeric completion
            p = p_q + f" {X}."
            prompts.append({"prompt": p, "X": X, "correct": A, "wrong": B})
    return prompts


# ── CPU-fallback probe ────────────────────────────────────────────────

def semantic_probe_on_cpu(model, tok, neuron_list):
    """Run probe on CPU to avoid CUBLAS issues after heavy GPU hook traffic."""
    print("   (moving model to CPU for probe)")
    model = model.cpu()
    # Update DEVICE for the internal calls inside semantic_probe -> neuron_activations_at_last_token
    # That helper uses `tokenizer(...).to(DEVICE)` — we need to patch.
    # Easiest path: monkey-patch _mod.DEVICE and torch default.
    orig_device = _mod.DEVICE
    _mod.DEVICE = "cpu"
    try:
        return semantic_probe(model, tok, neuron_list)
    finally:
        _mod.DEVICE = orig_device


# ── GPT-2 logit-based bug-rate ────────────────────────────────────────

def gpt2_logit_bug_rate(exp: GPT2Experiment, x_values=range(1, 10),
                        skip_x8: bool = True) -> Dict:
    """Measure error rate via logit-diff across X values (R12 approach).
    Returns dict with error_rate and per-X details. Deterministic (single
    forward per X), not sample-based."""
    per_x, errors = [], 0
    valid = 0
    for X in x_values:
        if skip_x8 and X == 8: continue
        prompt = f"Q: Which is bigger: {X}.8 or {X}.11?\nA:"
        r = exp.get_logit_difference(prompt, f"{X}.8", f"{X}.11")
        ld = r["logit_diff"]
        if np.isinf(ld) or np.isnan(ld):
            continue
        valid += 1
        is_error = not r["correct_preferred"]
        if is_error: errors += 1
        per_x.append({"x": X, "logit_diff": float(ld), "error": bool(is_error)})
    return {
        "error_rate": errors / valid if valid else None,
        "n_errors": errors, "n_valid": valid,
        "per_x": per_x,
    }


def ablate_and_measure_gpt2_logit(exp: GPT2Experiment, neuron_list, x_values=range(1, 10)):
    """Ablate given neurons on the GPT-2 model (exp.model) and measure logit-based error rate."""
    model, tok = exp.model, exp.tokenizer
    # Build ablation hooks on mlp.act
    ablation_dict: Dict[int, List[int]] = {}
    for L, ni, _ in neuron_list:
        ablation_dict.setdefault(L, []).append(ni)
    hooks = []
    def make_hook(idx):
        idx_t = torch.tensor(idx, dtype=torch.long, device=DEVICE)
        def fn(module, inp, out):
            out = out.clone(); out[..., idx_t] = 0.0; return out
        return fn
    for i, block in enumerate(model.transformer.h):
        if i in ablation_dict:
            h = block.mlp.act.register_forward_hook(make_hook(ablation_dict[i]))
            hooks.append(h)
    try:
        return gpt2_logit_bug_rate(exp, x_values=x_values)
    finally:
        for h in hooks: h.remove()


# ── Checkpoint helpers ────────────────────────────────────────────────

def load_ckpt():
    with open(CHECKPOINT_PATH) as f: return json.load(f)

def save_ckpt(d):
    tmp = str(CHECKPOINT_PATH) + ".tmp"
    with open(tmp, "w") as f: json.dump(d, f, indent=2, default=_json_default)
    os.replace(tmp, CHECKPOINT_PATH)


# ── Pythia section ────────────────────────────────────────────────────

def run_pythia_sweep():
    print("\n" + "=" * 60)
    print("PYTHIA: Format sweep + proper attribution on buggiest format")
    print("=" * 60)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "EleutherAI/pythia-160m"
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32,
                                                  device_map="auto")
    model.eval()

    out = {"format_sweep": {}}

    # 1. Sweep baseline bug rate across formats
    print("\n1. Baseline bug rate across formats (N=100, temp=0.6) ...")
    best_fmt = None; best_rate = -1.0
    for key, fmt in PYTHIA_FORMATS.items():
        r = measure_bug_rate(model, tok, fmt)
        out["format_sweep"][key] = {"prompt": fmt, **r}
        rate = r.get("bug_rate")
        rate_str = f"{rate:.3f}" if rate is not None else "None"
        print(f"   {key:18s} bug_rate={rate_str}  n_valid={r['n_valid']}  prompt={fmt!r}")
        if rate is not None and rate > best_rate:
            best_rate = rate; best_fmt = key

    print(f"\n   → Selected '{best_fmt}' (bug_rate={best_rate:.3f}) for attribution/ablation")
    out["selected_format"] = best_fmt
    out["selected_prompt"] = PYTHIA_FORMATS[best_fmt]

    if best_rate < 0.15:
        print(f"   ⚠ WARNING: best baseline bug rate is only {best_rate:.1%} — "
              "ablation effects will be uninterpretable. Running anyway for record.")

    # 2. Attribution using the selected format's prompt template
    print("\n2. Gradient attribution on Pythia (selected format) ...")
    prompts = pythia_attribution_prompts(PYTHIA_FORMATS[best_fmt])
    t0 = time.time()
    attr = compute_attribution(model, tok, prompts)
    print(f"   Done in {time.time()-t0:.1f}s  shape={attr.shape}  n_prompts={len(prompts)}")
    top_pos = top_k_from_attribution(attr, k=N_TOP, sign="positive")
    top_abs = top_k_from_attribution(attr, k=N_TOP, sign="abs")
    out["top_positive_attribution"] = [{"layer": L, "neuron": n, "score": s}
                                        for L, n, s in top_pos]
    out["top_abs_attribution"] = [{"layer": L, "neuron": n, "score": s}
                                   for L, n, s in top_abs]
    for L, n, s in top_pos[:10]:
        print(f"     L{L:2d} N{n:5d}: attr={s:+.4f}")

    # 3. Ablation (use selected format as the test prompt)
    test_prompt = PYTHIA_FORMATS[best_fmt]
    print("\n3. Ablation tests (bug-rate on selected format) ...")
    baseline = out["format_sweep"][best_fmt]
    r_pos = ablate_neurons_and_measure(model, tok, top_pos, test_prompt)
    r_abs = ablate_neurons_and_measure(model, tok, top_abs, test_prompt)
    inter = model.gpt_neox.layers[0].mlp.dense_h_to_4h.out_features
    layer_counts = {}
    for L, _, _ in top_pos: layer_counts[L] = layer_counts.get(L, 0) + 1
    rnd = [(L, random.randint(0, inter - 1), 0.0)
           for L, cnt in layer_counts.items() for _ in range(cnt)]
    r_rnd = ablate_neurons_and_measure(model, tok, rnd, test_prompt)
    print(f"   baseline:        bug_rate={baseline['bug_rate']}  n={baseline['n_valid']}")
    print(f"   top-50 positive: bug_rate={r_pos['bug_rate']}  n={r_pos['n_valid']}")
    print(f"   top-50 mixed:    bug_rate={r_abs['bug_rate']}  n={r_abs['n_valid']}")
    print(f"   random-matched:  bug_rate={r_rnd['bug_rate']}  n={r_rnd['n_valid']}")
    out["ablation"] = {"top50_positive": r_pos, "top50_abs": r_abs,
                       "random_layer_matched": r_rnd,
                       "baseline_selected_format": baseline}

    # 4. Semantic probe on CPU
    print("\n4. Semantic probe on CPU ...")
    try:
        if DEVICE == "cuda":
            torch.cuda.empty_cache(); torch.cuda.synchronize()
        probe = semantic_probe_on_cpu(model, tok, top_pos)
        summary = {}
        for theme in ["sept11", "bible", "gravity"]:
            scores = np.array(list(probe["selectivity"][theme].values()))
            summary[theme] = {
                "mean_selectivity": float(np.nanmean(scores)),
                "median_selectivity": float(np.nanmedian(scores)),
                "n_positive": int(np.sum(scores > 0)),
                "max_selectivity": float(np.nanmax(scores)),
            }
            print(f"   {theme:8s}: mean={summary[theme]['mean_selectivity']:+.3f}  "
                  f"n>0={summary[theme]['n_positive']}/{len(scores)}  "
                  f"max={summary[theme]['max_selectivity']:+.3f}")
        out["semantic_probe"] = probe
        out["semantic_probe_summary"] = summary
    except Exception as e:
        print(f"   PROBE FAILED (CPU fallback): {type(e).__name__}: {e}")
        out["semantic_probe"] = None
        out["semantic_probe_error"] = f"{type(e).__name__}: {e}"

    del model
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return out


# ── GPT-2 section ─────────────────────────────────────────────────────

def run_gpt2_logit():
    print("\n" + "=" * 60)
    print("GPT-2: Logit-based bug-rate eval + probe on CPU")
    print("=" * 60)
    exp = GPT2Experiment("gpt2")
    out = {}

    # 1. Baseline via logit diff (sweep X=1..9, skip X=8 edge case)
    print("\n1. Baseline (R12 logit-diff, X∈{1..9}\\{8}) ...")
    base = gpt2_logit_bug_rate(exp)
    print(f"   error_rate={base['error_rate']:.3f}  n_valid={base['n_valid']}")
    out["baseline_logit"] = base

    # 2. Attribution + ablation use exp.model
    print("\n2. Gradient attribution ...")
    prompts = make_attribution_prompts()
    t0 = time.time()
    attr = compute_attribution(exp.model, exp.tokenizer, prompts)
    print(f"   Done in {time.time()-t0:.1f}s  shape={attr.shape}")
    top_pos = top_k_from_attribution(attr, k=N_TOP, sign="positive")
    top_abs = top_k_from_attribution(attr, k=N_TOP, sign="abs")
    out["top_positive_attribution"] = [{"layer": L, "neuron": n, "score": s} for L, n, s in top_pos]
    out["top_abs_attribution"] = [{"layer": L, "neuron": n, "score": s} for L, n, s in top_abs]
    for L, n, s in top_pos[:10]:
        print(f"     L{L:2d} N{n:5d}: attr={s:+.4f}")

    # 3. Ablations
    print("\n3. Ablations (logit-based error rate) ...")
    r_pos = ablate_and_measure_gpt2_logit(exp, top_pos)
    r_abs = ablate_and_measure_gpt2_logit(exp, top_abs)
    inter = exp.model.transformer.h[0].mlp.c_proj.weight.shape[0]
    layer_counts = {}
    for L, _, _ in top_pos: layer_counts[L] = layer_counts.get(L, 0) + 1
    rnd = [(L, random.randint(0, inter - 1), 0.0)
           for L, cnt in layer_counts.items() for _ in range(cnt)]
    r_rnd = ablate_and_measure_gpt2_logit(exp, rnd)
    print(f"   baseline:        err={base['error_rate']:.3f}  n={base['n_valid']}")
    print(f"   top-50 positive: err={r_pos['error_rate']:.3f}  n={r_pos['n_valid']}")
    print(f"   top-50 mixed:    err={r_abs['error_rate']:.3f}  n={r_abs['n_valid']}")
    print(f"   random-matched:  err={r_rnd['error_rate']:.3f}  n={r_rnd['n_valid']}")
    out["ablation_logit"] = {
        "baseline": base, "top50_positive": r_pos,
        "top50_abs": r_abs, "random_layer_matched": r_rnd,
    }

    # 4. Probe on CPU
    print("\n4. Semantic probe on CPU ...")
    try:
        if DEVICE == "cuda":
            torch.cuda.empty_cache(); torch.cuda.synchronize()
        probe = semantic_probe_on_cpu(exp.model, exp.tokenizer, top_pos)
        summary = {}
        for theme in ["sept11", "bible", "gravity"]:
            scores = np.array(list(probe["selectivity"][theme].values()))
            summary[theme] = {
                "mean_selectivity": float(np.nanmean(scores)),
                "median_selectivity": float(np.nanmedian(scores)),
                "n_positive": int(np.sum(scores > 0)),
                "max_selectivity": float(np.nanmax(scores)),
            }
            print(f"   {theme:8s}: mean={summary[theme]['mean_selectivity']:+.3f}  "
                  f"n>0={summary[theme]['n_positive']}/{len(scores)}  "
                  f"max={summary[theme]['max_selectivity']:+.3f}")
        out["semantic_probe"] = probe
        out["semantic_probe_summary"] = summary
    except Exception as e:
        print(f"   PROBE FAILED (CPU fallback): {type(e).__name__}: {e}")
        out["semantic_probe"] = None
        out["semantic_probe_error"] = f"{type(e).__name__}: {e}"

    exp.cleanup()
    return out


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}  TS: {TIMESTAMP}")
    t0 = time.time()

    pythia_out = run_pythia_sweep()
    gpt2_out = run_gpt2_logit()

    # Update main checkpoint (preserve llama, overwrite pythia/gpt2)
    ckpt = load_ckpt()
    ckpt["pythia"] = pythia_out
    ckpt["gpt2"] = gpt2_out
    save_ckpt(ckpt)

    # Save followup-specific results file
    out = {
        "experiment": "16b_followup",
        "timestamp": TIMESTAMP,
        "total_time_seconds": time.time() - t0,
        "pythia": pythia_out,
        "gpt2": gpt2_out,
    }
    outfile = _HERE / f"results_followup_{TIMESTAMP}.json"
    with open(outfile, "w") as f:
        json.dump(out, f, indent=2, default=_json_default)
    print(f"\n✓ Saved {outfile}")
    print(f"  Updated main checkpoint at {CHECKPOINT_PATH}")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
