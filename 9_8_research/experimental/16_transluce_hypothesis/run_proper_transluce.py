#!/usr/bin/env python3
"""
Experiment 16b: Proper replication of Transluce's methodology.

Differs from 16 (run_experiment.py) in that we:
  - Use a SINGLE format per model (not chat-vs-simple contrast)
  - Identify neurons via GRADIENT ATTRIBUTION (e · dz/de), not activation contrast
  - Ablate top-K by attribution and measure bug-rate change
  - Add a semantic probe check: do the top neurons fire selectively on
    Sept-11 / Bible / Gravity-physics contexts vs. neutral?

Run:
  cd /home/paperspace/MATS/
  python3 9_8_research/experimental/16_transluce_hypothesis/run_proper_transluce.py
"""

import sys, os, json, time, random, importlib.util
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[3]

# reuse helpers from exp 16 main script
_spec = importlib.util.spec_from_file_location("exp16_main", _HERE / "run_experiment.py")
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
measure_bug_rate = _mod.measure_bug_rate
ablate_neurons_and_measure = _mod.ablate_neurons_and_measure
_get_layers = _mod._get_layers
_detect_arch = _mod._detect_arch
get_target_positions = _mod.get_target_positions
wilson_ci = _mod.wilson_ci

from probe_texts import ALL_SETS  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = _HERE
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint_proper.json"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RANDOM_SEED = 20260414
random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)

N_TRIALS = 100
N_TOP = 50

# Transluce-style prompt families (use their format, sweep X values).
# Format: "Which is bigger, X.A or X.B? Answer: X."  — the next token
# discriminates between A and B (their decimal parts).
def make_attribution_prompts():
    prompts = []
    for X in range(1, 16):
        for (A, B) in [("8", "11"), ("9", "10"), ("9", "12"), ("8", "13")]:
            p = f"Which is bigger, {X}.{A} or {X}.{B}? Answer: {X}."
            prompts.append({
                "prompt": p, "X": X, "correct": A, "wrong": B,
            })
    return prompts


def _json_default(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, torch.Tensor): return o.tolist()
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


# ── Gradient attribution ──────────────────────────────────────────────

def compute_attribution(model, tokenizer, prompts_meta: List[dict]) -> np.ndarray:
    """Compute per-neuron attribution: e · dz/de averaged over prompts.

    z = logit[wrong_first_token] - logit[correct_first_token]
    Positive attribution  →  neuron pushes toward WRONG answer.

    Returns: np.ndarray of shape [n_layers, intermediate_size].
    """
    layers, arch = _get_layers(model)
    n_layers = len(layers)

    # Determine intermediate_size by probing
    if arch == "llama":
        inter = layers[0].mlp.down_proj.in_features
    elif arch == "pythia":
        inter = layers[0].mlp.dense_h_to_4h.out_features
    else:  # gpt2
        inter = layers[0].mlp.c_proj.weight.shape[0]

    accum = np.zeros((n_layers, inter), dtype=np.float64)
    n_accum = 0

    def resolve_token(tok_str: str) -> Optional[int]:
        """Pick the single-token encoding if available; prefer raw over space-prefixed.
        (Space-prefixed variants often tokenize to [space_id, rest], yielding
        space_id as the 'first' token — shared across correct/wrong sides.)"""
        for cand in [tok_str, " " + tok_str]:
            ids = tokenizer.encode(cand, add_special_tokens=False)
            if len(ids) == 1:
                return ids[0]
        # Fallback: first token of raw
        ids = tokenizer.encode(tok_str, add_special_tokens=False)
        return ids[0] if ids else None

    for meta in prompts_meta:
        wrong_id = resolve_token(meta["wrong"])
        correct_id = resolve_token(meta["correct"])
        if wrong_id is None or correct_id is None or wrong_id == correct_id:
            continue

        inputs = tokenizer(meta["prompt"], return_tensors="pt").to(DEVICE)

        # Capture activations with grad retained
        captured: Dict[int, torch.Tensor] = {}
        hooks = []

        def make_llama_pre_hook(layer_idx):
            def fn(module, inp):
                x = inp[0] if isinstance(inp, tuple) else inp
                x.requires_grad_(True)
                x.retain_grad()
                captured[layer_idx] = x
                # return tuple preserving structure so PyTorch uses our tensor
                return (x,) if isinstance(inp, tuple) else x
            return fn

        def make_post_hook(layer_idx):
            def fn(module, inp, out):
                out.requires_grad_(True)
                out.retain_grad()
                captured[layer_idx] = out
                return out
            return fn

        for i, layer in enumerate(layers):
            if arch == "llama":
                h = layer.mlp.down_proj.register_forward_pre_hook(make_llama_pre_hook(i))
            elif arch == "pythia":
                h = layer.mlp.act.register_forward_hook(make_post_hook(i))
            else:
                h = layer.mlp.act.register_forward_hook(make_post_hook(i))
            hooks.append(h)

        try:
            model.zero_grad(set_to_none=True)
            out = model(**inputs)
            logits = out.logits[0, -1, :]  # last position
            z = logits[wrong_id] - logits[correct_id]
            z.backward()

            seq_len = inputs.input_ids.shape[1]
            tgt = get_target_positions(seq_len)

            for layer_idx, act in captured.items():
                grad = act.grad
                if grad is None:
                    continue
                # act/grad: [1, seq_len, inter]
                contrib = (act.detach() * grad.detach())[0, tgt, :].sum(0)
                accum[layer_idx] += contrib.float().cpu().numpy().astype(np.float64)
            n_accum += 1
        finally:
            for h in hooks:
                h.remove()

    if n_accum == 0:
        raise RuntimeError("No valid prompts processed for attribution.")
    accum /= n_accum
    return accum


def top_k_from_attribution(attr: np.ndarray, k: int = N_TOP,
                           sign: str = "positive") -> List[Tuple[int, int, float]]:
    """Pick top-k neurons by attribution magnitude.
    sign="positive": neurons pushing toward WRONG answer (Transluce target).
    sign="abs": top by |attribution|.
    """
    n_layers, inter = attr.shape
    flat = attr.flatten()
    if sign == "positive":
        flat_for_rank = flat.copy()
        flat_for_rank[flat_for_rank < 0] = -np.inf  # exclude negative
        idx = np.argsort(flat_for_rank)[-k:][::-1]
    else:
        idx = np.argsort(np.abs(flat))[-k:][::-1]
    out = []
    for fi in idx:
        L = int(fi // inter); n = int(fi % inter); s = float(flat[fi])
        out.append((L, n, s))
    return out


# ── Semantic probe ────────────────────────────────────────────────────

def neuron_activations_at_last_token(model, tokenizer, text: str,
                                     neuron_list: List[Tuple[int, int, float]]) -> Dict[Tuple[int,int], float]:
    """Return activation of each (layer, neuron) at last token of `text`."""
    layers, arch = _get_layers(model)
    needed_layers = {L for L, _, _ in neuron_list}
    captured: Dict[int, torch.Tensor] = {}
    hooks = []

    def make_llama_pre_hook(layer_idx):
        def fn(module, inp):
            x = inp[0] if isinstance(inp, tuple) else inp
            captured[layer_idx] = x.detach().cpu()
        return fn
    def make_post_hook(layer_idx):
        def fn(module, inp, out):
            captured[layer_idx] = out.detach().cpu()
        return fn

    for i, layer in enumerate(layers):
        if i not in needed_layers:
            continue
        if arch == "llama":
            h = layer.mlp.down_proj.register_forward_pre_hook(make_llama_pre_hook(i))
        elif arch == "pythia":
            h = layer.mlp.act.register_forward_hook(make_post_hook(i))
        else:
            h = layer.mlp.act.register_forward_hook(make_post_hook(i))
        hooks.append(h)

    try:
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        for h in hooks:
            h.remove()

    out = {}
    for L, n, _ in neuron_list:
        if L not in captured:
            out[(L, n)] = float("nan"); continue
        act = captured[L]
        if act.dim() == 3:
            out[(L, n)] = float(act[0, -1, n].float())
        else:
            out[(L, n)] = float(act[0, n].float())
    return out


def semantic_probe(model, tokenizer, neuron_list: List[Tuple[int, int, float]]) -> Dict:
    """For each theme, compute mean activation of each neuron across probes.
    Returns per-neuron theme selectivity scores (theme_mean - neutral_mean).
    """
    set_means: Dict[str, Dict[Tuple[int,int], float]] = {}
    for theme, probes in ALL_SETS.items():
        per_neuron_vals: Dict[Tuple[int,int], List[float]] = {(L,n): [] for L,n,_ in neuron_list}
        for text in probes:
            acts = neuron_activations_at_last_token(model, tokenizer, text, neuron_list)
            for k, v in acts.items():
                per_neuron_vals[k].append(v)
        set_means[theme] = {k: float(np.nanmean(vs)) for k, vs in per_neuron_vals.items()}

    neutral = set_means["neutral"]
    selectivity = {}
    for theme in ["sept11", "bible", "gravity"]:
        selectivity[theme] = {
            str(k): set_means[theme][k] - neutral[k] for k in neutral
        }
    # For each theme, identify neurons that are top-5 selective
    top_selective = {}
    for theme, scores in selectivity.items():
        items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_selective[theme] = items[:10]

    return {
        "set_means": {th: {f"{k[0]}_{k[1]}": v for k, v in d.items()}
                      for th, d in set_means.items()},
        "selectivity": selectivity,
        "top_selective_per_theme": top_selective,
    }


# ── Per-model runner ──────────────────────────────────────────────────

def run_model(model_id: str, key: str, torch_dtype, ckpt: dict,
              single_format_for_bugrate: str) -> dict:
    if key in ckpt:
        print(f"\n[checkpoint] Restoring {key} section")
        return ckpt[key]

    print(f"\n{'=' * 60}\n{key}: {model_id}\n{'=' * 60}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype, device_map="auto")
    model.eval()

    out: dict = {}

    # 1. Baseline bug rate
    print(f"\n1. Baseline bug rate (format used: {single_format_for_bugrate!r}) ...")
    base = measure_bug_rate(model, tok, single_format_for_bugrate)
    print(f"   bug_rate={base['bug_rate']}  n_valid={base['n_valid']}  CI={base.get('ci')}")
    out["baseline"] = base

    # 2. Attribution
    print("\n2. Computing gradient attribution over Transluce-style prompts ...")
    prompts = make_attribution_prompts()
    t0 = time.time()
    attr = compute_attribution(model, tok, prompts)
    print(f"   Done in {time.time()-t0:.1f}s. shape={attr.shape}")

    top_positive = top_k_from_attribution(attr, k=N_TOP, sign="positive")
    top_abs = top_k_from_attribution(attr, k=N_TOP, sign="abs")
    print(f"   Top-10 positive-attribution neurons (push toward wrong answer):")
    for L, n, s in top_positive[:10]:
        print(f"     L{L:2d} N{n:5d}: attr={s:+.4f}")

    out["top_positive_attribution"] = [
        {"layer": L, "neuron": n, "score": s} for L, n, s in top_positive
    ]
    out["top_abs_attribution"] = [
        {"layer": L, "neuron": n, "score": s} for L, n, s in top_abs
    ]

    # 3. Ablation tests
    print("\n3. Ablation: top-50 positive attribution ...")
    r_pos = ablate_neurons_and_measure(model, tok, top_positive, single_format_for_bugrate)
    print(f"   bug_rate={r_pos['bug_rate']}  n_valid={r_pos['n_valid']}")
    out["ablation_top50_positive"] = r_pos

    print("\n   Ablation: top-50 abs attribution (mixed sign) ...")
    r_abs = ablate_neurons_and_measure(model, tok, top_abs, single_format_for_bugrate)
    print(f"   bug_rate={r_abs['bug_rate']}  n_valid={r_abs['n_valid']}")
    out["ablation_top50_abs"] = r_abs

    # Layer-matched random control: same layer distribution as top_positive
    layer_counts = {}
    for L, _, _ in top_positive:
        layer_counts[L] = layer_counts.get(L, 0) + 1
    n_layers = len(model.model.layers if _detect_arch(model) == "llama"
                   else (model.gpt_neox.layers if _detect_arch(model) == "pythia"
                         else model.transformer.h))
    if _detect_arch(model) == "llama":
        inter = model.model.layers[0].mlp.down_proj.in_features
    elif _detect_arch(model) == "pythia":
        inter = model.gpt_neox.layers[0].mlp.dense_h_to_4h.out_features
    else:
        inter = model.transformer.h[0].mlp.c_proj.weight.shape[0]

    rnd = []
    for L, cnt in layer_counts.items():
        for _ in range(cnt):
            rnd.append((L, random.randint(0, inter - 1), 0.0))
    print(f"\n   Ablation: layer-matched random-{len(rnd)} control ...")
    r_rnd = ablate_neurons_and_measure(model, tok, rnd, single_format_for_bugrate)
    print(f"   bug_rate={r_rnd['bug_rate']}  n_valid={r_rnd['n_valid']}")
    out["ablation_random_layer_matched"] = r_rnd

    # 4. Semantic probe on top-50 positive (fault-tolerant)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("\n4. Semantic probe (Sept11 / Bible / Gravity vs Neutral) on top-50 positive ...")
    try:
        probe_results = semantic_probe(model, tok, top_positive)
        summary = {}
        for theme in ["sept11", "bible", "gravity"]:
            scores = np.array(list(probe_results["selectivity"][theme].values()))
            summary[theme] = {
                "mean_selectivity": float(np.nanmean(scores)),
                "median_selectivity": float(np.nanmedian(scores)),
                "n_positive": int(np.sum(scores > 0)),
                "max_selectivity": float(np.nanmax(scores)),
            }
            print(f"   {theme:8s}: mean={summary[theme]['mean_selectivity']:+.3f}  "
                  f"n>0={summary[theme]['n_positive']}/{len(scores)}  "
                  f"max={summary[theme]['max_selectivity']:+.3f}")
        out["semantic_probe"] = probe_results
        out["semantic_probe_summary"] = summary
    except Exception as e:
        print(f"   SEMANTIC PROBE FAILED: {type(e).__name__}: {e}")
        out["semantic_probe"] = None
        out["semantic_probe_error"] = f"{type(e).__name__}: {e}"

    ckpt[key] = out
    save_checkpoint(ckpt)

    del model
    torch.cuda.empty_cache()
    return out


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}  Timestamp: {TIMESTAMP}")
    t_start = time.time()
    ckpt = load_checkpoint()

    # Single-format bug-rate prompts per model (Transluce-style, unambiguous)
    TRANSLUCE_PROMPT = "Which is bigger, 9.11 or 9.8?"

    llama_out = run_model("meta-llama/Meta-Llama-3.1-8B-Instruct", "llama",
                          torch.float16, ckpt, TRANSLUCE_PROMPT)
    pythia_out = run_model("EleutherAI/pythia-160m", "pythia",
                           torch.float32, ckpt, TRANSLUCE_PROMPT)
    gpt2_out = run_model("gpt2", "gpt2",
                         torch.float32, ckpt, TRANSLUCE_PROMPT)

    total = time.time() - t_start
    results = {
        "experiment": "16b_proper_transluce",
        "timestamp": TIMESTAMP,
        "total_time_seconds": total,
        "prompt_for_bugrate": TRANSLUCE_PROMPT,
        "n_attribution_prompts": len(make_attribution_prompts()),
        "llama": llama_out, "pythia": pythia_out, "gpt2": gpt2_out,
    }
    outfile = RESULTS_DIR / f"results_proper_{TIMESTAMP}.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"\n✓ Saved {outfile}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Proper Transluce Replication")
    print("=" * 60)
    for key in ["llama", "pythia", "gpt2"]:
        d = results[key]
        base = d["baseline"].get("bug_rate")
        pos = d["ablation_top50_positive"].get("bug_rate")
        rnd = d["ablation_random_layer_matched"].get("bug_rate")
        print(f"\n{key}:")
        print(f"  Baseline: {base}")
        print(f"  Top-50 positive-attr ablation: {pos}")
        print(f"  Layer-matched random control: {rnd}")
        sp = d.get("semantic_probe_summary", {})
        for theme in ["sept11", "bible", "gravity"]:
            s = sp.get(theme, {})
            print(f"  {theme:8s}: mean_sel={s.get('mean_selectivity')}  "
                  f"max_sel={s.get('max_selectivity')}  n>0={s.get('n_positive')}")

    print(f"\nTotal time: {total/60:.1f} min")


if __name__ == "__main__":
    main()
