#!/usr/bin/env python3
"""Pythia rerun on `answer_prompt` format (58.9% bug, n_valid=90) —
the compare_two selection in run_proper_followup.py was n_valid=2 (noise)."""
import sys, os, json, time, random, importlib.util
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("exp16b_main", _HERE / "run_proper_transluce.py")
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
_fspec = importlib.util.spec_from_file_location("exp16b_fup", _HERE / "run_proper_followup.py")
_fmod = importlib.util.module_from_spec(_fspec); _fspec.loader.exec_module(_fmod)

compute_attribution = _mod.compute_attribution
top_k_from_attribution = _mod.top_k_from_attribution
ablate_neurons_and_measure = _mod.ablate_neurons_and_measure
measure_bug_rate = _mod.measure_bug_rate
_json_default = _mod._json_default
CHECKPOINT_PATH = _mod.CHECKPOINT_PATH
N_TOP = _mod.N_TOP
pythia_attribution_prompts = _fmod.pythia_attribution_prompts
semantic_probe_on_cpu = _fmod.semantic_probe_on_cpu

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(20260414); np.random.seed(20260414); torch.manual_seed(20260414)

SELECTED_FMT = "answer_prompt"
SELECTED_PROMPT = "Which is bigger: 9.8 or 9.11? Answer:"


def main():
    print(f"Device: {DEVICE}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "EleutherAI/pythia-160m"
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32,
                                                  device_map="auto")
    model.eval()

    print(f"\nSelected format: '{SELECTED_FMT}' → {SELECTED_PROMPT!r}")
    print("\n1. Baseline (N=100, temp=0.6) ...")
    base = measure_bug_rate(model, tok, SELECTED_PROMPT)
    print(f"   bug_rate={base['bug_rate']:.3f}  n_valid={base['n_valid']}  CI={base['ci']}")

    print("\n2. Attribution over parameterized answer_prompt prompts ...")
    prompts = pythia_attribution_prompts(SELECTED_PROMPT)
    t0 = time.time()
    attr = compute_attribution(model, tok, prompts)
    print(f"   Done in {time.time()-t0:.1f}s  shape={attr.shape}  n={len(prompts)}")
    top_pos = top_k_from_attribution(attr, k=N_TOP, sign="positive")
    top_abs = top_k_from_attribution(attr, k=N_TOP, sign="abs")
    for L, n, s in top_pos[:10]:
        print(f"     L{L:2d} N{n:5d}: attr={s:+.4f}")

    print("\n3. Ablations (on same format) ...")
    r_pos = ablate_neurons_and_measure(model, tok, top_pos, SELECTED_PROMPT)
    r_abs = ablate_neurons_and_measure(model, tok, top_abs, SELECTED_PROMPT)
    inter = model.gpt_neox.layers[0].mlp.dense_h_to_4h.out_features
    layer_counts = {}
    for L, _, _ in top_pos: layer_counts[L] = layer_counts.get(L, 0) + 1
    rnd = [(L, random.randint(0, inter - 1), 0.0)
           for L, cnt in layer_counts.items() for _ in range(cnt)]
    r_rnd = ablate_neurons_and_measure(model, tok, rnd, SELECTED_PROMPT)
    print(f"   baseline:        bug={base['bug_rate']:.3f}  n={base['n_valid']}")
    print(f"   top-50 positive: bug={r_pos['bug_rate']}  n={r_pos['n_valid']}")
    print(f"   top-50 mixed:    bug={r_abs['bug_rate']}  n={r_abs['n_valid']}")
    print(f"   random-matched:  bug={r_rnd['bug_rate']}  n={r_rnd['n_valid']}")

    print("\n4. Semantic probe on CPU ...")
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

    out = {
        "selected_format": SELECTED_FMT,
        "selected_prompt": SELECTED_PROMPT,
        "baseline": base,
        "top_positive_attribution": [{"layer": L, "neuron": n, "score": s} for L, n, s in top_pos],
        "top_abs_attribution": [{"layer": L, "neuron": n, "score": s} for L, n, s in top_abs],
        "ablation": {"top50_positive": r_pos, "top50_abs": r_abs,
                     "random_layer_matched": r_rnd, "baseline_selected_format": base},
        "semantic_probe": probe, "semantic_probe_summary": summary,
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = _HERE / f"results_pythia_answerprompt_{ts}.json"
    with open(outfile, "w") as f:
        json.dump(out, f, indent=2, default=_json_default)
    print(f"\n✓ Saved {outfile}")

    # Update main checkpoint to use this better Pythia result
    with open(CHECKPOINT_PATH) as f: ckpt = json.load(f)
    # Preserve format_sweep if present
    old_sweep = ckpt.get("pythia", {}).get("format_sweep")
    ckpt["pythia"] = out
    if old_sweep is not None:
        ckpt["pythia"]["format_sweep"] = old_sweep
    tmp = str(CHECKPOINT_PATH) + ".tmp"
    with open(tmp, "w") as f: json.dump(ckpt, f, indent=2, default=_json_default)
    os.replace(tmp, CHECKPOINT_PATH)
    print(f"  Updated main checkpoint (preserved format_sweep)")


if __name__ == "__main__":
    main()
