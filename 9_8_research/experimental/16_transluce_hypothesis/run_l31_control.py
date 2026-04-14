#!/usr/bin/env python3
"""
Follow-up control for Experiment 16: Random-50 ablation restricted to layer 31.

Disambiguates: is the top-50 effect (100%→86.3%) from the SPECIFIC neurons,
or just from "any 50 neurons at the late layer where top-50 happens to live"?

Also adds: random-50 at layer 31 (sign-agnostic draw from ALL 14336 neurons there).
"""

import sys, json, random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
import importlib.util
_spec = importlib.util.spec_from_file_location("exp16_main", _HERE / "run_experiment.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
measure_bug_rate = _mod.measure_bug_rate
ablate_neurons_and_measure = _mod.ablate_neurons_and_measure
LLAMA_CHAT = _mod.LLAMA_CHAT
DEVICE = _mod.DEVICE
RANDOM_SEED = _mod.RANDOM_SEED

random.seed(RANDOM_SEED + 1)
np.random.seed(RANDOM_SEED + 1)
torch.manual_seed(RANDOM_SEED + 1)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    print(f"Loading {model_id} ...")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    intermediate_size = model.model.layers[0].mlp.down_proj.in_features
    n_layers = len(model.model.layers)

    results = {"timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
               "controls": {}}

    # Baseline (sanity check — should be ~100%)
    print("\nBaseline chat ...")
    base = measure_bug_rate(model, tok, LLAMA_CHAT)
    print(f"  bug_rate={base['bug_rate']:.3f}  n_valid={base['n_valid']}  CI={base['ci']}")
    results["controls"]["baseline"] = base

    # Control A: 50 random neurons restricted to layer 31
    print("\nControl A: Random 50 at layer 31 only ...")
    rnd_l31 = [(31, random.randint(0, intermediate_size - 1), 0.0) for _ in range(50)]
    a = ablate_neurons_and_measure(model, tok, rnd_l31, LLAMA_CHAT)
    print(f"  bug_rate={a['bug_rate']:.3f}  n_valid={a['n_valid']}  CI={a['ci']}")
    print(f"  samples={a['samples']}")
    results["controls"]["random50_at_L31"] = a

    # Control B: 50 random neurons at layers 28-31 (match top-50's actual layer dist.)
    print("\nControl B: Random 50 at layers 28-31 ...")
    rnd_late = [(random.randint(28, 31), random.randint(0, intermediate_size - 1), 0.0)
                for _ in range(50)]
    b = ablate_neurons_and_measure(model, tok, rnd_late, LLAMA_CHAT)
    print(f"  bug_rate={b['bug_rate']:.3f}  n_valid={b['n_valid']}  CI={b['ci']}")
    print(f"  samples={b['samples']}")
    results["controls"]["random50_at_L28_31"] = b

    # Control C: 100 random neurons at layer 31 (match top-100 magnitude)
    print("\nControl C: Random 100 at layer 31 only ...")
    rnd_l31_100 = [(31, random.randint(0, intermediate_size - 1), 0.0) for _ in range(100)]
    c = ablate_neurons_and_measure(model, tok, rnd_l31_100, LLAMA_CHAT)
    print(f"  bug_rate={c['bug_rate']:.3f}  n_valid={c['n_valid']}  CI={c['ci']}")
    print(f"  samples={c['samples']}")
    results["controls"]["random100_at_L31"] = c

    out_path = _HERE / f"l31_control_{results['timestamp']}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Saved to {out_path}")

    print("\n" + "=" * 60)
    print("COMPARISON vs. main experiment:")
    print("=" * 60)
    print(f"  Baseline:          100.0% (main)     | {base['bug_rate']*100:.1f}% (this run)")
    print(f"  Top-50 targeted:    86.3% (main)")
    print(f"  Top-100 targeted:   81.0% (main)")
    print(f"  Random 50 @ L31:                      | {a['bug_rate']*100:.1f}%")
    print(f"  Random 50 @ L28-31:                   | {b['bug_rate']*100:.1f}%")
    print(f"  Random 100 @ L31:                     | {c['bug_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
