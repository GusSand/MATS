#!/usr/bin/env python3
"""
Experiment R8: Exhaustive vs Sampled Subset Testing + Confidence Intervals
==========================================================================
Addresses Reviewer WzrL: "Clarify whether subset testing was exhaustive or
sampled, and provide confidence intervals near the 7→8 threshold."

Components:
  1. Document the combinatorics: C(16,8) = 12,870 possible 8-even-head subsets
  2. Boundary analysis: sample 500 7-head and 500 8-head even combos (n=50 each)
  3. Test if threshold is REALLY sharp (any 7-combo with >0%, any 8-combo <100%?)
  4. Transition: 7 even + 1-9 odd heads — do odd heads help at all?
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
from contextlib import contextmanager
from math import comb
import json
from datetime import datetime
import time
import os

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LAYER_IDX = 10
N_HEADS = 32
EVEN_HEADS = list(range(0, 32, 2))
ODD_HEADS = list(range(1, 32, 2))
CORRECT_PROMPT = "Which is bigger: 9.8 or 9.11?\nAnswer:"
BUGGY_PROMPT = "Q: Which is bigger: 9.8 or 9.11?\nA:"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "checkpoint.json")


def save_checkpoint(data: dict):
    tmp = CHECKPOINT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=lambda o: int(o) if isinstance(o, np.integer)
                  else float(o) if isinstance(o, np.floating)
                  else o.tolist() if isinstance(o, np.ndarray) else str(o))
    os.replace(tmp, CHECKPOINT_PATH)


def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return None


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


class HeadPatcher:
    def __init__(self):
        print(f"Loading {MODEL_NAME} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
        )
        self.model.eval()
        self.hooks = []
        self.saved = {}
        print("Model loaded.")

    def _attn_module(self):
        return self.model.model.layers[LAYER_IDX].self_attn

    def _save_hook(self, key):
        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            self.saved[key] = h.detach().cpu()
        return fn

    def _patch_hook(self, saved, heads):
        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            bs, sl, hs = h.shape
            hd = hs // N_HEADS
            h_r = h.view(bs, sl, N_HEADS, hd)
            s_r = saved.to(h.device).view(bs, -1, N_HEADS, hd)
            new = h_r.clone()
            ml = min(sl, s_r.shape[1])
            for idx in heads:
                new[:, :ml, idx, :] = s_r[:, :ml, idx, :]
            new = new.view(bs, sl, hs)
            return (new,) + out[1:] if isinstance(out, tuple) else new
        return fn

    def _clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.saved.clear()

    @contextmanager
    def capture(self, prompt):
        try:
            hook = self._attn_module().register_forward_hook(self._save_hook("attn"))
            self.hooks.append(hook)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                self.model(**inputs)
            yield self.saved["attn"]
        finally:
            self._clear()

    @contextmanager
    def patch(self, saved, heads):
        try:
            hook = self._attn_module().register_forward_hook(self._patch_hook(saved, heads))
            self.hooks.append(hook)
            yield
        finally:
            self._clear()

    def generate(self, prompt, max_new=20):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                      pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)

    def is_correct(self, text):
        lo = text.lower()
        correct = any(p in lo for p in [
            "9.8 is bigger", "9.8 is larger", "9.8 is greater",
            "9.8 is the bigger", "9.8 is the larger"])
        bug = any(p in lo for p in [
            "9.11 is bigger", "9.11 is larger", "9.11 is greater",
            "9.11 is the bigger", "9.11 is the larger"])
        if not correct and not bug:
            lines = text.strip().split("\n")
            if lines and "9.8" in lines[0] and "9.11" not in lines[0]:
                correct = True
        return correct and not bug

    def test_subset(self, good_attn, heads, n_trials):
        successes = 0
        for _ in range(n_trials):
            with self.patch(good_attn, heads):
                out = self.generate(BUGGY_PROMPT)
            if self.is_correct(out):
                successes += 1
        rate = successes / n_trials
        ci = wilson_ci(successes, n_trials)
        return {"heads": heads, "k": len(heads),
                "success_rate": rate, "successes": successes,
                "n_trials": n_trials, "ci": list(ci)}


def main():
    rng = np.random.RandomState(42)
    patcher = HeadPatcher()

    with patcher.capture(CORRECT_PROMPT) as good_attn:
        good_attn_saved = good_attn.clone()

    t_start = time.time()
    ckpt = load_checkpoint() or {}

    # ── Section 1: Document combinatorics ───────────────────────────
    print("\n" + "=" * 60)
    print("SECTION 1: Combinatorics")
    print("=" * 60)
    print(f"  C(16, 7) = {comb(16, 7):,} possible 7-even-head subsets")
    print(f"  C(16, 8) = {comb(16, 8):,} possible 8-even-head subsets")
    print(f"  Exhaustive testing is infeasible → sampling 500 each")

    # ── Section 2: Boundary analysis ────────────────────────────────
    N_SAMPLE = 500
    N_TRIALS_PER = 50

    if "seven_head" in ckpt:
        print("\n  [checkpoint] Restoring 7-head results")
        seven_results = ckpt["seven_head"]
    else:
        print(f"\n  Testing {N_SAMPLE} random 7-even-head combos ({N_TRIALS_PER} trials each)...")
        seven_results = []
        t0 = time.time()
        for i in range(N_SAMPLE):
            heads = sorted(rng.choice(EVEN_HEADS, size=7, replace=False).tolist())
            r = patcher.test_subset(good_attn_saved, heads, N_TRIALS_PER)
            seven_results.append(r)
            if (i + 1) % 50 == 0:
                wins = sum(1 for x in seven_results if x["success_rate"] >= 0.8)
                elapsed = time.time() - t0
                print(f"    [{i+1}/{N_SAMPLE}] {wins} successful, {elapsed:.0f}s")
        save_checkpoint({"seven_head": seven_results})

    if "eight_head" in ckpt:
        print("\n  [checkpoint] Restoring 8-head results")
        eight_results = ckpt["eight_head"]
    else:
        print(f"\n  Testing {N_SAMPLE} random 8-even-head combos ({N_TRIALS_PER} trials each)...")
        eight_results = []
        t0 = time.time()
        for i in range(N_SAMPLE):
            heads = sorted(rng.choice(EVEN_HEADS, size=8, replace=False).tolist())
            r = patcher.test_subset(good_attn_saved, heads, N_TRIALS_PER)
            eight_results.append(r)
            if (i + 1) % 50 == 0:
                wins = sum(1 for x in eight_results if x["success_rate"] >= 0.8)
                elapsed = time.time() - t0
                print(f"    [{i+1}/{N_SAMPLE}] {wins} successful, {elapsed:.0f}s")
        save_checkpoint({"seven_head": seven_results, "eight_head": eight_results})

    # ── Section 3: Transition — 7 even + N odd ─────────────────────
    print("\n" + "=" * 60)
    print("SECTION 3: 7 even heads + N odd heads")
    print("=" * 60)

    if "transition" in ckpt:
        print("  [checkpoint] Restoring transition results")
        transition_results = ckpt["transition"]
    else:
        transition_results = {}
        # Pick one specific 7-even combo
        base_7_even = sorted(rng.choice(EVEN_HEADS, size=7, replace=False).tolist())
        print(f"  Base 7-even: {base_7_even}")

        for n_odd in [0, 1, 2, 3, 5, 9, 16]:
            if n_odd == 0:
                heads = base_7_even
            else:
                odd_sample = sorted(rng.choice(ODD_HEADS, size=min(n_odd, 16), replace=False).tolist())
                heads = sorted(base_7_even + odd_sample)

            r = patcher.test_subset(good_attn_saved, heads, N_TRIALS_PER)
            r["n_odd_added"] = n_odd
            r["total_heads"] = len(heads)
            transition_results[n_odd] = r
            print(f"  7 even + {n_odd} odd ({len(heads)} total): "
                  f"{r['success_rate']:.0%} [{r['ci'][0]:.0%}, {r['ci'][1]:.0%}]")

    total_time = time.time() - t_start

    # ── Analysis ────────────────────────────────────────────────────
    seven_rates = [x["success_rate"] for x in seven_results]
    eight_rates = [x["success_rate"] for x in eight_results]

    any_seven_success = any(r >= 0.5 for r in seven_rates)
    any_eight_fail = any(r < 0.5 for r in eight_rates)

    # ── Save results ────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "experiment": "R8_threshold_analysis",
        "timestamp": timestamp,
        "model": MODEL_NAME,
        "combinatorics": {
            "C_16_7": comb(16, 7),
            "C_16_8": comb(16, 8),
            "sampling_method": f"Random {N_SAMPLE} from each, seed=42",
        },
        "seven_head_results": seven_results,
        "eight_head_results": eight_results,
        "transition_results": {str(k): v for k, v in transition_results.items()},
        "analysis": {
            "seven_mean_rate": float(np.mean(seven_rates)),
            "seven_max_rate": float(np.max(seven_rates)),
            "seven_any_success_above_50pct": any_seven_success,
            "seven_fraction_above_80pct": float(np.mean([r >= 0.8 for r in seven_rates])),
            "eight_mean_rate": float(np.mean(eight_rates)),
            "eight_min_rate": float(np.min(eight_rates)),
            "eight_any_fail_below_50pct": any_eight_fail,
            "eight_fraction_above_80pct": float(np.mean([r >= 0.8 for r in eight_rates])),
            "threshold_is_sharp": not any_seven_success and not any_eight_fail,
        },
        "total_time_seconds": total_time,
    }

    results_path = os.path.join(RESULTS_DIR, f"results_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda o: int(o) if isinstance(o, np.integer)
                  else float(o) if isinstance(o, np.floating)
                  else o.tolist() if isinstance(o, np.ndarray) else str(o))
    print(f"\nResults saved to {results_path}")

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("R8 SUMMARY: Threshold Analysis")
    print("=" * 70)

    print(f"\n7-even-head combos ({N_SAMPLE} sampled of {comb(16,7):,}):")
    print(f"  Mean success rate: {np.mean(seven_rates):.1%}")
    print(f"  Max success rate:  {np.max(seven_rates):.1%}")
    print(f"  Any >50% success:  {any_seven_success}")
    print(f"  Fraction >=80%:    {np.mean([r >= 0.8 for r in seven_rates]):.1%}")

    print(f"\n8-even-head combos ({N_SAMPLE} sampled of {comb(16,8):,}):")
    print(f"  Mean success rate: {np.mean(eight_rates):.1%}")
    print(f"  Min success rate:  {np.min(eight_rates):.1%}")
    print(f"  Any <50% success:  {any_eight_fail}")
    print(f"  Fraction >=80%:    {np.mean([r >= 0.8 for r in eight_rates]):.1%}")

    print(f"\nIs threshold sharp? {not any_seven_success and not any_eight_fail}")

    print(f"\nTransition (7 even + N odd):")
    for n_odd in sorted(transition_results.keys(), key=lambda x: int(x) if isinstance(x, str) else x):
        r = transition_results[n_odd]
        print(f"  +{int(n_odd) if isinstance(n_odd, str) else n_odd} odd: "
              f"{r['success_rate']:.0%}")

    print(f"\nTotal time: {total_time/60:.1f} minutes")
    return results


if __name__ == "__main__":
    results = main()
