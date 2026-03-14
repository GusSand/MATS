#!/usr/bin/env python3
"""
Experiment R1: Arbitrary Head Combinations
==========================================
Addresses Reviewer joQU's central criticism: does even/odd parity matter,
or is there a simple majority-vote threshold effect where any ~half of the
heads suffice?

Builds on Experiment 13 (bandwidth competition) which showed:
- Only 63% of random even-head combos succeed
- Spatial organization matters more than parity

This experiment adds the MIXED-PARITY dimension that Exp 13 did not test.

Components:
  1. 200 random mixed-parity 8-head subsets (n=50 trials each)
  2. Varying subset sizes k={4,6,7,8,9,10,12,16} x 100 random subsets each
  3. Odd-only subsets at k=8,10,12,14,16
  4. Matched even/odd controls (same spatial pattern, shifted by 1)
  5. Reconciliation with Experiment 13 results
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple, Optional
from contextlib import contextmanager
import json
from datetime import datetime
import time
import os
import sys
from itertools import combinations
from scipy import stats as scipy_stats

# ── Config ──────────────────────────────────────────────────────────
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LAYER_IDX = 10
N_HEADS = 32
CORRECT_PROMPT = "Which is bigger: 9.8 or 9.11?\nAnswer:"
BUGGY_PROMPT = "Q: Which is bigger: 9.8 or 9.11?\nA:"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "checkpoint.json")


def save_checkpoint(data: dict):
    """Save intermediate results for crash recovery."""
    tmp = CHECKPOINT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=lambda o: int(o) if isinstance(o, np.integer) else float(o) if isinstance(o, np.floating) else o.tolist() if isinstance(o, np.ndarray) else str(o))
    os.replace(tmp, CHECKPOINT_PATH)


def load_checkpoint() -> Optional[dict]:
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return None


class HeadPatcher:
    """Reuses validated patching methodology from bandwidth/working_validated/."""

    def __init__(self):
        print(f"Loading {MODEL_NAME} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
        )
        self.model.eval()
        self.hooks: list = []
        self.saved_activations: dict = {}
        print("Model loaded.")

    # ── hooks (identical to working_validated/test_distributed_coverage_proper.py) ──

    def _attn_module(self):
        return self.model.model.layers[LAYER_IDX].self_attn

    def _save_hook(self, key: str):
        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            self.saved_activations[key] = h.detach().cpu()
        return fn

    def _patch_hook(self, saved: torch.Tensor, heads: List[int]):
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
        self.saved_activations.clear()

    @contextmanager
    def capture(self, prompt: str):
        try:
            key = "attn"
            hook = self._attn_module().register_forward_hook(self._save_hook(key))
            self.hooks.append(hook)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                self.model(**inputs)
            yield self.saved_activations[key]
        finally:
            self._clear()

    @contextmanager
    def patch(self, saved: torch.Tensor, heads: List[int]):
        try:
            hook = self._attn_module().register_forward_hook(self._patch_hook(saved, heads))
            self.hooks.append(hook)
            yield
        finally:
            self._clear()

    def generate(self, prompt: str, max_new: int = 20) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)

    def is_correct(self, text: str) -> bool:
        lo = text.lower()
        correct = any(p in lo for p in [
            "9.8 is bigger", "9.8 is larger", "9.8 is greater",
            "9.8 is the bigger", "9.8 is the larger",
        ])
        bug = any(p in lo for p in [
            "9.11 is bigger", "9.11 is larger", "9.11 is greater",
            "9.11 is the bigger", "9.11 is the larger",
        ])
        if not correct and not bug:
            lines = text.strip().split("\n")
            if lines and "9.8" in lines[0] and "9.11" not in lines[0]:
                correct = True
        return correct and not bug


# ── Helpers ─────────────────────────────────────────────────────────

def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score confidence interval."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def n_even(heads: List[int]) -> int:
    return sum(1 for h in heads if h % 2 == 0)


def spatial_metrics(heads: List[int]) -> Dict:
    heads = sorted(heads)
    if len(heads) < 2:
        return {"gap_std": 0.0, "span": 0, "gap_regularity": 1.0}
    gaps = [heads[i + 1] - heads[i] for i in range(len(heads) - 1)]
    return {
        "gap_std": float(np.std(gaps)),
        "span": heads[-1] - heads[0],
        "gap_regularity": float(1 / (1 + np.std(gaps))),
    }


# ── Experiment sections ────────────────────────────────────────────

def run_subset(patcher: HeadPatcher, good_attn: torch.Tensor,
               heads: List[int], n_trials: int) -> Dict:
    """Run n_trials for a single head subset. Returns result dict."""
    successes = 0
    samples = []
    for t in range(n_trials):
        with patcher.patch(good_attn, heads):
            out = patcher.generate(BUGGY_PROMPT)
        if patcher.is_correct(out):
            successes += 1
        if t < 2:
            samples.append(out.strip()[:80])
    rate = successes / n_trials
    ci_lo, ci_hi = wilson_ci(successes, n_trials)
    return {
        "heads": heads,
        "k": len(heads),
        "n_even": n_even(heads),
        "n_odd": len(heads) - n_even(heads),
        "success_rate": rate,
        "successes": successes,
        "n_trials": n_trials,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "samples": samples,
        "spatial": spatial_metrics(heads),
    }


def section1_random_mixed_8(patcher, good_attn, rng, n_subsets=200, n_trials=50):
    """200 random mixed-parity 8-head subsets."""
    print("\n" + "=" * 60)
    print("SECTION 1: Random mixed-parity 8-head subsets")
    print(f"  {n_subsets} subsets, {n_trials} trials each")
    print("=" * 60)
    results = []
    all_heads = list(range(N_HEADS))
    t0 = time.time()
    for i in range(n_subsets):
        heads = sorted(rng.choice(all_heads, size=8, replace=False).tolist())
        r = run_subset(patcher, good_attn, heads, n_trials)
        results.append(r)
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            avg = elapsed / (i + 1)
            remaining = avg * (n_subsets - i - 1)
            wins = sum(1 for x in results if x["success_rate"] >= 0.8)
            print(f"  [{i+1}/{n_subsets}] {wins} successful so far  "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")
    return results


def section2_varying_k(patcher, good_attn, rng, ks=(4, 6, 7, 8, 9, 10, 12, 16),
                       n_subsets=100, n_trials=50):
    """Random subsets of varying sizes."""
    print("\n" + "=" * 60)
    print("SECTION 2: Varying subset sizes (unconstrained parity)")
    print(f"  k in {ks}, {n_subsets} subsets each, {n_trials} trials")
    print("=" * 60)
    all_heads = list(range(N_HEADS))
    results_by_k = {}
    for k in ks:
        print(f"\n  k={k}:")
        t0 = time.time()
        res = []
        for i in range(n_subsets):
            heads = sorted(rng.choice(all_heads, size=k, replace=False).tolist())
            r = run_subset(patcher, good_attn, heads, n_trials)
            res.append(r)
            if (i + 1) % 25 == 0:
                wins = sum(1 for x in res if x["success_rate"] >= 0.8)
                print(f"    [{i+1}/{n_subsets}] {wins} successful ({time.time()-t0:.0f}s)")
        results_by_k[k] = res
        rates = [x["success_rate"] for x in res]
        print(f"    Mean success: {np.mean(rates):.1%}, "
              f"Fraction >=80%: {sum(1 for r in rates if r >= 0.8)/len(rates):.1%}")
    return results_by_k


def section3_odd_only(patcher, good_attn, rng, ks=(8, 10, 12, 14, 16),
                      n_trials=50):
    """All-odd subsets at various k."""
    print("\n" + "=" * 60)
    print("SECTION 3: Odd-only subsets at varying k")
    print("=" * 60)
    odd_heads = list(range(1, N_HEADS, 2))  # 16 odd heads
    results = {}
    for k in ks:
        if k > len(odd_heads):
            print(f"  k={k}: skipped (only {len(odd_heads)} odd heads)")
            continue
        print(f"\n  k={k} (all odd):")
        heads = sorted(rng.choice(odd_heads, size=k, replace=False).tolist())
        r = run_subset(patcher, good_attn, heads, n_trials)
        results[k] = r
        print(f"    Heads: {heads}")
        print(f"    Success: {r['success_rate']:.1%} [{r['ci_lower']:.1%}, {r['ci_upper']:.1%}]")
    # Also test ALL 16 odd heads
    print(f"\n  k=16 (ALL odd heads):")
    r = run_subset(patcher, good_attn, odd_heads, n_trials)
    results[16] = r
    print(f"    Success: {r['success_rate']:.1%} [{r['ci_lower']:.1%}, {r['ci_upper']:.1%}]")
    return results


def section4_matched_controls(patcher, good_attn, rng, n_pairs=30, n_trials=50):
    """For each successful even-only 8-head subset, create matched odd-only (shifted +1)."""
    print("\n" + "=" * 60)
    print("SECTION 4: Matched even/odd controls")
    print(f"  {n_pairs} pairs, {n_trials} trials each")
    print("=" * 60)
    even_heads = list(range(0, N_HEADS, 2))
    results = []
    t0 = time.time()
    for i in range(n_pairs):
        even_subset = sorted(rng.choice(even_heads, size=8, replace=False).tolist())
        # Matched odd: shift each head index by +1
        odd_subset = [h + 1 for h in even_subset]
        # Both should be valid (0-30 even → 1-31 odd)
        r_even = run_subset(patcher, good_attn, even_subset, n_trials)
        r_odd = run_subset(patcher, good_attn, odd_subset, n_trials)
        results.append({
            "pair_idx": i,
            "even": r_even,
            "odd": r_odd,
            "even_rate": r_even["success_rate"],
            "odd_rate": r_odd["success_rate"],
            "advantage": r_even["success_rate"] - r_odd["success_rate"],
        })
        if (i + 1) % 10 == 0:
            avg_adv = np.mean([x["advantage"] for x in results])
            print(f"  [{i+1}/{n_pairs}] mean even advantage: {avg_adv:+.1%} "
                  f"({time.time()-t0:.0f}s)")
    return results


# ── Analysis ────────────────────────────────────────────────────────

def analyze(s1, s2, s3, s4):
    """Compute summary statistics and statistical tests."""
    analysis = {}

    # Section 1 analysis
    s1_rates = [x["success_rate"] for x in s1]
    s1_even_counts = [x["n_even"] for x in s1]
    s1_success_flag = [1 if r >= 0.8 else 0 for r in s1_rates]

    analysis["section1"] = {
        "n_subsets": len(s1),
        "mean_success_rate": float(np.mean(s1_rates)),
        "median_success_rate": float(np.median(s1_rates)),
        "fraction_successful": float(np.mean(s1_success_flag)),
        "correlation_n_even_vs_rate": float(np.corrcoef(s1_even_counts, s1_rates)[0, 1])
            if len(set(s1_even_counts)) > 1 else None,
    }

    # Point-biserial correlation: is success enriched for more even heads?
    if len(set(s1_even_counts)) > 1 and len(set(s1_success_flag)) > 1:
        corr, pval = scipy_stats.pointbiserialr(s1_success_flag, s1_even_counts)
        analysis["section1"]["enrichment_correlation"] = float(corr)
        analysis["section1"]["enrichment_pvalue"] = float(pval)

    # Section 2 analysis
    s2_summary = {}
    for k, res_list in s2.items():
        rates = [x["success_rate"] for x in res_list]
        s2_summary[int(k)] = {
            "mean_rate": float(np.mean(rates)),
            "fraction_successful": float(np.mean([1 if r >= 0.8 else 0 for r in rates])),
            "std_rate": float(np.std(rates)),
        }
    analysis["section2"] = s2_summary

    # Section 3 analysis
    s3_summary = {}
    for k, res in s3.items():
        s3_summary[int(k)] = {
            "success_rate": res["success_rate"],
            "ci": [res["ci_lower"], res["ci_upper"]],
        }
    analysis["section3_odd_only"] = s3_summary

    # Section 4 analysis
    even_rates = [x["even_rate"] for x in s4]
    odd_rates = [x["odd_rate"] for x in s4]
    advantages = [x["advantage"] for x in s4]
    analysis["section4"] = {
        "n_pairs": len(s4),
        "mean_even_rate": float(np.mean(even_rates)),
        "mean_odd_rate": float(np.mean(odd_rates)),
        "mean_advantage": float(np.mean(advantages)),
        "paired_ttest_pvalue": float(scipy_stats.ttest_rel(even_rates, odd_rates).pvalue)
            if len(s4) > 1 else None,
        "wilcoxon_pvalue": float(scipy_stats.wilcoxon(
            even_rates, odd_rates, alternative="greater").pvalue)
            if len(s4) > 1 and any(a != 0 for a in advantages) else None,
    }

    # Reconciliation with Exp 13
    # Exp 13: 63% of random even-only 8-head combos succeeded (19/30)
    exp13_rate = 19 / 30  # 0.633
    s1_mixed_frac = analysis["section1"]["fraction_successful"]
    analysis["reconciliation_with_exp13"] = {
        "exp13_even_only_success_fraction": exp13_rate,
        "r1_mixed_parity_success_fraction": s1_mixed_frac,
        "difference": float(s1_mixed_frac - exp13_rate),
        "interpretation": (
            "Mixed sets succeed at LOWER rate → even heads contribute disproportionately"
            if s1_mixed_frac < exp13_rate - 0.05
            else "Mixed sets succeed at SIMILAR rate → parity is irrelevant"
            if abs(s1_mixed_frac - exp13_rate) <= 0.05
            else "Mixed sets succeed at HIGHER rate → constraint was even-only, not parity"
        ),
    }

    return analysis


# ── Main ────────────────────────────────────────────────────────────

def main():
    seed = 42
    rng = np.random.RandomState(seed)

    patcher = HeadPatcher()

    # Capture good-state activation once
    print("\nCapturing correct-format activation at Layer 10 ...")
    with patcher.capture(CORRECT_PROMPT) as good_attn:
        good_attn_saved = good_attn.clone()
    print(f"  Shape: {good_attn_saved.shape}")

    # Verify baselines
    print("\n── Baseline verification ──")
    baseline_buggy = patcher.generate(BUGGY_PROMPT)
    baseline_correct = patcher.generate(CORRECT_PROMPT)
    print(f"  Buggy (Q&A):  {baseline_buggy.strip()[:80]}")
    print(f"  Correct (Simple): {baseline_correct.strip()[:80]}")

    t_start = time.time()

    # Run all sections with checkpointing
    ckpt = load_checkpoint() or {}

    if "section1" in ckpt:
        print("\n  [checkpoint] Restoring section 1 from checkpoint")
        s1 = ckpt["section1"]
    else:
        s1 = section1_random_mixed_8(patcher, good_attn_saved, rng,
                                      n_subsets=200, n_trials=50)
        save_checkpoint({"section1": s1})

    if "section2" in ckpt:
        print("\n  [checkpoint] Restoring section 2 from checkpoint")
        s2 = ckpt["section2"]
    else:
        s2 = section2_varying_k(patcher, good_attn_saved, rng,
                                n_subsets=100, n_trials=50)
        save_checkpoint({"section1": s1, "section2": {str(k): v for k, v in s2.items()}})

    if "section3" in ckpt:
        print("\n  [checkpoint] Restoring section 3 from checkpoint")
        s3 = ckpt["section3"]
    else:
        s3 = section3_odd_only(patcher, good_attn_saved, rng, n_trials=50)
        save_checkpoint({"section1": s1, "section2": {str(k): v for k, v in s2.items()},
                         "section3": {str(k): v for k, v in s3.items()}})

    if "section4" in ckpt:
        print("\n  [checkpoint] Restoring section 4 from checkpoint")
        s4 = ckpt["section4"]
    else:
        s4 = section4_matched_controls(patcher, good_attn_saved, rng,
                                        n_pairs=30, n_trials=50)

    # Normalize s2/s3 keys back to int if loaded from checkpoint
    if isinstance(s2, dict) and all(isinstance(k, str) for k in s2.keys()):
        s2 = {int(k): v for k, v in s2.items()}
    if isinstance(s3, dict) and all(isinstance(k, str) for k in s3.keys()):
        s3 = {int(k): v for k, v in s3.items()}

    # Analysis
    analysis = analyze(s1, s2, s3, s4)

    total_time = time.time() - t_start

    # ── Save results ────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Convert section results to serialisable form
    def ser(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serialisable: {type(obj)}")

    results = {
        "experiment": "R1_arbitrary_heads",
        "timestamp": timestamp,
        "seed": seed,
        "model": MODEL_NAME,
        "layer": LAYER_IDX,
        "total_time_seconds": total_time,
        "section1_random_mixed_8": s1,
        "section2_varying_k": {str(k): v for k, v in s2.items()},
        "section3_odd_only": {str(k): (v if isinstance(v, dict) and "heads" in v else v)
                              for k, v in s3.items()},
        "section4_matched_controls": s4,
        "analysis": analysis,
    }

    results_path = os.path.join(RESULTS_DIR, f"results_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=ser)
    print(f"\nResults saved to {results_path}")

    # ── Print summary ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("R1 EXPERIMENT SUMMARY")
    print("=" * 70)

    a = analysis
    print(f"\nSection 1 — 200 random mixed-parity 8-head subsets:")
    print(f"  Mean success rate: {a['section1']['mean_success_rate']:.1%}")
    print(f"  Fraction >=80% success: {a['section1']['fraction_successful']:.1%}")
    if a["section1"].get("correlation_n_even_vs_rate") is not None:
        print(f"  Correlation(n_even, success_rate): "
              f"r={a['section1']['correlation_n_even_vs_rate']:.3f}")
    if a["section1"].get("enrichment_pvalue") is not None:
        print(f"  Enrichment test: r={a['section1']['enrichment_correlation']:.3f}, "
              f"p={a['section1']['enrichment_pvalue']:.4f}")

    print(f"\nSection 2 — Success rate by subset size k:")
    for k in sorted(a["section2"].keys()):
        d = a["section2"][k]
        print(f"  k={k:2d}: mean {d['mean_rate']:.1%}, "
              f"fraction >=80%: {d['fraction_successful']:.1%}")

    print(f"\nSection 3 — Odd-only subsets:")
    for k in sorted(a["section3_odd_only"].keys()):
        d = a["section3_odd_only"][k]
        print(f"  k={k:2d}: {d['success_rate']:.1%} "
              f"[{d['ci'][0]:.1%}, {d['ci'][1]:.1%}]")

    print(f"\nSection 4 — Matched even/odd controls ({a['section4']['n_pairs']} pairs):")
    print(f"  Mean even rate: {a['section4']['mean_even_rate']:.1%}")
    print(f"  Mean odd rate:  {a['section4']['mean_odd_rate']:.1%}")
    print(f"  Mean advantage: {a['section4']['mean_advantage']:+.1%}")
    if a["section4"].get("paired_ttest_pvalue") is not None:
        print(f"  Paired t-test p={a['section4']['paired_ttest_pvalue']:.4f}")
    if a["section4"].get("wilcoxon_pvalue") is not None:
        print(f"  Wilcoxon p={a['section4']['wilcoxon_pvalue']:.4f}")

    print(f"\nReconciliation with Experiment 13:")
    rec = a["reconciliation_with_exp13"]
    print(f"  Exp 13 even-only success fraction: {rec['exp13_even_only_success_fraction']:.1%}")
    print(f"  R1 mixed-parity success fraction:  {rec['r1_mixed_parity_success_fraction']:.1%}")
    print(f"  Interpretation: {rec['interpretation']}")

    print(f"\nTotal experiment time: {total_time/60:.1f} minutes")

    return results


if __name__ == "__main__":
    results = main()
