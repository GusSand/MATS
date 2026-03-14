#!/usr/bin/env python3
"""
Experiment R2: Memorization vs Generalization on Llama-3.1-8B
=============================================================
Extends Experiment 12 (Pythia memorization finding) to the primary model.

On Pythia-160M, the even-head "specialization" was pure memorization of one
exact phrase (0% generalization). This experiment determines whether
Llama-3.1-8B-Instruct — the paper's primary model — shows genuine
generalization or is also memorised.

Components:
  1. Prompt variation sweep (synonyms, rewordings, order swaps)
  2. Broader bug hunting across 50 decimal pairs
  3. Intervention testing on every newly found buggy pair
  4. Generalization score computation
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

# ── Config ──────────────────────────────────────────────────────────
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LAYER_IDX = 10
N_HEADS = 32
EVEN_HEADS = list(range(0, 32, 2))  # The standard 16 even heads
INTERVENTION_HEADS = EVEN_HEADS[:8]  # First 8 even heads (known working)
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


def load_checkpoint() -> Optional[dict]:
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return None


class HeadPatcher:
    """Reuses validated patching from working_scripts/bidirectional_patching.py
    and bandwidth/working_validated/test_distributed_coverage_proper.py"""

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

    def generate(self, prompt: str, max_new: int = 30) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)


def check_answer(text: str, correct_val: str, wrong_val: str) -> str:
    """Classify model output: 'correct', 'bug', or 'unclear'.
    correct_val is the numerically larger value (e.g. '9.8').
    wrong_val is the numerically smaller value (e.g. '9.11').
    """
    lo = text.lower()
    patterns_correct = [
        f"{correct_val} is bigger", f"{correct_val} is larger",
        f"{correct_val} is greater", f"{correct_val} is the bigger",
        f"{correct_val} is the larger",
    ]
    patterns_bug = [
        f"{wrong_val} is bigger", f"{wrong_val} is larger",
        f"{wrong_val} is greater", f"{wrong_val} is the bigger",
        f"{wrong_val} is the larger",
    ]
    has_correct = any(p in lo for p in patterns_correct)
    has_bug = any(p in lo for p in patterns_bug)

    # Fallback: check first line for the value
    if not has_correct and not has_bug:
        first_line = text.strip().split("\n")[0]
        if correct_val in first_line and wrong_val not in first_line:
            has_correct = True
        elif wrong_val in first_line and correct_val not in first_line:
            has_bug = True

    if has_correct and not has_bug:
        return "correct"
    elif has_bug and not has_correct:
        return "bug"
    return "unclear"


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ── Section 1: Prompt Variation Sweep ───────────────────────────────

def section1_prompt_variations(patcher: HeadPatcher, n_trials: int = 100):
    """Test even-head intervention on systematic prompt variations."""
    print("\n" + "=" * 60)
    print("SECTION 1: Prompt Variation Sweep")
    print(f"  Intervention: first 8 even heads at Layer 10")
    print(f"  n_trials per variation: {n_trials}")
    print("=" * 60)

    # correct_val is the larger number, wrong_val is the smaller
    variations = [
        # (name, qa_prompt, simple_prompt, correct_val, wrong_val)
        ("original",
         "Q: Which is bigger: 9.8 or 9.11?\nA:",
         "Which is bigger: 9.8 or 9.11?\nAnswer:",
         "9.8", "9.11"),
        ("reversed_order",
         "Q: Which is bigger: 9.11 or 9.8?\nA:",
         "Which is bigger: 9.11 or 9.8?\nAnswer:",
         "9.8", "9.11"),
        ("synonym_larger",
         "Q: Which is larger: 9.8 or 9.11?\nA:",
         "Which is larger: 9.8 or 9.11?\nAnswer:",
         "9.8", "9.11"),
        ("synonym_greater",
         "Q: Which is greater: 9.8 or 9.11?\nA:",
         "Which is greater: 9.8 or 9.11?\nAnswer:",
         "9.8", "9.11"),
        ("rephrased_1",
         "Q: Is 9.8 or 9.11 the bigger number?\nA:",
         "Is 9.8 or 9.11 the bigger number?\nAnswer:",
         "9.8", "9.11"),
        ("rephrased_2",
         "Q: Compare 9.8 and 9.11. Which is bigger?\nA:",
         "Compare 9.8 and 9.11. Which is bigger?\nAnswer:",
         "9.8", "9.11"),
        ("trailing_zero",
         "Q: Which is bigger: 9.80 or 9.11?\nA:",
         "Which is bigger: 9.80 or 9.11?\nAnswer:",
         "9.8", "9.11"),  # may output 9.80
        ("diff_decimal_97",
         "Q: Which is bigger: 9.7 or 9.11?\nA:",
         "Which is bigger: 9.7 or 9.11?\nAnswer:",
         "9.7", "9.11"),
        ("diff_decimal_93",
         "Q: Which is bigger: 9.3 or 9.11?\nA:",
         "Which is bigger: 9.3 or 9.11?\nAnswer:",
         "9.3", "9.11"),
    ]

    results = []
    for name, qa_prompt, simple_prompt, correct_val, wrong_val in variations:
        print(f"\n  Testing '{name}'...")

        # Step 1: Check if bug exists (Q&A vs Simple baseline)
        baseline_qa_results = []
        baseline_simple_results = []
        for _ in range(min(n_trials, 20)):  # 20 baseline trials
            out_qa = patcher.generate(qa_prompt)
            baseline_qa_results.append(check_answer(out_qa, correct_val, wrong_val))
            out_simple = patcher.generate(simple_prompt)
            baseline_simple_results.append(check_answer(out_simple, correct_val, wrong_val))

        qa_bug_rate = sum(1 for r in baseline_qa_results if r == "bug") / len(baseline_qa_results)
        simple_correct_rate = sum(1 for r in baseline_simple_results if r == "correct") / len(baseline_simple_results)

        bug_exists = qa_bug_rate >= 0.5 and simple_correct_rate >= 0.5
        print(f"    Q&A bug rate: {qa_bug_rate:.0%}, Simple correct rate: {simple_correct_rate:.0%}")
        print(f"    Bug exists: {bug_exists}")

        # Step 2: If bug exists, test intervention
        intervention_results = []
        if bug_exists:
            # Capture from simple format
            with patcher.capture(simple_prompt) as good_attn:
                good_attn_saved = good_attn.clone()

            for t in range(n_trials):
                with patcher.patch(good_attn_saved, INTERVENTION_HEADS):
                    out = patcher.generate(qa_prompt)
                intervention_results.append(check_answer(out, correct_val, wrong_val))
                if (t + 1) % 25 == 0:
                    successes = sum(1 for r in intervention_results if r == "correct")
                    print(f"    [{t+1}/{n_trials}] {successes} correct")

            fix_rate = sum(1 for r in intervention_results if r == "correct") / n_trials
            ci = wilson_ci(sum(1 for r in intervention_results if r == "correct"), n_trials)
            print(f"    Intervention fix rate: {fix_rate:.0%} [{ci[0]:.0%}, {ci[1]:.0%}]")
        else:
            fix_rate = None
            ci = None

        results.append({
            "name": name,
            "qa_prompt": qa_prompt,
            "simple_prompt": simple_prompt,
            "correct_val": correct_val,
            "wrong_val": wrong_val,
            "qa_bug_rate": qa_bug_rate,
            "simple_correct_rate": simple_correct_rate,
            "bug_exists": bug_exists,
            "intervention_fix_rate": fix_rate,
            "intervention_ci": list(ci) if ci else None,
            "n_trials": n_trials if bug_exists else 0,
            "baseline_qa": baseline_qa_results,
            "baseline_simple": baseline_simple_results,
            "intervention_detail": intervention_results if intervention_results else [],
            "sample_qa": patcher.generate(qa_prompt).strip()[:100],
        })

    return results


# ── Section 2: Broader Bug Hunting ──────────────────────────────────

def generate_decimal_pairs() -> List[Tuple[str, str, str, str]]:
    """Generate decimal pairs where string comparison gives wrong answer.
    Returns: (larger_val, smaller_val, description, category)
    """
    pairs = []

    # Category 1: X.Y vs X.YZ where X.Y > X.YZ  (1-digit vs 2-digit decimal)
    for x in range(1, 11):
        for y in range(1, 10):
            for yz in range(10, min(y * 10, 100)):
                # X.Y > X.YZ only if Y/10 > YZ/100, i.e. Y*10 > YZ
                if y * 10 > yz:
                    larger = f"{x}.{y}"
                    smaller = f"{x}.{yz}"
                    pairs.append((larger, smaller, f"{larger} vs {smaller}", "1d_vs_2d"))
                    if len(pairs) >= 60:
                        break
            if len(pairs) >= 60:
                break
        if len(pairs) >= 60:
            break

    # Add specific interesting cases
    extra = [
        ("5.6", "5.12", "5.6 vs 5.12", "1d_vs_2d"),
        ("7.4", "7.13", "7.4 vs 7.13", "1d_vs_2d"),
        ("2.9", "2.15", "2.9 vs 2.15", "1d_vs_2d"),
        ("3.5", "3.12", "3.5 vs 3.12", "1d_vs_2d"),
        ("8.7", "8.12", "8.7 vs 8.12", "1d_vs_2d"),
        ("6.3", "6.11", "6.3 vs 6.11", "1d_vs_2d"),
        ("4.8", "4.15", "4.8 vs 4.15", "1d_vs_2d"),
        ("1.5", "1.12", "1.5 vs 1.12", "1d_vs_2d"),
        # Two-digit integers
        ("10.9", "10.11", "10.9 vs 10.11", "2d_int"),
        ("11.8", "11.12", "11.8 vs 11.12", "2d_int"),
        ("12.7", "12.13", "12.7 vs 12.13", "2d_int"),
        ("15.6", "15.14", "15.6 vs 15.14", "2d_int"),
        # Edge cases
        ("9.9", "9.11", "9.9 vs 9.11", "edge"),
        ("9.5", "9.11", "9.5 vs 9.11", "edge"),
        ("9.2", "9.11", "9.2 vs 9.11", "edge"),
        ("9.8", "9.12", "9.8 vs 9.12", "edge"),
        ("9.8", "9.13", "9.8 vs 9.13", "edge"),
    ]

    # De-duplicate
    existing = {(a, b) for a, b, _, _ in pairs}
    for item in extra:
        if (item[0], item[1]) not in existing:
            pairs.append(item)

    return pairs[:70]  # Cap at 70 pairs


def section2_bug_hunting(patcher: HeadPatcher, n_trials_baseline: int = 20,
                         n_trials_intervention: int = 100):
    """Test 50+ decimal pairs for format-dependent bug."""
    print("\n" + "=" * 60)
    print("SECTION 2: Broader Bug Hunting")
    print("=" * 60)

    pairs = generate_decimal_pairs()
    print(f"  Testing {len(pairs)} decimal pairs")

    results = []
    buggy_pairs = []

    for idx, (larger, smaller, desc, category) in enumerate(pairs):
        qa_prompt = f"Q: Which is bigger: {larger} or {smaller}?\nA:"
        simple_prompt = f"Which is bigger: {larger} or {smaller}?\nAnswer:"

        # Baseline: check bug
        qa_results = []
        simple_results = []
        for _ in range(n_trials_baseline):
            out_qa = patcher.generate(qa_prompt)
            qa_results.append(check_answer(out_qa, larger, smaller))
            out_simple = patcher.generate(simple_prompt)
            simple_results.append(check_answer(out_simple, larger, smaller))

        qa_bug_rate = sum(1 for r in qa_results if r == "bug") / len(qa_results)
        simple_correct_rate = sum(1 for r in simple_results if r == "correct") / len(simple_results)
        bug_exists = qa_bug_rate >= 0.5 and simple_correct_rate >= 0.5

        result = {
            "pair": desc,
            "larger": larger,
            "smaller": smaller,
            "category": category,
            "qa_bug_rate": qa_bug_rate,
            "simple_correct_rate": simple_correct_rate,
            "bug_exists": bug_exists,
            "tokenization": {
                "larger": patcher.tokenizer.encode(larger),
                "smaller": patcher.tokenizer.encode(smaller),
                "larger_tokens": patcher.tokenizer.tokenize(larger),
                "smaller_tokens": patcher.tokenizer.tokenize(smaller),
            },
        }

        if bug_exists:
            buggy_pairs.append((larger, smaller, desc, category))
            print(f"  [{idx+1}/{len(pairs)}] BUG FOUND: {desc} "
                  f"(Q&A bug: {qa_bug_rate:.0%}, Simple correct: {simple_correct_rate:.0%})")

            # Test intervention
            with patcher.capture(simple_prompt) as good_attn:
                good_attn_saved = good_attn.clone()

            intervention_results = []
            for t in range(n_trials_intervention):
                with patcher.patch(good_attn_saved, INTERVENTION_HEADS):
                    out = patcher.generate(qa_prompt)
                intervention_results.append(check_answer(out, larger, smaller))

            fix_rate = sum(1 for r in intervention_results if r == "correct") / n_trials_intervention
            ci = wilson_ci(sum(1 for r in intervention_results if r == "correct"), n_trials_intervention)
            result["intervention_fix_rate"] = fix_rate
            result["intervention_ci"] = list(ci)
            result["intervention_detail"] = intervention_results
            print(f"         Intervention fix rate: {fix_rate:.0%} [{ci[0]:.0%}, {ci[1]:.0%}]")
        else:
            if (idx + 1) % 10 == 0:
                print(f"  [{idx+1}/{len(pairs)}] No bug: {desc} "
                      f"(Q&A bug: {qa_bug_rate:.0%})")

        results.append(result)

    print(f"\n  Total buggy pairs found: {len(buggy_pairs)}/{len(pairs)}")
    return results, buggy_pairs


# ── Main ────────────────────────────────────────────────────────────

def main():
    patcher = HeadPatcher()

    t_start = time.time()
    ckpt = load_checkpoint() or {}

    # Section 1: Prompt variations
    if "section1" in ckpt:
        print("\n  [checkpoint] Restoring section 1")
        s1 = ckpt["section1"]
    else:
        s1 = section1_prompt_variations(patcher, n_trials=100)
        save_checkpoint({"section1": s1})

    # Section 2: Bug hunting
    if "section2" in ckpt:
        print("\n  [checkpoint] Restoring section 2")
        s2 = ckpt["section2"]
        buggy_pairs = [(r["larger"], r["smaller"], r["pair"], r["category"])
                       for r in s2 if r.get("bug_exists")]
    else:
        s2, buggy_pairs = section2_bug_hunting(patcher)
        save_checkpoint({"section1": s1, "section2": s2})

    total_time = time.time() - t_start

    # ── Compute generalization score ────────────────────────────────
    # From section 1: variations where bug exists AND intervention works
    s1_bug_exists = [r for r in s1 if r["bug_exists"]]
    s1_intervention_works = [r for r in s1_bug_exists
                             if r.get("intervention_fix_rate", 0) >= 0.8]

    # From section 2: buggy pairs where intervention works
    s2_buggy = [r for r in s2 if r.get("bug_exists")]
    s2_intervention_works = [r for r in s2_buggy
                             if r.get("intervention_fix_rate", 0) >= 0.8]

    total_buggy = len(s1_bug_exists) + len(s2_buggy)
    total_fixable = len(s1_intervention_works) + len(s2_intervention_works)

    generalization_score = total_fixable / total_buggy if total_buggy > 0 else 0.0

    # ── Save results ────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        "experiment": "R2_generalization",
        "timestamp": timestamp,
        "model": MODEL_NAME,
        "layer": LAYER_IDX,
        "intervention_heads": INTERVENTION_HEADS,
        "total_time_seconds": total_time,
        "section1_prompt_variations": s1,
        "section2_bug_hunting": s2,
        "summary": {
            "s1_variations_tested": len(s1),
            "s1_bug_exists": len(s1_bug_exists),
            "s1_intervention_works": len(s1_intervention_works),
            "s2_pairs_tested": len(s2),
            "s2_bug_exists": len(s2_buggy),
            "s2_intervention_works": len(s2_intervention_works),
            "total_buggy": total_buggy,
            "total_fixable": total_fixable,
            "generalization_score": generalization_score,
            "buggy_pairs_found": [r["pair"] for r in s2 if r.get("bug_exists")],
        },
    }

    results_path = os.path.join(RESULTS_DIR, f"results_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda o: int(o) if isinstance(o, np.integer)
                  else float(o) if isinstance(o, np.floating)
                  else o.tolist() if isinstance(o, np.ndarray) else str(o))
    print(f"\nResults saved to {results_path}")

    # ── Print summary ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("R2 EXPERIMENT SUMMARY")
    print("=" * 70)

    print(f"\nSection 1 — Prompt Variations:")
    for r in s1:
        status = "NO BUG" if not r["bug_exists"] else \
                 f"FIX {r['intervention_fix_rate']:.0%}" if r.get("intervention_fix_rate") is not None else "???"
        print(f"  {r['name']:20s}: Q&A bug={r['qa_bug_rate']:.0%}  "
              f"Simple correct={r['simple_correct_rate']:.0%}  → {status}")

    print(f"\nSection 2 — Bug Hunting ({len(s2)} pairs tested):")
    print(f"  Buggy pairs found: {len(s2_buggy)}")
    for r in s2_buggy:
        fix = f"FIX {r['intervention_fix_rate']:.0%}" if r.get("intervention_fix_rate") is not None else "???"
        print(f"    {r['pair']:20s}: Q&A bug={r['qa_bug_rate']:.0%}  → {fix}")

    print(f"\nGeneralization Score: {generalization_score:.2f}")
    print(f"  ({total_fixable} fixable / {total_buggy} buggy)")

    if generalization_score >= 0.8:
        print("  → STRONG GENERALIZATION: Intervention works broadly")
    elif generalization_score >= 0.5:
        print("  → MODERATE GENERALIZATION: Intervention works on some variants")
    elif generalization_score > 0:
        print("  → WEAK GENERALIZATION: Intervention mostly fails on variants")
    else:
        print("  → NO GENERALIZATION or no bugs found beyond original")

    print(f"\nTotal time: {total_time/60:.1f} minutes")

    return results


if __name__ == "__main__":
    results = main()
