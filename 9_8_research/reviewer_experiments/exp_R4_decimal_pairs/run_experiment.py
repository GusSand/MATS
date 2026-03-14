#!/usr/bin/env python3
"""
Experiment R4: Expanded Decimal Pair Generalization
====================================================
Addresses both reviewers on narrow scope. Complements R2's bug hunting with:

1. 10.9 vs 10.11 deep dive (specifically called out by joQU)
   - Full tokenization analysis
   - Logit lens divergence
   - Attention patching at ALL layers
   - MLP patching for this pair
2. Tokenization grouping: group R2 results by tokenization pattern
3. Additional systematically generated pairs not in R2

NOTE: Run AFTER R2 to avoid redundant computation. Uses R2 results.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
from contextlib import contextmanager
import json
from datetime import datetime
import time
import os
import glob

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
R2_DIR = os.path.join(os.path.dirname(RESULTS_DIR), "exp_R2_generalization")
N_LAYERS = 32
N_HEADS = 32
EVEN_HEADS_8 = [0, 2, 4, 6, 8, 10, 12, 14]


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


class DeepDiveExperiment:
    def __init__(self):
        print(f"Loading {MODEL_NAME} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
        )
        self.model.eval()
        self.lm_head = self.model.lm_head.weight
        self.norm = self.model.model.norm
        print("Model loaded.")

    def generate(self, prompt, max_new=30):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                      pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)

    def check_answer(self, text, correct_val, wrong_val):
        lo = text.lower()
        correct = any(f"{correct_val} is {w}" in lo for w in
                      ["bigger", "larger", "greater", "the bigger", "the larger"])
        bug = any(f"{wrong_val} is {w}" in lo for w in
                  ["bigger", "larger", "greater", "the bigger", "the larger"])
        if not correct and not bug:
            first = text.strip().split("\n")[0]
            if correct_val in first and wrong_val not in first:
                correct = True
            elif wrong_val in first and correct_val not in first:
                bug = True
        if correct and not bug:
            return "correct"
        elif bug and not correct:
            return "bug"
        return "unclear"

    def tokenization_analysis(self, val: str) -> Dict:
        """Detailed tokenization of a value."""
        ids = self.tokenizer.encode(val, add_special_tokens=False)
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return {"value": val, "token_ids": ids, "tokens": tokens, "n_tokens": len(ids)}

    def logit_lens_all_layers(self, prompt: str, token_ids: List[int]) -> Dict:
        """Get logit lens projections at all layers for specified token IDs."""
        hidden_states = {}
        hooks = []
        for L in range(N_LAYERS):
            def make_hook(layer_idx):
                def fn(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    hidden_states[layer_idx] = h.detach()
                return fn
            hooks.append(self.model.model.layers[L].register_forward_hook(make_hook(L)))

        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            self.model(**inputs)
        for h in hooks:
            h.remove()

        results = {}
        for L in range(N_LAYERS):
            h = hidden_states[L][:, -1, :]  # final token
            with torch.no_grad():
                normed = self.norm(h.float()).half()
                logits = normed @ self.lm_head.T
            results[L] = {tid: float(logits[0, tid].cpu()) for tid in token_ids}
        return results

    def attn_patching_all_layers(self, simple_prompt, buggy_prompt,
                                  correct_val, wrong_val, n_trials=50) -> Dict:
        """Test attention patching at every layer."""
        results = {}
        for L in range(N_LAYERS):
            # Capture
            saved = None
            def save_fn(module, inp, out):
                nonlocal saved
                h = out[0] if isinstance(out, tuple) else out
                saved = h.detach().cpu()
            hook = self.model.model.layers[L].self_attn.register_forward_hook(save_fn)
            inputs = self.tokenizer(simple_prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                self.model(**inputs)
            hook.remove()

            if saved is None:
                results[L] = {"success_rate": 0, "error": "capture failed"}
                continue

            # Patch
            def make_patch(s):
                def fn(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    sv = s.to(h.device)
                    ml = min(h.shape[1], sv.shape[1])
                    new = h.clone()
                    new[:, :ml, :] = sv[:, :ml, :]
                    return (new,) + out[1:] if isinstance(out, tuple) else new
                return fn

            successes = 0
            for _ in range(n_trials):
                hook = self.model.model.layers[L].self_attn.register_forward_hook(
                    make_patch(saved))
                try:
                    out = self.generate(buggy_prompt)
                    if self.check_answer(out, correct_val, wrong_val) == "correct":
                        successes += 1
                finally:
                    hook.remove()

            rate = successes / n_trials
            ci = wilson_ci(successes, n_trials)
            results[L] = {"success_rate": rate, "ci": list(ci),
                          "successes": successes, "n_trials": n_trials}
            if rate > 0:
                print(f"      Layer {L:2d}: {rate:.0%}")

        return results

    def mlp_patching_all_layers(self, simple_prompt, buggy_prompt,
                                 correct_val, wrong_val, n_trials=50) -> Dict:
        """Test MLP patching at every layer."""
        results = {}
        for L in range(N_LAYERS):
            saved = None
            def save_fn(module, inp, out):
                nonlocal saved
                h = out[0] if isinstance(out, tuple) else out
                saved = h.detach().cpu()
            hook = self.model.model.layers[L].mlp.register_forward_hook(save_fn)
            inputs = self.tokenizer(simple_prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                self.model(**inputs)
            hook.remove()

            if saved is None:
                results[L] = {"success_rate": 0, "error": "capture failed"}
                continue

            def make_patch(s):
                def fn(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    sv = s.to(h.device)
                    ml = min(h.shape[1], sv.shape[1])
                    new = h.clone()
                    new[:, :ml, :] = sv[:, :ml, :]
                    return (new,) + out[1:] if isinstance(out, tuple) else new
                return fn

            successes = 0
            for _ in range(n_trials):
                hook = self.model.model.layers[L].mlp.register_forward_hook(
                    make_patch(saved))
                try:
                    out = self.generate(buggy_prompt)
                    if self.check_answer(out, correct_val, wrong_val) == "correct":
                        successes += 1
                finally:
                    hook.remove()

            rate = successes / n_trials
            ci = wilson_ci(successes, n_trials)
            results[L] = {"success_rate": rate, "ci": list(ci),
                          "successes": successes, "n_trials": n_trials}
            if rate > 0:
                print(f"      Layer {L:2d}: {rate:.0%}")

        return results

    def deep_dive_10_9_vs_10_11(self) -> Dict:
        """Full analysis of the 10.9 vs 10.11 counterexample."""
        print("\n" + "=" * 60)
        print("DEEP DIVE: 10.9 vs 10.11")
        print("=" * 60)

        larger = "10.9"
        smaller = "10.11"
        qa = f"Q: Which is bigger: {larger} or {smaller}?\nA:"
        simple = f"Which is bigger: {larger} or {smaller}?\nAnswer:"

        results = {}

        # 1. Tokenization
        print("\n  1. Tokenization:")
        tok_larger = self.tokenization_analysis(larger)
        tok_smaller = self.tokenization_analysis(smaller)
        tok_98 = self.tokenization_analysis("9.8")
        tok_911 = self.tokenization_analysis("9.11")
        results["tokenization"] = {
            "10.9": tok_larger, "10.11": tok_smaller,
            "9.8_reference": tok_98, "9.11_reference": tok_911,
        }
        print(f"    10.9:  {tok_larger['tokens']} ({tok_larger['n_tokens']} tokens)")
        print(f"    10.11: {tok_smaller['tokens']} ({tok_smaller['n_tokens']} tokens)")
        print(f"    9.8:   {tok_98['tokens']} ({tok_98['n_tokens']} tokens) [reference]")
        print(f"    9.11:  {tok_911['tokens']} ({tok_911['n_tokens']} tokens) [reference]")

        # 2. Baseline
        print("\n  2. Baseline (20 trials):")
        for name, prompt in [("qa", qa), ("simple", simple)]:
            answers = []
            for _ in range(20):
                out = self.generate(prompt)
                answers.append(self.check_answer(out, larger, smaller))
            bug_rate = sum(1 for a in answers if a == "bug") / len(answers)
            correct_rate = sum(1 for a in answers if a == "correct") / len(answers)
            results[f"baseline_{name}"] = {
                "bug_rate": bug_rate, "correct_rate": correct_rate,
                "answers": answers,
            }
            print(f"    {name:6s}: bug={bug_rate:.0%}, correct={correct_rate:.0%}")

        bug_exists = (results["baseline_qa"]["bug_rate"] >= 0.5 and
                      results["baseline_simple"]["correct_rate"] >= 0.5)
        results["bug_exists"] = bug_exists
        print(f"    Format-dependent bug exists: {bug_exists}")

        # 3. Logit lens
        print("\n  3. Logit lens comparison:")
        # Get relevant token IDs
        rel_tokens = {}
        for s in ["10", ".9", ".11", "9", "10.9", "10.11"]:
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            rel_tokens[s] = ids
        all_ids = set()
        for ids in rel_tokens.values():
            all_ids.update(ids)
        all_ids = list(all_ids)

        ll_qa = self.logit_lens_all_layers(qa, all_ids)
        ll_simple = self.logit_lens_all_layers(simple, all_ids)
        results["logit_lens"] = {
            "qa": {str(k): v for k, v in ll_qa.items()},
            "simple": {str(k): v for k, v in ll_simple.items()},
            "token_ids": rel_tokens,
        }

        # Find divergence layer
        for L in range(N_LAYERS):
            # Compare logit patterns between formats
            qa_logits = ll_qa[L]
            simple_logits = ll_simple[L]
            max_diff = max(abs(qa_logits.get(tid, 0) - simple_logits.get(tid, 0))
                          for tid in all_ids)
            if max_diff > 2.0:
                print(f"    Layer {L}: max logit divergence = {max_diff:.2f}")

        # 4. Attention patching at ALL layers
        print("\n  4. Attention patching at all 32 layers:")
        results["attn_patching"] = self.attn_patching_all_layers(
            simple, qa, larger, smaller, n_trials=50)
        working_layers = [L for L, r in results["attn_patching"].items()
                         if r["success_rate"] >= 0.5]
        print(f"    Working layers (>=50%): {working_layers}")

        # 5. MLP patching at all layers
        print("\n  5. MLP patching at all 32 layers:")
        results["mlp_patching"] = self.mlp_patching_all_layers(
            simple, qa, larger, smaller, n_trials=50)
        working_mlp = [L for L, r in results["mlp_patching"].items()
                      if r["success_rate"] >= 0.5]
        print(f"    Working layers (>=50%): {working_mlp}")

        return results

    def deep_dive_original(self) -> Dict:
        """Same analysis for 9.8 vs 9.11 as comparison."""
        print("\n" + "=" * 60)
        print("COMPARISON: 9.8 vs 9.11 (original)")
        print("=" * 60)

        larger = "9.8"
        smaller = "9.11"
        qa = f"Q: Which is bigger: {larger} or {smaller}?\nA:"
        simple = f"Which is bigger: {larger} or {smaller}?\nAnswer:"

        results = {}

        # Tokenization
        results["tokenization"] = {
            "9.8": self.tokenization_analysis("9.8"),
            "9.11": self.tokenization_analysis("9.11"),
        }
        print(f"  9.8:  {results['tokenization']['9.8']['tokens']}")
        print(f"  9.11: {results['tokenization']['9.11']['tokens']}")

        return results


def main():
    exp = DeepDiveExperiment()
    t_start = time.time()

    # Main deep dive: 10.9 vs 10.11
    dd_10 = exp.deep_dive_10_9_vs_10_11()

    # Reference: 9.8 vs 9.11 tokenization
    dd_9 = exp.deep_dive_original()

    total_time = time.time() - t_start

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "experiment": "R4_decimal_pairs",
        "timestamp": timestamp,
        "model": MODEL_NAME,
        "total_time_seconds": total_time,
        "deep_dive_10_9_vs_10_11": dd_10,
        "reference_9_8_vs_9_11": dd_9,
    }

    results_path = os.path.join(RESULTS_DIR, f"results_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda o: int(o) if isinstance(o, np.integer)
                  else float(o) if isinstance(o, np.floating)
                  else o.tolist() if isinstance(o, np.ndarray) else str(o))
    print(f"\nResults saved to {results_path}")

    print(f"\n" + "=" * 70)
    print("R4 SUMMARY: 10.9 vs 10.11 Deep Dive")
    print("=" * 70)
    print(f"\n  Bug exists: {dd_10['bug_exists']}")
    attn_working = [L for L, r in dd_10["attn_patching"].items() if r["success_rate"] >= 0.5]
    mlp_working = [L for L, r in dd_10["mlp_patching"].items() if r["success_rate"] >= 0.5]
    print(f"  Attn patching works at layers: {attn_working}")
    print(f"  MLP patching works at layers: {mlp_working}")
    print(f"  Tokenization difference from 9.8/9.11: "
          f"{dd_10['tokenization']['10.9']['tokens']} vs "
          f"{dd_10['tokenization']['10.11']['tokens']}")
    print(f"\n  Total time: {total_time/60:.1f} minutes")

    return results


if __name__ == "__main__":
    results = main()
