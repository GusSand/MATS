#!/usr/bin/env python3
"""
Experiment R3: MLP Contribution Analysis
=========================================
Addresses Reviewer joQU's critique: "the authors focus on attention throughout
and fail to rationalize why they believe attention is the focal mechanism in
this language model error, rather than how MLP layers represent the numeric
information."

Components:
  1. MLP-only patching sweep across layers
  2. MLP + attention combined patching
  3. Attention-fixed, MLP-varied (confirmation of sufficiency)
  4. MLP knockout at Layer 10
  5. Logit lens decomposition: attention vs MLP contribution
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
CORRECT_PROMPT = "Which is bigger: 9.8 or 9.11?\nAnswer:"
BUGGY_PROMPT = "Q: Which is bigger: 9.8 or 9.11?\nA:"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "checkpoint.json")
TEST_LAYERS = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25]
N_TRIALS = 100


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


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


class MLPExperiment:
    """Test MLP vs Attention contributions to the decimal bug."""

    def __init__(self):
        print(f"Loading {MODEL_NAME} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
        )
        self.model.eval()
        self.hooks: list = []
        self.saved: dict = {}
        print("Model loaded.")

    def _clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.saved.clear()

    def _make_save_hook(self, key: str):
        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            self.saved[key] = h.detach().cpu()
        return fn

    def _make_patch_hook(self, saved_val: torch.Tensor):
        """Replace entire output with saved value."""
        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            bs, sl, hs = h.shape
            s = saved_val.to(h.device)
            ml = min(sl, s.shape[1])
            new = h.clone()
            new[:, :ml, :] = s[:, :ml, :]
            return (new,) + out[1:] if isinstance(out, tuple) else new
        return fn

    def _make_zero_hook(self):
        """Zero out the output."""
        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            zeros = torch.zeros_like(h)
            return (zeros,) + out[1:] if isinstance(out, tuple) else zeros
        return fn

    def generate(self, prompt: str, max_new: int = 30) -> str:
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

    def capture_activations(self, prompt: str, layers: List[int]):
        """Capture both attention and MLP outputs at specified layers."""
        results = {}
        hooks = []
        try:
            for L in layers:
                attn_key = f"attn_{L}"
                mlp_key = f"mlp_{L}"
                h1 = self.model.model.layers[L].self_attn.register_forward_hook(
                    self._make_save_hook(attn_key))
                h2 = self.model.model.layers[L].mlp.register_forward_hook(
                    self._make_save_hook(mlp_key))
                hooks.extend([h1, h2])

            inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                self.model(**inputs)

            for key, val in self.saved.items():
                results[key] = val.clone()
        finally:
            for h in hooks:
                h.remove()
            hooks.clear()
            self.saved.clear()
        return results

    def test_with_hooks(self, prompt: str, hook_targets: list, n_trials: int) -> Dict:
        """Run generation n times with specified hooks active."""
        successes = 0
        samples = []
        for t in range(n_trials):
            hooks = []
            try:
                for module, hook_fn in hook_targets:
                    h = module.register_forward_hook(hook_fn)
                    hooks.append(h)
                out = self.generate(prompt)
                if self.is_correct(out):
                    successes += 1
                if t < 2:
                    samples.append(out.strip()[:100])
            finally:
                for h in hooks:
                    h.remove()

        ci = wilson_ci(successes, n_trials)
        return {
            "success_rate": successes / n_trials,
            "successes": successes,
            "n_trials": n_trials,
            "ci": list(ci),
            "samples": samples,
        }

    # ── Section 1: MLP-only patching ────────────────────────────────

    def section1_mlp_only(self, n_trials: int = 100) -> Dict:
        print("\n" + "=" * 60)
        print("SECTION 1: MLP-only patching (Simple → Q&A)")
        print("=" * 60)

        # Capture activations from correct format
        good = self.capture_activations(CORRECT_PROMPT, TEST_LAYERS)

        results = {}
        for L in TEST_LAYERS:
            mlp_key = f"mlp_{L}"
            if mlp_key not in good:
                print(f"  Layer {L}: SKIPPED (no activation)")
                continue

            hook_targets = [
                (self.model.model.layers[L].mlp, self._make_patch_hook(good[mlp_key]))
            ]
            r = self.test_with_hooks(BUGGY_PROMPT, hook_targets, n_trials)
            results[L] = r
            print(f"  Layer {L:2d} MLP-only: {r['success_rate']:.0%} "
                  f"[{r['ci'][0]:.0%}, {r['ci'][1]:.0%}]  {r['samples'][:1]}")

        return results

    # ── Section 2: MLP + Attention combined ─────────────────────────

    def section2_combined(self, n_trials: int = 100) -> Dict:
        print("\n" + "=" * 60)
        print("SECTION 2: MLP + Attention combined patching")
        print("=" * 60)

        good = self.capture_activations(CORRECT_PROMPT, TEST_LAYERS)

        results = {}
        for L in TEST_LAYERS:
            attn_key = f"attn_{L}"
            mlp_key = f"mlp_{L}"
            if attn_key not in good or mlp_key not in good:
                continue

            hook_targets = [
                (self.model.model.layers[L].self_attn, self._make_patch_hook(good[attn_key])),
                (self.model.model.layers[L].mlp, self._make_patch_hook(good[mlp_key])),
            ]
            r = self.test_with_hooks(BUGGY_PROMPT, hook_targets, n_trials)
            results[L] = r
            print(f"  Layer {L:2d} Attn+MLP: {r['success_rate']:.0%} "
                  f"[{r['ci'][0]:.0%}, {r['ci'][1]:.0%}]")

        return results

    # ── Section 3: Attention-only at Layer 10 (baseline) ────────────

    def section3_attn_only(self, n_trials: int = 100) -> Dict:
        print("\n" + "=" * 60)
        print("SECTION 3: Attention-only patching at Layer 10 (baseline)")
        print("=" * 60)

        good = self.capture_activations(CORRECT_PROMPT, [LAYER_IDX])

        attn_key = f"attn_{LAYER_IDX}"
        hook_targets = [
            (self.model.model.layers[LAYER_IDX].self_attn,
             self._make_patch_hook(good[attn_key]))
        ]
        r = self.test_with_hooks(BUGGY_PROMPT, hook_targets, n_trials)
        print(f"  Layer {LAYER_IDX} Attn-only: {r['success_rate']:.0%} "
              f"[{r['ci'][0]:.0%}, {r['ci'][1]:.0%}]")
        return r

    # ── Section 4: MLP knockout at Layer 10 ─────────────────────────

    def section4_mlp_knockout(self, n_trials: int = 100) -> Dict:
        print("\n" + "=" * 60)
        print("SECTION 4: MLP knockout (zero out) at Layer 10")
        print("=" * 60)

        results = {}

        # Test on buggy format: does bug persist without MLP?
        hook_targets_buggy = [
            (self.model.model.layers[LAYER_IDX].mlp, self._make_zero_hook())
        ]
        r_buggy = self.test_with_hooks(BUGGY_PROMPT, hook_targets_buggy, n_trials)
        results["buggy_format_no_mlp"] = r_buggy
        print(f"  Q&A format, MLP zeroed: correct={r_buggy['success_rate']:.0%}")
        print(f"    Samples: {r_buggy['samples']}")

        # Test on correct format: does model still work?
        r_correct = self.test_with_hooks(CORRECT_PROMPT, hook_targets_buggy, n_trials)
        results["correct_format_no_mlp"] = r_correct
        print(f"  Simple format, MLP zeroed: correct={r_correct['success_rate']:.0%}")
        print(f"    Samples: {r_correct['samples']}")

        return results

    # ── Section 5: Logit lens decomposition ─────────────────────────

    def section5_logit_lens(self) -> Dict:
        """Project attention and MLP outputs separately through unembedding."""
        print("\n" + "=" * 60)
        print("SECTION 5: Logit Lens Decomposition at Layer 10")
        print("=" * 60)

        # Get relevant token IDs
        tokens_of_interest = {}
        for tok_str in ["9", ".8", ".11", "8", "11", "9.8", "9.11"]:
            ids = self.tokenizer.encode(tok_str, add_special_tokens=False)
            tokens_of_interest[tok_str] = ids
        print(f"  Token IDs: {tokens_of_interest}")

        results = {"token_ids": tokens_of_interest}

        for prompt_name, prompt in [("simple", CORRECT_PROMPT), ("qa", BUGGY_PROMPT)]:
            print(f"\n  Analyzing {prompt_name} format...")

            # Hook to capture both attn and mlp outputs AND the residual before layer 10
            captured = {}
            hooks = []

            def make_capture(name):
                def fn(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    captured[name] = h.detach()
                return fn

            # Also capture input to layer 10 (residual stream before attention)
            def capture_input(name):
                def fn(module, inp, out):
                    # Input to the layer is the first element of inp
                    if isinstance(inp, tuple) and len(inp) > 0:
                        captured[name] = inp[0].detach()
                return fn

            hooks.append(self.model.model.layers[LAYER_IDX].register_forward_hook(
                capture_input("layer_input")))
            hooks.append(self.model.model.layers[LAYER_IDX].self_attn.register_forward_hook(
                make_capture("attn_out")))
            hooks.append(self.model.model.layers[LAYER_IDX].mlp.register_forward_hook(
                make_capture("mlp_out")))

            inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                _ = self.model(**inputs)

            for h in hooks:
                h.remove()

            # Get unembedding matrix and layer norm
            lm_head = self.model.lm_head.weight  # [vocab, hidden]
            norm = self.model.model.norm  # final RMSNorm

            # Project at last token position
            last_pos = -1  # final token
            attn_h = captured["attn_out"][:, last_pos, :]  # [1, hidden]
            mlp_h = captured["mlp_out"][:, last_pos, :]

            # Apply layer norm then project
            with torch.no_grad():
                attn_normed = norm(attn_h.float()).half()
                mlp_normed = norm(mlp_h.float()).half()
                attn_logits = attn_normed @ lm_head.T  # [1, vocab]
                mlp_logits = mlp_normed @ lm_head.T

            # Extract logits for tokens of interest
            prompt_results = {}
            for tok_str, tok_ids in tokens_of_interest.items():
                for tid in tok_ids:
                    key = f"{tok_str}_id{tid}"
                    prompt_results[key] = {
                        "attn_logit": float(attn_logits[0, tid].cpu()),
                        "mlp_logit": float(mlp_logits[0, tid].cpu()),
                    }

            # Key comparison: logit(".8") - logit(".11") from each component
            # Find token IDs for .8 and .11
            id_8 = tokens_of_interest.get(".8", tokens_of_interest.get("8", [None]))[0]
            id_11 = tokens_of_interest.get(".11", tokens_of_interest.get("11", [None]))[0]

            if id_8 is not None and id_11 is not None:
                prompt_results["attn_logit_diff_8_minus_11"] = float(
                    attn_logits[0, id_8].cpu() - attn_logits[0, id_11].cpu())
                prompt_results["mlp_logit_diff_8_minus_11"] = float(
                    mlp_logits[0, id_8].cpu() - mlp_logits[0, id_11].cpu())

            results[prompt_name] = prompt_results
            print(f"    Attn logit diff (.8-.11): "
                  f"{prompt_results.get('attn_logit_diff_8_minus_11', 'N/A')}")
            print(f"    MLP logit diff (.8-.11): "
                  f"{prompt_results.get('mlp_logit_diff_8_minus_11', 'N/A')}")

        return results


# ── Main ────────────────────────────────────────────────────────────

def main():
    exp = MLPExperiment()
    t_start = time.time()
    ckpt = load_checkpoint() or {}

    if "section1" in ckpt:
        print("\n  [checkpoint] Restoring section 1")
        s1 = ckpt["section1"]
    else:
        s1 = exp.section1_mlp_only(N_TRIALS)
        save_checkpoint({"section1": {str(k): v for k, v in s1.items()}})

    if "section2" in ckpt:
        print("\n  [checkpoint] Restoring section 2")
        s2 = ckpt["section2"]
    else:
        s2 = exp.section2_combined(N_TRIALS)
        save_checkpoint({"section1": {str(k): v for k, v in s1.items()},
                         "section2": {str(k): v for k, v in s2.items()}})

    if "section3" in ckpt:
        print("\n  [checkpoint] Restoring section 3")
        s3 = ckpt["section3"]
    else:
        s3 = exp.section3_attn_only(N_TRIALS)

    if "section4" in ckpt:
        print("\n  [checkpoint] Restoring section 4")
        s4 = ckpt["section4"]
    else:
        s4 = exp.section4_mlp_knockout(N_TRIALS)

    s5 = exp.section5_logit_lens()

    total_time = time.time() - t_start

    # ── Save results ────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        "experiment": "R3_mlp_analysis",
        "timestamp": timestamp,
        "model": MODEL_NAME,
        "layer": LAYER_IDX,
        "test_layers": TEST_LAYERS,
        "n_trials": N_TRIALS,
        "total_time_seconds": total_time,
        "section1_mlp_only": {str(k): v for k, v in s1.items()} if isinstance(s1, dict) and any(isinstance(k, int) for k in s1.keys()) else s1,
        "section2_attn_plus_mlp": {str(k): v for k, v in s2.items()} if isinstance(s2, dict) and any(isinstance(k, int) for k in s2.keys()) else s2,
        "section3_attn_only_layer10": s3,
        "section4_mlp_knockout": s4,
        "section5_logit_lens": s5,
    }

    results_path = os.path.join(RESULTS_DIR, f"results_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda o: int(o) if isinstance(o, np.integer)
                  else float(o) if isinstance(o, np.floating)
                  else o.tolist() if isinstance(o, np.ndarray) else str(o))
    print(f"\nResults saved to {results_path}")

    # ── Summary table ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("R3 EXPERIMENT SUMMARY: MLP vs Attention")
    print("=" * 70)

    print("\n┌─────────┬───────────────────┬───────────────────┬───────────────────┐")
    print("│  Layer  │    MLP-only       │   Attn+MLP        │   (Attn-only L10) │")
    print("├─────────┼───────────────────┼───────────────────┼───────────────────┤")
    for L in TEST_LAYERS:
        lk = str(L) if isinstance(list(s1.keys())[0], str) else L
        mlp_r = s1.get(lk, {}).get("success_rate", -1)
        comb_r = s2.get(lk, {}).get("success_rate", -1)
        mlp_str = f"{mlp_r:.0%}" if mlp_r >= 0 else "N/A"
        comb_str = f"{comb_r:.0%}" if comb_r >= 0 else "N/A"
        attn_str = f"{s3['success_rate']:.0%}" if L == LAYER_IDX else ""
        print(f"│   {L:2d}    │     {mlp_str:^8s}      │     {comb_str:^8s}      │     {attn_str:^8s}      │")
    print("└─────────┴───────────────────┴───────────────────┴───────────────────┘")

    print(f"\nMLP Knockout at Layer 10:")
    print(f"  Q&A (buggy) without MLP: correct={s4['buggy_format_no_mlp']['success_rate']:.0%}")
    print(f"  Simple (correct) without MLP: correct={s4['correct_format_no_mlp']['success_rate']:.0%}")

    print(f"\nLogit Lens Decomposition:")
    for fmt in ["simple", "qa"]:
        d = s5.get(fmt, {})
        attn_diff = d.get("attn_logit_diff_8_minus_11", "N/A")
        mlp_diff = d.get("mlp_logit_diff_8_minus_11", "N/A")
        print(f"  {fmt:6s}: Attn logit(.8-.11)={attn_diff}, MLP logit(.8-.11)={mlp_diff}")

    print(f"\nConclusion:")
    mlp_any_work = any(s1.get(str(L) if isinstance(list(s1.keys())[0], str) else L, {}).get("success_rate", 0) >= 0.8
                       for L in TEST_LAYERS)
    attn_works = s3["success_rate"] >= 0.8
    if attn_works and not mlp_any_work:
        print("  ATTENTION is the critical component. MLP patching at NO layer fixes the bug.")
    elif attn_works and mlp_any_work:
        print("  Both Attention and MLP patching can fix the bug at some layers.")
    else:
        print("  Unexpected result — review data.")

    print(f"\nTotal time: {total_time/60:.1f} minutes")
    return results


if __name__ == "__main__":
    results = main()
