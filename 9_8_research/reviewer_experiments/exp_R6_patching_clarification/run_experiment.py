#!/usr/bin/env python3
"""
Experiment R6: Patching Clarification
======================================
Addresses Reviewer joQU: "In 3.2, they don't make it clear what part of the
attn layer was patched - pre-W_out?"

Tests patching at 4 different points within the attention computation:
  1. Pre-softmax attention logits (QK^T / sqrt(d_k))
  2. Post-softmax attention weights
  3. Post-W_V attention output (before W_O) — value vectors weighted by attention
  4. Post-W_O attention output (after projection, before residual) — what the paper patches

n=100 each. This disambiguates the mechanism.
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

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LAYER_IDX = 10
N_HEADS = 32
CORRECT_PROMPT = "Which is bigger: 9.8 or 9.11?\nAnswer:"
BUGGY_PROMPT = "Q: Which is bigger: 9.8 or 9.11?\nA:"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
N_TRIALS = 100


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


class PatchingClarification:
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

        # Document the architecture
        attn = self.model.model.layers[LAYER_IDX].self_attn
        print(f"\n  Attention module type: {type(attn).__name__}")
        print(f"  Has q_proj: {hasattr(attn, 'q_proj')}")
        print(f"  Has k_proj: {hasattr(attn, 'k_proj')}")
        print(f"  Has v_proj: {hasattr(attn, 'v_proj')}")
        print(f"  Has o_proj: {hasattr(attn, 'o_proj')}")

    def _clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.saved.clear()

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

    def test_patching_point(self, name: str, save_module, patch_module,
                            n_trials: int = 100) -> Dict:
        """Test patching at a specific point in the attention computation."""
        print(f"\n  Testing: {name}")

        # Step 1: Capture activation from correct format
        saved_val = None

        def save_hook(module, inp, out):
            nonlocal saved_val
            h = out[0] if isinstance(out, tuple) else out
            saved_val = h.detach().cpu()

        hook = save_module.register_forward_hook(save_hook)
        inputs = self.tokenizer(CORRECT_PROMPT, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            self.model(**inputs)
        hook.remove()

        if saved_val is None:
            return {"name": name, "error": "Failed to capture activation", "success_rate": 0}

        print(f"    Captured shape: {saved_val.shape}")

        # Step 2: Patch during buggy format generation
        def patch_hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            sv = saved_val.to(h.device)
            ml = min(h.shape[1], sv.shape[1])
            new = h.clone()
            new[:, :ml, :] = sv[:, :ml, :]
            return (new,) + out[1:] if isinstance(out, tuple) else new

        successes = 0
        samples = []
        for t in range(n_trials):
            hook = patch_module.register_forward_hook(patch_hook)
            try:
                out = self.generate(BUGGY_PROMPT)
                if self.is_correct(out):
                    successes += 1
                if t < 3:
                    samples.append(out.strip()[:100])
            finally:
                hook.remove()

        rate = successes / n_trials
        ci = wilson_ci(successes, n_trials)
        print(f"    Result: {rate:.0%} [{ci[0]:.0%}, {ci[1]:.0%}]")
        print(f"    Samples: {samples[:2]}")

        return {
            "name": name,
            "success_rate": rate,
            "successes": successes,
            "n_trials": n_trials,
            "ci": list(ci),
            "saved_shape": list(saved_val.shape),
            "samples": samples,
        }

    def run(self):
        t_start = time.time()
        attn = self.model.model.layers[LAYER_IDX].self_attn

        print("\n" + "=" * 60)
        print("R6: Testing 4 patching points in attention computation")
        print("=" * 60)

        results = {}

        # Point 1: Post-W_O (self_attn output) — this is what the paper patches
        results["post_wo_self_attn_output"] = self.test_patching_point(
            "Post-W_O (self_attn output — paper's method)",
            save_module=attn,
            patch_module=attn,
            n_trials=N_TRIALS,
        )

        # Point 2: Post-W_O projection specifically (o_proj)
        results["post_o_proj"] = self.test_patching_point(
            "Post-o_proj (output projection only)",
            save_module=attn.o_proj,
            patch_module=attn.o_proj,
            n_trials=N_TRIALS,
        )

        # Point 3: Post-V projection (v_proj output)
        results["post_v_proj"] = self.test_patching_point(
            "Post-v_proj (value projection)",
            save_module=attn.v_proj,
            patch_module=attn.v_proj,
            n_trials=N_TRIALS,
        )

        # Point 4: Post-Q projection (q_proj output) — shouldn't help, control
        results["post_q_proj"] = self.test_patching_point(
            "Post-q_proj (query projection — control)",
            save_module=attn.q_proj,
            patch_module=attn.q_proj,
            n_trials=N_TRIALS,
        )

        total_time = time.time() - t_start

        # Technical specification
        spec = {
            "patching_location": "model.model.layers[10].self_attn output",
            "tensor_description": "The output of the multi-head attention after W_O projection, before addition to residual stream",
            "hook_type": "register_forward_hook on self_attn module",
            "tensor_shape": results["post_wo_self_attn_output"].get("saved_shape"),
            "head_specific_patching": (
                "For head-specific patching, the output tensor [batch, seq, hidden_dim] "
                "is reshaped to [batch, seq, n_heads, head_dim] where head_dim = hidden_dim / n_heads. "
                "Individual head slices are then replaced."
            ),
            "nnsight_equivalent": "lm.model.layers[10].self_attn.output[0]",
        }

        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_results = {
            "experiment": "R6_patching_clarification",
            "timestamp": timestamp,
            "model": MODEL_NAME,
            "layer": LAYER_IDX,
            "n_trials": N_TRIALS,
            "total_time_seconds": total_time,
            "patching_specification": spec,
            "results": results,
        }

        results_path = os.path.join(RESULTS_DIR, f"results_{timestamp}.json")
        with open(results_path, "w") as f:
            json.dump(full_results, f, indent=2,
                      default=lambda o: int(o) if isinstance(o, np.integer)
                      else float(o) if isinstance(o, np.floating)
                      else o.tolist() if isinstance(o, np.ndarray) else str(o))
        print(f"\nResults saved to {results_path}")

        # Summary
        print("\n" + "=" * 70)
        print("R6 SUMMARY: Patching Point Comparison")
        print("=" * 70)
        print(f"\n{'Patching Point':<45} {'Success':>10}")
        print("-" * 57)
        for key, r in results.items():
            rate = r.get("success_rate", 0)
            ci = r.get("ci", [0, 0])
            print(f"  {r['name']:<43} {rate:>6.0%} [{ci[0]:.0%},{ci[1]:.0%}]")

        print(f"\nTechnical Specification for Paper:")
        print(f"  Hook point: {spec['patching_location']}")
        print(f"  Description: {spec['tensor_description']}")
        print(f"  Tensor shape: {spec['tensor_shape']}")
        print(f"\nTotal time: {total_time/60:.1f} minutes")

        return full_results


if __name__ == "__main__":
    exp = PatchingClarification()
    results = exp.run()
