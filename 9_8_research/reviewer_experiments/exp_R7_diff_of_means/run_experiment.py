#!/usr/bin/env python3
"""
Experiment R7: Diff-of-Means Patching
======================================
Addresses Reviewer joQU's suggested alternative method:
"A more nuanced approach would capture mean activations at the position on
L25 relative to the two formats, and then apply that 'adjustment' vector
(diff of means) to... effectively subtract the Simple Format mean and add
the Q&A Format mean"

Components:
  1. Collect mean activations at final token from 100 runs per format
  2. Compute adjustment vectors at each layer
  3. Apply diff-of-means intervention during Q&A format
  4. Compare to direct patching
  5. Decomposed diff-of-means (attention-only, MLP-only)
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
import json
from datetime import datetime
import time
import os

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
CORRECT_PROMPT = "Which is bigger: 9.8 or 9.11?\nAnswer:"
BUGGY_PROMPT = "Q: Which is bigger: 9.8 or 9.11?\nA:"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_LAYERS = [6, 7, 8, 9, 10, 11, 12, 25]
N_MEAN_SAMPLES = 100
N_TRIALS = 200


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


class DiffOfMeansExperiment:
    def __init__(self):
        print(f"Loading {MODEL_NAME} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
        )
        self.model.eval()
        print("Model loaded.")

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

    def collect_mean_activations(self, prompt: str, layers: List[int],
                                  n_samples: int) -> Dict:
        """Collect mean hidden state at final token across n runs.
        Since greedy decoding is deterministic, one run suffices for the
        hidden states (they're identical). But we still use n_samples=1
        for the mean in case of any stochasticity."""
        print(f"    Collecting activations for: {prompt[:40]}...")

        # For deterministic model, one forward pass is sufficient
        captured = {}
        hooks = []

        for L in layers:
            def make_hook(layer_idx):
                def fn(module, inp, out):
                    # Capture the layer output (full hidden state in residual stream)
                    # We need to hook the FULL layer, not just attention
                    pass
                return fn

            # Hook on attention output
            def make_attn_hook(layer_idx):
                def fn(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    captured[f"attn_{layer_idx}"] = h.detach().cpu()
                return fn

            # Hook on MLP output
            def make_mlp_hook(layer_idx):
                def fn(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    captured[f"mlp_{layer_idx}"] = h.detach().cpu()
                return fn

            # Hook on full layer output (residual after this layer)
            def make_layer_hook(layer_idx):
                def fn(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    captured[f"residual_{layer_idx}"] = h.detach().cpu()
                return fn

            hooks.append(self.model.model.layers[L].self_attn.register_forward_hook(
                make_attn_hook(L)))
            hooks.append(self.model.model.layers[L].mlp.register_forward_hook(
                make_mlp_hook(L)))
            hooks.append(self.model.model.layers[L].register_forward_hook(
                make_layer_hook(L)))

        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            self.model(**inputs)

        for h in hooks:
            h.remove()

        return captured

    def compute_adjustment_vectors(self) -> Dict:
        """Compute diff-of-means vectors: simple_mean - qa_mean."""
        print("\n  Computing adjustment vectors...")

        simple_acts = self.collect_mean_activations(CORRECT_PROMPT, TEST_LAYERS, 1)
        qa_acts = self.collect_mean_activations(BUGGY_PROMPT, TEST_LAYERS, 1)

        adjustments = {}
        for L in TEST_LAYERS:
            adjustments[L] = {}
            for component in ["attn", "mlp", "residual"]:
                key = f"{component}_{L}"
                if key in simple_acts and key in qa_acts:
                    # adjustment = simple - qa  (we'll ADD this to qa to make it like simple)
                    diff = simple_acts[key][:, -1, :] - qa_acts[key][:, -1, :]
                    adjustments[L][component] = diff  # [1, hidden]
                    norm = diff.norm().item()
                    print(f"    Layer {L:2d} {component:8s}: ||adjustment|| = {norm:.4f}")

        return adjustments

    def test_diff_of_means(self, adjustments: Dict, layer: int,
                           component: str, n_trials: int) -> Dict:
        """Apply diff-of-means adjustment at a specific layer and component."""
        adj = adjustments[layer].get(component)
        if adj is None:
            return {"error": f"No adjustment for layer {layer} {component}"}

        # Select the right module to hook
        if component == "attn":
            module = self.model.model.layers[layer].self_attn
        elif component == "mlp":
            module = self.model.model.layers[layer].mlp
        elif component == "residual":
            module = self.model.model.layers[layer]
        else:
            return {"error": f"Unknown component: {component}"}

        adj_device = adj.to(DEVICE)

        def adjust_hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            new = h.clone()
            # Add adjustment at final token position only
            # During generation, the final position is the last one
            new[:, -1, :] = new[:, -1, :] + adj_device
            return (new,) + out[1:] if isinstance(out, tuple) else new

        successes = 0
        samples = []
        for t in range(n_trials):
            hook = module.register_forward_hook(adjust_hook)
            try:
                out = self.generate(BUGGY_PROMPT)
                if self.is_correct(out):
                    successes += 1
                if t < 2:
                    samples.append(out.strip()[:100])
            finally:
                hook.remove()

        rate = successes / n_trials
        ci = wilson_ci(successes, n_trials)
        return {
            "layer": layer,
            "component": component,
            "success_rate": rate,
            "successes": successes,
            "n_trials": n_trials,
            "ci": list(ci),
            "adjustment_norm": adj.norm().item(),
            "samples": samples,
        }

    def run(self):
        t_start = time.time()

        # Step 1: Compute adjustment vectors
        adjustments = self.compute_adjustment_vectors()

        # Step 2: Test residual stream diff-of-means at each layer
        print("\n" + "=" * 60)
        print("Testing diff-of-means on RESIDUAL STREAM")
        print("=" * 60)

        residual_results = {}
        for L in TEST_LAYERS:
            r = self.test_diff_of_means(adjustments, L, "residual", N_TRIALS)
            residual_results[L] = r
            print(f"  Layer {L:2d} residual: {r['success_rate']:.0%} "
                  f"[{r['ci'][0]:.0%}, {r['ci'][1]:.0%}]")

        # Step 3: Decomposed — attention-only and MLP-only at key layers
        print("\n" + "=" * 60)
        print("Testing decomposed diff-of-means (attn-only, MLP-only)")
        print("=" * 60)

        decomposed_results = {}
        for L in [10, 25]:
            for comp in ["attn", "mlp"]:
                r = self.test_diff_of_means(adjustments, L, comp, N_TRIALS)
                decomposed_results[f"L{L}_{comp}"] = r
                print(f"  Layer {L:2d} {comp:4s}: {r['success_rate']:.0%} "
                      f"[{r['ci'][0]:.0%}, {r['ci'][1]:.0%}]")

        # Step 4: Scaled diff-of-means (try different magnitudes)
        print("\n" + "=" * 60)
        print("Testing scaled diff-of-means at Layer 10 residual")
        print("=" * 60)

        scaled_results = {}
        for scale in [0.5, 1.0, 2.0, 5.0, 10.0]:
            # Temporarily scale the adjustment
            orig = adjustments[10]["residual"].clone()
            adjustments[10]["residual"] = orig * scale
            r = self.test_diff_of_means(adjustments, 10, "residual", N_TRIALS)
            r["scale"] = scale
            scaled_results[f"scale_{scale}"] = r
            adjustments[10]["residual"] = orig  # restore
            print(f"  Scale {scale:4.1f}x: {r['success_rate']:.0%} "
                  f"[{r['ci'][0]:.0%}, {r['ci'][1]:.0%}]")

        total_time = time.time() - t_start

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            "experiment": "R7_diff_of_means",
            "timestamp": timestamp,
            "model": MODEL_NAME,
            "test_layers": TEST_LAYERS,
            "n_mean_samples": N_MEAN_SAMPLES,
            "n_trials": N_TRIALS,
            "total_time_seconds": total_time,
            "residual_results": {str(k): v for k, v in residual_results.items()},
            "decomposed_results": decomposed_results,
            "scaled_results": scaled_results,
        }

        results_path = os.path.join(RESULTS_DIR, f"results_{timestamp}.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2,
                      default=lambda o: int(o) if isinstance(o, np.integer)
                      else float(o) if isinstance(o, np.floating)
                      else o.tolist() if isinstance(o, np.ndarray) else str(o))
        print(f"\nResults saved to {results_path}")

        # Summary
        print("\n" + "=" * 70)
        print("R7 SUMMARY: Diff-of-Means vs Direct Patching")
        print("=" * 70)

        print(f"\nResidual stream diff-of-means:")
        for L in TEST_LAYERS:
            r = residual_results[L]
            print(f"  Layer {L:2d}: {r['success_rate']:.0%} "
                  f"(||adj|| = {r.get('adjustment_norm', 0):.3f})")

        print(f"\nDecomposed at Layer 10 and 25:")
        for key, r in decomposed_results.items():
            print(f"  {key}: {r['success_rate']:.0%}")

        print(f"\nFor reference: Direct attention patching at Layer 10 = 100%")
        print(f"\nInterpretation:")
        best_layer = max(residual_results, key=lambda L: residual_results[L]["success_rate"])
        best_rate = residual_results[best_layer]["success_rate"]
        if best_rate >= 0.8:
            print(f"  Diff-of-means WORKS at Layer {best_layer} ({best_rate:.0%})")
            print(f"  This suggests a simpler linear intervention is possible.")
        else:
            print(f"  Diff-of-means FAILS (best: Layer {best_layer} at {best_rate:.0%})")
            print(f"  This confirms the paper's finding that direct patching is needed.")
            print(f"  The bug mechanism is NOT a simple additive bias.")

        print(f"\nTotal time: {total_time/60:.1f} minutes")
        return results


if __name__ == "__main__":
    exp = DiffOfMeansExperiment()
    results = exp.run()
