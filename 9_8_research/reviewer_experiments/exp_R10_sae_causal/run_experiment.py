#!/usr/bin/env python3
"""
Experiment R10: SAE Causal Validation
======================================
Validates the attention patching mechanism at the SAE feature level.
Uses Llama-Scope residual stream SAEs to:

1. Identify which SAE features at Layer 10 differ between Simple and Q&A formats
2. Causally test: does clamping Q&A features to Simple feature values fix the bug?
3. Ablation: which individual features are necessary/sufficient for the fix?
4. Compare feature-level vs attention-level intervention

Reuses SAE loading from experimental/sae/working_sae_loader.py
Reuses patching infrastructure from other R-experiments.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import time
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
N_LAYERS = 32

CORRECT_PROMPT = "Which is bigger: 9.8 or 9.11?\nAnswer:"
BUGGY_PROMPT = "Q: Which is bigger: 9.8 or 9.11?\nA:"


class SAECausalValidator:
    def __init__(self):
        print(f"Loading {MODEL_NAME} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
        )
        self.model.eval()

        # Load residual stream SAE for Layer 10
        print("Loading Layer 10 residual stream SAE...")
        self.sae = SAE.from_pretrained(
            release="llama_scope_lxr_8x",
            sae_id="l10r_8x",
            device=DEVICE
        )
        print(f"  SAE: d_in={self.sae.cfg.d_in}, d_sae={self.sae.cfg.d_sae}")
        print("Model and SAE loaded.")

    def generate(self, prompt, max_new=30):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                      pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)

    def check_answer(self, text):
        """Check if answer says 9.8 is bigger (correct) or 9.11 (bug)."""
        lo = text.lower()
        correct = any(f"9.8 is {w}" in lo or f"9.8 is the {w}" in lo
                      for w in ["bigger", "larger", "greater"])
        bug = any(f"9.11 is {w}" in lo or f"9.11 is the {w}" in lo
                  for w in ["bigger", "larger", "greater"])
        if not correct and not bug:
            first = text.strip().split("\n")[0]
            if "9.8" in first and "9.11" not in first:
                correct = True
            elif "9.11" in first and "9.8" not in first:
                bug = True
        if correct and not bug:
            return "correct"
        elif bug and not correct:
            return "bug"
        return "unclear"

    def get_layer10_residual(self, prompt: str) -> torch.Tensor:
        """Get residual stream output after Layer 10."""
        saved = {}

        def hook_fn(module, inp, out):
            if isinstance(out, tuple):
                saved['residual'] = out[0].detach()
            else:
                saved['residual'] = out.detach()

        hook = self.model.model.layers[10].register_forward_hook(hook_fn)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            self.model(**inputs)
        hook.remove()
        return saved['residual']  # [1, seq_len, 4096]

    def encode_to_features(self, residual: torch.Tensor) -> torch.Tensor:
        """Encode residual stream activations to SAE features."""
        # residual: [1, seq_len, 4096]
        # SAE expects [*, d_in]
        batch_seq = residual.shape[:-1]
        flat = residual.reshape(-1, residual.shape[-1]).float()
        with torch.no_grad():
            features = self.sae.encode(flat)  # [batch*seq, d_sae]
        return features.reshape(*batch_seq, -1)  # [1, seq_len, d_sae]

    def decode_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """Decode SAE features back to residual stream."""
        batch_seq = features.shape[:-1]
        flat = features.reshape(-1, features.shape[-1])
        with torch.no_grad():
            residual = self.sae.decode(flat)  # [batch*seq, d_in]
        return residual.reshape(*batch_seq, -1).half()

    def section1_feature_identification(self) -> Dict:
        """Identify SAE features that differ between formats."""
        print("\n" + "=" * 60)
        print("SECTION 1: Feature Identification")
        print("=" * 60)

        # Get residual activations for both formats
        res_simple = self.get_layer10_residual(CORRECT_PROMPT)
        res_qa = self.get_layer10_residual(BUGGY_PROMPT)

        # Encode to SAE features
        feat_simple = self.encode_to_features(res_simple)  # [1, seq, 32768]
        feat_qa = self.encode_to_features(res_qa)

        # Focus on final token position (where answer is generated)
        feat_simple_final = feat_simple[0, -1, :]  # [32768]
        feat_qa_final = feat_qa[0, -1, :]

        # Find active features
        threshold = 0.01
        active_simple = (feat_simple_final > threshold)
        active_qa = (feat_qa_final > threshold)

        n_active_simple = active_simple.sum().item()
        n_active_qa = active_qa.sum().item()

        # Feature differences
        diff = feat_simple_final - feat_qa_final
        abs_diff = diff.abs()

        # Top features by absolute difference
        top_k = 50
        top_indices = abs_diff.topk(top_k).indices.cpu().numpy()
        top_diffs = diff[top_indices].cpu().numpy()
        simple_vals = feat_simple_final[top_indices].cpu().numpy()
        qa_vals = feat_qa_final[top_indices].cpu().numpy()

        # Categorize features
        simple_dominant = []  # Higher in simple (correct) format
        qa_dominant = []      # Higher in Q&A (buggy) format

        for i, idx in enumerate(top_indices):
            if top_diffs[i] > 0.1:
                simple_dominant.append(int(idx))
            elif top_diffs[i] < -0.1:
                qa_dominant.append(int(idx))

        # Shared features
        shared = (active_simple & active_qa).sum().item()
        unique_simple = (active_simple & ~active_qa).sum().item()
        unique_qa = (~active_simple & active_qa).sum().item()
        total_active = (active_simple | active_qa).sum().item()
        overlap_pct = shared / total_active * 100 if total_active > 0 else 0

        print(f"\n  Active features (final token):")
        print(f"    Simple format:  {n_active_simple}")
        print(f"    Q&A format:     {n_active_qa}")
        print(f"    Shared:         {shared} ({overlap_pct:.1f}%)")
        print(f"    Unique simple:  {unique_simple}")
        print(f"    Unique Q&A:     {unique_qa}")
        print(f"\n  Top differential features:")
        print(f"    Simple-dominant: {len(simple_dominant)} features")
        print(f"    Q&A-dominant:    {len(qa_dominant)} features")
        print(f"\n  Top 10 by |diff|:")
        for i in range(min(10, len(top_indices))):
            idx = top_indices[i]
            print(f"    Feature {idx:5d}: simple={simple_vals[i]:+.3f}, "
                  f"qa={qa_vals[i]:+.3f}, diff={top_diffs[i]:+.3f}")

        results = {
            "n_active_simple": n_active_simple,
            "n_active_qa": n_active_qa,
            "shared": shared,
            "unique_simple": unique_simple,
            "unique_qa": unique_qa,
            "overlap_pct": overlap_pct,
            "simple_dominant_features": simple_dominant,
            "qa_dominant_features": qa_dominant,
            "top_features": [
                {"index": int(top_indices[i]),
                 "simple_val": float(simple_vals[i]),
                 "qa_val": float(qa_vals[i]),
                 "diff": float(top_diffs[i])}
                for i in range(len(top_indices))
            ],
        }
        return results, feat_simple_final, feat_qa_final

    def section2_full_feature_swap(self, n_trials=50) -> Dict:
        """Causal test: replace ALL Layer 10 SAE features from Simple→Q&A."""
        print("\n" + "=" * 60)
        print("SECTION 2: Full SAE Feature Swap (Simple→Q&A)")
        print("=" * 60)

        # Capture simple format residual and encode to features
        res_simple = self.get_layer10_residual(CORRECT_PROMPT)
        feat_simple = self.encode_to_features(res_simple)  # [1, seq, 32768]

        # Decode simple features back to residual (SAE reconstruction)
        res_simple_reconstructed = self.decode_from_features(feat_simple)

        # Test 1: Full feature replacement (decode simple features, inject into Q&A)
        print("\n  Test 1: Replace L10 residual with SAE-reconstructed simple activations")
        successes = 0
        for trial in range(n_trials):
            def patch_fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                new = h.clone()
                # Replace with SAE-reconstructed simple activations
                ml = min(h.shape[1], res_simple_reconstructed.shape[1])
                new[:, :ml, :] = res_simple_reconstructed[:, :ml, :].to(h.device)
                return (new,) + out[1:] if isinstance(out, tuple) else new

            hook = self.model.model.layers[10].register_forward_hook(patch_fn)
            try:
                out = self.generate(BUGGY_PROMPT)
                if self.check_answer(out) == "correct":
                    successes += 1
            finally:
                hook.remove()

        rate1 = successes / n_trials
        print(f"    Success rate: {successes}/{n_trials} = {rate1:.0%}")

        # Test 2: Direct residual swap (no SAE, for comparison)
        print("\n  Test 2: Direct residual swap (no SAE, baseline comparison)")
        successes2 = 0
        for trial in range(n_trials):
            def patch_fn2(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                new = h.clone()
                ml = min(h.shape[1], res_simple.shape[1])
                new[:, :ml, :] = res_simple[:, :ml, :].to(h.device)
                return (new,) + out[1:] if isinstance(out, tuple) else new

            hook = self.model.model.layers[10].register_forward_hook(patch_fn2)
            try:
                out = self.generate(BUGGY_PROMPT)
                if self.check_answer(out) == "correct":
                    successes2 += 1
            finally:
                hook.remove()

        rate2 = successes2 / n_trials
        print(f"    Success rate: {successes2}/{n_trials} = {rate2:.0%}")

        # Compute reconstruction error
        recon_error = (res_simple - res_simple_reconstructed.to(res_simple.device)).norm().item()
        print(f"\n  SAE reconstruction error (L2): {recon_error:.4f}")

        return {
            "sae_reconstructed_swap": {"success_rate": rate1, "successes": successes, "n_trials": n_trials},
            "direct_swap": {"success_rate": rate2, "successes": successes2, "n_trials": n_trials},
            "reconstruction_error_l2": recon_error,
        }

    def section3_feature_clamping(self, simple_features, qa_features,
                                    simple_dominant, qa_dominant, n_trials=50) -> Dict:
        """Causal test: clamp specific feature subsets from Simple→Q&A."""
        print("\n" + "=" * 60)
        print("SECTION 3: Selective Feature Clamping")
        print("=" * 60)

        results = {}

        # Get full residual for Q&A
        res_qa = self.get_layer10_residual(BUGGY_PROMPT)
        feat_qa_full = self.encode_to_features(res_qa)  # [1, seq, 32768]
        feat_simple_full_tensor = self.encode_to_features(
            self.get_layer10_residual(CORRECT_PROMPT))

        conditions = [
            ("simple_dominant_only", simple_dominant,
             "Clamp simple-dominant features to simple values"),
            ("qa_dominant_only", qa_dominant,
             "Zero out Q&A-dominant features"),
            ("top_10_diff", list(simple_features.topk(10).indices.cpu().numpy()),
             "Clamp top 10 simple features"),
            ("top_20_diff", list(simple_features.topk(20).indices.cpu().numpy()),
             "Clamp top 20 simple features"),
            ("top_50_diff", list(simple_features.topk(50).indices.cpu().numpy()),
             "Clamp top 50 simple features"),
        ]

        for name, feature_indices, desc in conditions:
            print(f"\n  {desc} ({len(feature_indices)} features):")

            if len(feature_indices) == 0:
                print(f"    Skipped (no features)")
                results[name] = {"success_rate": 0, "skipped": True, "n_features": 0}
                continue

            successes = 0
            for trial in range(n_trials):
                # For each trial: modify the Q&A features at specified indices
                def make_patch(feat_indices, simple_feat_full, qa_feat_full):
                    def patch_fn(module, inp, out):
                        h = out[0] if isinstance(out, tuple) else out
                        # Encode current activations
                        flat = h.reshape(-1, h.shape[-1]).float()
                        with torch.no_grad():
                            features = self.sae.encode(flat)

                        # Clamp specific features to simple format values
                        for idx in feat_indices:
                            if name == "qa_dominant_only":
                                # Zero out Q&A-dominant features
                                features[:, idx] = 0
                            else:
                                # Replace with simple format feature values
                                ml = min(features.shape[0], simple_feat_full.shape[1])
                                features[:ml, idx] = simple_feat_full[0, :ml, idx]

                        # Decode back
                        with torch.no_grad():
                            new_h = self.sae.decode(features)
                        new_h = new_h.reshape(h.shape).half()
                        return (new_h,) + out[1:] if isinstance(out, tuple) else new_h
                    return patch_fn

                hook = self.model.model.layers[10].register_forward_hook(
                    make_patch(feature_indices, feat_simple_full_tensor, feat_qa_full))
                try:
                    out = self.generate(BUGGY_PROMPT)
                    if self.check_answer(out) == "correct":
                        successes += 1
                finally:
                    hook.remove()

            rate = successes / n_trials
            print(f"    Success rate: {successes}/{n_trials} = {rate:.0%}")
            results[name] = {
                "success_rate": rate,
                "successes": successes,
                "n_trials": n_trials,
                "n_features": len(feature_indices),
                "feature_indices": feature_indices[:20],  # Save first 20 for reference
            }

        return results

    def section4_feature_necessity(self, simple_features, qa_features, n_trials=50) -> Dict:
        """Test which feature groups are necessary: ablate from full swap."""
        print("\n" + "=" * 60)
        print("SECTION 4: Feature Necessity (Ablation from Full Swap)")
        print("=" * 60)

        # Get diff between formats
        diff = simple_features - qa_features
        abs_diff = diff.abs()

        # Sort features by importance (largest absolute difference)
        sorted_indices = abs_diff.argsort(descending=True).cpu().numpy()

        # Get full simple residual and features
        res_simple = self.get_layer10_residual(CORRECT_PROMPT)
        feat_simple_full = self.encode_to_features(res_simple)

        # For each ablation: do full swap BUT keep N features at Q&A values
        results = {}
        keep_qa_counts = [0, 5, 10, 20, 50, 100, 200, 500]

        for n_keep in keep_qa_counts:
            keep_at_qa = sorted_indices[:n_keep] if n_keep > 0 else []
            desc = f"Full swap, keep top {n_keep} features at Q&A values"
            print(f"\n  {desc}:")

            successes = 0
            for trial in range(n_trials):
                def make_patch(keep_indices, simple_feat):
                    def patch_fn(module, inp, out):
                        h = out[0] if isinstance(out, tuple) else out

                        # Start with simple features
                        flat = h.reshape(-1, h.shape[-1]).float()
                        with torch.no_grad():
                            current_features = self.sae.encode(flat)

                        # Get simple features
                        ml = min(flat.shape[0], simple_feat.shape[1])
                        new_features = simple_feat[0, :ml, :].clone()

                        # Keep specified features at their Q&A (current) values
                        for idx in keep_indices:
                            new_features[:, idx] = current_features[:ml, idx]

                        # Pad if needed
                        if flat.shape[0] > ml:
                            padded = current_features.clone()
                            padded[:ml, :] = new_features
                            new_features = padded

                        with torch.no_grad():
                            new_h = self.sae.decode(new_features)
                        new_h = new_h.reshape(h.shape).half()
                        return (new_h,) + out[1:] if isinstance(out, tuple) else new_h
                    return patch_fn

                hook = self.model.model.layers[10].register_forward_hook(
                    make_patch(keep_at_qa, feat_simple_full))
                try:
                    out = self.generate(BUGGY_PROMPT)
                    if self.check_answer(out) == "correct":
                        successes += 1
                finally:
                    hook.remove()

            rate = successes / n_trials
            print(f"    Success rate: {successes}/{n_trials} = {rate:.0%}")
            results[f"keep_{n_keep}"] = {
                "success_rate": rate,
                "successes": successes,
                "n_trials": n_trials,
                "n_features_kept_at_qa": n_keep,
            }

        return results

    def section5_multi_layer_sae(self, n_trials=50) -> Dict:
        """Test SAE feature swap at other layers for comparison."""
        print("\n" + "=" * 60)
        print("SECTION 5: SAE Feature Swap at Multiple Layers")
        print("=" * 60)

        test_layers = [8, 9, 10, 11, 12, 25]
        results = {}

        for layer in test_layers:
            print(f"\n  Layer {layer}:")

            # Load SAE for this layer
            try:
                sae = SAE.from_pretrained(
                    release="llama_scope_lxr_8x",
                    sae_id=f"l{layer}r_8x",
                    device=DEVICE
                )
            except Exception as e:
                print(f"    Failed to load SAE: {e}")
                results[layer] = {"error": str(e)}
                continue

            # Get simple format residual at this layer
            saved = {}
            def make_capture(l):
                def hook_fn(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    saved['residual'] = h.detach()
                return hook_fn

            hook = self.model.model.layers[layer].register_forward_hook(make_capture(layer))
            inputs = self.tokenizer(CORRECT_PROMPT, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                self.model(**inputs)
            hook.remove()
            res_simple = saved['residual']

            # Encode and decode through SAE
            flat = res_simple.reshape(-1, res_simple.shape[-1]).float()
            with torch.no_grad():
                features_simple = sae.encode(flat)
                reconstructed = sae.decode(features_simple).reshape(res_simple.shape).half()

            # Test: inject SAE-reconstructed simple activations at this layer during Q&A
            successes = 0
            for trial in range(n_trials):
                def make_patch(recon, l):
                    def patch_fn(module, inp, out):
                        h = out[0] if isinstance(out, tuple) else out
                        new = h.clone()
                        ml = min(h.shape[1], recon.shape[1])
                        new[:, :ml, :] = recon[:, :ml, :].to(h.device)
                        return (new,) + out[1:] if isinstance(out, tuple) else new
                    return patch_fn

                hook = self.model.model.layers[layer].register_forward_hook(
                    make_patch(reconstructed, layer))
                try:
                    out = self.generate(BUGGY_PROMPT)
                    if self.check_answer(out) == "correct":
                        successes += 1
                finally:
                    hook.remove()

            rate = successes / n_trials
            recon_error = (res_simple - reconstructed.to(res_simple.device)).norm().item()
            print(f"    SAE swap success: {successes}/{n_trials} = {rate:.0%}")
            print(f"    Reconstruction error: {recon_error:.4f}")

            results[layer] = {
                "success_rate": rate,
                "successes": successes,
                "n_trials": n_trials,
                "reconstruction_error": recon_error,
            }

            # Clean up SAE if not Layer 10
            if layer != 10:
                del sae
                torch.cuda.empty_cache()

        return results


def main():
    exp = SAECausalValidator()
    t_start = time.time()

    # Section 1: Feature identification
    s1, simple_features, qa_features = exp.section1_feature_identification()

    # Section 2: Full feature swap
    s2 = exp.section2_full_feature_swap(n_trials=50)

    # Section 3: Selective feature clamping
    s3 = exp.section3_feature_clamping(
        simple_features, qa_features,
        s1["simple_dominant_features"], s1["qa_dominant_features"],
        n_trials=50)

    # Section 4: Feature necessity
    s4 = exp.section4_feature_necessity(simple_features, qa_features, n_trials=50)

    # Section 5: Multi-layer comparison
    s5 = exp.section5_multi_layer_sae(n_trials=50)

    total_time = time.time() - t_start

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "experiment": "R10_sae_causal",
        "timestamp": timestamp,
        "model": MODEL_NAME,
        "sae_release": "llama_scope_lxr_8x",
        "sae_layer": 10,
        "total_time_seconds": total_time,
        "section1_feature_identification": s1,
        "section2_full_feature_swap": s2,
        "section3_selective_clamping": s3,
        "section4_feature_necessity": s4,
        "section5_multi_layer": {str(k): v for k, v in s5.items()},
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
    print("R10 SUMMARY: SAE Causal Validation")
    print("=" * 70)
    print(f"\n  Section 1: {s1['overlap_pct']:.1f}% feature overlap, "
          f"{len(s1['simple_dominant_features'])} simple-dominant, "
          f"{len(s1['qa_dominant_features'])} qa-dominant")
    print(f"  Section 2: SAE swap={s2['sae_reconstructed_swap']['success_rate']:.0%}, "
          f"Direct swap={s2['direct_swap']['success_rate']:.0%}")
    print(f"  Section 3 (selective clamping):")
    for name, r in s3.items():
        if not r.get("skipped"):
            print(f"    {name}: {r['success_rate']:.0%} ({r['n_features']} features)")
    print(f"  Section 4 (necessity):")
    for name, r in s4.items():
        print(f"    {name}: {r['success_rate']:.0%}")
    print(f"  Section 5 (multi-layer):")
    for layer, r in s5.items():
        if "success_rate" in r:
            print(f"    Layer {layer}: {r['success_rate']:.0%}")
    print(f"\n  Total time: {total_time/60:.1f} minutes")

    return results


if __name__ == "__main__":
    results = main()
