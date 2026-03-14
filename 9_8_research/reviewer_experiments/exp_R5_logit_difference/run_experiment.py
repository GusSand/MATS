#!/usr/bin/env python3
"""
Experiment R5: Logit Difference Analysis for Figure 1
=====================================================
Addresses Reviewer joQU's critique: "The authors did not analyze when the
'.8' and '.11' tokens are boosted at the '9' token position, either
absolutely or in logit difference ('.8' vs '.11')."

Components:
  1. Layer-by-layer logit difference tracking at final token position
  2. Full heatmap: (layer × token_position) for logit(".8") - logit(".11")
  3. Multi-token tracking ("9", ".", "8" sequence)
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List
import json
from datetime import datetime
import time
import os

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CORRECT_PROMPT = "Which is bigger: 9.8 or 9.11?\nAnswer:"
BUGGY_PROMPT = "Q: Which is bigger: 9.8 or 9.11?\nA:"
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
N_LAYERS = 32


class LogitDifferenceAnalyzer:
    def __init__(self):
        print(f"Loading {MODEL_NAME} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
        )
        self.model.eval()

        # Get unembedding and norm
        self.lm_head = self.model.lm_head.weight  # [vocab, hidden]
        self.norm = self.model.model.norm

        # Find relevant token IDs
        # Key insight: "9.8" tokenizes as [9, ., 8] and "9.11" as [9, ., 11]
        # The critical divergence is at token "8" (id=23) vs "11" (id=806)
        self.token_map = {}
        for s in ["9", "8", "11", "."]:
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            self.token_map[s] = ids
        # The key token IDs we track:
        self.id_8 = self.tokenizer.encode("8", add_special_tokens=False)[0]   # 23
        self.id_11 = self.tokenizer.encode("11", add_special_tokens=False)[0]  # 806
        self.id_9 = self.tokenizer.encode("9", add_special_tokens=False)[0]   # 24
        self.id_dot = self.tokenizer.encode(".", add_special_tokens=False)[0]  # 13
        print(f"  Token IDs: 8={self.id_8}, 11={self.id_11}, 9={self.id_9}, .={self.id_dot}")
        print(f"  Token map: {self.token_map}")

    def get_all_hidden_states(self, prompt: str) -> Dict:
        """Run forward pass and capture hidden states at every layer."""
        hidden_states = {}
        hooks = []

        def make_hook(layer_idx):
            def fn(module, inp, out):
                # Capture the layer's output (residual stream after this layer)
                if isinstance(out, tuple):
                    hidden_states[layer_idx] = out[0].detach()
                else:
                    hidden_states[layer_idx] = out.detach()
            return fn

        for L in range(N_LAYERS):
            h = self.model.model.layers[L].register_forward_hook(make_hook(L))
            hooks.append(h)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        input_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        with torch.no_grad():
            _ = self.model(**inputs)

        for h in hooks:
            h.remove()

        return hidden_states, input_tokens

    def logit_lens(self, hidden: torch.Tensor, token_ids: List[int]) -> Dict[int, float]:
        """Apply logit lens: norm → unembedding → extract specific token logits."""
        with torch.no_grad():
            normed = self.norm(hidden.float()).half()
            logits = normed @ self.lm_head.T  # [1, seq, vocab] or [seq, vocab]
            if logits.dim() == 3:
                logits = logits[0]  # [seq, vocab]
        return {tid: logits[:, tid].cpu().numpy() for tid in token_ids}

    def section1_layer_by_layer(self) -> Dict:
        """Logit difference at final token, every layer, both formats."""
        print("\n" + "=" * 60)
        print("SECTION 1: Layer-by-layer logit difference at final token")
        print("=" * 60)

        # Key token IDs: "8" (id=23) vs "11" (id=806) — the diverging token after "9."
        all_ids = [self.id_8, self.id_11, self.id_9, self.id_dot]

        results = {}
        for fmt_name, prompt in [("simple", CORRECT_PROMPT), ("qa", BUGGY_PROMPT)]:
            print(f"\n  Format: {fmt_name}")
            hidden_states, input_tokens = self.get_all_hidden_states(prompt)

            layer_data = {}
            for L in range(N_LAYERS):
                h = hidden_states[L]  # [1, seq, hidden]
                token_logits = self.logit_lens(h, all_ids)

                # At final position
                final_pos = -1
                logit_vals = {}
                for tid in all_ids:
                    logit_vals[tid] = float(token_logits[tid][final_pos])

                # Compute differences
                layer_data[L] = {
                    "logits_at_final": logit_vals,
                }

                # logit("8") - logit("11") — the key divergence
                diff = logit_vals.get(self.id_8, 0) - logit_vals.get(self.id_11, 0)
                layer_data[L]["logit_diff_8_minus_11"] = diff
                layer_data[L]["logit_8"] = logit_vals.get(self.id_8, 0)
                layer_data[L]["logit_11"] = logit_vals.get(self.id_11, 0)
                layer_data[L]["logit_9"] = logit_vals.get(self.id_9, 0)

            results[fmt_name] = {
                "layer_data": {str(k): v for k, v in layer_data.items()},
                "input_tokens": input_tokens,
                "n_input_tokens": len(input_tokens),
            }

            # Print summary
            print(f"    Layer | logit(.8) - logit(.11) | logit(9)")
            print(f"    ------+------------------------+---------")
            for L in [0, 5, 8, 10, 15, 20, 25, 28, 31]:
                d = layer_data[L]
                diff = d.get("logit_diff_8_minus_11", "N/A")
                l9 = d.get("logit_9", "N/A")
                if isinstance(diff, float):
                    print(f"      {L:2d}  |  {diff:+8.3f}               |  {l9:+8.3f}")

        return results

    def section2_heatmap(self) -> Dict:
        """Full (layer × position) heatmap of logit(.8) - logit(.11)."""
        print("\n" + "=" * 60)
        print("SECTION 2: Full heatmap (layer × position)")
        print("=" * 60)

        id_8 = self.id_8
        id_11 = self.id_11

        results = {}
        for fmt_name, prompt in [("simple", CORRECT_PROMPT), ("qa", BUGGY_PROMPT)]:
            print(f"\n  Computing heatmap for {fmt_name} format...")
            hidden_states, input_tokens = self.get_all_hidden_states(prompt)

            heatmap = np.zeros((N_LAYERS, len(input_tokens)))
            for L in range(N_LAYERS):
                h = hidden_states[L]
                token_logits = self.logit_lens(h, [id_8, id_11])
                diff = token_logits[id_8] - token_logits[id_11]
                heatmap[L, :] = diff[:len(input_tokens)]

            results[fmt_name] = {
                "heatmap": heatmap.tolist(),
                "input_tokens": input_tokens,
                "shape": list(heatmap.shape),
            }
            print(f"    Heatmap shape: {heatmap.shape}")
            print(f"    At final token - min layer diff: {heatmap[:, -1].min():.3f} "
                  f"(layer {heatmap[:, -1].argmin()})")
            print(f"    At final token - max layer diff: {heatmap[:, -1].max():.3f} "
                  f"(layer {heatmap[:, -1].argmax()})")

        return results


def main():
    analyzer = LogitDifferenceAnalyzer()
    t_start = time.time()

    s1 = analyzer.section1_layer_by_layer()
    s2 = analyzer.section2_heatmap()

    total_time = time.time() - t_start

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "experiment": "R5_logit_difference",
        "timestamp": timestamp,
        "model": MODEL_NAME,
        "token_map": {k: v for k, v in analyzer.token_map.items()},
        "total_time_seconds": total_time,
        "section1_layer_by_layer": s1,
        "section2_heatmap": s2,
    }

    results_path = os.path.join(RESULTS_DIR, f"results_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda o: int(o) if isinstance(o, np.integer)
                  else float(o) if isinstance(o, np.floating)
                  else o.tolist() if isinstance(o, np.ndarray) else str(o))
    print(f"\nResults saved to {results_path}")

    # Print divergence analysis
    print("\n" + "=" * 70)
    print("KEY FINDING: Where do .8 and .11 logits diverge between formats?")
    print("=" * 70)

    for L in range(N_LAYERS):
        simple_diff = s1["simple"]["layer_data"][str(L)].get("logit_diff_8_minus_11")
        qa_diff = s1["qa"]["layer_data"][str(L)].get("logit_diff_8_minus_11")
        if simple_diff is not None and qa_diff is not None:
            format_diff = simple_diff - qa_diff
            marker = " ***" if abs(format_diff) > 1.0 else ""
            if L % 5 == 0 or abs(format_diff) > 1.0:
                print(f"  Layer {L:2d}: Simple={simple_diff:+.3f}, "
                      f"Q&A={qa_diff:+.3f}, gap={format_diff:+.3f}{marker}")

    print(f"\nTotal time: {total_time/60:.1f} minutes")
    return results


if __name__ == "__main__":
    results = main()
