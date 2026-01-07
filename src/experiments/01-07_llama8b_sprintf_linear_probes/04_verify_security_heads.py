#!/usr/bin/env python3
"""
Verify Security Context Heads - Causal Testing

Three tests to verify if candidate heads are actually "Security Context Heads":
1. ABLATION: Zero out heads → does P(snprintf) drop?
2. PATH PATCHING: Patch only these heads from secure→neutral → does behavior transfer?
3. OUTPUT ANALYSIS: What vectors do these heads write to the residual stream?
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Prompts
SECURE_PROMPT = '''// WARNING: Use snprintf to prevent buffer overflows
int list_item_to_string(node *head, char *str, size_t str_size) {
    if (head == NULL) return EXIT_FAILURE;
    '''

NEUTRAL_PROMPT = '''int list_item_to_string(node *head, char *str, size_t str_size) {
    if (head == NULL) return EXIT_FAILURE;
    '''

# Candidate heads to test (from attention analysis)
CANDIDATE_HEADS = [
    (20, 24),  # L20H24 - 61.1% attention to security
    (25, 13),  # L25H13 - 47.7%
    (17, 29),  # L17H29 - 44.0%
    (16, 8),   # L16H8 - 42.2%
    (22, 14),  # L22H14 - 40.8%
    (24, 27),  # L24H27 - 40.4%
    (17, 25),  # L17H25 - 40.2%
    (0, 11),   # L0H11 - 35.3% (early layer)
]


class HeadVerifier:
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        print("Loading model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()

        self.n_layers = self.model.config.num_hidden_layers
        self.n_heads = self.model.config.num_attention_heads
        self.head_dim = self.model.config.hidden_size // self.n_heads
        self.hidden_size = self.model.config.hidden_size

        print(f"Model loaded: {self.n_layers} layers, {self.n_heads} heads, head_dim={self.head_dim}")

        # Token IDs for measurement
        self.snprintf_token = self.tokenizer.encode(" snprintf", add_special_tokens=False)[0]
        self.sprintf_token = self.tokenizer.encode(" sprintf", add_special_tokens=False)[0]

        self.hooks = []
        self.saved_activations = {}

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.saved_activations = {}

    def get_snprintf_prob(self, prompt: str) -> float:
        """Get probability of snprintf token."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
        return probs[self.snprintf_token].item()

    # =========================================================================
    # TEST 1: ABLATION
    # =========================================================================

    def ablate_head_hook(self, layer_idx: int, head_idx: int):
        """Hook to zero out a specific head's output."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output

            batch, seq, hidden = h.shape

            # Reshape to [batch, seq, n_heads, head_dim]
            h_reshaped = h.view(batch, seq, self.n_heads, self.head_dim)

            # Zero out the specific head
            h_reshaped = h_reshaped.clone()
            h_reshaped[:, :, head_idx, :] = 0.0

            new_h = h_reshaped.view(batch, seq, hidden)

            if isinstance(output, tuple):
                return (new_h,) + output[1:]
            return new_h
        return hook_fn

    def test_ablation(self, heads_to_ablate: list) -> dict:
        """
        Test 1: Ablate specific heads and measure impact on P(snprintf).
        If these are security context heads, ablating them should REDUCE P(snprintf).
        """
        print("\n" + "="*70)
        print("TEST 1: ABLATION")
        print("="*70)
        print("If heads are causal, ablating them should REDUCE P(snprintf) on secure prompt")

        # Baseline
        baseline_secure = self.get_snprintf_prob(SECURE_PROMPT)
        baseline_neutral = self.get_snprintf_prob(NEUTRAL_PROMPT)
        print(f"\nBaselines:")
        print(f"  Secure prompt:  P(snprintf) = {baseline_secure*100:.2f}%")
        print(f"  Neutral prompt: P(snprintf) = {baseline_neutral*100:.2f}%")

        results = {
            'baseline_secure': baseline_secure,
            'baseline_neutral': baseline_neutral,
            'ablations': {}
        }

        print(f"\nAblating individual heads on SECURE prompt:")
        print("-" * 60)

        for layer_idx, head_idx in heads_to_ablate:
            head_key = f"L{layer_idx}H{head_idx}"

            # Register ablation hook
            attn = self.model.model.layers[layer_idx].self_attn
            hook = attn.register_forward_hook(self.ablate_head_hook(layer_idx, head_idx))
            self.hooks.append(hook)

            # Measure
            ablated_prob = self.get_snprintf_prob(SECURE_PROMPT)
            self.clear_hooks()

            drop = baseline_secure - ablated_prob
            drop_pct = (drop / baseline_secure) * 100 if baseline_secure > 0 else 0

            results['ablations'][head_key] = {
                'layer': layer_idx,
                'head': head_idx,
                'ablated_prob': ablated_prob,
                'drop': drop,
                'drop_pct': drop_pct
            }

            # Impact indicator
            if drop_pct > 10:
                symbol = "🔴 SIGNIFICANT"
            elif drop_pct > 5:
                symbol = "🟡 MODERATE"
            elif drop_pct > 0:
                symbol = "🟢 SMALL"
            else:
                symbol = "⚪ NONE"

            print(f"  {head_key}: P(snprintf) = {ablated_prob*100:.2f}% (drop: {drop_pct:+.1f}%) {symbol}")

        # Test ablating ALL candidate heads together
        print(f"\nAblating ALL {len(heads_to_ablate)} candidate heads together:")
        for layer_idx, head_idx in heads_to_ablate:
            attn = self.model.model.layers[layer_idx].self_attn
            hook = attn.register_forward_hook(self.ablate_head_hook(layer_idx, head_idx))
            self.hooks.append(hook)

        all_ablated_prob = self.get_snprintf_prob(SECURE_PROMPT)
        self.clear_hooks()

        total_drop = baseline_secure - all_ablated_prob
        total_drop_pct = (total_drop / baseline_secure) * 100 if baseline_secure > 0 else 0

        results['all_ablated'] = {
            'prob': all_ablated_prob,
            'drop': total_drop,
            'drop_pct': total_drop_pct
        }

        print(f"  ALL HEADS: P(snprintf) = {all_ablated_prob*100:.2f}% (drop: {total_drop_pct:+.1f}%)")

        return results

    # =========================================================================
    # TEST 2: PATH PATCHING
    # =========================================================================

    def save_head_hook(self, layer_idx: int, head_idx: int, key: str):
        """Hook to save a specific head's output."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output

            batch, seq, hidden = h.shape
            h_reshaped = h.view(batch, seq, self.n_heads, self.head_dim)

            # Save only the specific head, last position
            self.saved_activations[key] = h_reshaped[:, -1:, head_idx, :].detach().clone()
            return output
        return hook_fn

    def patch_head_hook(self, layer_idx: int, head_idx: int, saved_activation: torch.Tensor):
        """Hook to patch a specific head's output at the last position."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output

            batch, seq, hidden = h.shape
            h_reshaped = h.view(batch, seq, self.n_heads, self.head_dim).clone()

            # Patch the specific head at last position
            h_reshaped[:, -1:, head_idx, :] = saved_activation.to(h.device)

            new_h = h_reshaped.view(batch, seq, hidden)

            if isinstance(output, tuple):
                return (new_h,) + output[1:]
            return new_h
        return hook_fn

    def test_path_patching(self, heads_to_patch: list) -> dict:
        """
        Test 2: Path patching - transfer head outputs from secure to neutral.
        If heads carry security signal, patching them should INCREASE P(snprintf) on neutral.
        """
        print("\n" + "="*70)
        print("TEST 2: PATH PATCHING")
        print("="*70)
        print("If heads carry security signal, patching secure→neutral should INCREASE P(snprintf)")

        # Baselines
        baseline_secure = self.get_snprintf_prob(SECURE_PROMPT)
        baseline_neutral = self.get_snprintf_prob(NEUTRAL_PROMPT)
        gap = baseline_secure - baseline_neutral

        print(f"\nBaselines:")
        print(f"  Secure:  P(snprintf) = {baseline_secure*100:.2f}%")
        print(f"  Neutral: P(snprintf) = {baseline_neutral*100:.2f}%")
        print(f"  Gap: {gap*100:.2f}%")

        results = {
            'baseline_secure': baseline_secure,
            'baseline_neutral': baseline_neutral,
            'gap': gap,
            'patches': {}
        }

        # First, save activations from secure prompt for all heads
        print(f"\nSaving activations from secure prompt...")
        saved = {}
        for layer_idx, head_idx in heads_to_patch:
            key = f"L{layer_idx}H{head_idx}"
            attn = self.model.model.layers[layer_idx].self_attn
            hook = attn.register_forward_hook(self.save_head_hook(layer_idx, head_idx, key))
            self.hooks.append(hook)

        _ = self.get_snprintf_prob(SECURE_PROMPT)  # Run forward pass to save
        saved = {k: v.clone() for k, v in self.saved_activations.items()}
        self.clear_hooks()

        print(f"\nPatching individual heads into NEUTRAL prompt:")
        print("-" * 60)

        for layer_idx, head_idx in heads_to_patch:
            head_key = f"L{layer_idx}H{head_idx}"

            # Register patching hook
            attn = self.model.model.layers[layer_idx].self_attn
            hook = attn.register_forward_hook(
                self.patch_head_hook(layer_idx, head_idx, saved[head_key])
            )
            self.hooks.append(hook)

            # Measure
            patched_prob = self.get_snprintf_prob(NEUTRAL_PROMPT)
            self.clear_hooks()

            lift = patched_prob - baseline_neutral
            lift_pct = (lift / gap) * 100 if gap > 0 else 0

            results['patches'][head_key] = {
                'layer': layer_idx,
                'head': head_idx,
                'patched_prob': patched_prob,
                'lift': lift,
                'lift_pct': lift_pct
            }

            # Impact indicator
            if lift_pct > 10:
                symbol = "🔴 SIGNIFICANT"
            elif lift_pct > 5:
                symbol = "🟡 MODERATE"
            elif lift_pct > 0:
                symbol = "🟢 SMALL"
            else:
                symbol = "⚪ NONE"

            print(f"  {head_key}: P(snprintf) = {patched_prob*100:.2f}% (lift: {lift_pct:+.1f}% of gap) {symbol}")

        # Test patching ALL candidate heads together
        print(f"\nPatching ALL {len(heads_to_patch)} candidate heads together:")
        for layer_idx, head_idx in heads_to_patch:
            head_key = f"L{layer_idx}H{head_idx}"
            attn = self.model.model.layers[layer_idx].self_attn
            hook = attn.register_forward_hook(
                self.patch_head_hook(layer_idx, head_idx, saved[head_key])
            )
            self.hooks.append(hook)

        all_patched_prob = self.get_snprintf_prob(NEUTRAL_PROMPT)
        self.clear_hooks()

        total_lift = all_patched_prob - baseline_neutral
        total_lift_pct = (total_lift / gap) * 100 if gap > 0 else 0

        results['all_patched'] = {
            'prob': all_patched_prob,
            'lift': total_lift,
            'lift_pct': total_lift_pct
        }

        print(f"  ALL HEADS: P(snprintf) = {all_patched_prob*100:.2f}% (lift: {total_lift_pct:+.1f}% of gap)")

        return results

    # =========================================================================
    # TEST 3: OUTPUT ANALYSIS
    # =========================================================================

    def test_output_analysis(self, heads_to_analyze: list) -> dict:
        """
        Test 3: Analyze what vectors these heads write to the residual stream.
        Compare output directions between secure and neutral contexts.
        """
        print("\n" + "="*70)
        print("TEST 3: OUTPUT ANALYSIS")
        print("="*70)
        print("Analyzing head output vectors and their similarity to snprintf direction")

        # Get the unembedding vector for snprintf (this is what promotes snprintf in output)
        unembed = self.model.lm_head.weight  # (vocab_size, hidden_size)
        snprintf_direction = unembed[self.snprintf_token].detach().cpu().numpy()
        sprintf_direction = unembed[self.sprintf_token].detach().cpu().numpy()

        # Normalize
        snprintf_direction = snprintf_direction / np.linalg.norm(snprintf_direction)
        sprintf_direction = sprintf_direction / np.linalg.norm(sprintf_direction)

        results = {
            'heads': {}
        }

        # Collect head outputs for both prompts
        print("\nCollecting head outputs...")

        for layer_idx, head_idx in heads_to_analyze:
            head_key = f"L{layer_idx}H{head_idx}"

            head_outputs = {'secure': None, 'neutral': None}

            for prompt_name, prompt in [('secure', SECURE_PROMPT), ('neutral', NEUTRAL_PROMPT)]:
                key = f"{head_key}_{prompt_name}"
                attn = self.model.model.layers[layer_idx].self_attn
                hook = attn.register_forward_hook(self.save_head_hook(layer_idx, head_idx, key))
                self.hooks.append(hook)

                _ = self.get_snprintf_prob(prompt)
                head_outputs[prompt_name] = self.saved_activations[key].squeeze().cpu().numpy()
                self.clear_hooks()

            # The head output needs to be projected to full hidden size
            # For now, analyze the head_dim vector and its properties

            secure_out = head_outputs['secure']
            neutral_out = head_outputs['neutral']

            # Compute difference
            diff = secure_out - neutral_out
            diff_norm = np.linalg.norm(diff)

            # Compute norms
            secure_norm = np.linalg.norm(secure_out)
            neutral_norm = np.linalg.norm(neutral_out)

            # Cosine similarity between secure and neutral outputs
            if secure_norm > 0 and neutral_norm > 0:
                cos_sim = np.dot(secure_out, neutral_out) / (secure_norm * neutral_norm)
            else:
                cos_sim = 0.0

            results['heads'][head_key] = {
                'layer': layer_idx,
                'head': head_idx,
                'secure_norm': float(secure_norm),
                'neutral_norm': float(neutral_norm),
                'diff_norm': float(diff_norm),
                'cosine_similarity': float(cos_sim),
                'norm_ratio': float(secure_norm / neutral_norm) if neutral_norm > 0 else 0
            }

            # Indicator
            if cos_sim < 0.9:
                symbol = "🔴 DIFFERENT"
            elif cos_sim < 0.95:
                symbol = "🟡 SOMEWHAT SIMILAR"
            else:
                symbol = "🟢 VERY SIMILAR"

            print(f"  {head_key}: cos_sim={cos_sim:.3f}, diff_norm={diff_norm:.3f} {symbol}")

        return results


def main():
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    verifier = HeadVerifier()

    # Run all three tests
    ablation_results = verifier.test_ablation(CANDIDATE_HEADS)
    patching_results = verifier.test_path_patching(CANDIDATE_HEADS)
    output_results = verifier.test_output_analysis(CANDIDATE_HEADS)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Are these Security Context Heads?")
    print("="*70)

    print("\n| Head   | Ablation Drop | Path Patch Lift | Output Diff | Verdict |")
    print("|--------|---------------|-----------------|-------------|---------|")

    verdicts = {}
    for layer_idx, head_idx in CANDIDATE_HEADS:
        head_key = f"L{layer_idx}H{head_idx}"

        abl = ablation_results['ablations'].get(head_key, {})
        patch = patching_results['patches'].get(head_key, {})
        out = output_results['heads'].get(head_key, {})

        abl_drop = abl.get('drop_pct', 0)
        patch_lift = patch.get('lift_pct', 0)
        cos_sim = out.get('cosine_similarity', 1.0)

        # Verdict: head is causal if ablation hurts OR patching helps significantly
        is_causal = (abl_drop > 5) or (patch_lift > 5)
        is_different = cos_sim < 0.95

        if is_causal and is_different:
            verdict = "✅ VERIFIED"
        elif is_causal:
            verdict = "🟡 LIKELY"
        elif is_different:
            verdict = "🟡 MAYBE"
        else:
            verdict = "❌ NOT CAUSAL"

        verdicts[head_key] = verdict

        print(f"| {head_key:6s} | {abl_drop:+12.1f}% | {patch_lift:+14.1f}% | {1-cos_sim:11.3f} | {verdict} |")

    # Overall
    print(f"\nAll heads ablated: {ablation_results['all_ablated']['drop_pct']:+.1f}% drop")
    print(f"All heads patched: {patching_results['all_patched']['lift_pct']:+.1f}% lift")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    full_results = {
        'timestamp': timestamp,
        'candidate_heads': [f"L{l}H{h}" for l, h in CANDIDATE_HEADS],
        'ablation': ablation_results,
        'path_patching': patching_results,
        'output_analysis': output_results,
        'verdicts': verdicts
    }

    with open(results_dir / f"head_verification_{timestamp}.json", 'w') as f:
        json.dump(full_results, f, indent=2, default=float)

    print(f"\n💾 Results saved to: {results_dir}")

    return full_results


if __name__ == "__main__":
    results = main()
