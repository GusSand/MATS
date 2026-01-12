#!/usr/bin/env python3
"""
Experiment 7: Test patching at different layers.

Maybe layer 10 isn't special - we just happened to test there.
Let's see if the even/odd pattern holds at other layers.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from contextlib import contextmanager
from datetime import datetime
import json

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEVICE = "cuda"

EVEN_HEADS = list(range(0, 32, 2))
ODD_HEADS = list(range(1, 32, 2))

PROMPT_BUGGY = "Q: Which is bigger: 9.8 or 9.11?\nA:"
PROMPT_CORRECT = "Which is bigger: 9.8 or 9.11?\nAnswer:"


class LayerTester:
    def __init__(self):
        print(f"Loading {MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map=DEVICE
        )
        self.model.eval()

        self.n_heads = 32
        self.head_dim = self.model.config.hidden_size // self.n_heads
        self.hooks = []
        self.saved = {}

    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    @contextmanager
    def save_context(self, prompt: str, layer_idx: int):
        """Save attention output from correct prompt."""
        try:
            def hook(module, input, output):
                self.saved[layer_idx] = output[0].detach().cpu()

            h = self.model.model.layers[layer_idx].self_attn.register_forward_hook(hook)
            self.hooks.append(h)

            inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                _ = self.model(**inputs)

            yield self.saved
        finally:
            self.clear_hooks()

    def create_patch_hook(self, saved_activation, head_indices, layer_idx):
        """Create hook for patching specific heads."""
        def hook(module, input, output):
            hidden = output[0]
            batch, seq_len, hidden_size = hidden.shape

            hidden_reshaped = hidden.view(batch, seq_len, self.n_heads, self.head_dim)
            saved_reshaped = saved_activation.to(hidden.device).view(batch, -1, self.n_heads, self.head_dim)

            new_hidden = hidden_reshaped.clone()
            min_seq = min(seq_len, saved_reshaped.shape[1])

            for head_idx in head_indices:
                new_hidden[:, :min_seq, head_idx, :] = saved_reshaped[:, :min_seq, head_idx, :]

            new_hidden = new_hidden.view(batch, seq_len, hidden_size)
            return (new_hidden,) + output[1:]

        return hook

    @contextmanager
    def patch_context(self, saved_activation, head_indices, layer_idx):
        """Apply patching during forward pass."""
        try:
            hook = self.create_patch_hook(saved_activation, head_indices, layer_idx)
            h = self.model.model.layers[layer_idx].self_attn.register_forward_hook(hook)
            self.hooks.append(h)
            yield
        finally:
            self.clear_hooks()

    def generate(self, prompt: str, max_new_tokens: int = 20) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        return self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    def check_correct(self, output: str) -> bool:
        output_lower = output.lower()
        correct = any(p in output_lower for p in ["9.8 is bigger", "9.8 is larger", "9.8 is greater", "9.8 >", "9.8."])
        wrong = any(p in output_lower for p in ["9.11 is bigger", "9.11 is larger", "9.11 is greater", "9.11 >"])
        return correct and not wrong

    def test_layer(self, layer_idx: int, n_trials: int = 10):
        """Test even/odd patching at a specific layer."""
        print(f"\n--- Layer {layer_idx} ---")

        # Save correct activation
        with self.save_context(PROMPT_CORRECT, layer_idx) as saved:
            correct_activation = saved[layer_idx].clone()

        results = {}

        # Test even heads
        even_success = 0
        for _ in range(n_trials):
            with self.patch_context(correct_activation, EVEN_HEADS, layer_idx):
                output = self.generate(PROMPT_BUGGY)
            if self.check_correct(output):
                even_success += 1

        results['even'] = even_success / n_trials
        print(f"  Even heads: {even_success}/{n_trials} = {results['even']:.0%}")

        # Test odd heads
        odd_success = 0
        for _ in range(n_trials):
            with self.patch_context(correct_activation, ODD_HEADS, layer_idx):
                output = self.generate(PROMPT_BUGGY)
            if self.check_correct(output):
                odd_success += 1

        results['odd'] = odd_success / n_trials
        print(f"  Odd heads:  {odd_success}/{n_trials} = {results['odd']:.0%}")

        # Test all heads (sanity check)
        all_success = 0
        for _ in range(n_trials):
            with self.patch_context(correct_activation, list(range(32)), layer_idx):
                output = self.generate(PROMPT_BUGGY)
            if self.check_correct(output):
                all_success += 1

        results['all'] = all_success / n_trials
        print(f"  All heads:  {all_success}/{n_trials} = {results['all']:.0%}")

        return results

    def run_all_layers(self, layers_to_test, n_trials=10):
        """Test multiple layers."""
        print("\n" + "="*70)
        print("EXPERIMENT 7: TESTING DIFFERENT LAYERS")
        print("="*70)

        all_results = {
            'model': MODEL_NAME,
            'timestamp': datetime.now().isoformat(),
            'n_trials': n_trials,
            'layers': {}
        }

        for layer in layers_to_test:
            all_results['layers'][layer] = self.test_layer(layer, n_trials)

        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"\n{'Layer':<8} {'Even':<10} {'Odd':<10} {'All':<10} {'Even-Odd':<10}")
        print("-" * 50)
        for layer in layers_to_test:
            r = all_results['layers'][layer]
            diff = r['even'] - r['odd']
            print(f"L{layer:<6} {r['even']:<10.0%} {r['odd']:<10.0%} {r['all']:<10.0%} {diff:+.0%}")

        # Save
        with open('layer_test_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)

        self.write_markdown(all_results, layers_to_test)

        return all_results

    def write_markdown(self, results, layers):
        md = f"""# Layer Testing Results

**Model**: {results['model']}
**Date**: {results['timestamp']}
**Trials per test**: {results['n_trials']}

---

## Results by Layer

| Layer | Even Heads | Odd Heads | All Heads | Even-Odd Diff |
|-------|------------|-----------|-----------|---------------|
"""
        for layer in layers:
            r = results['layers'][layer]
            diff = r['even'] - r['odd']
            md += f"| {layer} | {r['even']:.0%} | {r['odd']:.0%} | {r['all']:.0%} | {diff:+.0%} |\n"

        md += """
---

## Interpretation

If the even/odd pattern appears at multiple layers → it's a general property
If it only appears at layer 10 → layer 10 is special for this task

---

## One-Slide Summary

```
LAYER TESTING: Is layer 10 special?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Results will show if even/odd pattern is layer-specific]
```
"""

        with open('layer_test_results.md', 'w') as f:
            f.write(md)

        print("\nResults written to: layer_test_results.md")


def main():
    tester = LayerTester()
    # Test layers around 10
    layers_to_test = [8, 9, 10, 11, 12, 15, 20]
    tester.run_all_layers(layers_to_test, n_trials=10)


if __name__ == "__main__":
    main()
