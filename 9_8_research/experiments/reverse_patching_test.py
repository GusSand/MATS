#!/usr/bin/env python3
"""
Experiment 4: Reverse Patching Test - Is it POSITION or CONTENT?

Question: Why do even heads fix the bug but odd heads don't?

Hypothesis A (POSITION): Even-indexed positions are processed differently by downstream layers
Hypothesis B (CONTENT): Even heads produce outputs with specific information that helps

Test Design:
1. Baseline Even: Patch even heads from correct→buggy (expect: 100%)
2. Baseline Odd: Patch odd heads from correct→buggy (expect: 0%)
3. Cross-patch A: Put EVEN head content into ODD positions
4. Cross-patch B: Put ODD head content into EVEN positions

If POSITION matters: Cross-patch A fails (odd positions are broken)
If CONTENT matters: Cross-patch A succeeds (good content fixes regardless of position)
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from contextlib import contextmanager
from datetime import datetime
import json

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LAYER_IDX = 10
DEVICE = "cuda"

EVEN_HEADS = list(range(0, 32, 2))
ODD_HEADS = list(range(1, 32, 2))

PROMPT_BUGGY = "Q: Which is bigger: 9.8 or 9.11?\nA:"
PROMPT_CORRECT = "Which is bigger: 9.8 or 9.11?\nAnswer:"


class ReversePatchingTest:
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
        self.saved_activations = {}

    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    @contextmanager
    def save_context(self, prompt: str):
        """Save attention output from a prompt."""
        try:
            def hook(module, input, output):
                self.saved_activations['attn'] = output[0].detach().cpu()

            h = self.model.model.layers[LAYER_IDX].self_attn.register_forward_hook(hook)
            self.hooks.append(h)

            inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                _ = self.model(**inputs)

            yield self.saved_activations
        finally:
            self.clear_hooks()

    def create_patch_hook(self, saved_activation, source_heads, target_heads):
        """
        Create a hook that patches specific heads.

        source_heads: which heads to take content FROM (in saved_activation)
        target_heads: which head POSITIONS to patch (in current forward pass)
        """
        def hook(module, input, output):
            hidden = output[0]  # [batch, seq, hidden]
            batch, seq_len, hidden_size = hidden.shape

            # Reshape to per-head
            hidden_reshaped = hidden.view(batch, seq_len, self.n_heads, self.head_dim)
            saved_reshaped = saved_activation.to(hidden.device).view(batch, -1, self.n_heads, self.head_dim)

            new_hidden = hidden_reshaped.clone()
            min_seq = min(seq_len, saved_reshaped.shape[1])

            # Patch: take content from source_heads, put into target_heads positions
            for src, tgt in zip(source_heads, target_heads):
                new_hidden[:, :min_seq, tgt, :] = saved_reshaped[:, :min_seq, src, :]

            new_hidden = new_hidden.view(batch, seq_len, hidden_size)
            return (new_hidden,) + output[1:]

        return hook

    @contextmanager
    def patch_context(self, saved_activation, source_heads, target_heads):
        """Apply patching during forward pass."""
        try:
            hook = self.create_patch_hook(saved_activation, source_heads, target_heads)
            h = self.model.model.layers[LAYER_IDX].self_attn.register_forward_hook(hook)
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
        correct = any(p in output_lower for p in ["9.8 is bigger", "9.8 is larger", "9.8 is greater", "9.8 >"])
        wrong = any(p in output_lower for p in ["9.11 is bigger", "9.11 is larger", "9.11 is greater", "9.11 >"])
        return correct and not wrong

    def run_test(self, source_heads, target_heads, n_trials: int = 20, name: str = "") -> dict:
        """Run patching test with given configuration."""
        print(f"\nTesting: {name}")
        print(f"  Source heads (content from): {source_heads[:4]}... ({len(source_heads)} heads)")
        print(f"  Target heads (positions):    {target_heads[:4]}... ({len(target_heads)} heads)")

        # Save activation from correct prompt
        with self.save_context(PROMPT_CORRECT) as saved:
            correct_activation = saved['attn'].clone()

        successes = 0
        for trial in range(n_trials):
            with self.patch_context(correct_activation, source_heads, target_heads):
                output = self.generate(PROMPT_BUGGY)

            if self.check_correct(output):
                successes += 1

            if trial == 0:
                print(f"  Example output: {output[:60]}...")

        rate = successes / n_trials
        print(f"  Result: {successes}/{n_trials} = {rate:.0%}")

        return {
            'name': name,
            'source_heads': source_heads,
            'target_heads': target_heads,
            'success_rate': rate,
            'successes': successes,
            'n_trials': n_trials
        }

    def run_all_tests(self, n_trials: int = 20):
        """Run all patching configurations."""
        results = {
            'model': MODEL_NAME,
            'layer': LAYER_IDX,
            'timestamp': datetime.now().isoformat(),
            'tests': []
        }

        print("\n" + "="*70)
        print("EXPERIMENT 4: REVERSE PATCHING TEST")
        print("Is the effect about HEAD POSITION or HEAD CONTENT?")
        print("="*70)

        # Test 1: Baseline - patch even heads normally
        r1 = self.run_test(
            source_heads=EVEN_HEADS,
            target_heads=EVEN_HEADS,
            n_trials=n_trials,
            name="BASELINE: Even→Even (expect ~100%)"
        )
        results['tests'].append(r1)

        # Test 2: Baseline - patch odd heads normally
        r2 = self.run_test(
            source_heads=ODD_HEADS,
            target_heads=ODD_HEADS,
            n_trials=n_trials,
            name="BASELINE: Odd→Odd (expect ~0%)"
        )
        results['tests'].append(r2)

        # Test 3: Cross-patch - even content into odd positions
        r3 = self.run_test(
            source_heads=EVEN_HEADS,
            target_heads=ODD_HEADS,
            n_trials=n_trials,
            name="CROSS: Even content → Odd positions"
        )
        results['tests'].append(r3)

        # Test 4: Cross-patch - odd content into even positions
        r4 = self.run_test(
            source_heads=ODD_HEADS,
            target_heads=EVEN_HEADS,
            n_trials=n_trials,
            name="CROSS: Odd content → Even positions"
        )
        results['tests'].append(r4)

        # Test 5: All heads from correct (sanity check)
        r5 = self.run_test(
            source_heads=list(range(32)),
            target_heads=list(range(32)),
            n_trials=n_trials,
            name="SANITY: All 32 heads (expect ~100%)"
        )
        results['tests'].append(r5)

        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)

        print(f"\n{'Test':<45} {'Result':<10}")
        print("-" * 55)
        for t in results['tests']:
            print(f"{t['name']:<45} {t['success_rate']:>6.0%}")

        # Interpretation
        print("\n" + "-"*70)
        print("INTERPRETATION")
        print("-"*70)

        even_even = r1['success_rate']
        odd_odd = r2['success_rate']
        even_to_odd = r3['success_rate']
        odd_to_even = r4['success_rate']

        if even_to_odd > 0.5:
            print("\n✓ Even content → Odd positions WORKS")
            print("  → The CONTENT of even heads matters, not their position")
        else:
            print("\n✗ Even content → Odd positions FAILS")
            print("  → The POSITION matters - odd positions can't use even content")

        if odd_to_even > 0.5:
            print("\n✓ Odd content → Even positions WORKS")
            print("  → Even positions can use any content")
        else:
            print("\n✗ Odd content → Even positions FAILS")
            print("  → The CONTENT matters - even positions need specific content")

        # Save results
        with open('reverse_patching_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        # Write markdown
        self.write_markdown(results)

        return results

    def write_markdown(self, results):
        tests = results['tests']

        md = f"""# Reverse Patching Test Results

**Model**: {results['model']}
**Layer**: {results['layer']}
**Date**: {results['timestamp']}

---

## Question

Why do even heads fix the bug but odd heads don't?

- **Hypothesis A (POSITION)**: Even-indexed positions are processed differently by downstream layers
- **Hypothesis B (CONTENT)**: Even heads produce outputs with specific information that helps

---

## Results

| Test | Source → Target | Success Rate |
|------|-----------------|--------------|
| Baseline Even | Even → Even | {tests[0]['success_rate']:.0%} |
| Baseline Odd | Odd → Odd | {tests[1]['success_rate']:.0%} |
| **Cross-patch A** | **Even → Odd** | **{tests[2]['success_rate']:.0%}** |
| **Cross-patch B** | **Odd → Even** | **{tests[3]['success_rate']:.0%}** |
| Sanity Check | All → All | {tests[4]['success_rate']:.0%} |

---

## Interpretation

"""
        even_to_odd = tests[2]['success_rate']
        odd_to_even = tests[3]['success_rate']

        if even_to_odd > 0.5 and odd_to_even < 0.5:
            md += """**CONTENT matters, not POSITION.**

- Even head content works even in odd positions
- Odd head content fails even in even positions
- The information encoded by even heads is what fixes the bug
"""
        elif even_to_odd < 0.5 and odd_to_even > 0.5:
            md += """**POSITION matters, not CONTENT.**

- Even head content fails in odd positions
- Odd head content works in even positions
- Downstream layers treat even positions differently
"""
        elif even_to_odd < 0.5 and odd_to_even < 0.5:
            md += """**BOTH POSITION and CONTENT matter.**

- Even content in odd positions fails
- Odd content in even positions also fails
- Need both: the RIGHT content in the RIGHT position
"""
        else:
            md += """**Unexpected result - needs further investigation.**
"""

        md += """
---

## One-Slide Summary

```
IS IT POSITION OR CONTENT?
━━━━━━━━━━━━━━━━━━━━━━━━━━

Baseline:
  Even→Even: """ + f"{tests[0]['success_rate']:.0%}" + """
  Odd→Odd:   """ + f"{tests[1]['success_rate']:.0%}" + """

Cross-patch:
  Even content → Odd positions: """ + f"{tests[2]['success_rate']:.0%}" + """
  Odd content → Even positions: """ + f"{tests[3]['success_rate']:.0%}" + """

CONCLUSION: [Based on results above]
```
"""

        with open('reverse_patching_results.md', 'w') as f:
            f.write(md)

        print("\nResults written to: reverse_patching_results.md")


def main():
    tester = ReversePatchingTest()
    results = tester.run_all_tests(n_trials=20)


if __name__ == "__main__":
    main()
