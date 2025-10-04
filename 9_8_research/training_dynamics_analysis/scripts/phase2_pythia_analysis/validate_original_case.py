#!/usr/bin/env python3
"""
Validate Original Case - Ensure 9.8 vs 9.11 still shows specialization
====================================================================

Before analyzing the comprehensive results, let's validate that our
test setup correctly detects the known 9.8 vs 9.11 specialization.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from contextlib import contextmanager

class OriginalCaseValidator:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model_name = "EleutherAI/pythia-160m"
        self.n_heads = 12
        self.target_layer = 6
        self.hooks = []

    @contextmanager
    def temporary_hooks(self):
        try:
            yield
        finally:
            for hook in self.hooks:
                hook.remove()
            self.hooks.clear()

    def get_attention_module(self, model, layer_idx: int):
        return model.gpt_neox.layers[layer_idx].attention

    def selective_patch_hook(self, saved_activation: torch.Tensor, head_indices: List[int]):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            batch_size, seq_len, hidden_dim = hidden_states.shape
            head_dim = hidden_dim // self.n_heads
            reshaped = hidden_states.view(batch_size, seq_len, self.n_heads, head_dim)

            saved_batch, saved_seq, saved_heads, saved_head_dim = saved_activation.shape
            min_seq_len = min(seq_len, saved_seq)

            new_hidden = reshaped.clone()
            saved_reshaped = saved_activation.view(saved_batch, saved_seq, saved_heads, saved_head_dim)

            for head_idx in head_indices:
                if head_idx < self.n_heads:
                    new_hidden[:, :min_seq_len, head_idx, :] = saved_reshaped[:, :min_seq_len, head_idx, :]

            new_output = new_hidden.view(batch_size, seq_len, hidden_dim)

            if isinstance(output, tuple):
                return (new_output,) + output[1:]
            else:
                return new_output
        return hook_fn

    def test_original_case(self, trials: int = 15):
        """Test the original 9.8 vs 9.11 case"""
        print("🔍 VALIDATING ORIGINAL CASE: 9.8 vs 9.11")
        print("="*50)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        model.eval()

        clean_prompt = "Which is bigger: 9.8 or 9.11?"
        buggy_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"

        even_heads = list(range(0, self.n_heads, 2))
        odd_heads = list(range(1, self.n_heads, 2))

        def check_bug_fixed(output: str) -> bool:
            output_lower = output.lower()
            correct_patterns = ["9.8 is bigger", "9.8 is larger", "9.8"]
            bug_patterns = ["9.11 is bigger", "9.11 is larger", "9.11"]
            has_correct = any(pattern in output_lower for pattern in correct_patterns)
            has_bug = any(pattern in output_lower for pattern in bug_patterns)
            return has_correct and not has_bug

        try:
            with self.temporary_hooks():
                # Test baseline
                print("Testing baseline...")
                baseline_success = 0
                baseline_samples = []

                for trial in range(trials):
                    inputs = tokenizer(buggy_prompt, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=20,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id
                        )
                    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                    baseline_samples.append(response.strip())
                    if check_bug_fixed(response):
                        baseline_success += 1

                baseline_rate = baseline_success / trials
                print(f"Baseline success: {baseline_rate:.1%}")
                print(f"Sample baseline: {baseline_samples[0][:50]}...")

                # Get clean activation
                print("\nGetting clean activation...")
                saved_activations = {}
                clean_inputs = tokenizer(clean_prompt, return_tensors="pt").to(self.device)
                attention_module = self.get_attention_module(model, self.target_layer)

                def save_hook(module, input, output):
                    if isinstance(output, tuple):
                        saved_activations['clean'] = output[0].detach()
                    else:
                        saved_activations['clean'] = output.detach()

                save_handle = attention_module.register_forward_hook(save_hook)
                with torch.no_grad():
                    model(**clean_inputs)
                save_handle.remove()

                print(f"Clean activation shape: {saved_activations['clean'].shape}")

                # Test even heads
                print("\nTesting even heads...")
                even_success = 0
                even_samples = []

                if 'clean' in saved_activations:
                    patch_hook = self.selective_patch_hook(saved_activations['clean'], even_heads)
                    hook_handle = attention_module.register_forward_hook(patch_hook)
                    self.hooks.append(hook_handle)

                    for trial in range(trials):
                        inputs = tokenizer(buggy_prompt, return_tensors="pt").to(self.device)
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=20,
                                do_sample=False,
                                pad_token_id=tokenizer.pad_token_id
                            )
                        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                        even_samples.append(response.strip())
                        if check_bug_fixed(response):
                            even_success += 1

                    hook_handle.remove()
                    self.hooks = []

                even_rate = even_success / trials
                print(f"Even heads success: {even_rate:.1%}")
                print(f"Sample even: {even_samples[0][:50]}...")

                # Test odd heads
                print("\nTesting odd heads...")
                odd_success = 0
                odd_samples = []

                if 'clean' in saved_activations:
                    patch_hook = self.selective_patch_hook(saved_activations['clean'], odd_heads)
                    hook_handle = attention_module.register_forward_hook(patch_hook)
                    self.hooks.append(hook_handle)

                    for trial in range(trials):
                        inputs = tokenizer(buggy_prompt, return_tensors="pt").to(self.device)
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=20,
                                do_sample=False,
                                pad_token_id=tokenizer.pad_token_id
                            )
                        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                        odd_samples.append(response.strip())
                        if check_bug_fixed(response):
                            odd_success += 1

                    hook_handle.remove()
                    self.hooks = []

                odd_rate = odd_success / trials
                print(f"Odd heads success: {odd_rate:.1%}")
                print(f"Sample odd: {odd_samples[0][:50]}...")

                # Results
                specialization = even_rate - odd_rate
                print(f"\n{'='*50}")
                print(f"VALIDATION RESULTS:")
                print(f"Baseline: {baseline_rate:.1%}")
                print(f"Even heads: {even_rate:.1%}")
                print(f"Odd heads: {odd_rate:.1%}")
                print(f"Specialization strength: {specialization:+.2f}")

                if abs(specialization) > 0.5:
                    print("✅ Strong specialization detected - test setup working!")
                elif abs(specialization) > 0.3:
                    print("⚡ Moderate specialization - test setup partially working")
                else:
                    print("❌ No specialization - test setup issue!")

        except Exception as e:
            print(f"❌ Error: {e}")

        finally:
            del model
            del tokenizer
            torch.cuda.empty_cache()

if __name__ == "__main__":
    validator = OriginalCaseValidator()
    validator.test_original_case()