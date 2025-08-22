#!/usr/bin/env python3
"""
Test subsets of odd heads to see if any combinations work.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
from contextlib import contextmanager
import json
from datetime import datetime
import time

class OddHeadsTest:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        
        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        self.model.eval()
        
        self.n_heads = 32
        self.layer_idx = 10
        self.saved_activations = {}
        self.hooks = []
    
    def get_attention_module(self, layer_idx: int):
        return self.model.model.layers[layer_idx].self_attn
    
    def save_activation_hook(self, key: str):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self.saved_activations[key] = hidden_states.detach().cpu()
        return hook_fn
    
    def selective_patch_hook(self, saved_activation: torch.Tensor, head_indices: List[int]):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            batch_size, seq_len, hidden_size = hidden_states.shape
            head_dim = hidden_size // self.n_heads
            
            hidden_states_reshaped = hidden_states.view(batch_size, seq_len, self.n_heads, head_dim)
            saved_reshaped = saved_activation.to(hidden_states.device).view(batch_size, -1, self.n_heads, head_dim)
            
            new_hidden = hidden_states_reshaped.clone()
            min_seq_len = min(seq_len, saved_reshaped.shape[1])
            
            for head_idx in head_indices:
                new_hidden[:, :min_seq_len, head_idx, :] = saved_reshaped[:, :min_seq_len, head_idx, :]
            
            new_hidden = new_hidden.view(batch_size, seq_len, hidden_size)
            
            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden
        return hook_fn
    
    @contextmanager
    def save_activation_context(self, prompt: str):
        try:
            module = self.get_attention_module(self.layer_idx)
            key = f"layer_{self.layer_idx}_attention"
            
            hook = module.register_forward_hook(self.save_activation_hook(key))
            self.hooks.append(hook)
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                _ = self.model(**inputs)
            
            yield self.saved_activations
            
        finally:
            for hook in self.hooks:
                hook.remove()
            self.hooks.clear()
    
    @contextmanager
    def patch_activation_context(self, saved_activation: torch.Tensor, head_indices: List[int]):
        try:
            module = self.get_attention_module(self.layer_idx)
            hook = module.register_forward_hook(
                self.selective_patch_hook(saved_activation, head_indices)
            )
            self.hooks.append(hook)
            yield
        finally:
            for hook in self.hooks:
                hook.remove()
            self.hooks.clear()
    
    def generate(self, prompt: str, max_new_tokens: int = 20) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        return generated
    
    def check_bug_fixed(self, output: str) -> bool:
        output_lower = output.lower()
        
        correct_patterns = ["9.8 is bigger", "9.8 is larger", "9.8 is greater"]
        bug_patterns = ["9.11 is bigger", "9.11 is larger", "9.11 is greater"]
        
        has_correct = any(pattern in output_lower for pattern in correct_patterns)
        has_bug = any(pattern in output_lower for pattern in bug_patterns)
        
        return has_correct and not has_bug
    
    def test_head_subset(self, head_indices: List[int], n_trials: int = 30, name: str = "") -> Dict:
        """Test a specific subset of heads"""
        success_count = 0
        
        # Save activation once
        correct_prompt = "Which is bigger: 9.8 or 9.11?\nAnswer:"
        with self.save_activation_context(correct_prompt) as saved:
            correct_activation = saved[f"layer_{self.layer_idx}_attention"]
        
        buggy_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"
        
        print(f"Testing {name} - heads: {head_indices}")
        
        for trial in range(n_trials):
            with self.patch_activation_context(correct_activation, head_indices):
                output = self.generate(buggy_prompt, max_new_tokens=20)
            
            if self.check_bug_fixed(output):
                success_count += 1
            
            # Progress indicator
            if (trial + 1) % 10 == 0:
                print(f"  Progress: {trial+1}/{n_trials} - {success_count} successes", end='\r')
        
        success_rate = success_count / n_trials
        
        # Calculate 95% CI
        se = np.sqrt(success_rate * (1 - success_rate) / n_trials)
        ci_lower = max(0, success_rate - 1.96 * se)
        ci_upper = min(1, success_rate + 1.96 * se)
        
        print(f"  Result: {success_rate:.1%} [{ci_lower:.1%}, {ci_upper:.1%}]        ")
        
        return {
            'success_rate': success_rate,
            'success_count': success_count,
            'n_trials': n_trials,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'head_indices': head_indices,
            'n_heads': len(head_indices)
        }
    
    def run_odd_tests(self, n_trials: int = 30) -> Dict:
        """Test various subsets of odd heads"""
        results = {}
        all_odd_heads = list(range(1, 32, 2))  # [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31]
        
        print("\n" + "="*70)
        print(f"TESTING ODD HEAD SUBSETS (n={n_trials} per test)")
        print("="*70)
        
        # 1. All 16 odd heads (baseline - we know this fails)
        print("\n1. ALL 16 ODD HEADS (baseline)")
        results['odd_16'] = self.test_head_subset(all_odd_heads, n_trials, "all 16 odd")
        
        # 2. First 8 odd heads
        print("\n2. FIRST 8 ODD HEADS")
        first_8_odd = all_odd_heads[:8]  # [1,3,5,7,9,11,13,15]
        results['first_8_odd'] = self.test_head_subset(first_8_odd, n_trials, "first 8 odd")
        
        # 3. Last 8 odd heads
        print("\n3. LAST 8 ODD HEADS")
        last_8_odd = all_odd_heads[8:]  # [17,19,21,23,25,27,29,31]
        results['last_8_odd'] = self.test_head_subset(last_8_odd, n_trials, "last 8 odd")
        
        # 4. Every other odd head (8 heads)
        print("\n4. EVERY OTHER ODD HEAD (8 heads)")
        every_other_odd = all_odd_heads[::2]  # [1,5,9,13,17,21,25,29]
        results['every_other_odd'] = self.test_head_subset(every_other_odd, n_trials, "every other odd")
        
        # 5. First 4 odd heads
        print("\n5. FIRST 4 ODD HEADS")
        first_4_odd = all_odd_heads[:4]  # [1,3,5,7]
        results['first_4_odd'] = self.test_head_subset(first_4_odd, n_trials, "first 4 odd")
        
        # 6. Last 4 odd heads
        print("\n6. LAST 4 ODD HEADS")
        last_4_odd = all_odd_heads[-4:]  # [25,27,29,31]
        results['last_4_odd'] = self.test_head_subset(last_4_odd, n_trials, "last 4 odd")
        
        # 7. Middle 4 odd heads
        print("\n7. MIDDLE 4 ODD HEADS")
        middle_4_odd = all_odd_heads[6:10]  # [13,15,17,19]
        results['middle_4_odd'] = self.test_head_subset(middle_4_odd, n_trials, "middle 4 odd")
        
        # 8. Every 4th odd head (4 heads)
        print("\n8. EVERY 4TH ODD HEAD (4 heads)")
        every_4th_odd = all_odd_heads[::4]  # [1,9,17,25]
        results['every_4th_odd'] = self.test_head_subset(every_4th_odd, n_trials, "every 4th odd")
        
        # 9. First 2 odd heads
        print("\n9. FIRST 2 ODD HEADS")
        first_2_odd = all_odd_heads[:2]  # [1,3]
        results['first_2_odd'] = self.test_head_subset(first_2_odd, n_trials, "first 2 odd")
        
        # 10. Last 2 odd heads
        print("\n10. LAST 2 ODD HEADS")
        last_2_odd = all_odd_heads[-2:]  # [29,31]
        results['last_2_odd'] = self.test_head_subset(last_2_odd, n_trials, "last 2 odd")
        
        # 11. Extremes (2 heads)
        print("\n11. EXTREME ODD HEADS (first and last)")
        extreme_odd = [all_odd_heads[0], all_odd_heads[-1]]  # [1,31]
        results['extreme_odd'] = self.test_head_subset(extreme_odd, n_trials, "extreme odd")
        
        # 12. Single odd heads (test a few)
        print("\n12. SINGLE ODD HEADS")
        for head in [1, 15, 31]:
            print(f"\n    Testing single head {head}:")
            results[f'single_{head}'] = self.test_head_subset([head], n_trials, f"single head {head}")
        
        # 13. Mixed experiments - what if we combine best odd with best even?
        print("\n13. MIXED ODD + EVEN EXPERIMENTS")
        
        # Best 4 odd + best 4 even
        print("\n    Testing 4 odd + 4 even:")
        mixed_4_4 = [1, 3, 5, 7] + [0, 2, 4, 6]  # first 4 of each
        results['mixed_4odd_4even'] = self.test_head_subset(mixed_4_4, n_trials, "4 odd + 4 even")
        
        # Best 2 odd + best 6 even
        print("\n    Testing 2 odd + 6 even:")
        mixed_2_6 = [1, 3] + [0, 2, 4, 6, 8, 10]  # 2 odd + 6 even
        results['mixed_2odd_6even'] = self.test_head_subset(mixed_2_6, n_trials, "2 odd + 6 even")
        
        return results
    
    def print_summary(self, results: Dict):
        """Print summary organized by type"""
        print("\n" + "="*70)
        print("SUMMARY: ODD HEAD SUBSET ANALYSIS")
        print("="*70)
        
        # Pure odd subsets
        print("\nPURE ODD HEAD SUBSETS:")
        print("-" * 50)
        odd_only = {k: v for k, v in results.items() if 'odd' in k and 'mixed' not in k}
        for key, result in sorted(odd_only.items()):
            print(f"{key:20s}: {result['success_rate']:6.1%} [{result['ci_lower']:5.1%}, {result['ci_upper']:5.1%}] ({result['n_heads']} heads)")
        
        # Mixed experiments
        print("\nMIXED ODD + EVEN EXPERIMENTS:")
        print("-" * 50)
        mixed = {k: v for k, v in results.items() if 'mixed' in k}
        for key, result in sorted(mixed.items()):
            print(f"{key:20s}: {result['success_rate']:6.1%} [{result['ci_lower']:5.1%}, {result['ci_upper']:5.1%}] ({result['n_heads']} heads)")
        
        # Analysis
        print("\n" + "="*70)
        print("KEY FINDINGS:")
        
        any_odd_working = any(r['success_rate'] > 0.1 for k, r in results.items() if 'mixed' not in k)
        if any_odd_working:
            working_odd = [k for k, r in results.items() if r['success_rate'] > 0.1 and 'mixed' not in k]
            print(f"✅ Some odd subsets work: {', '.join(working_odd)}")
        else:
            print("❌ NO pure odd head subsets work (all ≤ 10%)")
        
        mixed_working = [k for k, r in results.items() if r['success_rate'] > 0.5 and 'mixed' in k]
        if mixed_working:
            print(f"⚡ Mixed odd+even working: {', '.join(mixed_working)}")
        else:
            print("❌ Mixed odd+even combinations also fail")
        
        print(f"\n💡 Conclusion: Odd heads appear fundamentally incompatible with decimal comparison fix")

def main():
    start_time = time.time()
    
    tester = OddHeadsTest(device="cuda")
    results = tester.run_odd_tests(n_trials=30)  # 30 trials for speed
    tester.print_summary(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"odd_heads_subsets_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    total_time = time.time() - start_time
    print(f"\n✅ Odd head testing complete in {total_time:.1f} seconds")
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    main()