#!/usr/bin/env python3
"""
Test progressively smaller subsets of even heads to find the minimal set.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
from contextlib import contextmanager
import json
from datetime import datetime
import time

class MinimalEvenHeadsTest:
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
    
    def test_head_subset(self, head_indices: List[int], n_trials: int = 50, name: str = "") -> Dict:
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
    
    def run_minimal_tests(self, n_trials: int = 50) -> Dict:
        """Test progressively smaller subsets of even heads"""
        results = {}
        all_even_heads = list(range(0, 32, 2))  # [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
        
        print("\n" + "="*70)
        print(f"TESTING MINIMAL EVEN HEAD SUBSETS (n={n_trials} per test)")
        print("="*70)
        
        # 1. All 16 even heads (baseline)
        print("\n1. ALL 16 EVEN HEADS (baseline)")
        results['even_16'] = self.test_head_subset(all_even_heads, n_trials, "all 16 even")
        
        # 2. First 8 even heads
        print("\n2. FIRST 8 EVEN HEADS")
        first_8_even = all_even_heads[:8]  # [0,2,4,6,8,10,12,14]
        results['first_8_even'] = self.test_head_subset(first_8_even, n_trials, "first 8 even")
        
        # 3. Last 8 even heads
        print("\n3. LAST 8 EVEN HEADS")
        last_8_even = all_even_heads[8:]  # [16,18,20,22,24,26,28,30]
        results['last_8_even'] = self.test_head_subset(last_8_even, n_trials, "last 8 even")
        
        # 4. Every other even head (8 heads)
        print("\n4. EVERY OTHER EVEN HEAD (8 heads)")
        every_other_even = all_even_heads[::2]  # [0,4,8,12,16,20,24,28]
        results['every_other_even'] = self.test_head_subset(every_other_even, n_trials, "every other even")
        
        # 5. First 4 even heads
        print("\n5. FIRST 4 EVEN HEADS")
        first_4_even = all_even_heads[:4]  # [0,2,4,6]
        results['first_4_even'] = self.test_head_subset(first_4_even, n_trials, "first 4 even")
        
        # 6. Last 4 even heads
        print("\n6. LAST 4 EVEN HEADS")
        last_4_even = all_even_heads[-4:]  # [24,26,28,30]
        results['last_4_even'] = self.test_head_subset(last_4_even, n_trials, "last 4 even")
        
        # 7. Middle 4 even heads
        print("\n7. MIDDLE 4 EVEN HEADS")
        middle_4_even = all_even_heads[6:10]  # [12,14,16,18]
        results['middle_4_even'] = self.test_head_subset(middle_4_even, n_trials, "middle 4 even")
        
        # 8. Every 4th even head (4 heads)
        print("\n8. EVERY 4TH EVEN HEAD (4 heads)")
        every_4th_even = all_even_heads[::4]  # [0,8,16,24]
        results['every_4th_even'] = self.test_head_subset(every_4th_even, n_trials, "every 4th even")
        
        # 9. First 2 even heads
        print("\n9. FIRST 2 EVEN HEADS")
        first_2_even = all_even_heads[:2]  # [0,2]
        results['first_2_even'] = self.test_head_subset(first_2_even, n_trials, "first 2 even")
        
        # 10. Last 2 even heads
        print("\n10. LAST 2 EVEN HEADS")
        last_2_even = all_even_heads[-2:]  # [28,30]
        results['last_2_even'] = self.test_head_subset(last_2_even, n_trials, "last 2 even")
        
        # 11. Extremes (2 heads)
        print("\n11. EXTREME EVEN HEADS (first and last)")
        extreme_even = [all_even_heads[0], all_even_heads[-1]]  # [0,30]
        results['extreme_even'] = self.test_head_subset(extreme_even, n_trials, "extreme even")
        
        # 12. Single even heads (test a few)
        print("\n12. SINGLE EVEN HEADS")
        for head in [0, 14, 30]:
            print(f"\n    Testing single head {head}:")
            results[f'single_{head}'] = self.test_head_subset([head], n_trials, f"single head {head}")
        
        return results
    
    def print_summary(self, results: Dict):
        """Print summary organized by number of heads"""
        print("\n" + "="*70)
        print("SUMMARY: MINIMAL EVEN HEAD SUBSET DISCOVERY")
        print("="*70)
        
        # Group by number of heads
        by_n_heads = {}
        for key, result in results.items():
            n = result['n_heads']
            if n not in by_n_heads:
                by_n_heads[n] = []
            by_n_heads[n].append((key, result))
        
        # Print organized by size
        for n_heads in sorted(by_n_heads.keys(), reverse=True):
            print(f"\n{n_heads} HEADS:")
            print("-" * 50)
            for key, result in by_n_heads[n_heads]:
                print(f"{key:20s}: {result['success_rate']:6.1%} [{result['ci_lower']:5.1%}, {result['ci_upper']:5.1%}]")
        
        # Find threshold
        print("\n" + "="*70)
        print("KEY FINDINGS:")
        
        working = [k for k, r in results.items() if r['success_rate'] > 0.8]
        partial = [k for k, r in results.items() if 0.2 < r['success_rate'] <= 0.8]
        failing = [k for k, r in results.items() if r['success_rate'] <= 0.2]
        
        if working:
            min_working = min([results[k]['n_heads'] for k in working])
            print(f"✅ Minimum heads for >80% success: {min_working}")
            print(f"   Working subsets: {', '.join(working)}")
        
        if partial:
            print(f"⚠️ Partially working (20-80%): {', '.join(partial)}")
        
        if failing:
            max_failing = max([results[k]['n_heads'] for k in failing if results[k]['n_heads'] > 1])
            print(f"❌ Maximum heads that still fail: {max_failing}")

def main():
    start_time = time.time()
    
    tester = MinimalEvenHeadsTest(device="cuda")
    results = tester.run_minimal_tests(n_trials=50)
    tester.print_summary(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"minimal_even_heads_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    total_time = time.time() - start_time
    print(f"\n✅ Testing complete in {total_time:.1f} seconds")
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    main()