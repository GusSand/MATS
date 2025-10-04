#!/usr/bin/env python3
"""
Quick test of structured head subsets with fewer trials for speed.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
from contextlib import contextmanager
import json
from datetime import datetime
import random

class QuickHeadAnalysis:
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
        """Get attention module for a specific layer"""
        return self.model.model.layers[layer_idx].self_attn
    
    def save_activation_hook(self, key: str):
        """Create hook to save activation"""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self.saved_activations[key] = hidden_states.detach().cpu()
        return hook_fn
    
    def selective_patch_hook(self, saved_activation: torch.Tensor, head_indices: List[int]):
        """Create hook to patch only specific attention heads"""
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
        """Context manager to save activation from a prompt"""
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
        """Context manager to patch specific attention heads during generation"""
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
        """Generate text from prompt"""
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
        """Check if the bug is fixed (model says 9.8 > 9.11)"""
        output_lower = output.lower()
        
        correct_patterns = ["9.8 is bigger", "9.8 is larger", "9.8 is greater"]
        bug_patterns = ["9.11 is bigger", "9.11 is larger", "9.11 is greater"]
        
        has_correct = any(pattern in output_lower for pattern in correct_patterns)
        has_bug = any(pattern in output_lower for pattern in bug_patterns)
        
        return has_correct and not has_bug
    
    def test_head_subset(self, head_indices: List[int], n_trials: int = 20) -> Dict:
        """Test a specific subset of heads"""
        success_count = 0
        
        # Save activation once from correct format
        correct_prompt = "Which is bigger: 9.8 or 9.11?\nAnswer:"
        with self.save_activation_context(correct_prompt) as saved:
            correct_activation = saved[f"layer_{self.layer_idx}_attention"]
        
        # Test multiple times with buggy format
        buggy_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"
        
        for trial in range(n_trials):
            with self.patch_activation_context(correct_activation, head_indices):
                output = self.generate(buggy_prompt, max_new_tokens=20)
            
            if self.check_bug_fixed(output):
                success_count += 1
            
            # Print progress
            if (trial + 1) % 5 == 0:
                print(f"    Trial {trial + 1}/{n_trials}: {success_count}/{trial + 1} successes")
        
        success_rate = success_count / n_trials
        
        return {
            'success_rate': success_rate,
            'success_count': success_count,
            'n_trials': n_trials,
            'head_indices': head_indices
        }
    
    def run_quick_experiments(self) -> Dict:
        """Run quick structured subset experiments"""
        results = {}
        n_trials = 20  # Quick test with 20 trials
        
        print("\n" + "="*60)
        print("QUICK STRUCTURED HEAD SUBSET EXPERIMENTS")
        print(f"Testing with n={n_trials} trials per subset")
        print("="*60)
        
        # 1. First 16 heads
        print("\n1. Testing FIRST 16 heads (0-15)...")
        first_16 = list(range(16))
        results['first_16'] = self.test_head_subset(first_16, n_trials)
        print(f"   Result: {results['first_16']['success_rate']:.1%}")
        
        # 2. Last 16 heads
        print("\n2. Testing LAST 16 heads (16-31)...")
        last_16 = list(range(16, 32))
        results['last_16'] = self.test_head_subset(last_16, n_trials)
        print(f"   Result: {results['last_16']['success_rate']:.1%}")
        
        # 3. Every other head (even)
        print("\n3. Testing EVERY OTHER head (even: 0,2,4,...,30)...")
        every_other_even = list(range(0, 32, 2))
        results['every_other_even'] = self.test_head_subset(every_other_even, n_trials)
        print(f"   Result: {results['every_other_even']['success_rate']:.1%}")
        
        # 4. Every other head (odd)
        print("\n4. Testing EVERY OTHER head (odd: 1,3,5,...,31)...")
        every_other_odd = list(range(1, 32, 2))
        results['every_other_odd'] = self.test_head_subset(every_other_odd, n_trials)
        print(f"   Result: {results['every_other_odd']['success_rate']:.1%}")
        
        # 5. Random 16 heads (3 seeds for quick test)
        print("\n5. Testing RANDOM 16 heads (3 different seeds)...")
        results['random_16'] = []
        
        for seed in range(3):
            print(f"\n   Seed {seed}:")
            random.seed(seed)
            random_16 = sorted(random.sample(range(32), 16))
            print(f"   Heads: {random_16[:8]}...{random_16[-4:]}")  # Show subset
            seed_result = self.test_head_subset(random_16, n_trials)
            seed_result['seed'] = seed
            results['random_16'].append(seed_result)
            print(f"   Result: {seed_result['success_rate']:.1%}")
        
        # 6. Baseline: All 32 heads
        print("\n6. Testing ALL 32 heads (baseline)...")
        all_heads = list(range(32))
        results['all_32'] = self.test_head_subset(all_heads, n_trials)
        print(f"   Result: {results['all_32']['success_rate']:.1%}")
        
        return results
    
    def print_summary(self, results: Dict):
        """Print summary of results"""
        print("\n" + "="*60)
        print("SUMMARY OF QUICK STRUCTURED HEAD EXPERIMENTS")
        print("="*60)
        
        print(f"\nLayer 10 Attention Head Subsets (16 heads each):")
        print("-" * 50)
        
        print(f"First 16 heads (0-15):     {results['first_16']['success_rate']:6.1%}")
        print(f"Last 16 heads (16-31):     {results['last_16']['success_rate']:6.1%}")
        print(f"Every other (even):        {results['every_other_even']['success_rate']:6.1%}")
        print(f"Every other (odd):         {results['every_other_odd']['success_rate']:6.1%}")
        
        if results['random_16']:
            random_rates = [r['success_rate'] for r in results['random_16']]
            print(f"Random 16 (mean):          {np.mean(random_rates):6.1%}")
        
        print("-" * 50)
        print(f"All 32 heads (baseline):   {results['all_32']['success_rate']:6.1%}")
        
        print("\nKey Finding:")
        if results['all_32']['success_rate'] > 0.8:
            if all(results[k]['success_rate'] < 0.2 for k in ['first_16', 'last_16', 'every_other_even', 'every_other_odd']):
                print("✅ ALL 32 heads are required together - no 16-head subset works!")
            else:
                working = [k for k in ['first_16', 'last_16', 'every_other_even', 'every_other_odd'] 
                          if results[k]['success_rate'] > 0.5]
                if working:
                    print(f"⚠️ Some 16-head subsets partially work: {', '.join(working)}")
                else:
                    print("✅ No 16-head subset achieves >50% success")

def main():
    print("Starting quick structured head subset experiments...")
    print("This will test different combinations of 16 attention heads")
    
    analyzer = QuickHeadAnalysis(device="cuda")
    results = analyzer.run_quick_experiments()
    analyzer.print_summary(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"quick_structured_heads_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {filename}")

if __name__ == "__main__":
    main()