#!/usr/bin/env python3
"""
Test structured subsets of attention heads for Layer 10 intervention.
Tests: first 16, last 16, every other, and random 16 heads.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional, Tuple
from contextlib import contextmanager
import json
from datetime import datetime
import random

class StructuredHeadAnalysis:
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
        
        self.n_heads = 32  # Llama-3.1-8B has 32 attention heads
        self.layer_idx = 10  # Focus on Layer 10
        
        self.saved_activations = {}
        self.hooks = []
    
    def get_attention_module(self, layer_idx: int):
        """Get attention module for a specific layer"""
        return self.model.model.layers[layer_idx].self_attn
    
    def save_activation_hook(self, key: str):
        """Create hook to save activation"""
        def hook_fn(module, input, output):
            # For attention modules, output is (hidden_states, attn_weights, past_key_values)
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
            
            # Calculate dimensions per head
            batch_size, seq_len, hidden_size = hidden_states.shape
            head_dim = hidden_size // self.n_heads
            
            # Reshape to separate heads
            hidden_states_reshaped = hidden_states.view(batch_size, seq_len, self.n_heads, head_dim)
            saved_reshaped = saved_activation.to(hidden_states.device).view(batch_size, -1, self.n_heads, head_dim)
            
            # Create new tensor for patched output
            new_hidden = hidden_states_reshaped.clone()
            
            # Patch only specified heads
            min_seq_len = min(seq_len, saved_reshaped.shape[1])
            for head_idx in head_indices:
                new_hidden[:, :min_seq_len, head_idx, :] = saved_reshaped[:, :min_seq_len, head_idx, :]
            
            # Reshape back
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
    
    def generate(self, prompt: str, max_new_tokens: int = 30) -> str:
        """Generate text from prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
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
        
        # Check for correct answer
        correct_patterns = [
            "9.8 is bigger",
            "9.8 is larger",
            "9.8 is greater",
            "9.8"  # Sometimes just outputs the number
        ]
        
        # Check for bug
        bug_patterns = [
            "9.11 is bigger",
            "9.11 is larger", 
            "9.11 is greater",
            "9.11"
        ]
        
        has_correct = any(pattern in output_lower for pattern in correct_patterns)
        has_bug = any(pattern in output_lower for pattern in bug_patterns)
        
        # Fixed if has correct answer and no bug
        return has_correct and not has_bug
    
    def test_head_subset(self, head_indices: List[int], n_trials: int = 100) -> Dict:
        """Test a specific subset of heads"""
        success_count = 0
        
        for _ in range(n_trials):
            # Save activation from correct format
            correct_prompt = "Which is bigger: 9.8 or 9.11?\nAnswer:"
            with self.save_activation_context(correct_prompt) as saved:
                correct_activation = saved[f"layer_{self.layer_idx}_attention"]
            
            # Generate with patched activation in buggy format
            buggy_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"
            with self.patch_activation_context(correct_activation, head_indices):
                output = self.generate(buggy_prompt, max_new_tokens=30)
            
            if self.check_bug_fixed(output):
                success_count += 1
        
        success_rate = success_count / n_trials
        
        # Calculate confidence interval
        se = np.sqrt(success_rate * (1 - success_rate) / n_trials)
        ci_lower = max(0, success_rate - 1.96 * se)
        ci_upper = min(1, success_rate + 1.96 * se)
        
        return {
            'success_rate': success_rate,
            'success_count': success_count,
            'n_trials': n_trials,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'head_indices': head_indices
        }
    
    def run_structured_experiments(self, n_trials: int = 100, n_random_seeds: int = 5) -> Dict:
        """Run all structured subset experiments"""
        results = {}
        
        # 1. First 16 heads
        print("\n" + "="*60)
        print("Testing FIRST 16 heads (0-15)...")
        first_16 = list(range(16))
        results['first_16'] = self.test_head_subset(first_16, n_trials)
        print(f"Success rate: {results['first_16']['success_rate']:.1%}")
        
        # 2. Last 16 heads
        print("\n" + "="*60)
        print("Testing LAST 16 heads (16-31)...")
        last_16 = list(range(16, 32))
        results['last_16'] = self.test_head_subset(last_16, n_trials)
        print(f"Success rate: {results['last_16']['success_rate']:.1%}")
        
        # 3. Every other head (even indices)
        print("\n" + "="*60)
        print("Testing EVERY OTHER head (even indices: 0,2,4,...,30)...")
        every_other_even = list(range(0, 32, 2))
        results['every_other_even'] = self.test_head_subset(every_other_even, n_trials)
        print(f"Success rate: {results['every_other_even']['success_rate']:.1%}")
        
        # 4. Every other head (odd indices)
        print("\n" + "="*60)
        print("Testing EVERY OTHER head (odd indices: 1,3,5,...,31)...")
        every_other_odd = list(range(1, 32, 2))
        results['every_other_odd'] = self.test_head_subset(every_other_odd, n_trials)
        print(f"Success rate: {results['every_other_odd']['success_rate']:.1%}")
        
        # 5. Random 16 heads (multiple seeds)
        print("\n" + "="*60)
        print(f"Testing RANDOM 16 heads ({n_random_seeds} different seeds)...")
        results['random_16'] = []
        
        for seed in range(n_random_seeds):
            print(f"\n  Seed {seed}:")
            random.seed(seed)
            random_16 = sorted(random.sample(range(32), 16))
            seed_result = self.test_head_subset(random_16, n_trials)
            seed_result['seed'] = seed
            results['random_16'].append(seed_result)
            print(f"    Heads: {random_16}")
            print(f"    Success rate: {seed_result['success_rate']:.1%}")
        
        # Calculate average for random subsets
        avg_random = np.mean([r['success_rate'] for r in results['random_16']])
        std_random = np.std([r['success_rate'] for r in results['random_16']])
        results['random_16_summary'] = {
            'mean_success_rate': avg_random,
            'std_success_rate': std_random,
            'n_seeds': n_random_seeds
        }
        
        # 6. Baseline: All 32 heads
        print("\n" + "="*60)
        print("Testing ALL 32 heads (baseline)...")
        all_heads = list(range(32))
        results['all_32'] = self.test_head_subset(all_heads, n_trials)
        print(f"Success rate: {results['all_32']['success_rate']:.1%}")
        
        return results
    
    def save_results(self, results: Dict, filename: str = None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"structured_head_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {filename}")
        return filename
    
    def print_summary(self, results: Dict):
        """Print summary of results"""
        print("\n" + "="*60)
        print("SUMMARY OF STRUCTURED HEAD SUBSET EXPERIMENTS")
        print("="*60)
        
        print(f"\nLayer 10 Attention Head Subsets (16 heads each):")
        print("-" * 50)
        
        # Main structured subsets
        print(f"First 16 heads (0-15):     {results['first_16']['success_rate']:6.1%} "
              f"[{results['first_16']['ci_lower']:.1%}, {results['first_16']['ci_upper']:.1%}]")
        print(f"Last 16 heads (16-31):     {results['last_16']['success_rate']:6.1%} "
              f"[{results['last_16']['ci_lower']:.1%}, {results['last_16']['ci_upper']:.1%}]")
        print(f"Every other (even):        {results['every_other_even']['success_rate']:6.1%} "
              f"[{results['every_other_even']['ci_lower']:.1%}, {results['every_other_even']['ci_upper']:.1%}]")
        print(f"Every other (odd):         {results['every_other_odd']['success_rate']:6.1%} "
              f"[{results['every_other_odd']['ci_lower']:.1%}, {results['every_other_odd']['ci_upper']:.1%}]")
        
        # Random subsets
        if 'random_16_summary' in results:
            print(f"Random 16 (mean ± std):    {results['random_16_summary']['mean_success_rate']:6.1%} "
                  f"± {results['random_16_summary']['std_success_rate']:.1%}")
        
        # Baseline
        print("-" * 50)
        print(f"All 32 heads (baseline):   {results['all_32']['success_rate']:6.1%} "
              f"[{results['all_32']['ci_lower']:.1%}, {results['all_32']['ci_upper']:.1%}]")
        
        print("\n" + "="*60)

def main():
    # Initialize analyzer
    analyzer = StructuredHeadAnalysis(device="cuda")
    
    # Run experiments
    print("\nStarting structured head subset experiments...")
    print("This will test different combinations of 16 attention heads")
    
    # Use 100 trials for good statistical power
    results = analyzer.run_structured_experiments(n_trials=100, n_random_seeds=5)
    
    # Print summary
    analyzer.print_summary(results)
    
    # Save results
    filename = analyzer.save_results(results)
    
    print(f"\n✅ Experiments complete! Results saved to {filename}")

if __name__ == "__main__":
    main()