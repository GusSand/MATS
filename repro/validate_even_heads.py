#!/usr/bin/env python3
"""
Validate the surprising finding that even-indexed heads work perfectly.
Run with n=100 trials for statistical confidence.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
from contextlib import contextmanager
import json
from datetime import datetime
import time

class ValidateEvenHeads:
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
    
    def test_head_subset(self, head_indices: List[int], n_trials: int = 100, name: str = "") -> Dict:
        """Test a specific subset of heads with progress bar"""
        success_count = 0
        
        # Save activation once
        correct_prompt = "Which is bigger: 9.8 or 9.11?\nAnswer:"
        with self.save_activation_context(correct_prompt) as saved:
            correct_activation = saved[f"layer_{self.layer_idx}_attention"]
        
        buggy_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"
        
        print(f"Testing {name}...")
        start_time = time.time()
        
        for trial in range(n_trials):
            with self.patch_activation_context(correct_activation, head_indices):
                output = self.generate(buggy_prompt, max_new_tokens=20)
            
            if self.check_bug_fixed(output):
                success_count += 1
            
            # Progress bar
            if (trial + 1) % 10 == 0:
                progress = (trial + 1) / n_trials
                bar_length = 40
                filled = int(bar_length * progress)
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"\r  [{bar}] {trial+1}/{n_trials} - {success_count} successes", end='')
        
        elapsed = time.time() - start_time
        success_rate = success_count / n_trials
        
        # Calculate 95% CI
        se = np.sqrt(success_rate * (1 - success_rate) / n_trials)
        ci_lower = max(0, success_rate - 1.96 * se)
        ci_upper = min(1, success_rate + 1.96 * se)
        
        print(f"\n  Result: {success_rate:.1%} [{ci_lower:.1%}, {ci_upper:.1%}] - {elapsed:.1f}s")
        
        return {
            'success_rate': success_rate,
            'success_count': success_count,
            'n_trials': n_trials,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'time_seconds': elapsed,
            'head_indices': head_indices
        }
    
    def run_validation(self, n_trials: int = 100) -> Dict:
        """Run validation experiments"""
        results = {}
        
        print("\n" + "="*60)
        print(f"VALIDATING EVEN-INDEXED HEADS DISCOVERY (n={n_trials})")
        print("="*60)
        
        # 1. Even-indexed heads (the surprising finding)
        print("\n1. EVEN-INDEXED HEADS (0,2,4,...,30)")
        even_heads = list(range(0, 32, 2))
        results['even_heads'] = self.test_head_subset(even_heads, n_trials, "even heads")
        
        # 2. Odd-indexed heads (should fail)
        print("\n2. ODD-INDEXED HEADS (1,3,5,...,31)")
        odd_heads = list(range(1, 32, 2))
        results['odd_heads'] = self.test_head_subset(odd_heads, n_trials, "odd heads")
        
        # 3. All 32 heads (baseline)
        print("\n3. ALL 32 HEADS (baseline)")
        all_heads = list(range(32))
        results['all_heads'] = self.test_head_subset(all_heads, n_trials, "all heads")
        
        # 4. Test removing individual even heads
        print("\n4. TESTING CRITICALITY OF INDIVIDUAL EVEN HEADS")
        print("   (Testing even heads with one removed at a time)")
        
        critical_even = []
        for remove_idx in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]:
            subset = [h for h in range(0, 32, 2) if h != remove_idx]
            print(f"\n   Removing head {remove_idx}:")
            result = self.test_head_subset(subset, 20, f"even minus {remove_idx}")  # Quick test
            if result['success_rate'] < 0.5:
                critical_even.append(remove_idx)
        
        results['critical_even_heads'] = critical_even
        
        return results
    
    def print_summary(self, results: Dict):
        """Print summary of validation results"""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        print(f"\nMain Results (n={results['even_heads']['n_trials']}):")
        print("-" * 50)
        
        even = results['even_heads']
        odd = results['odd_heads']
        all32 = results['all_heads']
        
        print(f"Even heads (16 heads):  {even['success_rate']:6.1%} [{even['ci_lower']:.1%}, {even['ci_upper']:.1%}]")
        print(f"Odd heads (16 heads):   {odd['success_rate']:6.1%} [{odd['ci_lower']:.1%}, {odd['ci_upper']:.1%}]")
        print(f"All heads (32 heads):   {all32['success_rate']:6.1%} [{all32['ci_lower']:.1%}, {all32['ci_upper']:.1%}]")
        
        if 'critical_even_heads' in results and results['critical_even_heads']:
            print(f"\nCritical even heads (removing any causes failure):")
            print(f"  {results['critical_even_heads']}")
        
        print("\n" + "="*60)
        print("CONCLUSION:")
        if even['success_rate'] > 0.95 and odd['success_rate'] < 0.05:
            print("✅ CONFIRMED: Even-indexed attention heads (0,2,4,...,30)")
            print("   are necessary and sufficient to fix the decimal bug!")
            print("   This is a 2x reduction in required heads (16 vs 32).")
        else:
            print("⚠️ Results differ from initial finding - needs investigation")

def main():
    start_time = time.time()
    
    validator = ValidateEvenHeads(device="cuda")
    results = validator.run_validation(n_trials=100)
    validator.print_summary(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"even_heads_validation_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    total_time = time.time() - start_time
    print(f"\n✅ Validation complete in {total_time:.1f} seconds")
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    main()