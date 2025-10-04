#!/usr/bin/env python3
"""
Bidirectional Attention Output Patching Experiment
Based on the working implementation in layer25/attention_control_experiment.py
Tests both forward patching (fix bug) and reverse patching (induce bug)
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import os
from contextlib import contextmanager
from datetime import datetime
import json

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class BidirectionalPatcher:
    """Test both forward and reverse attention output patching at Layer 10"""
    
    def __init__(self):
        print("Loading model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.model.eval()
        
        # Storage for activations
        self.saved_activations = {}
        self.hooks = []
        
        print("Model loaded successfully!")
        
    def get_attention_module(self, layer_idx: int) -> nn.Module:
        """Get the attention module for a specific layer"""
        layer = self.model.model.layers[layer_idx]
        if hasattr(layer, 'self_attn'):
            return layer.self_attn
        else:
            raise AttributeError(f"Layer {layer_idx} does not have self_attn module")
    
    def save_activation_hook(self, layer_idx: int):
        """Create a hook that saves the activation"""
        def hook_fn(module, input, output):
            # Handle tuple output structure
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            key = f"layer_{layer_idx}"
            self.saved_activations[key] = hidden_states.detach().clone()
            return output
        return hook_fn
    
    def patch_activation_hook(self, layer_idx: int, saved_activation: torch.Tensor):
        """Create a hook that patches in a saved activation"""
        def hook_fn(module, input, output):
            # Handle tuple output structure
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Get dimensions
            batch_size, seq_len, hidden_dim = hidden_states.shape
            saved_seq_len = saved_activation.shape[1]
            
            # Patch the overlapping sequence positions
            min_seq_len = min(seq_len, saved_seq_len)
            
            # Clone to avoid in-place modification
            new_hidden = hidden_states.clone()
            new_hidden[:, :min_seq_len, :] = saved_activation[:, :min_seq_len, :]
            
            # Return modified output maintaining tuple structure
            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden
        
        return hook_fn
    
    def clear_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.saved_activations = {}
    
    @contextmanager
    def save_attention_context(self, prompt: str, layer_idx: int = 10):
        """Context manager to save attention output at Layer 10"""
        try:
            # Register hook on attention module
            module = self.get_attention_module(layer_idx)
            hook = module.register_forward_hook(self.save_activation_hook(layer_idx))
            self.hooks.append(hook)
            
            # Run forward pass
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                _ = self.model(**inputs)
            
            yield self.saved_activations.copy()
            
        finally:
            self.clear_hooks()
    
    @contextmanager
    def patch_attention_context(self, saved_activations: dict, layer_idx: int = 10):
        """Context manager to patch attention during generation"""
        try:
            # Register patching hook
            for key, activation in saved_activations.items():
                module = self.get_attention_module(layer_idx)
                hook = module.register_forward_hook(
                    self.patch_activation_hook(layer_idx, activation)
                )
                self.hooks.append(hook)
            
            yield
            
        finally:
            self.clear_hooks()
    
    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Generate text from prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,  # Deterministic
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated[len(prompt):]  # Return only generated part
    
    def classify_output(self, output: str) -> dict:
        """Classify the output as correct, bug, or unclear"""
        output_lower = output.lower()
        
        # Check for 9.8 > 9.11 (correct)
        correct = ("9.8" in output and 
                  any(phrase in output_lower for phrase in [
                      "9.8 is bigger", "9.8 is larger", "9.8 is greater",
                      "9.8 is the bigger", "9.8 is the larger"
                  ]))
        
        # Check for 9.11 > 9.8 (bug)
        bug = ("9.11" in output and 
               any(phrase in output_lower for phrase in [
                   "9.11 is bigger", "9.11 is larger", "9.11 is greater",
                   "9.11 is the bigger", "9.11 is the larger"
               ]))
        
        # Check for gibberish (repeated patterns)
        gibberish = ("Q: W" in output or 
                    "Which Which" in output or
                    output.count("9.8") > 5 or
                    output.count("9.11") > 5)
        
        return {
            'is_correct': correct and not bug,
            'shows_bug': bug and not correct,
            'is_gibberish': gibberish,
            'is_unclear': not (correct or bug or gibberish)
        }
    
    def run_experiment(self, n_trials: int = 5):
        """Run the complete bidirectional patching experiment"""
        
        # Define prompts
        correct_prompt = "Which is bigger: 9.8 or 9.11?\nAnswer:"
        buggy_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"
        
        results = {
            'baselines': {},
            'forward_patch': [],
            'reverse_patch': []
        }
        
        print("\n" + "="*70)
        print("BIDIRECTIONAL ATTENTION OUTPUT PATCHING EXPERIMENT")
        print("="*70)
        print("Testing Layer 10 attention output patching")
        
        # Step 1: Test baselines
        print("\n📊 TESTING BASELINES")
        print("-" * 50)
        
        print("\n1. Correct format baseline (should be correct):")
        for i in range(n_trials):
            output = self.generate(correct_prompt)
            classification = self.classify_output(output)
            
            if i == 0:
                print(f"  Sample: {output[:60]}...")
                symbol = "✅" if classification['is_correct'] else "❌" if classification['shows_bug'] else "❓"
                print(f"  {symbol} {classification}")
            
            if 'correct_baseline' not in results['baselines']:
                results['baselines']['correct_baseline'] = []
            results['baselines']['correct_baseline'].append(classification)
        
        print("\n2. Buggy format baseline (should show bug):")
        for i in range(n_trials):
            output = self.generate(buggy_prompt)
            classification = self.classify_output(output)
            
            if i == 0:
                print(f"  Sample: {output[:60]}...")
                symbol = "✅" if classification['is_correct'] else "❌" if classification['shows_bug'] else "❓"
                print(f"  {symbol} {classification}")
            
            if 'buggy_baseline' not in results['baselines']:
                results['baselines']['buggy_baseline'] = []
            results['baselines']['buggy_baseline'].append(classification)
        
        # Step 2: Save attention outputs
        print("\n🔬 SAVING ATTENTION OUTPUTS")
        print("-" * 50)
        
        print("\nSaving CORRECT format attention at Layer 10...")
        with self.save_attention_context(correct_prompt, layer_idx=10) as correct_attention:
            saved_correct = correct_attention.copy()
            print(f"  ✅ Saved shape: {saved_correct['layer_10'].shape}")
        
        print("\nSaving BUGGY format attention at Layer 10...")
        with self.save_attention_context(buggy_prompt, layer_idx=10) as buggy_attention:
            saved_buggy = buggy_attention.copy()
            print(f"  ✅ Saved shape: {saved_buggy['layer_10'].shape}")
        
        # Step 3: Forward patching (fix the bug)
        print("\n🔧 FORWARD PATCHING: Buggy format + Correct attention")
        print("-" * 50)
        print("Should FIX the bug if attention is causal")
        
        for i in range(n_trials):
            with self.patch_attention_context(saved_correct, layer_idx=10):
                output = self.generate(buggy_prompt)
                classification = self.classify_output(output)
                
                if i == 0:
                    print(f"  Sample: {output[:60]}...")
                    symbol = "✅" if classification['is_correct'] else "❌" if classification['shows_bug'] else "💥" if classification['is_gibberish'] else "❓"
                    print(f"  {symbol} {classification}")
                
                results['forward_patch'].append(classification)
        
        # Step 4: Reverse patching (induce the bug)
        print("\n🔄 REVERSE PATCHING: Correct format + Buggy attention")
        print("-" * 50)
        print("Should INDUCE the bug if attention is causal")
        
        for i in range(n_trials):
            with self.patch_attention_context(saved_buggy, layer_idx=10):
                output = self.generate(correct_prompt)
                classification = self.classify_output(output)
                
                if i == 0:
                    print(f"  Sample: {output[:60]}...")
                    symbol = "✅" if classification['is_correct'] else "❌" if classification['shows_bug'] else "💥" if classification['is_gibberish'] else "❓"
                    print(f"  {symbol} {classification}")
                
                results['reverse_patch'].append(classification)
        
        return results
    
    def analyze_results(self, results: dict):
        """Analyze and print results summary"""
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        
        def calculate_rates(classifications):
            n = len(classifications)
            if n == 0:
                return {'correct': 0, 'bug': 0, 'gibberish': 0, 'unclear': 0}
            
            return {
                'correct': sum(c['is_correct'] for c in classifications) / n * 100,
                'bug': sum(c['shows_bug'] for c in classifications) / n * 100,
                'gibberish': sum(c['is_gibberish'] for c in classifications) / n * 100,
                'unclear': sum(c['is_unclear'] for c in classifications) / n * 100
            }
        
        # Calculate rates for each condition
        conditions = {
            'Correct Baseline': results['baselines'].get('correct_baseline', []),
            'Buggy Baseline': results['baselines'].get('buggy_baseline', []),
            'Forward Patch (Bug→Fix)': results['forward_patch'],
            'Reverse Patch (Fix→Bug)': results['reverse_patch']
        }
        
        print("\n| Condition | Correct% | Bug% | Gibberish% | Unclear% |")
        print("|-----------|----------|------|------------|----------|")
        
        for name, classifications in conditions.items():
            rates = calculate_rates(classifications)
            print(f"| {name:23} | {rates['correct']:7.1f}% | {rates['bug']:4.1f}% | {rates['gibberish']:9.1f}% | {rates['unclear']:7.1f}% |")
        
        # Key findings
        print("\n" + "="*70)
        print("KEY FINDINGS")
        print("="*70)
        
        forward_rates = calculate_rates(results['forward_patch'])
        reverse_rates = calculate_rates(results['reverse_patch'])
        
        if forward_rates['correct'] >= 80:
            print("✅ FORWARD PATCHING SUCCESS: Correct attention FIXES the bug!")
            print(f"   Success rate: {forward_rates['correct']:.1f}%")
        elif forward_rates['gibberish'] >= 80:
            print("💥 FORWARD PATCHING: Produces gibberish")
            print("   Suggests dimension mismatch or incompatibility")
        else:
            print("❌ FORWARD PATCHING: Does not reliably fix the bug")
        
        if reverse_rates['bug'] >= 80:
            print("\n✅ REVERSE PATCHING SUCCESS: Buggy attention INDUCES the bug!")
            print(f"   Bug induction rate: {reverse_rates['bug']:.1f}%")
        elif reverse_rates['gibberish'] >= 80:
            print("\n💥 REVERSE PATCHING: Produces gibberish")
            print("   Suggests dimension mismatch or incompatibility")
        else:
            print("\n❌ REVERSE PATCHING: Does not reliably induce the bug")
        
        # Final verdict
        if forward_rates['correct'] >= 80 and reverse_rates['bug'] >= 80:
            print("\n🎉 BIDIRECTIONAL CAUSALITY CONFIRMED!")
            print("Layer 10 attention output is definitively causal for the decimal bug.")
        elif forward_rates['correct'] >= 80 or reverse_rates['bug'] >= 80:
            print("\n⚠️ PARTIAL CAUSALITY")
            print("Layer 10 attention output shows some causal influence.")
        else:
            print("\n❓ CAUSALITY NOT ESTABLISHED")
            print("Results do not support strong causal relationship.")
        
        return conditions

def main():
    # Initialize
    patcher = BidirectionalPatcher()
    
    # Run experiment
    results = patcher.run_experiment(n_trials=5)
    
    # Analyze
    conditions = patcher.analyze_results(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(f'bidirectional_results_{timestamp}.json', 'w') as f:
        # Convert to serializable format
        serializable_results = {
            'timestamp': timestamp,
            'baselines': results['baselines'],
            'forward_patch': results['forward_patch'],
            'reverse_patch': results['reverse_patch']
        }
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n✅ Results saved to bidirectional_results_{timestamp}.json")
    
    return results

if __name__ == "__main__":
    results = main()