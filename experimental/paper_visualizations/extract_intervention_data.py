#!/usr/bin/env python3
"""
Extract intervention success rates data for surgical precision visualization
Tests different layers and components to find what works
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import warnings
import os
from contextlib import contextmanager

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

class InterventionTester:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        print("Loading model for intervention testing...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.model.eval()
        
        self.saved_activations = {}
        self.hooks = []
        
    def get_component(self, layer_idx, component_type):
        """Get specific component of a layer"""
        layer = self.model.model.layers[layer_idx]
        
        if component_type == 'attention':
            return layer.self_attn
        elif component_type == 'mlp':
            return layer.mlp
        elif component_type == 'full':
            return layer
        else:
            raise ValueError(f"Unknown component type: {component_type}")
    
    def save_activation_hook(self, key):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self.saved_activations[key] = hidden_states.detach().clone()
            return output
        return hook_fn
    
    def patch_activation_hook(self, saved_activation):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Patch the activation
            batch_size, seq_len, hidden_dim = hidden_states.shape
            saved_seq_len = saved_activation.shape[1]
            min_seq_len = min(seq_len, saved_seq_len)
            
            new_hidden = hidden_states.clone()
            new_hidden[:, :min_seq_len, :] = saved_activation[:, :min_seq_len, :]
            
            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden
        return hook_fn
    
    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.saved_activations = {}
    
    @contextmanager
    def save_context(self, prompt, layer_idx, component_type):
        """Save activations from a specific component"""
        try:
            module = self.get_component(layer_idx, component_type)
            hook = module.register_forward_hook(
                self.save_activation_hook(f"{layer_idx}_{component_type}")
            )
            self.hooks.append(hook)
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                _ = self.model(**inputs)
            
            yield self.saved_activations.copy()
        finally:
            self.clear_hooks()
    
    @contextmanager
    def patch_context(self, saved_activations, layer_idx, component_type):
        """Patch activations during generation"""
        try:
            for key, activation in saved_activations.items():
                module = self.get_component(layer_idx, component_type)
                hook = module.register_forward_hook(
                    self.patch_activation_hook(activation)
                )
                self.hooks.append(hook)
            yield
        finally:
            self.clear_hooks()
    
    def test_intervention(self, source_prompt, target_prompt, layer_idx, component_type, n_trials=10):
        """Test intervention success rate"""
        successes = 0
        
        # Save activations from correct format
        with self.save_context(source_prompt, layer_idx, component_type) as saved_acts:
            for trial in range(n_trials):
                # Generate with patched activations
                with self.patch_context(saved_acts, layer_idx, component_type):
                    inputs = self.tokenizer(target_prompt, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=30,
                            temperature=0.0,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id
                        )
                    
                    response = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    )
                    
                    # Check if intervention fixed the bug
                    if "9.8" in response and ("bigger" in response.lower() or "larger" in response.lower()):
                        if "9.11" not in response or response.index("9.8") < response.index("9.11"):
                            successes += 1
        
        return (successes / n_trials) * 100
    
    def run_full_experiment(self):
        """Test all layer-component combinations"""
        # Define prompts
        correct_prompt = "Which is bigger: 9.8 or 9.11? Answer:"
        buggy_prompt = "Q: Which is bigger: 9.8 or 9.11? A:"
        
        # Test layers and components
        layers = [8, 9, 10, 11, 12]
        components = ['attention', 'mlp', 'full']
        
        results = np.zeros((len(layers), len(components)))
        
        for i, layer_idx in enumerate(layers):
            for j, component in enumerate(components):
                print(f"Testing Layer {layer_idx}, Component: {component}")
                
                success_rate = self.test_intervention(
                    correct_prompt, buggy_prompt, layer_idx, component, n_trials=10
                )
                
                results[i, j] = success_rate
                print(f"  Success rate: {success_rate:.1f}%")
        
        return {
            'layers': layers,
            'components': components,
            'results': results.tolist(),
            'metadata': {
                'source_prompt': correct_prompt,
                'target_prompt': buggy_prompt,
                'n_trials_per_config': 10
            }
        }

def main():
    tester = InterventionTester()
    
    print("Running intervention experiments...")
    print("="*50)
    
    results = tester.run_full_experiment()
    
    # Save results
    with open('intervention_success_rates.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*50)
    print("INTERVENTION SUCCESS RATES")
    print("="*50)
    
    # Print results table
    print("\n" + " "*10, end="")
    for comp in results['components']:
        print(f"{comp:^12}", end="")
    print()
    
    for i, layer in enumerate(results['layers']):
        print(f"Layer {layer:2d}: ", end="")
        for j, comp in enumerate(results['components']):
            rate = results['results'][i][j]
            if rate > 95:
                print(f"  ✓ {rate:5.1f}%", end="")
            elif rate > 20:
                print(f"  △ {rate:5.1f}%", end="")
            else:
                print(f"  ✗ {rate:5.1f}%", end="")
        print()
    
    print("\nData saved to intervention_success_rates.json")
    
    return results

if __name__ == "__main__":
    results = main()