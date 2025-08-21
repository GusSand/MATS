#!/usr/bin/env python3
"""
Comprehensive Patching Test
============================
Tests both full layer and attention-only patching for specified layers.
Outputs sample results for documentation.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from typing import Dict, List
import warnings
import os
from contextlib import contextmanager
import json
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'


class PatchingModel:
    """Model wrapper for both full layer and attention patching"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        print(f"Loading model: {model_name}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
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
        
        print("Model loaded successfully")
    
    def get_full_layer(self, layer_idx: int) -> nn.Module:
        """Get the full transformer layer"""
        return self.model.model.layers[layer_idx]
    
    def get_attention_module(self, layer_idx: int) -> nn.Module:
        """Get the attention module for a specific layer"""
        layer = self.model.model.layers[layer_idx]
        return layer.self_attn
    
    def save_activation_hook(self, key: str):
        """Create a hook that saves the activation"""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self.saved_activations[key] = hidden_states.detach().clone()
            return output
        return hook_fn
    
    def patch_activation_hook(self, key: str):
        """Create a hook that patches in a saved activation"""
        def hook_fn(module, input, output):
            saved_activation = self.saved_activations[key]
            
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
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
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.saved_activations = {}
    
    @contextmanager
    def save_and_patch_context(self, source_prompt: str, layer: int, component: str):
        """Combined context for saving from source and patching during generation"""
        try:
            # Save from source prompt
            if component == 'full':
                module = self.get_full_layer(layer)
                key = f"layer_{layer}_full"
            else:  # attention
                module = self.get_attention_module(layer)
                key = f"layer_{layer}_attention"
            
            # Register save hook
            save_hook = module.register_forward_hook(self.save_activation_hook(key))
            self.hooks.append(save_hook)
            
            # Run forward pass to save
            inputs = self.tokenizer(source_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                _ = self.model(**inputs)
            
            # Remove save hook
            save_hook.remove()
            self.hooks.remove(save_hook)
            
            # Register patch hook
            patch_hook = module.register_forward_hook(self.patch_activation_hook(key))
            self.hooks.append(patch_hook)
            
            yield
            
        finally:
            self.clear_hooks()
    
    def generate_with_patching(self, target_prompt: str, source_prompt: str, 
                              layer: int, component: str, max_new_tokens: int = 50) -> str:
        """Generate with patching intervention"""
        
        with self.save_and_patch_context(source_prompt, layer, component):
            inputs = self.tokenizer(target_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Temperature 0
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            generated = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=False
            )
            return generated
    
    def generate_baseline(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Generate without any intervention"""
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
            skip_special_tokens=False
        )
        return generated


def classify_output(output: str) -> str:
    """Classify output as correct, bug, or gibberish"""
    # Take only first 100 chars for classification
    output_sample = output[:100]
    
    if "9.11 is bigger than 9.8" in output_sample:
        return "Bug (9.11 > 9.8)"
    elif "9.8 is bigger than 9.11" in output_sample:
        return "Correct (9.8 > 9.11)"
    elif "://" in output_sample or "php" in output_sample:
        return "Gibberish"
    elif "Question" * 3 in output_sample:
        return "Gibberish"
    else:
        return "Other"


def main():
    """Run comprehensive patching test"""
    
    print("="*70)
    print("COMPREHENSIVE PATCHING TEST")
    print("="*70)
    print("Testing both full layer and attention-only patching")
    print("Temperature: 0.0 (deterministic)")
    print()
    
    # Define prompts
    WRONG_FORMAT = "Q: Which is bigger: 9.8 or 9.11?\nA:"
    CORRECT_FORMAT = "Which is bigger: 9.8 or 9.11?\nAnswer:"
    
    # Layers to test
    test_layers = [6, 7, 8, 9, 10, 11, 12, 15, 20, 23, 25, 27, 28]
    
    # Initialize model
    model = PatchingModel()
    
    # Test baselines
    print("\n📊 BASELINES")
    print("-"*50)
    
    wrong_output = model.generate_baseline(WRONG_FORMAT)
    print(f"Wrong format: {wrong_output[:60]}...")
    print(f"Classification: {classify_output(wrong_output)}")
    
    correct_output = model.generate_baseline(CORRECT_FORMAT)
    print(f"\nCorrect format: {correct_output[:60]}...")
    print(f"Classification: {classify_output(correct_output)}")
    
    # Store results
    results = []
    
    print("\n" + "="*70)
    print("RUNNING PATCHING EXPERIMENTS")
    print("="*70)
    
    for layer in test_layers:
        print(f"\n--- Layer {layer} ---")
        
        # Full layer patching
        print("Full layer patching: ", end="")
        full_output = model.generate_with_patching(
            WRONG_FORMAT, CORRECT_FORMAT, layer, 'full'
        )
        full_class = classify_output(full_output)
        print(full_class)
        print(f"  Output: {full_output[:60]}...")
        
        # Attention-only patching
        print("Attention patching:  ", end="")
        att_output = model.generate_with_patching(
            WRONG_FORMAT, CORRECT_FORMAT, layer, 'attention'
        )
        att_class = classify_output(att_output)
        print(att_class)
        print(f"  Output: {att_output[:60]}...")
        
        results.append({
            'Layer': layer,
            'Full Layer Patching': full_class,
            'Full Output Sample': full_output[:40].replace('\n', ' '),
            'Attention Patching': att_class,
            'Attention Output Sample': att_output[:40].replace('\n', ' ')
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    df.to_csv('comprehensive_patching_results.csv', index=False)
    
    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    
    summary_df = df[['Layer', 'Full Layer Patching', 'Attention Patching']]
    print(summary_df.to_string(index=False))
    
    # Identify successes
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    full_success = df[df['Full Layer Patching'].str.contains('Correct')]
    att_success = df[df['Attention Patching'].str.contains('Correct')]
    
    if not full_success.empty:
        print("\n✓ Full layer patching succeeded at:")
        for _, row in full_success.iterrows():
            print(f"  Layer {row['Layer']}")
    else:
        print("\n✗ No successful full layer patching")
    
    if not att_success.empty:
        print("\n✓ Attention patching succeeded at:")
        for _, row in att_success.iterrows():
            print(f"  Layer {row['Layer']}")
    else:
        print("\n✗ No successful attention patching")
    
    # Save detailed results for markdown
    with open('patching_results_for_markdown.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to:")
    print(f"  - comprehensive_patching_results.csv")
    print(f"  - patching_results_for_markdown.json")


if __name__ == "__main__":
    main()