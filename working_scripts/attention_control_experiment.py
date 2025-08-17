#!/usr/bin/env python3
"""
Attention Head Control Experiment
==================================
This script isolates the MLP's role by patching ONLY the attention block output,
allowing us to test if the incompatibility exists even when the MLP gets the
exact same input.

Based on intervention_layers_6_8.py but modified to patch only attention outputs.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import logging
import sys
import os
import warnings
from dataclasses import dataclass
from contextlib import contextmanager
import json
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Set up logging
log_filename = f'attention_control_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_filename)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class InterventionResult:
    """Store results from an intervention experiment"""
    layer: int
    intervention_type: str  # 'attention_only' or 'full_layer'
    prompt_type: str
    output: str
    correct: bool
    has_bug: bool
    tokens_generated: int


class AttentionControlModel:
    """Wrapper for Llama model with attention-specific intervention capabilities"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        logger.info(f"Loading model: {model_name}")
        
        # Load model and tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
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
        
        logger.info("Model loaded successfully")
        logger.info(f"Model architecture: {type(self.model.model)}")
        
        # Verify layer structure
        sample_layer = self.model.model.layers[0]
        logger.info(f"Layer structure: {list(sample_layer._modules.keys())}")
    
    def get_attention_module(self, layer_idx: int) -> nn.Module:
        """Get the attention module for a specific layer"""
        layer = self.model.model.layers[layer_idx]
        # In Llama models, the attention module is typically called 'self_attn'
        if hasattr(layer, 'self_attn'):
            return layer.self_attn
        else:
            raise AttributeError(f"Layer {layer_idx} does not have self_attn module")
    
    def get_full_layer(self, layer_idx: int) -> nn.Module:
        """Get the full transformer layer"""
        return self.model.model.layers[layer_idx]
    
    def save_activation_hook(self, layer_idx: int, component: str = 'full'):
        """Create a hook that saves the activation at a specific layer/component"""
        def hook_fn(module, input, output):
            # output structure depends on the module
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            key = f"{layer_idx}_{component}"
            self.saved_activations[key] = hidden_states.detach().clone()
            return output
        return hook_fn
    
    def patch_activation_hook(self, layer_idx: int, saved_activation: torch.Tensor, component: str = 'full'):
        """Create a hook that patches in a saved activation"""
        def hook_fn(module, input, output):
            # Handle different output structures
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Get dimensions
            batch_size, seq_len, hidden_dim = hidden_states.shape
            saved_seq_len = saved_activation.shape[1]
            
            # Patch the overlapping sequence positions
            min_seq_len = min(seq_len, saved_seq_len)
            
            # Clone to avoid in-place modification issues
            new_hidden = hidden_states.clone()
            new_hidden[:, :min_seq_len, :] = saved_activation[:, :min_seq_len, :]
            
            # Log the patching
            logger.debug(f"Patched {component} at layer {layer_idx}: shape {new_hidden.shape}")
            
            # Return modified output (maintain tuple structure if needed)
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
    def save_activations_context(self, prompt: str, layers: List[int], component: str = 'attention'):
        """Context manager to save activations at specified layers and components"""
        try:
            # Register hooks
            for layer_idx in layers:
                if component == 'attention':
                    module = self.get_attention_module(layer_idx)
                else:  # 'full'
                    module = self.get_full_layer(layer_idx)
                
                hook = module.register_forward_hook(
                    self.save_activation_hook(layer_idx, component)
                )
                self.hooks.append(hook)
            
            # Run forward pass
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                _ = self.model(**inputs)
            
            yield self.saved_activations.copy()
            
        finally:
            self.clear_hooks()
    
    @contextmanager  
    def patch_activations_context(self, saved_activations: Dict[str, torch.Tensor], component: str = 'attention'):
        """Context manager to patch activations during generation"""
        try:
            # Register patching hooks
            for key, activation in saved_activations.items():
                layer_idx = int(key.split('_')[0])
                
                if component == 'attention':
                    module = self.get_attention_module(layer_idx)
                else:  # 'full'
                    module = self.get_full_layer(layer_idx)
                
                hook = module.register_forward_hook(
                    self.patch_activation_hook(layer_idx, activation, component)
                )
                self.hooks.append(hook)
            
            yield
            
        finally:
            self.clear_hooks()
    
    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Generate text from a prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode only the generated tokens
        generated = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=False
        )
        return generated
    
    def generate_with_intervention(
        self, 
        target_prompt: str,
        source_prompt: str, 
        layers: List[int],
        component: str = 'attention',
        max_new_tokens: int = 50
    ) -> str:
        """Generate from target prompt with activations from source prompt patched in"""
        
        # Step 1: Save activations from source prompt
        with self.save_activations_context(source_prompt, layers, component) as saved_acts:
            # Step 2: Generate from target prompt with patched activations
            with self.patch_activations_context(saved_acts, component):
                output = self.generate(target_prompt, max_new_tokens)
        
        return output


def test_baseline(model: AttentionControlModel, prompt: str, num_samples: int = 3) -> Dict:
    """Test baseline generation without intervention"""
    logger.info(f"Testing baseline for prompt: {prompt[:50]}...")
    
    results = {
        'outputs': [],
        'correct_count': 0,
        'bug_count': 0
    }
    
    for i in range(num_samples):
        output = model.generate(prompt, max_new_tokens=50)
        results['outputs'].append(output)
        
        # Check for correct/bug
        output_lower = output.lower()
        if "9.11" in output and "bigger than 9.8" in output:
            results['bug_count'] += 1
            logger.info(f"  Sample {i+1}: ✗ Bug (says 9.11 > 9.8)")
        elif "9.8" in output and "bigger than 9.11" in output:
            results['correct_count'] += 1
            logger.info(f"  Sample {i+1}: ✓ Correct (says 9.8 > 9.11)")
        else:
            logger.info(f"  Sample {i+1}: ? Unclear: {output[:50]}")
    
    results['correct_rate'] = results['correct_count'] / num_samples * 100
    results['bug_rate'] = results['bug_count'] / num_samples * 100
    
    return results


def test_attention_intervention(
    model: AttentionControlModel,
    wrong_prompt: str,
    correct_prompt: str,
    layer_idx: int,
    num_samples: int = 3
) -> Dict:
    """Test intervention at attention block only"""
    logger.info(f"Testing ATTENTION-ONLY intervention at layer {layer_idx}")
    
    results = {
        'layer': layer_idx,
        'component': 'attention',
        'outputs': [],
        'correct_count': 0,
        'bug_count': 0,
        'gibberish_count': 0
    }
    
    for i in range(num_samples):
        output = model.generate_with_intervention(
            target_prompt=wrong_prompt,
            source_prompt=correct_prompt,
            layers=[layer_idx],
            component='attention',
            max_new_tokens=50
        )
        results['outputs'].append(output)
        
        # Check for correct/bug/gibberish
        output_lower = output.lower()
        
        # Check for gibberish patterns
        if "://" in output or "php" in output or output.count(output[:3]) > 5:
            results['gibberish_count'] += 1
            logger.info(f"  Sample {i+1}: 💥 Gibberish: {output[:30]}")
        elif "9.11" in output and "bigger than 9.8" in output:
            results['bug_count'] += 1
            logger.info(f"  Sample {i+1}: ✗ Bug (says 9.11 > 9.8)")
        elif "9.8" in output and "bigger than 9.11" in output:
            results['correct_count'] += 1
            logger.info(f"  Sample {i+1}: ✓ Correct (says 9.8 > 9.11)")
        else:
            logger.info(f"  Sample {i+1}: ? Unclear: {output[:50]}")
    
    results['correct_rate'] = results['correct_count'] / num_samples * 100
    results['bug_rate'] = results['bug_count'] / num_samples * 100
    results['gibberish_rate'] = results['gibberish_count'] / num_samples * 100
    
    return results


def test_full_layer_intervention(
    model: AttentionControlModel,
    wrong_prompt: str,
    correct_prompt: str,
    layer_idx: int,
    num_samples: int = 3
) -> Dict:
    """Test intervention at full layer (for comparison)"""
    logger.info(f"Testing FULL LAYER intervention at layer {layer_idx}")
    
    results = {
        'layer': layer_idx,
        'component': 'full_layer',
        'outputs': [],
        'correct_count': 0,
        'bug_count': 0,
        'gibberish_count': 0
    }
    
    for i in range(num_samples):
        output = model.generate_with_intervention(
            target_prompt=wrong_prompt,
            source_prompt=correct_prompt,
            layers=[layer_idx],
            component='full',
            max_new_tokens=50
        )
        results['outputs'].append(output)
        
        # Check for correct/bug/gibberish
        output_lower = output.lower()
        
        # Check for gibberish patterns
        if "://" in output or "php" in output or output.count(output[:3]) > 5:
            results['gibberish_count'] += 1
            logger.info(f"  Sample {i+1}: 💥 Gibberish: {output[:30]}")
        elif "9.11" in output and "bigger than 9.8" in output:
            results['bug_count'] += 1
            logger.info(f"  Sample {i+1}: ✗ Bug (says 9.11 > 9.8)")
        elif "9.8" in output and "bigger than 9.11" in output:
            results['correct_count'] += 1
            logger.info(f"  Sample {i+1}: ✓ Correct (says 9.8 > 9.11)")
        else:
            logger.info(f"  Sample {i+1}: ? Unclear: {output[:50]}")
    
    results['correct_rate'] = results['correct_count'] / num_samples * 100
    results['bug_rate'] = results['bug_count'] / num_samples * 100
    results['gibberish_rate'] = results['gibberish_count'] / num_samples * 100
    
    return results


def main():
    """Run the attention control experiment"""
    
    logger.info("="*60)
    logger.info("ATTENTION HEAD CONTROL EXPERIMENT")
    logger.info("="*60)
    logger.info("Testing if incompatibility exists even when MLP gets same input")
    logger.info("Methodology: Patch ONLY attention outputs, let MLP process normally")
    
    # Define prompts
    WRONG_FORMAT = "Q: Which is bigger: 9.8 or 9.11?\nA:"
    CORRECT_FORMAT = "Which is bigger: 9.8 or 9.11?\nAnswer:"
    
    # Initialize model
    model = AttentionControlModel()
    
    # Test baselines first
    print("\n📊 TESTING BASELINES")
    print("="*50)
    
    print("\n1. Wrong format baseline (should show bug):")
    wrong_baseline = test_baseline(model, WRONG_FORMAT, num_samples=3)
    print(f"  Correct: {wrong_baseline['correct_rate']:.1f}%")
    print(f"  Bug: {wrong_baseline['bug_rate']:.1f}%")
    
    print("\n2. Correct format baseline (should be correct):")
    correct_baseline = test_baseline(model, CORRECT_FORMAT, num_samples=3)
    print(f"  Correct: {correct_baseline['correct_rate']:.1f}%")
    print(f"  Bug: {correct_baseline['bug_rate']:.1f}%")
    
    # Store all results
    all_results = {
        'baselines': {
            'wrong_format': wrong_baseline,
            'correct_format': correct_baseline
        },
        'attention_only_interventions': [],
        'full_layer_interventions': []
    }
    
    # Only proceed if we reproduced the bug
    if wrong_baseline['bug_rate'] > 0 and correct_baseline['correct_rate'] > 0:
        print("\n✅ Bug successfully reproduced! Proceeding with controlled experiments...")
        
        # Test layers
        test_layers = [6, 8, 10, 15, 20, 25]
        
        attention_results = []
        full_layer_results = []
        
        print("\n🔬 ATTENTION-ONLY PATCHING EXPERIMENTS")
        print("="*50)
        print("Patching ONLY attention outputs from CORRECT into WRONG format")
        print("This tests if MLP can work correctly with proper attention input")
        
        for layer_idx in test_layers:
            print(f"\n--- Layer {layer_idx} ---")
            
            # Test attention-only patching
            print("Attention-only patching:")
            att_result = test_attention_intervention(
                model, WRONG_FORMAT, CORRECT_FORMAT, layer_idx, num_samples=3
            )
            print(f"  Correct: {att_result['correct_rate']:.1f}%")
            print(f"  Bug: {att_result['bug_rate']:.1f}%")
            print(f"  Gibberish: {att_result['gibberish_rate']:.1f}%")
            
            attention_results.append({
                'Layer': layer_idx,
                'Correct %': att_result['correct_rate'],
                'Bug %': att_result['bug_rate'],
                'Gibberish %': att_result['gibberish_rate']
            })
            all_results['attention_only_interventions'].append(att_result)
        
        print("\n🔬 FULL LAYER PATCHING (FOR COMPARISON)")
        print("="*50)
        print("Patching entire layer outputs from CORRECT into WRONG format")
        
        for layer_idx in test_layers:
            print(f"\n--- Layer {layer_idx} ---")
            
            # Test full layer patching for comparison
            print("Full layer patching:")
            full_result = test_full_layer_intervention(
                model, WRONG_FORMAT, CORRECT_FORMAT, layer_idx, num_samples=3
            )
            print(f"  Correct: {full_result['correct_rate']:.1f}%")
            print(f"  Bug: {full_result['bug_rate']:.1f}%")
            print(f"  Gibberish: {full_result['gibberish_rate']:.1f}%")
            
            full_layer_results.append({
                'Layer': layer_idx,
                'Correct %': full_result['correct_rate'],
                'Bug %': full_result['bug_rate'],
                'Gibberish %': full_result['gibberish_rate']
            })
            all_results['full_layer_interventions'].append(full_result)
        
        # Save results
        attention_df = pd.DataFrame(attention_results)
        full_df = pd.DataFrame(full_layer_results)
        
        attention_df.to_csv('attention_only_results.csv', index=False)
        full_df.to_csv('full_layer_comparison_results.csv', index=False)
        
        # Save detailed results
        with open('attention_control_detailed.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Final analysis
        print("\n📊 FINAL COMPARISON")
        print("="*50)
        
        print("\nAttention-Only Patching Results:")
        print(attention_df.to_string())
        
        print("\nFull Layer Patching Results (Reference):")
        print(full_df.to_string())
        
        # Analyze the difference
        print("\n🔍 KEY INSIGHTS")
        print("="*50)
        
        avg_att_gibberish = attention_df['Gibberish %'].mean()
        avg_full_gibberish = full_df['Gibberish %'].mean()
        
        print(f"Average gibberish rate (Attention-only): {avg_att_gibberish:.1f}%")
        print(f"Average gibberish rate (Full layer): {avg_full_gibberish:.1f}%")
        
        if avg_att_gibberish < avg_full_gibberish - 10:
            print("\n✓ ATTENTION-ONLY patching produces LESS gibberish!")
            print("  → This suggests MLPs can partially handle correct attention inputs")
            print("  → The incompatibility is not solely in the MLP")
        elif avg_att_gibberish > avg_full_gibberish + 10:
            print("\n✗ ATTENTION-ONLY patching produces MORE gibberish!")
            print("  → This suggests mismatched attention-MLP interaction")
            print("  → MLPs expect specific attention patterns")
        else:
            print("\n≈ Similar gibberish rates for both approaches")
            print("  → The incompatibility affects both attention and MLP equally")
            print("  → Format differences create systemic incompatibility")
        
        # Check if any attention-only intervention succeeded
        successful_attention = attention_df[attention_df['Correct %'] > 50]
        if not successful_attention.empty:
            print(f"\n🎯 BREAKTHROUGH: Attention-only patching succeeded at layers:")
            for _, row in successful_attention.iterrows():
                print(f"  Layer {row['Layer']}: {row['Correct %']:.1f}% correct")
            print("  → This proves MLPs CAN process correct attention outputs!")
            print("  → The bug is primarily in the attention mechanism!")
        else:
            print("\n❌ No successful attention-only interventions")
            print("  → Even with correct attention, MLPs fail")
            print("  → Suggests deep entanglement between attention and MLP")
    
    else:
        print("\n❌ Failed to reproduce the bug!")
        print("Cannot proceed with controlled experiments.")
    
    logger.info("Attention control experiment complete!")
    print(f"\nResults saved to:")
    print(f"  - attention_only_results.csv")
    print(f"  - full_layer_comparison_results.csv")
    print(f"  - attention_control_detailed.json")
    print(f"  - {log_filename}")


if __name__ == "__main__":
    main()