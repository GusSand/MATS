#!/usr/bin/env python3
"""
Attention Control 10x Repetition Experiment
============================================
Runs the attention control experiment 10 times to verify reproducibility,
especially the Layer 10 success finding.

Tests layers: 6, 8, 9, 10, 11, 12, 15, 20, 25
Temperature: 0.0 (deterministic)
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
from typing import Dict, List
import logging
import sys
import os
import warnings
from contextlib import contextmanager
import json
from datetime import datetime
import gc

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Set up logging
log_filename = f'attention_10x_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_filename)
    ]
)
logger = logging.getLogger(__name__)


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
    
    def get_attention_module(self, layer_idx: int) -> nn.Module:
        """Get the attention module for a specific layer"""
        layer = self.model.model.layers[layer_idx]
        if hasattr(layer, 'self_attn'):
            return layer.self_attn
        else:
            raise AttributeError(f"Layer {layer_idx} does not have self_attn module")
    
    def save_activation_hook(self, layer_idx: int):
        """Create a hook that saves the activation at a specific layer"""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            self.saved_activations[layer_idx] = hidden_states.detach().clone()
            return output
        return hook_fn
    
    def patch_activation_hook(self, layer_idx: int, saved_activation: torch.Tensor):
        """Create a hook that patches in a saved activation"""
        def hook_fn(module, input, output):
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
    def save_activations_context(self, prompt: str, layers: List[int]):
        """Context manager to save activations at specified layers"""
        try:
            for layer_idx in layers:
                module = self.get_attention_module(layer_idx)
                hook = module.register_forward_hook(self.save_activation_hook(layer_idx))
                self.hooks.append(hook)
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                _ = self.model(**inputs)
            
            yield self.saved_activations.copy()
            
        finally:
            self.clear_hooks()
    
    @contextmanager  
    def patch_activations_context(self, saved_activations: Dict[int, torch.Tensor]):
        """Context manager to patch activations during generation"""
        try:
            for layer_idx, activation in saved_activations.items():
                module = self.get_attention_module(layer_idx)
                hook = module.register_forward_hook(
                    self.patch_activation_hook(layer_idx, activation)
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
                do_sample=False,  # Temperature 0.0 (greedy/deterministic)
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=False
        )
        return generated
    
    def generate_with_attention_intervention(
        self, 
        target_prompt: str,
        source_prompt: str, 
        layer: int,
        max_new_tokens: int = 50
    ) -> str:
        """Generate with attention-only intervention at a single layer"""
        
        with self.save_activations_context(source_prompt, [layer]) as saved_acts:
            with self.patch_activations_context(saved_acts):
                output = self.generate(target_prompt, max_new_tokens)
        
        return output


def classify_output(output: str) -> str:
    """Classify output as correct, bug, or gibberish"""
    output_lower = output.lower()
    
    # First check for correct/bug patterns before gibberish
    if "9.11 is bigger than 9.8" in output or "9.11 is greater than 9.8" in output:
        return "bug"
    elif "9.8 is bigger than 9.11" in output or "9.8 is greater than 9.11" in output:
        return "correct"
    # Check for gibberish patterns
    elif "://" in output or "php" in output:
        return "gibberish"
    # Check for repetitive patterns (but not if it contains the answer first)
    elif "Q: Which" in output and output.count("Q: Which") > 1:
        # It's repeating the question, but check if it answered first
        first_part = output.split("Q: Which")[0]
        if "9.11 is bigger" in first_part:
            return "bug"
        elif "9.8 is bigger" in first_part:
            return "correct"
        return "gibberish"
    else:
        # Check for other correct patterns
        if "9.8" in output and ("greater" in output_lower or "larger" in output_lower):
            return "correct"
        return "unclear"


def run_single_experiment(model: AttentionControlModel, layer: int, 
                         wrong_prompt: str, correct_prompt: str) -> Dict:
    """Run a single experiment for one layer"""
    
    output = model.generate_with_attention_intervention(
        target_prompt=wrong_prompt,
        source_prompt=correct_prompt,
        layer=layer,
        max_new_tokens=50
    )
    
    classification = classify_output(output)
    
    return {
        'layer': layer,
        'output': output[:100],  # Truncate for storage
        'classification': classification
    }


def main():
    """Run the 10x repetition experiment"""
    
    print("="*70)
    print("ATTENTION CONTROL 10X REPETITION EXPERIMENT")
    print("="*70)
    print("Running 10 independent experiments to verify reproducibility")
    print("Layers tested: 6, 8, 9, 10, 11, 12, 15, 20, 25")
    print("Temperature: 0.0 (deterministic)")
    print("="*70)
    
    # Define prompts
    WRONG_FORMAT = "Q: Which is bigger: 9.8 or 9.11?\nA:"
    CORRECT_FORMAT = "Which is bigger: 9.8 or 9.11?\nAnswer:"
    
    # Layers to test (added 9, 11, 12)
    test_layers = [6, 8, 9, 10, 11, 12, 15, 20, 25]
    
    # Number of experiment runs
    num_runs = 10
    
    # Initialize model once
    print("\nLoading model...")
    model = AttentionControlModel()
    
    # Test baselines first
    print("\n📊 BASELINE VERIFICATION")
    print("-"*50)
    
    print("Wrong format output (should show bug):")
    wrong_output = model.generate(WRONG_FORMAT, max_new_tokens=50)
    print(f"  {wrong_output[:100]}")
    print(f"  Classification: {classify_output(wrong_output)}")
    
    print("\nCorrect format output (should be correct):")
    correct_output = model.generate(CORRECT_FORMAT, max_new_tokens=50)
    print(f"  {correct_output[:100]}")
    print(f"  Classification: {classify_output(correct_output)}")
    
    # Check if bug is reproduced
    if classify_output(wrong_output) != "bug" or classify_output(correct_output) != "correct":
        print("\n❌ Failed to reproduce bug in baselines! Aborting.")
        return
    
    print("\n✅ Bug reproduced! Starting 10x experiment...")
    
    # Store all results
    all_results = []
    
    # Run experiments
    for run_idx in range(num_runs):
        print(f"\n{'='*70}")
        print(f"RUN {run_idx + 1}/{num_runs}")
        print(f"{'='*70}")
        
        run_results = {}
        
        for layer in test_layers:
            result = run_single_experiment(model, layer, WRONG_FORMAT, CORRECT_FORMAT)
            run_results[layer] = result['classification']
            
            # Print inline result
            symbol = {
                'correct': '✓',
                'bug': '✗',
                'gibberish': '💥',
                'unclear': '?'
            }.get(result['classification'], '?')
            
            print(f"Layer {layer:2d}: {symbol} {result['classification']:10s} | {result['output'][:40]}...")
            
            # Store full result
            all_results.append({
                'run': run_idx + 1,
                'layer': layer,
                'classification': result['classification'],
                'output_sample': result['output']
            })
        
        # Quick summary for this run
        print(f"\nRun {run_idx + 1} Summary:")
        for layer in test_layers:
            print(f"  Layer {layer:2d}: {run_results[layer]}")
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(all_results)
    
    # Calculate statistics
    print("\n" + "="*70)
    print("FINAL STATISTICS ACROSS ALL 10 RUNS")
    print("="*70)
    
    # Create summary table
    summary_data = []
    for layer in test_layers:
        layer_df = df[df['layer'] == layer]
        
        correct_count = (layer_df['classification'] == 'correct').sum()
        bug_count = (layer_df['classification'] == 'bug').sum()
        gibberish_count = (layer_df['classification'] == 'gibberish').sum()
        unclear_count = (layer_df['classification'] == 'unclear').sum()
        
        summary_data.append({
            'Layer': layer,
            'Correct': correct_count,
            'Bug': bug_count,
            'Gibberish': gibberish_count,
            'Unclear': unclear_count,
            'Success Rate %': (correct_count / num_runs) * 100
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    
    # Highlight key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    # Find layers with any success
    successful_layers = summary_df[summary_df['Correct'] > 0]
    if not successful_layers.empty:
        print("\n🎯 LAYERS WITH SUCCESSFUL INTERVENTIONS:")
        for _, row in successful_layers.iterrows():
            print(f"  Layer {int(row['Layer'])}: {row['Correct']}/{num_runs} correct ({row['Success Rate %']:.0f}%)")
    else:
        print("\n❌ No layers had successful interventions")
    
    # Check consistency
    print("\n📊 CONSISTENCY ANALYSIS:")
    for layer in test_layers:
        layer_df = df[df['layer'] == layer]
        unique_outputs = layer_df['classification'].unique()
        if len(unique_outputs) == 1:
            print(f"  Layer {layer}: 100% consistent ({unique_outputs[0]})")
        else:
            print(f"  Layer {layer}: Inconsistent - {', '.join(unique_outputs)}")
    
    # Save detailed results
    df.to_csv('attention_10x_detailed_results.csv', index=False)
    summary_df.to_csv('attention_10x_summary.csv', index=False)
    
    # Save as JSON for further analysis
    with open('attention_10x_all_data.json', 'w') as f:
        json.dump({
            'metadata': {
                'num_runs': num_runs,
                'layers_tested': test_layers,
                'temperature': 0.0,
                'wrong_prompt': WRONG_FORMAT,
                'correct_prompt': CORRECT_FORMAT
            },
            'results': all_results,
            'summary': summary_data
        }, f, indent=2)
    
    print(f"\n📁 Results saved to:")
    print(f"  - attention_10x_detailed_results.csv")
    print(f"  - attention_10x_summary.csv")
    print(f"  - attention_10x_all_data.json")
    print(f"  - {log_filename}")
    
    # Final conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    layer_10_row = summary_df[summary_df['Layer'] == 10].iloc[0]
    if layer_10_row['Success Rate %'] >= 80:
        print(f"✅ LAYER 10 FINDING CONFIRMED: {layer_10_row['Success Rate %']:.0f}% success rate!")
        print("   The MLP at layer 10 CAN process correct attention outputs.")
    elif layer_10_row['Success Rate %'] >= 50:
        print(f"⚠️ LAYER 10 PARTIALLY CONFIRMED: {layer_10_row['Success Rate %']:.0f}% success rate")
        print("   Results are inconsistent but show promise.")
    else:
        print(f"❌ LAYER 10 FINDING NOT REPRODUCED: Only {layer_10_row['Success Rate %']:.0f}% success rate")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()