#!/usr/bin/env python3
"""
Layer 6 and 8 Intervention Analysis using PyTorch Hooks
=========================================================
This script extends the layer 25 analysis to test interventions
at earlier layers (6 and 8) where initial processing happens.

The goal is to test whether patching activations from "good" prompts 
(simple format) into "bad" prompts (chat format) at earlier layers
can fix the decimal comparison bug (9.8 vs 9.11).
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
log_filename = f'layers_6_8_intervention_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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
    prompt_type: str
    output: str
    correct: bool
    has_bug: bool
    tokens_generated: int


class LayerInterventionModel:
    """Wrapper for Llama model with layer intervention capabilities using PyTorch hooks"""
    
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
    
    def get_layer(self, layer_idx: int) -> nn.Module:
        """Get a specific transformer layer"""
        return self.model.model.layers[layer_idx]
    
    def save_activation_hook(self, layer_idx: int):
        """Create a hook that saves the activation at a specific layer"""
        def hook_fn(module, input, output):
            # output is a tuple, we want the hidden states
            hidden_states = output[0] if isinstance(output, tuple) else output
            self.saved_activations[layer_idx] = hidden_states.detach().clone()
            return output
        return hook_fn
    
    def patch_activation_hook(self, layer_idx: int, saved_activation: torch.Tensor):
        """Create a hook that patches in a saved activation"""
        def hook_fn(module, input, output):
            # output is a tuple (hidden_states, ...) 
            hidden_states = output[0] if isinstance(output, tuple) else output
            
            # Get dimensions
            batch_size, seq_len, hidden_dim = hidden_states.shape
            saved_seq_len = saved_activation.shape[1]
            
            # Patch the overlapping sequence positions
            min_seq_len = min(seq_len, saved_seq_len)
            
            # Clone to avoid in-place modification issues
            new_hidden = hidden_states.clone()
            new_hidden[:, :min_seq_len, :] = saved_activation[:, :min_seq_len, :]
            
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
    def save_activations_context(self, prompt: str, layers: List[int]):
        """Context manager to save activations at specified layers"""
        try:
            # Register hooks
            for layer_idx in layers:
                layer = self.get_layer(layer_idx)
                hook = layer.register_forward_hook(self.save_activation_hook(layer_idx))
                self.hooks.append(hook)
            
            # Run forward pass
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
            # Register patching hooks
            for layer_idx, activation in saved_activations.items():
                layer = self.get_layer(layer_idx)
                hook = layer.register_forward_hook(
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
        max_new_tokens: int = 50
    ) -> str:
        """Generate from target prompt with activations from source prompt patched in"""
        
        # Step 1: Save activations from source prompt
        with self.save_activations_context(source_prompt, layers) as saved_acts:
            # Step 2: Generate from target prompt with patched activations
            with self.patch_activations_context(saved_acts):
                output = self.generate(target_prompt, max_new_tokens)
        
        return output


def test_baseline(model: LayerInterventionModel, prompt: str, num_samples: int = 5) -> Dict:
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
        
        # Check for correct/bug - need to look at the actual claim
        output_lower = output.lower()
        if "9.8" in output and "bigger" in output_lower:
            # Check if it's saying 9.8 is bigger (correct) or 9.11 is bigger (wrong)
            if "9.11" in output and "bigger than 9.8" in output:
                # This is saying "9.11 is bigger than 9.8" - WRONG!
                results['bug_count'] += 1
                logger.info(f"  Sample {i+1}: ✗ Bug (says 9.11 > 9.8)")
            elif "9.8" in output and "bigger than 9.11" in output:
                # This is saying "9.8 is bigger than 9.11" - CORRECT!
                results['correct_count'] += 1
                logger.info(f"  Sample {i+1}: ✓ Correct (says 9.8 > 9.11)")
            else:
                # Ambiguous
                logger.info(f"  Sample {i+1}: ? Unclear")
        else:
            logger.info(f"  Sample {i+1}: ? Unclear")
    
    results['correct_rate'] = results['correct_count'] / num_samples * 100
    results['bug_rate'] = results['bug_count'] / num_samples * 100
    
    return results


def test_layer_intervention(
    model: LayerInterventionModel,
    chat_prompt: str,
    simple_prompt: str,
    layer_idx: int,
    num_samples: int = 5
) -> Dict:
    """Test intervention at a specific layer"""
    logger.info(f"Testing intervention at layer {layer_idx}")
    
    results = {
        'layer': layer_idx,
        'outputs': [],
        'correct_count': 0,
        'bug_count': 0
    }
    
    for i in range(num_samples):
        output = model.generate_with_intervention(
            target_prompt=chat_prompt,
            source_prompt=simple_prompt,
            layers=[layer_idx],
            max_new_tokens=50
        )
        results['outputs'].append(output)
        
        # Check for correct/bug - need to look at the actual claim
        output_lower = output.lower()
        if "9.11" in output and "bigger than 9.8" in output:
            # This is saying "9.11 is bigger than 9.8" - WRONG!
            results['bug_count'] += 1
            logger.info(f"  Sample {i+1}: ✗ Bug (says 9.11 > 9.8)")
        elif "9.8" in output and "bigger than 9.11" in output:
            # This is saying "9.8 is bigger than 9.11" - CORRECT!
            results['correct_count'] += 1
            logger.info(f"  Sample {i+1}: ✓ Correct (says 9.8 > 9.11)")
        else:
            logger.info(f"  Sample {i+1}: ? Unclear: {output[:50]}")
    
    results['correct_rate'] = results['correct_count'] / num_samples * 100
    results['bug_rate'] = results['bug_count'] / num_samples * 100
    
    return results


def test_multi_layer_intervention(
    model: LayerInterventionModel,
    chat_prompt: str,
    simple_prompt: str,
    layers: List[int],
    num_samples: int = 5
) -> Dict:
    """Test intervention at multiple layers simultaneously"""
    logger.info(f"Testing intervention at layers {layers}")
    
    results = {
        'layers': layers,
        'outputs': [],
        'correct_count': 0,
        'bug_count': 0
    }
    
    for i in range(num_samples):
        output = model.generate_with_intervention(
            target_prompt=chat_prompt,
            source_prompt=simple_prompt,
            layers=layers,
            max_new_tokens=50
        )
        results['outputs'].append(output)
        
        # Check for correct/bug - need to look at the actual claim
        output_lower = output.lower()
        if "9.8" in output and "bigger" in output_lower:
            # Check if it's saying 9.8 is bigger (correct) or 9.11 is bigger (wrong)
            if "9.11" in output and "bigger than 9.8" in output:
                # This is saying "9.11 is bigger than 9.8" - WRONG!
                results['bug_count'] += 1
                logger.info(f"  Sample {i+1}: ✗ Bug (says 9.11 > 9.8)")
            elif "9.8" in output and "bigger than 9.11" in output:
                # This is saying "9.8 is bigger than 9.11" - CORRECT!
                results['correct_count'] += 1
                logger.info(f"  Sample {i+1}: ✓ Correct (says 9.8 > 9.11)")
            else:
                # Ambiguous
                logger.info(f"  Sample {i+1}: ? Unclear")
        else:
            logger.info(f"  Sample {i+1}: ? Unclear")
    
    results['correct_rate'] = results['correct_count'] / num_samples * 100
    results['bug_rate'] = results['bug_count'] / num_samples * 100
    
    return results


def analyze_token_positions(model: LayerInterventionModel, chat_prompt: str, simple_prompt: str):
    """Analyze token positions to understand prompt structure"""
    chat_tokens = model.tokenizer(chat_prompt, return_tensors="pt")
    simple_tokens = model.tokenizer(simple_prompt, return_tensors="pt")
    
    chat_token_strs = [model.tokenizer.decode([t]) for t in chat_tokens.input_ids[0]]
    simple_token_strs = [model.tokenizer.decode([t]) for t in simple_tokens.input_ids[0]]
    
    logger.info(f"Chat prompt: {len(chat_token_strs)} tokens")
    logger.info(f"Simple prompt: {len(simple_token_strs)} tokens")
    logger.info(f"Chat tokens (last 10): {' | '.join(chat_token_strs[-10:])}")
    logger.info(f"Simple tokens (last 10): {' | '.join(simple_token_strs[-10:])}")
    
    # Find number positions
    chat_numbers = [i for i, tok in enumerate(chat_token_strs) if '9' in tok or '8' in tok or '11' in tok]
    simple_numbers = [i for i, tok in enumerate(simple_token_strs) if '9' in tok or '8' in tok or '11' in tok]
    
    logger.info(f"Number positions in chat: {chat_numbers}")
    logger.info(f"Number positions in simple: {simple_numbers}")


def main():
    """Run the complete layer 6 and 8 intervention analysis using PyTorch hooks"""
    
    logger.info("="*60)
    logger.info("LAYER 6 AND 8 INTERVENTION ANALYSIS WITH PYTORCH HOOKS")
    logger.info("="*60)
    logger.info("Using temperature=0.0 (deterministic) to reproduce the bug")
    logger.info("Testing early layer interventions where initial processing happens")
    
    # Define prompts based on logitlens findings
    # Wrong format: "Q: ... A:" produces wrong answer with temp=0.0
    WRONG_FORMAT = "Q: Which is bigger: 9.8 or 9.11?\nA:"
    
    # Correct format: Simple "Answer:" produces correct answer  
    CORRECT_FORMAT = "Which is bigger: 9.8 or 9.11?\nAnswer:"
    
    # Chat template also produces wrong answer with temp=0.0
    CHAT_PROMPT = """<|start_header_id|>user<|end_header_id|>
Which is bigger: 9.8 or 9.11?
<|end_header_id|>
<|start_header_id|>assistant<|end_header_id|>"""
    
    # Initialize model
    model = LayerInterventionModel()
    
    print("\n🔍 ANALYZING TOKEN STRUCTURE")
    print("="*50)
    analyze_token_positions(model, WRONG_FORMAT, CORRECT_FORMAT)
    analyze_token_positions(model, CHAT_PROMPT, CORRECT_FORMAT)
    
    # Test baselines
    print("\n📊 TESTING BASELINES (temperature=0.0, deterministic)")
    print("="*50)
    
    print("\n1. Wrong format (Q:...A:) baseline - should show bug:")
    wrong_baseline = test_baseline(model, WRONG_FORMAT, num_samples=3)
    print(f"  Correct: {wrong_baseline['correct_rate']:.1f}%")
    print(f"  Bug: {wrong_baseline['bug_rate']:.1f}%")
    if wrong_baseline['outputs']:
        print(f"  Sample output: {wrong_baseline['outputs'][0][:100]}")
    
    print("\n2. Correct format (Answer:) baseline - should be correct:")
    correct_baseline = test_baseline(model, CORRECT_FORMAT, num_samples=3)
    print(f"  Correct: {correct_baseline['correct_rate']:.1f}%")
    print(f"  Bug: {correct_baseline['bug_rate']:.1f}%")
    if correct_baseline['outputs']:
        print(f"  Sample output: {correct_baseline['outputs'][0][:100]}")
    
    print("\n3. Chat template baseline - should show bug:")
    chat_baseline = test_baseline(model, CHAT_PROMPT, num_samples=3)
    print(f"  Correct: {chat_baseline['correct_rate']:.1f}%")
    print(f"  Bug: {chat_baseline['bug_rate']:.1f}%")
    if chat_baseline['outputs']:
        print(f"  Sample output: {chat_baseline['outputs'][0][:100]}")
    
    # Store all results for analysis
    all_results = {
        'baselines': {
            'wrong_format': wrong_baseline,
            'correct_format': correct_baseline,
            'chat_template': chat_baseline
        },
        'single_layer_interventions': [],
        'multi_layer_interventions': []
    }
    
    # Only proceed with interventions if we successfully reproduced the bug
    if wrong_baseline['bug_rate'] > 0 and correct_baseline['correct_rate'] > 0:
        print("\n✅ Bug successfully reproduced! Proceeding with interventions...")
        
        # Test single layer interventions focusing on layers 6 and 8
        print("\n🔧 TESTING SINGLE LAYER INTERVENTIONS (EARLY LAYERS)")
        print("="*50)
        print("Patching activations from CORRECT format into WRONG format")
        
        # Test early layers (including 6 and 8) and some surrounding layers
        layers_to_test = [4, 5, 6, 7, 8, 9, 10, 12, 15]
        single_layer_results = []
        
        for layer_idx in layers_to_test:
            print(f"\nLayer {layer_idx}:")
            result = test_layer_intervention(
                model, WRONG_FORMAT, CORRECT_FORMAT, layer_idx, num_samples=3
            )
            print(f"  Correct: {result['correct_rate']:.1f}%")
            print(f"  Bug: {result['bug_rate']:.1f}%")
            
            single_layer_results.append({
                'Layer': layer_idx,
                'Correct %': result['correct_rate'],
                'Bug %': result['bug_rate']
            })
            all_results['single_layer_interventions'].append(result)
    
        # Test multi-layer interventions focusing on combinations with layers 6 and 8
        print("\n🔬 TESTING MULTI-LAYER INTERVENTIONS (EARLY LAYER COMBINATIONS)")
        print("="*50)
        print("Patching activations from CORRECT format into WRONG format")
        
        # Focus on layer 6 and 8 combinations
        multi_layer_configs = [
            [6],  # Just layer 6
            [8],  # Just layer 8
            [6, 8],  # Both critical early layers
            [5, 6, 7],  # Layer 6 with neighbors
            [7, 8, 9],  # Layer 8 with neighbors
            [6, 7, 8],  # Bridge between 6 and 8
            [5, 6, 7, 8, 9],  # Broader early layer intervention
            [4, 6, 8, 10],  # Spaced early layers
            [6, 8, 12, 15],  # Early to mid layers
        ]
        
        multi_layer_results = []
        
        for layers in multi_layer_configs:
            print(f"\nLayers {layers}:")
            result = test_multi_layer_intervention(
                model, WRONG_FORMAT, CORRECT_FORMAT, layers, num_samples=3
            )
            print(f"  Correct: {result['correct_rate']:.1f}%")
            print(f"  Bug: {result['bug_rate']:.1f}%")
            
            multi_layer_results.append({
                'Layers': str(layers),
                'Correct %': result['correct_rate'],
                'Bug %': result['bug_rate']
            })
            all_results['multi_layer_interventions'].append(result)
        
        # Compare with layer 25 for reference
        print("\n📊 COMPARISON WITH LAYER 25 (REFERENCE)")
        print("="*50)
        
        reference_layers = [20, 22, 25, 28, 30]
        reference_results = []
        
        for layer_idx in reference_layers:
            print(f"\nLayer {layer_idx}:")
            result = test_layer_intervention(
                model, WRONG_FORMAT, CORRECT_FORMAT, layer_idx, num_samples=3
            )
            print(f"  Correct: {result['correct_rate']:.1f}%")
            print(f"  Bug: {result['bug_rate']:.1f}%")
            
            reference_results.append({
                'Layer': layer_idx,
                'Correct %': result['correct_rate'],
                'Bug %': result['bug_rate']
            })
    
        # Save results
        single_df = pd.DataFrame(single_layer_results)
        multi_df = pd.DataFrame(multi_layer_results)
        reference_df = pd.DataFrame(reference_results)
        
        single_df.to_csv('layers_6_8_single_results.csv', index=False)
        multi_df.to_csv('layers_6_8_multi_results.csv', index=False)
        reference_df.to_csv('layers_6_8_reference_results.csv', index=False)
        
        # Save detailed results as JSON
        with open('layers_6_8_detailed_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print("\n📊 FINAL SUMMARY")
        print("="*50)
        print("\nEarly Layer (4-15) Single Layer Results:")
        print(single_df.to_string())
        
        print("\nEarly Layer Multi-Layer Results:")
        print(multi_df.to_string())
        
        print("\nReference Layer (20-30) Results:")
        print(reference_df.to_string())
        
        # Find best configurations
        if not single_df.empty:
            best_single = single_df.loc[single_df['Correct %'].idxmax()]
            print(f"\n🏆 Best Single Early Layer: Layer {best_single['Layer']}")
            print(f"   Correct: {best_single['Correct %']:.1f}%")
            print(f"   Bug: {best_single['Bug %']:.1f}%")
        
        if not multi_df.empty:
            best_multi = multi_df.loc[multi_df['Correct %'].idxmax()]
            print(f"\n🏆 Best Multi-Layer Configuration: {best_multi['Layers']}")
            print(f"   Correct: {best_multi['Correct %']:.1f}%")
            print(f"   Bug: {best_multi['Bug %']:.1f}%")
        
        # Check for "sweet spots"
        sweet_spots = single_df[single_df['Correct %'] > 80]
        if not sweet_spots.empty:
            print(f"\n🎯 EARLY LAYER SWEET SPOTS FOUND (>80% correct):")
            for _, spot in sweet_spots.iterrows():
                print(f"   Layer {spot['Layer']}: {spot['Correct %']:.1f}% correct")
        else:
            print(f"\n❌ No early layer sweet spots found (>80% correct)")
            print("   Early layers may not contain sufficient information for correction")
        
        # Compare early vs late layer effectiveness
        print("\n🔬 EARLY VS LATE LAYER COMPARISON")
        print("="*50)
        
        early_avg = single_df[single_df['Layer'] <= 10]['Correct %'].mean()
        late_avg = reference_df['Correct %'].mean()
        
        print(f"Average correct rate for early layers (≤10): {early_avg:.1f}%")
        print(f"Average correct rate for late layers (20-30): {late_avg:.1f}%")
        
        if late_avg > early_avg + 20:
            print("✓ Late layers are significantly more effective for this intervention")
            print("  This suggests the bug emerges in later processing stages")
        elif early_avg > late_avg + 20:
            print("✓ Early layers are surprisingly effective!")
            print("  This suggests the bug might originate in early processing")
        else:
            print("✓ Similar effectiveness across layers")
            print("  This suggests distributed processing of the comparison")
    
    else:
        print("\n❌ Failed to reproduce the bug!")
        print("Wrong format produced correct answers or correct format produced wrong answers.")
        print("Check the model and prompt settings.")
    
    logger.info("Analysis complete!")
    print(f"\nResults saved to:")
    print(f"  - layers_6_8_single_results.csv")
    print(f"  - layers_6_8_multi_results.csv")
    print(f"  - layers_6_8_reference_results.csv")
    print(f"  - layers_6_8_detailed_results.json")
    print(f"  - {log_filename}")


if __name__ == "__main__":
    main()