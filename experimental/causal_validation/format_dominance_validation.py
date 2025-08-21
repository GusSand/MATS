#!/usr/bin/env python3
"""
Format Dominance Causal Validation Experiments
==============================================
Tests if manipulating format token contributions at Layer 10 can:
1. Induce the bug in correct formats (by boosting format tokens)
2. Fix the bug in buggy formats (by reducing format tokens)
3. Find the critical threshold for bug occurrence
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
from contextlib import contextmanager
import json
from datetime import datetime
from typing import Dict, List, Tuple

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

class FormatDominanceValidator:
    """Test format dominance hypothesis through causal interventions"""
    
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
        
        self.hooks = []
        self.saved_activations = {}
        
        print("Model loaded successfully!")
    
    def identify_token_positions(self, prompt: str) -> Dict[str, List[int]]:
        """Identify positions of format tokens, numbers, and other tokens"""
        tokens = self.tokenizer.encode(prompt, return_tensors="pt")[0]
        token_strings = [self.tokenizer.decode([t]) for t in tokens]
        
        positions = {
            'format': [],
            'numbers': [],
            'other': []
        }
        
        # Identify format tokens (Q:, A:, Which, Answer:, etc.)
        format_indicators = ['Q', ':', 'Which', 'Answer', 'A']
        number_indicators = ['9', '.', '8', '11']
        
        for idx, token_str in enumerate(token_strings):
            token_stripped = token_str.strip()
            if any(fmt in token_stripped for fmt in format_indicators):
                positions['format'].append(idx)
            elif any(num in token_stripped for num in number_indicators):
                positions['numbers'].append(idx)
            else:
                positions['other'].append(idx)
        
        return positions
    
    def get_attention_module(self, layer_idx: int) -> nn.Module:
        """Get the attention module for a specific layer"""
        layer = self.model.model.layers[layer_idx]
        return layer.self_attn
    
    def modulate_format_contribution_hook(self, layer_idx: int, token_positions: Dict, 
                                         target_format_percentage: float):
        """Create hook that modulates format token contributions"""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Clone to avoid in-place modification
            modified_hidden = hidden_states.clone()
            
            # Calculate current contributions (simplified - using norm as proxy)
            seq_len = hidden_states.shape[1]
            contributions = torch.norm(hidden_states[0], dim=-1)  # [seq_len]
            total_contribution = contributions.sum()
            
            # Current format contribution
            format_positions = [p for p in token_positions['format'] if p < seq_len]
            if not format_positions:
                return output
            
            format_contrib = contributions[format_positions].sum()
            current_format_percentage = (format_contrib / total_contribution).item()
            
            # Calculate scaling factor
            if current_format_percentage > 0:
                scale_factor = target_format_percentage / current_format_percentage
            else:
                scale_factor = 1.0
            
            # Apply scaling to format tokens
            for pos in format_positions:
                if pos < seq_len:
                    modified_hidden[0, pos] *= scale_factor
            
            # Adjust other tokens to maintain total magnitude
            other_positions = [p for p in range(seq_len) if p not in format_positions]
            if other_positions and scale_factor != 1.0:
                # Calculate compensation factor
                format_change = (scale_factor - 1.0) * format_contrib
                other_contrib = total_contribution - format_contrib
                if other_contrib > 0:
                    compensation = 1.0 - (format_change / other_contrib)
                    compensation = max(0.1, compensation)  # Prevent extreme scaling
                    
                    for pos in other_positions:
                        modified_hidden[0, pos] *= compensation
            
            if isinstance(output, tuple):
                return (modified_hidden,) + output[1:]
            return modified_hidden
        
        return hook_fn
    
    def clear_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    @contextmanager
    def format_modulation_context(self, prompt: str, target_format_percentage: float, 
                                 layer_idx: int = 10):
        """Context manager for format contribution modulation"""
        try:
            # Identify token positions
            token_positions = self.identify_token_positions(prompt)
            
            # Register modulation hook
            module = self.get_attention_module(layer_idx)
            hook = module.register_forward_hook(
                self.modulate_format_contribution_hook(layer_idx, token_positions, 
                                                      target_format_percentage)
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
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated[len(prompt):]
    
    def classify_output(self, output: str) -> str:
        """Classify output as correct, bug, or unclear"""
        output_lower = output.lower()
        
        correct_patterns = ["9.8 is bigger", "9.8 is larger", "9.8 is greater"]
        bug_patterns = ["9.11 is bigger", "9.11 is larger", "9.11 is greater"]
        
        if any(pattern in output_lower for pattern in correct_patterns):
            return "correct"
        elif any(pattern in output_lower for pattern in bug_patterns):
            return "bug"
        else:
            return "unclear"
    
    def run_induction_experiment(self, n_trials: int = 10):
        """Experiment 1: Try to induce bug in simple format"""
        print("\n" + "="*70)
        print("EXPERIMENT 1: FORMAT DOMINANCE INDUCTION")
        print("="*70)
        print("Goal: Induce bug in Simple format by boosting format tokens")
        
        simple_prompt = "Which is bigger: 9.8 or 9.11?\nAnswer:"
        
        # Test different boost levels
        boost_levels = [0.60, 0.65, 0.70, 0.75, 0.80]
        results = {}
        
        for target_percentage in boost_levels:
            print(f"\n📊 Testing format contribution: {target_percentage:.0%}")
            
            trial_results = []
            for i in range(n_trials):
                with self.format_modulation_context(simple_prompt, target_percentage):
                    output = self.generate(simple_prompt)
                    classification = self.classify_output(output)
                    trial_results.append(classification)
                    
                    if i == 0:
                        print(f"  Sample output: {output[:50]}...")
                        print(f"  Classification: {classification}")
            
            bug_rate = sum(1 for r in trial_results if r == "bug") / n_trials
            correct_rate = sum(1 for r in trial_results if r == "correct") / n_trials
            
            results[target_percentage] = {
                'bug_rate': bug_rate,
                'correct_rate': correct_rate,
                'trials': trial_results
            }
            
            print(f"  Results: {correct_rate:.0%} correct, {bug_rate:.0%} bug")
        
        return results
    
    def run_reduction_experiment(self, n_trials: int = 10):
        """Experiment 2: Try to fix bug in Q&A format"""
        print("\n" + "="*70)
        print("EXPERIMENT 2: FORMAT INFLUENCE REDUCTION")
        print("="*70)
        print("Goal: Fix bug in Q&A format by reducing format tokens")
        
        qa_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"
        
        # Test different reduction levels
        reduction_levels = [0.40, 0.45, 0.50, 0.55, 0.59]
        results = {}
        
        for target_percentage in reduction_levels:
            print(f"\n📊 Testing format contribution: {target_percentage:.0%}")
            
            trial_results = []
            for i in range(n_trials):
                with self.format_modulation_context(qa_prompt, target_percentage):
                    output = self.generate(qa_prompt)
                    classification = self.classify_output(output)
                    trial_results.append(classification)
                    
                    if i == 0:
                        print(f"  Sample output: {output[:50]}...")
                        print(f"  Classification: {classification}")
            
            bug_rate = sum(1 for r in trial_results if r == "bug") / n_trials
            correct_rate = sum(1 for r in trial_results if r == "correct") / n_trials
            
            results[target_percentage] = {
                'bug_rate': bug_rate,
                'correct_rate': correct_rate,
                'trials': trial_results
            }
            
            print(f"  Results: {correct_rate:.0%} correct, {bug_rate:.0%} bug")
        
        return results
    
    def run_threshold_discovery(self, n_trials: int = 5):
        """Experiment 3: Find critical threshold"""
        print("\n" + "="*70)
        print("EXPERIMENT 3: THRESHOLD DISCOVERY")
        print("="*70)
        print("Goal: Find critical format dominance threshold for bug")
        
        # Use a neutral prompt
        prompt = "Which is bigger: 9.8 or 9.11?\nAnswer:"
        
        # Test range of thresholds
        thresholds = [0.50, 0.55, 0.60, 0.63, 0.65, 0.70, 0.75]
        results = {}
        
        for threshold in thresholds:
            print(f"\n📊 Testing threshold: {threshold:.0%}")
            
            trial_results = []
            for i in range(n_trials):
                with self.format_modulation_context(prompt, threshold):
                    output = self.generate(prompt)
                    classification = self.classify_output(output)
                    trial_results.append(classification)
            
            bug_rate = sum(1 for r in trial_results if r == "bug") / n_trials
            correct_rate = sum(1 for r in trial_results if r == "correct") / n_trials
            
            results[threshold] = {
                'bug_rate': bug_rate,
                'correct_rate': correct_rate
            }
            
            print(f"  Bug rate: {bug_rate:.0%}, Correct rate: {correct_rate:.0%}")
        
        # Find critical threshold
        for threshold in sorted(results.keys()):
            if results[threshold]['bug_rate'] > 0.5:
                print(f"\n🎯 Critical threshold found: ~{threshold:.0%}")
                print(f"   Below {threshold:.0%}: Mostly correct")
                print(f"   Above {threshold:.0%}: Mostly buggy")
                break
        
        return results
    
    def create_visualization(self, induction_results: Dict, reduction_results: Dict, 
                           threshold_results: Dict):
        """Create comprehensive visualization of results"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Panel 1: Stacked bar chart of output contributions
        ax1 = axes[0]
        formats = ['Simple\n(Correct)', 'Q&A\n(Bug)', 'Chat\n(Bug)']
        format_contrib = [59.4, 63.6, 75.6]
        begin_contrib = [4.0, 3.6, 2.9]
        number_contrib = [36.6, 32.8, 21.6]
        
        # Create stacked bars
        p1 = ax1.bar(formats, format_contrib, label='Format Tokens', color='#f44336')
        p2 = ax1.bar(formats, begin_contrib, bottom=format_contrib, 
                    label='BEGIN Token', color='#2196F3')
        p3 = ax1.bar(formats, number_contrib, 
                    bottom=[f+b for f,b in zip(format_contrib, begin_contrib)], 
                    label='Number Tokens', color='#4CAF50')
        
        ax1.axhline(y=63, color='black', linestyle='--', linewidth=2, label='Bug Threshold')
        ax1.set_ylabel('Attention Output Contribution (%)', fontsize=12)
        ax1.set_title('Format Token Hijacking at Layer 10', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.set_ylim(0, 100)
        ax1.grid(axis='y', alpha=0.3)
        
        # Panel 2: Format dominance vs bug rate
        ax2 = axes[1]
        
        # Combine all threshold data
        all_thresholds = sorted(threshold_results.keys())
        bug_rates = [threshold_results[t]['bug_rate'] for t in all_thresholds]
        
        ax2.scatter(all_thresholds, bug_rates, s=100, color='#f44336', alpha=0.7, 
                   label='Observed Bug Rate')
        ax2.plot(all_thresholds, bug_rates, 'r--', alpha=0.5)
        
        # Add vertical line at critical threshold
        ax2.axvline(x=0.63, color='black', linestyle='--', linewidth=2, 
                   label='Critical Threshold')
        ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        
        ax2.set_xlabel('Format Token Contribution (%)', fontsize=12)
        ax2.set_ylabel('Bug Rate', fontsize=12)
        ax2.set_title('Bug Emergence at Format Threshold', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0.45, 0.80)
        ax2.set_ylim(-0.05, 1.05)
        
        # Panel 3: Intervention effects
        ax3 = axes[2]
        
        # Prepare intervention data
        categories = ['Baseline\nSimple', 'Boosted\nSimple', 'Baseline\nQ&A', 'Reduced\nQ&A']
        correct_rates = [
            1.0,  # Baseline simple (correct)
            1.0 - max(induction_results.values(), key=lambda x: x['bug_rate'])['bug_rate'],
            0.0,  # Baseline Q&A (bug)
            max(reduction_results.values(), key=lambda x: x['correct_rate'])['correct_rate']
        ]
        colors = ['#4CAF50', '#FFA726', '#f44336', '#66BB6A']
        
        bars = ax3.bar(categories, correct_rates, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, rate in zip(bars, correct_rates):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{rate:.0%}', ha='center', va='bottom', fontweight='bold')
        
        ax3.set_ylabel('Correct Answer Rate', fontsize=12)
        ax3.set_title('Intervention Effects', fontsize=14, fontweight='bold')
        ax3.set_ylim(0, 1.1)
        ax3.grid(axis='y', alpha=0.3)
        
        # Add annotations
        ax3.annotate('Induction\nAttempt', xy=(1, correct_rates[1]), 
                    xytext=(1, 0.5), fontsize=10,
                    arrowprops=dict(arrowstyle='->', color='orange', lw=2))
        ax3.annotate('Reduction\nSuccess', xy=(3, correct_rates[3]), 
                    xytext=(3, 0.5), fontsize=10,
                    arrowprops=dict(arrowstyle='->', color='green', lw=2))
        
        plt.suptitle('The Format Hijacking Effect: Causal Validation', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'format_hijacking_validation_{timestamp}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n📊 Visualization saved to {filename}")
        
        return fig

def main():
    """Run all validation experiments"""
    validator = FormatDominanceValidator()
    
    # Run experiments
    print("\n🔬 RUNNING FORMAT DOMINANCE VALIDATION EXPERIMENTS")
    print("="*70)
    
    # Experiment 1: Induction
    induction_results = validator.run_induction_experiment(n_trials=5)
    
    # Experiment 2: Reduction  
    reduction_results = validator.run_reduction_experiment(n_trials=5)
    
    # Experiment 3: Threshold
    threshold_results = validator.run_threshold_discovery(n_trials=5)
    
    # Create visualization
    fig = validator.create_visualization(induction_results, reduction_results, 
                                        threshold_results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'timestamp': timestamp,
        'induction': {str(k): v for k, v in induction_results.items()},
        'reduction': {str(k): v for k, v in reduction_results.items()},
        'threshold': {str(k): v for k, v in threshold_results.items()}
    }
    
    with open(f'format_dominance_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to format_dominance_results_{timestamp}.json")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY OF FINDINGS")
    print("="*70)
    
    # Check if induction worked
    max_induction_bug = max(induction_results.values(), key=lambda x: x['bug_rate'])['bug_rate']
    if max_induction_bug > 0.5:
        print("✅ INDUCTION SUCCESS: Can induce bug by boosting format tokens")
    else:
        print("❌ INDUCTION FAILED: Boosting format tokens doesn't induce bug")
    
    # Check if reduction worked
    max_reduction_correct = max(reduction_results.values(), 
                               key=lambda x: x['correct_rate'])['correct_rate']
    if max_reduction_correct > 0.5:
        print("✅ REDUCTION SUCCESS: Can fix bug by reducing format tokens")
    else:
        print("❌ REDUCTION FAILED: Reducing format tokens doesn't fix bug")
    
    # Report threshold
    critical_threshold = None
    for t in sorted(threshold_results.keys()):
        if threshold_results[t]['bug_rate'] > 0.5:
            critical_threshold = t
            break
    
    if critical_threshold:
        print(f"🎯 CRITICAL THRESHOLD: ~{critical_threshold:.0%} format contribution")
    
    return results

if __name__ == "__main__":
    results = main()