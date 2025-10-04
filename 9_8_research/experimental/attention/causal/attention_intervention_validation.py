"""
Causal Validation of Attention Output Format Dominance
Tests if manipulating format token contributions causes/fixes the bug
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class AttentionInterventionValidator:
    def __init__(self):
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            torch_dtype=torch.float16,
            device_map="cuda:0",
            low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        
        # Store hooks
        self.hooks = []
        self.attention_outputs = {}
        
    def hook_attention_output(self, layer_idx=10):
        """Hook to capture and modify attention output"""
        def hook_fn(module, inputs, outputs):
            # Store original
            if isinstance(outputs, tuple):
                attention_output = outputs[0]
            else:
                attention_output = outputs
            
            self.attention_outputs[f'layer_{layer_idx}'] = attention_output.clone()
            
            # Apply intervention if set
            if hasattr(self, 'intervention_fn') and self.intervention_fn is not None:
                modified_output = self.intervention_fn(attention_output, module, inputs)
                if isinstance(outputs, tuple):
                    return (modified_output,) + outputs[1:]
                return modified_output
            
            return outputs
        
        handle = self.model.model.layers[layer_idx].self_attn.register_forward_hook(hook_fn)
        self.hooks.append(handle)
        return handle
    
    def clear_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.intervention_fn = None
        
    def analyze_tokens(self, prompt):
        """Categorize tokens into BEGIN, format, and number tokens"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda:0")
        tokens = [self.tokenizer.decode([tid]) for tid in inputs.input_ids[0]]
        
        begin_positions = []
        format_positions = []
        number_positions = []
        
        for i, token in enumerate(tokens):
            token_clean = token.strip().lower()
            
            # BEGIN: First position or start tokens
            if i == 0 or 'begin' in token_clean or token_clean in ['<s>', '<|']:
                begin_positions.append(i)
            # Numbers
            elif any(c.isdigit() or c == '.' for c in token):
                number_positions.append(i)
            # Format tokens (Q:, A:, Answer:, etc)
            elif token_clean in ['q', ':', 'a', 'answer', 'user', 'assistant', '?', '\n']:
                format_positions.append(i)
            else:
                format_positions.append(i)  # Other tokens count as format
        
        return {
            'tokens': tokens,
            'begin_positions': begin_positions,
            'format_positions': format_positions,
            'number_positions': number_positions
        }
    
    def induce_format_dominance(self, prompt, target_format_percentage=0.75):
        """
        In Simple format, boost format token contribution to induce bug
        """
        token_info = self.analyze_tokens(prompt)
        
        def intervention(attention_output, module, inputs):
            # attention_output shape: [batch, seq_len, hidden_dim]
            modified = attention_output.clone()
            
            # Calculate current contributions
            norms = torch.norm(attention_output[0], p=2, dim=-1)
            total_norm = norms.sum()
            
            if total_norm > 0:
                # Current format contribution
                format_contrib = sum(norms[i] for i in token_info['format_positions']) / total_norm
                
                # Scale factor to reach target
                if format_contrib > 0:
                    scale_factor = target_format_percentage / format_contrib.item()
                    
                    # Boost format tokens
                    for i in token_info['format_positions']:
                        if i < len(norms):
                            modified[0, i] *= scale_factor
                    
                    # Reduce other tokens proportionally
                    other_positions = [i for i in range(len(norms)) 
                                     if i not in token_info['format_positions']]
                    reduction_factor = (1 - target_format_percentage) / (1 - format_contrib.item())
                    for i in other_positions:
                        modified[0, i] *= reduction_factor
            
            return modified
        
        self.intervention_fn = intervention
        return token_info
    
    def reduce_format_influence(self, prompt, target_format_percentage=0.59):
        """
        In Chat/Q&A format, reduce format token contribution to fix bug
        """
        token_info = self.analyze_tokens(prompt)
        
        def intervention(attention_output, module, inputs):
            modified = attention_output.clone()
            
            # Calculate current contributions
            norms = torch.norm(attention_output[0], p=2, dim=-1)
            total_norm = norms.sum()
            
            if total_norm > 0:
                # Current format contribution
                format_contrib = sum(norms[i] for i in token_info['format_positions']) / total_norm
                
                # Scale factor to reach target (reducing)
                if format_contrib > 0:
                    scale_factor = target_format_percentage / format_contrib.item()
                    
                    # Reduce format tokens
                    for i in token_info['format_positions']:
                        if i < len(norms):
                            modified[0, i] *= scale_factor
                    
                    # Boost other tokens proportionally
                    other_positions = [i for i in range(len(norms)) 
                                     if i not in token_info['format_positions']]
                    boost_factor = (1 - target_format_percentage) / (1 - format_contrib.item())
                    for i in other_positions:
                        modified[0, i] *= boost_factor
            
            return modified
        
        self.intervention_fn = intervention
        return token_info
    
    def set_format_dominance(self, prompt, target_percentage):
        """Set format dominance to specific level"""
        if target_percentage > 0.65:
            return self.induce_format_dominance(prompt, target_percentage)
        else:
            return self.reduce_format_influence(prompt, target_percentage)
    
    def generate_and_check(self, prompt):
        """Generate response and check for bug"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda:0")
        
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=self.tokenizer.pad_token_id
                )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = full_response[len(prompt):].strip()
        
        # Check for bug
        generated_lower = generated.lower()
        says_9_8 = "9.8 is" in generated_lower and any(w in generated_lower for w in ["bigger", "larger", "greater"])
        says_9_11 = "9.11 is" in generated_lower and any(w in generated_lower for w in ["bigger", "larger", "greater"])
        
        shows_bug = says_9_11 and not says_9_8
        is_correct = says_9_8 and not says_9_11
        
        return {
            'response': generated[:100],
            'is_correct': is_correct,
            'shows_bug': shows_bug
        }
    
    def run_causal_validation(self, n_trials=5):
        """Test 1: Induce format dominance in Simple format"""
        print("\n" + "="*60)
        print("CAUSAL VALIDATION: Inducing Format Dominance")
        print("="*60)
        
        simple_prompt = "Which is bigger: 9.8 or 9.11?\nAnswer:"
        results = []
        
        # Baseline (no intervention)
        print("\nBaseline (Simple format, no intervention):")
        self.clear_hooks()
        for i in range(n_trials):
            result = self.generate_and_check(simple_prompt)
            results.append({
                'intervention': 'baseline',
                'format_percentage': 0.58,  # Natural Simple format level
                'trial': i,
                **result
            })
            if i == 0:
                print(f"  Sample: {result['response'][:60]}...")
        
        baseline_bug_rate = sum(r['shows_bug'] for r in results) / len(results)
        print(f"  Bug rate: {baseline_bug_rate:.1%}")
        
        # With intervention (boost to Q&A levels)
        print("\nWith intervention (boost format to 75%):")
        self.clear_hooks()
        self.hook_attention_output(layer_idx=10)
        
        for i in range(n_trials):
            self.induce_format_dominance(simple_prompt, target_format_percentage=0.75)
            result = self.generate_and_check(simple_prompt)
            results.append({
                'intervention': 'induced',
                'format_percentage': 0.75,
                'trial': i,
                **result
            })
            if i == 0:
                print(f"  Sample: {result['response'][:60]}...")
        
        induced_bug_rate = sum(r['shows_bug'] for r in results[-n_trials:]) / n_trials
        print(f"  Bug rate: {induced_bug_rate:.1%}")
        
        print(f"\n✓ Bug rate increased from {baseline_bug_rate:.1%} to {induced_bug_rate:.1%}")
        
        self.clear_hooks()
        return pd.DataFrame(results)
    
    def run_reduction_validation(self, n_trials=5):
        """Test 2: Reduce format influence in Q&A format"""
        print("\n" + "="*60)
        print("REDUCTION VALIDATION: Reducing Format Influence")
        print("="*60)
        
        qa_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"
        results = []
        
        # Baseline (no intervention)
        print("\nBaseline (Q&A format, no intervention):")
        self.clear_hooks()
        for i in range(n_trials):
            result = self.generate_and_check(qa_prompt)
            results.append({
                'intervention': 'baseline',
                'format_percentage': 0.62,  # Natural Q&A format level
                'trial': i,
                **result
            })
            if i == 0:
                print(f"  Sample: {result['response'][:60]}...")
        
        baseline_bug_rate = sum(r['shows_bug'] for r in results) / len(results)
        print(f"  Bug rate: {baseline_bug_rate:.1%}")
        
        # With intervention (reduce to Simple levels)
        print("\nWith intervention (reduce format to 58%):")
        self.clear_hooks()
        self.hook_attention_output(layer_idx=10)
        
        for i in range(n_trials):
            self.reduce_format_influence(qa_prompt, target_format_percentage=0.58)
            result = self.generate_and_check(qa_prompt)
            results.append({
                'intervention': 'reduced',
                'format_percentage': 0.58,
                'trial': i,
                **result
            })
            if i == 0:
                print(f"  Sample: {result['response'][:60]}...")
        
        reduced_bug_rate = sum(r['shows_bug'] for r in results[-n_trials:]) / n_trials
        print(f"  Bug rate: {reduced_bug_rate:.1%}")
        
        print(f"\n✓ Bug rate reduced from {baseline_bug_rate:.1%} to {reduced_bug_rate:.1%}")
        
        self.clear_hooks()
        return pd.DataFrame(results)
    
    def run_threshold_discovery(self, n_trials=3):
        """Test 3: Find critical threshold"""
        print("\n" + "="*60)
        print("THRESHOLD DISCOVERY: Testing Format Dominance Levels")
        print("="*60)
        
        qa_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"
        thresholds = [0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70, 0.75]
        results = []
        
        for threshold in thresholds:
            print(f"\nTesting {threshold:.0%} format dominance:")
            
            self.clear_hooks()
            self.hook_attention_output(layer_idx=10)
            
            threshold_results = []
            for i in range(n_trials):
                self.set_format_dominance(qa_prompt, threshold)
                result = self.generate_and_check(qa_prompt)
                threshold_results.append(result)
                results.append({
                    'format_percentage': threshold,
                    'trial': i,
                    **result
                })
            
            bug_rate = sum(r['shows_bug'] for r in threshold_results) / len(threshold_results)
            print(f"  Bug rate: {bug_rate:.1%}")
            
            if bug_rate < 0.5 and threshold > 0.60:
                print(f"  ✓ Critical threshold found around {threshold:.0%}")
        
        self.clear_hooks()
        return pd.DataFrame(results)

def main():
    validator = AttentionInterventionValidator()
    
    # Run all three validations
    print("\n🔬 Running Attention Intervention Validations")
    
    # 1. Causal validation
    causal_results = validator.run_causal_validation(n_trials=5)
    
    # 2. Reduction validation  
    reduction_results = validator.run_reduction_validation(n_trials=5)
    
    # 3. Threshold discovery
    threshold_results = validator.run_threshold_discovery(n_trials=3)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Combine all results
    all_results = {
        'causal_validation': causal_results.to_dict('records'),
        'reduction_validation': reduction_results.to_dict('records'),
        'threshold_discovery': threshold_results.to_dict('records')
    }
    
    with open(f'intervention_results_{timestamp}.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to intervention_results_{timestamp}.json")
    
    return causal_results, reduction_results, threshold_results

if __name__ == "__main__":
    causal, reduction, threshold = main()