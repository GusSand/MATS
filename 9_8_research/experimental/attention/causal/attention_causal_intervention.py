"""
Fixed Attention Causal Intervention
Directly manipulates attention outputs to test causality
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class CausalInterventionExperiment:
    def __init__(self):
        print("Loading model...")
        torch.cuda.empty_cache()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            torch_dtype=torch.float16,
            device_map="cuda:0",
            low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        print("Model loaded!")
        
        # Store for interventions
        self.intervention_config = None
        self.hooks = []
        
    def analyze_prompt_tokens(self, prompt):
        """Categorize tokens in the prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda:0")
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        
        begin_positions = []
        format_positions = []
        number_positions = []
        
        for i, token in enumerate(tokens):
            token_str = str(token).lower()
            
            # First token is always BEGIN
            if i == 0:
                begin_positions.append(i)
            # Format tokens (punctuation, question words, format markers)
            elif any(x in token_str for x in ['?', ':', 'which', 'bigger', 'answer', 'user', 'assistant', 'q', 'a', '▁q', '▁a']):
                format_positions.append(i)
            # Number tokens
            elif any(c.isdigit() or c == '.' for c in token_str):
                number_positions.append(i)
            # Whitespace and newlines are format
            elif token_str in ['\n', '▁', ' ', '']:
                format_positions.append(i)
            else:
                # Everything else counts as format for now
                format_positions.append(i)
        
        print(f"  Token analysis: {len(tokens)} total, {len(begin_positions)} begin, {len(format_positions)} format, {len(number_positions)} number")
        
        return {
            'tokens': tokens,
            'begin_positions': begin_positions,
            'format_positions': format_positions,
            'number_positions': number_positions,
            'total_tokens': len(tokens)
        }
    
    def hook_layer_10_attention(self, target_format_percentage=None, token_info=None):
        """Hook Layer 10 attention to modify format token contributions"""
        
        def attention_hook(module, inputs, outputs):
            if target_format_percentage is None or token_info is None:
                return outputs
            
            # Handle tuple output from attention
            if isinstance(outputs, tuple):
                attn_output = outputs[0]  # [batch, seq_len, hidden_dim]
            else:
                attn_output = outputs
            
            # Clone to avoid in-place modification
            modified_output = attn_output.clone()
            
            # Only modify if we're in the prompt phase (not generation)
            batch_size, seq_len, hidden_dim = modified_output.shape
            
            # Check if this is the prompt forward pass
            if seq_len == token_info['total_tokens']:
                print(f"    Intervening on attention output (seq_len={seq_len})")
                
                # Calculate current format contribution
                with torch.no_grad():
                    # Calculate L2 norm for each position
                    position_norms = torch.norm(modified_output[0], p=2, dim=-1)  # [seq_len]
                    total_norm = position_norms.sum()
                    
                    if total_norm > 0:
                        # Current format contribution
                        format_contrib = 0.0
                        for pos in token_info['format_positions']:
                            if pos < seq_len:
                                format_contrib += position_norms[pos].item()
                        format_contrib = format_contrib / total_norm.item()
                        
                        print(f"    Current format contribution: {format_contrib:.1%}")
                        print(f"    Target format contribution: {target_format_percentage:.1%}")
                        
                        # Calculate scaling factor
                        if format_contrib > 0.001:  # Avoid division by zero
                            format_scale = target_format_percentage / format_contrib
                        else:
                            format_scale = 1.0
                        
                        # Scale format positions
                        for pos in token_info['format_positions']:
                            if pos < seq_len:
                                modified_output[0, pos] *= format_scale
                        
                        # Scale non-format positions to maintain total norm
                        non_format_positions = [i for i in range(seq_len) 
                                               if i not in token_info['format_positions']]
                        
                        if len(non_format_positions) > 0 and format_contrib < 0.999:
                            non_format_scale = (1 - target_format_percentage) / (1 - format_contrib)
                            for pos in non_format_positions:
                                modified_output[0, pos] *= non_format_scale
                        
                        # Verify the intervention
                        new_norms = torch.norm(modified_output[0], p=2, dim=-1)
                        new_total = new_norms.sum()
                        if new_total > 0:
                            new_format_contrib = sum(new_norms[pos] for pos in token_info['format_positions'] if pos < seq_len) / new_total
                            print(f"    After intervention: {new_format_contrib:.1%}")
            
            # Return modified output in same format
            if isinstance(outputs, tuple):
                return (modified_output,) + outputs[1:]
            return modified_output
        
        # Register hook on Layer 10 self-attention
        handle = self.model.model.layers[10].self_attn.register_forward_hook(attention_hook)
        self.hooks.append(handle)
        return handle
    
    def clear_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def generate_with_intervention(self, prompt, target_format_percentage=None):
        """Generate with optional intervention"""
        # Analyze tokens
        token_info = self.analyze_prompt_tokens(prompt) if target_format_percentage else None
        
        # Set up hook if intervening
        if target_format_percentage is not None:
            self.clear_hooks()
            self.hook_layer_10_attention(target_format_percentage, token_info)
        
        # Generate
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
        
        # Clear hooks after generation
        if target_format_percentage is not None:
            self.clear_hooks()
        
        # Decode
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = full_response[len(prompt):].strip()
        
        # Check for bug
        generated_lower = generated.lower()
        says_9_8 = "9.8 is" in generated_lower and any(w in generated_lower for w in ["bigger", "larger", "greater"])
        says_9_11 = "9.11 is" in generated_lower and any(w in generated_lower for w in ["bigger", "larger", "greater"])
        
        shows_bug = says_9_11 and not says_9_8
        is_correct = says_9_8 and not says_9_11
        
        torch.cuda.empty_cache()
        
        return {
            'response': generated[:100],
            'is_correct': is_correct,
            'shows_bug': shows_bug
        }
    
    def run_causal_validation(self, n_trials=5):
        """Test 1: Induce format dominance in Simple format to cause bug"""
        print("\n" + "="*70)
        print("CAUSAL TEST 1: Inducing Bug in Simple Format")
        print("="*70)
        
        simple_prompt = "Which is bigger: 9.8 or 9.11?\nAnswer:"
        results = []
        
        # Baseline (no intervention)
        print("\n1. Baseline (Simple format, natural ~58% format):")
        for i in range(n_trials):
            result = self.generate_with_intervention(simple_prompt, target_format_percentage=None)
            results.append({
                'test': 'induce_bug',
                'condition': 'baseline',
                'format_dominance': 0.58,
                'trial': i,
                **result
            })
            if i == 0:
                symbol = "✅" if result['is_correct'] else "❌" if result['shows_bug'] else "❓"
                print(f"  {symbol} Sample: {result['response'][:60]}...")
        
        baseline_bug_rate = sum(r['shows_bug'] for r in results) / len(results)
        baseline_correct_rate = sum(r['is_correct'] for r in results) / len(results)
        print(f"  Results: {baseline_bug_rate:.0%} bug rate, {baseline_correct_rate:.0%} correct rate")
        
        # Intervention: Boost to 75% (like Q&A format)
        print("\n2. Intervention (boost format to 75%, like Q&A):")
        for i in range(n_trials):
            result = self.generate_with_intervention(simple_prompt, target_format_percentage=0.75)
            results.append({
                'test': 'induce_bug',
                'condition': 'boosted',
                'format_dominance': 0.75,
                'trial': i,
                **result
            })
            if i == 0:
                symbol = "✅" if result['is_correct'] else "❌" if result['shows_bug'] else "❓"
                print(f"  {symbol} Sample: {result['response'][:60]}...")
        
        boosted_bug_rate = sum(r['shows_bug'] for r in results[-n_trials:]) / n_trials
        boosted_correct_rate = sum(r['is_correct'] for r in results[-n_trials:]) / n_trials
        print(f"  Results: {boosted_bug_rate:.0%} bug rate, {boosted_correct_rate:.0%} correct rate")
        
        # Summary
        print(f"\n📊 CAUSAL EFFECT:")
        print(f"  Bug rate: {baseline_bug_rate:.0%} → {boosted_bug_rate:.0%} (Δ = {boosted_bug_rate - baseline_bug_rate:+.0%})")
        print(f"  Correct rate: {baseline_correct_rate:.0%} → {boosted_correct_rate:.0%} (Δ = {boosted_correct_rate - baseline_correct_rate:+.0%})")
        
        if boosted_bug_rate > baseline_bug_rate:
            print("  ✅ CAUSAL LINK CONFIRMED: Boosting format dominance induced the bug!")
        else:
            print("  ❌ No causal effect observed")
        
        return pd.DataFrame(results)
    
    def run_reduction_validation(self, n_trials=5):
        """Test 2: Reduce format dominance in Q&A format to fix bug"""
        print("\n" + "="*70)
        print("CAUSAL TEST 2: Fixing Bug in Q&A Format")
        print("="*70)
        
        qa_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"
        results = []
        
        # Baseline (no intervention)
        print("\n1. Baseline (Q&A format, natural ~62% format):")
        for i in range(n_trials):
            result = self.generate_with_intervention(qa_prompt, target_format_percentage=None)
            results.append({
                'test': 'fix_bug',
                'condition': 'baseline',
                'format_dominance': 0.62,
                'trial': i,
                **result
            })
            if i == 0:
                symbol = "✅" if result['is_correct'] else "❌" if result['shows_bug'] else "❓"
                print(f"  {symbol} Sample: {result['response'][:60]}...")
        
        baseline_bug_rate = sum(r['shows_bug'] for r in results) / len(results)
        baseline_correct_rate = sum(r['is_correct'] for r in results) / len(results)
        print(f"  Results: {baseline_bug_rate:.0%} bug rate, {baseline_correct_rate:.0%} correct rate")
        
        # Intervention: Reduce to 50% (below threshold)
        print("\n2. Intervention (reduce format to 50%, below threshold):")
        for i in range(n_trials):
            result = self.generate_with_intervention(qa_prompt, target_format_percentage=0.50)
            results.append({
                'test': 'fix_bug',
                'condition': 'reduced',
                'format_dominance': 0.50,
                'trial': i,
                **result
            })
            if i == 0:
                symbol = "✅" if result['is_correct'] else "❌" if result['shows_bug'] else "❓"
                print(f"  {symbol} Sample: {result['response'][:60]}...")
        
        reduced_bug_rate = sum(r['shows_bug'] for r in results[-n_trials:]) / n_trials
        reduced_correct_rate = sum(r['is_correct'] for r in results[-n_trials:]) / n_trials
        print(f"  Results: {reduced_bug_rate:.0%} bug rate, {reduced_correct_rate:.0%} correct rate")
        
        # Summary
        print(f"\n📊 CAUSAL EFFECT:")
        print(f"  Bug rate: {baseline_bug_rate:.0%} → {reduced_bug_rate:.0%} (Δ = {reduced_bug_rate - baseline_bug_rate:+.0%})")
        print(f"  Correct rate: {baseline_correct_rate:.0%} → {reduced_correct_rate:.0%} (Δ = {reduced_correct_rate - baseline_correct_rate:+.0%})")
        
        if reduced_bug_rate < baseline_bug_rate:
            print("  ✅ CAUSAL LINK CONFIRMED: Reducing format dominance fixed the bug!")
        else:
            print("  ❌ No causal effect observed")
        
        return pd.DataFrame(results)
    
    def run_threshold_discovery(self, n_trials=3):
        """Test 3: Find exact threshold where bug emerges"""
        print("\n" + "="*70)
        print("CAUSAL TEST 3: Threshold Discovery")
        print("="*70)
        
        # Use Q&A prompt as base
        qa_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"
        thresholds = [0.45, 0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70]
        results = []
        
        print("\nTesting different format dominance levels:")
        for threshold in thresholds:
            print(f"\n  {threshold:.0%} format dominance:")
            
            threshold_results = []
            for i in range(n_trials):
                result = self.generate_with_intervention(qa_prompt, target_format_percentage=threshold)
                threshold_results.append(result)
                results.append({
                    'test': 'threshold',
                    'format_dominance': threshold,
                    'trial': i,
                    **result
                })
            
            bug_rate = sum(r['shows_bug'] for r in threshold_results) / len(threshold_results)
            correct_rate = sum(r['is_correct'] for r in threshold_results) / len(threshold_results)
            
            status = "🔴 BUG" if bug_rate > 0.5 else "🟢 OK" if correct_rate > 0.5 else "🟡 MIXED"
            print(f"    {status} - {bug_rate:.0%} bug, {correct_rate:.0%} correct")
        
        # Find threshold
        df = pd.DataFrame(results)
        threshold_summary = df.groupby('format_dominance')['shows_bug'].mean()
        
        critical_threshold = None
        for i in range(len(threshold_summary) - 1):
            if threshold_summary.iloc[i] < 0.5 and threshold_summary.iloc[i+1] >= 0.5:
                critical_threshold = (threshold_summary.index[i] + threshold_summary.index[i+1]) / 2
                break
        
        if critical_threshold:
            print(f"\n✅ CRITICAL THRESHOLD FOUND: ~{critical_threshold:.1%}")
            print(f"  Below {critical_threshold:.1%}: Bug rate low")
            print(f"  Above {critical_threshold:.1%}: Bug rate high")
        else:
            print("\n❓ No clear threshold identified")
        
        return pd.DataFrame(results), critical_threshold

def main():
    experiment = CausalInterventionExperiment()
    
    print("\n" + "="*70)
    print("🔬 CAUSAL INTERVENTION EXPERIMENTS")
    print("="*70)
    print("Testing if format dominance CAUSES the decimal comparison bug")
    
    # Run all three tests
    induce_results = experiment.run_causal_validation(n_trials=5)
    fix_results = experiment.run_reduction_validation(n_trials=5)
    threshold_results, critical_threshold = experiment.run_threshold_discovery(n_trials=3)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    all_results = pd.concat([induce_results, fix_results, threshold_results], ignore_index=True)
    all_results.to_csv(f'causal_intervention_results_{timestamp}.csv', index=False)
    
    # Create summary
    summary = {
        'timestamp': timestamp,
        'induce_bug': {
            'baseline_bug_rate': induce_results[induce_results['condition'] == 'baseline']['shows_bug'].mean(),
            'boosted_bug_rate': induce_results[induce_results['condition'] == 'boosted']['shows_bug'].mean(),
            'causal_effect': induce_results[induce_results['condition'] == 'boosted']['shows_bug'].mean() - 
                           induce_results[induce_results['condition'] == 'baseline']['shows_bug'].mean()
        },
        'fix_bug': {
            'baseline_bug_rate': fix_results[fix_results['condition'] == 'baseline']['shows_bug'].mean(),
            'reduced_bug_rate': fix_results[fix_results['condition'] == 'reduced']['shows_bug'].mean(),
            'causal_effect': fix_results[fix_results['condition'] == 'reduced']['shows_bug'].mean() - 
                           fix_results[fix_results['condition'] == 'baseline']['shows_bug'].mean()
        },
        'critical_threshold': critical_threshold
    }
    
    with open(f'causal_summary_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Results saved:")
    print(f"  - causal_intervention_results_{timestamp}.csv")
    print(f"  - causal_summary_{timestamp}.json")
    
    return all_results, summary

if __name__ == "__main__":
    results, summary = main()