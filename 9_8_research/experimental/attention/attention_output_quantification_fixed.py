#!/usr/bin/env python3
"""
Attention OUTPUT Quantification Analysis - FIXED VERSION
=========================================================
Uses exact bug reproduction conditions from verify_llama_bug.py
Measures actual attention output contributions at Layer 10
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Set style for clean visualizations
try:
    plt.style.use('seaborn-darkgrid')
except:
    pass
sns.set_palette("husl")

@dataclass
class DecimalPair:
    """Represents a decimal comparison pair"""
    larger: float
    smaller: float
    
    def __str__(self):
        return f"{self.larger} vs {self.smaller}"

@dataclass
class AttentionOutputResult:
    """Results from attention output analysis"""
    prompt: str
    format: str
    decimal_pair: str
    response: str
    is_correct: bool
    begin_output_contribution: float
    format_output_contribution: float
    number_output_contribution: float
    total_norm: float
    confidence: float

class AttentionOutputAnalyzer:
    """Analyzes attention OUTPUT contributions at Layer 10 with correct bug conditions"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        print(f"Loading model: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        print(f"Model loaded on {self.device}")
        
        # Set pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def create_test_suite(self) -> List[DecimalPair]:
        """Create test examples matching verify_llama_bug.py"""
        return [
            DecimalPair(9.8, 9.11),  # The classic bug example
            DecimalPair(9.9, 9.09),
            DecimalPair(8.8, 8.08),
            DecimalPair(7.7, 7.17),
            DecimalPair(6.6, 6.16),
            DecimalPair(5.5, 5.15),
        ]
    
    def create_prompts(self, decimal_pair: DecimalPair) -> Dict[str, List[str]]:
        """Create prompts in formats that reproduce the bug"""
        larger, smaller = decimal_pair.larger, decimal_pair.smaller
        
        prompts = {}
        
        # 1. Chat template format (most reliable for bug)
        user_message = f"Which is bigger: {smaller} or {larger}?"
        chat_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            tokenize=False,
            add_generation_prompt=True
        )
        prompts['chat'] = [chat_prompt]
        
        user_message2 = f"Which is bigger: {larger} or {smaller}?"
        chat_prompt2 = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message2}],
            tokenize=False,
            add_generation_prompt=True
        )
        prompts['chat'].append(chat_prompt2)
        
        # 2. Raw format with special tokens (from ablation work)
        prompts['raw'] = [
            f"<|start_header_id|>user<|end_header_id|>\n\nWhich is bigger: {smaller} or {larger}?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            f"<|start_header_id|>user<|end_header_id|>\n\nWhich is bigger: {larger} or {smaller}?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        ]
        
        # 3. Simple format
        prompts['simple'] = [
            f"Which is bigger: {smaller} or {larger}?\nAnswer:",
            f"Which is bigger: {larger} or {smaller}?\nAnswer:",
        ]
        
        # 4. Q&A format (should show bug)
        prompts['qa'] = [
            f"Q: Which is bigger: {smaller} or {larger}?\nA:",
            f"Q: Which is bigger: {larger} or {smaller}?\nA:",
        ]
        
        return prompts
    
    def hook_attention_output(self, layer_idx: int = 10):
        """Hook to capture attention output at specified layer"""
        self.attention_output = None
        self.value_vectors = None
        self.attention_weights = None
        
        def hook_fn(module, inputs, outputs):
            # For Llama, attention returns (output, weights, cache)
            if isinstance(outputs, tuple):
                self.attention_output = outputs[0].detach().cpu()  # [batch, seq, hidden]
                if len(outputs) > 1 and outputs[1] is not None:
                    self.attention_weights = outputs[1].detach().cpu()
            else:
                self.attention_output = outputs.detach().cpu()
        
        # Register hook on attention module
        handle = self.model.model.layers[layer_idx].self_attn.register_forward_hook(hook_fn)
        return handle
    
    def analyze_attention_output_contributions(
        self, 
        prompt: str,
        layer_idx: int = 10,
        temperature: float = 0.0
    ) -> Tuple[Dict[str, float], str, float]:
        """
        Analyze attention output contributions with exact bug conditions
        """
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        
        # Register hook
        hook_handle = self.hook_attention_output(layer_idx)
        
        try:
            # Generate response with EXACT conditions from verify_llama_bug.py
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids,
                    max_new_tokens=50,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=temperature > 0,  # Critical: False when temp=0
                )
                
                # Also get logits for confidence
                outputs = self.model(input_ids)
                logits = outputs.logits
                next_token_logits = logits[0, -1, :]
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                confidence = next_token_probs.max().item()
            
            # Decode response
            full_response = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            prompt_decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            generated_text = full_response[len(prompt_decoded):].strip()
            
            if self.attention_output is None:
                hook_handle.remove()
                return {}, generated_text, confidence
            
            # Analyze attention output
            attention_output = self.attention_output[0]  # [seq_len, hidden_dim]
            
            # Move to CPU for analysis
            if attention_output.is_cuda:
                attention_output = attention_output.cpu()
            
            # Calculate L2 norm for each position's output
            position_norms = torch.norm(attention_output, p=2, dim=-1)  # [seq_len]
            total_norm = position_norms.sum().item()
            
            # Decode tokens for categorization
            tokens = [self.tokenizer.decode([tid]) for tid in input_ids[0]]
            
            # Categorize positions
            begin_positions = []
            format_positions = []
            number_positions = []
            
            for i, token in enumerate(tokens):
                token_clean = token.strip().lower()
                
                # BEGIN token (first position or special start tokens)
                if i == 0 or 'start' in token_clean or 'begin' in token_clean or token_clean == '<s>':
                    begin_positions.append(i)
                # Format tokens
                elif token_clean in ['q', ':', 'a', 'which', 'is', 'bigger', 'or', '?', 
                                    'answer', 'user', 'assistant', 'header', 'end']:
                    format_positions.append(i)
                # Number tokens
                elif any(c.isdigit() or c == '.' for c in token):
                    number_positions.append(i)
                # Also consider format tokens that weren't caught
                elif len(token_clean) <= 3 and not token_clean.isalnum():
                    format_positions.append(i)
            
            # Calculate contributions
            if total_norm > 0:
                begin_contribution = sum(position_norms[i].item() 
                                       for i in begin_positions 
                                       if i < len(position_norms)) / total_norm
                format_contribution = sum(position_norms[i].item() 
                                        for i in format_positions 
                                        if i < len(position_norms)) / total_norm
                number_contribution = sum(position_norms[i].item() 
                                        for i in number_positions 
                                        if i < len(position_norms)) / total_norm
            else:
                begin_contribution = format_contribution = number_contribution = 0.0
            
            contributions = {
                'begin_output': begin_contribution,
                'format_output': format_contribution,
                'number_output': number_contribution,
                'total_norm': total_norm
            }
            
            return contributions, generated_text, confidence
            
        finally:
            hook_handle.remove()
    
    def check_correctness(self, response: str, larger: float, smaller: float) -> bool:
        """Check if response is correct using logic from verify_llama_bug.py"""
        response_lower = response.lower()
        
        # Look for patterns indicating which number is bigger
        says_larger_bigger = (
            (str(larger) in response and any(w in response_lower for w in ["bigger", "larger", "greater"])) or
            (f"bigger than {smaller}" in response_lower) or
            (f"larger than {smaller}" in response_lower) or
            (f"greater than {smaller}" in response_lower)
        )
        
        says_smaller_bigger = (
            (str(smaller) in response and any(w in response_lower for w in ["bigger", "larger", "greater"])) or
            (f"bigger than {larger}" in response_lower) or
            (f"larger than {larger}" in response_lower) or
            (f"greater than {larger}" in response_lower)
        )
        
        # If both detected, look for first clear statement
        if says_larger_bigger and says_smaller_bigger:
            words = response_lower.split()
            for i, word in enumerate(words):
                if word in [str(larger), str(smaller)] and i + 1 < len(words):
                    if words[i + 1] in ["is", "are"] and i + 2 < len(words):
                        if words[i + 2] in ["bigger", "larger", "greater"]:
                            return word == str(larger)
        
        return says_larger_bigger and not says_smaller_bigger
    
    def analyze_all_examples(self) -> pd.DataFrame:
        """Analyze all test examples with correct bug conditions"""
        results = []
        
        # Create test suite
        decimal_pairs = self.create_test_suite()
        
        print(f"\nAnalyzing attention OUTPUT with EXACT bug conditions...")
        print(f"Temperature: 0.0, do_sample: False")
        
        total_examples = len(decimal_pairs) * 4 * 2  # 4 formats, 2 orderings each
        
        with tqdm(total=total_examples, desc="Processing") as pbar:
            for pair in decimal_pairs:
                prompts_dict = self.create_prompts(pair)
                
                for format_name, format_prompts in prompts_dict.items():
                    for prompt in format_prompts:
                        # Analyze with temperature=0.0 (exact bug conditions)
                        contributions, response, confidence = self.analyze_attention_output_contributions(
                            prompt, 
                            layer_idx=10,
                            temperature=0.0
                        )
                        
                        if not contributions:
                            pbar.update(1)
                            continue
                        
                        # Check correctness
                        is_correct = self.check_correctness(response, pair.larger, pair.smaller)
                        
                        # Store result
                        result = AttentionOutputResult(
                            prompt=prompt[:100] + "..." if len(prompt) > 100 else prompt,
                            format=format_name,
                            decimal_pair=str(pair),
                            response=response[:50],
                            is_correct=is_correct,
                            begin_output_contribution=contributions['begin_output'],
                            format_output_contribution=contributions['format_output'],
                            number_output_contribution=contributions['number_output'],
                            total_norm=contributions['total_norm'],
                            confidence=confidence
                        )
                        
                        results.append(asdict(result))
                        pbar.update(1)
        
        return pd.DataFrame(results)

def perform_statistical_analysis(df: pd.DataFrame) -> Tuple[Dict, Optional[object]]:
    """Perform statistical analysis"""
    from scipy import stats
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS")
    print("="*70)
    
    results = {}
    
    # 1. Format statistics
    print("\n1. ATTENTION OUTPUT BY FORMAT")
    print("-"*40)
    format_stats = df.groupby('format').agg({
        'begin_output_contribution': ['mean', 'std'],
        'format_output_contribution': ['mean', 'std'],
        'number_output_contribution': ['mean', 'std'],
        'is_correct': 'mean'
    }).round(3)
    print(format_stats)
    results['format_stats'] = format_stats.to_dict()
    
    # 2. Correlations
    print("\n2. CORRELATIONS")
    print("-"*40)
    
    if df['is_correct'].std() > 0:  # Check if there's variance
        begin_corr = stats.pearsonr(df['begin_output_contribution'], df['is_correct'].astype(int))
        format_corr = stats.pearsonr(df['format_output_contribution'], df['is_correct'].astype(int))
        print(f"Correlation(BEGIN output, correctness): r={begin_corr[0]:.3f}, p={begin_corr[1]:.4f}")
        print(f"Correlation(format output, correctness): r={format_corr[0]:.3f}, p={format_corr[1]:.4f}")
        
        results['correlations'] = {
            'begin_correctness': {'r': begin_corr[0], 'p': begin_corr[1]},
            'format_correctness': {'r': format_corr[0], 'p': format_corr[1]}
        }
    else:
        print("No variance in correctness - all same class")
        results['correlations'] = {
            'begin_correctness': {'r': 0, 'p': 1},
            'format_correctness': {'r': 0, 'p': 1}
        }
    
    # 3. Logistic regression if possible
    print("\n3. LOGISTIC REGRESSION")
    print("-"*40)
    
    X = df['begin_output_contribution'].values.reshape(-1, 1)
    y = df['is_correct'].astype(int).values
    
    if len(np.unique(y)) > 1:
        log_reg = LogisticRegression(random_state=42)
        log_reg.fit(X, y)
        
        print(f"Coefficient: {log_reg.coef_[0][0]:.3f}")
        print(f"Intercept: {log_reg.intercept_[0]:.3f}")
        
        y_pred_proba = log_reg.predict_proba(X)[:, 1]
        roc_auc = roc_auc_score(y, y_pred_proba)
        print(f"ROC AUC: {roc_auc:.3f}")
        
        results['logistic_regression'] = {
            'coefficient': log_reg.coef_[0][0],
            'intercept': log_reg.intercept_[0],
            'roc_auc': roc_auc
        }
    else:
        print("Cannot perform - single class only")
        log_reg = None
        results['logistic_regression'] = None
    
    return results, log_reg

def main():
    """Main analysis pipeline"""
    print("="*70)
    print("ATTENTION OUTPUT QUANTIFICATION - FIXED VERSION")
    print("Using exact bug reproduction conditions")
    print("="*70)
    
    # Initialize analyzer
    analyzer = AttentionOutputAnalyzer()
    
    # Analyze examples
    df = analyzer.analyze_all_examples()
    
    # Save raw data
    output_file = '/home/paperspace/dev/MATS9/attention/attention_output_fixed_data.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ Raw data saved to {output_file}")
    
    # Statistical analysis
    stats_results, log_reg = perform_statistical_analysis(df)
    
    # Save stats - fix tuple keys
    stats_file = '/home/paperspace/dev/MATS9/attention/attention_output_fixed_stats.json'
    json_safe_results = {}
    for key, value in stats_results.items():
        if isinstance(value, dict):
            json_safe_results[key] = {}
            for k, v in value.items():
                # Convert tuple keys to strings
                if isinstance(k, tuple):
                    json_safe_results[key][str(k)] = v
                else:
                    json_safe_results[key][k] = v
        else:
            json_safe_results[key] = value
    
    with open(stats_file, 'w') as f:
        json.dump(json_safe_results, f, indent=2, default=str)
    print(f"✅ Statistics saved to {stats_file}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\nAnalyzed {len(df)} examples")
    print(f"Overall accuracy: {df['is_correct'].mean():.1%}")
    
    print("\nAccuracy by format:")
    for fmt in df['format'].unique():
        fmt_data = df[df['format'] == fmt]
        print(f"  {fmt:8s}: {fmt_data['is_correct'].mean():.1%} "
              f"(BEGIN output: {fmt_data['begin_output_contribution'].mean():.1%})")
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()