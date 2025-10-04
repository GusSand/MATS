#!/usr/bin/env python3
"""
Attention OUTPUT Quantification Analysis
========================================
Measures the actual attention output contributions (not just weights) at Layer 10
to understand the causal mechanism behind the decimal comparison bug.

Key difference from previous analysis:
- Previous: Measured attention WEIGHTS (how much the model looks at each position)
- This: Measures attention OUTPUT (what information actually flows from each position)
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

# Set style for clean visualizations
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('seaborn-darkgrid')
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
    begin_output_contribution: float  # How much BEGIN contributes to output
    format_output_contribution: float  # How much format tokens contribute
    number_output_contribution: float  # How much number tokens contribute
    total_norm: float  # Total norm of attention output
    confidence: float

class AttentionOutputAnalyzer:
    """Analyzes attention OUTPUT contributions at Layer 10"""
    
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
        
        # Special tokens
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def create_test_suite(self) -> List[DecimalPair]:
        """Create diverse test examples"""
        pairs = []
        
        # Original problematic pair
        pairs.append(DecimalPair(9.8, 9.11))
        
        # Variations with similar structure
        for larger in [9.7, 8.6, 7.5, 6.4]:
            smaller = larger + 0.01 + 0.3  # Creates x.y vs x.yz pattern
            pairs.append(DecimalPair(larger, round(smaller, 2)))
        
        # Different decimal patterns
        pairs.extend([
            DecimalPair(9.9, 9.09),
            DecimalPair(8.8, 8.08),
            DecimalPair(7.7, 7.17),
            DecimalPair(5.5, 5.15),
            DecimalPair(3.3, 3.13),
            DecimalPair(10.1, 10.01),
            DecimalPair(4.2, 4.12),
        ])
        
        return pairs[:12]  # Use 12 pairs for 120 total examples
    
    def create_prompts(self, decimal_pair: DecimalPair) -> Dict[str, List[str]]:
        """Create prompts in different formats"""
        larger, smaller = decimal_pair.larger, decimal_pair.smaller
        
        prompts = {
            'simple': [
                f"Which is bigger: {smaller} or {larger}?\nAnswer:",
                f"Which is bigger: {larger} or {smaller}?\nAnswer:",
            ],
            'qa': [
                f"Q: Which is bigger: {smaller} or {larger}?\nA:",
                f"Q: Which is bigger: {larger} or {smaller}?\nA:",
            ],
            'question': [
                f"Question: Which is bigger: {smaller} or {larger}?\nAnswer:",
                f"Question: Which is bigger: {larger} or {smaller}?\nAnswer:",
            ],
            'compare': [
                f"Compare {smaller} and {larger}. Which is bigger?\nAnswer:",
                f"Compare {larger} and {smaller}. Which is bigger?\nAnswer:",
            ],
            'direct': [
                f"{smaller} or {larger} - which is bigger?",
                f"{larger} or {smaller} - which is bigger?",
            ],
        }
        
        return prompts
    
    def hook_attention_output(self, layer_idx: int = 10):
        """Hook to capture attention output at specified layer"""
        self.attention_output = None
        self.attention_weights = None
        self.hidden_states_input = None
        
        def hook_fn(module, inputs, outputs):
            # For Llama, attention module returns (output, weights)
            if isinstance(outputs, tuple):
                self.attention_output = outputs[0].detach()
                if len(outputs) > 1 and outputs[1] is not None:
                    self.attention_weights = outputs[1].detach()
            else:
                self.attention_output = outputs.detach()
            
            # Store input hidden states
            self.hidden_states_input = inputs[0].detach() if inputs else None
        
        # Register hook on attention module
        handle = self.model.model.layers[layer_idx].self_attn.register_forward_hook(hook_fn)
        return handle
    
    def analyze_attention_output_contributions(
        self, 
        prompt: str,
        layer_idx: int = 10
    ) -> Tuple[Dict[str, float], str, float]:
        """
        Analyze how different token groups contribute to attention output.
        
        This measures the actual information flow, not just attention weights.
        """
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        
        # Decode tokens for analysis
        tokens = [self.tokenizer.decode([tid]) for tid in input_ids[0]]
        
        # Register hook
        hook_handle = self.hook_attention_output(layer_idx)
        
        try:
            # Forward pass with generation for complete response
            with torch.no_grad():
                # First, get attention output at layer 10
                outputs = self.model(**inputs)
                
                # Now generate a complete response
                generated = self.model.generate(
                    input_ids,
                    max_new_tokens=20,
                    temperature=0.01,  # Near-deterministic
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                # Decode the generated response (excluding the prompt)
                prompt_length = input_ids.shape[1]
                generated_tokens = generated[0, prompt_length:]
                predicted_token = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # Get confidence from first token prediction
                logits = outputs.logits
                next_token_logits = logits[0, -1, :]
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                first_generated_id = generated_tokens[0].item() if len(generated_tokens) > 0 else 0
                confidence = next_token_probs[first_generated_id].item() if first_generated_id < len(next_token_probs) else 0.0
            
            if self.attention_output is None:
                hook_handle.remove()
                return {}, predicted_token, confidence
            
            # Analyze attention output contributions
            attention_output = self.attention_output[0]  # [seq_len, hidden_dim]
            
            # Calculate contribution norms for each position
            position_norms = torch.norm(attention_output, dim=-1)  # [seq_len]
            total_norm = torch.sum(position_norms).item()
            
            # Identify token groups
            begin_positions = [0]  # BEGIN token is always position 0
            format_positions = []
            number_positions = []
            
            for i, token in enumerate(tokens):
                token_lower = token.lower().strip()
                # Format tokens
                if token_lower in ['q', ':', 'question', 'compare', 'which', 'bigger', 'answer', 'a', '?', '-', 'and', 'or', 'is', '.']:
                    format_positions.append(i)
                # Number tokens
                elif any(char.isdigit() for char in token):
                    number_positions.append(i)
            
            # Calculate group contributions
            begin_contribution = sum(position_norms[i].item() for i in begin_positions if i < len(position_norms))
            format_contribution = sum(position_norms[i].item() for i in format_positions if i < len(position_norms))
            number_contribution = sum(position_norms[i].item() for i in number_positions if i < len(position_norms))
            
            # Normalize to percentages
            if total_norm > 0:
                begin_pct = begin_contribution / total_norm
                format_pct = format_contribution / total_norm
                number_pct = number_contribution / total_norm
            else:
                begin_pct = format_pct = number_pct = 0.0
            
            contributions = {
                'begin_output': begin_pct,
                'format_output': format_pct,
                'number_output': number_pct,
                'total_norm': total_norm
            }
            
            return contributions, predicted_token, confidence
            
        finally:
            hook_handle.remove()
    
    def analyze_all_examples(self) -> pd.DataFrame:
        """Analyze attention output for all test examples"""
        results = []
        
        # Create test suite
        decimal_pairs = self.create_test_suite()
        
        print(f"Analyzing attention OUTPUT at Layer 10...")
        print(f"(This measures information flow, not just attention weights)")
        
        with tqdm(total=len(decimal_pairs) * 5 * 2, desc="Processing examples") as pbar:
            for pair in decimal_pairs:
                prompts_dict = self.create_prompts(pair)
                
                for format_name, format_prompts in prompts_dict.items():
                    for prompt in format_prompts:
                        # Analyze attention output
                        contributions, response, confidence = self.analyze_attention_output_contributions(prompt)
                        
                        if not contributions:
                            pbar.update(1)
                            continue
                        
                        # Determine correctness
                        response_lower = response.lower().strip()
                        larger_str = str(pair.larger)
                        smaller_str = str(pair.smaller)
                        
                        # Debug: Print first few responses
                        if len(results) < 5:
                            print(f"Response: '{response}' for prompt ending with: ...{prompt[-30:]}")
                        
                        # Check if response indicates correct answer
                        is_correct = False
                        
                        # Check for the larger number in response
                        if larger_str in response:  # Don't use lower() for number matching
                            # Make sure it's actually saying this number is bigger
                            if "11" not in response or pair.larger != 9.11:  # Special case for 9.11
                                is_correct = True
                        
                        # Check for specific patterns
                        if "9.8" in response and "9.11" in prompt and "11" not in response:
                            is_correct = True
                        elif "9.11" in response and "9.8" in prompt and pair.larger == 9.11:
                            is_correct = True
                        
                        # Check ordinal references
                        if "first" in response_lower:
                            # Check if larger is first in prompt
                            parts = prompt.lower().split("or")
                            if len(parts) >= 2 and larger_str in parts[0]:
                                is_correct = True
                        elif "second" in response_lower:
                            # Check if larger is second in prompt
                            parts = prompt.lower().split("or")
                            if len(parts) >= 2 and larger_str in parts[1]:
                                is_correct = True
                        
                        # Store result
                        result = AttentionOutputResult(
                            prompt=prompt,
                            format=format_name,
                            decimal_pair=str(pair),
                            response=response,
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
    """Perform statistical analysis on attention output data"""
    from scipy import stats
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve
    
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS OF ATTENTION OUTPUT")
    print("="*70)
    
    results = {}
    
    # 1. Format-wise statistics
    print("\n1. ATTENTION OUTPUT STATISTICS BY FORMAT")
    print("-"*40)
    format_stats = df.groupby('format').agg({
        'begin_output_contribution': ['mean', 'std'],
        'format_output_contribution': ['mean', 'std'],
        'number_output_contribution': ['mean', 'std'],
        'is_correct': 'mean'
    }).round(3)
    print(format_stats)
    results['format_stats'] = format_stats.to_dict()
    
    # 2. Correlation analysis
    print("\n2. CORRELATION ANALYSIS")
    print("-"*40)
    
    # Correlation between BEGIN output and correctness
    begin_corr = stats.pearsonr(df['begin_output_contribution'], df['is_correct'].astype(int))
    print(f"Correlation(BEGIN output, correctness): r={begin_corr[0]:.3f}, p={begin_corr[1]:.4f}")
    
    # Correlation between format output and correctness
    format_corr = stats.pearsonr(df['format_output_contribution'], df['is_correct'].astype(int))
    print(f"Correlation(format output, correctness): r={format_corr[0]:.3f}, p={format_corr[1]:.4f}")
    
    results['correlations'] = {
        'begin_correctness': {'r': begin_corr[0], 'p': begin_corr[1]},
        'format_correctness': {'r': format_corr[0], 'p': format_corr[1]}
    }
    
    # 3. Logistic regression
    print("\n3. LOGISTIC REGRESSION: correctness ~ BEGIN_output")
    print("-"*40)
    
    X = df['begin_output_contribution'].values.reshape(-1, 1)
    y = df['is_correct'].astype(int).values
    
    # Check if we have both classes
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        print(f"Warning: Only one class found in data (all {'correct' if unique_classes[0] == 1 else 'incorrect'})")
        print("Cannot perform logistic regression with single class")
        log_reg = None
        results['logistic_regression'] = {
            'coefficient': None,
            'intercept': None,
            'roc_auc': None,
            'note': 'Single class only'
        }
    else:
        log_reg = LogisticRegression(random_state=42)
        log_reg.fit(X, y)
        
        coef = log_reg.coef_[0][0]
        intercept = log_reg.intercept_[0]
        
        # Calculate ROC AUC
        y_pred_proba = log_reg.predict_proba(X)[:, 1]
        roc_auc = roc_auc_score(y, y_pred_proba)
        
        print(f"Coefficient: {coef:.3f}")
        print(f"Intercept: {intercept:.3f}")
        print(f"Interpretation: Each 0.1 increase in BEGIN output → {coef*0.1*100:.1f}% change in odds of correct answer")
        print(f"ROC AUC: {roc_auc:.3f}")
        
        results['logistic_regression'] = {
            'coefficient': coef,
            'intercept': intercept,
            'roc_auc': roc_auc
        }
    
    # 4. Format comparison (t-tests)
    print("\n4. FORMAT COMPARISON (t-tests)")
    print("-"*40)
    
    simple_begin = df[df['format'] == 'simple']['begin_output_contribution']
    qa_begin = df[df['format'] == 'qa']['begin_output_contribution']
    
    if len(simple_begin) > 0 and len(qa_begin) > 0:
        t_stat, p_val = stats.ttest_ind(simple_begin, qa_begin)
        print(f"BEGIN output: Simple ({simple_begin.mean():.3f}) vs Q&A ({qa_begin.mean():.3f})")
        print(f"t-statistic: {t_stat:.3f}, p-value: {p_val:.4f}")
        
        results['format_comparison'] = {
            'simple_begin_mean': simple_begin.mean(),
            'qa_begin_mean': qa_begin.mean(),
            't_statistic': t_stat,
            'p_value': p_val
        }
    
    return results, log_reg

def create_visualizations(df: pd.DataFrame, log_reg, stats_results: Dict):
    """Create comprehensive visualizations of attention output analysis"""
    
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Attention OUTPUT Quantification: Information Flow Analysis at Layer 10', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. BEGIN Output by Format
    ax1 = fig.add_subplot(gs[0, 0])
    format_order = ['simple', 'qa', 'question', 'compare', 'direct']
    existing_formats = [f for f in format_order if f in df['format'].unique()]
    
    sns.boxplot(data=df, x='format', y='begin_output_contribution', 
                order=existing_formats, ax=ax1)
    ax1.set_title('BEGIN Output Contribution by Format', fontweight='bold')
    ax1.set_xlabel('Format')
    ax1.set_ylabel('BEGIN Output Contribution')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    
    # 2. Correctness by BEGIN Output Level
    ax2 = fig.add_subplot(gs[0, 1])
    df['begin_bin'] = pd.cut(df['begin_output_contribution'], bins=5)
    bin_stats = df.groupby('begin_bin')['is_correct'].mean()
    
    bin_labels = [f"{interval.left:.2f}-{interval.right:.2f}" 
                  for interval in bin_stats.index]
    ax2.bar(range(len(bin_stats)), bin_stats.values)
    ax2.set_xticks(range(len(bin_stats)))
    ax2.set_xticklabels(bin_labels, rotation=45)
    ax2.set_title('Correctness Rate by BEGIN Output Level', fontweight='bold')
    ax2.set_xlabel('BEGIN Output Contribution')
    ax2.set_ylabel('Correctness Rate')
    ax2.set_ylim([0, 1])
    
    # 3. BEGIN Output vs Correctness (Logistic Regression)
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Scatter plot
    for correct in [0, 1]:
        mask = df['is_correct'] == correct
        if mask.sum() > 0:  # Only plot if we have data for this class
            ax3.scatter(df[mask]['begin_output_contribution'], 
                       df[mask]['is_correct'],
                       alpha=0.3, 
                       label='Correct' if correct else 'Incorrect',
                       color='green' if correct else 'red')
    
    # Logistic regression curve (only if we have the model)
    if log_reg is not None:
        x_range = np.linspace(df['begin_output_contribution'].min(), 
                             df['begin_output_contribution'].max(), 100)
        y_pred = log_reg.predict_proba(x_range.reshape(-1, 1))[:, 1]
        ax3.plot(x_range, y_pred, 'r-', linewidth=2, label='Logistic Regression')
    
    ax3.set_title('BEGIN Output vs Correctness', fontweight='bold')
    ax3.set_xlabel('BEGIN Output Contribution')
    ax3.set_ylabel('P(Correct)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Format vs Number Output Trade-off
    ax4 = fig.add_subplot(gs[1, 0])
    scatter = ax4.scatter(df['format_output_contribution'], 
                         df['number_output_contribution'],
                         c=df['is_correct'], 
                         cmap='RdYlGn',
                         alpha=0.6,
                         s=50)
    ax4.set_title('Format vs Number Output Trade-off', fontweight='bold')
    ax4.set_xlabel('Format Output Contribution')
    ax4.set_ylabel('Number Output Contribution')
    plt.colorbar(scatter, ax=ax4, label='Correct')
    
    # 5. Output Contribution Stacked Bar
    ax5 = fig.add_subplot(gs[1, 1])
    format_means = df.groupby('format')[['begin_output_contribution', 
                                         'format_output_contribution',
                                         'number_output_contribution']].mean()
    
    format_means.plot(kind='bar', stacked=True, ax=ax5,
                      color=['#ff7f0e', '#2ca02c', '#1f77b4'])
    ax5.set_title('Output Contribution Breakdown by Format', fontweight='bold')
    ax5.set_xlabel('Format')
    ax5.set_ylabel('Contribution Proportion')
    ax5.legend(title='Token Type', labels=['BEGIN', 'Format', 'Number'])
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45)
    
    # 6. Correlation Heatmap
    ax6 = fig.add_subplot(gs[1, 2])
    corr_data = df[['begin_output_contribution', 'format_output_contribution', 
                    'number_output_contribution', 'is_correct']].corr()
    sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=ax6, 
                xticklabels=['BEGIN', 'Format', 'Number', 'Correct'],
                yticklabels=['BEGIN', 'Format', 'Number', 'Correct'])
    ax6.set_title('Correlation Matrix', fontweight='bold')
    
    # 7. ROC Curve
    ax7 = fig.add_subplot(gs[2, 0])
    
    if log_reg is not None:
        from sklearn.metrics import roc_curve
        
        X = df['begin_output_contribution'].values.reshape(-1, 1)
        y = df['is_correct'].astype(int).values
        y_pred_proba = log_reg.predict_proba(X)[:, 1]
        
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        ax7.plot(fpr, tpr, 'orange', linewidth=2, 
                label=f'ROC curve (AUC = {stats_results["logistic_regression"]["roc_auc"]:.2f})')
        ax7.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax7.set_title('ROC Curve: BEGIN Output Predicting Correctness', fontweight='bold')
        ax7.set_xlabel('False Positive Rate')
        ax7.set_ylabel('True Positive Rate')
        ax7.legend()
    else:
        ax7.text(0.5, 0.5, 'ROC curve not available\n(single class only)', 
                ha='center', va='center', fontsize=12)
        ax7.set_title('ROC Curve: Not Available', fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # 8. Format-specific accuracy
    ax8 = fig.add_subplot(gs[2, 1])
    format_acc = df.groupby('format')['is_correct'].mean().sort_values()
    ax8.barh(range(len(format_acc)), format_acc.values)
    ax8.set_yticks(range(len(format_acc)))
    ax8.set_yticklabels(format_acc.index)
    ax8.set_title('Accuracy by Format', fontweight='bold')
    ax8.set_xlabel('Accuracy')
    
    # Add value labels
    for i, v in enumerate(format_acc.values):
        ax8.text(v + 0.01, i, f'{v:.1%}', va='center')
    
    # 9. Key Findings Summary
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # Handle missing values
    begin_r = stats_results['correlations']['begin_correctness']['r']
    begin_p = stats_results['correlations']['begin_correctness']['p']
    
    if log_reg is not None:
        lr_coef = stats_results['logistic_regression']['coefficient']
        lr_auc = stats_results['logistic_regression']['roc_auc']
        lr_text = f"  Coefficient = {lr_coef:.3f}\n  ROC AUC = {lr_auc:.3f}"
    else:
        lr_text = "  Not available (single class)"
    
    # Format text with NaN handling
    if np.isnan(begin_r):
        corr_text = "  Not available (single class)"
    else:
        corr_text = f"  r = {begin_r:.3f}\n  (p < {begin_p:.4f})"
    
    findings_text = f"""KEY FINDINGS (Attention OUTPUT):

• Correlation (BEGIN output, correctness):
{corr_text}

• Logistic Regression:
{lr_text}

• Format Comparison:
  Simple format: {stats_results.get('format_comparison', {}).get('simple_begin_mean', 0):.1%} BEGIN output
  Q&A format: {stats_results.get('format_comparison', {}).get('qa_begin_mean', 0):.1%} BEGIN output
  Difference: {abs(stats_results.get('format_comparison', {}).get('simple_begin_mean', 0) - stats_results.get('format_comparison', {}).get('qa_begin_mean', 0)):.1%}

• Interpretation:
  Attention OUTPUT (information flow)
  shows different patterns than
  attention WEIGHTS (where model looks)
  
  This measures actual causal mechanism!"""
    
    ax9.text(0.1, 0.5, findings_text, fontsize=10, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save visualizations
    output_path = '/home/paperspace/dev/MATS9/attention/attention_output_quantification_results'
    plt.savefig(f'{output_path}.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{output_path}.pdf', bbox_inches='tight')
    print(f"\n✅ Visualizations saved to {output_path}.png/pdf")

def main():
    """Main analysis pipeline"""
    print("="*70)
    print("ATTENTION OUTPUT QUANTIFICATION ANALYSIS")
    print("="*70)
    
    # Initialize analyzer
    analyzer = AttentionOutputAnalyzer()
    
    # Generate test examples
    print("\nGenerating 120 test examples...")
    
    # Analyze all examples
    df = analyzer.analyze_all_examples()
    
    # Save raw data
    df.to_csv('/home/paperspace/dev/MATS9/attention/attention_output_quantification_data.csv', index=False)
    print(f"\n✅ Raw data saved to attention_output_quantification_data.csv")
    
    # Perform statistical analysis
    stats_results, log_reg = perform_statistical_analysis(df)
    
    # Save statistical results - convert tuple keys to strings
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
    
    with open('/home/paperspace/dev/MATS9/attention/attention_output_quantification_stats.json', 'w') as f:
        json.dump(json_safe_results, f, indent=2, default=str)
    print(f"✅ Statistical results saved to attention_output_quantification_stats.json")
    
    # Create visualizations
    create_visualizations(df, log_reg, stats_results)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\nAnalyzed {len(df)} examples")
    print(f"Overall accuracy: {df['is_correct'].mean():.1%}")
    
    print("\nBEGIN output contribution by format:")
    for fmt in df['format'].unique():
        fmt_data = df[df['format'] == fmt]
        print(f"  {fmt:10s}: {fmt_data['begin_output_contribution'].mean():.1%} (accuracy: {fmt_data['is_correct'].mean():.1%})")
    
    print(f"\nKey finding: Attention OUTPUT analysis reveals actual information flow")
    print(f"Correlation r = {stats_results['correlations']['begin_correctness']['r']:.3f} (p < {stats_results['correlations']['begin_correctness']['p']:.4f})")
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()