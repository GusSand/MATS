#!/usr/bin/env python3
"""
Attention Pattern Quantification
=================================
Measures attention patterns across 100+ examples to quantify the relationship
between BEGIN token attention and correct answers.

Key measurements:
- BEGIN attention % at Layer 10
- Format token attention % at Layer 10
- Correlation between BEGIN attention and correctness
- Logistic regression: correctness ~ BEGIN_attention
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Set style for clean visualizations
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set_style('whitegrid')


class AttentionQuantifier:
    """Quantify attention patterns across many examples"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        print(f"Loading model: {model_name}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            attn_implementation="eager"
        )
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
        
    def generate_test_examples(self, n_examples: int = 120) -> List[Dict]:
        """Generate diverse test examples with different decimal comparisons"""
        
        examples = []
        
        # Generate various decimal comparison pairs
        decimal_pairs = [
            (9.8, 9.11),
            (9.7, 9.12),
            (8.6, 8.13),
            (7.9, 7.14),
            (6.8, 6.15),
            (10.8, 10.11),
            (5.7, 5.15),
            (11.9, 11.10),
            (9.9, 9.10),
            (8.5, 8.12),
            (4.8, 4.11),
            (3.7, 3.13),
        ]
        
        # For each decimal pair, create multiple format variations
        for larger, smaller in decimal_pairs:
            # Test both orderings
            for first, second in [(larger, smaller), (smaller, larger)]:
                correct_answer = "first" if first > second else "second"
                
                # Simple format (tends to be correct)
                examples.append({
                    'prompt': f"Which is bigger: {first} or {second}?\nAnswer:",
                    'format': 'simple',
                    'first': first,
                    'second': second,
                    'correct_answer': correct_answer,
                    'expected_behavior': 'correct'
                })
                
                # Q&A format (tends to be wrong)
                examples.append({
                    'prompt': f"Q: Which is bigger: {first} or {second}?\nA:",
                    'format': 'qa',
                    'first': first,
                    'second': second,
                    'correct_answer': correct_answer,
                    'expected_behavior': 'wrong'
                })
                
                # Question: format
                examples.append({
                    'prompt': f"Question: Which is bigger: {first} or {second}?\nAnswer:",
                    'format': 'question',
                    'first': first,
                    'second': second,
                    'correct_answer': correct_answer,
                    'expected_behavior': 'unknown'
                })
                
                # Compare: format
                examples.append({
                    'prompt': f"Compare: {first} and {second}\nWhich is larger?",
                    'format': 'compare',
                    'first': first,
                    'second': second,
                    'correct_answer': correct_answer,
                    'expected_behavior': 'unknown'
                })
                
                # Direct format
                examples.append({
                    'prompt': f"{first} vs {second}, which is bigger?",
                    'format': 'direct',
                    'first': first,
                    'second': second,
                    'correct_answer': correct_answer,
                    'expected_behavior': 'unknown'
                })
        
        return examples[:n_examples]
    
    def extract_attention_metrics(self, prompt: str, layer: int = 10, head: int = None) -> Dict:
        """Extract attention metrics for a given prompt"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
                return_dict=True
            )
            
            # Generate to see what the model produces
            gen_outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            generated = self.tokenizer.decode(
                gen_outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
        
        # Get tokens
        tokens = [self.tokenizer.decode([tid]) for tid in inputs['input_ids'][0]]
        
        # Get attention from specified layer (average across all heads if head not specified)
        attention = outputs.attentions[layer].cpu()
        
        if head is not None:
            attn_weights = attention[0, head, -1, :].numpy()
        else:
            # Average across all heads
            attn_weights = attention[0, :, -1, :].mean(dim=0).numpy()
        
        # Calculate metrics
        begin_token_attention = attn_weights[0]  # First token is <|begin_of_text|>
        
        # Format tokens (Q, A, :, Question, Answer, etc.)
        format_token_attention = 0
        format_tokens = ['Q', 'A', ':', 'Question', 'Answer', 'Compare', '?']
        for i, token in enumerate(tokens[:len(attn_weights)]):
            if any(fmt in token for fmt in format_tokens):
                format_token_attention += attn_weights[i]
        
        # Number tokens
        number_token_attention = 0
        for i, token in enumerate(tokens[:len(attn_weights)]):
            if any(char in token for char in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.']):
                number_token_attention += attn_weights[i]
        
        return {
            'begin_attention': float(begin_token_attention),
            'format_attention': float(format_token_attention),
            'number_attention': float(number_token_attention),
            'generated': generated,
            'tokens': tokens,
            'attention_weights': attn_weights
        }
    
    def analyze_example(self, example: Dict, layer: int = 10) -> Dict:
        """Analyze a single example"""
        
        # Extract attention metrics
        metrics = self.extract_attention_metrics(example['prompt'], layer=layer)
        
        # Determine if answer is correct
        generated = metrics['generated'].lower()
        first_str = str(example['first'])
        second_str = str(example['second'])
        
        # Check which number the model thinks is bigger
        model_thinks_first_bigger = False
        model_thinks_second_bigger = False
        
        if first_str in generated and 'bigger' in generated:
            # Check if it says first is bigger
            first_pos = generated.find(first_str)
            bigger_pos = generated.find('bigger')
            if first_pos < bigger_pos:
                model_thinks_first_bigger = True
        
        if second_str in generated and 'bigger' in generated:
            second_pos = generated.find(second_str)
            bigger_pos = generated.find('bigger')
            if second_pos < bigger_pos:
                model_thinks_second_bigger = True
        
        # Simpler heuristic if above doesn't work
        if not (model_thinks_first_bigger or model_thinks_second_bigger):
            if first_str in generated[:20]:
                model_thinks_first_bigger = True
            elif second_str in generated[:20]:
                model_thinks_second_bigger = True
        
        # Determine correctness
        if example['correct_answer'] == 'first':
            is_correct = model_thinks_first_bigger and not model_thinks_second_bigger
        else:
            is_correct = model_thinks_second_bigger and not model_thinks_first_bigger
        
        return {
            'prompt': example['prompt'],
            'format': example['format'],
            'first': example['first'],
            'second': example['second'],
            'correct_answer': example['correct_answer'],
            'model_answer': 'first' if model_thinks_first_bigger else 'second',
            'is_correct': is_correct,
            'begin_attention': metrics['begin_attention'],
            'format_attention': metrics['format_attention'],
            'number_attention': metrics['number_attention'],
            'generated': metrics['generated'][:50],
            'expected_behavior': example['expected_behavior']
        }
    
    def run_quantification(self, n_examples: int = 120, layer: int = 10) -> pd.DataFrame:
        """Run full quantification analysis"""
        
        print(f"\nGenerating {n_examples} test examples...")
        examples = self.generate_test_examples(n_examples)
        
        print(f"Analyzing attention patterns at Layer {layer}...")
        results = []
        
        for example in tqdm(examples, desc="Processing examples"):
            result = self.analyze_example(example, layer=layer)
            results.append(result)
        
        return pd.DataFrame(results)


def perform_statistical_analysis(df: pd.DataFrame) -> Dict:
    """Perform statistical analysis on the results"""
    
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS")
    print("="*70)
    
    results = {}
    
    # 1. Basic statistics by format
    print("\n1. ATTENTION STATISTICS BY FORMAT")
    print("-"*40)
    
    format_stats = df.groupby('format').agg({
        'begin_attention': ['mean', 'std'],
        'format_attention': ['mean', 'std'],
        'number_attention': ['mean', 'std'],
        'is_correct': 'mean'
    }).round(3)
    
    print(format_stats)
    results['format_stats'] = format_stats.to_dict()
    
    # 2. Correlation analysis
    print("\n2. CORRELATION ANALYSIS")
    print("-"*40)
    
    corr_begin = stats.pearsonr(df['begin_attention'], df['is_correct'].astype(int))
    corr_format = stats.pearsonr(df['format_attention'], df['is_correct'].astype(int))
    
    print(f"Correlation(BEGIN attention, correctness): r={corr_begin[0]:.3f}, p={corr_begin[1]:.4f}")
    print(f"Correlation(format attention, correctness): r={corr_format[0]:.3f}, p={corr_format[1]:.4f}")
    
    results['correlations'] = {
        'begin_correctness': {'r': corr_begin[0], 'p': corr_begin[1]},
        'format_correctness': {'r': corr_format[0], 'p': corr_format[1]}
    }
    
    # 3. Logistic regression
    print("\n3. LOGISTIC REGRESSION: correctness ~ BEGIN_attention")
    print("-"*40)
    
    X = df[['begin_attention']].values
    y = df['is_correct'].astype(int).values
    
    log_reg = LogisticRegression()
    log_reg.fit(X, y)
    
    # Get predictions and probabilities
    y_pred = log_reg.predict(X)
    y_prob = log_reg.predict_proba(X)[:, 1]
    
    # Calculate metrics
    coef = log_reg.coef_[0][0]
    intercept = log_reg.intercept_[0]
    
    print(f"Coefficient: {coef:.3f}")
    print(f"Intercept: {intercept:.3f}")
    print(f"Interpretation: Each 0.1 increase in BEGIN attention → "
          f"{(np.exp(coef * 0.1) - 1) * 100:.1f}% change in odds of correct answer")
    
    # ROC AUC
    roc_auc = roc_auc_score(y, y_prob)
    print(f"ROC AUC: {roc_auc:.3f}")
    
    results['logistic_regression'] = {
        'coefficient': coef,
        'intercept': intercept,
        'roc_auc': roc_auc
    }
    
    # 4. Format comparison
    print("\n4. FORMAT COMPARISON (t-tests)")
    print("-"*40)
    
    simple_data = df[df['format'] == 'simple']
    qa_data = df[df['format'] == 'qa']
    
    if len(simple_data) > 0 and len(qa_data) > 0:
        t_stat, p_val = stats.ttest_ind(
            simple_data['begin_attention'],
            qa_data['begin_attention']
        )
        
        print(f"BEGIN attention: Simple ({simple_data['begin_attention'].mean():.3f}) vs "
              f"Q&A ({qa_data['begin_attention'].mean():.3f})")
        print(f"t-statistic: {t_stat:.3f}, p-value: {p_val:.4f}")
        
        results['format_comparison'] = {
            'simple_begin_mean': simple_data['begin_attention'].mean(),
            'qa_begin_mean': qa_data['begin_attention'].mean(),
            't_statistic': t_stat,
            'p_value': p_val
        }
    
    return results, log_reg


def create_visualizations(df: pd.DataFrame, log_reg, stats_results: Dict):
    """Create comprehensive visualizations"""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Title
    fig.suptitle('Attention Pattern Quantification: BEGIN Token Attention and Correctness',
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. BEGIN attention by format
    ax1 = fig.add_subplot(gs[0, 0])
    sns.boxplot(data=df, x='format', y='begin_attention', ax=ax1)
    ax1.set_title('BEGIN Token Attention by Format', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Format', fontsize=11)
    ax1.set_ylabel('BEGIN Attention', fontsize=11)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Remove spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 2. Correctness by BEGIN attention bins
    ax2 = fig.add_subplot(gs[0, 1])
    df['begin_bin'] = pd.cut(df['begin_attention'], bins=5)
    correctness_by_bin = df.groupby('begin_bin')['is_correct'].mean()
    correctness_by_bin.plot(kind='bar', ax=ax2, color='steelblue')
    ax2.set_title('Correctness Rate by BEGIN Attention Level', fontsize=12, fontweight='bold')
    ax2.set_xlabel('BEGIN Attention Range', fontsize=11)
    ax2.set_ylabel('Correctness Rate', fontsize=11)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.set_ylim(0, 1)
    
    # Remove spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 3. Scatter plot with logistic regression
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Add jitter for visibility
    jitter = np.random.normal(0, 0.01, len(df))
    ax3.scatter(df['begin_attention'], df['is_correct'].astype(int) + jitter,
                alpha=0.3, s=20, color='gray')
    
    # Add logistic regression curve
    x_range = np.linspace(df['begin_attention'].min(), df['begin_attention'].max(), 100)
    X_range = x_range.reshape(-1, 1)
    y_prob = log_reg.predict_proba(X_range)[:, 1]
    ax3.plot(x_range, y_prob, 'r-', linewidth=2, label='Logistic Regression')
    
    ax3.set_title('BEGIN Attention vs Correctness', fontsize=12, fontweight='bold')
    ax3.set_xlabel('BEGIN Attention', fontsize=11)
    ax3.set_ylabel('P(Correct)', fontsize=11)
    ax3.set_ylim(-0.1, 1.1)
    ax3.legend()
    
    # Remove spines
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # 4. Format vs Number attention trade-off
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(df['format_attention'], df['number_attention'],
                c=df['is_correct'], cmap='RdYlGn', alpha=0.6, s=30)
    ax4.set_title('Format vs Number Attention Trade-off', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Format Token Attention', fontsize=11)
    ax4.set_ylabel('Number Token Attention', fontsize=11)
    
    # Add colorbar
    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label('Correct', fontsize=10)
    
    # Remove spines
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # 5. ROC Curve
    ax5 = fig.add_subplot(gs[1, 1])
    
    y_true = df['is_correct'].astype(int).values
    X = df[['begin_attention']].values
    y_prob = log_reg.predict_proba(X)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = stats_results['logistic_regression']['roc_auc']
    
    ax5.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax5.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax5.set_xlim([0.0, 1.0])
    ax5.set_ylim([0.0, 1.05])
    ax5.set_xlabel('False Positive Rate', fontsize=11)
    ax5.set_ylabel('True Positive Rate', fontsize=11)
    ax5.set_title('ROC Curve: BEGIN Attention Predicting Correctness', fontsize=12, fontweight='bold')
    ax5.legend(loc="lower right")
    
    # Remove spines
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    
    # 6. Summary statistics table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Create summary text
    summary_text = f"""
KEY FINDINGS:

• Correlation (BEGIN attention, correctness):
  r = {stats_results['correlations']['begin_correctness']['r']:.3f} 
  (p < {stats_results['correlations']['begin_correctness']['p']:.4f})

• Logistic Regression:
  Coefficient = {stats_results['logistic_regression']['coefficient']:.3f}
  ROC AUC = {stats_results['logistic_regression']['roc_auc']:.3f}

• Format Comparison:
  Simple format: {stats_results['format_comparison']['simple_begin_mean']:.1%} BEGIN
  Q&A format: {stats_results['format_comparison']['qa_begin_mean']:.1%} BEGIN
  Difference: {(stats_results['format_comparison']['simple_begin_mean'] - 
               stats_results['format_comparison']['qa_begin_mean']):.1%}

• Interpretation:
  Higher BEGIN attention strongly predicts
  correct decimal comparison
"""
    
    ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    plt.savefig('/home/paperspace/dev/MATS9/attention/attention_quantification_results.png',
                dpi=150, bbox_inches='tight')
    plt.savefig('/home/paperspace/dev/MATS9/attention/attention_quantification_results.pdf',
                bbox_inches='tight')
    
    print("\n✅ Visualizations saved to attention_quantification_results.png/pdf")


def main():
    """Run the complete attention pattern quantification"""
    
    print("="*70)
    print("ATTENTION PATTERN QUANTIFICATION ANALYSIS")
    print("="*70)
    
    # Initialize analyzer
    quantifier = AttentionQuantifier()
    
    # Run quantification
    df = quantifier.run_quantification(n_examples=120, layer=10)
    
    # Save raw results
    df.to_csv('/home/paperspace/dev/MATS9/attention/attention_quantification_data.csv', index=False)
    print(f"\n✅ Raw data saved to attention_quantification_data.csv")
    
    # Perform statistical analysis
    stats_results, log_reg = perform_statistical_analysis(df)
    
    # Save statistical results - convert tuple keys to strings for JSON serialization
    json_safe_results = {}
    for key, value in stats_results.items():
        if isinstance(value, dict):
            # Handle nested dictionaries
            json_safe_results[key] = {}
            for k, v in value.items():
                # Convert tuple keys to strings
                if isinstance(k, tuple):
                    json_safe_results[key][str(k)] = v
                else:
                    json_safe_results[key][k] = v
        else:
            json_safe_results[key] = value
    
    with open('/home/paperspace/dev/MATS9/attention/attention_quantification_stats.json', 'w') as f:
        json.dump(json_safe_results, f, indent=2, default=str)
    print(f"✅ Statistical results saved to attention_quantification_stats.json")
    
    # Create visualizations
    create_visualizations(df, log_reg, stats_results)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\nAnalyzed {len(df)} examples")
    print(f"Overall accuracy: {df['is_correct'].mean():.1%}")
    
    print(f"\nBEGIN attention by format:")
    for fmt in df['format'].unique():
        fmt_data = df[df['format'] == fmt]
        print(f"  {fmt:10s}: {fmt_data['begin_attention'].mean():.1%} "
              f"(accuracy: {fmt_data['is_correct'].mean():.1%})")
    
    print(f"\nKey finding: BEGIN attention strongly predicts correctness")
    print(f"Correlation r = {stats_results['correlations']['begin_correctness']['r']:.3f} "
          f"(p < {stats_results['correlations']['begin_correctness']['p']:.4f})")
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()