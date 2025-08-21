#!/usr/bin/env python3
"""
Comprehensive Analysis of All Heads in Layer 10
================================================
Analyzes all 32 heads in Layer 10 to determine which ones show
bug-fixing patterns similar to Head 27.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class Layer10HeadAnalyzer:
    """Analyze all heads in Layer 10 for bug-fixing patterns"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        print("="*70)
        print("LAYER 10 ALL HEADS ANALYSIS")
        print("="*70)
        print(f"\nLoading model: {model_name}")
        
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
        
        # Llama-3.1-8B has 32 heads per layer
        self.n_heads = 32
        self.layer_idx = 10
        
    def analyze_single_head(self, prompt: str, head_idx: int) -> Dict:
        """Analyze attention pattern for a single head"""
        
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
        
        # Get attention from Layer 10, specific head
        attention = outputs.attentions[self.layer_idx].cpu()
        attn_weights = attention[0, head_idx, -1, :].numpy()
        
        # Calculate metrics
        begin_attention = attn_weights[0]  # First token is <|begin_of_text|>
        
        # Format tokens (Q, A, :, etc.)
        format_attention = 0
        format_tokens = ['Q', 'A', ':', 'Question', 'Answer', '?']
        for i, token in enumerate(tokens[:len(attn_weights)]):
            if any(fmt in token for fmt in format_tokens):
                format_attention += attn_weights[i]
        
        # Number tokens
        number_attention = 0
        for i, token in enumerate(tokens[:len(attn_weights)]):
            if any(char in token for char in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.']):
                number_attention += attn_weights[i]
        
        # Check if answer is correct
        is_correct = "9.8" in generated[:20] and "bigger" in generated.lower()
        
        return {
            'head_idx': head_idx,
            'begin_attention': float(begin_attention),
            'format_attention': float(format_attention),
            'number_attention': float(number_attention),
            'is_correct': is_correct,
            'generated': generated[:50],
            'attention_weights': attn_weights
        }
    
    def analyze_all_heads(self) -> pd.DataFrame:
        """Analyze all heads in Layer 10 for both formats"""
        
        test_prompts = {
            "simple": "Which is bigger: 9.8 or 9.11?\nAnswer:",
            "qa": "Q: Which is bigger: 9.8 or 9.11?\nA:",
        }
        
        print(f"\nAnalyzing all {self.n_heads} heads in Layer {self.layer_idx}...")
        
        results = []
        
        for format_name, prompt in test_prompts.items():
            print(f"\nProcessing {format_name} format...")
            
            for head_idx in tqdm(range(self.n_heads), desc=f"Analyzing heads ({format_name})"):
                head_result = self.analyze_single_head(prompt, head_idx)
                head_result['format'] = format_name
                results.append(head_result)
        
        return pd.DataFrame(results)
    
    def identify_key_heads(self, df: pd.DataFrame) -> Dict:
        """Identify heads that show strong bug-fixing patterns"""
        
        # Calculate differences between formats for each head
        head_analysis = []
        
        for head_idx in range(self.n_heads):
            simple_data = df[(df['format'] == 'simple') & (df['head_idx'] == head_idx)]
            qa_data = df[(df['format'] == 'qa') & (df['head_idx'] == head_idx)]
            
            if len(simple_data) > 0 and len(qa_data) > 0:
                # Calculate BEGIN attention difference
                begin_diff = simple_data['begin_attention'].values[0] - qa_data['begin_attention'].values[0]
                
                # Check correctness patterns
                simple_correct = simple_data['is_correct'].values[0]
                qa_correct = qa_data['is_correct'].values[0]
                
                # A head shows bug-fixing pattern if:
                # 1. Higher BEGIN attention in simple format
                # 2. Simple format is correct while QA is wrong
                shows_pattern = (begin_diff > 0.1) and simple_correct and not qa_correct
                
                head_analysis.append({
                    'head_idx': head_idx,
                    'begin_attention_simple': simple_data['begin_attention'].values[0],
                    'begin_attention_qa': qa_data['begin_attention'].values[0],
                    'begin_attention_diff': begin_diff,
                    'simple_correct': simple_correct,
                    'qa_correct': qa_correct,
                    'shows_bug_fix_pattern': shows_pattern
                })
        
        analysis_df = pd.DataFrame(head_analysis)
        
        # Identify key heads
        key_heads = analysis_df[analysis_df['shows_bug_fix_pattern']].sort_values(
            'begin_attention_diff', ascending=False
        )
        
        return {
            'analysis_df': analysis_df,
            'key_heads': key_heads,
            'n_heads_with_pattern': len(key_heads),
            'head_27_analysis': analysis_df[analysis_df['head_idx'] == 27].to_dict('records')[0] if 27 < len(analysis_df) else None
        }
    
    def visualize_head_patterns(self, df: pd.DataFrame, head_analysis: Dict):
        """Create comprehensive visualization of all heads"""
        
        fig = plt.figure(figsize=(18, 12))
        
        # Title
        fig.suptitle(f'Layer {self.layer_idx}: All Heads Analysis - Bug-Fixing Patterns',
                     fontsize=16, fontweight='bold', y=0.98)
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. BEGIN attention for all heads
        ax1 = fig.add_subplot(gs[0, :2])
        
        analysis_df = head_analysis['analysis_df']
        x = np.arange(len(analysis_df))
        width = 0.35
        
        ax1.bar(x - width/2, analysis_df['begin_attention_simple'], width, 
                label='Simple Format', color='green', alpha=0.7)
        ax1.bar(x + width/2, analysis_df['begin_attention_qa'], width,
                label='Q&A Format', color='red', alpha=0.7)
        
        # Highlight Head 27
        if 27 < len(analysis_df):
            ax1.axvline(x=27, color='blue', linestyle='--', alpha=0.5, label='Head 27')
        
        ax1.set_xlabel('Head Index')
        ax1.set_ylabel('BEGIN Token Attention')
        ax1.set_title('BEGIN Token Attention by Head and Format')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Attention difference heatmap
        ax2 = fig.add_subplot(gs[0, 2])
        
        diff_values = analysis_df['begin_attention_diff'].values.reshape(-1, 1)
        sns.heatmap(diff_values, cmap='RdBu_r', center=0, 
                    yticklabels=range(self.n_heads),
                    xticklabels=['Δ BEGIN'],
                    ax=ax2, cbar_kws={'label': 'Simple - QA'})
        ax2.set_title('BEGIN Attention Difference')
        ax2.set_ylabel('Head Index')
        
        # 3. Key heads with bug-fixing pattern
        ax3 = fig.add_subplot(gs[1, :])
        
        key_heads = head_analysis['key_heads']
        if len(key_heads) > 0:
            ax3.barh(range(len(key_heads)), key_heads['begin_attention_diff'].values)
            ax3.set_yticks(range(len(key_heads)))
            ax3.set_yticklabels([f"Head {h}" for h in key_heads['head_idx'].values])
            ax3.set_xlabel('BEGIN Attention Difference (Simple - QA)')
            ax3.set_title(f'Heads Showing Bug-Fixing Pattern (n={len(key_heads)})')
            ax3.grid(True, alpha=0.3)
            
            # Highlight Head 27 if present
            if 27 in key_heads['head_idx'].values:
                idx = key_heads['head_idx'].values.tolist().index(27)
                ax3.barh(idx, key_heads['begin_attention_diff'].values[idx], color='blue', alpha=0.7)
        else:
            ax3.text(0.5, 0.5, 'No heads show clear bug-fixing pattern',
                    ha='center', va='center', fontsize=12)
            ax3.set_title('Heads Showing Bug-Fixing Pattern')
        
        # 4. Scatter plot: BEGIN attention vs correctness
        ax4 = fig.add_subplot(gs[2, 0])
        
        for format_name, color in [('simple', 'green'), ('qa', 'red')]:
            format_data = df[df['format'] == format_name]
            ax4.scatter(format_data['begin_attention'], 
                       format_data['is_correct'].astype(int),
                       label=format_name.upper(), alpha=0.6, s=50, color=color)
        
        # Highlight Head 27
        head27_data = df[df['head_idx'] == 27]
        if len(head27_data) > 0:
            ax4.scatter(head27_data['begin_attention'],
                       head27_data['is_correct'].astype(int),
                       s=200, marker='*', color='blue', label='Head 27', zorder=5)
        
        ax4.set_xlabel('BEGIN Token Attention')
        ax4.set_ylabel('Correct Answer')
        ax4.set_title('BEGIN Attention vs Correctness')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Format vs Number attention trade-off
        ax5 = fig.add_subplot(gs[2, 1])
        
        ax5.scatter(df['format_attention'], df['number_attention'],
                   c=df['is_correct'], cmap='RdYlGn', alpha=0.6, s=30)
        
        # Highlight Head 27
        if len(head27_data) > 0:
            ax5.scatter(head27_data['format_attention'],
                       head27_data['number_attention'],
                       s=200, marker='*', color='blue', zorder=5)
        
        ax5.set_xlabel('Format Token Attention')
        ax5.set_ylabel('Number Token Attention')
        ax5.set_title('Attention Distribution Trade-off')
        
        # 6. Summary statistics
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        
        # Create summary
        head27_info = head_analysis['head_27_analysis']
        if head27_info:
            head27_text = f"""Head 27:
  BEGIN (Simple): {head27_info['begin_attention_simple']:.1%}
  BEGIN (Q&A): {head27_info['begin_attention_qa']:.1%}
  Difference: {head27_info['begin_attention_diff']:.1%}
  Pattern: {'YES' if head27_info['shows_bug_fix_pattern'] else 'NO'}"""
        else:
            head27_text = "Head 27: Not found"
        
        summary_text = f"""
KEY FINDINGS:
{'='*30}

Total heads analyzed: {self.n_heads}
Heads with bug-fix pattern: {head_analysis['n_heads_with_pattern']}

Top 3 heads by BEGIN difference:
{analysis_df.nlargest(3, 'begin_attention_diff')[['head_idx', 'begin_attention_diff']].to_string()}

{head27_text}

Conclusion:
{'Multiple heads show bug-fixing' if head_analysis['n_heads_with_pattern'] > 1 else 'Only Head 27 shows'}
{'patterns similar to Head 27' if head_analysis['n_heads_with_pattern'] > 1 else 'the bug-fixing pattern'}
"""
        
        ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes,
                fontsize=9, verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        
        # Save
        plt.savefig('/home/paperspace/dev/MATS9/attention/layer10_all_heads_analysis.png',
                    dpi=150, bbox_inches='tight')
        plt.savefig('/home/paperspace/dev/MATS9/attention/layer10_all_heads_analysis.pdf',
                    bbox_inches='tight')
        
        print("\n✅ Visualizations saved")
        
    def generate_report(self, df: pd.DataFrame, head_analysis: Dict) -> str:
        """Generate detailed report of findings"""
        
        report = []
        report.append("="*70)
        report.append(f"LAYER {self.layer_idx} - ALL HEADS ANALYSIS REPORT")
        report.append("="*70)
        
        report.append("\n## SUMMARY")
        report.append("-"*40)
        report.append(f"Total heads analyzed: {self.n_heads}")
        report.append(f"Heads showing bug-fix pattern: {head_analysis['n_heads_with_pattern']}")
        
        report.append("\n## HEAD-BY-HEAD ANALYSIS")
        report.append("-"*40)
        
        analysis_df = head_analysis['analysis_df']
        
        # Sort by BEGIN attention difference
        sorted_df = analysis_df.sort_values('begin_attention_diff', ascending=False)
        
        report.append("\nTop 5 heads by BEGIN attention difference:")
        for _, row in sorted_df.head(5).iterrows():
            report.append(f"\nHead {row['head_idx']:2d}:")
            report.append(f"  BEGIN Simple: {row['begin_attention_simple']:6.1%}")
            report.append(f"  BEGIN Q&A:    {row['begin_attention_qa']:6.1%}")
            report.append(f"  Difference:   {row['begin_attention_diff']:+6.1%}")
            report.append(f"  Bug-fix pattern: {'YES ✓' if row['shows_bug_fix_pattern'] else 'NO'}")
        
        report.append("\n## HEAD 27 SPECIAL ANALYSIS")
        report.append("-"*40)
        
        if head_analysis['head_27_analysis']:
            h27 = head_analysis['head_27_analysis']
            report.append(f"BEGIN attention (Simple): {h27['begin_attention_simple']:.1%}")
            report.append(f"BEGIN attention (Q&A):    {h27['begin_attention_qa']:.1%}")
            report.append(f"Difference:               {h27['begin_attention_diff']:+.1%}")
            report.append(f"Shows bug-fix pattern:    {'YES ✓' if h27['shows_bug_fix_pattern'] else 'NO'}")
            
            # Rank among all heads
            rank = (analysis_df['begin_attention_diff'] > h27['begin_attention_diff']).sum() + 1
            report.append(f"Rank by BEGIN difference: {rank}/{self.n_heads}")
        else:
            report.append("Head 27 not found in this layer")
        
        report.append("\n## HEADS WITH BUG-FIXING PATTERN")
        report.append("-"*40)
        
        key_heads = head_analysis['key_heads']
        if len(key_heads) > 0:
            report.append(f"\n{len(key_heads)} heads show the bug-fixing pattern:")
            for _, row in key_heads.iterrows():
                report.append(f"  Head {row['head_idx']:2d}: Δ={row['begin_attention_diff']:+.1%}")
        else:
            report.append("\nNo heads show clear bug-fixing pattern")
        
        report.append("\n## CONCLUSION")
        report.append("-"*40)
        
        if head_analysis['n_heads_with_pattern'] == 0:
            report.append("• No heads in Layer 10 show the bug-fixing pattern")
            report.append("• This suggests the bug mechanism may be elsewhere")
        elif head_analysis['n_heads_with_pattern'] == 1:
            if head_analysis['key_heads']['head_idx'].values[0] == 27:
                report.append("• ONLY Head 27 shows the bug-fixing pattern")
                report.append("• This makes Head 27 uniquely important")
            else:
                report.append(f"• Only Head {head_analysis['key_heads']['head_idx'].values[0]} shows the pattern")
                report.append("• Head 27 does not show the expected pattern")
        else:
            report.append(f"• {head_analysis['n_heads_with_pattern']} heads show bug-fixing patterns")
            if 27 in head_analysis['key_heads']['head_idx'].values:
                report.append("• Head 27 is among them but not unique")
            else:
                report.append("• Head 27 does NOT show the pattern")
            report.append("• The bug-fixing mechanism is distributed")
        
        report.append("\n" + "="*70)
        
        return "\n".join(report)
    
    def save_results(self, df: pd.DataFrame, head_analysis: Dict, report: str):
        """Save all results to files"""
        
        # Save raw data
        df.to_csv('/home/paperspace/dev/MATS9/attention/layer10_all_heads_data.csv', index=False)
        
        # Save analysis
        head_analysis['analysis_df'].to_csv(
            '/home/paperspace/dev/MATS9/attention/layer10_heads_analysis.csv', 
            index=False
        )
        
        # Save key findings as JSON
        key_findings = {
            'n_heads_analyzed': self.n_heads,
            'n_heads_with_pattern': head_analysis['n_heads_with_pattern'],
            'key_heads': head_analysis['key_heads']['head_idx'].tolist() if len(head_analysis['key_heads']) > 0 else [],
            'head_27_shows_pattern': head_analysis['head_27_analysis']['shows_bug_fix_pattern'] if head_analysis['head_27_analysis'] else False,
            'head_27_rank': int((head_analysis['analysis_df']['begin_attention_diff'] > 
                                 head_analysis['head_27_analysis']['begin_attention_diff']).sum() + 1) if head_analysis['head_27_analysis'] else None
        }
        
        with open('/home/paperspace/dev/MATS9/attention/layer10_key_findings.json', 'w') as f:
            json.dump(key_findings, f, indent=2)
        
        # Save report
        with open('/home/paperspace/dev/MATS9/attention/layer10_all_heads_report.txt', 'w') as f:
            f.write(report)
        
        print("✅ Results saved to:")
        print("   - layer10_all_heads_data.csv")
        print("   - layer10_heads_analysis.csv")
        print("   - layer10_key_findings.json")
        print("   - layer10_all_heads_report.txt")


def main():
    """Run comprehensive Layer 10 heads analysis"""
    
    print("\n" + "="*70)
    print("STARTING COMPREHENSIVE LAYER 10 ANALYSIS")
    print("="*70)
    
    # Initialize analyzer
    analyzer = Layer10HeadAnalyzer()
    
    # Analyze all heads
    df = analyzer.analyze_all_heads()
    
    # Identify key heads
    head_analysis = analyzer.identify_key_heads(df)
    
    # Generate report
    report = analyzer.generate_report(df, head_analysis)
    print("\n" + report)
    
    # Create visualizations
    analyzer.visualize_head_patterns(df, head_analysis)
    
    # Save results
    analyzer.save_results(df, head_analysis, report)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    # Print key finding
    if head_analysis['n_heads_with_pattern'] == 1 and 27 in head_analysis['key_heads']['head_idx'].values:
        print("\n🎯 KEY FINDING: Only Head 27 shows the bug-fixing pattern!")
    elif head_analysis['n_heads_with_pattern'] > 1:
        print(f"\n📊 KEY FINDING: {head_analysis['n_heads_with_pattern']} heads show bug-fixing patterns")
    else:
        print("\n❓ KEY FINDING: No clear bug-fixing pattern found in Layer 10 heads")


if __name__ == "__main__":
    main()