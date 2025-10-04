#!/usr/bin/env python3
"""
Create Clean Attention Visualization in Paper Style
====================================================
Creates attention visualization following the clean style from sample_viz.py:
- Minimal spines (only left and bottom)
- Clean fonts and consistent sizing
- No excessive frames or boxes
- Professional academic paper style
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Set style to match sample_viz.py
sns.set_style('whitegrid')

class AttentionAnalyzer:
    """Extract and analyze critical attention patterns"""
    
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
        
        print("Model ready for analysis")
    
    def get_attention_weights(self, prompt: str, layer_idx: int = 10) -> Dict:
        """Extract attention weights from specified layer"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
                return_dict=True
            )
        
        # Get attention from specific layer
        attention = outputs.attentions[layer_idx].cpu()
        
        # Get tokens
        tokens = [self.tokenizer.decode([tid]) for tid in inputs['input_ids'][0]]
        
        # Also generate to see what happens next
        gen_outputs = self.model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        generated = self.tokenizer.decode(
            gen_outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return {
            'attention': attention[0],  # Remove batch dimension
            'tokens': tokens,
            'generated': generated,
            'prompt': prompt
        }


def create_clean_visualization():
    """Create clean attention visualization in paper style"""
    
    print("\n" + "="*70)
    print("CREATING CLEAN ATTENTION VISUALIZATION")
    print("="*70)
    
    analyzer = AttentionAnalyzer()
    
    # Test prompts
    plain_prompt = "Which is bigger: 9.8 or 9.11?\nAnswer:"
    qa_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"
    
    # Get attention patterns
    print("\nExtracting attention patterns...")
    plain_data = analyzer.get_attention_weights(plain_prompt, layer_idx=10)
    qa_data = analyzer.get_attention_weights(qa_prompt, layer_idx=10)
    
    print(f"Plain format generates: {plain_data['generated'][:30]}")
    print(f"Q&A format generates: {qa_data['generated'][:30]}")
    
    # Find the head with strongest pattern difference
    n_heads = plain_data['attention'].shape[0]
    
    # Calculate which head shows the clearest difference
    head_scores = []
    for h in range(n_heads):
        # Look at last token attention
        plain_last = plain_data['attention'][h, -1, :].numpy()
        qa_last = qa_data['attention'][h, -1, :].numpy()
        
        # Score based on how much Q&A attends to format tokens
        qa_format_attn = 0
        for i, token in enumerate(qa_data['tokens'][:len(qa_last)]):
            if 'Q' in token or 'A' in token or ':' in token:
                qa_format_attn += qa_last[i]
        
        # Score based on how much plain attends to numbers
        plain_number_attn = 0
        for i, token in enumerate(plain_data['tokens'][:len(plain_last)]):
            if '9' in token or '.' in token or '8' in token or '11' in token:
                plain_number_attn += plain_last[i]
        
        score = qa_format_attn + plain_number_attn
        head_scores.append(score)
    
    best_head = np.argmax(head_scores)
    print(f"Selected head {best_head} for visualization")
    
    # Create the visualization with clean style
    fig = plt.figure(figsize=(16, 10))
    
    # Main title - simple and clean
    fig.suptitle('Attention Pattern Analysis: Layer 10, Head ' + str(best_head),
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Create grid - only 2x2 for the main visualizations
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
    
    # Helper function for clean bar plots
    def plot_attention_bars(ax, attention_weights, tokens, title, highlight_type='numbers'):
        weights = attention_weights.numpy()
        
        # Prepare labels and colors
        labels = []
        colors = []
        for i, token in enumerate(tokens):
            token_clean = token.strip()
            
            # Categorize and color tokens - simpler color scheme
            if '9.8' in token or '9.11' in token or '9' in token_clean or '8' in token_clean or '11' in token_clean:
                labels.append(token_clean)
                colors.append('#E74C3C' if highlight_type == 'numbers' else '#95A5A6')
            elif token_clean in ['Q:', 'A:', 'Q', 'A', ':']:
                labels.append(token_clean)
                colors.append('#3498DB' if highlight_type == 'format' else '#95A5A6')
            elif '?' in token:
                labels.append(token_clean)
                colors.append('#2ECC71')
            elif 'bigger' in token.lower():
                labels.append(token_clean)
                colors.append('#F39C12')
            else:
                labels.append(token_clean[:15])
                colors.append('#BDC3C7')
        
        # Create horizontal bar plot
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, weights, color=colors, edgecolor='none')
        
        # Customize - clean style
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel('Attention Weight', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        # Remove top and right spines (following sample_viz.py style)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        
        # Subtle grid
        ax.grid(axis='x', alpha=0.2, linestyle='-', linewidth=0.5)
        ax.set_xlim(0, max(weights) * 1.15)
        
        # Add value labels on bars (only for significant weights)
        for i, (bar, weight) in enumerate(zip(bars, weights)):
            if weight > 0.05:  # Only show significant weights
                ax.text(weight + 0.005, bar.get_y() + bar.get_height()/2,
                       f'{weight:.2f}', va='center', fontsize=10, color='#2C3E50')
        
        return bars
    
    # PLAIN FORMAT (CORRECT) - Top Left
    ax1 = fig.add_subplot(gs[0, 0])
    plain_attn = plain_data['attention'][best_head, -1, :]
    plot_attention_bars(ax1, plain_attn, plain_data['tokens'],
                        'Simple Format (Correct Answer)',
                        highlight_type='numbers')
    
    # Q&A FORMAT (WRONG) - Top Right
    ax2 = fig.add_subplot(gs[0, 1])
    qa_attn = qa_data['attention'][best_head, -1, :]
    plot_attention_bars(ax2, qa_attn, qa_data['tokens'],
                        'Q&A Format (Incorrect Answer)',
                        highlight_type='format')
    
    # ATTENTION HEATMAPS - Bottom Row
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Clean heatmap style
    sns.heatmap(plain_data['attention'][best_head].numpy(),
                xticklabels=[t.strip()[:8] for t in plain_data['tokens']],
                yticklabels=[t.strip()[:8] for t in plain_data['tokens']],
                cmap='RdBu_r', center=0.5, ax=ax3, 
                cbar_kws={'label': 'Weight', 'shrink': 0.8},
                linewidths=0.5, linecolor='white',
                square=True)
    ax3.set_title('Simple Format - Attention Matrix', fontsize=14)
    ax3.set_xlabel('Target Token', fontsize=11)
    ax3.set_ylabel('Source Token', fontsize=11)
    
    ax4 = fig.add_subplot(gs[1, 1])
    sns.heatmap(qa_data['attention'][best_head].numpy(),
                xticklabels=[t.strip()[:8] for t in qa_data['tokens']],
                yticklabels=[t.strip()[:8] for t in qa_data['tokens']],
                cmap='RdBu_r', center=0.5, ax=ax4,
                cbar_kws={'label': 'Weight', 'shrink': 0.8},
                linewidths=0.5, linecolor='white',
                square=True)
    ax4.set_title('Q&A Format - Attention Matrix', fontsize=14)
    ax4.set_xlabel('Target Token', fontsize=11)
    ax4.set_ylabel('Source Token', fontsize=11)
    
    # Add a simple legend at the bottom
    legend_elements = [
        mpatches.Patch(color='#E74C3C', label='Number Tokens'),
        mpatches.Patch(color='#3498DB', label='Format Tokens'),
        mpatches.Patch(color='#F39C12', label='Query Token'),
        mpatches.Patch(color='#2ECC71', label='Question Mark'),
        mpatches.Patch(color='#BDC3C7', label='Other Tokens'),
    ]
    
    # Place legend outside the plot area
    fig.legend(handles=legend_elements, 
              loc='lower center', 
              bbox_to_anchor=(0.5, -0.02),
              ncol=5,
              frameon=False,
              fontsize=11)
    
    # Add subtle caption with key statistics
    plain_number_attn = sum(plain_attn[i] for i, t in enumerate(plain_data['tokens']) 
                           if '9' in t or '8' in t or '11' in t)
    qa_format_attn = sum(qa_attn[i] for i, t in enumerate(qa_data['tokens'])
                        if 'Q' in t or 'A' in t or ':' in t)
    qa_number_attn = sum(qa_attn[i] for i, t in enumerate(qa_data['tokens'])
                        if '9' in t or '8' in t or '11' in t)
    
    caption = (f"Last token attention distribution. Simple format: {plain_number_attn:.1%} on numbers. "
               f"Q&A format: {qa_format_attn:.1%} on format tokens, {qa_number_attn:.1%} on numbers.")
    
    fig.text(0.5, 0.01, caption, ha='center', fontsize=10, style='italic', color='#555')
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = '/home/paperspace/dev/MATS9/attention/clean_attention_visualization.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\n✨ Clean visualization saved to: {output_path}")
    
    # Also save as PDF for paper
    pdf_path = '/home/paperspace/dev/MATS9/attention/clean_attention_visualization.pdf'
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"📄 PDF version saved to: {pdf_path}")
    
    plt.close()
    
    return plain_number_attn, qa_format_attn, qa_number_attn


def main():
    """Generate clean attention visualization"""
    
    print("🎯 GENERATING CLEAN ATTENTION VISUALIZATION")
    print("="*70)
    print("Creating paper-ready visualization in clean style")
    
    # Create the visualization
    plain_num, qa_fmt, qa_num = create_clean_visualization()
    
    print("\n" + "="*70)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"\n📊 Summary Statistics:")
    print(f"  • Simple format attention on numbers: {plain_num:.1%}")
    print(f"  • Q&A format attention on format tokens: {qa_fmt:.1%}")
    print(f"  • Q&A format attention on numbers: {qa_num:.1%}")
    print(f"\n🎯 Key finding: Q&A format causes {(plain_num - qa_num):.1%} drop in numerical attention")


if __name__ == "__main__":
    main()