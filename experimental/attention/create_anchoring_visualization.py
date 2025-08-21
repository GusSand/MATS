#!/usr/bin/env python3
"""
Create Attention Anchoring Visualization
=========================================
Shows how Q&A format disrupts the critical attention anchoring to <|begin_of_text|>
which causes the model to fall into a superficial processing mode.
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
    """Extract and analyze attention patterns across layers and heads"""
    
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
    
    def get_attention_patterns(self, prompt: str, layers_heads: List[Tuple[int, int]]) -> Dict:
        """Extract attention patterns for specific layer-head combinations"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
                return_dict=True
            )
        
        # Get tokens
        tokens = [self.tokenizer.decode([tid]) for tid in inputs['input_ids'][0]]
        
        # Extract attention for each requested layer-head
        attention_data = {}
        for layer_idx, head_idx in layers_heads:
            attention = outputs.attentions[layer_idx].cpu()
            attention_data[(layer_idx, head_idx)] = attention[0, head_idx, -1, :].numpy()
        
        # Also get generation
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
            'attention_data': attention_data,
            'tokens': tokens,
            'generated': generated,
            'prompt': prompt
        }


def create_anchoring_visualization():
    """Create visualization showing attention anchoring disruption"""
    
    print("\n" + "="*70)
    print("CREATING ATTENTION ANCHORING VISUALIZATION")
    print("="*70)
    
    analyzer = AttentionAnalyzer()
    
    # Test prompts
    plain_prompt = "Which is bigger: 9.8 or 9.11?\nAnswer:"
    qa_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"
    
    # Key layer-head combinations that show the pattern
    key_heads = [
        (6, 27),   # Layer 6, Head 27 - strongest anchoring difference
        (8, 19),   # Layer 8, Head 19 - mid-layer pattern
        (10, 27),  # Layer 10, Head 27 - shows redistribution
    ]
    
    print("\nExtracting attention patterns across layers...")
    plain_data = analyzer.get_attention_patterns(plain_prompt, key_heads)
    qa_data = analyzer.get_attention_patterns(qa_prompt, key_heads)
    
    print(f"Plain format generates: '{plain_data['generated'][:20]}'")
    print(f"Q&A format generates: '{qa_data['generated'][:20]}'")
    
    # Create the visualization
    fig = plt.figure(figsize=(18, 11))  # Moderate height for balanced spacing
    
    # Main title
    fig.suptitle('Attention Anchoring Disruption: The Real Mechanism Behind the Decimal Bug',
                 fontsize=18, fontweight='bold', y=0.96)  # Balanced position for title
    
    # Create grid: 3 rows (one per layer), 3 columns with moderate vertical spacing
    gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.3, width_ratios=[1.2, 1.2, 1],
                         top=0.85, bottom=0.06)  # Lowered top to give more space after legend
    
    # Analyze each layer-head combination
    begin_token_attention_plain = []
    begin_token_attention_qa = []
    number_attention_plain = []
    number_attention_qa = []
    format_attention_qa = []
    
    for idx, (layer, head) in enumerate(key_heads):
        # Plain format bar chart
        ax1 = fig.add_subplot(gs[idx, 0])
        plain_attn = plain_data['attention_data'][(layer, head)]
        
        # Calculate key metrics
        begin_attn_plain = plain_attn[0]  # <|begin_of_text|> is first token
        number_attn_plain = sum(plain_attn[i] for i, t in enumerate(plain_data['tokens']) 
                               if any(n in t for n in ['9', '8', '11', '.']))
        
        begin_token_attention_plain.append(begin_attn_plain)
        number_attention_plain.append(number_attn_plain)
        
        # Create bar plot for plain format
        colors = []
        labels = []
        for i, token in enumerate(plain_data['tokens']):
            token_clean = token.strip()
            if i == 0:  # <|begin_of_text|>
                colors.append('#2E86AB')  # Blue for anchor
                labels.append('⚓ BEGIN')
            elif any(n in token for n in ['9', '8', '11', '.']):
                colors.append('#A23B72')  # Purple for numbers
                labels.append(token_clean)
            else:
                colors.append('#C0C0C0')  # Gray for others
                labels.append(token_clean[:10])
        
        y_pos = np.arange(len(labels))
        bars = ax1.barh(y_pos, plain_attn, color=colors, edgecolor='none')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(labels, fontsize=9)
        ax1.set_xlabel('Attention Weight', fontsize=10)
        ax1.set_title(f'Layer {layer}, Head {head}\nPlain Format (Correct)', fontsize=11, fontweight='bold')
        ax1.set_xlim(0, 1.0)
        
        # Remove spines
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Add percentage labels
        for bar, weight in zip(bars, plain_attn):
            if weight > 0.05:
                ax1.text(weight + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{weight:.0%}', va='center', fontsize=9)
        
        # Q&A format bar chart
        ax2 = fig.add_subplot(gs[idx, 1])
        qa_attn = qa_data['attention_data'][(layer, head)]
        
        # Calculate key metrics
        begin_attn_qa = qa_attn[0]
        number_attn_qa = sum(qa_attn[i] for i, t in enumerate(qa_data['tokens']) 
                            if any(n in t for n in ['9', '8', '11', '.']))
        format_attn_qa = sum(qa_attn[i] for i, t in enumerate(qa_data['tokens'])
                            if any(f in t for f in ['Q', 'A', ':']))
        
        begin_token_attention_qa.append(begin_attn_qa)
        number_attention_qa.append(number_attn_qa)
        format_attention_qa.append(format_attn_qa)
        
        # Create bar plot for Q&A format
        colors = []
        labels = []
        for i, token in enumerate(qa_data['tokens']):
            token_clean = token.strip()
            if i == 0:  # <|begin_of_text|>
                colors.append('#2E86AB')  # Blue for anchor
                labels.append('⚓ BEGIN')
            elif any(n in token for n in ['9', '8', '11', '.']):
                colors.append('#A23B72')  # Purple for numbers
                labels.append(token_clean)
            elif any(f in token for f in ['Q', 'A', ':']):
                colors.append('#F18F01')  # Orange for format tokens
                labels.append(token_clean)
            else:
                colors.append('#C0C0C0')  # Gray for others
                labels.append(token_clean[:10])
        
        y_pos = np.arange(len(labels))
        bars = ax2.barh(y_pos, qa_attn, color=colors, edgecolor='none')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels, fontsize=9)
        ax2.set_xlabel('Attention Weight', fontsize=10)
        ax2.set_title(f'Layer {layer}, Head {head}\nQ&A Format (Wrong)', fontsize=11, fontweight='bold')
        ax2.set_xlim(0, 1.0)
        
        # Remove spines
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Add percentage labels
        for bar, weight in zip(bars, qa_attn):
            if weight > 0.05:
                ax2.text(weight + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{weight:.0%}', va='center', fontsize=9)
        
        # Difference visualization
        ax3 = fig.add_subplot(gs[idx, 2])
        
        # Calculate attention shift
        anchor_shift = begin_attn_qa - begin_attn_plain
        number_shift = number_attn_qa - number_attn_plain
        format_total = format_attn_qa
        
        # Create summary bars
        categories = ['⚓ BEGIN\nToken', '🔢 Number\nTokens', '📝 Format\nTokens']
        plain_values = [begin_attn_plain, number_attn_plain, 0]
        qa_values = [begin_attn_qa, number_attn_qa, format_attn_qa]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, plain_values, width, label='Plain', color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.7)
        bars2 = ax3.bar(x + width/2, qa_values, width, label='Q&A', color=['#2E86AB', '#A23B72', '#F18F01'])
        
        ax3.set_ylabel('Total Attention', fontsize=10)
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories, fontsize=9)
        ax3.legend(fontsize=9)
        ax3.set_ylim(0, 1.0)
        
        # Remove spines
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0.05:
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{height:.0%}', ha='center', va='bottom', fontsize=8)
        
        # Add shift annotation
        ax3.text(0, 0.95, f'Δ = {anchor_shift:+.0%}', ha='center', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    # Add summary statistics box at bottom
    # summary_text = f"""
    # KEY FINDINGS:
    
    # • BEGIN Token Attention (Plain → Q&A):
    #   Layer 6:  {begin_token_attention_plain[0]:.0%} → {begin_token_attention_qa[0]:.0%} (Δ = {begin_token_attention_qa[0] - begin_token_attention_plain[0]:+.0%})
    #   Layer 8:  {begin_token_attention_plain[1]:.0%} → {begin_token_attention_qa[1]:.0%} (Δ = {begin_token_attention_qa[1] - begin_token_attention_plain[1]:+.0%})
    #   Layer 10: {begin_token_attention_plain[2]:.0%} → {begin_token_attention_qa[2]:.0%} (Δ = {begin_token_attention_qa[2] - begin_token_attention_plain[2]:+.0%})
    
    # • Average BEGIN token attention drop: {np.mean([qa - plain for qa, plain in zip(begin_token_attention_qa, begin_token_attention_plain)]):.0%}
    # • This disrupted anchoring causes the model to fall into superficial pattern matching
    # """
    
    # fig.text(0.5, 0.02, summary_text, ha='center', fontsize=10, 
    #         bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F0F0', edgecolor='black', linewidth=0.5))
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='#2E86AB', label='BEGIN Token (<|begin_of_text|>)'),
        mpatches.Patch(color='#A23B72', label='Number Tokens (9, 8, 11, .)'),
        mpatches.Patch(color='#F18F01', label='Format Tokens (Q, :, A)'),
        mpatches.Patch(color='#C0C0C0', label='Other Tokens'),
    ]
    
    fig.legend(handles=legend_elements, 
              loc='upper center', 
              bbox_to_anchor=(0.5, 0.92),  # Moved up slightly since we lowered the grid
              ncol=4,
              frameon=False,
              fontsize=10)
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = '/home/paperspace/dev/MATS9/attention/attention_anchoring_visualization.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\n✨ Visualization saved to: {output_path}")
    
    # Also save as PDF for paper
    pdf_path = '/home/paperspace/dev/MATS9/attention/attention_anchoring_visualization.pdf'
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"📄 PDF version saved to: {pdf_path}")
    
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    avg_begin_plain = np.mean(begin_token_attention_plain)
    avg_begin_qa = np.mean(begin_token_attention_qa)
    avg_drop = avg_begin_qa - avg_begin_plain
    
    print(f"\nAverage BEGIN token attention:")
    print(f"  Plain format:  {avg_begin_plain:.1%}")
    print(f"  Q&A format:    {avg_begin_qa:.1%}")
    print(f"  Drop:          {avg_drop:.1%}")
    
    print(f"\nThis {abs(avg_drop):.0%} drop in attention anchoring is the root cause of the bug.")
    print("Without strong anchoring, the model falls into superficial pattern matching.")


def main():
    """Generate attention anchoring visualization"""
    
    print("🎯 GENERATING ATTENTION ANCHORING VISUALIZATION")
    print("="*70)
    print("This will show how Q&A format disrupts critical attention anchoring")
    
    # Create the visualization
    create_anchoring_visualization()
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()