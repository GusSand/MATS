#!/usr/bin/env python3
"""
Enhanced visualizations for the 9.11 vs 9.8 bug analysis
Combines insights from existing logitlens analysis with improved visualization techniques
"""

import torch
from nnsight import LanguageModel
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import logging
from datetime import datetime
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patheffects import withStroke

# Set clean style with larger titles
try:
    plt.style.use('seaborn-darkgrid')
except:
    plt.style.use('ggplot')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['font.family'] = 'DejaVu Sans'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Enhanced color scheme
COLORS = {
    'correct': '#2ECC71',  # Emerald green
    'wrong': '#E74C3C',    # Alizarin red
    'neutral': '#95A5A6',  # Gray
    'highlight': '#F39C12', # Orange
    'layer25': '#9B59B6',  # Purple for critical layer
    'background': '#ECF0F1' # Light gray
}

# Token evolution colors
TOKEN_PALETTE = [
    '#E74C3C', '#3498DB', '#F39C12', '#9B59B6', 
    '#1ABC9C', '#E67E22', '#2ECC71', '#34495E'
]

logger.info("Loading model...")
model = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", device_map="auto")
logger.info("✓ Model loaded")

def run_logit_lens(prompt):
    """Run logit lens analysis with enhanced tracking"""
    with model.trace(prompt) as tracer:
        hidden_states = []
        for layer_idx in range(model.config.num_hidden_layers):
            if layer_idx == 0:
                hidden = model.model.layers[layer_idx].input[0][0]
            else:
                hidden = model.model.layers[layer_idx].output[0]
            hidden_states.append(hidden.save())
        final_hidden = model.model.norm.output.save()
        hidden_states.append(final_hidden)
    
    predictions = []
    token_probs = {
        '9': [], '8': [], '11': [], 
        'Both': [], 'neither': [], '': []
    }
    
    with torch.no_grad():
        for layer_idx, hidden_state in enumerate(hidden_states):
            if layer_idx < len(hidden_states) - 1:
                normalized = model.model.norm(hidden_state.value)
            else:
                normalized = hidden_state.value
            
            logits = model.lm_head(normalized)
            
            if len(logits.shape) == 1:
                answer_logits = logits
            elif len(logits.shape) == 2:
                answer_logits = logits[-1, :]
            elif len(logits.shape) == 3:
                answer_logits = logits[0, -1, :]
            
            probs = torch.softmax(answer_logits, dim=-1)
            top10_probs, top10_indices = torch.topk(probs, 10)
            top_tokens = [model.tokenizer.decode(idx.item()) for idx in top10_indices]
            
            # Track key tokens
            token_probs['9'].append(probs[24].item() if 24 < len(probs) else 0)
            token_probs['8'].append(probs[23].item() if 23 < len(probs) else 0)
            token_probs['11'].append(probs[806].item() if 806 < len(probs) else 0)
            
            # Track hedge tokens
            for token, idx in [('Both', 11995), ('neither', 14911)]:
                if idx < len(probs):
                    token_probs[token].append(probs[idx].item())
                else:
                    token_probs[token].append(0)
            
            predictions.append({
                'layer': layer_idx,
                'top_token': top_tokens[0],
                'top_prob': top10_probs[0].item(),
                'top10_tokens': list(zip(top_tokens, top10_probs.tolist()))
            })
    
    return predictions, token_probs

# Run analysis for both formats
logger.info("Running analysis for wrong format (Q&A)...")
wrong_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"
wrong_preds, wrong_probs = run_logit_lens(wrong_prompt)

logger.info("Running analysis for correct format (Simple)...")
correct_prompt = "Which is bigger: 9.8 or 9.11?\nAnswer:"
correct_preds, correct_probs = run_logit_lens(correct_prompt)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

def create_enhanced_token_evolution():
    """Create an enhanced visualization showing token evolution for both paths"""
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 2, height_ratios=[1, 1, 1.2], width_ratios=[1, 1], hspace=0.3, wspace=0.2)
    
    # Top row: Token probability evolution
    ax1 = fig.add_subplot(gs[0, :])
    
    layers = list(range(33))
    
    # Plot token "9" probability
    ax1.plot(layers, wrong_probs['9'], color=COLORS['wrong'], linewidth=2.5, 
             label='Wrong Format (Q&A)', linestyle='--', marker='o', markersize=4)
    ax1.plot(layers, correct_probs['9'], color=COLORS['correct'], linewidth=2.5,
             label='Correct Format (Simple)', linestyle='-', marker='s', markersize=4)
    
    # Highlight Layer 25
    ax1.axvline(x=25, color=COLORS['layer25'], linestyle=':', linewidth=2, alpha=0.7)
    ax1.fill_betweenx([0, 1], 24.5, 25.5, color=COLORS['layer25'], alpha=0.1)
    
    # Add annotation for Layer 25
    ax1.annotate('Layer 25\nCritical Divergence', xy=(25, 0.22), xytext=(27, 0.4),
                arrowprops=dict(arrowstyle='->', color=COLORS['layer25'], lw=2),
                fontsize=12, color=COLORS['layer25'], fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=COLORS['layer25']))
    
    ax1.set_xlabel('Layer', fontsize=14)
    ax1.set_ylabel('P(token "9")', fontsize=14)
    ax1.set_title('Token "9" Probability Evolution: The Divergence Point', fontsize=18, fontweight='bold')
    ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.5, 32.5)
    ax1.set_ylim(-0.02, 0.65)
    
    # Middle row: Token evolution paths
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Wrong format token evolution
    plot_token_path(ax2, wrong_preds, "Wrong Format (Q&A → 9.11)", COLORS['wrong'])
    
    # Correct format token evolution
    plot_token_path(ax3, correct_preds, "Correct Format (Answer → 9.8)", COLORS['correct'])
    
    # Bottom row: Heatmap comparison
    ax4 = fig.add_subplot(gs[2, :])
    create_probability_heatmap(ax4, wrong_probs, correct_probs)
    
    plt.suptitle('9.11 vs 9.8 Bug: Complete Token Evolution Analysis', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Save figure
    for ext in ['png', 'pdf']:
        filename = f'enhanced_token_evolution_{timestamp}.{ext}'
        plt.savefig(filename, dpi=300 if ext == 'png' else None, bbox_inches='tight')
        logger.info(f"✓ Saved {filename}")
    
    plt.show()

def plot_token_path(ax, predictions, title, color):
    """Plot token evolution as a path through layers"""
    key_layers = [0, 5, 10, 15, 20, 25, 28, 30, 32]
    
    # Create nodes
    y_positions = np.linspace(0.9, 0.1, len(key_layers))
    x_position = 0.5
    
    token_colors = {}
    color_idx = 0
    
    for i, layer_idx in enumerate(key_layers):
        pred = predictions[layer_idx]
        token = pred['top_token'].strip()[:12]
        prob = pred['top_prob']
        
        # Assign colors to unique tokens
        if token not in token_colors and token:
            token_colors[token] = TOKEN_PALETTE[color_idx % len(TOKEN_PALETTE)]
            color_idx += 1
        
        node_color = token_colors.get(token, COLORS['neutral'])
        
        # Special highlight for Layer 25
        if layer_idx == 25:
            edge_color = COLORS['layer25']
            edge_width = 3
            bbox_props = dict(boxstyle="round,pad=0.05", 
                            facecolor='white', 
                            edgecolor=edge_color, 
                            linewidth=edge_width)
        else:
            edge_color = node_color
            edge_width = 2
            bbox_props = dict(boxstyle="round,pad=0.03", 
                            facecolor='white', 
                            edgecolor=edge_color, 
                            linewidth=edge_width, 
                            alpha=0.8)
        
        # Draw node
        ax.text(x_position, y_positions[i], f'{token}\n{prob:.3f}',
               ha='center', va='center', fontsize=10,
               bbox=bbox_props, fontweight='bold' if layer_idx == 25 else 'normal')
        
        # Layer label
        ax.text(0.1, y_positions[i], f'L{layer_idx}', 
               ha='center', va='center', fontsize=11, fontweight='bold')
        
        # Draw connection to next node
        if i < len(key_layers) - 1:
            ax.annotate('', xy=(x_position, y_positions[i+1] + 0.05),
                       xytext=(x_position, y_positions[i] - 0.05),
                       arrowprops=dict(arrowstyle='->', lw=1.5, 
                                     color=node_color, alpha=0.6))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=14, fontweight='bold', color=color)
    ax.axis('off')

def create_probability_heatmap(ax, wrong_probs, correct_probs):
    """Create a heatmap showing probability differences across layers"""
    tokens = ['9', '8', '11', 'Both', 'neither']
    layers = list(range(0, 33, 2))  # Sample every 2nd layer for clarity
    
    # Create difference matrix
    diff_matrix = np.zeros((len(tokens), len(layers)))
    
    for i, token in enumerate(tokens):
        for j, layer in enumerate(layers):
            correct_val = correct_probs[token][layer] if token in correct_probs else 0
            wrong_val = wrong_probs[token][layer] if token in wrong_probs else 0
            diff_matrix[i, j] = correct_val - wrong_val
    
    # Create heatmap
    im = ax.imshow(diff_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.3, vmax=0.3)
    
    # Set ticks and labels
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('P(correct) - P(wrong)', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(tokens)):
        for j in range(len(layers)):
            if abs(diff_matrix[i, j]) > 0.05:
                text = ax.text(j, i, f'{diff_matrix[i, j]:.2f}',
                             ha="center", va="center", color="white" if abs(diff_matrix[i, j]) > 0.15 else "black",
                             fontsize=8)
    
    # Highlight Layer 25
    ax.axvline(x=12.5, color=COLORS['layer25'], linestyle='--', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Layer', fontsize=14)
    ax.set_ylabel('Token', fontsize=14)
    ax.set_title('Probability Difference Heatmap (Correct - Wrong Format)', fontsize=14, fontweight='bold')

def create_interactive_visualization():
    """Create an interactive visualization showing layer-by-layer token changes"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    def update(frame):
        ax1.clear()
        ax2.clear()
        
        layer = frame
        
        # Plot probability bars for wrong format
        wrong_pred = wrong_preds[layer]
        tokens, probs = zip(*wrong_pred['top10_tokens'][:5])
        colors = [COLORS['wrong'] if t == '9' else COLORS['neutral'] for t in tokens]
        ax1.barh(range(5), probs, color=colors, alpha=0.7)
        ax1.set_yticks(range(5))
        ax1.set_yticklabels([t[:15] for t in tokens])
        ax1.set_xlabel('Probability')
        ax1.set_title(f'Wrong Format - Layer {layer}', fontweight='bold', color=COLORS['wrong'])
        ax1.set_xlim(0, 1)
        
        # Plot probability bars for correct format
        correct_pred = correct_preds[layer]
        tokens, probs = zip(*correct_pred['top10_tokens'][:5])
        colors = [COLORS['correct'] if t == '9' else COLORS['neutral'] for t in tokens]
        ax2.barh(range(5), probs, color=colors, alpha=0.7)
        ax2.set_yticks(range(5))
        ax2.set_yticklabels([t[:15] for t in tokens])
        ax2.set_xlabel('Probability')
        ax2.set_title(f'Correct Format - Layer {layer}', fontweight='bold', color=COLORS['correct'])
        ax2.set_xlim(0, 1)
        
        if layer == 25:
            fig.suptitle(f'Layer {layer} - CRITICAL DIVERGENCE POINT', 
                        fontsize=16, fontweight='bold', color=COLORS['layer25'])
        else:
            fig.suptitle(f'Layer {layer} Token Predictions', fontsize=16, fontweight='bold')
    
    ani = animation.FuncAnimation(fig, update, frames=range(33), interval=500, repeat=True)
    
    # Save as GIF
    gif_filename = f'token_evolution_animation_{timestamp}.gif'
    ani.save(gif_filename, writer='pillow', fps=2)
    logger.info(f"✓ Saved {gif_filename}")
    
    plt.show()

def create_comparison_table():
    """Create a detailed comparison table of key metrics"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    headers = ['Metric', 'Wrong Format (Q&A)', 'Correct Format (Simple)', 'Difference']
    
    rows = [
        ['Prompt Format', 'Q: ... A:', '... Answer:', '—'],
        ['Final Answer', '9.11 is bigger ❌', '9.8 is bigger ✓', '—'],
        ['Layer 25 Top Token', wrong_preds[25]['top_token'], correct_preds[25]['top_token'], '—'],
        ['Layer 25 P("9")', f"{wrong_probs['9'][25]:.3f}", f"{correct_probs['9'][25]:.3f}", 
         f"+{correct_probs['9'][25] - wrong_probs['9'][25]:.3f}"],
        ['Layer 30 P("9")', f"{wrong_probs['9'][30]:.3f}", f"{correct_probs['9'][30]:.3f}",
         f"+{correct_probs['9'][30] - wrong_probs['9'][30]:.3f}"],
        ['Max P("9")', f"{max(wrong_probs['9']):.3f}", f"{max(correct_probs['9']):.3f}", '—'],
        ['Peak P("Both")', f"{max(wrong_probs['Both']):.3f}", f"{max(correct_probs['Both']):.3f}", '—'],
        ['Temperature', '0.0', '0.0', '—'],
        ['Model', 'Llama-3.1-8B-Instruct', 'Llama-3.1-8B-Instruct', '—']
    ]
    
    # Create table
    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#34495E')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color-code rows
    for i in range(1, len(rows) + 1):
        if i == 4 or i == 5:  # Layer 25 and 30 rows
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#F0E6FF')
    
    ax.set_title('Detailed Comparison: 9.11 vs 9.8 Bug Analysis', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Save
    for ext in ['png', 'pdf']:
        filename = f'comparison_table_{timestamp}.{ext}'
        plt.savefig(filename, dpi=300 if ext == 'png' else None, bbox_inches='tight')
        logger.info(f"✓ Saved {filename}")
    
    plt.show()

# Generate all visualizations
logger.info("Creating enhanced visualizations...")
create_enhanced_token_evolution()
create_interactive_visualization()
create_comparison_table()

logger.info("✓ All visualizations complete!")