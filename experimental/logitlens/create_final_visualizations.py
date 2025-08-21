#!/usr/bin/env python3
"""
Final visualizations with requested improvements:
- Better labels
- Fixed y-axis scale
- Multi-color token evolution
- Larger titles
- Both PNG and PDF output
"""

import torch
from nnsight import LanguageModel
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import logging
from datetime import datetime
import seaborn as sns
import matplotlib.cm as cm
import matplotlib
from textwrap import wrap

# Set up matplotlib for publication quality
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.pyplot.switch_backend('agg')

# Set clean style with MUCH larger fonts to match sample_viz.py
plt.rcParams['font.size'] = 20  # Increased from 12
plt.rcParams['axes.labelsize'] = 40  # Increased from 14
plt.rcParams['axes.titlesize'] = 50  # Increased from 20
plt.rcParams['xtick.labelsize'] = 35  # Increased from 11
plt.rcParams['ytick.labelsize'] = 35  # Increased from 11
plt.rcParams['legend.fontsize'] = 30  # Increased from 12
plt.rcParams['figure.titlesize'] = 60  # Increased from 22

# Set seaborn style to match sample_viz.py
sns.set_style('whitegrid')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Colors
WRONG_COLOR = '#E74C3C'  # Red
CORRECT_COLOR = '#27AE60'  # Green
NEUTRAL_COLOR = '#95A5A6'  # Gray

# Create a color palette for different tokens
TOKEN_COLORS = [
    '#E74C3C',  # Red
    '#3498DB',  # Blue
    '#F39C12',  # Orange
    '#9B59B6',  # Purple
    '#1ABC9C',  # Turquoise
    '#E67E22',  # Carrot
    '#2ECC71',  # Emerald
    '#34495E',  # Dark Gray
]

logger.info("Loading model...")
model = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", device_map="auto")
logger.info("✓ Model loaded")

def run_logit_lens(prompt):
    """Run logit lens analysis"""
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
    token_9_probs = []
    token_8_probs = []
    token_11_probs = []
    
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
            top5_probs, top5_indices = torch.topk(probs, 5)
            top_tokens = [model.tokenizer.decode(idx.item()) for idx in top5_indices]
            
            token_9_probs.append(probs[24].item() if 24 < len(probs) else 0)
            token_8_probs.append(probs[23].item() if 23 < len(probs) else 0)
            token_11_probs.append(probs[806].item() if 806 < len(probs) else 0)
            
            predictions.append({
                'layer': layer_idx,
                'top_token': top_tokens[0],
                'top_prob': top5_probs[0].item(),
                'top5': list(zip(top_tokens[:3], top5_probs[:3].cpu().numpy()))
            })
    
    return predictions, token_9_probs, token_8_probs, token_11_probs

# Run analysis
logger.info("Running logit lens analysis...")
wrong_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"
correct_prompt = "Which is bigger: 9.8 or 9.11?\nAnswer:"

wrong_preds, wrong_9, wrong_8, wrong_11 = run_logit_lens(wrong_prompt)
correct_preds, correct_9, correct_8, correct_11 = run_logit_lens(correct_prompt)
logger.info("✓ Analysis complete")

layers = list(range(33))
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Helper function to save both PNG and PDF
def save_figure(fig, base_filename):
    """Save figure in both PNG and PDF formats"""
    png_name = f"{base_filename}.png"
    pdf_name = f"{base_filename}.pdf"
    fig.savefig(png_name, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_name, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Saved: {png_name} and {pdf_name}")
    return png_name, pdf_name

# ==========================================
# VISUALIZATION 1: Token "9" Probability
# ==========================================
def create_token_9_probability_plot():
    fig, ax = plt.subplots(figsize=(16, 10))  # Increased from (10, 6)
    
    # Plot lines with corrected labels
    ax.plot(layers, wrong_9, color=WRONG_COLOR, linewidth=3,  # Increased from 2.5
            marker='o', markersize=8, alpha=0.9, label='Wrong Format (Q&A)')  # Increased markersize
    ax.plot(layers, correct_9, color=CORRECT_COLOR, linewidth=3,
            marker='s', markersize=8, alpha=0.9, label='Correct Format (Simple)')
    
    # Highlight critical layers
    for layer in [25, 30]:
        ax.axvline(x=layer, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    # Add shading for key regions
    ax.axvspan(24, 26, alpha=0.05, color='blue', label='Critical region')
    ax.axvspan(29, 31, alpha=0.05, color='blue')
    
    # Annotations for peaks - FIXED: Move annotation lower to avoid title overlap
    max_correct = max(correct_9)
    max_layer = correct_9.index(max_correct)
    ax.annotate(f'{max_correct:.3f}',
                xy=(max_layer, max_correct),
                xytext=(max_layer-3, max_correct-0.05),  # Changed from +0.15 to -0.05 to move lower
                arrowprops=dict(arrowstyle='->', color=CORRECT_COLOR, lw=1.5),
                fontsize=20, color=CORRECT_COLOR, weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=CORRECT_COLOR))
    
    # Labels and formatting - MUCH larger fonts
    ax.set_xlabel('Layer', fontsize=40, fontweight='bold')
    ax.set_ylabel('Probability of Token "9"', fontsize=40, fontweight='bold')
    ax.set_title('Token "9" Probability Across Transformer Layers', fontsize=50, pad=30, weight='bold')  # Increased pad from 20 to 30
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=30)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_xlim(-0.5, 32.5)
    ax.set_ylim(-0.01, max(max(wrong_9), max(correct_9)) * 1.25)  # Increased from 1.15 to 1.25 for more space
    
    # Remove spines to match sample_viz.py style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add layer markers on x-axis
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 32])
    
    # FIXED: Add more space between y-axis label and tick marks
    ax.yaxis.set_label_coords(-0.08, 0.5)  # Move label further left (default is -0.05)
    
    plt.tight_layout()
    base_filename = f'viz_1_token_9_probability_{timestamp}'
    return save_figure(fig, base_filename)

# ==========================================
# VISUALIZATION 2: Token Preference (Fixed Scale)
# ==========================================
def create_token_preference_plot():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    wrong_diff = np.array([w11 - w8 for w11, w8 in zip(wrong_11, wrong_8)])
    correct_diff = np.array([c11 - c8 for c11, c8 in zip(correct_11, correct_8)])
    
    # Apply smoothing for better visibility
    from scipy.ndimage import gaussian_filter1d
    wrong_diff_smooth = gaussian_filter1d(wrong_diff, sigma=0.5)
    correct_diff_smooth = gaussian_filter1d(correct_diff, sigma=0.5)
    
    # Scale by 1000 for visibility
    scale = 1000
    
    # Plot with both raw and smoothed
    ax.plot(layers, wrong_diff * scale, color=WRONG_COLOR, linewidth=1, alpha=0.3)
    ax.plot(layers, wrong_diff_smooth * scale, color=WRONG_COLOR, linewidth=2.5,
            label='Wrong Format (Q&A)', alpha=0.9)
    
    ax.plot(layers, correct_diff * scale, color=CORRECT_COLOR, linewidth=1, alpha=0.3)
    ax.plot(layers, correct_diff_smooth * scale, color=CORRECT_COLOR, linewidth=2.5,
            label='Correct Format (Simple)', alpha=0.9)
    
    # Zero line
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1.5)
    
    # Shaded regions
    ax.fill_between(layers, 0, -2.5, alpha=0.05, color=CORRECT_COLOR)
    ax.fill_between(layers, 0, 2.5, alpha=0.05, color=WRONG_COLOR)
    
    # Find significant differences
    significant_layers = []
    for i, (w, c) in enumerate(zip(wrong_diff, correct_diff)):
        if abs(c) > 0.001:  # Threshold for significance
            significant_layers.append(i)
    
    # Mark significant layers
    for layer in significant_layers[-5:]:  # Last 5 significant layers
        if layer < len(correct_diff):
            ax.scatter(layer, correct_diff[layer] * scale, 
                      s=50, color=CORRECT_COLOR, zorder=5, alpha=0.7)
    
    # Labels
    ax.set_xlabel('Layer', fontsize=14)
    ax.set_ylabel('P("11") - P("8") × 10³', fontsize=14)
    ax.set_title('Token Preference: Decimal Ending Selection', fontsize=20, pad=15, weight='bold')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    
    # Add annotations
    ax.text(0.5, 0.95, '↑ Prefers "11" (wrong)', transform=ax.transAxes,
            fontsize=11, color=WRONG_COLOR, ha='center', va='top')
    ax.text(0.5, 0.05, '↓ Prefers "8" (correct)', transform=ax.transAxes,
            fontsize=11, color=CORRECT_COLOR, ha='center', va='bottom')
    
    # FIXED Y-AXIS SCALE
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlim(-0.5, 32.5)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 32])
    
    plt.tight_layout()
    base_filename = f'viz_2_token_preference_{timestamp}'
    return save_figure(fig, base_filename)

# ==========================================
# VISUALIZATION 3: Token Evolution (Wrong) - Multi-color
# ==========================================
def create_token_evolution_wrong():
    fig, ax = plt.subplots(figsize=(14, 4))
    
    key_layers = [0, 5, 10, 15, 20, 25, 30, 32]
    
    # Calculate positions
    positions = np.linspace(0.1, 0.9, len(key_layers))
    
    # Create token-to-color mapping
    unique_tokens = {}
    color_idx = 0
    
    for i, layer_idx in enumerate(key_layers):
        pred = wrong_preds[layer_idx]
        token = pred['top_token'].strip()[:15]
        prob = pred['top_prob']
        
        # Assign color to unique tokens
        if token not in unique_tokens and token:
            unique_tokens[token] = TOKEN_COLORS[color_idx % len(TOKEN_COLORS)]
            color_idx += 1
        
        token_color = unique_tokens.get(token, NEUTRAL_COLOR)
        
        # Box style based on probability
        alpha = min(0.5 + prob * 0.5, 1.0)
        
        # Create box
        rect = mpatches.FancyBboxPatch(
            (positions[i] - 0.05, 0.35), 0.1, 0.3,
            boxstyle="round,pad=0.02",
            facecolor='white',
            edgecolor=token_color,
            linewidth=2.5,
            alpha=alpha
        )
        ax.add_patch(rect)
        
        # Layer number above
        ax.text(positions[i], 0.75, f'L{layer_idx}',
                ha='center', va='center', fontsize=12, weight='bold')
        
        # Token in box - BIGGER and BLACK text
        font_size = 13 if len(token) < 10 else 11  # Increased from 11/9
        ax.text(positions[i], 0.5, f'"{token}"',
                ha='center', va='center', fontsize=font_size,
                color='black',  # Always black for readability
                weight='bold')  # Always bold
        
        # Probability below - also black and bigger
        ax.text(positions[i], 0.25, f'{prob:.3f}',
                ha='center', va='center', fontsize=11,  # Increased from 9
                color='black', style='italic', weight='bold')
        
        # Arrow to next
        if i < len(key_layers) - 1:
            ax.annotate('', xy=(positions[i+1] - 0.05, 0.5),
                       xytext=(positions[i] + 0.05, 0.5),
                       arrowprops=dict(arrowstyle='->', color='gray',
                                     lw=1.5, alpha=0.5))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    ax.set_title('Token Evolution - Wrong Format (Produces: "9.11 is bigger")',
                fontsize=20, color=WRONG_COLOR, pad=20, weight='bold')
    
    plt.tight_layout()
    base_filename = f'viz_3_token_evolution_wrong_{timestamp}'
    return save_figure(fig, base_filename)

# ==========================================
# VISUALIZATION 4: Token Evolution (Correct) - Multi-color
# ==========================================
def create_token_evolution_correct():
    fig, ax = plt.subplots(figsize=(14, 4))
    
    key_layers = [0, 5, 10, 15, 20, 25, 30, 32]
    positions = np.linspace(0.1, 0.9, len(key_layers))
    
    # Create token-to-color mapping
    unique_tokens = {}
    color_idx = 0
    
    for i, layer_idx in enumerate(key_layers):
        pred = correct_preds[layer_idx]
        token = pred['top_token'].strip()[:15]
        prob = pred['top_prob']
        
        # Assign color to unique tokens
        if token not in unique_tokens and token:
            unique_tokens[token] = TOKEN_COLORS[color_idx % len(TOKEN_COLORS)]
            color_idx += 1
        
        token_color = unique_tokens.get(token, NEUTRAL_COLOR)
        
        # Highlight layer 25 where "9" appears
        is_critical = (layer_idx == 25 and "9" in token)
        
        alpha = min(0.5 + prob * 0.5, 1.0)
        
        rect = mpatches.FancyBboxPatch(
            (positions[i] - 0.05, 0.35), 0.1, 0.3,
            boxstyle="round,pad=0.02",
            facecolor='#E8F8F5' if is_critical else 'white',
            edgecolor=CORRECT_COLOR if is_critical else token_color,
            linewidth=3.5 if is_critical else 2.5,
            alpha=1.0 if is_critical else alpha
        )
        ax.add_patch(rect)
        
        # Layer number
        ax.text(positions[i], 0.75, f'L{layer_idx}',
                ha='center', va='center', fontsize=12, weight='bold')
        
        # Token - BIGGER and BLACK text
        font_size = 13 if len(token) < 10 else 11  # Increased from 11/9
        ax.text(positions[i], 0.5, f'"{token}"',
                ha='center', va='center', fontsize=font_size,
                color='black',  # Always black for readability
                weight='bold')  # Always bold
        
        # Probability - also black and bigger
        ax.text(positions[i], 0.25, f'{prob:.3f}',
                ha='center', va='center', fontsize=11,  # Increased from 9
                color='black', style='italic', weight='bold')
        
        # Critical point annotation
        if is_critical:
            ax.annotate('Critical!', xy=(positions[i], 0.15),
                       xytext=(positions[i], 0.05),
                       arrowprops=dict(arrowstyle='->', color=CORRECT_COLOR, lw=2),
                       fontsize=11, color=CORRECT_COLOR, ha='center', weight='bold')
        
        # Arrow to next
        if i < len(key_layers) - 1:
            ax.annotate('', xy=(positions[i+1] - 0.05, 0.5),
                       xytext=(positions[i] + 0.05, 0.5),
                       arrowprops=dict(arrowstyle='->', color='gray',
                                     lw=1.5, alpha=0.5))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    ax.set_title('Token Evolution - Correct Format (Produces: "9.8 is bigger")',
                fontsize=20, color=CORRECT_COLOR, pad=20, weight='bold')
    
    plt.tight_layout()
    base_filename = f'viz_4_token_evolution_correct_{timestamp}'
    return save_figure(fig, base_filename)

# ==========================================
# VISUALIZATION 5: Combined Comparison
# ==========================================
def create_combined_comparison():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Subplot 1: Token 9 probability
    ax = axes[0, 0]
    ax.plot(layers, wrong_9, color=WRONG_COLOR, linewidth=2, marker='o', 
            markersize=3, alpha=0.8, label='Wrong (Q&A)')
    ax.plot(layers, correct_9, color=CORRECT_COLOR, linewidth=2, marker='s',
            markersize=3, alpha=0.8, label='Correct (Simple)')
    ax.set_xlabel('Layer')
    ax.set_ylabel('P(token "9")')
    ax.set_title('Token "9" Probability', fontsize=16)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.2)
    
    # Subplot 2: Token preference with fixed scale
    ax = axes[0, 1]
    wrong_diff = [(w11 - w8) * 1000 for w11, w8 in zip(wrong_11, wrong_8)]
    correct_diff = [(c11 - c8) * 1000 for c11, c8 in zip(correct_11, correct_8)]
    ax.plot(layers, wrong_diff, color=WRONG_COLOR, linewidth=2, marker='o',
            markersize=3, alpha=0.8, label='Wrong (Q&A)')
    ax.plot(layers, correct_diff, color=CORRECT_COLOR, linewidth=2, marker='s',
            markersize=3, alpha=0.8, label='Correct (Simple)')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Layer')
    ax.set_ylabel('P("11") - P("8") × 10³')
    ax.set_title('Token Preference', fontsize=16)
    ax.set_ylim(-2.5, 2.5)  # Fixed scale
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)
    
    # Subplot 3: Key layers comparison (Wrong)
    ax = axes[1, 0]
    key_layers = [0, 10, 20, 25, 30, 32]
    tokens_wrong = [wrong_preds[l]['top_token'].strip()[:10] for l in key_layers]
    probs_wrong = [wrong_preds[l]['top_prob'] for l in key_layers]
    
    bars = ax.bar(range(len(key_layers)), probs_wrong, color=WRONG_COLOR, alpha=0.7)
    ax.set_xticks(range(len(key_layers)))
    ax.set_xticklabels([f'L{l}\n"{t}"' for l, t in zip(key_layers, tokens_wrong)],
                       rotation=0, fontsize=9)
    ax.set_ylabel('Probability')
    ax.set_title('Wrong Format - Top Tokens', color=WRONG_COLOR, fontsize=16)
    ax.set_ylim(0, max(probs_wrong) * 1.2 if max(probs_wrong) > 0 else 1)
    
    # Subplot 4: Key layers comparison (Correct)
    ax = axes[1, 1]
    tokens_correct = [correct_preds[l]['top_token'].strip()[:10] for l in key_layers]
    probs_correct = [correct_preds[l]['top_prob'] for l in key_layers]
    
    bars = ax.bar(range(len(key_layers)), probs_correct, color=CORRECT_COLOR, alpha=0.7)
    ax.set_xticks(range(len(key_layers)))
    ax.set_xticklabels([f'L{l}\n"{t}"' for l, t in zip(key_layers, tokens_correct)],
                       rotation=0, fontsize=9)
    ax.set_ylabel('Probability')
    ax.set_title('Correct Format - Top Tokens', color=CORRECT_COLOR, fontsize=16)
    ax.set_ylim(0, max(probs_correct) * 1.2 if max(probs_correct) > 0 else 1)
    
    plt.suptitle('Logit Lens Analysis: Format-Dependent Processing',
                fontsize=22, fontweight='bold')
    plt.tight_layout()
    
    base_filename = f'viz_5_combined_comparison_{timestamp}'
    return save_figure(fig, base_filename)

# Create all visualizations
logger.info("Creating visualizations...")
files = []
files.extend(create_token_9_probability_plot())
files.extend(create_token_preference_plot())
files.extend(create_token_evolution_wrong())
files.extend(create_token_evolution_correct())
files.extend(create_combined_comparison())

logger.info("\n✅ All visualizations created successfully!")
logger.info("Files created:")
for f in files:
    logger.info(f"  - {f}")

print("\n" + "="*60)
print("VISUALIZATION SUMMARY")
print("="*60)
print("1. Token 9 Probability: Shows divergence at layer 25-30")
print("2. Token Preference: Fixed scale (-2.5 to 2.5) for better visibility")
print("3. Token Evolution (Wrong): Multi-color tokens for visual distinction")
print("4. Token Evolution (Correct): Multi-color with L25 highlight")
print("5. Combined Comparison: Overview with all improvements")
print("\nAll figures saved in both PNG and PDF formats!")
print("="*60)