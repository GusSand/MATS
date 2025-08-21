#!/usr/bin/env python3
"""
Grid visualization for the 9.11 vs 9.8 bug analysis
Shows token predictions at key layers in a clean grid format
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
import matplotlib.patches as patches

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure matplotlib for better text rendering
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['text.usetex'] = False

logger.info("Loading model...")
model = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", device_map="auto")
logger.info("✓ Model loaded")

def clean_token(token):
    """Clean token for display"""
    # Remove leading/trailing whitespace indicators
    token = token.strip()
    
    # Handle special tokens
    if not token:
        return "∅"
    if token == "\n":
        return "\\n"
    if token == "\t":
        return "\\t"
    if token == " ":
        return "⎵"
    
    # Truncate long tokens
    if len(token) > 15:
        return token[:12] + "..."
    
    return token

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
    
    predictions = {}
    
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
            top_k = 10  # Get top 10 predictions
            top_probs, top_indices = torch.topk(probs, top_k)
            
            # Store predictions for this layer
            layer_preds = []
            for i in range(top_k):
                token = model.tokenizer.decode(top_indices[i].item())
                prob = top_probs[i].item()
                layer_preds.append((clean_token(token), prob))
            
            predictions[layer_idx] = layer_preds
    
    return predictions

def create_grid_visualization():
    """Create a grid visualization showing token predictions at key layers"""
    
    # Key layers to visualize
    key_layers = [0, 7, 14, 15, 20, 25, 28, 30, 31, 32]
    
    # Run analysis for both formats
    logger.info("Running analysis for wrong format (Q&A)...")
    wrong_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"
    wrong_preds = run_logit_lens(wrong_prompt)
    
    logger.info("Running analysis for correct format (Simple)...")
    correct_prompt = "Which is bigger: 9.8 or 9.11?\nAnswer:"
    correct_preds = run_logit_lens(correct_prompt)
    
    # Create figure with optimal size for readability
    num_layers = len(key_layers)
    fig_width = 20
    cell_height = 0.8  # Height per cell
    fig_height = num_layers * cell_height + 3  # Add space for titles
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Create two subplots side by side
    gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.15)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    def plot_grid(ax, predictions, title, highlight_color):
        """Plot prediction grid for one format"""
        
        # Number of tokens to show
        num_tokens = 8
        
        # Create grid data
        grid_data = []
        layer_labels = []
        
        for layer in key_layers:
            layer_preds = predictions[layer][:num_tokens]
            # Pad if needed
            while len(layer_preds) < num_tokens:
                layer_preds.append(("", 0.0))
            grid_data.append(layer_preds)
            layer_labels.append(f"Layer {layer}")
        
        # Set up the plot
        ax.set_xlim(0, num_tokens)
        ax.set_ylim(0, num_layers)
        
        # Draw cells
        for y, (layer_label, row_data) in enumerate(zip(layer_labels, grid_data)):
            for x, (token, prob) in enumerate(row_data):
                # Determine cell color based on probability
                if prob > 0.5:
                    color = '#2ecc71'  # Strong green
                    text_color = 'white'
                elif prob > 0.2:
                    color = '#a3d5a3'  # Light green
                    text_color = 'black'
                elif prob > 0.1:
                    color = '#f9e79f'  # Light yellow
                    text_color = 'black'
                elif prob > 0.05:
                    color = '#fdebd0'  # Very light yellow
                    text_color = 'black'
                else:
                    color = 'white'
                    text_color = 'gray'
                
                # Special highlighting for key tokens
                if token in ['9', '9.']:
                    rect = patches.Rectangle((x, num_layers-y-1), 1, 1, 
                                            linewidth=3, edgecolor='red', 
                                            facecolor=color)
                elif token in ['11', '11.', 'Both']:
                    rect = patches.Rectangle((x, num_layers-y-1), 1, 1, 
                                            linewidth=2, edgecolor='blue', 
                                            facecolor=color)
                else:
                    rect = patches.Rectangle((x, num_layers-y-1), 1, 1, 
                                            linewidth=0.5, edgecolor='gray', 
                                            facecolor=color)
                ax.add_patch(rect)
                
                # Add token text
                if token and prob > 0.001:
                    # Main token text
                    ax.text(x + 0.5, num_layers-y-1 + 0.6, token,
                           ha='center', va='center', fontsize=10,
                           color=text_color, weight='bold' if prob > 0.2 else 'normal')
                    
                    # Probability text
                    prob_text = f"{prob:.1%}" if prob > 0.01 else f"{prob:.2%}"
                    ax.text(x + 0.5, num_layers-y-1 + 0.3, prob_text,
                           ha='center', va='center', fontsize=7,
                           color=text_color, alpha=0.7)
        
        # Add layer labels
        for y, layer_label in enumerate(layer_labels):
            # Highlight Layer 25
            if "25" in layer_label:
                ax.text(-0.1, num_layers-y-1 + 0.5, layer_label,
                       ha='right', va='center', fontsize=10,
                       color='purple', weight='bold')
            else:
                ax.text(-0.1, num_layers-y-1 + 0.5, layer_label,
                       ha='right', va='center', fontsize=9,
                       color='black')
        
        # Add column headers (ranks)
        for x in range(num_tokens):
            ax.text(x + 0.5, num_layers + 0.2, f"#{x+1}",
                   ha='center', va='center', fontsize=8,
                   color='gray', style='italic')
        
        # Set title and remove axes
        ax.set_title(title, fontsize=14, weight='bold', color=highlight_color, pad=20)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    # Plot both grids
    plot_grid(ax1, wrong_preds, "Wrong Format (Q: ... A:) → 9.11", '#e74c3c')
    plot_grid(ax2, correct_preds, "Correct Format (Answer:) → 9.8", '#27ae60')
    
    # Add main title
    fig.suptitle('Logit Lens Analysis: 9.11 vs 9.8 Bug\nToken Predictions Across Key Layers', 
                 fontsize=16, weight='bold', y=0.98)
    
    # Add legend
    legend_elements = [
        patches.Patch(edgecolor='red', facecolor='none', linewidth=3, label='Token "9"'),
        patches.Patch(edgecolor='blue', facecolor='none', linewidth=2, label='Hedge tokens (Both, 11)'),
        patches.Patch(facecolor='#2ecc71', label='High probability (>50%)'),
        patches.Patch(facecolor='#a3d5a3', label='Medium probability (20-50%)'),
        patches.Patch(facecolor='#f9e79f', label='Low probability (10-20%)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, 
              frameon=True, fancybox=True, shadow=True, bbox_to_anchor=(0.5, -0.02))
    
    # Add explanatory text
    fig.text(0.5, 0.01, 
            'Layer 25 (highlighted in purple) is the critical divergence point where paths separate',
            ha='center', fontsize=10, style='italic', color='purple')
    
    plt.tight_layout()
    
    # Save with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for ext in ['png', 'pdf']:
        filename = f'grid_visualization_{timestamp}.{ext}'
        plt.savefig(filename, dpi=300 if ext == 'png' else None, 
                   bbox_inches='tight', pad_inches=0.2)
        logger.info(f"✓ Saved {filename}")
    
    plt.show()

# Create alternative heatmap visualization
def create_heatmap_visualization():
    """Create a heatmap-style visualization with better text handling"""
    
    key_layers = [0, 7, 14, 15, 20, 25, 28, 30, 31, 32]
    
    # Run analysis
    logger.info("Running heatmap analysis...")
    wrong_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"
    wrong_preds = run_logit_lens(wrong_prompt)
    
    correct_prompt = "Which is bigger: 9.8 or 9.11?\nAnswer:"
    correct_preds = run_logit_lens(correct_prompt)
    
    # Prepare data for heatmap
    num_tokens = 10
    
    def prepare_heatmap_data(predictions):
        prob_matrix = []
        token_matrix = []
        
        for layer in key_layers:
            layer_preds = predictions[layer][:num_tokens]
            probs = [p for _, p in layer_preds]
            tokens = [t for t, _ in layer_preds]
            
            # Pad if needed
            while len(probs) < num_tokens:
                probs.append(0.0)
                tokens.append("")
            
            prob_matrix.append(probs)
            token_matrix.append(tokens)
        
        return np.array(prob_matrix), token_matrix
    
    wrong_probs, wrong_tokens = prepare_heatmap_data(wrong_preds)
    correct_probs, correct_tokens = prepare_heatmap_data(correct_preds)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    
    # Plot heatmaps
    sns.heatmap(wrong_probs, annot=wrong_tokens, fmt='', 
                cmap='YlOrRd', cbar_kws={'label': 'Probability'},
                ax=ax1, vmin=0, vmax=1,
                yticklabels=[f"Layer {l}" for l in key_layers],
                xticklabels=[f"Rank {i+1}" for i in range(num_tokens)])
    ax1.set_title("Wrong Format (Q: ... A:) → 9.11", fontsize=14, weight='bold', color='#e74c3c')
    
    sns.heatmap(correct_probs, annot=correct_tokens, fmt='', 
                cmap='YlGn', cbar_kws={'label': 'Probability'},
                ax=ax2, vmin=0, vmax=1,
                yticklabels=[f"Layer {l}" for l in key_layers],
                xticklabels=[f"Rank {i+1}" for i in range(num_tokens)])
    ax2.set_title("Correct Format (Answer:) → 9.8", fontsize=14, weight='bold', color='#27ae60')
    
    # Highlight Layer 25
    for ax in [ax1, ax2]:
        layer_25_idx = key_layers.index(25)
        ax.axhline(y=layer_25_idx, color='purple', linewidth=3, linestyle='--', alpha=0.7)
        ax.axhline(y=layer_25_idx+1, color='purple', linewidth=3, linestyle='--', alpha=0.7)
    
    plt.suptitle('Heatmap Visualization: Token Probabilities Across Layers', 
                fontsize=16, weight='bold')
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for ext in ['png', 'pdf']:
        filename = f'heatmap_visualization_{timestamp}.{ext}'
        plt.savefig(filename, dpi=300 if ext == 'png' else None, bbox_inches='tight')
        logger.info(f"✓ Saved {filename}")
    
    plt.show()

if __name__ == "__main__":
    logger.info("Creating grid visualization...")
    create_grid_visualization()
    
    logger.info("Creating heatmap visualization...")
    create_heatmap_visualization()
    
    logger.info("✓ All visualizations complete!")