import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as path_effects

def create_complete_mechanism_figure():
    """
    A hand-drawn style diagram showing the cascade mechanism with two paths
    """
    
    # Set up the figure with dark background
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_facecolor('#1a1a2e')  # Dark blue background
    fig.patch.set_facecolor('#1a1a2e')
    
    # Use a readable font
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 12
    
    # Define colors
    colors = {
        'input': '#90EE90',      # Light green for input nodes
        'hidden': '#87CEEB',     # Light blue for hidden nodes
        'white': '#FFFFFF',      # White for text and lines
        'orange': '#FFB347',     # Light orange for highlights
        'green_path': '#27AE60', # Green for correct path
        'red_path': '#E74C3C',   # Red for wrong path
        'entanglement': '#FFD700' # Gold for entanglement
    }
    
    # Layer positions (left to right) - cascade mechanism
    x_positions = [2, 4, 6, 8, 10, 12, 14]
    y_center = 6
    
    # Draw input layer (format detection)
    input_nodes = 3
    for i in range(input_nodes):
        y_pos = y_center - 1 + i * 1
        circle = plt.Circle((x_positions[0], y_pos), 0.3, 
                           facecolor=colors['input'], 
                           edgecolor=colors['white'], 
                           linewidth=2, alpha=0.8)
        ax.add_patch(circle)
        labels = ['9.9', '9.11', 'Format']
        ax.text(x_positions[0], y_pos, labels[i], 
                ha='center', va='center', fontsize=14, 
                color=colors['white'], fontweight='bold')
    
    # Add ellipsis for input layer
    ax.text(x_positions[0], y_center + 2.5, '...', 
            ha='center', va='center', fontsize=20, 
            color=colors['white'], fontweight='bold')
    
    # Define the two paths
    green_path_layers = [
        (4, "Layer 6", "Format Detection\n2.499 KL divergence", "entanglement"),
        (6, "Layer 14", "Entangled Features\nL14/N12639", "normal"),
        (8, "Layer 23", "Hedging Begins\nN290, N2742", "normal"),
        (10, "Layer 25", "Commitment Point\n22.1% vs 12.9%", "entanglement"),
        (12, "Layer 30", "Final Amplification\nPath locked in", "normal")
    ]
    
    red_path_layers = [
        (4, "Layer 6", "Format Detection\n2.499 KL divergence", "entanglement"),
        (6, "Layer 14", "Entangled Features\nL14/N12639", "normal"),
        (8, "Layer 23", "Hedging Begins\nN290, N2742", "normal"),
        (10, "Layer 25", "Commitment Point\n22.1% vs 12.9%", "entanglement"),
        (12, "Layer 30", "Final Amplification\nPath locked in", "normal")
    ]
    
    # Draw green path (correct)
    for i, (x_pos, title, description, highlight_type) in enumerate(green_path_layers):
        # Draw nodes
        for j in range(3):
            y_pos = y_center - 1 + j * 1
            circle = plt.Circle((x_pos, y_pos), 0.3, 
                               facecolor=colors['hidden'], 
                               edgecolor=colors['green_path'], 
                               linewidth=2, alpha=0.8)
            ax.add_patch(circle)
        
        # Add ellipsis
        ax.text(x_pos, y_center + 2, '...', 
                ha='center', va='center', fontsize=20, 
                color=colors['green_path'], fontweight='bold')
        
        # Add layer title - MUCH LARGER
        ax.text(x_pos, y_center - 2.5, title, 
                ha='center', va='center', fontsize=16, 
                color=colors['green_path'], fontweight='bold')
        
        # Add description - MUCH LARGER
        ax.text(x_pos, y_center + 3.5, description, 
                ha='center', va='center', fontsize=14, 
                color=colors['green_path'], alpha=0.9)
        
        # Draw connections from previous layer
        if i > 0:
            prev_x = green_path_layers[i-1][0]
            for j in range(3):
                y_pos = y_center - 1 + j * 1
                # Sketchy arrows in green
                for k in range(2):
                    offset = (k-0.5) * 0.2
                    x1, y1 = prev_x + 0.3, y_center - 1 + j * 1 + offset
                    x2, y2 = x_pos - 0.3, y_pos + offset
                    x_mid = (x1 + x2) / 2
                    y_mid = (y1 + y2) / 2 + np.random.uniform(-0.1, 0.1)
                    ax.plot([x1, x_mid, x2], [y1, y_mid, y2], 
                           color=colors['green_path'], linewidth=1.5, alpha=0.7)
        
        # Add entanglement highlighting
        if highlight_type == "entanglement":
            # Gold highlight around entanglement layers
            ent_rect = patches.Rectangle((x_pos - 0.8, y_center - 3), 1.6, 6, 
                                       linewidth=3, edgecolor=colors['entanglement'], 
                                       facecolor='none', linestyle='-', alpha=0.8)
            ax.add_patch(ent_rect)
            # Add entanglement label - MUCH LARGER
            ax.text(x_pos, y_center - 3.5, "ENTANGLEMENT", 
                    ha='center', va='center', fontsize=14, 
                    color=colors['entanglement'], fontweight='bold')
    
    # Draw red path (wrong) - offset to the right
    for i, (x_pos, title, description, highlight_type) in enumerate(red_path_layers):
        x_pos_red = x_pos + 0.8  # Offset to the right
        
        # Draw nodes
        for j in range(3):
            y_pos = y_center - 1 + j * 1
            circle = plt.Circle((x_pos_red, y_pos), 0.3, 
                               facecolor=colors['hidden'], 
                               edgecolor=colors['red_path'], 
                               linewidth=2, alpha=0.8)
            ax.add_patch(circle)
        
        # Add ellipsis
        ax.text(x_pos_red, y_center + 2, '...', 
                ha='center', va='center', fontsize=20, 
                color=colors['red_path'], fontweight='bold')
        
        # Add layer title - MUCH LARGER
        ax.text(x_pos_red, y_center - 2.5, title, 
                ha='center', va='center', fontsize=16, 
                color=colors['red_path'], fontweight='bold')
        
        # Add description - MUCH LARGER
        ax.text(x_pos_red, y_center + 3.5, description, 
                ha='center', va='center', fontsize=14, 
                color=colors['red_path'], alpha=0.9)
        
        # Draw connections from previous layer
        if i > 0:
            prev_x = red_path_layers[i-1][0] + 0.8
            for j in range(3):
                y_pos = y_center - 1 + j * 1
                # Sketchy arrows in red
                for k in range(2):
                    offset = (k-0.5) * 0.2
                    x1, y1 = prev_x + 0.3, y_center - 1 + j * 1 + offset
                    x2, y2 = x_pos_red - 0.3, y_pos + offset
                    x_mid = (x1 + x2) / 2
                    y_mid = (y1 + y2) / 2 + np.random.uniform(-0.1, 0.1)
                    ax.plot([x1, x_mid, x2], [y1, y_mid, y2], 
                           color=colors['red_path'], linewidth=1.5, alpha=0.7)
        
        # Add entanglement highlighting
        if highlight_type == "entanglement":
            # Gold highlight around entanglement layers
            ent_rect = patches.Rectangle((x_pos_red - 0.8, y_center - 3), 1.6, 6, 
                                       linewidth=3, edgecolor=colors['entanglement'], 
                                       facecolor='none', linestyle='-', alpha=0.8)
            ax.add_patch(ent_rect)
            # Add entanglement label - MUCH LARGER
            ax.text(x_pos_red, y_center - 3.5, "ENTANGLEMENT", 
                    ha='center', va='center', fontsize=14, 
                    color=colors['entanglement'], fontweight='bold')
    
    # Add path labels - MUCH LARGER
    ax.text(4, y_center + 4.5, "Simple Format\n→ Correct Path", 
            ha='center', va='center', fontsize=16, 
            color=colors['green_path'], fontweight='bold')
    
    ax.text(4.8, y_center + 4.5, "Q&A Format\n→ Wrong Path", 
            ha='center', va='center', fontsize=16, 
            color=colors['red_path'], fontweight='bold')
    
    # Add intervention markers
    intervention_layers = [8, 8.5, 9, 9.5, 10]
    for layer in intervention_layers:
        ax.scatter(15, layer, marker='x', s=150, color=colors['red_path'], 
                  linewidth=3, alpha=0.8)
    
    # Add intervention label - MUCH LARGER
    ax.text(15, y_center - 1, "Failed\nInterventions", 
            ha='center', va='center', fontsize=14, 
            color=colors['red_path'], fontweight='bold', rotation=90)
    
    # Add annotation - MUCH LARGER
    ax.text(8, y_center + 5.5, "Entanglement at Layers 6 & 25:\nFormat determines outcome", 
            ha='center', va='center', fontsize=16, 
            color=colors['white'], fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='none', 
                     edgecolor=colors['white'], alpha=0.5))
    
    # Add title - MUCH LARGER
    ax.text(8, y_center + 6.5, "Complete Cascade: Two Paths with Entanglement", 
            ha='center', va='center', fontsize=20, 
            color=colors['white'], fontweight='bold')
    
    # Set up the plot
    ax.set_xlim(0, 16)
    ax.set_ylim(y_center - 5, y_center + 7)
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Create the figure
    fig = create_complete_mechanism_figure()
    
    # Save the figure
    plt.savefig('complete_cascade_mechanism.png', dpi=300, bbox_inches='tight')
    plt.savefig('complete_cascade_mechanism.pdf', bbox_inches='tight')
    
    # Show the figure
    plt.show()
    
    print("Complete cascade mechanism figure saved as:")
    print("- complete_cascade_mechanism.png")
    print("- complete_cascade_mechanism.pdf")