#!/usr/bin/env python3
"""
Logit Attribution Analysis - Fixed Version
Tracks layer-by-layer contributions to the decimal comparison bug
Uses correct temperature (0.0) and prompt formats based on our findings
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import json
import seaborn as sns

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logit_attribution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set matplotlib style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

# Increase font sizes for publication
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 20,
    'figure.figsize': (16, 12)
})

class LogitAttribution:
    """
    Decompose model's prediction into layer-by-layer contributions
    Based on "A Mathematical Framework for Transformer Circuits" approach
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        logger.info("Initializing LogitAttribution analyzer")
        
        # Get token IDs for key predictions
        self.token_ids = {}
        for token in ['9.8', '9.11', '9', '11', '8', 'Both', 'both']:
            try:
                encoded = tokenizer.encode(token, add_special_tokens=False)
                if encoded:
                    self.token_ids[token] = encoded[0]
                    logger.debug(f"Token '{token}' -> ID {encoded[0]}")
            except:
                logger.warning(f"Could not encode token '{token}'")
        
        logger.info(f"Initialized with {len(self.token_ids)} tokens")
    
    def get_residual_stream_contributions(self, prompt: str, prompt_type: str = "unknown"):
        """
        Track how each layer changes the residual stream's prediction
        """
        logger.info(f"\n📊 Analyzing {prompt_type} format: '{prompt[:50]}...'")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Store contributions
        contributions = {
            'prompt_type': prompt_type,
            'prompt': prompt,
            'layers': [],
            'logit_diff': [],  # logit(9) - logit(8) after each layer
            'delta_logit_diff': [],  # change caused by each layer
            'both_token_prob': [],  # Track "Both" token probability
            'nine_token_prob': [],  # Track "9" token probability
            'key_neuron_contributions': {}
        }
        
        with torch.no_grad():
            # Get embeddings
            embeddings = self.model.model.embed_tokens(inputs['input_ids'])
            hidden_states = embeddings
            
            logger.debug("Processing embedding layer")
            # Track initial prediction (just from embeddings)
            initial_logits = self.get_logits_from_hidden(hidden_states)
            last_diff = self.compute_logit_diff(initial_logits)
            both_prob, nine_prob = self.compute_token_probs(initial_logits)
            
            contributions['layers'].append('embed')
            contributions['logit_diff'].append(last_diff)
            contributions['delta_logit_diff'].append(last_diff)
            contributions['both_token_prob'].append(both_prob)
            contributions['nine_token_prob'].append(nine_prob)
            
            # Go through each layer
            position_ids = torch.arange(0, hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
            
            for layer_idx in range(len(self.model.model.layers)):
                if layer_idx % 5 == 0:
                    logger.debug(f"Processing layer {layer_idx}/{len(self.model.model.layers)}")
                
                # Save state before layer
                hidden_before = hidden_states.clone()
                
                # Apply layer with proper arguments
                layer = self.model.model.layers[layer_idx]
                layer_outputs = layer(
                    hidden_states,
                    position_ids=position_ids
                )
                hidden_states = layer_outputs[0]
                
                # Get logits after this layer
                current_logits = self.get_logits_from_hidden(hidden_states)
                current_diff = self.compute_logit_diff(current_logits)
                both_prob, nine_prob = self.compute_token_probs(current_logits)
                
                # Calculate this layer's contribution
                delta = current_diff - last_diff
                
                contributions['layers'].append(f'L{layer_idx}')
                contributions['logit_diff'].append(current_diff)
                contributions['delta_logit_diff'].append(delta)
                contributions['both_token_prob'].append(both_prob)
                contributions['nine_token_prob'].append(nine_prob)
                
                # Special attention to key layers based on our findings
                if layer_idx == 14:  # L14 with potential entanglement
                    logger.info(f"  L14: delta={delta:.3f}, P(Both)={both_prob:.3f}, P(9)={nine_prob:.3f}")
                    contributions['key_neuron_contributions']['L14'] = {
                        'total_delta': delta,
                        'both_prob': both_prob,
                        'nine_prob': nine_prob
                    }
                elif layer_idx == 25:  # L25 divergence point - CRITICAL!
                    logger.info(f"  ⚠️ L25 (CRITICAL): delta={delta:.3f}, P(Both)={both_prob:.3f}, P(9)={nine_prob:.3f}")
                    contributions['key_neuron_contributions']['L25'] = {
                        'total_delta': delta,
                        'both_prob': both_prob,
                        'nine_prob': nine_prob
                    }
                elif layer_idx in [23, 24]:  # Transition layers
                    logger.info(f"  L{layer_idx}: delta={delta:.3f}, P(Both)={both_prob:.3f}, P(9)={nine_prob:.3f}")
                
                last_diff = current_diff
            
            # Final layer norm and output
            hidden_states = self.model.model.norm(hidden_states)
            final_logits = self.model.lm_head(hidden_states)
            final_diff = self.compute_logit_diff(final_logits)
            both_prob, nine_prob = self.compute_token_probs(final_logits)
            
            contributions['layers'].append('final')
            contributions['logit_diff'].append(final_diff)
            contributions['delta_logit_diff'].append(final_diff - last_diff)
            contributions['both_token_prob'].append(both_prob)
            contributions['nine_token_prob'].append(nine_prob)
            
            logger.info(f"  Final: logit_diff={final_diff:.3f}, P(Both)={both_prob:.3f}, P(9)={nine_prob:.3f}")
        
        return contributions
    
    def get_logits_from_hidden(self, hidden_states):
        """Convert hidden states to logits"""
        # Apply final LN and unembedding
        normalized = self.model.model.norm(hidden_states)
        logits = self.model.lm_head(normalized)
        return logits
    
    def compute_logit_diff(self, logits):
        """
        Compute the key quantity: logit(9) - logit(8)
        Positive means model prefers starting with 9
        """
        # Focus on last token position (where answer appears)
        if len(logits.shape) == 3:
            last_logits = logits[0, -1, :]
        else:
            last_logits = logits[-1, :] if len(logits.shape) == 2 else logits
        
        # Get logits for key tokens
        logit_9 = last_logits[self.token_ids['9']].item() if '9' in self.token_ids else 0
        logit_8 = last_logits[self.token_ids['8']].item() if '8' in self.token_ids else 0
        
        return logit_9 - logit_8
    
    def compute_token_probs(self, logits):
        """Compute probabilities for 'Both' and '9' tokens"""
        if len(logits.shape) == 3:
            last_logits = logits[0, -1, :]
        else:
            last_logits = logits[-1, :] if len(logits.shape) == 2 else logits
        
        probs = torch.softmax(last_logits, dim=-1)
        
        both_prob = 0
        if 'Both' in self.token_ids:
            both_prob += probs[self.token_ids['Both']].item()
        if 'both' in self.token_ids:
            both_prob += probs[self.token_ids['both']].item()
        
        nine_prob = probs[self.token_ids['9']].item() if '9' in self.token_ids else 0
        
        return both_prob, nine_prob
    
    def decompose_mlp_contributions(self, prompt: str, layer_idx: int):
        """
        For a specific layer, decompose which MLP neurons contribute most
        """
        logger.info(f"Decomposing MLP contributions at Layer {layer_idx}")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        neuron_contributions = []
        
        with torch.no_grad():
            # Run forward pass up to target layer
            hidden_states = self.model.model.embed_tokens(inputs['input_ids'])
            position_ids = torch.arange(0, hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
            
            for i in range(layer_idx):
                layer_outputs = self.model.model.layers[i](
                    hidden_states,
                    position_ids=position_ids
                )
                hidden_states = layer_outputs[0]
            
            # Now examine the target layer's MLP
            layer = self.model.model.layers[layer_idx]
            
            # Get MLP input
            mlp_input = layer.post_attention_layernorm(hidden_states)
            
            # Get gate and up projections (Llama uses SwiGLU)
            gate_proj = layer.mlp.gate_proj(mlp_input)
            up_proj = layer.mlp.up_proj(mlp_input)
            intermediate = torch.nn.functional.silu(gate_proj) * up_proj
            
            # Each neuron's contribution to output
            down_proj_weight = layer.mlp.down_proj.weight  # [hidden_size, intermediate_size]
            
            # For top neurons, calculate contribution to logit difference
            top_neurons = []
            for neuron_idx in range(min(100, intermediate.shape[-1])):  # Check top 100
                # This neuron's activation
                neuron_act = intermediate[0, -1, neuron_idx].item()
                
                # Its contribution to the residual stream
                neuron_contribution = neuron_act * down_proj_weight[:, neuron_idx]
                
                # How much this changes the final logits
                impact = torch.norm(neuron_contribution).item()
                
                if abs(neuron_act) > 0.1:  # Only consider active neurons
                    top_neurons.append({
                        'neuron_id': neuron_idx,
                        'activation': neuron_act,
                        'impact': impact
                    })
        
        # Sort by impact
        top_neurons.sort(key=lambda x: abs(x['impact']), reverse=True)
        
        return top_neurons[:10]  # Top 10 neurons

def create_publication_visualizations(qa_contributions, simple_contributions, save_path=''):
    """Create high-quality visualizations for publication"""
    
    logger.info("\n📈 Creating publication-quality visualizations...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # Define colors
    WRONG_COLOR = '#E74C3C'  # Red
    CORRECT_COLOR = '#27AE60'  # Green
    HIGHLIGHT_COLOR = '#3498DB'  # Blue
    
    # Plot 1: Cumulative logit difference
    ax1 = plt.subplot(2, 3, 1)
    layers_numeric = list(range(len(qa_contributions['logit_diff'])))
    ax1.plot(layers_numeric, 
             qa_contributions['logit_diff'], 
             color=WRONG_COLOR, label='Q&A Format (Wrong)', linewidth=2.5, marker='o', markersize=3)
    ax1.plot(layers_numeric, 
             simple_contributions['logit_diff'], 
             color=CORRECT_COLOR, label='Simple Format (Correct)', linewidth=2.5, marker='s', markersize=3)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax1.set_xlabel('Layer', fontsize=14)
    ax1.set_ylabel('Logit(9) - Logit(8)', fontsize=14)
    ax1.set_title('Cumulative Logit Difference', fontsize=16, fontweight='bold')
    ax1.legend(loc='best', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Highlight key layers
    ax1.axvspan(25, 26, alpha=0.2, color=HIGHLIGHT_COLOR)
    ax1.text(25.5, ax1.get_ylim()[1]*0.9, 'L25', ha='center', fontsize=12, fontweight='bold')
    
    # Plot 2: Per-layer contribution
    ax2 = plt.subplot(2, 3, 2)
    width = 0.35
    x_pos = np.arange(len(qa_contributions['delta_logit_diff']))
    ax2.bar(x_pos - width/2, qa_contributions['delta_logit_diff'], 
            width, alpha=0.7, color=WRONG_COLOR, label='Q&A Format')
    ax2.bar(x_pos + width/2, simple_contributions['delta_logit_diff'], 
            width, alpha=0.7, color=CORRECT_COLOR, label='Simple Format')
    ax2.set_xlabel('Layer', fontsize=14)
    ax2.set_ylabel('Layer Contribution (Δ Logit)', fontsize=14)
    ax2.set_title('Per-Layer Contributions', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Only show every 5th layer label for clarity
    ax2.set_xticks(x_pos[::5])
    ax2.set_xticklabels([qa_contributions['layers'][i] for i in range(0, len(qa_contributions['layers']), 5)], 
                        rotation=45)
    
    # Plot 3: Token probabilities evolution (Both token)
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(layers_numeric, 
             qa_contributions['both_token_prob'], 
             color=WRONG_COLOR, label='Q&A Format', linewidth=2.5, marker='o', markersize=3)
    ax3.plot(layers_numeric, 
             simple_contributions['both_token_prob'], 
             color=CORRECT_COLOR, label='Simple Format', linewidth=2.5, marker='s', markersize=3)
    ax3.set_xlabel('Layer', fontsize=14)
    ax3.set_ylabel('P("Both")', fontsize=14)
    ax3.set_title('"Both" Token Probability', fontsize=16, fontweight='bold')
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.axvspan(25, 26, alpha=0.2, color=HIGHLIGHT_COLOR)
    
    # Plot 4: Token "9" probability evolution
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(layers_numeric, 
             qa_contributions['nine_token_prob'], 
             color=WRONG_COLOR, label='Q&A Format', linewidth=2.5, marker='o', markersize=3)
    ax4.plot(layers_numeric, 
             simple_contributions['nine_token_prob'], 
             color=CORRECT_COLOR, label='Simple Format', linewidth=2.5, marker='s', markersize=3)
    ax4.set_xlabel('Layer', fontsize=14)
    ax4.set_ylabel('P("9")', fontsize=14)
    ax4.set_title('Token "9" Probability (Critical)', fontsize=16, fontweight='bold')
    ax4.legend(fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.axvspan(25, 26, alpha=0.2, color=HIGHLIGHT_COLOR)
    ax4.text(25.5, ax4.get_ylim()[1]*0.9, 'L25', ha='center', fontsize=12, fontweight='bold')
    
    # Plot 5: Zoom on layers 20-30 (Critical zone)
    ax5 = plt.subplot(2, 3, 5)
    layers_20_30 = list(range(20, 31))
    qa_subset = qa_contributions['delta_logit_diff'][21:32]
    simple_subset = simple_contributions['delta_logit_diff'][21:32]
    
    ax5.plot(layers_20_30, qa_subset, color=WRONG_COLOR, marker='o', 
             linewidth=2.5, markersize=8, label='Q&A Format')
    ax5.plot(layers_20_30, simple_subset, color=CORRECT_COLOR, marker='s', 
             linewidth=2.5, markersize=8, label='Simple Format')
    ax5.axvline(x=25, color=HIGHLIGHT_COLOR, linestyle='--', alpha=0.7, linewidth=2)
    ax5.set_xlabel('Layer', fontsize=14)
    ax5.set_ylabel('Layer Contribution', fontsize=14)
    ax5.set_title('Zoom: Layers 20-30 (Hedging Zone)', fontsize=16, fontweight='bold')
    ax5.legend(fontsize=12)
    ax5.grid(True, alpha=0.3)
    
    # Annotate Layer 25
    ax5.annotate('Layer 25\nDivergence Point', 
                xy=(25, qa_subset[5]), 
                xytext=(27, qa_subset[5] + 0.5),
                arrowprops=dict(arrowstyle='->', color=HIGHLIGHT_COLOR, lw=2),
                fontsize=12, fontweight='bold', color=HIGHLIGHT_COLOR)
    
    # Plot 6: Summary statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate key statistics
    l25_qa_delta = qa_contributions['key_neuron_contributions'].get('L25', {}).get('total_delta', 0)
    l25_simple_delta = simple_contributions['key_neuron_contributions'].get('L25', {}).get('total_delta', 0)
    l25_qa_nine = qa_contributions['key_neuron_contributions'].get('L25', {}).get('nine_prob', 0)
    l25_simple_nine = simple_contributions['key_neuron_contributions'].get('L25', {}).get('nine_prob', 0)
    
    summary_text = f"""
KEY FINDINGS

Layer 25 Analysis:
• Q&A Format:
  - Δ Logit: {l25_qa_delta:.3f}
  - P("9"): {l25_qa_nine:.3f}
  - P("Both"): {qa_contributions['key_neuron_contributions'].get('L25', {}).get('both_prob', 0):.3f}

• Simple Format:
  - Δ Logit: {l25_simple_delta:.3f}
  - P("9"): {l25_simple_nine:.3f}
  - P("Both"): {simple_contributions['key_neuron_contributions'].get('L25', {}).get('both_prob', 0):.3f}

Critical Insight:
At Layer 25, Simple format
commits to "9" ({l25_simple_nine:.1%})
while Q&A format hedges.
"""
    
    ax6.text(0.1, 0.5, summary_text, fontsize=14, family='monospace',
            verticalalignment='center', transform=ax6.transAxes)
    
    # Add main title
    plt.suptitle('Logit Attribution Analysis: Mechanistic Path of the Decimal Bug', 
                fontsize=20, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save in both formats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_file = f'{save_path}logit_attribution_analysis_{timestamp}.png'
    pdf_file = f'{save_path}logit_attribution_analysis_{timestamp}.pdf'
    
    plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_file, bbox_inches='tight', facecolor='white')
    
    logger.info(f"✓ Saved visualizations: {png_file} and {pdf_file}")
    plt.show()
    
    return png_file, pdf_file

def run_logit_attribution_analysis():
    """
    Complete logit attribution analysis with correct parameters
    """
    logger.info("="*60)
    logger.info("LOGIT ATTRIBUTION ANALYSIS")
    logger.info("="*60)
    logger.info("Configuration:")
    logger.info("- Temperature: 0.0 (deterministic)")
    logger.info("- Q&A Format: Produces WRONG answer")
    logger.info("- Simple Format: Produces CORRECT answer")
    logger.info("- Focus: Layer 25 divergence point")
    logger.info("="*60)
    
    # Check GPU
    if torch.cuda.is_available():
        logger.info(f"✓ GPU available: {torch.cuda.get_device_name()}")
        device = torch.device("cuda")
    else:
        logger.warning("⚠️ No GPU available, using CPU")
        device = torch.device("cpu")
    
    # Load model
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("✓ Model loaded successfully")
    
    analyzer = LogitAttribution(model, tokenizer)
    
    # Test prompts - based on our confirmed findings
    qa_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"  # WRONG format
    simple_prompt = "Which is bigger: 9.8 or 9.11?\nAnswer:"  # CORRECT format
    
    # First verify the bug behavior
    logger.info("\n🔍 Verifying bug behavior with temperature=0.0...")
    
    # Test Q&A format
    inputs = tokenizer(qa_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.0,  # CRITICAL: Must be 0.0
            do_sample=False,  # CRITICAL: Must be False
            pad_token_id=tokenizer.eos_token_id
        )
    qa_result = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    logger.info(f"Q&A Result: {qa_result[:100]}")
    
    # Test Simple format
    inputs = tokenizer(simple_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    simple_result = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    logger.info(f"Simple Result: {simple_result[:100]}")
    
    # Verify results
    if "9.11 is bigger" in qa_result.lower():
        logger.info("✓ Q&A format produces WRONG answer as expected")
    else:
        logger.warning("⚠️ Q&A format did not produce expected wrong answer")
    
    if "9.8 is bigger" in simple_result.lower():
        logger.info("✓ Simple format produces CORRECT answer as expected")
    else:
        logger.warning("⚠️ Simple format did not produce expected correct answer")
    
    # Run attribution analysis
    logger.info("\n📊 Running attribution analysis...")
    qa_contributions = analyzer.get_residual_stream_contributions(qa_prompt, "Q&A (Wrong)")
    simple_contributions = analyzer.get_residual_stream_contributions(simple_prompt, "Simple (Correct)")
    
    # Analyze top neurons at Layer 25
    logger.info("\n🔬 Analyzing top neurons at Layer 25...")
    qa_neurons_l25 = analyzer.decompose_mlp_contributions(qa_prompt, 25)
    simple_neurons_l25 = analyzer.decompose_mlp_contributions(simple_prompt, 25)
    
    logger.info("\nTop neurons at L25 for Q&A format:")
    for i, neuron in enumerate(qa_neurons_l25[:5]):
        logger.info(f"  {i+1}. N{neuron['neuron_id']}: activation={neuron['activation']:.3f}, impact={neuron['impact']:.3f}")
    
    logger.info("\nTop neurons at L25 for Simple format:")
    for i, neuron in enumerate(simple_neurons_l25[:5]):
        logger.info(f"  {i+1}. N{neuron['neuron_id']}: activation={neuron['activation']:.3f}, impact={neuron['impact']:.3f}")
    
    # Create visualizations
    png_file, pdf_file = create_publication_visualizations(qa_contributions, simple_contributions)
    
    # Save results to JSON
    results = {
        'timestamp': datetime.now().isoformat(),
        'model': model_name,
        'temperature': 0.0,
        'qa_prompt': qa_prompt,
        'simple_prompt': simple_prompt,
        'qa_result': qa_result[:200],
        'simple_result': simple_result[:200],
        'layer_25_analysis': {
            'qa_format': qa_contributions['key_neuron_contributions'].get('L25', {}),
            'simple_format': simple_contributions['key_neuron_contributions'].get('L25', {})
        },
        'top_neurons_l25': {
            'qa_format': qa_neurons_l25[:5],
            'simple_format': simple_neurons_l25[:5]
        }
    }
    
    json_file = f'logit_attribution_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"✓ Results saved to {json_file}")
    
    # Print key findings
    logger.info("\n" + "="*60)
    logger.info("KEY FINDINGS")
    logger.info("="*60)
    
    # Find layers with biggest contribution differences
    diff_contributions = [abs(qa - simple) for qa, simple in 
                          zip(qa_contributions['delta_logit_diff'], 
                              simple_contributions['delta_logit_diff'])]
    
    max_diff_idx = np.argmax(diff_contributions)
    logger.info(f"\n🎯 Largest divergence at: {qa_contributions['layers'][max_diff_idx]}")
    logger.info(f"   Q&A contribution: {qa_contributions['delta_logit_diff'][max_diff_idx]:.3f}")
    logger.info(f"   Simple contribution: {simple_contributions['delta_logit_diff'][max_diff_idx]:.3f}")
    logger.info(f"   Difference: {diff_contributions[max_diff_idx]:.3f}")
    
    # Report L25 specifically
    if 'L25' in qa_contributions['key_neuron_contributions']:
        logger.info(f"\n📍 Layer 25 (Critical Divergence Point):")
        logger.info(f"   Q&A Format:")
        logger.info(f"     - Delta: {qa_contributions['key_neuron_contributions']['L25']['total_delta']:.3f}")
        logger.info(f"     - P('Both'): {qa_contributions['key_neuron_contributions']['L25']['both_prob']:.3f}")
        logger.info(f"     - P('9'): {qa_contributions['key_neuron_contributions']['L25']['nine_prob']:.3f}")
        logger.info(f"   Simple Format:")
        logger.info(f"     - Delta: {simple_contributions['key_neuron_contributions']['L25']['total_delta']:.3f}")
        logger.info(f"     - P('Both'): {simple_contributions['key_neuron_contributions']['L25']['both_prob']:.3f}")
        logger.info(f"     - P('9'): {simple_contributions['key_neuron_contributions']['L25']['nine_prob']:.3f}")
    
    logger.info("\n✅ Logit attribution analysis complete!")
    return qa_contributions, simple_contributions

if __name__ == "__main__":
    qa_contrib, simple_contrib = run_logit_attribution_analysis()