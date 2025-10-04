import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt

class LogitAttribution:
    """
    Decompose model's prediction into layer-by-layer contributions
    Based on "A Mathematical Framework for Transformer Circuits" approach
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        # Get token IDs for key predictions
        self.token_ids = {
            '9.8': tokenizer.encode('9.8', add_special_tokens=False)[0],
            '9.11': tokenizer.encode('9.11', add_special_tokens=False)[0],
            '9': tokenizer.encode('9', add_special_tokens=False)[0],
            '11': tokenizer.encode('11', add_special_tokens=False)[0],
            '8': tokenizer.encode('8', add_special_tokens=False)[0],
        }
    
    def get_residual_stream_contributions(self, prompt: str):
        """
        Track how each layer changes the residual stream's prediction
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Store contributions
        contributions = {
            'layers': [],
            'logit_diff': [],  # logit(9.11) - logit(9.8) after each layer
            'delta_logit_diff': [],  # change caused by each layer
            'key_neuron_contributions': {}
        }
        
        with torch.no_grad():
            # Get embeddings
            embeddings = self.model.model.embed_tokens(inputs['input_ids'])
            hidden_states = embeddings
            
            # Track initial prediction (just from embeddings)
            initial_logits = self.get_logits_from_hidden(hidden_states)
            last_diff = self.compute_logit_diff(initial_logits)
            
            contributions['layers'].append('embed')
            contributions['logit_diff'].append(last_diff)
            contributions['delta_logit_diff'].append(last_diff)
            
            # Go through each layer
            for layer_idx in range(len(self.model.model.layers)):
                # Save state before layer
                hidden_before = hidden_states.clone()
                
                # Apply layer
                layer = self.model.model.layers[layer_idx]
                hidden_states = layer(hidden_states)[0]
                
                # Get logits after this layer
                current_logits = self.get_logits_from_hidden(hidden_states)
                current_diff = self.compute_logit_diff(current_logits)
                
                # Calculate this layer's contribution
                delta = current_diff - last_diff
                
                contributions['layers'].append(f'L{layer_idx}')
                contributions['logit_diff'].append(current_diff)
                contributions['delta_logit_diff'].append(delta)
                
                # Special attention to key layers
                if layer_idx == 14:  # L14 with entangled neuron
                    contributions['key_neuron_contributions']['L14'] = {
                        'total_delta': delta,
                        'hidden_states': hidden_states.cpu().numpy()
                    }
                elif layer_idx == 25:  # L25 divergence point
                    contributions['key_neuron_contributions']['L25'] = {
                        'total_delta': delta,
                        'hidden_states': hidden_states.cpu().numpy()
                    }
                
                last_diff = current_diff
            
            # Final layer norm and output
            hidden_states = self.model.model.norm(hidden_states)
            final_logits = self.model.lm_head(hidden_states)
            final_diff = self.compute_logit_diff(final_logits)
            
            contributions['layers'].append('final')
            contributions['logit_diff'].append(final_diff)
            contributions['delta_logit_diff'].append(final_diff - last_diff)
        
        return contributions
    
    def get_logits_from_hidden(self, hidden_states):
        """Convert hidden states to logits"""
        # Apply final LN and unembedding
        normalized = self.model.model.norm(hidden_states)
        logits = self.model.lm_head(normalized)
        return logits
    
    def compute_logit_diff(self, logits):
        """
        Compute the key quantity: logit(wrong) - logit(correct)
        Positive means model prefers wrong answer
        """
        # Focus on last token position (where answer appears)
        last_logits = logits[0, -1, :]
        
        # For 9.11 vs 9.8, we might need to look at first token of each
        # Since "9.11" might tokenize differently
        logit_9 = last_logits[self.token_ids['9']].item()
        logit_8 = last_logits[self.token_ids['8']].item() if '8' in self.token_ids else 0
        
        # Simplified: track preference for starting with 9 vs 8
        # (In practice, you'd track the full sequence)
        return logit_9 - logit_8
    
    def decompose_mlp_contributions(self, prompt: str, layer_idx: int):
        """
        For a specific layer, decompose which MLP neurons contribute most
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        neuron_contributions = []
        
        with torch.no_grad():
            # Run forward pass up to target layer
            hidden_states = self.model.model.embed_tokens(inputs['input_ids'])
            
            for i in range(layer_idx):
                hidden_states = self.model.model.layers[i](hidden_states)[0]
            
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
            
            # For each neuron, calculate its contribution to logit difference
            for neuron_idx in range(intermediate.shape[-1]):
                # This neuron's activation
                neuron_act = intermediate[0, -1, neuron_idx].item()
                
                # Its contribution to the residual stream
                neuron_contribution = neuron_act * down_proj_weight[:, neuron_idx]
                
                # How much this changes the final logits
                # (Simplified - in practice you'd propagate through remaining layers)
                impact = torch.norm(neuron_contribution).item()
                
                neuron_contributions.append({
                    'neuron_id': neuron_idx,
                    'activation': neuron_act,
                    'impact': impact
                })
        
        # Sort by impact
        neuron_contributions.sort(key=lambda x: abs(x['impact']), reverse=True)
        
        return neuron_contributions[:10]  # Top 10 neurons

def run_logit_attribution_analysis():
    """
    Complete logit attribution analysis for the paper
    """
    print("="*60)
    print("LOGIT ATTRIBUTION ANALYSIS")
    print("="*60)
    
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    analyzer = LogitAttribution(model, tokenizer)
    
    # Test prompts
    qa_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"
    simple_prompt = "Which is bigger: 9.8 or 9.11?\nAnswer:"
    
    print("\n📊 Analyzing Q&A Format (Wrong)...")
    qa_contributions = analyzer.get_residual_stream_contributions(qa_prompt)
    
    print("\n📊 Analyzing Simple Format (Correct)...")
    simple_contributions = analyzer.get_residual_stream_contributions(simple_prompt)
    
    # Visualize the results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Cumulative logit difference
    ax1 = axes[0, 0]
    ax1.plot(range(len(qa_contributions['logit_diff'])), 
             qa_contributions['logit_diff'], 
             'r-', label='Q&A (Wrong)', linewidth=2)
    ax1.plot(range(len(simple_contributions['logit_diff'])), 
             simple_contributions['logit_diff'], 
             'g-', label='Simple (Correct)', linewidth=2)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Logit Difference (9 vs 8)')
    ax1.set_title('Cumulative Logit Difference Through Layers')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Highlight key layers
    ax1.axvspan(14, 15, alpha=0.2, color='orange', label='L14 (Entangled)')
    ax1.axvspan(25, 26, alpha=0.2, color='red', label='L25 (Divergence)')
    
    # Plot 2: Per-layer contribution
    ax2 = axes[0, 1]
    x_pos = range(len(qa_contributions['delta_logit_diff']))
    ax2.bar(x_pos, qa_contributions['delta_logit_diff'], 
            alpha=0.6, color='red', label='Q&A')
    ax2.bar(x_pos, simple_contributions['delta_logit_diff'], 
            alpha=0.6, color='green', label='Simple')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Layer Contribution to Logit Diff')
    ax2.set_title('Per-Layer Contributions')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Focus on layers 20-30
    ax3 = axes[1, 0]
    layers_20_30 = range(20, 31)
    qa_subset = qa_contributions['delta_logit_diff'][21:32]  # Adjust indices
    simple_subset = simple_contributions['delta_logit_diff'][21:32]
    
    ax3.plot(layers_20_30, qa_subset, 'r-o', label='Q&A', linewidth=2)
    ax3.plot(layers_20_30, simple_subset, 'g-o', label='Simple', linewidth=2)
    ax3.axvline(x=25, color='blue', linestyle='--', alpha=0.7, label='L25')
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Layer Contribution')
    ax3.set_title('Zoom: Layers 20-30 (Hedging Zone)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Top contributing neurons at L25
    ax4 = axes[1, 1]
    print("\n🔬 Analyzing top neurons at Layer 25...")
    qa_neurons = analyzer.decompose_mlp_contributions(qa_prompt, 25)
    
    neuron_ids = [n['neuron_id'] for n in qa_neurons[:5]]
    neuron_impacts = [n['impact'] for n in qa_neurons[:5]]
    
    ax4.barh(range(5), neuron_impacts, color='red', alpha=0.7)
    ax4.set_yticks(range(5))
    ax4.set_yticklabels([f"N{nid}" for nid in neuron_ids])
    ax4.set_xlabel('Impact on Logit Difference')
    ax4.set_title('Top 5 Neurons at Layer 25 (Q&A Format)')
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Logit Attribution: Mechanistic Path of the Decimal Bug', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('logit_attribution_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print key findings
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    
    # Find layers with biggest contribution differences
    diff_contributions = [abs(qa - simple) for qa, simple in 
                          zip(qa_contributions['delta_logit_diff'], 
                              simple_contributions['delta_logit_diff'])]
    
    max_diff_idx = np.argmax(diff_contributions)
    print(f"\n🎯 Largest divergence at: {qa_contributions['layers'][max_diff_idx]}")
    print(f"   Q&A contribution: {qa_contributions['delta_logit_diff'][max_diff_idx]:.3f}")
    print(f"   Simple contribution: {simple_contributions['delta_logit_diff'][max_diff_idx]:.3f}")
    
    # Report L14 and L25 specifically
    if 'L14' in qa_contributions['key_neuron_contributions']:
        print(f"\n📍 L14 (Entangled) contribution:")
        print(f"   Q&A: {qa_contributions['key_neuron_contributions']['L14']['total_delta']:.3f}")
    
    if 'L25' in qa_contributions['key_neuron_contributions']:
        print(f"\n📍 L25 (Divergence) contribution:")
        print(f"   Q&A: {qa_contributions['key_neuron_contributions']['L25']['total_delta']:.3f}")
    
    return qa_contributions, simple_contributions

if __name__ == "__main__":
    qa_contrib, simple_contrib = run_logit_attribution_analysis()