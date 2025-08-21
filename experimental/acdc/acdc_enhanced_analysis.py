#!/usr/bin/env python3
"""
Enhanced ACDC Circuit Discovery with Path-Specific Analysis

This enhanced version performs more detailed analysis to identify
the distinct computational paths for correct vs incorrect answers.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PathSpecificACDC:
    """Enhanced ACDC with path-specific circuit discovery"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        logger.info(f"Loading model: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        
        # Model configuration
        self.n_layers = self.model.config.num_hidden_layers  # 32
        self.n_heads = self.model.config.num_attention_heads  # 32
        self.head_dim = self.model.config.hidden_size // self.n_heads
        
        # Storage for different prompt types
        self.correct_activations = {}
        self.incorrect_activations = {}
        self.correct_attention = {}
        self.incorrect_attention = {}
        
    def create_test_suite(self) -> Dict[str, List[str]]:
        """Create comprehensive test prompts"""
        return {
            "correct_formats": [
                "Which is bigger: 9.8 or 9.11?\nAnswer:",
                "9.8 vs 9.11, which is larger?\nAnswer:",
                "Compare: 9.8 and 9.11\nAnswer:",
            ],
            "incorrect_formats": [
                "Q: Which is bigger: 9.8 or 9.11?\nA:",
                "Question: Which is bigger: 9.8 or 9.11?\nAnswer:",
                "<|start_header_id|>user<|end_header_id|>\n\nWhich is bigger: 9.8 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            ]
        }
    
    def extract_path_specific_features(self, prompts: List[str], path_type: str):
        """Extract activations and attention patterns for a specific path"""
        all_layer_activations = {f"layer_{i}": [] for i in range(self.n_layers)}
        all_attention_patterns = {f"layer_{i}": [] for i in range(self.n_layers)}
        
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                # Get model outputs with attention
                outputs = self.model(
                    **inputs,
                    output_attentions=True,
                    output_hidden_states=True
                )
                
                # Extract hidden states for each layer
                hidden_states = outputs.hidden_states  # Tuple of tensors (n_layers + 1)
                for layer_idx in range(self.n_layers):
                    # Average over sequence length and batch
                    layer_activation = hidden_states[layer_idx + 1].mean(dim=1).squeeze(0)
                    all_layer_activations[f"layer_{layer_idx}"].append(layer_activation.cpu())
                
                # Extract attention patterns
                attentions = outputs.attentions  # Tuple of tensors (n_layers)
                seq_len = attentions[0].shape[-1]
                for layer_idx in range(self.n_layers):
                    # Average over heads and extract last token's attention
                    attn = attentions[layer_idx][0, :, -1, :].mean(dim=0)  # Average over heads
                    # Pad or truncate to fixed size for consistency
                    if attn.shape[0] < 20:
                        attn = F.pad(attn, (0, 20 - attn.shape[0]))
                    else:
                        attn = attn[:20]
                    all_attention_patterns[f"layer_{layer_idx}"].append(attn.cpu())
        
        # Average across all prompts of the same type
        avg_activations = {}
        avg_attention = {}
        
        for layer_key in all_layer_activations:
            if all_layer_activations[layer_key]:
                avg_activations[layer_key] = torch.stack(all_layer_activations[layer_key]).mean(dim=0)
                if all_attention_patterns[layer_key]:
                    avg_attention[layer_key] = torch.stack(all_attention_patterns[layer_key]).mean(dim=0)
        
        if path_type == "correct":
            self.correct_activations = avg_activations
            self.correct_attention = avg_attention
        else:
            self.incorrect_activations = avg_activations
            self.incorrect_attention = avg_attention
    
    def compute_path_divergence(self) -> Dict[str, float]:
        """Compute divergence between correct and incorrect paths at each layer"""
        divergence_scores = {}
        
        for layer_idx in range(self.n_layers):
            layer_key = f"layer_{layer_idx}"
            
            if layer_key in self.correct_activations and layer_key in self.incorrect_activations:
                # Compute cosine distance between activation patterns
                correct_act = self.correct_activations[layer_key]
                incorrect_act = self.incorrect_activations[layer_key]
                
                cos_sim = F.cosine_similarity(
                    correct_act.unsqueeze(0),
                    incorrect_act.unsqueeze(0)
                ).item()
                
                # Divergence is 1 - similarity
                divergence_scores[layer_idx] = 1.0 - cos_sim
                
                # Also compute L2 distance
                l2_dist = torch.norm(correct_act - incorrect_act).item()
                divergence_scores[f"layer_{layer_idx}_l2"] = l2_dist
        
        return divergence_scores
    
    def identify_critical_neurons(self, layer_idx: int, top_k: int = 100) -> Dict:
        """Identify neurons with largest activation differences at a specific layer"""
        layer_key = f"layer_{layer_idx}"
        
        if layer_key not in self.correct_activations or layer_key not in self.incorrect_activations:
            return {}
        
        correct_act = self.correct_activations[layer_key]
        incorrect_act = self.incorrect_activations[layer_key]
        
        # Compute absolute difference
        diff = torch.abs(correct_act - incorrect_act)
        
        # Get top-k neurons with largest differences
        top_values, top_indices = torch.topk(diff, min(top_k, diff.size(0)))
        
        critical_neurons = {
            "layer": layer_idx,
            "neuron_indices": top_indices.tolist(),
            "difference_values": top_values.tolist(),
            "correct_stronger": [],
            "incorrect_stronger": []
        }
        
        # Identify which path has stronger activation for each critical neuron
        for idx in top_indices:
            if correct_act[idx] > incorrect_act[idx]:
                critical_neurons["correct_stronger"].append(idx.item())
            else:
                critical_neurons["incorrect_stronger"].append(idx.item())
        
        return critical_neurons
    
    def trace_information_flow(self) -> Dict:
        """Trace how information flows differently through the network"""
        flow_analysis = {
            "layer_importance": {},
            "cumulative_divergence": [],
            "critical_transition_layers": []
        }
        
        cumulative_div = 0
        prev_div = 0
        
        for layer_idx in range(self.n_layers):
            layer_key = f"layer_{layer_idx}"
            
            if layer_key in self.correct_attention and layer_key in self.incorrect_attention:
                # Analyze attention pattern differences
                correct_attn = self.correct_attention[layer_key]
                incorrect_attn = self.incorrect_attention[layer_key]
                
                # KL divergence between attention distributions
                kl_div = F.kl_div(
                    torch.log_softmax(correct_attn, dim=-1),
                    torch.softmax(incorrect_attn, dim=-1),
                    reduction='sum'
                ).item()
                
                flow_analysis["layer_importance"][layer_idx] = kl_div
                cumulative_div += kl_div
                flow_analysis["cumulative_divergence"].append(cumulative_div)
                
                # Identify critical transition layers (sharp increases in divergence)
                if layer_idx > 0 and kl_div > 2 * prev_div and kl_div > 0.1:
                    flow_analysis["critical_transition_layers"].append(layer_idx)
                
                prev_div = kl_div
        
        return flow_analysis
    
    def visualize_path_analysis(self, save_prefix: str = "acdc_path"):
        """Create comprehensive visualizations of path-specific circuits"""
        fig = plt.figure(figsize=(20, 12))
        
        # Compute analyses
        divergence = self.compute_path_divergence()
        flow = self.trace_information_flow()
        
        # Plot 1: Layer-wise divergence
        ax1 = plt.subplot(2, 3, 1)
        layers = list(range(self.n_layers))
        div_values = [divergence.get(i, 0) for i in layers]
        ax1.plot(layers, div_values, 'b-', linewidth=2, marker='o')
        ax1.axvline(x=25, color='red', linestyle='--', alpha=0.7, label='Layer 25')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Path Divergence (1 - cosine similarity)')
        ax1.set_title('Activation Path Divergence')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Information flow importance
        ax2 = plt.subplot(2, 3, 2)
        importance = list(flow["layer_importance"].values())
        ax2.bar(layers[:len(importance)], importance, color='green', alpha=0.7)
        ax2.axvline(x=25, color='red', linestyle='--', alpha=0.7, label='Layer 25')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('KL Divergence of Attention')
        ax2.set_title('Attention Pattern Divergence')
        ax2.legend()
        
        # Plot 3: Cumulative divergence
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(layers[:len(flow["cumulative_divergence"])], 
                flow["cumulative_divergence"], 'purple', linewidth=2)
        ax3.axvline(x=25, color='red', linestyle='--', alpha=0.7, label='Layer 25')
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('Cumulative KL Divergence')
        ax3.set_title('Cumulative Information Divergence')
        ax3.legend()
        
        # Plot 4: Critical neurons at Layer 25
        ax4 = plt.subplot(2, 3, 4)
        layer_25_neurons = self.identify_critical_neurons(25, top_k=50)
        if layer_25_neurons:
            neuron_diffs = layer_25_neurons["difference_values"][:20]
            ax4.bar(range(len(neuron_diffs)), neuron_diffs, color='orange')
            ax4.set_xlabel('Top 20 Critical Neurons')
            ax4.set_ylabel('Activation Difference')
            ax4.set_title('Layer 25: Most Different Neurons')
        
        # Plot 5: Layer importance heatmap
        ax5 = plt.subplot(2, 3, 5)
        importance_matrix = np.zeros((1, self.n_layers))
        for i in range(self.n_layers):
            importance_matrix[0, i] = divergence.get(i, 0)
        
        sns.heatmap(importance_matrix, cmap='YlOrRd', cbar_kws={'label': 'Divergence'},
                   xticklabels=layers, yticklabels=[''], ax=ax5)
        ax5.set_title('Layer-wise Path Divergence Heatmap')
        ax5.axvline(x=25.5, color='blue', linewidth=2)
        
        # Plot 6: Critical transition layers
        ax6 = plt.subplot(2, 3, 6)
        if flow["critical_transition_layers"]:
            ax6.scatter(flow["critical_transition_layers"], 
                       [1] * len(flow["critical_transition_layers"]),
                       s=200, c='red', marker='v', label='Critical Transitions')
        ax6.set_xlim(-1, self.n_layers)
        ax6.set_ylim(0, 2)
        ax6.set_xlabel('Layer')
        ax6.set_title('Critical Transition Layers')
        ax6.axvline(x=25, color='green', linestyle='--', alpha=0.7, label='Layer 25')
        ax6.legend()
        
        plt.suptitle('ACDC Path-Specific Circuit Analysis: Decimal Comparison Bug', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_analysis.png", dpi=150, bbox_inches='tight')
        logger.info(f"Path analysis visualization saved to {save_prefix}_analysis.png")
        
        return fig
    
    def generate_circuit_report(self) -> Dict:
        """Generate comprehensive circuit analysis report"""
        divergence = self.compute_path_divergence()
        flow = self.trace_information_flow()
        
        # Analyze multiple critical layers
        critical_layers_analysis = {}
        for layer in [20, 23, 24, 25, 26, 27, 28, 30]:
            critical_layers_analysis[f"layer_{layer}"] = self.identify_critical_neurons(layer, top_k=20)
        
        # Find layer with maximum divergence
        max_div_layer = max(range(self.n_layers), key=lambda i: divergence.get(i, 0))
        
        report = {
            "summary": {
                "max_divergence_layer": max_div_layer,
                "max_divergence_value": divergence.get(max_div_layer, 0),
                "layer_25_divergence": divergence.get(25, 0),
                "critical_transition_layers": flow["critical_transition_layers"],
                "total_information_divergence": flow["cumulative_divergence"][-1] if flow["cumulative_divergence"] else 0
            },
            "layer_analysis": critical_layers_analysis,
            "divergence_profile": {str(k): v for k, v in divergence.items() if isinstance(k, int)},
            "attention_divergence": flow["layer_importance"]
        }
        
        return report
    
    def run_enhanced_analysis(self):
        """Run the enhanced ACDC analysis"""
        logger.info("Starting enhanced ACDC path-specific analysis")
        
        # Get test prompts
        test_suite = self.create_test_suite()
        
        # Extract features for each path
        logger.info("Extracting correct path features...")
        self.extract_path_specific_features(test_suite["correct_formats"], "correct")
        
        logger.info("Extracting incorrect path features...")
        self.extract_path_specific_features(test_suite["incorrect_formats"], "incorrect")
        
        # Generate comprehensive report
        report = self.generate_circuit_report()
        
        # Log key findings
        logger.info("\n" + "="*60)
        logger.info("ENHANCED CIRCUIT ANALYSIS RESULTS")
        logger.info("="*60)
        
        logger.info(f"\n🔍 Maximum Divergence:")
        logger.info(f"  - Layer: {report['summary']['max_divergence_layer']}")
        logger.info(f"  - Value: {report['summary']['max_divergence_value']:.4f}")
        
        logger.info(f"\n📍 Layer 25 Analysis:")
        logger.info(f"  - Divergence: {report['summary']['layer_25_divergence']:.4f}")
        layer_25_neurons = report['layer_analysis'].get('layer_25', {})
        if layer_25_neurons:
            logger.info(f"  - Critical neurons favoring correct: {len(layer_25_neurons.get('correct_stronger', []))}")
            logger.info(f"  - Critical neurons favoring incorrect: {len(layer_25_neurons.get('incorrect_stronger', []))}")
        
        logger.info(f"\n🔄 Critical Transition Layers: {report['summary']['critical_transition_layers']}")
        
        # Check if Layer 25 is among critical layers
        if 25 in report['summary']['critical_transition_layers']:
            logger.info("\n✅ Layer 25 confirmed as a critical transition layer!")
        elif 25 in [24, 26] or abs(report['summary']['max_divergence_layer'] - 25) <= 2:
            logger.info("\n⚠️ Layer 25 region (±2 layers) shows critical behavior!")
        
        # Create visualizations
        self.visualize_path_analysis("acdc_enhanced")
        
        # Save detailed report
        with open("acdc_enhanced_report.json", "w") as f:
            json.dump(report, f, indent=2, default=lambda x: x.tolist() if isinstance(x, torch.Tensor) else str(x))
        
        logger.info("\n📊 Enhanced analysis complete! Report saved to acdc_enhanced_report.json")
        
        return report


def main():
    """Main entry point for enhanced ACDC analysis"""
    analyzer = PathSpecificACDC()
    report = analyzer.run_enhanced_analysis()
    
    # Additional focused analysis on Layer 25
    logger.info("\n" + "="*60)
    logger.info("LAYER 25 DEEP DIVE")
    logger.info("="*60)
    
    layer_25_analysis = analyzer.identify_critical_neurons(25, top_k=100)
    if layer_25_analysis:
        logger.info(f"\nTop 10 most different neurons at Layer 25:")
        for i in range(min(10, len(layer_25_analysis["neuron_indices"]))):
            neuron_idx = layer_25_analysis["neuron_indices"][i]
            diff_value = layer_25_analysis["difference_values"][i]
            path = "correct" if neuron_idx in layer_25_analysis["correct_stronger"] else "incorrect"
            logger.info(f"  - Neuron {neuron_idx}: diff={diff_value:.4f}, stronger in {path} path")
    
    logger.info("\n🎯 Enhanced ACDC analysis complete!")


if __name__ == "__main__":
    main()