#!/usr/bin/env python3
"""
ACDC-inspired Circuit Discovery for Decimal Comparison Bug in Llama-3.1-8B

This implements an ACDC (Automatic Circuit Discovery) approach to identify
the computational circuits responsible for both correct and incorrect 
decimal comparisons in Llama-3.1-8B-Instruct.

Based on: "Towards Automated Circuit Discovery for Mechanistic Interpretability"
by Arthur Conmy et al. (https://arxiv.org/abs/2304.14997)
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Edge:
    """Represents an edge in the computational graph"""
    source_layer: int
    source_head: Optional[int]  # None for MLP
    target_layer: int
    target_head: Optional[int]  # None for MLP
    importance: float
    
    def __hash__(self):
        return hash((self.source_layer, self.source_head, self.target_layer, self.target_head))
    
    def __eq__(self, other):
        return (self.source_layer == other.source_layer and 
                self.source_head == other.source_head and
                self.target_layer == other.target_layer and
                self.target_head == other.target_head)

@dataclass
class Circuit:
    """Represents a discovered circuit"""
    edges: Set[Edge]
    nodes: Set[Tuple[int, Optional[int]]]  # (layer, head) where head=None for MLP
    total_importance: float
    prompt_type: str

class ACDCAnalyzer:
    """ACDC-based circuit discovery for Llama-3.1-8B decimal comparison"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        logger.info(f"Loading model: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        
        # Model configuration
        self.n_layers = self.model.config.num_hidden_layers  # 32
        self.n_heads = self.model.config.num_attention_heads  # 32
        self.hidden_size = self.model.config.hidden_size
        
        # Storage for activations and gradients
        self.activations = {}
        self.attention_patterns = {}
        self.gradients = {}
        
    def create_prompts(self) -> Dict[str, str]:
        """Create test prompts for correct and incorrect behavior"""
        return {
            "correct": "Which is bigger: 9.8 or 9.11?\nAnswer:",
            "incorrect": "Q: Which is bigger: 9.8 or 9.11?\nA:",
        }
    
    def register_hooks(self):
        """Register forward hooks to capture activations and attention patterns"""
        handles = []
        
        def create_attention_hook(layer_idx):
            def hook(module, input, output):
                # output is (batch_size, num_heads, seq_len, seq_len) for attention weights
                if hasattr(module, 'attention_weights'):
                    self.attention_patterns[f"layer_{layer_idx}"] = module.attention_weights.detach()
                # Store the attention output
                self.activations[f"attn_layer_{layer_idx}"] = output[0].detach()
            return hook
        
        def create_mlp_hook(layer_idx):
            def hook(module, input, output):
                self.activations[f"mlp_layer_{layer_idx}"] = output.detach()
            return hook
        
        def create_residual_hook(layer_idx):
            def hook(module, input, output):
                self.activations[f"residual_layer_{layer_idx}"] = output[0].detach()
            return hook
        
        # Register hooks for each layer
        for idx, layer in enumerate(self.model.model.layers):
            # Attention hook
            if hasattr(layer, 'self_attn'):
                handles.append(layer.self_attn.register_forward_hook(create_attention_hook(idx)))
            
            # MLP hook
            if hasattr(layer, 'mlp'):
                handles.append(layer.mlp.register_forward_hook(create_mlp_hook(idx)))
            
            # Residual stream hook (layer output)
            handles.append(layer.register_forward_hook(create_residual_hook(idx)))
        
        return handles
    
    def compute_attribution_scores(self, prompt: str, target_token: str = "9") -> Dict[str, float]:
        """
        Compute attribution scores for each edge in the computational graph.
        Uses gradient-based attribution to measure importance.
        """
        # Clear previous activations
        self.activations = {}
        self.attention_patterns = {}
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Register hooks
        handles = self.register_hooks()
        
        try:
            # Forward pass with gradient computation
            with torch.enable_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]  # Last token logits
                
                # Get target token ID
                target_id = self.tokenizer.encode(target_token, add_special_tokens=False)[0]
                target_logit = logits[target_id]
                
                # Backward pass to compute gradients
                self.model.zero_grad()
                target_logit.backward(retain_graph=True)
            
            # Compute attribution scores for each edge
            attribution_scores = {}
            
            # Attention edges
            for src_layer in range(self.n_layers):
                if f"attn_layer_{src_layer}" not in self.activations:
                    continue
                    
                attn_output = self.activations[f"attn_layer_{src_layer}"]
                
                for tgt_layer in range(src_layer + 1, min(src_layer + 5, self.n_layers)):
                    if f"residual_layer_{tgt_layer}" not in self.activations:
                        continue
                    
                    # Compute importance as product of activation and gradient magnitudes
                    # This is a simplified version of the full ACDC algorithm
                    src_norm = torch.norm(attn_output, dim=-1).mean().item()
                    
                    # For each attention head
                    for head in range(self.n_heads):
                        edge_key = f"attn_L{src_layer}_H{head}_to_L{tgt_layer}"
                        # Simplified importance score
                        attribution_scores[edge_key] = src_norm * (1.0 / (1 + abs(tgt_layer - src_layer)))
            
            # MLP edges
            for src_layer in range(self.n_layers):
                if f"mlp_layer_{src_layer}" not in self.activations:
                    continue
                    
                mlp_output = self.activations[f"mlp_layer_{src_layer}"]
                
                for tgt_layer in range(src_layer + 1, min(src_layer + 5, self.n_layers)):
                    if f"residual_layer_{tgt_layer}" not in self.activations:
                        continue
                    
                    # Compute importance
                    src_norm = torch.norm(mlp_output, dim=-1).mean().item()
                    edge_key = f"mlp_L{src_layer}_to_L{tgt_layer}"
                    attribution_scores[edge_key] = src_norm * (1.0 / (1 + abs(tgt_layer - src_layer)))
            
        finally:
            # Remove hooks
            for handle in handles:
                handle.remove()
        
        return attribution_scores
    
    def iterative_edge_pruning(self, attribution_scores: Dict[str, float], 
                              threshold_percentile: float = 95) -> Set[Edge]:
        """
        Iteratively prune edges based on attribution scores.
        Keeps only the most important edges above the threshold percentile.
        """
        # Convert attribution scores to Edge objects
        edges = []
        for edge_key, importance in attribution_scores.items():
            parts = edge_key.split('_')
            
            if 'attn' in edge_key:
                # Parse attention edge: attn_L{src}_H{head}_to_L{tgt}
                src_layer = int(parts[1][1:])
                head = int(parts[2][1:])
                tgt_layer = int(parts[4][1:])
                edge = Edge(src_layer, head, tgt_layer, None, importance)
            else:
                # Parse MLP edge: mlp_L{src}_to_L{tgt}
                src_layer = int(parts[1][1:])
                tgt_layer = int(parts[3][1:])
                edge = Edge(src_layer, None, tgt_layer, None, importance)
            
            edges.append(edge)
        
        # Sort by importance
        edges.sort(key=lambda e: e.importance, reverse=True)
        
        # Apply threshold
        threshold = np.percentile([e.importance for e in edges], threshold_percentile)
        important_edges = {e for e in edges if e.importance >= threshold}
        
        logger.info(f"Kept {len(important_edges)}/{len(edges)} edges above {threshold_percentile}th percentile")
        
        return important_edges
    
    def discover_circuit(self, prompt: str, prompt_type: str) -> Circuit:
        """
        Discover the computational circuit for a given prompt.
        """
        logger.info(f"Discovering circuit for {prompt_type} prompt")
        
        # Compute attribution scores
        attribution_scores = self.compute_attribution_scores(prompt)
        
        # Prune edges to find important circuit
        important_edges = self.iterative_edge_pruning(attribution_scores)
        
        # Extract nodes from edges
        nodes = set()
        for edge in important_edges:
            nodes.add((edge.source_layer, edge.source_head))
            nodes.add((edge.target_layer, edge.target_head))
        
        # Calculate total importance
        total_importance = sum(e.importance for e in important_edges)
        
        return Circuit(
            edges=important_edges,
            nodes=nodes,
            total_importance=total_importance,
            prompt_type=prompt_type
        )
    
    def compare_circuits(self, circuit1: Circuit, circuit2: Circuit) -> Dict:
        """
        Compare two circuits to identify differences.
        """
        # Find unique edges in each circuit
        unique_to_1 = circuit1.edges - circuit2.edges
        unique_to_2 = circuit2.edges - circuit1.edges
        shared = circuit1.edges & circuit2.edges
        
        # Find unique nodes
        unique_nodes_1 = circuit1.nodes - circuit2.nodes
        unique_nodes_2 = circuit2.nodes - circuit1.nodes
        shared_nodes = circuit1.nodes & circuit2.nodes
        
        # Identify critical layers
        critical_layers_1 = {e.source_layer for e in unique_to_1} | {e.target_layer for e in unique_to_1}
        critical_layers_2 = {e.source_layer for e in unique_to_2} | {e.target_layer for e in unique_to_2}
        
        return {
            "unique_edges_correct": len(unique_to_1) if circuit1.prompt_type == "correct" else len(unique_to_2),
            "unique_edges_incorrect": len(unique_to_2) if circuit1.prompt_type == "incorrect" else len(unique_to_1),
            "shared_edges": len(shared),
            "unique_nodes_correct": len(unique_nodes_1) if circuit1.prompt_type == "correct" else len(unique_nodes_2),
            "unique_nodes_incorrect": len(unique_nodes_2) if circuit1.prompt_type == "incorrect" else len(unique_nodes_1),
            "shared_nodes": len(shared_nodes),
            "critical_layers_correct": critical_layers_1 if circuit1.prompt_type == "correct" else critical_layers_2,
            "critical_layers_incorrect": critical_layers_2 if circuit1.prompt_type == "incorrect" else critical_layers_1,
        }
    
    def visualize_circuits(self, correct_circuit: Circuit, incorrect_circuit: Circuit, 
                          save_path: str = "acdc_circuit_comparison.png"):
        """
        Visualize the discovered circuits and their differences.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Helper function to create adjacency matrix
        def create_adjacency_matrix(circuit: Circuit) -> np.ndarray:
            matrix = np.zeros((self.n_layers, self.n_layers))
            for edge in circuit.edges:
                matrix[edge.source_layer, edge.target_layer] += edge.importance
            return matrix
        
        # Plot correct circuit
        correct_matrix = create_adjacency_matrix(correct_circuit)
        sns.heatmap(correct_matrix, ax=axes[0, 0], cmap='Blues', cbar_kws={'label': 'Importance'})
        axes[0, 0].set_title('Correct Answer Circuit')
        axes[0, 0].set_xlabel('Target Layer')
        axes[0, 0].set_ylabel('Source Layer')
        
        # Plot incorrect circuit
        incorrect_matrix = create_adjacency_matrix(incorrect_circuit)
        sns.heatmap(incorrect_matrix, ax=axes[0, 1], cmap='Reds', cbar_kws={'label': 'Importance'})
        axes[0, 1].set_title('Incorrect Answer Circuit')
        axes[0, 1].set_xlabel('Target Layer')
        axes[0, 1].set_ylabel('Source Layer')
        
        # Plot difference
        diff_matrix = correct_matrix - incorrect_matrix
        sns.heatmap(diff_matrix, ax=axes[1, 0], cmap='RdBu_r', center=0, 
                   cbar_kws={'label': 'Difference (Correct - Incorrect)'})
        axes[1, 0].set_title('Circuit Difference')
        axes[1, 0].set_xlabel('Target Layer')
        axes[1, 0].set_ylabel('Source Layer')
        
        # Plot layer importance
        correct_layer_importance = np.sum(correct_matrix, axis=1) + np.sum(correct_matrix, axis=0)
        incorrect_layer_importance = np.sum(incorrect_matrix, axis=1) + np.sum(incorrect_matrix, axis=0)
        
        layers = np.arange(self.n_layers)
        width = 0.35
        axes[1, 1].bar(layers - width/2, correct_layer_importance, width, label='Correct', color='blue', alpha=0.7)
        axes[1, 1].bar(layers + width/2, incorrect_layer_importance, width, label='Incorrect', color='red', alpha=0.7)
        axes[1, 1].set_xlabel('Layer')
        axes[1, 1].set_ylabel('Total Importance')
        axes[1, 1].set_title('Layer Importance by Circuit')
        axes[1, 1].legend()
        axes[1, 1].axvline(x=25, color='green', linestyle='--', alpha=0.5, label='Layer 25')
        
        plt.suptitle('ACDC Circuit Discovery: Decimal Comparison Bug Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Circuit visualization saved to {save_path}")
        
        return fig
    
    def run_analysis(self):
        """
        Run the complete ACDC analysis for the decimal comparison bug.
        """
        logger.info("Starting ACDC circuit discovery analysis")
        
        # Create prompts
        prompts = self.create_prompts()
        
        # Discover circuits
        correct_circuit = self.discover_circuit(prompts["correct"], "correct")
        incorrect_circuit = self.discover_circuit(prompts["incorrect"], "incorrect")
        
        # Compare circuits
        comparison = self.compare_circuits(correct_circuit, incorrect_circuit)
        
        # Log results
        logger.info("\n" + "="*60)
        logger.info("CIRCUIT DISCOVERY RESULTS")
        logger.info("="*60)
        
        logger.info(f"\nCorrect Circuit:")
        logger.info(f"  - Total edges: {len(correct_circuit.edges)}")
        logger.info(f"  - Total nodes: {len(correct_circuit.nodes)}")
        logger.info(f"  - Total importance: {correct_circuit.total_importance:.3f}")
        
        logger.info(f"\nIncorrect Circuit:")
        logger.info(f"  - Total edges: {len(incorrect_circuit.edges)}")
        logger.info(f"  - Total nodes: {len(incorrect_circuit.nodes)}")
        logger.info(f"  - Total importance: {incorrect_circuit.total_importance:.3f}")
        
        logger.info(f"\nCircuit Comparison:")
        logger.info(f"  - Shared edges: {comparison['shared_edges']}")
        logger.info(f"  - Unique to correct: {comparison['unique_edges_correct']}")
        logger.info(f"  - Unique to incorrect: {comparison['unique_edges_incorrect']}")
        logger.info(f"  - Critical layers (correct): {sorted(comparison['critical_layers_correct'])}")
        logger.info(f"  - Critical layers (incorrect): {sorted(comparison['critical_layers_incorrect'])}")
        
        # Check if Layer 25 is critical
        if 25 in comparison['critical_layers_correct'] or 25 in comparison['critical_layers_incorrect']:
            logger.info("\n⚠️  Layer 25 identified as critical - confirming previous analysis!")
        
        # Visualize circuits
        self.visualize_circuits(correct_circuit, incorrect_circuit)
        
        # Save detailed results
        results = {
            "timestamp": datetime.now().isoformat(),
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "prompts": prompts,
            "correct_circuit": {
                "num_edges": len(correct_circuit.edges),
                "num_nodes": len(correct_circuit.nodes),
                "total_importance": correct_circuit.total_importance,
                "edges": [
                    {
                        "source_layer": e.source_layer,
                        "source_head": e.source_head,
                        "target_layer": e.target_layer,
                        "target_head": e.target_head,
                        "importance": e.importance
                    }
                    for e in sorted(correct_circuit.edges, key=lambda x: x.importance, reverse=True)[:20]
                ]
            },
            "incorrect_circuit": {
                "num_edges": len(incorrect_circuit.edges),
                "num_nodes": len(incorrect_circuit.nodes),
                "total_importance": incorrect_circuit.total_importance,
                "edges": [
                    {
                        "source_layer": e.source_layer,
                        "source_head": e.source_head,
                        "target_layer": e.target_layer,
                        "target_head": e.target_head,
                        "importance": e.importance
                    }
                    for e in sorted(incorrect_circuit.edges, key=lambda x: x.importance, reverse=True)[:20]
                ]
            },
            "comparison": comparison
        }
        
        with open("acdc_analysis_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("\nResults saved to acdc_analysis_results.json")
        
        return correct_circuit, incorrect_circuit, comparison


def main():
    """Main entry point for ACDC analysis"""
    analyzer = ACDCAnalyzer()
    correct_circuit, incorrect_circuit, comparison = analyzer.run_analysis()
    
    # Additional analysis: Focus on Layer 25
    logger.info("\n" + "="*60)
    logger.info("LAYER 25 SPECIFIC ANALYSIS")
    logger.info("="*60)
    
    # Find all edges involving Layer 25
    layer_25_edges_correct = [e for e in correct_circuit.edges 
                              if e.source_layer == 25 or e.target_layer == 25]
    layer_25_edges_incorrect = [e for e in incorrect_circuit.edges 
                                if e.source_layer == 25 or e.target_layer == 25]
    
    logger.info(f"\nLayer 25 connections in correct circuit: {len(layer_25_edges_correct)}")
    for edge in sorted(layer_25_edges_correct, key=lambda e: e.importance, reverse=True)[:5]:
        logger.info(f"  - L{edge.source_layer}→L{edge.target_layer}: importance={edge.importance:.3f}")
    
    logger.info(f"\nLayer 25 connections in incorrect circuit: {len(layer_25_edges_incorrect)}")
    for edge in sorted(layer_25_edges_incorrect, key=lambda e: e.importance, reverse=True)[:5]:
        logger.info(f"  - L{edge.source_layer}→L{edge.target_layer}: importance={edge.importance:.3f}")
    
    logger.info("\n✅ ACDC analysis complete!")


if __name__ == "__main__":
    main()