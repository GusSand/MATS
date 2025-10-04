#!/usr/bin/env python3
"""
Functional Clustering Analysis: Testing the Even/Odd Critique
============================================================

This investigation tests whether our even/odd head findings represent genuine
functional specialization or coincidental indexing by analyzing whether heads
cluster functionally independent of their indices.

Critical Question: Do the "working" heads work because they're even-indexed,
or because they happen to have similar functional properties that could be
found in other heads regardless of index?
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager
import json
from datetime import datetime
import warnings
import os
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
import itertools

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Model configuration
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LAYER_OF_INTEREST = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FunctionalClusteringAnalyzer:
    """Analyze whether attention heads cluster by function or by index"""

    def __init__(self, model_name: str = MODEL_NAME, device: str = DEVICE):
        self.device = device
        self.model_name = model_name

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()

        self.n_heads = 32
        self.layer_idx = LAYER_OF_INTEREST
        self.saved_activations = {}
        self.hooks = []

        # Known results from previous experiments
        self.known_working_even_heads = [
            [0, 2, 4, 6, 8, 10, 12, 14],     # First 8 even - 100% success
            [16, 18, 20, 22, 24, 26, 28, 30], # Last 8 even - 100% success
            [0, 4, 8, 12, 16, 20, 24, 28]     # Every other even - 100% success
        ]

        self.known_failing_combinations = [
            [0, 4, 12, 14, 16, 18, 26, 30],   # Failed even combination
            [1, 3, 5, 7, 9, 11, 13, 15],      # First 8 odd - always fail
            [2, 4, 6, 8, 12, 14, 18, 30]     # Another failed even combination
        ]

    def extract_attention_weight_matrices(self) -> Dict[str, torch.Tensor]:
        """Extract weight matrices for all attention heads at target layer"""
        print("Extracting attention weight matrices...")

        attention_layer = self.model.model.layers[self.layer_idx].self_attn

        # Get the weight matrices
        weight_matrices = {
            'q_proj': attention_layer.q_proj.weight.data.cpu(),
            'k_proj': attention_layer.k_proj.weight.data.cpu(),
            'v_proj': attention_layer.v_proj.weight.data.cpu(),
            'o_proj': attention_layer.o_proj.weight.data.cpu()
        }

        # Print actual shapes to debug
        for name, weight in weight_matrices.items():
            print(f"{name} shape: {weight.shape}")

        hidden_size = weight_matrices['q_proj'].shape[1]  # Input dimension
        head_dim = hidden_size // self.n_heads

        print(f"Hidden size: {hidden_size}, Head dim: {head_dim}, N heads: {self.n_heads}")

        # Since we need to analyze functional similarity, let's just concatenate all weights
        # for each head rather than trying to reshape complex tensor structures
        head_weights = {}

        # Handle different projection dimensions (Llama uses grouped query attention)
        # Q: [4096, 4096] -> 32 heads * 128 dim each
        # K,V: [1024, 4096] -> 8 groups * 128 dim each (shared across 4 heads each)

        # Q projection: full multi-head
        q_weight = weight_matrices['q_proj']  # [4096, 4096]
        head_weights['q_proj'] = q_weight.view(self.n_heads, head_dim, hidden_size)

        # K,V projections: grouped (8 groups for 32 heads = 4 heads per group)
        for proj_name in ['k_proj', 'v_proj']:
            weight = weight_matrices[proj_name]  # [1024, 4096]
            out_features, in_features = weight.shape

            if out_features == 1024:  # 8 groups * 128 dim
                n_groups = 8
                group_dim = 128
                # Reshape to [n_groups, group_dim, hidden_size]
                grouped_weight = weight.view(n_groups, group_dim, in_features)

                # Expand to all 32 heads (each group covers 4 heads)
                expanded_weight = torch.zeros(self.n_heads, head_dim, hidden_size)
                for group_idx in range(n_groups):
                    for head_in_group in range(4):
                        head_idx = group_idx * 4 + head_in_group
                        expanded_weight[head_idx] = grouped_weight[group_idx]

                head_weights[proj_name] = expanded_weight
            else:
                print(f"Warning: Unexpected shape for {proj_name}: {weight.shape}")
                head_weights[proj_name] = weight

        # For o_proj: input is concatenated head outputs, output is hidden_size
        o_weight = weight_matrices['o_proj']  # [hidden_size, n_heads * head_dim]
        out_features, in_features = o_weight.shape

        if in_features == hidden_size:  # n_heads * head_dim
            head_weights['o_proj'] = o_weight.view(out_features, self.n_heads, head_dim)
        else:
            print(f"Warning: Unexpected shape for o_proj: {o_weight.shape}")
            head_weights['o_proj'] = o_weight

        print(f"Extracted weight matrices for {self.n_heads} heads")
        print(f"Q/K/V shapes: {head_weights['q_proj'].shape}")
        print(f"O shape: {head_weights['o_proj'].shape}")

        return head_weights

    def compute_head_functional_similarity(self, head_weights: Dict[str, torch.Tensor]) -> np.ndarray:
        """Compute functional similarity between all pairs of attention heads"""
        print("Computing functional similarity between heads...")

        # Create feature vectors for each head by concatenating all its weight matrices
        head_features = []

        for head_idx in range(self.n_heads):
            # Extract weights for this head
            q_weights = head_weights['q_proj'][head_idx].flatten()  # [head_dim * hidden_size]
            k_weights = head_weights['k_proj'][head_idx].flatten()
            v_weights = head_weights['v_proj'][head_idx].flatten()
            o_weights = head_weights['o_proj'][:, head_idx, :].flatten()  # [hidden_size * head_dim]

            # Concatenate all weights for this head
            head_feature = torch.cat([q_weights, k_weights, v_weights, o_weights])
            head_features.append(head_feature.numpy())

        head_features = np.array(head_features)  # [n_heads, total_features]
        print(f"Head feature vectors shape: {head_features.shape}")

        # Normalize features (important for similarity computation)
        scaler = StandardScaler()
        head_features_normalized = scaler.fit_transform(head_features)

        # Compute pairwise cosine similarity
        similarity_matrix = np.zeros((self.n_heads, self.n_heads))

        for i in range(self.n_heads):
            for j in range(self.n_heads):
                # Cosine similarity
                dot_product = np.dot(head_features_normalized[i], head_features_normalized[j])
                norm_i = np.linalg.norm(head_features_normalized[i])
                norm_j = np.linalg.norm(head_features_normalized[j])
                similarity_matrix[i, j] = dot_product / (norm_i * norm_j)

        print("Functional similarity matrix computed")
        return similarity_matrix, head_features_normalized

    def perform_hierarchical_clustering(self, similarity_matrix: np.ndarray, n_clusters: int = 8) -> Dict:
        """Perform hierarchical clustering on attention heads"""
        print(f"Performing hierarchical clustering with {n_clusters} clusters...")

        # Convert similarity to distance (1 - similarity)
        distance_matrix = 1 - similarity_matrix

        # Fix diagonal for precomputed distance matrix
        np.fill_diagonal(distance_matrix, 0)

        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            linkage='average'
        )

        cluster_labels = clustering.fit_predict(distance_matrix)

        # Also compute linkage for dendrogram
        condensed_distances = pdist(1 - similarity_matrix, metric='euclidean')
        linkage_matrix = linkage(condensed_distances, method='average')

        # Compute clustering quality metrics
        silhouette_avg = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')

        # Group heads by cluster
        clusters = {}
        for cluster_id in range(n_clusters):
            clusters[cluster_id] = [i for i, label in enumerate(cluster_labels) if label == cluster_id]

        print(f"Clustering completed. Silhouette score: {silhouette_avg:.3f}")
        print("Cluster assignments:")
        for cluster_id, heads in clusters.items():
            even_heads = [h for h in heads if h % 2 == 0]
            odd_heads = [h for h in heads if h % 2 == 1]
            print(f"  Cluster {cluster_id}: {heads} (Even: {even_heads}, Odd: {odd_heads})")

        return {
            'cluster_labels': cluster_labels,
            'clusters': clusters,
            'linkage_matrix': linkage_matrix,
            'silhouette_score': silhouette_avg,
            'distance_matrix': distance_matrix
        }

    def test_even_odd_correlation(self, cluster_labels: np.ndarray) -> Dict:
        """Test correlation between functional clusters and even/odd indices"""
        print("Testing correlation between functional clusters and even/odd indices...")

        # Create even/odd labels (0 for even, 1 for odd)
        even_odd_labels = np.array([head_idx % 2 for head_idx in range(self.n_heads)])

        # Compute Adjusted Rand Index (measures clustering similarity)
        ari_score = adjusted_rand_score(even_odd_labels, cluster_labels)

        # Compute correlation between cluster assignments and even/odd
        cluster_correlation, cluster_p_value = spearmanr(even_odd_labels, cluster_labels)

        # Analyze cluster composition
        cluster_composition = {}
        n_clusters = len(set(cluster_labels))

        for cluster_id in range(n_clusters):
            cluster_heads = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            even_count = sum(1 for h in cluster_heads if h % 2 == 0)
            odd_count = len(cluster_heads) - even_count

            cluster_composition[cluster_id] = {
                'heads': cluster_heads,
                'even_count': even_count,
                'odd_count': odd_count,
                'even_ratio': even_count / len(cluster_heads) if cluster_heads else 0,
                'size': len(cluster_heads)
            }

        # Find most "pure" clusters (high even or odd ratio)
        pure_even_clusters = [cid for cid, comp in cluster_composition.items()
                             if comp['even_ratio'] > 0.75 and comp['size'] >= 4]
        pure_odd_clusters = [cid for cid, comp in cluster_composition.items()
                            if comp['even_ratio'] < 0.25 and comp['size'] >= 4]

        print(f"Adjusted Rand Index (clustering similarity): {ari_score:.3f}")
        print(f"Spearman correlation with even/odd: {cluster_correlation:.3f} (p={cluster_p_value:.3f})")
        print(f"Pure even clusters (>75% even): {pure_even_clusters}")
        print(f"Pure odd clusters (<25% even): {pure_odd_clusters}")

        return {
            'ari_score': ari_score,
            'cluster_correlation': cluster_correlation,
            'cluster_p_value': cluster_p_value,
            'cluster_composition': cluster_composition,
            'pure_even_clusters': pure_even_clusters,
            'pure_odd_clusters': pure_odd_clusters
        }

    def analyze_known_working_combinations(self, similarity_matrix: np.ndarray, cluster_labels: np.ndarray) -> Dict:
        """Analyze how known working/failing combinations relate to functional clusters"""
        print("Analyzing known working/failing combinations...")

        analysis = {}

        # Analyze known working combinations
        for i, working_combo in enumerate(self.known_working_even_heads):
            combo_name = f"working_combo_{i}"

            # Compute internal similarity within the combination
            combo_similarities = []
            for h1, h2 in itertools.combinations(working_combo, 2):
                combo_similarities.append(similarity_matrix[h1, h2])

            internal_similarity = np.mean(combo_similarities)

            # Check cluster assignments
            combo_clusters = [cluster_labels[h] for h in working_combo]
            unique_clusters = len(set(combo_clusters))

            analysis[combo_name] = {
                'heads': working_combo,
                'type': 'working',
                'internal_similarity': internal_similarity,
                'cluster_assignments': combo_clusters,
                'n_unique_clusters': unique_clusters,
                'cluster_purity': 1 - (unique_clusters - 1) / (len(working_combo) - 1)  # How clustered they are
            }

        # Analyze known failing combinations
        for i, failing_combo in enumerate(self.known_failing_combinations):
            combo_name = f"failing_combo_{i}"

            # Compute internal similarity
            combo_similarities = []
            for h1, h2 in itertools.combinations(failing_combo, 2):
                combo_similarities.append(similarity_matrix[h1, h2])

            internal_similarity = np.mean(combo_similarities)

            # Check cluster assignments
            combo_clusters = [cluster_labels[h] for h in failing_combo]
            unique_clusters = len(set(combo_clusters))

            analysis[combo_name] = {
                'heads': failing_combo,
                'type': 'failing',
                'internal_similarity': internal_similarity,
                'cluster_assignments': combo_clusters,
                'n_unique_clusters': unique_clusters,
                'cluster_purity': 1 - (unique_clusters - 1) / (len(failing_combo) - 1)
            }

        # Compare working vs failing combinations
        working_similarities = [data['internal_similarity'] for name, data in analysis.items()
                              if data['type'] == 'working']
        failing_similarities = [data['internal_similarity'] for name, data in analysis.items()
                              if data['type'] == 'failing']

        working_purities = [data['cluster_purity'] for name, data in analysis.items()
                          if data['type'] == 'working']
        failing_purities = [data['cluster_purity'] for name, data in analysis.items()
                          if data['type'] == 'failing']

        print(f"Working combinations - avg similarity: {np.mean(working_similarities):.3f}")
        print(f"Failing combinations - avg similarity: {np.mean(failing_similarities):.3f}")
        print(f"Working combinations - avg cluster purity: {np.mean(working_purities):.3f}")
        print(f"Failing combinations - avg cluster purity: {np.mean(failing_purities):.3f}")

        analysis['summary'] = {
            'working_avg_similarity': np.mean(working_similarities),
            'failing_avg_similarity': np.mean(failing_similarities),
            'working_avg_purity': np.mean(working_purities),
            'failing_avg_purity': np.mean(failing_purities)
        }

        return analysis

    def predict_working_heads_by_function(self, similarity_matrix: np.ndarray, cluster_labels: np.ndarray) -> Dict:
        """Predict which heads should work based purely on functional similarity"""
        print("Predicting working heads based on functional similarity...")

        # Strategy 1: Find the most functionally coherent groups of 8 heads
        predictions = {}

        # Method 1: Use the largest pure clusters
        cluster_composition = {}
        n_clusters = len(set(cluster_labels))

        for cluster_id in range(n_clusters):
            cluster_heads = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            cluster_composition[cluster_id] = cluster_heads

        # Find clusters that could form working 8-head combinations
        potential_combinations = []

        # Single large cluster (if any cluster has 8+ heads)
        for cluster_id, heads in cluster_composition.items():
            if len(heads) >= 8:
                potential_combinations.append({
                    'heads': heads[:8],  # Take first 8
                    'method': f'single_cluster_{cluster_id}',
                    'basis': 'largest_functional_cluster'
                })

        # Combine similar clusters
        for cluster_id1 in range(n_clusters):
            for cluster_id2 in range(cluster_id1 + 1, n_clusters):
                heads1 = cluster_composition[cluster_id1]
                heads2 = cluster_composition[cluster_id2]

                if len(heads1) + len(heads2) >= 8:
                    # Check inter-cluster similarity
                    inter_similarities = []
                    for h1 in heads1:
                        for h2 in heads2:
                            inter_similarities.append(similarity_matrix[h1, h2])

                    avg_inter_similarity = np.mean(inter_similarities)

                    if avg_inter_similarity > 0.5:  # Threshold for similarity
                        combined_heads = heads1 + heads2
                        potential_combinations.append({
                            'heads': combined_heads[:8],
                            'method': f'combined_clusters_{cluster_id1}_{cluster_id2}',
                            'basis': 'high_inter_cluster_similarity',
                            'inter_similarity': avg_inter_similarity
                        })

        # Method 2: Find 8 most similar heads to known working heads
        reference_working_heads = self.known_working_even_heads[0]  # Use first 8 even as reference

        # Compute average similarity to reference working heads
        similarity_to_working = []
        for head_idx in range(self.n_heads):
            similarities = [similarity_matrix[head_idx, ref_head] for ref_head in reference_working_heads]
            avg_similarity = np.mean(similarities)
            similarity_to_working.append((head_idx, avg_similarity))

        # Sort by similarity and take top 8
        similarity_to_working.sort(key=lambda x: x[1], reverse=True)
        top_8_similar = [head_idx for head_idx, _ in similarity_to_working[:8]]

        potential_combinations.append({
            'heads': top_8_similar,
            'method': 'most_similar_to_known_working',
            'basis': 'functional_similarity_to_reference'
        })

        predictions['potential_combinations'] = potential_combinations

        # Analyze predictions
        for combo in potential_combinations:
            heads = combo['heads']
            even_count = sum(1 for h in heads if h % 2 == 0)
            odd_count = 8 - even_count

            combo['even_count'] = even_count
            combo['odd_count'] = odd_count
            combo['even_ratio'] = even_count / 8

            print(f"Prediction ({combo['method']}): {heads}")
            print(f"  Even/Odd split: {even_count}E/{odd_count}O ({combo['even_ratio']:.1%} even)")

        return predictions

    def create_comprehensive_visualization(self, similarity_matrix: np.ndarray, clustering_results: Dict,
                                         correlation_results: Dict, combination_analysis: Dict):
        """Create comprehensive visualization of functional clustering results"""
        print("Creating comprehensive visualization...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Plot 1: Similarity matrix heatmap
        ax1 = axes[0, 0]
        im1 = ax1.imshow(similarity_matrix, cmap='RdYlBu', vmin=-1, vmax=1)
        ax1.set_title('Head Functional Similarity Matrix')
        ax1.set_xlabel('Head Index')
        ax1.set_ylabel('Head Index')

        # Add even/odd indicators
        for i in range(self.n_heads):
            color = 'red' if i % 2 == 0 else 'blue'
            ax1.axhline(i, color=color, alpha=0.3, linewidth=0.5)
            ax1.axvline(i, color=color, alpha=0.3, linewidth=0.5)

        plt.colorbar(im1, ax=ax1)

        # Plot 2: Dendrogram
        ax2 = axes[0, 1]
        dendrogram(clustering_results['linkage_matrix'], ax=ax2, orientation='top')
        ax2.set_title('Hierarchical Clustering Dendrogram')
        ax2.set_xlabel('Head Index')
        ax2.set_ylabel('Distance')

        # Color code even/odd
        xlbls = ax2.get_xmajorticklabels()
        for lbl in xlbls:
            if lbl.get_text():
                head_idx = int(lbl.get_text())
                color = 'red' if head_idx % 2 == 0 else 'blue'
                lbl.set_color(color)

        # Plot 3: Cluster composition
        ax3 = axes[0, 2]
        cluster_composition = correlation_results['cluster_composition']
        cluster_ids = list(cluster_composition.keys())
        even_ratios = [cluster_composition[cid]['even_ratio'] for cid in cluster_ids]
        cluster_sizes = [cluster_composition[cid]['size'] for cid in cluster_ids]

        scatter = ax3.scatter(cluster_ids, even_ratios, s=[size*50 for size in cluster_sizes],
                             c=even_ratios, cmap='RdYlBu', alpha=0.7)
        ax3.set_title('Cluster Even/Odd Composition')
        ax3.set_xlabel('Cluster ID')
        ax3.set_ylabel('Even Head Ratio')
        ax3.axhline(0.5, color='black', linestyle='--', alpha=0.5)
        plt.colorbar(scatter, ax=ax3)

        # Plot 4: Working vs Failing similarity
        ax4 = axes[1, 0]
        working_sims = [data['internal_similarity'] for name, data in combination_analysis.items()
                       if name.startswith('working_')]
        failing_sims = [data['internal_similarity'] for name, data in combination_analysis.items()
                       if name.startswith('failing_')]

        ax4.boxplot([working_sims, failing_sims], labels=['Working', 'Failing'])
        ax4.set_title('Internal Similarity: Working vs Failing')
        ax4.set_ylabel('Average Internal Similarity')

        # Plot 5: Cluster purity comparison
        ax5 = axes[1, 1]
        working_purities = [data['cluster_purity'] for name, data in combination_analysis.items()
                           if name.startswith('working_')]
        failing_purities = [data['cluster_purity'] for name, data in combination_analysis.items()
                           if name.startswith('failing_')]

        ax5.boxplot([working_purities, failing_purities], labels=['Working', 'Failing'])
        ax5.set_title('Cluster Purity: Working vs Failing')
        ax5.set_ylabel('Cluster Purity Score')

        # Plot 6: Summary statistics
        ax6 = axes[1, 2]
        ax6.axis('off')

        summary_text = "FUNCTIONAL CLUSTERING ANALYSIS\n" + "="*35 + "\n\n"

        # Correlation results
        ari_score = correlation_results['ari_score']
        cluster_corr = correlation_results['cluster_correlation']
        cluster_p = correlation_results['cluster_p_value']

        summary_text += f"EVEN/ODD CORRELATION:\n"
        summary_text += f"  Adjusted Rand Index: {ari_score:.3f}\n"
        summary_text += f"  Spearman correlation: {cluster_corr:.3f}\n"
        summary_text += f"  P-value: {cluster_p:.3f}\n\n"

        # Pure clusters
        pure_even = len(correlation_results['pure_even_clusters'])
        pure_odd = len(correlation_results['pure_odd_clusters'])
        summary_text += f"PURE CLUSTERS:\n"
        summary_text += f"  Pure even clusters: {pure_even}\n"
        summary_text += f"  Pure odd clusters: {pure_odd}\n\n"

        # Combination analysis
        working_avg_sim = combination_analysis['summary']['working_avg_similarity']
        failing_avg_sim = combination_analysis['summary']['failing_avg_similarity']

        summary_text += f"COMBINATION ANALYSIS:\n"
        summary_text += f"  Working avg similarity: {working_avg_sim:.3f}\n"
        summary_text += f"  Failing avg similarity: {failing_avg_sim:.3f}\n"
        summary_text += f"  Difference: {working_avg_sim - failing_avg_sim:+.3f}\n\n"

        # Interpretation
        if ari_score > 0.3:
            interpretation = "Strong functional clustering by index"
        elif ari_score > 0.1:
            interpretation = "Moderate functional clustering by index"
        else:
            interpretation = "Weak functional clustering by index"

        summary_text += f"INTERPRETATION:\n"
        summary_text += f"  {interpretation}\n"

        if working_avg_sim > failing_avg_sim + 0.05:
            summary_text += f"  Working heads more functionally similar\n"
        else:
            summary_text += f"  No clear functional advantage\n"

        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        plt.tight_layout()
        plt.savefig('/home/paperspace/dev/MATS9/bandwidth/figures/functional_clustering_analysis.png',
                   dpi=300, bbox_inches='tight')
        plt.show()

    def run_complete_analysis(self) -> Dict:
        """Run the complete functional clustering analysis"""
        print("="*70)
        print("FUNCTIONAL CLUSTERING ANALYSIS - TESTING THE EVEN/ODD CRITIQUE")
        print("="*70)

        results = {}

        # Phase 1: Extract weight matrices
        print("\n" + "="*50)
        print("PHASE 1: WEIGHT MATRIX EXTRACTION")
        print("="*50)
        head_weights = self.extract_attention_weight_matrices()
        results['head_weights'] = head_weights

        # Phase 2: Compute functional similarity
        print("\n" + "="*50)
        print("PHASE 2: FUNCTIONAL SIMILARITY COMPUTATION")
        print("="*50)
        similarity_matrix, head_features = self.compute_head_functional_similarity(head_weights)
        results['similarity_matrix'] = similarity_matrix
        results['head_features'] = head_features

        # Phase 3: Hierarchical clustering
        print("\n" + "="*50)
        print("PHASE 3: HIERARCHICAL CLUSTERING")
        print("="*50)
        clustering_results = self.perform_hierarchical_clustering(similarity_matrix)
        results['clustering'] = clustering_results

        # Phase 4: Test even/odd correlation
        print("\n" + "="*50)
        print("PHASE 4: EVEN/ODD CORRELATION TESTING")
        print("="*50)
        correlation_results = self.test_even_odd_correlation(clustering_results['cluster_labels'])
        results['correlation'] = correlation_results

        # Phase 5: Analyze known combinations
        print("\n" + "="*50)
        print("PHASE 5: KNOWN COMBINATION ANALYSIS")
        print("="*50)
        combination_analysis = self.analyze_known_working_combinations(
            similarity_matrix, clustering_results['cluster_labels']
        )
        results['combination_analysis'] = combination_analysis

        # Phase 6: Function-based predictions
        print("\n" + "="*50)
        print("PHASE 6: FUNCTION-BASED PREDICTIONS")
        print("="*50)
        functional_predictions = self.predict_working_heads_by_function(
            similarity_matrix, clustering_results['cluster_labels']
        )
        results['functional_predictions'] = functional_predictions

        # Phase 7: Visualization
        print("\n" + "="*50)
        print("PHASE 7: VISUALIZATION")
        print("="*50)
        self.create_comprehensive_visualization(
            similarity_matrix, clustering_results, correlation_results, combination_analysis
        )

        return results


def main():
    """Run the functional clustering analysis"""
    print("Starting Functional Clustering Analysis...")
    print("This will test whether our even/odd findings represent true functional specialization")
    print("or coincidental correlation with head indices.")
    print()

    # Initialize analyzer
    analyzer = FunctionalClusteringAnalyzer()

    # Run complete analysis
    results = analyzer.run_complete_analysis()

    # Final summary and conclusion
    print("\n" + "="*70)
    print("FINAL ANALYSIS AND CONCLUSION")
    print("="*70)

    ari_score = results['correlation']['ari_score']
    cluster_correlation = results['correlation']['cluster_correlation']
    working_avg_sim = results['combination_analysis']['summary']['working_avg_similarity']
    failing_avg_sim = results['combination_analysis']['summary']['failing_avg_similarity']

    print(f"\nKEY METRICS:")
    print(f"  Adjusted Rand Index (functional clustering by index): {ari_score:.3f}")
    print(f"  Spearman correlation (clusters vs even/odd): {cluster_correlation:.3f}")
    print(f"  Working combinations functional similarity: {working_avg_sim:.3f}")
    print(f"  Failing combinations functional similarity: {failing_avg_sim:.3f}")

    print(f"\nEVIDENCE EVALUATION:")

    # Test 1: Do heads cluster functionally by index?
    if ari_score > 0.3:
        print(f"  ✅ STRONG: Heads cluster functionally by even/odd index (ARI={ari_score:.3f})")
        functional_clustering_verdict = "SUPPORTS OUR CLAIM"
    elif ari_score > 0.1:
        print(f"  ⚠️  MODERATE: Some functional clustering by index (ARI={ari_score:.3f})")
        functional_clustering_verdict = "MIXED EVIDENCE"
    else:
        print(f"  ❌ WEAK: Little functional clustering by index (ARI={ari_score:.3f})")
        functional_clustering_verdict = "SUPPORTS CRITIC"

    # Test 2: Are working combinations more functionally similar?
    similarity_difference = working_avg_sim - failing_avg_sim
    if similarity_difference > 0.05:
        print(f"  ✅ Working combinations are more functionally similar (+{similarity_difference:.3f})")
        similarity_verdict = "SUPPORTS FUNCTIONAL HYPOTHESIS"
    elif similarity_difference > -0.05:
        print(f"  ⚠️  No clear functional advantage ({similarity_difference:+.3f})")
        similarity_verdict = "INCONCLUSIVE"
    else:
        print(f"  ❌ Working combinations less functionally similar ({similarity_difference:+.3f})")
        similarity_verdict = "CONTRADICTS FUNCTIONAL HYPOTHESIS"

    # Test 3: Function-based predictions
    functional_predictions = results['functional_predictions']['potential_combinations']
    predicted_even_ratios = [combo['even_ratio'] for combo in functional_predictions]
    avg_predicted_even_ratio = np.mean(predicted_even_ratios)

    if avg_predicted_even_ratio > 0.75:
        print(f"  ✅ Function-based predictions favor even heads ({avg_predicted_even_ratio:.1%} even)")
        prediction_verdict = "SUPPORTS OUR CLAIM"
    elif avg_predicted_even_ratio > 0.6:
        print(f"  ⚠️  Function-based predictions somewhat favor even heads ({avg_predicted_even_ratio:.1%} even)")
        prediction_verdict = "MIXED EVIDENCE"
    else:
        print(f"  ❌ Function-based predictions don't favor even heads ({avg_predicted_even_ratio:.1%} even)")
        prediction_verdict = "SUPPORTS CRITIC"

    # Overall conclusion
    print(f"\n" + "="*50)
    print("OVERALL CONCLUSION")
    print("="*50)

    evidence_scores = {
        functional_clustering_verdict: {"SUPPORTS OUR CLAIM": 2, "MIXED EVIDENCE": 1, "SUPPORTS CRITIC": 0},
        similarity_verdict: {"SUPPORTS FUNCTIONAL HYPOTHESIS": 2, "INCONCLUSIVE": 1, "CONTRADICTS FUNCTIONAL HYPOTHESIS": 0},
        prediction_verdict: {"SUPPORTS OUR CLAIM": 2, "MIXED EVIDENCE": 1, "SUPPORTS CRITIC": 0}
    }

    total_score = sum(evidence_scores[verdict][verdict] for verdict in evidence_scores.keys())
    max_score = 6

    if total_score >= 5:
        final_conclusion = "STRONG EVIDENCE FOR FUNCTIONAL SPECIALIZATION BY INDEX"
        response_to_critic = "The critic's challenge is REFUTED"
    elif total_score >= 3:
        final_conclusion = "MIXED EVIDENCE - PARTIAL FUNCTIONAL SPECIALIZATION"
        response_to_critic = "The critic raises valid points but evidence is mixed"
    else:
        final_conclusion = "EVIDENCE SUPPORTS CRITIC'S FUNCTIONAL COINCIDENCE THEORY"
        response_to_critic = "The critic's challenge appears VALID"

    print(f"Evidence Score: {total_score}/{max_score}")
    print(f"Conclusion: {final_conclusion}")
    print(f"Response to Critic: {response_to_critic}")

    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'/home/paperspace/dev/MATS9/bandwidth/functional_clustering_analysis_{timestamp}.json'

    # Prepare JSON-serializable results
    json_results = {
        'analysis_summary': {
            'ari_score': float(ari_score),
            'cluster_correlation': float(cluster_correlation),
            'working_avg_similarity': float(working_avg_sim),
            'failing_avg_similarity': float(failing_avg_sim),
            'evidence_score': int(total_score),
            'max_evidence_score': int(max_score),
            'final_conclusion': final_conclusion,
            'response_to_critic': response_to_critic
        },
        'detailed_results': {
            # Convert numpy arrays to lists for JSON serialization
            'similarity_matrix': results['similarity_matrix'].tolist(),
            'cluster_labels': results['clustering']['cluster_labels'].tolist(),
            'correlation_results': {k: v for k, v in results['correlation'].items()
                                  if k not in ['cluster_composition']},  # Exclude complex nested dict
            'combination_analysis': results['combination_analysis'],
            'functional_predictions': results['functional_predictions']
        }
    }

    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)

    print(f"\n✅ Analysis complete!")
    print(f"✅ Results saved to {results_file}")
    print(f"✅ Visualization saved to /home/paperspace/dev/MATS9/bandwidth/figures/functional_clustering_analysis.png")

    return results


if __name__ == "__main__":
    main()