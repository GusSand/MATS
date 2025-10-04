#!/usr/bin/env python3
"""
Verification Script: Cross-check clustering results against known behavioral patterns

This script addresses skepticism about the clustering results by:
1. Testing different clustering methods and parameters
2. Verifying against actual behavioral tests (9.8 vs 9.11)
3. Checking consistency with even/odd head specialization findings
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import torch.nn.functional as F
from scipy.cluster.hierarchy import dendrogram, linkage
from datetime import datetime
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Setup paths
BASE_DIR = Path("/home/paperspace/dev/MATS9/pythia_clustering_dynamics")
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

class ClusteringVerifier:
    """Verify clustering results against behavioral tests and different methods"""

    def __init__(self, model_name="EleutherAI/pythia-160m"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

        print(f"🔧 Loaded {model_name} on {self.device}")

    def extract_qkv_weights(self, layer_idx=6):
        """Extract Q, K, V weight matrices from specified layer"""
        layer = self.model.gpt_neox.layers[layer_idx]
        qkv_weight = layer.attention.query_key_value.weight.detach().cpu()

        # Split QKV
        hidden_size = self.model.config.hidden_size
        num_heads = self.model.config.num_attention_heads
        head_dim = hidden_size // num_heads

        qkv = qkv_weight.reshape(3, num_heads, head_dim, hidden_size)

        return {
            'query': qkv[0].numpy(),
            'key': qkv[1].numpy(),
            'value': qkv[2].numpy()
        }

    def test_multiple_clustering_methods(self, weights):
        """Test different clustering algorithms and parameters"""
        print(f"\n🔍 Testing Multiple Clustering Methods...")

        num_heads = weights.shape[0]
        head_weights = weights.reshape(num_heads, -1)

        results = {}

        # 1. K-Means with different k
        print("  📊 K-Means clustering:")
        for k in [2, 3, 4, 5]:
            if k <= num_heads:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(head_weights)
                silhouette = silhouette_score(head_weights, labels) if k > 1 else 0

                # Group heads by cluster
                clusters = {}
                for i in range(k):
                    cluster_heads = np.where(labels == i)[0].tolist()
                    clusters[f'cluster_{i}'] = cluster_heads

                results[f'kmeans_k{k}'] = {
                    'labels': labels.tolist(),
                    'silhouette': silhouette,
                    'clusters': clusters,
                    'method': f'K-Means (k={k})'
                }

                print(f"    k={k}: silhouette={silhouette:.3f}, clusters={list(clusters.values())}")

        # 2. Hierarchical clustering
        print("  🌳 Hierarchical clustering:")
        linkage_matrix = linkage(head_weights, method='ward')

        for k in [2, 3, 4]:
            if k <= num_heads:
                hierarchical = AgglomerativeClustering(n_clusters=k, linkage='ward')
                labels = hierarchical.fit_predict(head_weights)
                silhouette = silhouette_score(head_weights, labels) if k > 1 else 0

                clusters = {}
                for i in range(k):
                    cluster_heads = np.where(labels == i)[0].tolist()
                    clusters[f'cluster_{i}'] = cluster_heads

                results[f'hierarchical_k{k}'] = {
                    'labels': labels.tolist(),
                    'silhouette': silhouette,
                    'clusters': clusters,
                    'method': f'Hierarchical (k={k})'
                }

                print(f"    k={k}: silhouette={silhouette:.3f}, clusters={list(clusters.values())}")

        # 3. DBSCAN (density-based, can find outliers)
        print("  🔍 DBSCAN clustering:")
        for eps in [0.5, 1.0, 1.5, 2.0]:
            dbscan = DBSCAN(eps=eps, min_samples=2)
            labels = dbscan.fit_predict(head_weights)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            if n_clusters > 1:
                silhouette = silhouette_score(head_weights, labels)
            else:
                silhouette = -1

            clusters = {}
            for i in set(labels):
                if i != -1:  # Not noise
                    cluster_heads = np.where(labels == i)[0].tolist()
                    clusters[f'cluster_{i}'] = cluster_heads

            if n_noise > 0:
                outliers = np.where(labels == -1)[0].tolist()
                clusters['outliers'] = outliers

            results[f'dbscan_eps{eps}'] = {
                'labels': labels.tolist(),
                'silhouette': silhouette,
                'clusters': clusters,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'method': f'DBSCAN (eps={eps})'
            }

            print(f"    eps={eps}: {n_clusters} clusters, {n_noise} outliers, "
                  f"silhouette={silhouette:.3f}, clusters={list(clusters.values())}")

        return results

    def test_even_odd_hypothesis(self, weights):
        """Test if even/odd head grouping makes sense"""
        print(f"\n🎯 Testing Even/Odd Hypothesis...")

        num_heads = weights.shape[0]  # Should be 12
        head_weights = weights.reshape(num_heads, -1)

        # Define even/odd groups
        even_heads = [0, 2, 4, 6, 8, 10]
        odd_heads = [1, 3, 5, 7, 9, 11]

        # Create even/odd labels
        even_odd_labels = np.zeros(num_heads)
        for h in odd_heads:
            even_odd_labels[h] = 1

        # Calculate metrics for even/odd grouping
        silhouette = silhouette_score(head_weights, even_odd_labels)

        # Calculate within and between group similarities
        even_weights = head_weights[even_heads]
        odd_weights = head_weights[odd_heads]

        # Average within-group distance
        even_within_dist = np.mean([
            np.linalg.norm(even_weights[i] - even_weights[j])
            for i in range(len(even_weights))
            for j in range(i+1, len(even_weights))
        ])

        odd_within_dist = np.mean([
            np.linalg.norm(odd_weights[i] - odd_weights[j])
            for i in range(len(odd_weights))
            for j in range(i+1, len(odd_weights))
        ])

        # Average between-group distance
        between_dist = np.mean([
            np.linalg.norm(even_weights[i] - odd_weights[j])
            for i in range(len(even_weights))
            for j in range(len(odd_weights))
        ])

        even_odd_result = {
            'silhouette': silhouette,
            'even_heads': even_heads,
            'odd_heads': odd_heads,
            'even_within_distance': even_within_dist,
            'odd_within_distance': odd_within_dist,
            'between_distance': between_dist,
            'separation_ratio': between_dist / np.mean([even_within_dist, odd_within_dist])
        }

        print(f"  Even/Odd silhouette: {silhouette:.3f}")
        print(f"  Even heads: {even_heads}")
        print(f"  Odd heads: {odd_heads}")
        print(f"  Separation ratio: {even_odd_result['separation_ratio']:.3f}")

        return even_odd_result

    def test_behavioral_specialization(self):
        """Test actual behavioral differences on the 9.8 vs 9.11 task"""
        print(f"\n🧪 Testing Behavioral Specialization...")

        prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Get baseline
        with torch.no_grad():
            baseline_output = self.model(**inputs)
            baseline_logits = baseline_output.logits[0, -1]
            baseline_probs = F.softmax(baseline_logits, dim=-1)

        # Test individual heads
        print(f"  🔬 Testing individual head contributions...")

        layer_idx = 6
        head_results = {}

        for head_idx in range(12):
            result = self.test_single_head_active(prompt, layer_idx, head_idx)
            head_results[head_idx] = result
            print(f"    Head {head_idx}: {result['token']} ({result['confidence']:.2%})")

        # Test even vs odd groups
        print(f"  ⚖️  Testing even vs odd groups...")

        even_result = self.test_head_group_active(prompt, layer_idx, [0,2,4,6,8,10])
        odd_result = self.test_head_group_active(prompt, layer_idx, [1,3,5,7,9,11])

        print(f"    Even heads [0,2,4,6,8,10]: {even_result['token']} ({even_result['confidence']:.2%})")
        print(f"    Odd heads [1,3,5,7,9,11]: {odd_result['token']} ({odd_result['confidence']:.2%})")

        # Test our discovered clusters (Head 6 vs Others)
        print(f"  🎯 Testing discovered clusters...")

        head6_result = self.test_head_group_active(prompt, layer_idx, [6])
        others_result = self.test_head_group_active(prompt, layer_idx, [0,1,2,3,4,5,7,8,9,10,11])

        print(f"    Head 6 only: {head6_result['token']} ({head6_result['confidence']:.2%})")
        print(f"    Other heads: {others_result['token']} ({others_result['confidence']:.2%})")

        return {
            'baseline': {
                'top_tokens': [self.tokenizer.decode(t) for t in torch.topk(baseline_probs, 5).indices],
                'top_probs': torch.topk(baseline_probs, 5).values.tolist()
            },
            'individual_heads': head_results,
            'even_group': even_result,
            'odd_group': odd_result,
            'head6_only': head6_result,
            'others_group': others_result
        }

    def test_single_head_active(self, prompt, layer_idx, active_head):
        """Test model with only one head active"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        def hook_fn(module, input, output):
            batch_size, seq_len, _ = output.shape
            hidden_size = self.model.config.hidden_size
            num_heads = self.model.config.num_attention_heads
            head_dim = hidden_size // num_heads

            output_reshaped = output.view(batch_size, seq_len, num_heads, head_dim)

            # Zero out all heads except the active one
            for h in range(num_heads):
                if h != active_head:
                    output_reshaped[:, :, h, :] = 0

            return output_reshaped.view(batch_size, seq_len, hidden_size)

        handle = self.model.gpt_neox.layers[layer_idx].attention.dense.register_forward_hook(hook_fn)

        with torch.no_grad():
            output = self.model(**inputs)
            logits = output.logits[0, -1]
            probs = F.softmax(logits, dim=-1)
            top_token_id = torch.argmax(probs).item()
            top_token = self.tokenizer.decode(top_token_id)
            confidence = probs[top_token_id].item()

        handle.remove()

        return {'token': top_token, 'confidence': confidence}

    def test_head_group_active(self, prompt, layer_idx, active_heads):
        """Test model with only specified heads active"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        def hook_fn(module, input, output):
            batch_size, seq_len, _ = output.shape
            hidden_size = self.model.config.hidden_size
            num_heads = self.model.config.num_attention_heads
            head_dim = hidden_size // num_heads

            output_reshaped = output.view(batch_size, seq_len, num_heads, head_dim)

            # Zero out inactive heads
            for h in range(num_heads):
                if h not in active_heads:
                    output_reshaped[:, :, h, :] = 0

            return output_reshaped.view(batch_size, seq_len, hidden_size)

        handle = self.model.gpt_neox.layers[layer_idx].attention.dense.register_forward_hook(hook_fn)

        with torch.no_grad():
            output = self.model(**inputs)
            logits = output.logits[0, -1]
            probs = F.softmax(logits, dim=-1)
            top_token_id = torch.argmax(probs).item()
            top_token = self.tokenizer.decode(top_token_id)
            confidence = probs[top_token_id].item()

        handle.remove()

        return {'token': top_token, 'confidence': confidence}

    def visualize_clustering_comparison(self, weights, clustering_results):
        """Compare different clustering methods visually"""
        print(f"\n📊 Visualizing Clustering Comparisons...")

        num_heads = weights.shape[0]
        head_weights = weights.reshape(num_heads, -1)

        # Reduce to 2D for visualization
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(head_weights)

        # Select interesting methods to compare
        methods_to_plot = [
            'kmeans_k2',
            'hierarchical_k2',
            'dbscan_eps1.0'
        ]

        # Add even/odd if we have that result
        if hasattr(self, 'even_odd_result'):
            even_heads = [0, 2, 4, 6, 8, 10]
            odd_heads = [1, 3, 5, 7, 9, 11]
            even_odd_labels = np.zeros(num_heads)
            for h in odd_heads:
                even_odd_labels[h] = 1

            clustering_results['even_odd'] = {
                'labels': even_odd_labels.tolist(),
                'clusters': {'even': even_heads, 'odd': odd_heads},
                'method': 'Even/Odd Hypothesis'
            }
            methods_to_plot.append('even_odd')

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

        for idx, method in enumerate(methods_to_plot[:4]):
            ax = axes[idx]

            if method in clustering_results:
                labels = np.array(clustering_results[method]['labels'])
                method_name = clustering_results[method]['method']

                # Plot points
                for cluster_id in np.unique(labels):
                    if cluster_id != -1:  # Skip noise points in DBSCAN
                        mask = labels == cluster_id
                        ax.scatter(coords[mask, 0], coords[mask, 1],
                                 c=colors[cluster_id % len(colors)],
                                 s=100, alpha=0.7,
                                 label=f'Cluster {cluster_id}')

                        # Add head numbers
                        for j, (x, y) in enumerate(coords[mask]):
                            head_idx = np.where(mask)[0][j]
                            ax.annotate(str(head_idx), (x, y),
                                       fontsize=8, ha='center', va='center')

                # Plot noise points if any
                if -1 in labels:
                    noise_mask = labels == -1
                    ax.scatter(coords[noise_mask, 0], coords[noise_mask, 1],
                             c='black', s=100, alpha=0.3, marker='x',
                             label='Noise')

                ax.set_title(method_name)
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.suptitle('Clustering Method Comparison', fontsize=16)
        plt.tight_layout()

        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = FIGURES_DIR / f"clustering_comparison_{timestamp}.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"  Saved comparison to {filepath}")

        return fig

    def run_verification(self):
        """Run complete verification analysis"""
        print("🔍 CLUSTERING VERIFICATION ANALYSIS")
        print("="*50)

        # Extract weights
        weights = self.extract_qkv_weights(layer_idx=6)

        results = {}

        # Test each weight type
        for weight_type in ['query', 'key', 'value']:
            print(f"\n{'='*20} {weight_type.upper()} WEIGHTS {'='*20}")

            w = weights[weight_type]

            # Test multiple clustering methods
            clustering_results = self.test_multiple_clustering_methods(w)

            # Test even/odd hypothesis
            even_odd_result = self.test_even_odd_hypothesis(w)

            # Visualize comparison
            self.visualize_clustering_comparison(w, clustering_results)

            results[weight_type] = {
                'clustering_methods': clustering_results,
                'even_odd_analysis': even_odd_result
            }

        # Test behavioral specialization
        behavioral_results = self.test_behavioral_specialization()
        results['behavioral_tests'] = behavioral_results

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            else:
                return obj

        filepath = RESULTS_DIR / f"clustering_verification_{timestamp}.json"
        with open(filepath, 'w') as f:
            json.dump(convert_numpy(results), f, indent=2)

        print(f"\n💾 Saved verification results to {filepath}")

        return results


def main():
    """Run clustering verification"""
    verifier = ClusteringVerifier()
    results = verifier.run_verification()

    print("\n" + "="*50)
    print("🎯 VERIFICATION SUMMARY")
    print("="*50)

    # Compare clustering methods
    print("\n📊 Best Clustering Methods by Silhouette Score:")
    for weight_type in ['query', 'key', 'value']:
        print(f"\n  {weight_type.upper()}:")
        clustering_methods = results[weight_type]['clustering_methods']

        # Sort by silhouette score
        sorted_methods = sorted(
            [(k, v) for k, v in clustering_methods.items()],
            key=lambda x: x[1]['silhouette'],
            reverse=True
        )

        for method_name, method_data in sorted_methods[:3]:
            print(f"    {method_data['method']}: {method_data['silhouette']:.3f}")
            print(f"      Clusters: {list(method_data['clusters'].values())}")

    # Even/odd analysis
    print(f"\n⚖️  Even/Odd Hypothesis Analysis:")
    for weight_type in ['query', 'key', 'value']:
        even_odd = results[weight_type]['even_odd_analysis']
        print(f"  {weight_type.upper()}: silhouette={even_odd['silhouette']:.3f}, "
              f"separation={even_odd['separation_ratio']:.3f}")

    # Behavioral results
    print(f"\n🧪 Behavioral Test Results (9.8 vs 9.11):")
    behavioral = results['behavioral_tests']
    print(f"  Baseline: {behavioral['baseline']['top_tokens'][0]}")
    print(f"  Even heads: {behavioral['even_group']['token']}")
    print(f"  Odd heads: {behavioral['odd_group']['token']}")
    print(f"  Head 6 only: {behavioral['head6_only']['token']}")
    print(f"  Other heads: {behavioral['others_group']['token']}")

    print("\n✅ Verification complete!")


if __name__ == "__main__":
    main()