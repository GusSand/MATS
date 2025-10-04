#!/usr/bin/env python3
"""
Mechanistic Analysis: Understanding the causal role of weight clusters

This analyzes what makes the clusters different and tests if the differences
are causally responsible for the model's behavior.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch.nn.functional as F
from scipy import stats
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup paths
BASE_DIR = Path("/home/paperspace/dev/MATS9/pythia_clustering_dynamics")
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

class MechanisticAnalyzer:
    """Deep dive into what makes clusters different and why it matters"""

    def __init__(self, model_name="EleutherAI/pythia-160m"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Using device: {self.device}")

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

    def extract_all_weights(self, layer_idx=6):
        """Extract comprehensive weight information from layer"""
        layer = self.model.gpt_neox.layers[layer_idx]

        # Get QKV weights
        qkv_weight = layer.attention.query_key_value.weight.detach().cpu()

        # Get output projection
        out_proj = layer.attention.dense.weight.detach().cpu()

        # Get MLP weights
        mlp_in = layer.mlp.dense_h_to_4h.weight.detach().cpu()
        mlp_out = layer.mlp.dense_4h_to_h.weight.detach().cpu()

        # Split QKV
        hidden_size = self.model.config.hidden_size
        num_heads = self.model.config.num_attention_heads
        head_dim = hidden_size // num_heads

        qkv = qkv_weight.reshape(3, num_heads, head_dim, hidden_size)

        return {
            'query': qkv[0].numpy(),
            'key': qkv[1].numpy(),
            'value': qkv[2].numpy(),
            'out_proj': out_proj.numpy(),
            'mlp_in': mlp_in.numpy(),
            'mlp_out': mlp_out.numpy(),
            'layer_norm': layer.input_layernorm.weight.detach().cpu().numpy()
        }

    def cluster_and_analyze(self, weights, n_clusters=2):
        """Perform clustering and extract cluster characteristics"""
        num_heads = weights.shape[0]
        head_weights = weights.reshape(num_heads, -1)

        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(head_weights)

        # Analyze each cluster
        clusters = {}
        for i in range(n_clusters):
            cluster_heads = np.where(labels == i)[0]
            cluster_weights = head_weights[labels == i]

            clusters[f'cluster_{i}'] = {
                'heads': cluster_heads.tolist(),
                'size': len(cluster_heads),
                'mean': np.mean(cluster_weights, axis=0),
                'std': np.std(cluster_weights, axis=0),
                'norm': np.mean([np.linalg.norm(w) for w in cluster_weights]),
                'center': kmeans.cluster_centers_[i]
            }

        # Compare clusters
        if n_clusters == 2:
            diff = kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1]
            clusters['difference'] = {
                'vector': diff,
                'magnitude': np.linalg.norm(diff),
                'top_dims': np.argsort(np.abs(diff))[-10:].tolist(),
                'correlation': np.corrcoef(kmeans.cluster_centers_[0], kmeans.cluster_centers_[1])[0, 1]
            }

        return labels, clusters

    def test_causal_intervention(self, test_prompt="Q: Which is bigger: 9.8 or 9.11?\nA:"):
        """Test if swapping cluster assignments changes behavior"""
        print("\n🧪 Testing Causal Intervention...")

        # Get baseline output
        inputs = self.tokenizer(test_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            baseline_output = self.model(**inputs, output_hidden_states=True)
            baseline_logits = baseline_output.logits[0, -1]

        # Extract weights and cluster
        layer_idx = 6
        weights = self.extract_all_weights(layer_idx)
        q_labels, q_clusters = self.cluster_and_analyze(weights['query'])

        print(f"  Cluster 0 heads: {q_clusters['cluster_0']['heads']}")
        print(f"  Cluster 1 heads: {q_clusters['cluster_1']['heads']}")

        # Create intervention: swap cluster assignments
        layer = self.model.gpt_neox.layers[layer_idx]
        original_qkv = layer.attention.query_key_value.weight.clone()

        # Prepare swapped weights
        hidden_size = self.model.config.hidden_size
        num_heads = self.model.config.num_attention_heads
        head_dim = hidden_size // num_heads

        qkv_weight = original_qkv.reshape(3, num_heads, head_dim, hidden_size)
        q_weights = qkv_weight[0].clone()

        # Swap cluster 0 and cluster 1 average patterns
        if len(q_clusters['cluster_0']['heads']) > 0 and len(q_clusters['cluster_1']['heads']) > 0:
            # Calculate average patterns
            cluster0_avg = q_weights[q_clusters['cluster_0']['heads']].mean(0)
            cluster1_avg = q_weights[q_clusters['cluster_1']['heads']].mean(0)

            # Apply swap
            for head in q_clusters['cluster_0']['heads']:
                q_weights[head] = cluster1_avg
            for head in q_clusters['cluster_1']['heads']:
                q_weights[head] = cluster0_avg

            # Reconstruct QKV weight
            qkv_weight[0] = q_weights
            new_qkv = qkv_weight.reshape(3 * num_heads * head_dim, hidden_size)
            layer.attention.query_key_value.weight.data = new_qkv

            # Get intervention output
            with torch.no_grad():
                intervention_output = self.model(**inputs)
                intervention_logits = intervention_output.logits[0, -1]

            # Restore original weights
            layer.attention.query_key_value.weight.data = original_qkv

            # Compare outputs
            top_baseline = torch.topk(baseline_logits, 5)
            top_intervention = torch.topk(intervention_logits, 5)

            baseline_tokens = [self.tokenizer.decode(t) for t in top_baseline.indices]
            intervention_tokens = [self.tokenizer.decode(t) for t in top_intervention.indices]

            print("\n  Baseline top tokens:", baseline_tokens)
            print("  Intervention top tokens:", intervention_tokens)

            # Calculate KL divergence
            kl_div = F.kl_div(
                F.log_softmax(intervention_logits, dim=-1),
                F.softmax(baseline_logits, dim=-1),
                reduction='sum'
            ).item()

            print(f"  KL divergence: {kl_div:.4f}")

            return {
                'baseline_tokens': baseline_tokens,
                'intervention_tokens': intervention_tokens,
                'kl_divergence': kl_div,
                'changed': baseline_tokens[0] != intervention_tokens[0]
            }

        return None

    def analyze_cluster_functions(self):
        """Analyze what each cluster specializes in"""
        print("\n🔬 Analyzing Cluster Specialization...")

        test_prompts = [
            # Numerical comparisons
            "Q: Which is bigger: 9.8 or 9.11?\nA:",
            "Q: Which is bigger: 5.7 or 5.12?\nA:",
            # Basic math
            "Q: What is 2 + 2?\nA:",
            "Q: What is 10 - 3?\nA:",
            # Language tasks
            "The capital of France is",
            "The opposite of hot is",
            # Pattern completion
            "1, 2, 3, 4,",
            "A, B, C, D,"
        ]

        layer_idx = 6
        weights = self.extract_all_weights(layer_idx)
        q_labels, q_clusters = self.cluster_and_analyze(weights['query'])

        results = {}

        for prompt in test_prompts:
            print(f"\n  Testing: {prompt[:30]}...")

            # Test with only cluster 0 active
            cluster0_result = self.test_selective_activation(
                prompt, layer_idx, active_heads=q_clusters['cluster_0']['heads']
            )

            # Test with only cluster 1 active
            cluster1_result = self.test_selective_activation(
                prompt, layer_idx, active_heads=q_clusters['cluster_1']['heads']
            )

            results[prompt] = {
                'cluster_0': cluster0_result,
                'cluster_1': cluster1_result,
                'difference': cluster0_result['confidence'] - cluster1_result['confidence']
            }

            print(f"    Cluster 0: {cluster0_result['token']} ({cluster0_result['confidence']:.2%})")
            print(f"    Cluster 1: {cluster1_result['token']} ({cluster1_result['confidence']:.2%})")

        return results

    def test_selective_activation(self, prompt, layer_idx, active_heads):
        """Test model with only specific heads active"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Create hook to zero out inactive heads
        def hook_fn(module, input, output):
            # output shape: [batch, seq_len, hidden_size]
            batch_size, seq_len, _ = output.shape

            hidden_size = self.model.config.hidden_size
            num_heads = self.model.config.num_attention_heads
            head_dim = hidden_size // num_heads

            # Reshape to separate heads
            output_reshaped = output.view(batch_size, seq_len, num_heads, head_dim)

            # Zero out inactive heads
            for h in range(num_heads):
                if h not in active_heads:
                    output_reshaped[:, :, h, :] = 0

            return output_reshaped.view(batch_size, seq_len, hidden_size)

        # Register hook
        handle = self.model.gpt_neox.layers[layer_idx].attention.dense.register_forward_hook(hook_fn)

        # Get output
        with torch.no_grad():
            output = self.model(**inputs)
            logits = output.logits[0, -1]
            probs = F.softmax(logits, dim=-1)
            top_token_id = torch.argmax(probs).item()
            top_token = self.tokenizer.decode(top_token_id)
            confidence = probs[top_token_id].item()

        # Remove hook
        handle.remove()

        return {'token': top_token, 'confidence': confidence}

    def visualize_clusters(self, weights, labels, method='pca'):
        """Visualize cluster separation in 2D"""
        print(f"\n📊 Visualizing clusters using {method.upper()}...")

        num_heads = weights.shape[0]
        head_weights = weights.reshape(num_heads, -1)

        # Reduce dimensionality
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            coords = reducer.fit_transform(head_weights)
            explained_var = reducer.explained_variance_ratio_
            print(f"  Explained variance: {explained_var[0]:.2%} + {explained_var[1]:.2%}")
        else:  # tsne
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(5, num_heads-1))
            coords = reducer.fit_transform(head_weights)

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        colors = ['blue', 'red', 'green', 'orange']
        for i in np.unique(labels):
            mask = labels == i
            ax.scatter(coords[mask, 0], coords[mask, 1],
                      c=colors[i], s=100, alpha=0.7,
                      label=f'Cluster {i}')

            # Add head numbers
            for j, (x, y) in enumerate(coords[mask]):
                head_idx = np.where(mask)[0][j]
                ax.annotate(str(head_idx), (x, y),
                           fontsize=8, ha='center', va='center')

        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.set_title(f'Head Weight Clusters ({method.upper()} Projection)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = FIGURES_DIR / f"cluster_visualization_{method}_{timestamp}.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"  Saved to {filepath}")

        return fig

    def compare_weight_statistics(self, weights, labels):
        """Statistical comparison of cluster properties"""
        print("\n📈 Statistical Analysis of Clusters...")

        num_heads = weights.shape[0]
        head_weights = weights.reshape(num_heads, -1)

        stats_comparison = {}

        for i in np.unique(labels):
            cluster_weights = head_weights[labels == i]

            stats_comparison[f'cluster_{i}'] = {
                'mean_weight': float(np.mean(cluster_weights)),
                'std_weight': float(np.std(cluster_weights)),
                'min_weight': float(np.min(cluster_weights)),
                'max_weight': float(np.max(cluster_weights)),
                'sparsity': float(np.mean(np.abs(cluster_weights) < 0.01)),
                'mean_norm': float(np.mean([np.linalg.norm(w) for w in cluster_weights])),
                'weight_range': float(np.max(cluster_weights) - np.min(cluster_weights))
            }

        # Compare distributions
        if len(np.unique(labels)) == 2:
            cluster0_weights = head_weights[labels == 0].flatten()
            cluster1_weights = head_weights[labels == 1].flatten()

            # KS test for distribution difference
            ks_stat, ks_pvalue = stats.ks_2samp(cluster0_weights, cluster1_weights)

            # T-test for mean difference
            t_stat, t_pvalue = stats.ttest_ind(cluster0_weights, cluster1_weights)

            stats_comparison['statistical_tests'] = {
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(ks_pvalue),
                't_statistic': float(t_stat),
                't_pvalue': float(t_pvalue),
                'significantly_different': ks_pvalue < 0.05
            }

            print(f"  KS test p-value: {ks_pvalue:.6f}")
            print(f"  T-test p-value: {t_pvalue:.6f}")
            print(f"  Distributions are {'significantly' if ks_pvalue < 0.05 else 'not significantly'} different")

        return stats_comparison

    def run_complete_analysis(self):
        """Run all mechanistic analyses"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name
        }

        # Extract and analyze weights
        print("\n🔍 Extracting weight matrices...")
        weights = self.extract_all_weights(layer_idx=6)

        # Analyze each weight type
        for weight_type in ['query', 'key', 'value']:
            print(f"\n{'='*50}")
            print(f"Analyzing {weight_type.upper()} weights")
            print('='*50)

            w = weights[weight_type]
            labels, clusters = self.cluster_and_analyze(w)

            results[weight_type] = {
                'clusters': {k: v for k, v in clusters.items()
                           if k != 'difference' or 'vector' not in v},
                'labels': labels.tolist()
            }

            # Visualize
            self.visualize_clusters(w, labels, method='pca')
            self.visualize_clusters(w, labels, method='tsne')

            # Statistical analysis
            stats = self.compare_weight_statistics(w, labels)
            results[weight_type]['statistics'] = stats

        # Test causal intervention
        intervention_result = self.test_causal_intervention()
        results['causal_intervention'] = intervention_result

        # Analyze cluster functions
        functional_analysis = self.analyze_cluster_functions()
        results['functional_analysis'] = {
            k: {
                'cluster_0_token': v['cluster_0']['token'],
                'cluster_1_token': v['cluster_1']['token'],
                'difference': v['difference']
            }
            for k, v in functional_analysis.items()
        }

        return results

    def save_results(self, results):
        """Save analysis results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = RESULTS_DIR / f"mechanistic_analysis_{timestamp}.json"

        # Convert numpy arrays to lists for JSON
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            return obj

        serializable_results = convert_to_serializable(results)

        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\n💾 Saved results to {filepath}")
        return filepath


def main():
    """Run mechanistic analysis"""
    print("🚀 Starting Mechanistic Analysis of Weight Clusters")
    print("=" * 50)

    analyzer = MechanisticAnalyzer()
    results = analyzer.run_complete_analysis()
    analyzer.save_results(results)

    # Print summary
    print("\n" + "=" * 50)
    print("📊 MECHANISTIC ANALYSIS SUMMARY")
    print("=" * 50)

    print("\n🎯 Key Findings:")

    # Cluster separation
    for weight_type in ['query', 'key', 'value']:
        if weight_type in results and 'statistics' in results[weight_type]:
            stats_test = results[weight_type]['statistics'].get('statistical_tests', {})
            if stats_test.get('significantly_different'):
                print(f"  ✓ {weight_type.upper()}: Clusters are statistically distinct (p={stats_test['ks_pvalue']:.6f})")
            else:
                print(f"  ✗ {weight_type.upper()}: Clusters are not significantly different")

    # Causal intervention
    if results.get('causal_intervention'):
        if results['causal_intervention']['changed']:
            print(f"  ✓ Swapping clusters CHANGES output (KL={results['causal_intervention']['kl_divergence']:.4f})")
        else:
            print(f"  ✗ Swapping clusters does not change output significantly")

    # Functional specialization
    if 'functional_analysis' in results:
        different_outputs = sum(
            1 for v in results['functional_analysis'].values()
            if v['cluster_0_token'] != v['cluster_1_token']
        )
        total = len(results['functional_analysis'])
        print(f"  • Functional difference: {different_outputs}/{total} prompts show different cluster outputs")

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()