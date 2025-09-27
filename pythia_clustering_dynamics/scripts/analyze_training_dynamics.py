#!/usr/bin/env python3
"""
Training Dynamics Analysis: When do weight clusters emerge in Pythia?

This leverages Pythia's unique advantage - comprehensive training checkpoints.
We'll track the emergence and evolution of weight clustering throughout training.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup paths
BASE_DIR = Path("/home/paperspace/dev/MATS9/pythia_clustering_dynamics")
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
DATA_DIR = BASE_DIR / "data"

for dir in [RESULTS_DIR, FIGURES_DIR, DATA_DIR]:
    dir.mkdir(exist_ok=True)

class TrainingDynamicsAnalyzer:
    """Analyze weight clustering emergence across training checkpoints"""

    def __init__(self, model_base="EleutherAI/pythia-160m"):
        self.model_base = model_base
        self.checkpoints = [
            "step1000",
            "step5000",
            "step10000",
            "step20000",
            "step40000",
            "step80000",
            "step120000",
            "step143000"  # Final checkpoint
        ]
        self.results = {}

    def extract_qkv_weights(self, model, layer_idx=6):
        """Extract Q, K, V weight matrices from specified layer"""
        layer = model.gpt_neox.layers[layer_idx]

        # Get attention weights
        qkv_weight = layer.attention.query_key_value.weight.detach().cpu()

        # GPT-NeoX combines QKV, need to split
        hidden_size = model.config.hidden_size
        num_heads = model.config.num_attention_heads
        head_dim = hidden_size // num_heads

        # Reshape to separate Q, K, V
        qkv = qkv_weight.reshape(3, num_heads, head_dim, hidden_size)

        return {
            'query': qkv[0],  # [num_heads, head_dim, hidden_size]
            'key': qkv[1],
            'value': qkv[2],
            'combined': qkv_weight.numpy()
        }

    def analyze_clustering(self, weights, n_clusters=2):
        """Analyze if weights form distinct clusters"""
        # Flatten weights for each head
        num_heads = weights.shape[0]
        head_weights = weights.reshape(num_heads, -1)

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(head_weights)

        # Calculate metrics
        silhouette = silhouette_score(head_weights, labels)

        # Calculate cluster separation
        cluster_centers = kmeans.cluster_centers_
        inter_cluster_dist = cdist(cluster_centers, cluster_centers).max()

        # Calculate within-cluster variance
        within_cluster_vars = []
        for i in range(n_clusters):
            cluster_points = head_weights[labels == i]
            if len(cluster_points) > 0:
                var = np.var(cluster_points, axis=0).mean()
                within_cluster_vars.append(var)

        # Distribution overlap (Wasserstein distance between clusters)
        if n_clusters == 2:
            cluster_0 = head_weights[labels == 0].flatten()
            cluster_1 = head_weights[labels == 1].flatten()
            # Sample for efficiency
            sample_size = min(1000, len(cluster_0), len(cluster_1))
            c0_sample = np.random.choice(cluster_0, sample_size)
            c1_sample = np.random.choice(cluster_1, sample_size)
            overlap = wasserstein_distance(c0_sample, c1_sample)
        else:
            overlap = None

        return {
            'labels': labels,
            'silhouette_score': silhouette,
            'inter_cluster_distance': inter_cluster_dist,
            'within_cluster_variance': np.mean(within_cluster_vars),
            'cluster_separation_ratio': inter_cluster_dist / np.mean(within_cluster_vars) if within_cluster_vars else 0,
            'wasserstein_distance': overlap,
            'cluster_centers': cluster_centers,
            'cluster_sizes': [sum(labels == i) for i in range(n_clusters)]
        }

    def compare_checkpoints(self, early_weights, late_weights):
        """Compare weight distributions between checkpoints"""
        # Calculate weight divergence
        weight_diff = np.abs(late_weights - early_weights)
        mean_divergence = np.mean(weight_diff)
        max_divergence = np.max(weight_diff)

        # Per-head divergence
        num_heads = early_weights.shape[0]
        per_head_divergence = []
        for h in range(num_heads):
            head_diff = np.mean(np.abs(late_weights[h] - early_weights[h]))
            per_head_divergence.append(head_diff)

        return {
            'mean_divergence': mean_divergence,
            'max_divergence': max_divergence,
            'per_head_divergence': per_head_divergence
        }

    def analyze_checkpoint(self, checkpoint_name):
        """Analyze clustering in a single checkpoint"""
        print(f"\n📊 Analyzing {checkpoint_name}...")

        # Load model using revision parameter
        model = AutoModelForCausalLM.from_pretrained(
            self.model_base,
            revision=checkpoint_name,
            torch_dtype=torch.float32
        )
        model.eval()

        # Extract weights
        weights = self.extract_qkv_weights(model)

        # Analyze each weight type
        results = {}
        for weight_type in ['query', 'key', 'value']:
            w = weights[weight_type].numpy()
            clustering = self.analyze_clustering(w)
            results[weight_type] = clustering

            print(f"  {weight_type.upper()}: silhouette={clustering['silhouette_score']:.3f}, "
                  f"separation={clustering['cluster_separation_ratio']:.3f}")

        # Store raw weights for comparison
        results['raw_weights'] = {k: v.tolist() if isinstance(v, np.ndarray) else v
                                  for k, v in weights.items()}

        return results

    def run_full_analysis(self):
        """Run analysis across all checkpoints"""
        print("🚀 Starting Training Dynamics Analysis")
        print("=" * 50)

        all_results = {}
        prev_weights = None

        for ckpt in self.checkpoints:
            step_results = self.analyze_checkpoint(ckpt)

            # Compare to previous checkpoint
            if prev_weights is not None:
                for weight_type in ['query', 'key', 'value']:
                    curr = np.array(step_results['raw_weights'][weight_type])
                    prev = np.array(prev_weights[weight_type])
                    divergence = self.compare_checkpoints(prev, curr)
                    step_results[f'{weight_type}_divergence'] = divergence

            all_results[ckpt] = step_results
            prev_weights = step_results['raw_weights']

        self.results = all_results
        return all_results

    def plot_emergence_timeline(self):
        """Plot how clustering metrics evolve over training"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Extract step numbers
        steps = []
        for ckpt in self.checkpoints:
            if ckpt == "step143000":
                steps.append(143000)
            else:
                step = int(ckpt.replace("step", ""))
                steps.append(step)

        # Metrics to plot
        metrics = {
            'Silhouette Score': 'silhouette_score',
            'Cluster Separation Ratio': 'cluster_separation_ratio',
            'Inter-cluster Distance': 'inter_cluster_distance',
            'Within-cluster Variance': 'within_cluster_variance',
            'Wasserstein Distance': 'wasserstein_distance'
        }

        weight_types = ['query', 'key', 'value']
        colors = {'query': 'blue', 'key': 'green', 'value': 'red'}

        for idx, (metric_name, metric_key) in enumerate(metrics.items()):
            ax = axes[idx // 3, idx % 3]

            for wt in weight_types:
                values = []
                for ckpt in self.checkpoints:
                    if ckpt in self.results and wt in self.results[ckpt]:
                        val = self.results[ckpt][wt].get(metric_key)
                        values.append(val if val is not None else 0)
                    else:
                        values.append(0)

                ax.plot(steps, values, marker='o', label=wt.upper(), color=colors[wt], linewidth=2)

            ax.set_xlabel('Training Step')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} Evolution')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend()

        # Remove empty subplot
        fig.delaxes(axes[1, 2])

        plt.suptitle('Clustering Emergence During Training', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = FIGURES_DIR / f"clustering_emergence_{timestamp}.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"\n📈 Saved emergence timeline to {filepath}")

        return fig

    def identify_phase_transition(self):
        """Identify when clustering suddenly emerges"""
        print("\n🔍 Identifying Phase Transitions...")

        transitions = {}

        for weight_type in ['query', 'key', 'value']:
            scores = []
            steps = []

            for ckpt in self.checkpoints:
                if ckpt in self.results:
                    score = self.results[ckpt][weight_type]['silhouette_score']
                    scores.append(score)
                    step = 143000 if ckpt == "step143000" else int(ckpt.replace("step", ""))
                    steps.append(step)

            # Find biggest jump
            if len(scores) > 1:
                deltas = np.diff(scores)
                max_delta_idx = np.argmax(np.abs(deltas))

                transitions[weight_type] = {
                    'transition_step': f"{steps[max_delta_idx]}-{steps[max_delta_idx+1]}",
                    'score_before': scores[max_delta_idx],
                    'score_after': scores[max_delta_idx+1],
                    'delta': deltas[max_delta_idx],
                    'is_sudden': abs(deltas[max_delta_idx]) > 0.2
                }

                print(f"  {weight_type.upper()}: Biggest change at steps {transitions[weight_type]['transition_step']}")
                print(f"    Score: {transitions[weight_type]['score_before']:.3f} → "
                      f"{transitions[weight_type]['score_after']:.3f} "
                      f"(Δ={transitions[weight_type]['delta']:.3f})")

        return transitions

    def save_results(self):
        """Save all results to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        def convert_numpy(obj):
            """Recursively convert numpy types to native Python types"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            else:
                return obj

        # Convert numpy arrays to lists for JSON serialization
        json_safe_results = {}
        for ckpt, ckpt_results in self.results.items():
            json_safe_results[ckpt] = {}
            for key, value in ckpt_results.items():
                if key == 'raw_weights':
                    continue  # Skip raw weights (too large)
                else:
                    json_safe_results[ckpt][key] = convert_numpy(value)

        # Save JSON results
        filepath = RESULTS_DIR / f"training_dynamics_{timestamp}.json"
        with open(filepath, 'w') as f:
            json.dump(json_safe_results, f, indent=2)

        print(f"\n💾 Saved results to {filepath}")

        return filepath


def main():
    """Run the complete training dynamics analysis"""
    analyzer = TrainingDynamicsAnalyzer()

    # Run analysis
    results = analyzer.run_full_analysis()

    # Generate visualizations
    analyzer.plot_emergence_timeline()

    # Identify phase transitions
    transitions = analyzer.identify_phase_transition()

    # Save results
    analyzer.save_results()

    # Print summary
    print("\n" + "=" * 50)
    print("📊 TRAINING DYNAMICS SUMMARY")
    print("=" * 50)

    print("\n🎯 Key Findings:")

    # Find checkpoint with highest clustering
    best_clustering = {'checkpoint': None, 'score': -1, 'type': None}
    for ckpt in analyzer.checkpoints:
        for wt in ['query', 'key', 'value']:
            score = results[ckpt][wt]['silhouette_score']
            if score > best_clustering['score']:
                best_clustering = {'checkpoint': ckpt, 'score': score, 'type': wt}

    print(f"  • Strongest clustering: {best_clustering['type'].upper()} weights at {best_clustering['checkpoint']} "
          f"(score={best_clustering['score']:.3f})")

    # Check if gradual or sudden
    sudden_transitions = sum(1 for t in transitions.values() if t.get('is_sudden', False))
    if sudden_transitions > 0:
        print(f"  • Emergence pattern: SUDDEN ({sudden_transitions}/3 weight types show phase transition)")
    else:
        print(f"  • Emergence pattern: GRADUAL (continuous evolution)")

    # Early vs late emergence
    early_checkpoints = ['step1000', 'step5000', 'step10000', 'step20000']
    early_clustering = any(
        results[ckpt][wt]['silhouette_score'] > 0.3
        for ckpt in early_checkpoints
        for wt in ['query', 'key', 'value']
    )

    if early_clustering:
        print(f"  • Timing: EARLY emergence (visible in first 20k steps)")
    else:
        print(f"  • Timing: LATE emergence (appears after 20k steps)")

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()