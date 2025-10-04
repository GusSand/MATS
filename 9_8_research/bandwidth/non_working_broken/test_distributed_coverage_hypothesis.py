#!/usr/bin/env python3
"""
Test Distributed Coverage Hypothesis
==================================

This script designs and implements experiments to test whether the 8-head requirement
is due to "distributed coverage across attention space" vs other hypotheses.

Hypotheses to Test:
1. Distributed Coverage: Need 8 heads spread across 32-head space (25% coverage)
2. Functional Redundancy: Need multiple heads from same functional clusters
3. Critical Mass: Simply need 8 heads regardless of distribution
4. Even/Odd Specialization: Even heads have special properties
5. Architectural Constraint: GQA structure requires specific pattern
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

class DistributedCoverageExperiment:
    def __init__(self):
        self.model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.layer = 10
        self.n_heads = 32
        self.test_prompts = [
            "Which is larger: 9.8 or 9.11?",
            "Compare 9.8 and 9.11",
            "Is 9.8 > 9.11?",
            "Which number is bigger: 9.8 or 9.11?"
        ]

    def generate_test_patterns(self):
        """Generate patterns to test different hypotheses"""
        patterns = {}

        # Hypothesis 1: Distributed Coverage (25% coverage, evenly spaced)
        patterns['distributed_even_spacing'] = [
            [0, 4, 8, 12, 16, 20, 24, 28],  # Every 4th head
            [1, 5, 9, 13, 17, 21, 25, 29],  # Every 4th head, offset 1
            [2, 6, 10, 14, 18, 22, 26, 30], # Every 4th head, offset 2
            [3, 7, 11, 15, 19, 23, 27, 31], # Every 4th head, offset 3
        ]

        # Hypothesis 2: Functional Redundancy (multiple heads from same "clusters")
        patterns['functional_redundancy'] = [
            [0, 1, 4, 5, 8, 9, 12, 13],     # 2 from each "group of 4"
            [0, 2, 4, 6, 8, 10, 12, 14],    # Skip pattern within groups
            [16, 17, 20, 21, 24, 25, 28, 29], # High-index clusters only
        ]

        # Hypothesis 3: Critical Mass (random sets of 8)
        np.random.seed(42)
        patterns['critical_mass'] = []
        for i in range(5):
            random_8 = np.random.choice(32, 8, replace=False).tolist()
            patterns['critical_mass'].append(sorted(random_8))

        # Hypothesis 4: Even/Odd Specialization
        patterns['even_odd_test'] = [
            list(range(0, 16, 2)),  # First 8 even heads
            list(range(16, 32, 2)), # Last 8 even heads
            list(range(1, 16, 2)),  # First 8 odd heads
            list(range(17, 32, 2)), # Last 8 odd heads
        ]

        # Hypothesis 5: Coverage Density Tests
        patterns['coverage_density'] = [
            [0, 1, 2, 3, 4, 5, 6, 7],       # Dense coverage (first 8)
            [24, 25, 26, 27, 28, 29, 30, 31], # Dense coverage (last 8)
            [0, 5, 10, 15, 20, 25, 30, 31],  # Sparse coverage
            [4, 6, 10, 12, 16, 18, 22, 24], # Medium spacing
        ]

        # Test different head counts for coverage hypothesis
        patterns['head_count_test'] = {
            4: [[0, 8, 16, 24], [1, 9, 17, 25], [2, 10, 18, 26]],
            6: [[0, 5, 11, 16, 21, 27], [2, 7, 13, 18, 23, 29]],
            8: [[0, 4, 8, 12, 16, 20, 24, 28]],
            10: [[0, 3, 6, 9, 12, 15, 18, 21, 24, 27]],
            12: [[0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 31]],
        }

        return patterns

    def calculate_coverage_metrics(self, heads):
        """Calculate various coverage metrics for a head selection"""
        heads = sorted(heads)
        n_heads = len(heads)

        # Coverage efficiency (proportion of space covered)
        coverage_efficiency = n_heads / 32

        # Spatial distribution
        gaps = [heads[i+1] - heads[i] for i in range(len(heads)-1)]
        mean_gap = np.mean(gaps) if gaps else 0
        gap_variance = np.var(gaps) if gaps else 0
        gap_regularity = 1 / (1 + gap_variance) if gap_variance > 0 else 1

        # Coverage uniformity
        expected_gap = 32 / n_heads
        gap_deviation = np.mean([abs(gap - expected_gap) for gap in gaps]) if gaps else 0

        # Span coverage
        span = heads[-1] - heads[0] if len(heads) > 1 else 0
        span_efficiency = span / 31  # 31 is max possible span

        return {
            'coverage_efficiency': coverage_efficiency,
            'mean_gap': mean_gap,
            'gap_variance': gap_variance,
            'gap_regularity': gap_regularity,
            'gap_deviation': gap_deviation,
            'span': span,
            'span_efficiency': span_efficiency
        }

    def predict_success_by_hypothesis(self, patterns):
        """Predict which patterns should succeed under each hypothesis"""
        predictions = {}

        for hypothesis, pattern_sets in patterns.items():
            if hypothesis == 'head_count_test':
                # Special handling for head count tests
                predictions[hypothesis] = {}
                for count, head_lists in pattern_sets.items():
                    if count == 8:
                        predictions[hypothesis][count] = [1.0] * len(head_lists)  # Should work
                    elif count < 8:
                        predictions[hypothesis][count] = [0.0] * len(head_lists)  # Should fail
                    else:
                        predictions[hypothesis][count] = [0.8] * len(head_lists)  # Might work
                continue

            pred_list = []
            for heads in pattern_sets:
                metrics = self.calculate_coverage_metrics(heads)

                if hypothesis == 'distributed_even_spacing':
                    # Should succeed if gap regularity is high and coverage is ~25%
                    score = metrics['gap_regularity'] * metrics['coverage_efficiency'] * 4
                    pred_list.append(min(score, 1.0))

                elif hypothesis == 'functional_redundancy':
                    # Should succeed if heads cluster in groups
                    # Low prediction since our clustering showed this doesn't work
                    pred_list.append(0.3)

                elif hypothesis == 'critical_mass':
                    # Should succeed if simply having 8 heads matters
                    pred_list.append(0.7)  # Some should work

                elif hypothesis == 'even_odd_test':
                    # Even heads should work, odd shouldn't
                    heads_array = np.array(heads)
                    even_ratio = np.sum(heads_array % 2 == 0) / len(heads_array)
                    pred_list.append(even_ratio)

                elif hypothesis == 'coverage_density':
                    # Distributed coverage should work better than dense
                    inverse_density = metrics['span_efficiency']
                    pred_list.append(inverse_density)

            predictions[hypothesis] = pred_list

        return predictions

    def run_coverage_analysis(self):
        """Run the coverage analysis experiment"""

        print("=== DISTRIBUTED COVERAGE HYPOTHESIS TEST ===")
        print(f"Model: {self.model_name}")
        print(f"Layer: {self.layer}")
        print(f"Testing different head selection patterns...\n")

        # Generate test patterns
        patterns = self.generate_test_patterns()

        # Calculate predictions
        predictions = self.predict_success_by_hypothesis(patterns)

        # Analyze patterns
        results = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'layer': self.layer,
            'patterns': patterns,
            'predictions': predictions,
            'analysis': {}
        }

        print("PATTERN ANALYSIS:")
        print("=" * 50)

        for hypothesis, pattern_sets in patterns.items():
            if hypothesis == 'head_count_test':
                print(f"\n{hypothesis.upper()}:")
                for count, head_lists in pattern_sets.items():
                    print(f"  {count} heads: {len(head_lists)} patterns")
                    for i, heads in enumerate(head_lists):
                        metrics = self.calculate_coverage_metrics(heads)
                        print(f"    Pattern {i}: {heads}")
                        print(f"      Coverage: {metrics['coverage_efficiency']:.3f}")
                        print(f"      Gap regularity: {metrics['gap_regularity']:.3f}")
                continue

            print(f"\n{hypothesis.upper()}:")
            preds = predictions[hypothesis]

            for i, heads in enumerate(pattern_sets):
                metrics = self.calculate_coverage_metrics(heads)
                pred_score = preds[i]

                print(f"  Pattern {i+1}: {heads}")
                print(f"    Predicted success: {pred_score:.3f}")
                print(f"    Coverage efficiency: {metrics['coverage_efficiency']:.3f}")
                print(f"    Gap regularity: {metrics['gap_regularity']:.3f}")
                print(f"    Span efficiency: {metrics['span_efficiency']:.3f}")

                # Store detailed analysis
                results['analysis'][f"{hypothesis}_pattern_{i}"] = {
                    'heads': heads,
                    'predicted_success': pred_score,
                    'metrics': metrics
                }

        # Summary predictions by hypothesis
        print("\n" + "=" * 50)
        print("HYPOTHESIS PREDICTIONS SUMMARY:")
        print("=" * 50)

        for hypothesis, preds in predictions.items():
            if hypothesis == 'head_count_test':
                print(f"\n{hypothesis}: Success should depend on head count (8 optimal)")
                continue

            avg_pred = np.mean(preds)
            print(f"\n{hypothesis}: Average predicted success = {avg_pred:.3f}")

            if avg_pred > 0.7:
                print("  → Strong support expected")
            elif avg_pred > 0.4:
                print("  → Moderate support expected")
            else:
                print("  → Weak support expected")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"distributed_coverage_hypothesis_test_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_file}")

        return results

    def create_hypothesis_visualization(self, results):
        """Create visualization of different hypotheses"""

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        hypothesis_names = [
            'distributed_even_spacing',
            'functional_redundancy',
            'critical_mass',
            'even_odd_test',
            'coverage_density'
        ]

        for i, hypothesis in enumerate(hypothesis_names):
            ax = axes[i]

            if hypothesis not in results['patterns']:
                continue

            patterns = results['patterns'][hypothesis]
            predictions = results['predictions'][hypothesis]

            # Create head selection visualization
            for j, (heads, pred) in enumerate(zip(patterns, predictions)):
                y_pos = j

                # Plot all head positions
                ax.scatter(range(32), [y_pos] * 32, c='lightgray', s=30, alpha=0.3)

                # Plot selected heads
                color = plt.cm.RdYlGn(pred)  # Color by prediction
                ax.scatter(heads, [y_pos] * len(heads), c=[color], s=80, alpha=0.8)

                # Add prediction score
                ax.text(33, y_pos, f'{pred:.2f}', va='center', fontweight='bold')

            ax.set_xlim(-1, 35)
            ax.set_ylim(-0.5, len(patterns) - 0.5)
            ax.set_xlabel('Head Index')
            ax.set_ylabel('Pattern')
            ax.set_title(hypothesis.replace('_', ' ').title(), fontweight='bold')
            ax.grid(True, alpha=0.3)

        # Head count test in last subplot
        if 'head_count_test' in results['patterns']:
            ax = axes[5]

            head_counts = []
            avg_predictions = []

            for count, patterns in results['patterns']['head_count_test'].items():
                head_counts.append(count)
                avg_pred = np.mean(results['predictions']['head_count_test'][count])
                avg_predictions.append(avg_pred)

            ax.plot(head_counts, avg_predictions, 'bo-', linewidth=2, markersize=8)
            ax.axvline(x=8, color='red', linestyle='--', alpha=0.7, label='Required: 8 heads')
            ax.set_xlabel('Number of Heads')
            ax.set_ylabel('Predicted Success Rate')
            ax.set_title('Head Count vs Success', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()

        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"figures/hypothesis_testing_{timestamp}.png"
        os.makedirs("figures", exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_file}")

        plt.show()

def main():
    """Main experiment runner"""

    experiment = DistributedCoverageExperiment()

    print("Running Distributed Coverage Hypothesis Test...")
    print("This experiment tests multiple hypotheses about why 8 heads are needed:")
    print("1. Distributed Coverage: Need coverage across attention space")
    print("2. Functional Redundancy: Need multiple heads from same clusters")
    print("3. Critical Mass: Simply need 8 heads regardless of selection")
    print("4. Even/Odd Specialization: Even heads have special properties")
    print("5. Coverage Density: Sparse > dense coverage")
    print()

    # Run analysis
    results = experiment.run_coverage_analysis()

    # Create visualization
    experiment.create_hypothesis_visualization(results)

    print("\n" + "=" * 60)
    print("NEXT STEPS TO PROVE/DISPROVE HYPOTHESES:")
    print("=" * 60)
    print("1. IMPLEMENT ACTUAL PATCHING: Test these patterns with real model intervention")
    print("2. MEASURE PERFORMANCE: Run numerical reasoning tasks on each pattern")
    print("3. STATISTICAL ANALYSIS: Compare predicted vs actual success rates")
    print("4. CROSS-MODEL VALIDATION: Test on different Llama model sizes")
    print("5. ABLATION STUDIES: Gradually remove heads to find minimum effective set")

if __name__ == "__main__":
    main()