#!/usr/bin/env python3
"""
Synthesis: Combine all results from CWE-787 SR vs SCG experiments.

Generates:
1. Unified summary of all findings
2. Publication-quality figures
3. Final answer: Are SR and SCG separately encoded?
"""

import sys
from pathlib import Path

import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


def load_all_results(results_dir: Path) -> dict:
    """Load results from all experiment steps."""
    results = {}

    # Probe results
    probe_files = sorted(results_dir.glob("sr_scg_probes_*.json"))
    if probe_files:
        with open(probe_files[-1]) as f:
            results['probes'] = json.load(f)
        print(f"Loaded probe results: {probe_files[-1].name}")

    # Differential steering
    steering_files = sorted(results_dir.glob("differential_steering_*.json"))
    if steering_files:
        with open(steering_files[-1]) as f:
            results['steering'] = json.load(f)
        print(f"Loaded steering results: {steering_files[-1].name}")

    # Jailbreak
    jailbreak_files = sorted(results_dir.glob("jailbreak_test_*.json"))
    if jailbreak_files:
        with open(jailbreak_files[-1]) as f:
            results['jailbreak'] = json.load(f)
        print(f"Loaded jailbreak results: {jailbreak_files[-1].name}")

    # Latent guard
    guard_files = sorted(results_dir.glob("latent_guard_*.json"))
    if guard_files:
        with open(guard_files[-1]) as f:
            results['guard'] = json.load(f)
        print(f"Loaded guard results: {guard_files[-1].name}")

    return results


def generate_summary(results: dict) -> dict:
    """Generate unified summary."""
    summary = {
        'research_question': 'Are Security Recognition (SR) and Secure Code Generation (SCG) separately encoded in LLaMA-3.1-8B using validated CWE-787 prompt pairs?',
        'data_source': '7 validated CWE-787 prompt pairs with 100% separation',
        'findings': {}
    }

    # Probe findings
    if 'probes' in results:
        sr_results = results['probes']['sr_probe_results']
        scg_results = results['probes']['scg_probe_results']
        sims = results['probes']['similarities']

        sr_best = max((r['accuracy'] for r in sr_results if r['accuracy']), default=0)
        scg_best = max((r['accuracy'] for r in scg_results if r['accuracy']), default=0)
        valid_sims = [s['cosine_similarity'] for s in sims if s['cosine_similarity'] is not None]

        summary['findings']['probes'] = {
            'sr_best_accuracy': sr_best,
            'scg_best_accuracy': scg_best,
            'avg_similarity': np.mean(valid_sims) if valid_sims else None,
            'min_similarity': min(valid_sims) if valid_sims else None,
            'n_low_sim_layers': sum(1 for s in valid_sims if s < 0.5)
        }

    # Steering findings
    if 'steering' in results:
        summary['findings']['steering'] = {
            'avg_scg_sr_ratio': results['steering'].get('avg_ratio'),
            'conclusion': results['steering'].get('conclusion')
        }

    # Jailbreak findings
    if 'jailbreak' in results:
        summary['findings']['jailbreak'] = {
            'n_attempts': results['jailbreak']['analysis']['n_attempts'],
            'n_successes': results['jailbreak']['analysis']['n_successes'],
            'conclusion': results['jailbreak']['analysis']['conclusion']
        }

    # Guard findings
    if 'guard' in results:
        summary['findings']['guard'] = {
            'accuracy': results['guard']['eval_metrics']['accuracy'],
            'f1_score': results['guard']['eval_metrics']['f1']
        }

    return summary


def draw_conclusion(summary: dict) -> str:
    """Draw overall conclusion."""
    evidence_for = 0
    evidence_against = 0

    findings = summary.get('findings', {})

    # Similarity evidence
    if 'probes' in findings:
        avg_sim = findings['probes'].get('avg_similarity')
        if avg_sim is not None:
            if avg_sim < 0.5:
                evidence_for += 2
            elif avg_sim > 0.7:
                evidence_against += 2

    # Steering evidence
    if 'steering' in findings:
        ratio = findings['steering'].get('avg_scg_sr_ratio')
        if ratio is not None:
            if ratio > 2:
                evidence_for += 1
            elif ratio < 0.5:
                evidence_against += 1

    # Jailbreak evidence
    if 'jailbreak' in findings:
        if findings['jailbreak'].get('n_successes', 0) > 0:
            evidence_for += 2

    # Conclusion
    if evidence_for > evidence_against + 2:
        conclusion = "STRONG EVIDENCE FOR SEPARATION: SR and SCG appear to be separately encoded, similar to harmfulness/refusal separation."
    elif evidence_for > evidence_against:
        conclusion = "MODERATE EVIDENCE FOR SEPARATION: Some indicators suggest SR and SCG are distinct."
    elif evidence_against > evidence_for:
        conclusion = "EVIDENCE AGAINST SEPARATION: SR and SCG appear aligned/overlapping."
    else:
        conclusion = "INCONCLUSIVE: Mixed evidence."

    return conclusion


def create_synthesis_figure(results: dict, summary: dict, output_path: Path):
    """Create summary figure."""
    fig = plt.figure(figsize=(16, 12))

    # Panel 1: Probe accuracies
    ax1 = fig.add_subplot(2, 2, 1)
    if 'probes' in results:
        sr_acc = [r['accuracy'] if r['accuracy'] else 0.5 for r in results['probes']['sr_probe_results']]
        scg_acc = [r['accuracy'] if r['accuracy'] else 0.5 for r in results['probes']['scg_probe_results']]
        layers = range(len(sr_acc))

        ax1.plot(layers, sr_acc, 'b-o', markersize=4, label='SR (Recognition)')
        ax1.plot(layers, scg_acc, 'r-s', markersize=4, label='SCG (Generation)')
        ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('A. Probe Accuracy by Layer')
        ax1.legend()
        ax1.set_ylim(0.4, 1.05)
        ax1.grid(True, alpha=0.3)

    # Panel 2: Direction similarity
    ax2 = fig.add_subplot(2, 2, 2)
    if 'probes' in results:
        sims = results['probes']['similarities']
        layers = [s['layer'] for s in sims]
        cos_sims = [s['cosine_similarity'] if s['cosine_similarity'] else 0 for s in sims]

        ax2.bar(layers, cos_sims, color='purple', alpha=0.7)
        ax2.axhline(y=0.5, color='orange', linestyle='--', label='Separation threshold')
        ax2.fill_between(layers, 0, 0.5, alpha=0.1, color='green')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Cosine Similarity')
        ax2.set_title('B. SR-SCG Direction Similarity')
        ax2.legend()
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

    # Panel 3: Steering effects
    ax3 = fig.add_subplot(2, 2, 3)
    if 'steering' in results:
        analysis = results['steering'].get('analysis', {})
        if analysis:
            layers = sorted([int(k) for k in analysis.keys()])
            sr_effects = [analysis[str(l)]['sr_max_effect'] for l in layers]
            scg_effects = [analysis[str(l)]['scg_max_effect'] for l in layers]

            x = np.arange(len(layers))
            width = 0.35

            ax3.bar(x - width/2, sr_effects, width, label='SR steering', color='blue', alpha=0.7)
            ax3.bar(x + width/2, scg_effects, width, label='SCG steering', color='red', alpha=0.7)
            ax3.set_xlabel('Layer')
            ax3.set_ylabel('Max Effect on P(secure)')
            ax3.set_title('C. Differential Steering Effects')
            ax3.set_xticks(x)
            ax3.set_xticklabels([f'L{l}' for l in layers])
            ax3.legend()
            ax3.grid(True, alpha=0.3)

    # Panel 4: Summary text
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    conclusion = draw_conclusion(summary)

    text = "SYNTHESIS SUMMARY\n" + "=" * 40 + "\n\n"
    text += "Data: 7 validated CWE-787 pairs\n"
    text += "(100% separation in validation)\n\n"

    if 'probes' in summary['findings']:
        avg_sim = summary['findings']['probes'].get('avg_similarity')
        if avg_sim:
            text += f"Direction Similarity: {avg_sim:.3f}\n"

    if 'steering' in summary['findings']:
        ratio = summary['findings']['steering'].get('avg_scg_sr_ratio')
        if ratio:
            text += f"SCG/SR Effect Ratio: {ratio:.2f}x\n"

    if 'jailbreak' in summary['findings']:
        n_success = summary['findings']['jailbreak'].get('n_successes', 0)
        text += f"Jailbreak Successes: {n_success}\n"

    if 'guard' in summary['findings']:
        acc = summary['findings']['guard'].get('accuracy')
        if acc:
            text += f"Latent Guard Accuracy: {acc*100:.1f}%\n"

    text += "\n" + "=" * 40 + "\n"
    text += f"\n{conclusion}"

    ax4.text(0.1, 0.9, text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSynthesis figure saved to: {output_path}")


def main():
    results_dir = Path(__file__).parent / "results"

    print("\n" + "=" * 70)
    print("SYNTHESIS: CWE-787 SR vs SCG SEPARATION")
    print("=" * 70)

    # Load all results
    print("\nLoading results...")
    results = load_all_results(results_dir)

    if not results:
        print("No results found. Run the experiment pipeline first.")
        return None

    # Generate summary
    summary = generate_summary(results)
    conclusion = draw_conclusion(summary)

    # Print findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    for finding_name, finding_data in summary['findings'].items():
        print(f"\n{finding_name.upper()}:")
        for k, v in finding_data.items():
            if v is not None:
                if isinstance(v, float):
                    print(f"  {k}: {v:.3f}")
                else:
                    print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print(f"\n{conclusion}")

    # Create figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = results_dir / f"synthesis_{timestamp}.png"
    create_synthesis_figure(results, summary, fig_path)

    # Save summary
    summary['conclusion'] = conclusion
    summary['timestamp'] = timestamp

    with open(results_dir / f"synthesis_{timestamp}.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {results_dir}")

    return summary


if __name__ == "__main__":
    summary = main()
