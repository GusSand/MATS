#!/usr/bin/env python3
"""
Quick validation with smaller n for testing
"""

from comprehensive_validation import StatisticalValidator
import json
from datetime import datetime

def main():
    """Run quick validation with smaller n"""
    validator = StatisticalValidator()
    
    all_results = {}
    
    print("\n🔬 QUICK VALIDATION (smaller n for testing)")
    print("="*70)
    
    # Experiment 1: Statistical rigor (n=50 for speed)
    all_results['statistical_rigor'] = validator.experiment_1_statistical_rigor(n=50)
    
    # Experiment 2: Multiple decimal pairs
    all_results['multiple_pairs'] = validator.experiment_2_multiple_pairs()
    
    # Experiment 3: Head-level analysis (simplified)
    print("\n" + "="*70)
    print("EXPERIMENT 3: HEAD-LEVEL ANALYSIS (simplified)")
    print("="*70)
    print("Testing subset of heads for quick validation...")
    
    # Test just a few heads
    head_results = []
    test_heads = [0, 4, 8, 12, 16, 20, 24, 28]  # Sample 8 heads
    
    for head_idx in test_heads:
        success_rate = 0
        n_trials = 10
        
        for _ in range(n_trials):
            success = validator.run_single_intervention(
                "9.8", "9.11", head_indices=[head_idx]
            )
            success_rate += success
        
        success_rate /= n_trials
        head_results.append({
            'head_idx': head_idx,
            'success_rate': success_rate
        })
        print(f"  Head {head_idx}: {success_rate:.0%}")
    
    all_results['head_analysis'] = {
        'individual_heads': head_results,
        'note': 'Simplified - only tested subset of heads'
    }
    
    # Experiment 4: Ablation study
    all_results['ablation'] = validator.experiment_4_ablation()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(f'quick_validation_{timestamp}.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n✅ Quick validation results saved to quick_validation_{timestamp}.json")
    
    # Print summary
    print("\n" + "="*70)
    print("QUICK VALIDATION SUMMARY")
    print("="*70)
    
    print(f"""
Results (with reduced n for speed):
------------------------------------
1. Format Comparison: {all_results['statistical_rigor']['format_comparison']['success_rate']:.1%} (n=50)
2. Layer 10 Intervention: {all_results['statistical_rigor']['layer10_intervention']['success_rate']:.1%} (n=50)
3. Multiple pairs tested: {len(all_results['multiple_pairs'])}
4. Ablation threshold: {all_results['ablation'].get('threshold', 'N/A')}

Note: For publication-quality results, run comprehensive_validation.py with n=1000
    """)
    
    return all_results

if __name__ == "__main__":
    results = main()