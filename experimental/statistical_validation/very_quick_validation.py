#!/usr/bin/env python3
"""
Very quick validation with n=10 for rapid testing
"""

from comprehensive_validation import StatisticalValidator
import json
from datetime import datetime
import numpy as np

def main():
    """Run very quick validation with n=10"""
    validator = StatisticalValidator()
    
    all_results = {}
    
    print("\n🔬 VERY QUICK VALIDATION (n=10 for rapid testing)")
    print("="*70)
    
    # Experiment 1: Basic tests (n=10)
    print("\nEXPERIMENT 1: BASIC STATISTICAL TESTS (n=10)")
    print("-"*50)
    
    results_1 = validator.run_with_statistics(
        lambda: validator.run_single_intervention("9.8", "9.11"),
        n=10, description="Layer 10 Intervention"
    )
    
    all_results['layer10_intervention'] = results_1
    print(f"Layer 10 Success: {results_1['success_rate']:.0%} [{results_1['ci_lower']:.0%}, {results_1['ci_upper']:.0%}]")
    
    # Experiment 2: Test 3 decimal pairs
    print("\nEXPERIMENT 2: MULTIPLE DECIMAL PAIRS")
    print("-"*50)
    
    test_pairs = [
        ("9.8", "9.11"),
        ("8.7", "8.12"),
        ("3.4", "3.25")
    ]
    
    pair_results = {}
    for num1, num2 in test_pairs:
        success_count = 0
        for _ in range(5):
            if validator.run_single_intervention(num1, num2):
                success_count += 1
        success_rate = success_count / 5
        pair_results[f"{num1}_vs_{num2}"] = success_rate
        print(f"  {num1} vs {num2}: {success_rate:.0%} success")
    
    all_results['decimal_pairs'] = pair_results
    
    # Experiment 3: Test a few heads
    print("\nEXPERIMENT 3: HEAD ANALYSIS (testing 4 heads)")
    print("-"*50)
    
    test_heads = [0, 8, 16, 24]
    head_results = []
    
    for head_idx in test_heads:
        success_count = 0
        for _ in range(5):
            if validator.run_single_intervention("9.8", "9.11", head_indices=[head_idx]):
                success_count += 1
        success_rate = success_count / 5
        head_results.append({'head': head_idx, 'rate': success_rate})
        print(f"  Head {head_idx}: {success_rate:.0%}")
    
    all_results['heads'] = head_results
    
    # Experiment 4: Test 3 replacement percentages
    print("\nEXPERIMENT 4: ABLATION (3 percentages)")
    print("-"*50)
    
    percentages = [0.5, 0.8, 1.0]
    ablation_results = {}
    
    for pct in percentages:
        success_count = 0
        for _ in range(5):
            if validator.run_single_intervention("9.8", "9.11", replacement_percentage=pct):
                success_count += 1
        success_rate = success_count / 5
        ablation_results[pct] = success_rate
        print(f"  {pct:.0%} replacement: {success_rate:.0%}")
    
    all_results['ablation'] = ablation_results
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'very_quick_results_{timestamp}.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to very_quick_results_{timestamp}.json")
    
    # Summary
    print("\n" + "="*70)
    print("VERY QUICK VALIDATION SUMMARY")
    print("="*70)
    print(f"""
Key Results (n=10, for quick testing only):
-------------------------------------------
1. Layer 10 Intervention: {all_results['layer10_intervention']['success_rate']:.0%}
2. Works on {sum(1 for v in pair_results.values() if v > 0.8)}/{len(pair_results)} decimal pairs
3. Best head contribution: {max(h['rate'] for h in head_results):.0%}
4. Full replacement success: {ablation_results[1.0]:.0%}

Note: These are preliminary results with very small n.
Run comprehensive_validation.py for publication-quality results.
    """)
    
    return all_results

if __name__ == "__main__":
    results = main()