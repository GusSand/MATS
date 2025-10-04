# Bandwidth Competition Theory - Experimental Results Index

## Overview
This document indexes all experimental results from our investigation into the bandwidth competition theory and the critic's challenge about even/odd head indexing in transformer attention mechanisms.

---

## 1. Attention Weights vs Outputs Investigation
**File**: `investigation_weights_vs_outputs_20250926_155755.json`
**Script**: `investigation_attention_weights_vs_outputs.py`

### Key Findings:
- **Attention Weights**: 8-10% numerical bandwidth, odd heads outperform even heads
- **Attention Outputs**: 24-49% numerical bandwidth, minimal even/odd differences
- **Conclusion**: Successful patching works on outputs, not weights

### Critical Results:
```json
{
  "weights_analysis": {
    "overall_bandwidth": 0.08125,
    "even_heads_bandwidth": 0.075,
    "odd_heads_bandwidth": 0.0875
  },
  "outputs_analysis": {
    "overall_bandwidth": 0.49375,
    "even_heads_bandwidth": 0.49375,
    "odd_heads_bandwidth": 0.49375
  }
}
```

---

## 2. Specific Even Heads Investigation
**File**: `specific_heads_investigation_20250926_161028.json`
**Script**: `investigation_specific_even_heads.py`

### Key Findings:
- **Original Claim**: "ANY 8 even heads achieve 100% success" - **REFUTED**
- **Actual Results**: Only 19/30 random combinations work (63% success rate)
- **Failed Combinations**: 11 combinations achieved 0% success

### Critical Results:
```json
{
  "summary_statistics": {
    "total_combinations_tested": 30,
    "successful_combinations": 19,
    "failed_combinations": 11,
    "success_rate": 0.6333333333333333
  }
}
```

---

## 3. Spatial Organization Investigation
**File**: `spatial_organization_investigation_20250926_162904.json`
**Script**: `spatial_organization_investigation.py`

### Key Findings:
- **15 patterns tested**: 11 successful, 4 failed
- **Gap regularity predicts success**: Regular spacing > irregular spacing
- **Spatial organization matters more than even/odd indexing**

### Pattern Results:
```json
{
  "pattern_testing": {
    "consecutive": {"success_rate": 1.0, "patterns_tested": 3},
    "uniform_spacing": {"success_rate": 1.0, "patterns_tested": 3},
    "balanced": {"success_rate": 1.0, "patterns_tested": 3},
    "irregular": {"success_rate": 0.33, "patterns_tested": 6}
  }
}
```

### Spatial Metrics:
- **Coverage Efficiency**: 0.25 (optimal)
- **Gap Regularity**: 0.78-1.0 for successful patterns
- **Mean Gap**: 3.5-4.0 for successful patterns

---

## 4. Functional Clustering Analysis
**File**: `functional_clustering_analysis_20250926_164446.json`
**Script**: `functional_clustering_analysis.py`

### Key Findings:
- **Adjusted Rand Index**: -0.060 (NO functional clustering by even/odd)
- **Cluster Correlation**: 0.0
- **Heads cluster by spatial proximity**: Groups of 4 adjacent heads
- **Function-based prediction**: Still selects 100% even heads despite no clustering

### Clustering Evidence:
```json
{
  "analysis_summary": {
    "ari_score": -0.05982905982905983,
    "cluster_correlation": 0.0,
    "working_avg_similarity": -0.03170738240615243,
    "failing_avg_similarity": -0.03161162838694595,
    "final_conclusion": "MIXED EVIDENCE - PARTIAL FUNCTIONAL SPECIALIZATION"
  }
}
```

### Detailed Clustering Results:
- **Working combinations**: 4 unique clusters, 57% purity
- **Failing combinations**: 4-8 unique clusters, 0-57% purity
- **No pure even/odd clusters found**
- **Spatial clustering**: Groups of 4 (heads 0-3, 4-7, 8-11, etc.)

---

## Summary of All Experiments

### Research Question Timeline:
1. **Initial**: Test "ANY 8 even heads work" claim
2. **Refined**: Why do specific 8 even heads work while others fail?
3. **Spatial**: Does spatial organization predict success?
4. **Methodological**: Attention weights vs outputs difference?
5. **Theoretical**: Test critic's challenge about functional clustering

### Key Discoveries:
1. **Original claim overgeneralized** - only 63% of random even combinations work
2. **Spatial organization predicts success** - gap regularity matters more than indexing
3. **Attention outputs more relevant** - higher bandwidth, method used in successful patches
4. **No functional clustering by even/odd** - heads cluster spatially (groups of 4)
5. **Mixed evidence on specialization** - no clustering but function-based selection favors even heads

### Unresolved Questions:
1. **Why does function-based prediction still favor even heads?**
2. **Is this training dynamics or architectural constraints?**
3. **Does the pattern hold across different models?**

### Files Generated:
- `figures/functional_clustering_analysis.png` - Clustering analysis visualization
- `figures/spatial_organization_comprehensive.png` - Spatial organization results
- `figures/comprehensive_summary_analysis.png` - Overall summary figure

---

## Next Steps for Further Investigation:
1. **Cross-model validation**: Test patterns across different Llama model sizes
2. **Training dynamics**: Analyze when even/odd specialization emerges during training
3. **Architecture ablation**: Compare with models having different head arrangements
4. **Causal intervention**: Test direct manipulation of attention patterns