# Pythia Weight Clustering Dynamics Analysis

## Overview

This experiment leverages Pythia's unique advantage - comprehensive training checkpoints - to understand when and how attention head weight clustering emerges during training. We analyze the formation of distinct weight clusters and their causal role in model behavior.

## Research Questions

1. **When do weight clusters emerge during training?**
   - Is clustering present from initialization or learned?
   - Does it emerge gradually or suddenly (phase transition)?
   - At what training step does clustering become apparent?

2. **What are the mechanistic differences between clusters?**
   - How do cluster A and B weights differ statistically?
   - Are these differences causally important for model behavior?
   - Do different clusters specialize in different tasks?

## Experimental Design

### 1. Training Dynamics Analysis (`analyze_training_dynamics.py`)

Analyzes clustering emergence across 8 training checkpoints:
- **Checkpoints**: 1k, 5k, 10k, 20k, 40k, 80k, 120k, 143k steps
- **Metrics Tracked**:
  - Silhouette score (cluster quality)
  - Inter-cluster distance
  - Within-cluster variance
  - Cluster separation ratio
  - Wasserstein distance between clusters
  - Weight divergence from previous checkpoint

**Key Outputs**:
- Timeline plots showing clustering metric evolution
- Phase transition identification
- Classification of emergence pattern (gradual vs sudden)

### 2. Mechanistic Analysis (`mechanistic_analysis.py`)

Deep analysis of what makes clusters different and why it matters:

**Analyses Performed**:
- **Statistical Comparison**: KS test and t-test between cluster weight distributions
- **Visualization**: PCA and t-SNE projections of head weights
- **Causal Intervention**: Test if swapping cluster assignments changes outputs
- **Functional Specialization**: Analyze what each cluster specializes in

**Test Suite**:
- Numerical comparisons (9.8 vs 9.11)
- Basic arithmetic
- Language tasks
- Pattern completion

## Key Findings Expected

### Hypothesis 1: Early Architectural Bias
- Clustering visible from early checkpoints (< 10k steps)
- Gradual refinement throughout training
- Suggests architectural inductive bias

### Hypothesis 2: Late-Stage Optimization
- Clustering emerges late in training (> 80k steps)
- Sudden phase transition
- Suggests learned optimization for efficiency

### Hypothesis 3: Task-Specific Specialization
- Different clusters handle different types of inputs
- Causal intervention changes model behavior
- Clear functional differentiation

## Directory Structure

```
pythia_clustering_dynamics/
├── scripts/
│   ├── analyze_training_dynamics.py    # Checkpoint analysis
│   └── mechanistic_analysis.py         # Causal & functional analysis
├── docs/
│   └── README.md                        # This file
├── figures/
│   ├── clustering_emergence_*.png      # Timeline visualizations
│   └── cluster_visualization_*.png     # PCA/t-SNE plots
├── results/
│   ├── training_dynamics_*.json        # Checkpoint analysis results
│   └── mechanistic_analysis_*.json     # Mechanistic findings
└── data/
    └── [cached model weights]
```

## Running the Experiments

### 1. Training Dynamics Analysis

```bash
cd pythia_clustering_dynamics/scripts
python analyze_training_dynamics.py
```

**Expected Runtime**: ~15-20 minutes (downloading checkpoints + analysis)

**Outputs**:
- JSON file with all clustering metrics
- PNG visualization of emergence timeline
- Console summary of key findings

### 2. Mechanistic Analysis

```bash
python mechanistic_analysis.py
```

**Expected Runtime**: ~5-10 minutes

**Outputs**:
- JSON file with statistical tests and intervention results
- PCA/t-SNE visualizations
- Functional specialization analysis

## Interpretation Guide

### Clustering Metrics

- **Silhouette Score**: [-1, 1] range, higher = better defined clusters
  - > 0.5: Strong clustering
  - 0.3-0.5: Moderate clustering
  - < 0.3: Weak/no clustering

- **Cluster Separation Ratio**: Inter-cluster distance / within-cluster variance
  - > 2.0: Well-separated clusters
  - 1.0-2.0: Moderate separation
  - < 1.0: Poor separation

### Phase Transition Detection

The analysis identifies the training step window where the largest change in clustering metrics occurs. A "sudden" transition is defined as:
- Silhouette score change > 0.2 between consecutive checkpoints
- Indicates discrete optimization event rather than gradual evolution

### Causal Intervention

Tests if cluster identity is causally important:
1. Baseline: Normal model output
2. Intervention: Swap cluster assignments (A→B, B→A)
3. Measure: KL divergence between output distributions
   - KL > 1.0: Strong causal effect
   - KL 0.1-1.0: Moderate effect
   - KL < 0.1: Minimal effect

## Scientific Significance

This analysis provides:
1. **First systematic study** of weight clustering emergence during training
2. **Causal evidence** for functional importance of clusters
3. **Framework** for understanding attention head specialization
4. **Insights** into training dynamics and optimization

## Related Work

- Builds on activation patching methodology from prior experiments
- Extends findings about even/odd head specialization
- Provides mechanistic understanding complementary to behavioral observations

## Next Steps

Based on findings, potential follow-ups include:
1. Test larger Pythia models (410M, 1B, 2.8B)
2. Analyze other layers beyond Layer 6
3. Track clustering in other weight matrices (MLP, embeddings)
4. Correlate clustering emergence with capability development

---

*Experiment designed: September 2025*
*Purpose: Understand the training dynamics and mechanistic basis of weight clustering in transformers*