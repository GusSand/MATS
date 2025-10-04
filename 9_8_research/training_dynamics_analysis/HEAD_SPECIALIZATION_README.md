# Training Dynamics Analysis: When Does Even/Odd Head Specialization Emerge?

## Overview

This directory contains a comprehensive analysis of **when and how even/odd attention head specialization emerges during training** using Pythia-160M model checkpoints across the full training process.

## Research Question

**When does even/odd head specialization first appear during training?**
- Is it an early architectural bias or late-stage optimization effect?
- Does emergence happen gradually or suddenly (phase transition)?
- What training dynamics drive this specialization?

## Experimental Design

### Model: Pythia-160M
- **Why Pythia**: We know the final model shows perfect even/odd specialization
- **Architecture**: GPT-NeoX, 12 heads per layer, 12 layers
- **Target Layer**: Layer 6 (middle layer, known effective for final model)

### Checkpoints Tested (11 total)
EleutherAI provides comprehensive training snapshots:
1. `step1000` - Very early training
2. `step2000` - Early training
3. `step4000` - Early training
4. `step8000` - Early-mid training
5. `step16000` - Early-mid training
6. `step32000` - Mid training
7. `step64000` - Mid training
8. `step128000` - Mid-late training
9. `step256000` - Late training
10. `step512000` - Late training
11. `step1000000` - Final model (known to work)

### Methodology
1. **Activation Patching**: Use same methodology that works for final model
2. **Even vs Odd Heads**: Test [0,2,4,6,8,10] vs [1,3,5,7,9,11]
3. **Numerical Task**: 9.8 vs 9.11 comparison (known bug)
4. **Metrics**: Success rate difference (even - odd) = specialization strength

## Expected Timeline
- **Setup**: 5 minutes
- **Checkpoint downloads**: 30 minutes
- **Testing**: ~10 minutes per checkpoint × 11 = ~2 hours
- **Analysis & visualization**: 30 minutes
- **Total**: ~3 hours

## Directory Structure

```
training_dynamics_analysis/
├── scripts/
│   └── test_pythia_training_dynamics.py    # Main experiment script
├── results/
│   ├── pythia_training_dynamics_YYYYMMDD_HHMMSS.json  # Complete results
│   └── individual_checkpoint_results/      # Per-checkpoint detailed data
├── figures/
│   ├── specialization_emergence_timeline.png
│   ├── individual_success_rates.png
│   └── emergence_analysis.png
└── documentation/
    ├── README.md                           # This file
    ├── METHODOLOGY.md                      # Detailed methodology
    └── TRAINING_DYNAMICS_REPORT.md         # Final analysis report
```

## Key Hypotheses to Test

### Emergence Timing
- **H1 (Early)**: Specialization emerges in first 10% of training (steps 1k-100k)
- **H2 (Mid)**: Specialization emerges in middle training (steps 100k-500k)
- **H3 (Late)**: Specialization emerges in final training (steps 500k-1000k)

### Emergence Pattern
- **H4 (Gradual)**: Specialization strength increases smoothly over training
- **H5 (Sudden)**: Specialization appears suddenly in a phase transition
- **H6 (Fluctuating)**: Specialization appears, disappears, then stabilizes

### Training Correlations
- **H7**: Emergence correlates with major loss reduction milestones
- **H8**: Emergence correlates with general capability development
- **H9**: Emergence is independent of other training milestones

## Expected Scientific Impact

This will be the **first systematic study** of when attention head specialization emerges during training, providing:

1. **Precise emergence timeline** for even/odd specialization
2. **Training dynamics insights** that could inform future model development
3. **Evidence for/against architectural bias** vs learned optimization effects
4. **Framework for studying** other types of attention head specialization

## Files

### Scripts
- `test_pythia_training_dynamics.py` - Main experiment runner

### Results (Generated)
- `pythia_training_dynamics_YYYYMMDD_HHMMSS.json` - Complete experimental data
- Individual checkpoint result files

### Visualizations (Generated)
- Timeline plots showing specialization emergence
- Success rate comparisons across training
- Emergence pattern analysis

### Documentation
- `METHODOLOGY.md` - Detailed experimental methodology
- `TRAINING_DYNAMICS_REPORT.md` - Comprehensive results and analysis

---

*Experiment designed: September 26, 2025*
*Expected completion: ~3 hours from start*
*Scientific significance: First-ever timeline of attention head specialization emergence*