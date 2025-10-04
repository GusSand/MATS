# MATS9 - Decimal Comparison Bug Research

Research into the causal mechanisms behind the decimal comparison bug in Llama-3.1-8B-Instruct, where the model incorrectly compares 9.8 vs 9.11 depending on prompt format.

## 🎯 Main Finding

**Layer 10 attention output is causally responsible for the bug.** This has been confirmed with 100% bidirectional causality.

## 📁 Repository Structure

```
MATS9/
├── working_scripts/           # ✅ VERIFIED WORKING IMPLEMENTATIONS
│   ├── README.md             # Detailed documentation of working scripts
│   ├── bidirectional_patching.py  # Main result - 100% bidirectional causality
│   ├── attention_control_experiment.py  # Original working implementation
│   ├── verify_llama_bug.py  # Bug verification
│   └── format_comparison.py  # Format-bug correlation analysis
│
├── experimental/             # 🧪 EXPERIMENTAL & IN-PROGRESS WORK
│   ├── attention/           # Attention mechanism experiments
│   │   ├── causal/         # Causal intervention attempts
│   │   └── ...             # Various attention analyses
│   │
│   ├── attention_output_patching/  # Initial patching attempts
│   │   ├── README.md       # Documentation of attempts
│   │   └── ...             # Various implementations
│   │
│   ├── layer25/            # Layer-specific experiments
│   │   └── ...             # Multiple layer testing
│   │
│   ├── logitlens/          # Logit lens visualizations
│   ├── sae/                # SAE analysis
│   ├── acdc/               # ACDC experiments
│   └── submission/         # Submission materials
│
└── README.md               # This file
```

## 🚀 Quick Start

To reproduce the main finding:

```bash
cd working_scripts
python bidirectional_patching.py
```

This will demonstrate:
1. The bug exists (baselines)
2. Forward patching fixes the bug (100% success)
3. Reverse patching induces the bug (100% success)

## 📊 Key Results

| Experiment | Location | Success Rate |
|------------|----------|--------------|
| Bidirectional Patching | `working_scripts/bidirectional_patching.py` | 100% |
| Bug Verification | `working_scripts/verify_llama_bug.py` | 100% reproducible |
| Format Correlation | `working_scripts/format_comparison.py` | 100% correlation |

## 🔬 The Bug

**What**: Llama-3.1-8B incorrectly says 9.11 > 9.8 (should be 9.8 > 9.11)

**When**: Only in certain prompt formats:
- ✅ Works: `"Which is bigger: 9.8 or 9.11?\nAnswer:"`
- ❌ Fails: `"Q: Which is bigger: 9.8 or 9.11?\nA:"`

**Why**: Layer 10's attention module processes format information that affects numerical comparison

**Fix**: Replacing Layer 10 attention output from buggy format with correct format output

## 🧑‍🔬 Development Workflow

1. **Start with working_scripts/** - These are verified to work
2. **Experiment in experimental/** - Keep the same directory structure
3. **When something works** - Move it to working_scripts/ with documentation
4. **Document findings** - Update READMEs with results and learnings

## 📝 Important Findings

1. **Attention Outputs vs Weights**: Only output patching works, not weight patching
2. **Layer Specificity**: Only Layer 10 works cleanly for intervention
3. **Bidirectional Causality**: Can both fix and induce the bug
4. **Format Dependency**: 100% correlation between format and bug occurrence

## 🔗 Related Documentation

- `working_scripts/README.md` - Detailed documentation of working implementations
- `experimental/attention/BREAKTHROUGH_FINDINGS.md` - Theoretical insights
- `WORKING_SCRIPTS_SUMMARY.md` - Historical summary of what works vs doesn't

## 📈 Next Steps

1. Understand WHY Layer 10 is special
2. Investigate the specific attention heads involved
3. Explore potential model fixes
4. Test on other models for generalization

## ⚠️ Note on Experimental Directory

The `experimental/` directory contains:
- Work in progress
- Failed attempts (kept for documentation)
- Exploratory analyses
- Diagnostic tools

Not everything in experimental/ works as intended. See individual READMEs for status.

---

*Research conducted August 2024*  
*Model: Llama-3.1-8B-Instruct*