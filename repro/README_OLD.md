# Reproduction Package - Decimal Comparison Bug in Llama-3.1-8B

This directory contains all scripts and data needed to reproduce the key findings about the decimal comparison bug in Llama-3.1-8B-Instruct, including the critical discovery that **Layer 10 is the sole intervention point** for fixing this bug.

## 🎯 Key Discovery

**Layer 10's attention output is the ONLY single-layer intervention point that successfully fixes the decimal comparison bug.** This layer acts as a re-entanglement bottleneck where format-separated representations merge with different weights, creating the bug.

## 📋 Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (minimum 24GB VRAM, tested on A100-80GB)
- **RAM**: 32GB minimum
- **Storage**: ~20GB for model weights

### Software Requirements
```bash
# Install required packages
pip install torch transformers sae-lens numpy matplotlib seaborn tqdm
```

### Model Download
The scripts will automatically download `meta-llama/Llama-3.1-8B-Instruct` on first run.

## 🚀 Quick Start

### Step 1: Verify the Bug Exists (30 seconds)
```bash
python quick_bug_test.py
```
Expected output:
```
Simple format: 0/10 bug (0% error rate) ✅
Q&A format: 10/10 bug (100% error rate) ❌
```

### Step 2: Demonstrate Layer 10 Causality (2 minutes)
```bash
python bidirectional_patching.py
```
This proves Layer 10 can both fix and induce the bug with 100% success.

## 📊 Complete Analysis Reproduction

### Phase 1: Core Bug Analysis (15 minutes)

#### 1.1 Verify Bug Across Formats
```bash
python verify_llama_bug.py
```
**Purpose**: Confirms bug exists only in Q&A and chat formats, not simple format.
**Output**: Console display showing format-dependent behavior.

#### 1.2 Format Comparison
```bash
python format_comparison.py
```
**Purpose**: Tests multiple prompt formats systematically.
**Output**: Detailed comparison of bug rates across formats.

#### 1.3 Bidirectional Patching
```bash
python bidirectional_patching.py
```
**Purpose**: Proves Layer 10 attention output is causally responsible.
**Key Results**:
- Forward patch (buggy→correct): 100% fixes bug
- Reverse patch (correct→buggy): 100% induces bug

#### 1.4 Original Discovery
```bash
python attention_control_experiment.py
```
**Purpose**: Original script that discovered Layer 10's unique role.
**Output**: Shows only Layer 10 works; other layers produce gibberish.

### Phase 2: SAE Feature Analysis (45 minutes)

#### 2.1 Complete 32-Layer Analysis
```bash
python all_layers_batched.py
```
**Purpose**: Analyzes feature overlap across all 32 layers using Sparse Autoencoders.
**Key Findings**:
- Layer 7-8: Minimum overlap (10-20%) - format discrimination
- Layer 10: Maximum overlap (80%) - re-entanglement bottleneck
- Dramatic phase transition at Layer 9→10

**Output**: 
- Console display of layer-wise analysis
- `all_32_layers_analysis.json` with complete data

#### 2.2 Layer 10 Deep Dive
```bash
python layer_10_focused_analysis.py
```
**Purpose**: Detailed SAE analysis of Layer 10's features.
**Output**: Feature-level understanding of the re-entanglement mechanism.

### Phase 3: Statistical Validation (2-3 hours)

```bash
python statistical_validation.py
```
**Purpose**: Comprehensive statistical proof with n=1000 trials.
**Key Results**:
- 100% success rate for Layer 10 intervention
- p-value < 10⁻³⁰⁰ (essentially zero chance of randomness)
- Generalizes to 4/5 decimal pairs
- 60% minimum replacement threshold

**Output**:
- `statistical_validation_results.json` - Complete numerical results
- `statistical_validation_visualization.png` - 6-panel dashboard
- Console progress updates during execution

### Phase 4: Attention Head Analysis (1 hour)

#### 4.1 Structured Head Subsets
```bash
python structured_head_subsets.py
```
**Purpose**: Tests different configurations of attention heads.
**Tests**: First 16, last 16, alternating, random subsets.

#### 4.2 Even Head Validation
```bash
python validate_even_heads.py
```
**Purpose**: Tests hypothesis about even-numbered heads.
**Output**: JSON file with validation results.

#### 4.3 Odd Head Testing
```bash
python test_odd_heads_subsets.py
```
**Purpose**: Tests odd-numbered head patterns.
**Finding**: All 32 heads required for reliable intervention.

#### 4.4 Minimal Head Set
```bash
python test_minimal_even_heads.py
```
**Purpose**: Finds minimum head subset needed.
**Result**: Full set of 32 heads required.

### Phase 5: Visualization (10 minutes)

#### 5.1 Statistical Validation Figure
```bash
python create_statistical_validation_figure.py
```
**Output**: `statistical_validation_figure.png` - Publication-quality results visualization

#### 5.2 Even/Odd Analysis Figure
```bash
python create_even_odd_figure.py
```
**Output**: `even_odd_discovery.png` - Head pattern analysis visualization

#### 5.3 Publication Figures
```bash
python create_publication_pdf.py
```
**Output**: PDF versions of all figures for publication

## 📁 Data Files

### Input Data (Pre-computed)
- `statistical_validation_results.json` - Main validation results (n=1000)
- `all_32_layers_analysis.json` - Complete 32-layer SAE features
- `even_heads_validation_*.json` - Even head pattern results
- `odd_heads_subsets_*.json` - Odd head pattern results
- `minimal_even_heads_*.json` - Minimal subset exploration

### Documentation
- `ANALYSIS_SUMMARY.md` - Overview of all analyses
- `FINAL_RESULTS.md` - Comprehensive results report
- `LAYER_10_CRITICAL_DISCOVERY.md` - Detailed Layer 10 findings
- `STRUCTURED_HEAD_ANALYSIS_FINDINGS.md` - Head pattern analysis

## 🔬 Understanding the Results

### The Bug Mechanism
1. **Layer 6**: Format detection begins
2. **Layer 7-8**: Maximum format discrimination (10-20% feature overlap)
3. **Layer 10**: Re-entanglement bottleneck (80% overlap) - **INTERVENTION POINT**
4. **Layer 13-15**: Hijacker neurons activate
5. **Layer 25**: Answer commitment
6. **Layer 26-31**: Output generation

### Why Only Layer 10 Works
- **Timing**: After format separation but before commitment
- **Architecture**: Bottleneck where paths must merge
- **Feature overlap**: 80% shared features allow control over both paths
- **Amplification**: 1.24x suggests active bias application

### Key Metrics
| Metric | Value | Significance |
|--------|-------|--------------|
| Layer 10 Success Rate | 100% | Perfect intervention |
| Statistical Confidence | p < 10⁻³⁰⁰ | Essentially certain |
| Feature Overlap at L10 | 80% | Maximum in model |
| Feature Overlap at L7-8 | 10-20% | Minimum in model |
| Heads Required | 32/32 | All heads needed |
| Generalization | 4/5 pairs | Works for most decimals |

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Solution: Use smaller batch sizes in scripts
   - Alternative: Use CPU (slower but works)

2. **Model Download Fails**
   - Ensure you have Hugging Face access token if needed
   - Check internet connection and disk space

3. **SAE Loading Errors**
   - Ensure sae-lens is properly installed: `pip install sae-lens`
   - Check CUDA compatibility

4. **Slow Execution**
   - Statistical validation takes 2-3 hours (this is normal)
   - Use quick_bug_test.py for rapid verification

## 📚 Citation

If you use this code in your research, please cite:
```
[Your citation information here]
```

## 📧 Contact

For questions or issues with reproduction:
- Open an issue on GitHub
- [Contact information]

## 🔄 Version Information

- **Date**: August 2024
- **Model**: meta-llama/Llama-3.1-8B-Instruct
- **Key Finding**: Layer 10 is the sole intervention point
- **Success Rate**: 100% (n=1000, p < 10⁻³⁰⁰)

---

*This reproduction package provides complete code and data to verify that Layer 10's attention output is the unique architectural bottleneck responsible for the decimal comparison bug in Llama-3.1-8B.*