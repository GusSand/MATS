# Reproduction Scripts Summary - Decimal Comparison Bug Research

## Key Analyses to Reproduce

### 1. Core Bug Verification & Patching
- **Basic bug verification**: Confirms the decimal comparison bug exists
- **Bidirectional patching**: Layer 10 attention output patching (fixes and induces bug)
- **Statistical validation**: n=1000 trials proving 100% success rate

### 2. SAE (Sparse Autoencoder) Analysis
- **All 32 layers analysis**: Complete feature overlap analysis revealing phase transitions
- **Layer 10 focused analysis**: Deep dive into the re-entanglement bottleneck
- **Feature discrimination**: Identifying format-specific features

### 3. Attention Head Analysis
- **Even/odd head analysis**: Testing structured subsets of attention heads
- **Minimal head validation**: Finding minimum heads needed for intervention
- **Head subset testing**: First 16, last 16, alternating patterns

### 4. Visualization & Documentation
- **Statistical validation figures**: Publication-quality visualizations
- **Even/odd discovery plots**: Visual proof of head patterns
- **Comprehensive results**: JSON data files with all experimental results

## Script Categories

### Essential Core Scripts (Must Run)
1. `verify_llama_bug.py` - Verify bug exists
2. `bidirectional_patching.py` - Prove Layer 10 causality
3. `statistical_validation.py` - Complete statistical proof
4. `all_layers_batched.py` - SAE analysis across all 32 layers
5. `layer_10_focused_analysis.py` - Deep Layer 10 SAE analysis

### Head Pattern Analysis
6. `structured_head_subsets.py` - Test different head configurations
7. `validate_even_heads.py` - Validate even-numbered heads
8. `test_odd_heads_subsets.py` - Test odd-numbered head patterns
9. `test_minimal_even_heads.py` - Find minimal working subset

### Visualization Scripts
10. `create_statistical_validation_figure.py` - Generate main results figure
11. `create_even_odd_figure.py` - Generate head pattern visualization
12. `create_publication_pdf.py` - Create publication-ready figures

### Supporting Scripts
13. `format_comparison.py` - Test multiple prompt formats
14. `quick_bug_test.py` - Quick 30-second bug verification
15. `attention_control_experiment.py` - Original Layer 10 discovery

## Data Files Required
- `statistical_validation_results.json` - Main validation results
- `all_32_layers_analysis.json` - Complete layer-wise SAE features
- `even_heads_validation_*.json` - Even head validation data
- `odd_heads_subsets_*.json` - Odd head validation data
- `minimal_even_heads_*.json` - Minimal subset results

## Execution Order

### Phase 1: Verify Bug (5 minutes)
```bash
python verify_llama_bug.py
python quick_bug_test.py
python format_comparison.py
```

### Phase 2: Core Causality (10 minutes)
```bash
python bidirectional_patching.py
python attention_control_experiment.py
```

### Phase 3: SAE Analysis (30 minutes)
```bash
python all_layers_batched.py
python layer_10_focused_analysis.py
```

### Phase 4: Statistical Validation (2-3 hours)
```bash
python statistical_validation.py
```

### Phase 5: Head Pattern Analysis (1 hour)
```bash
python structured_head_subsets.py
python validate_even_heads.py
python test_odd_heads_subsets.py
python test_minimal_even_heads.py
```

### Phase 6: Visualization (10 minutes)
```bash
python create_statistical_validation_figure.py
python create_even_odd_figure.py
```

## Key Findings Summary

1. **Layer 10 is the ONLY single-layer intervention point** that fixes the bug
2. **80% feature overlap at Layer 10** creates re-entanglement bottleneck
3. **100% success rate** with Layer 10 attention output patching (n=1000)
4. **All 32 attention heads required** for successful intervention
5. **Phase transitions**: Layer 7-8 (10-20% overlap) → Layer 10 (80% overlap)
6. **Bidirectional causality**: Can both fix and induce the bug
7. **Format-dependent**: Bug only occurs with Q&A/Chat formats, not simple format

## Hardware Requirements
- GPU: NVIDIA with CUDA support (tested on A100-80GB)
- RAM: 32GB minimum
- Disk: ~20GB for model weights
- Time: Full reproduction ~4-5 hours

## Software Dependencies
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- SAE-Lens
- NumPy, Matplotlib, Seaborn
- CUDA toolkit