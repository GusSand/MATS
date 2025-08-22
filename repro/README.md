# Reproduction Package - Decimal Comparison Bug in Llama-3.1-8B

## 📂 Organized Directory Structure

This reproduction package is organized by analysis type for easier navigation:

```
repro/
├── core/                  # Core bug verification
│   ├── verify_llama_bug.py
│   ├── quick_bug_test.py
│   └── format_comparison.py
│
├── layer_10/             # Layer 10 specific analyses
│   ├── bidirectional_patching.py
│   └── attention_control_experiment.py
│
├── sae/                  # SAE (Sparse Autoencoder) analyses
│   ├── all_layers_batched.py
│   ├── layer_10_focused_analysis.py
│   └── enhanced_sae_analysis.py
│
├── head_analysis/        # Attention head pattern analyses
│   ├── structured_head_subsets.py
│   ├── validate_even_heads.py
│   ├── test_odd_heads_subsets.py
│   └── test_minimal_even_heads.py
│
├── validation/           # Statistical validation
│   └── statistical_validation.py
│
├── figures/              # Figure generation and outputs
│   ├── create_*.py       # Figure generation scripts
│   └── *.png/pdf         # Generated figures
│
├── data/                 # Data files and documentation
│   ├── *.json           # Analysis results and mappings
│   └── *.md             # Analysis documentation
│
├── README.md             # This file
├── REPRODUCIBILITY_STATEMENT.md
└── requirements.txt
```

## 🚀 Quick Start by Task

### 1. Verify the Bug Exists
```bash
cd core/
python quick_bug_test.py  # 30-second verification
python verify_llama_bug.py  # Detailed verification
```

### 2. Layer 10 Intervention Analysis
```bash
cd layer_10/
python bidirectional_patching.py  # Prove Layer 10 causality
python attention_control_experiment.py  # Original discovery
```

### 3. SAE Feature Analysis
```bash
cd sae/
python all_layers_batched.py  # Analyze all 32 layers
python layer_10_focused_analysis.py  # Deep dive on Layer 10
```

### 4. Head Pattern Analysis
```bash
cd head_analysis/
python structured_head_subsets.py  # Test head configurations
python validate_even_heads.py  # Validate even/odd pattern
```

### 5. Statistical Validation
```bash
cd validation/
python statistical_validation.py  # Full validation (2-3 hours)
```

### 6. Generate Figures
```bash
cd figures/
python create_feature_head_visualization_final.py  # Main figure
```

## 🎯 Key Findings by Category

### Layer 10 Discovery
- **Location**: `layer_10/` directory
- **Finding**: Layer 10 is the ONLY single-layer intervention point
- **Mechanism**: Re-entanglement bottleneck with 80% feature overlap

### SAE Analysis
- **Location**: `sae/` directory  
- **Finding**: Phase transitions at Layer 7-8 (discrimination) and Layer 10 (re-entanglement)
- **Features**: F00-F09 (numerical), F10-F19 (format)

### Head Patterns
- **Location**: `head_analysis/` directory
- **Finding**: Any 8 even heads sufficient, odd heads incompatible
- **Correlation**: Even heads 85-92% with numerical features

### Statistical Proof
- **Location**: `validation/` directory
- **Finding**: 100% success rate (n=1000, p < 10^-300)

## 📊 Data Files

All data files are in the `data/` directory:

- **Analysis Results**: `all_32_layers_analysis.json`, `statistical_validation_results.json`
- **Feature Mappings**: `feature_mapping_table.json`, `FEATURE_MAPPING.md`
- **Head Validations**: `even_heads_validation_*.json`, `odd_heads_subsets_*.json`
- **Documentation**: `LAYER_10_CRITICAL_DISCOVERY.md`, `FINAL_RESULTS.md`

## 🔬 Execution Order

For complete reproduction:

1. **Core verification** (`core/`): 5 minutes
2. **Layer 10 analysis** (`layer_10/`): 10 minutes  
3. **SAE analysis** (`sae/`): 30-45 minutes
4. **Head analysis** (`head_analysis/`): 1 hour
5. **Statistical validation** (`validation/`): 2-3 hours
6. **Figure generation** (`figures/`): 10 minutes

## 📋 Requirements

See `requirements.txt` for dependencies. Install with:
```bash
pip install -r requirements.txt
```

## 🔍 Finding Specific Analyses

- **Bug verification**: `core/verify_llama_bug.py`
- **Layer 10 patching**: `layer_10/bidirectional_patching.py`
- **32-layer SAE analysis**: `sae/all_layers_batched.py`
- **Even/odd heads**: `head_analysis/validate_even_heads.py`
- **Statistical validation**: `validation/statistical_validation.py`
- **Main figure**: `figures/create_feature_head_visualization_final.py`

## 📚 Citation

If using this code, please cite our work on Layer 10 as the sole intervention point for the decimal comparison bug in Llama-3.1-8B.

---

*Organized for clarity and ease of reproduction. Each directory contains related analyses for specific aspects of the decimal comparison bug research.*