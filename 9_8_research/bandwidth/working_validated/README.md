# Working Validated Scripts

This directory contains scripts that have been validated and produce reliable results for the bandwidth competition theory investigation.

## Key Scripts

### 🎯 **Definitive Experiments**

1. **`test_permutation_invariance.py`** - **MOST IMPORTANT**
   - **Purpose**: Direct test of critic's permutation invariance argument
   - **Result**: DEFINITIVELY REFUTES critic - head indices are functionally meaningful
   - **Key Finding**: Performance varies dramatically with permutations (0-100% range)
   - **Usage**: `python test_permutation_invariance.py`

2. **`test_distributed_coverage_proper.py`** - **CORE VALIDATION**
   - **Purpose**: Proper activation patching using correct methodology
   - **Result**: Establishes 8-head requirement and even/odd specialization
   - **Key Finding**: Exactly 8 even heads needed, spacing irrelevant
   - **Usage**: `python test_distributed_coverage_proper.py`

### 📊 **Supporting Analysis Scripts**

3. **`functional_clustering_analysis.py`**
   - **Purpose**: Test functional vs spatial clustering of attention heads
   - **Result**: Heads cluster spatially (groups of 4), not functionally by even/odd
   - **Key Finding**: Architecture drives clustering, not function

4. **`spatial_organization_investigation.py`**
   - **Purpose**: Test whether spatial organization predicts success
   - **Result**: Spatial patterns matter for some combinations but not others
   - **Key Finding**: Led to discovering distributed coverage hypothesis

5. **`investigation_specific_even_heads.py`**
   - **Purpose**: Test "ANY 8 even heads work" claim
   - **Result**: REFUTES overgeneralization - only 63% of random combinations work
   - **Key Finding**: Not all even head combinations are equivalent

6. **`investigation_attention_weights_vs_outputs.py`**
   - **Purpose**: Compare attention weights vs outputs analysis
   - **Result**: Outputs more relevant for successful patching
   - **Key Finding**: Methodology matters for bandwidth analysis

## Dependencies

All scripts require:
- `torch`
- `transformers`
- `numpy`
- `matplotlib`
- `json`

## Results Files

Each script generates timestamped JSON results files with detailed experimental data.

## Usage Notes

- Scripts use `meta-llama/Meta-Llama-3.1-8B-Instruct` by default
- Layer 10 is the target layer for attention patching
- All scripts include progress bars and detailed logging
- Results are automatically saved and visualized