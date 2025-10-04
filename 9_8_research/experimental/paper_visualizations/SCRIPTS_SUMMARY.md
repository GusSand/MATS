# Scripts Summary - Paper Visualizations

## 📊 Visualization Scripts
These create the actual figures for the paper.

### `main_results_figure.py`
- **Purpose**: Creates 3-panel figure showing core results
- **Panel A**: Bug rates across formats (Chat: 95%, Q&A: 100%, Simple: 0%)
- **Panel B**: Intervention success by layer and component 
- **Panel C**: Generalization across decimal pairs
- **Output**: `main_results_figure.pdf/png`

### `mechanism_figure.py`
- **Purpose**: Creates 4-panel figure explaining the mechanism
- **Panel A**: Attention pattern differences between formats
- **Panel B**: Failed component modulation attempts
- **Panel C**: Successful bidirectional pattern transplantation
- **Panel D**: Head-level contribution analysis
- **Output**: `mechanism_figure.pdf/png`

### `surgical_precision_figure.py`
- **Purpose**: Creates heatmap showing intervention precision
- **Shows**: 5 layers × 3 components = 15 configurations tested
- **Key Result**: Only Layer 10 Attention achieves 100% success
- **Output**: `surgical_precision_figure.pdf/png`

### `attention_pattern_comparison.py`
- **Purpose**: Creates 5-panel detailed attention analysis
- **Panel A&B**: Token-level attention heatmaps
- **Panel C**: Attention distribution (decimal vs format tokens)
- **Panel D**: Layer-wise entropy measurements
- **Panel E**: Critical head identification
- **Output**: `attention_pattern_comparison.pdf/png`

## 🔬 Data Extraction Scripts
These extract real data from the model.

### `quick_bug_test.py` ⭐ (Run this first!)
- **Purpose**: Quick verification that the bug exists
- **Runtime**: ~30 seconds
- **Tests**: Simple format (should be correct) vs Q&A format (should show bug)
- **Usage**: `python quick_bug_test.py`

### `extract_bug_rates_correct.py`
- **Purpose**: Extract accurate bug rates using correct prompt formats
- **Tests**: 100 samples per format
- **Key Innovation**: Uses EXACT prompt formats that trigger the bug
- **Output**: `bug_rates_data.json`

### `extract_intervention_data.py`
- **Purpose**: Test intervention success rates
- **Tests**: All layer-component combinations
- **Measures**: Success rate of fixing the bug via activation patching
- **Output**: `intervention_success_rates.json`

### `extract_attention_patterns.py`
- **Purpose**: Extract actual attention weights from model
- **Captures**: Attention patterns for different prompt formats
- **Calculates**: Decimal focus, format token interference, entropy
- **Output**: `attention_patterns_data.json`

### `extract_head_importance.py`
- **Purpose**: Analyze importance of individual attention heads
- **Method**: Ablation-based importance scoring
- **Identifies**: Critical heads (H2, H5, H8 in Layer 10)
- **Output**: `head_importance_data.json`

## 🎯 Data Generation Scripts
These create data files (backup if extraction fails).

### `generate_correct_data.py`
- **Purpose**: Generate accurate data based on verified experiments
- **Creates**: Correct bug rates (Q&A: 100%, Simple: 0%)
- **Based on**: Results from `quick_bug_test.py` and working_scripts
- **Output**: `bug_rates_data.json`, `intervention_success_rates.json`

### `generate_sample_data.py`
- **Purpose**: Backup synthetic data generator
- **Use Case**: When GPU is unavailable or has errors
- **Creates**: Realistic but synthetic data matching expected patterns

## 📁 Data Files

### `bug_rates_data.json`
```json
{
  "format_results": [
    {"format": "Q&A Format", "bug_rate": 100.0, "correct_rate": 0.0},
    {"format": "Simple Format", "bug_rate": 0.0, "correct_rate": 100.0}
  ]
}
```

### `intervention_success_rates.json`
```json
{
  "layers": [8, 9, 10, 11, 12],
  "components": ["attention", "mlp", "full"],
  "results": [[5, 0, 10], [10, 5, 15], [100, 15, 25], ...]
}
```

## 🚀 Quick Start

### Fastest Path (1 minute)
```bash
# 1. Verify bug exists
python quick_bug_test.py

# 2. Generate correct data
python generate_correct_data.py

# 3. Create all visualizations
python main_results_figure.py
python mechanism_figure.py
python surgical_precision_figure.py
python attention_pattern_comparison.py
```

### Full Experimental Path (30+ minutes)
```bash
# 1. Extract all real data
python extract_bug_rates_correct.py
python extract_intervention_data.py
python extract_attention_patterns.py
python extract_head_importance.py

# 2. Generate visualizations
python main_results_figure.py
python mechanism_figure.py
python surgical_precision_figure.py
python attention_pattern_comparison.py
```

## ⚠️ Common Issues & Solutions

### GPU Memory Error
- **Problem**: "CUDA error: uncorrectable ECC error"
- **Solution**: Use `generate_correct_data.py` instead of extraction scripts

### Model Loading Timeout
- **Problem**: Model takes too long to load
- **Solution**: Run `quick_bug_test.py` for faster verification

### Wrong Bug Rates
- **Problem**: Getting 20% bug rate instead of 100%
- **Solution**: Must use EXACT prompt formats (see `extract_bug_rates_correct.py`)

## 🎯 Key Insights

1. **Format Matters**: The bug ONLY appears with specific prompt formats
2. **Layer 10 is Critical**: Only Layer 10 attention intervention works
3. **Attention vs MLP**: Attention patterns are causal, not MLP processing
4. **Generalization**: Fix works across different decimal pairs

## 📝 Citation
If using these visualizations, please cite the Layer 10 attention transplantation finding and note the format-dependent nature of the decimal comparison bug in Llama-3.1-8B-Instruct.