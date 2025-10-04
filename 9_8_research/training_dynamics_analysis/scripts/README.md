# Training Dynamics Analysis Scripts

This directory contains all scripts used in the comprehensive training dynamics analysis investigation. Scripts are organized by research phase for easy navigation.

## 📁 Directory Structure

### **Phase 1: Training Dynamics** (`phase1_training_dynamics/`)
Scripts investigating when even/odd attention head specialization emerges during training.

- **`test_pythia_training_dynamics.py`** - Core training dynamics analysis
  - Tests 11 Pythia-160M checkpoints (step 1k → 143k)
  - Implements selective attention head patching for even/odd heads
  - Maps emergence timeline of apparent specialization
  - **Key finding**: Specialization emerges only at final checkpoint

### **Phase 2: Pythia Analysis** (`phase2_pythia_analysis/`)
Scripts analyzing the nature and scope of Pythia's decimal comparison pattern.

- **`test_comprehensive_decimal_patterns.py`** - Comprehensive pattern testing
  - Tests 4 categories of decimal comparison patterns
  - Evaluates generalization across different number pairs
  - **Key finding**: Only 12% success rate, ultra-specific pattern

- **`test_memorization_hypothesis.py`** - Memorization vs reasoning analysis
  - Tests phrase sensitivity, semantic equivalents, tokenization effects
  - Generates memorization score (0.14/1.0 = strong memorization evidence)
  - **Key finding**: Pattern is memorized, not reasoned

- **`test_biblical_interference_hypothesis.py`** - Biblical context testing
  - Tests whether pattern is patch for biblical verse interference
  - Evaluates systematic decimal comparison baseline (0% accuracy)
  - **Key finding**: Moderate evidence for biblical interference

### **Phase 3: Cross-Model Validation** (`phase3_cross_model_validation/`)
Scripts testing whether findings generalize to other model architectures.

- **`test_llama_comprehensive.py`** - Llama-3.1-8B comprehensive testing
  - Uses identical methodology to Pythia testing
  - Tests both base and instruct model variants
  - **Key finding**: Llama performs worse than Pythia on most categories

### **Phase 4: Format Sensitivity** (`phase4_format_sensitivity/`)
Scripts investigating impact of prompt format on model performance.

- **`test_llama_chat_format.py`** - Format comparison testing
  - Compares Q&A format vs official chat template vs Transluce format
  - **Key finding**: Format dramatically affects performance

- **`test_transluce_exact_format.py`** - Exact Transluce format replication
  - Tests precise minimal format used by Transluce study
  - **Key finding**: 41.7% accuracy - largely explains Transluce discrepancy

### **Utils** (`utils/`)
*Reserved for shared utility functions (none created yet)*

## 🚀 **How to Run Scripts**

### Prerequisites
```bash
# Ensure you have required packages
pip install torch transformers numpy matplotlib json datetime
```

### Running Individual Scripts
```bash
# Phase 1: Training dynamics
cd phase1_training_dynamics
python test_pythia_training_dynamics.py

# Phase 2: Pythia analysis
cd phase2_pythia_analysis
python test_comprehensive_decimal_patterns.py
python test_memorization_hypothesis.py
python test_biblical_interference_hypothesis.py

# Phase 3: Cross-model validation
cd phase3_cross_model_validation
python test_llama_comprehensive.py

# Phase 4: Format sensitivity
cd phase4_format_sensitivity
python test_llama_chat_format.py
python test_transluce_exact_format.py
```

## 📊 **Key Results Summary**

| Phase | Key Finding | Accuracy |
|-------|-------------|----------|
| **Phase 1** | Specialization emerges at final checkpoint only | 0% → 100% |
| **Phase 2** | Ultra-specific memorization, not reasoning | 12% generalization |
| **Phase 3** | Llama performs worse than Pythia | 0-5% vs 37.5% |
| **Phase 4** | Format sensitivity explains Transluce discrepancy | 0% → 41.7% |

## 🔧 **Script Dependencies**

All scripts require:
- **PyTorch** - For model loading and inference
- **Transformers** - For Pythia and Llama models
- **Standard libraries** - json, datetime, random, re

Model downloads (automatic on first run):
- `EleutherAI/pythia-160m` (various checkpoints)
- `meta-llama/Meta-Llama-3.1-8B`
- `meta-llama/Meta-Llama-3.1-8B-Instruct`

## 📁 **Output Files**

Each script generates:
- **JSON results file** with timestamp
- **Console output** with real-time progress
- **Visualization files** (for training dynamics)

Example output files are stored in the main directory with descriptive timestamps.

## 🔬 **Methodology Notes**

### **Consistent Testing Framework**
- All scripts use identical decimal comparison evaluation
- Same prompt structures across tests for comparability
- Standardized correctness evaluation

### **Statistical Rigor**
- Multiple test cases per category
- Documented success/failure rates
- Comprehensive boundary testing

### **Reproducibility**
- Fixed random seeds where applicable
- Exact model specifications documented
- Complete parameter settings recorded

## 🎯 **Research Impact**

These scripts collectively demonstrate:
1. **Memorization can masquerade as reasoning** (Phase 2)
2. **Capabilities don't transfer across architectures** (Phase 3)
3. **Prompt format dramatically affects performance** (Phase 4)
4. **Systematic evaluation reveals hidden limitations** (All phases)

The methodology developed here provides a template for rigorous AI capability evaluation.

---

*For complete analysis and findings, see `COMPREHENSIVE_FINDINGS_REPORT.md` in the parent directory.*