# Statistical Validation Directory

Comprehensive statistical validation of the Layer 10 attention causality hypothesis with rigorous experimental design.

## 📁 Contents

- `comprehensive_validation.py` - Full validation suite with n=1000 trials
- `quick_validation.py` - Fast validation with reduced n for testing  
- `statistical_validation_report.md` - Detailed report and expected results
- `validation_results_*.json` - Experimental results (when run)
- `comprehensive_validation_*.png` - Visualization dashboard (when generated)

## 🔬 Experiments Included

### 1. Statistical Rigor (n=1000)
- Format comparison with confidence intervals
- Layer 10 intervention success rate
- Bidirectional patching validation
- Bootstrap confidence intervals and p-values

### 2. Multiple Decimal Pairs
Tests generalization beyond 9.8 vs 9.11:
- 8.7 vs 8.12
- 10.9 vs 10.11  
- 7.85 vs 7.9
- 3.4 vs 3.25

### 3. Head-Level Analysis
- Individual head contributions
- Minimal head set discovery
- Cumulative success rates

### 4. Ablation Study
- Partial replacement (20%, 40%, 60%, 80%, 100%)
- Threshold identification
- Non-linear response curve

## 🚀 Usage

### Quick Testing (n=50)
```bash
python quick_validation.py
```
- Runs in ~10-15 minutes
- Good for development and testing
- Not for publication

### Full Validation (n=1000)
```bash
python comprehensive_validation.py
```
- Runs in ~2-3 hours on GPU
- Publication-quality results
- Full statistical power

## 📊 Expected Results

| Metric | Expected Value | Statistical Significance |
|--------|---------------|-------------------------|
| Format Comparison | ~100% | p < 0.0001 |
| Layer 10 Success | ~100% | p < 0.0001 |
| Works on Multiple Pairs | 5/5 | All significant |
| Minimum Heads Needed | ~8 | For >80% success |
| Minimum Replacement | ~80% | Threshold for reliability |

## 🎯 Key Features

1. **Proper Statistics**
   - Bootstrap confidence intervals
   - Binomial tests against chance
   - Multiple comparison corrections
   - Effect size calculations

2. **Comprehensive Testing**
   - 1000 trials for main claims
   - Multiple decimal pairs
   - Head-level granularity
   - Ablation analysis

3. **Visualization**
   - 6-panel dashboard
   - Error bars and confidence intervals
   - Summary statistics
   - Publication-ready figures

## ⚠️ Computational Requirements

- **GPU**: Strongly recommended (CUDA)
- **Memory**: ~16GB GPU memory for full runs
- **Time**: 2-3 hours for n=1000
- **Storage**: ~100MB for results

## 📝 Notes

- The full n=1000 experiments are computationally intensive
- Results are deterministic with temperature=0
- Caching can speed up repeated runs
- Can parallelize across multiple GPUs if available

## 🔄 Development Workflow

1. Test with `quick_validation.py` during development
2. Run `comprehensive_validation.py` for final results
3. Review `statistical_validation_report.md` for interpretation
4. Check generated JSON and PNG files for detailed results

---

*Part of the decimal comparison bug research project*