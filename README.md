# MATS - Mechanistic Interpretability Research

This repository contains the code and experiments for the paper **"Even Heads Fix Odd Errors"** and follow-up research on mechanistic interpretability of language models.

**Paper**: [Even Heads Fix Odd Errors](https://arxiv.org/html/2508.19414v1) (arXiv:2508.19414)

## Paper Abstract

> We present a mechanistic case study of a format-dependent reasoning failure in Llama-3.1-8B-Instruct, where the model incorrectly judges "9.11" as larger than "9.8" in chat or Q&A formats, but answers correctly in simple format.
>
> Through systematic intervention, we discover transformers implement **even/odd attention head specialization**: even indexed heads handle numerical comparison, while odd heads serve incompatible functions. The bug requires exactly 8 even heads at Layer 10 for perfect repair. Any combination of 8+ even heads succeeds, while 7 or fewer completely fails, revealing **sharp computational thresholds with perfect redundancy** among the 16 even heads.
>
> SAE analysis reveals the mechanism: format representations separate (10% feature overlap at Layer 7), then re-entangle with different weightings (80% feature overlap at Layer 10), with specific features showing 1.5× amplification in failing formats. We achieve perfect repair using only 25% of attention heads and identify a 60% pattern replacement threshold, demonstrating that apparent full-module requirements hide sophisticated substructure with implications for interpretability and efficiency.

## Key Findings

### Original Paper (9.8 vs 9.11 Bug)
- **Even/odd head specialization**: Even-indexed attention heads handle numerical comparison; odd heads serve other functions
- **Sharp threshold**: Exactly 8 even heads at Layer 10 required for repair (7 fails completely, 8+ succeeds)
- **Perfect redundancy**: Any combination of 8+ even heads works
- **Format dependence**: Simple format works; chat/Q&A formats fail

### Follow-up: Security Code Generation Steering
- **Activation steering works for security**: A simple mean-difference direction can convert vulnerable code prompts into secure outputs
- **+66.7 pp conversion rate** on held-out test data (0% → 66.7% secure code generation)
- **Validated**: No overfitting detected - effect generalizes to unseen prompts
- **Layer 31, α=3.0** is the optimal configuration for CWE-787 (buffer overflow) vulnerabilities

## Repository Structure

```
MATS/
├── 9_8_research/              # Original paper experiments (9.8 vs 9.11 bug)
│   ├── experiments/           # Head ablation, bandwidth analysis
│   ├── repro/                 # Reproduction scripts
│   └── pythia_clustering/     # Cross-model validation
├── src/experiments/           # Follow-up experiments
│   ├── 01-07_*                # Linear probes for sprintf security
│   ├── 01-08_*                # SR/SCG separation experiments
│   └── 01-12_*                # Cross-domain steering (CWE-787)
├── docs/
│   ├── research_journal.md    # Detailed experiment logs
│   ├── experiments/           # Per-experiment reports
│   └── DATA_INVENTORY.md      # Dataset documentation
└── README.md
```

## Experiments Summary

### 1. Original 9.8 Research (`9_8_research/`)
Mechanistic analysis of the format-dependent numerical comparison bug:
- Head ablation experiments identifying even/odd specialization
- Bandwidth analysis of attention patterns
- Cross-model validation with Pythia
- SAE feature analysis

### 2. Security Recognition vs Secure Code Generation (`01-08_*`)
Testing whether LLMs encode "recognizing security context" and "generating secure code" as separate features:
- **Function stub prompts**: SR and SCG are orthogonal (cosine sim = 0.026)
- **Full task prompts**: SR and SCG are aligned (cosine sim = 0.899)
- **Conclusion**: Separation depends on prompt structure

### 3. CWE-787 Dataset Expansion (`01-12_cwe787_dataset_expansion/`)
LLM-based augmentation of security prompt pairs:
- 7 base templates → 105 pairs via GPT-4o augmentation
- 87.6% behavioral separation maintained
- Used for steering experiments

### 4. Cross-Domain Steering (`01-12_cwe787_cross_domain_steering/`) ⭐
**Key positive result**: Activation steering improves code security

| Metric | Baseline | Steered (L31, α=3.0) |
|--------|----------|----------------------|
| Secure code | 0% | **66.7%** |
| Insecure code | 90.5% | 19.0% |
| **Conversion** | - | **+66.7 pp** |

*Results validated on held-out test set (21 pairs) with direction computed from separate train set (84 pairs)*

## Quick Start

```bash
# Clone the repo
git clone https://github.com/GusSand/MATS.git
cd MATS

# Set up environment
python -m venv env
source env/bin/activate
pip install -r requirements.txt

# Run the validated steering experiment
cd src/experiments/01-12_cwe787_cross_domain_steering
python 06_validated_experiment.py
```

## Key Files

| File | Description |
|------|-------------|
| `docs/research_journal.md` | Complete experiment log with results |
| `docs/DATA_INVENTORY.md` | All datasets documented |
| `src/experiments/01-12_cwe787_cross_domain_steering/06_validated_experiment.py` | Validated steering experiment |
| `9_8_research/experiments/` | Original paper experiments |

## Citation

If you use this work, please cite:

```bibtex
@article{evenheads2025,
  title={Even Heads Fix Odd Errors},
  author={...},
  journal={arXiv preprint arXiv:2508.19414},
  year={2025}
}
```

## License

[Add license information]

## Acknowledgments

This research was conducted as part of the MATS (ML Alignment Theory Scholars) program.
