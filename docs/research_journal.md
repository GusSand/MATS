# Research Journal

## 2026-01-07: sprintf vs snprintf Security Localization

### Prompt
> Run an experiment similar to the 9.8 vs 9.11 mechanistic analysis, but for security code. Have LLaMA-8B complete a C function and determine where the model decides to use sprintf (insecure) vs snprintf (secure).

### Research Question
Where in LLaMA-8B does the model decide to use insecure `sprintf` vs secure `snprintf`?

### Methods
- **Model**: meta-llama/Meta-Llama-3.1-8B-Instruct
- **Task**: C code completion for string formatting function
- **Measurement**: Logit probability shift for snprintf token
- **Technique**: Last-token activation patching across layers/heads

### Results (No Interpretation)

| Experiment | Result |
|------------|--------|
| Baseline without security context | 0% snprintf, 100% sprintf |
| Baseline with security warning | 100% snprintf, 0% sprintf |
| P(snprintf) gap | 33.9% (secure 37.1% vs neutral 3.2%) |
| Single layer patching (best L25) | 6.1% lift |
| All 32 layers last-token patching | 100% lift |
| Layers 16-31 | 94.7% lift |
| Layers 0-15 | 7.0% lift |
| All even heads (32 layers) | 46.1% lift |
| All odd heads (32 layers) | 28.4% lift |

### Interpretation (Claude's)
The security context ("use snprintf for buffer overflow prevention") is encoded as a **distributed representation** across all 32 layers, concentrated at the last token position. This is fundamentally different from the 9.8 decimal bug, which was localized to Layer 10 attention.

This suggests that high-level behavioral instructions (like "use secure code patterns") involve the entire model rather than specific circuits. This has implications for AI safety: complex behavioral properties may be harder to mechanistically interpret/edit than simple processing errors.

### Detailed Report
See: [docs/experiments/01-07_llama8b_security_sprintf_localization.md](experiments/01-07_llama8b_security_sprintf_localization.md)

### Code Location
`src/experiments/01-07_llama8b_sprintf_security/`

---
