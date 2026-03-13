# Experiment 30: Expanded CodeQL/Static Analysis Validation

**Date**: 2026-03-13
**Models**: Llama-3.1-8B-Instruct outputs (from prior LOBO experiments)
**Dataset**: 30 CWE-119 + 30 CWE-89 sampled outputs
**Experiment**: Static analysis validation of regex scoring

## Research Question

Does the regex-based security scorer agree with static analysis tools across CWE-119 and CWE-89, expanding beyond the existing CWE-787 CodeQL validation?

## Methods

### CWE-119 (C, CodeQL)
- Sampled 30 outputs from Llama-8B CWE-119 LOBO results (10 secure, 10 insecure, 10 other)
- Wrapped as compilable C files with standard headers
- Compiled with gcc (17/30 compiled = 57%)
- Ran CodeQL v2.16.1 with cpp-queries v0.9.0 (7 buffer overflow queries)

### CWE-89 (Python, Bandit + Semgrep)
- Sampled 30 outputs from Llama-8B CWE-89 baseline (10 secure, 10 insecure, 10 other)
- Wrapped as Python files with sqlite3 import
- Ran Bandit v1.9.4 (B608 SQL injection rule)
- Ran Semgrep v1.155.0 (auto config: sqlalchemy + formatted-sql-query rules)

## Results

### CWE-119 CodeQL Agreement

| Regex Label | n | CodeQL Secure | CodeQL Insecure |
|-------------|---|---------------|-----------------|
| secure | 10 | 10 | 0 |
| insecure | 10 | 6 | 4 |
| other | 10 | 10 | 0 |

**Agreement rate (excluding other): 14/20 (70.0%)**

- CodeQL found 4 alerts total (all DangerousFunctionOverflow — flags gets/strcpy usage)
- Zero false positives: no regex-secure samples flagged by CodeQL
- 6 false negatives: CodeQL missed 6/10 regex-insecure samples

### CWE-89 Bandit Agreement

| Regex Label | n | Bandit Secure | Bandit Insecure |
|-------------|---|---------------|-----------------|
| secure | 10 | 9 | 1 |
| insecure | 10 | 6 | 4 |
| other | 10 | 10 | 0 |

**Agreement rate: 13/20 (65.0%)**

### CWE-89 Semgrep Agreement

| Regex Label | n | Semgrep Secure | Semgrep Insecure |
|-------------|---|----------------|------------------|
| secure | 10 | 10 | 0 |
| insecure | 10 | 7 | 3 |
| other | 10 | 10 | 0 |

**Agreement rate: 13/20 (65.0%)**

### CWE-89 Combined (Bandit OR Semgrep)

| Regex Label | n | Combined Secure | Combined Insecure |
|-------------|---|-----------------|-------------------|
| secure | 10 | 9 | 1 |
| insecure | 10 | 6 | 4 |
| other | 10 | 10 | 0 |

**Combined agreement: 13/20 (65.0%)**

### Cross-CWE Comparison (including prior CWE-787 from Appendix C)

| CWE | Tool | Agreement Rate | False Positives | False Negatives |
|-----|------|---------------|-----------------|-----------------|
| CWE-787 | CodeQL | ~90% (prior) | Low | Low |
| CWE-119 | CodeQL | 70.0% | 0/10 | 6/10 |
| CWE-89 | Bandit | 65.0% | 1/10 | 6/10 |
| CWE-89 | Semgrep | 65.0% | 0/10 | 7/10 |

## Key Observations

1. **Static analysis tools are conservative**: Very low false positive rates (0-1/10) but high false negative rates (6-7/10)
2. **CodeQL DangerousFunctionOverflow was most effective query for CWE-119**: The other 6 queries found 0 issues (they require richer dataflow context than LLM snippets provide)
3. **Semgrep `--config auto` required**: The `p/python` pack didn't include SQL injection rules; `auto` config includes sqlalchemy and formatted-sql-query rules
4. **Code extraction limitations**: 5/10 insecure CWE-89 samples had no extractable SQL code (model outputs were prose or non-SQL code)

## Code

- [01_cwe119_codeql_validation.py](../../src/experiments/03-13_expanded_codeql_validation/01_cwe119_codeql_validation.py) - CWE-119 CodeQL pipeline
- [02_cwe89_bandit_semgrep_validation.py](../../src/experiments/03-13_expanded_codeql_validation/02_cwe89_bandit_semgrep_validation.py) - CWE-89 Bandit/Semgrep pipeline

## Configuration
- CWE-119 results: `results/cwe119_codeql_agreement_20260313_160520.json`
- CWE-89 results: `results/cwe89_bandit_semgrep_agreement_20260313_160521.json`
- CodeQL binary: `/opt/codeql/codeql` v2.16.1
- Query pack: `codeql/cpp-queries@0.9.0`
- Runtime: ~5 minutes total (CWE-119 CodeQL + CWE-89 Bandit/Semgrep)
