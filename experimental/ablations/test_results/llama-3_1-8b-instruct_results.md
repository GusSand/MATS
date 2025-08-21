# Test Results: Llama-3.1-8B-Instruct

**Model Path**: `meta-llama/Llama-3.1-8B-Instruct`
**Model Type**: instruct
**Test Date**: 2025-08-01T21:10:13.543817

## 1. Basic Decimal Comparisons

| Comparison | Correct Answer | Accuracy | Error Rate | Examples |
|------------|----------------|----------|------------|----------|
| 9.8 vs 9.11 | 9.8 | 0% | 0% | ? ... / ? ... |
| 3.9 vs 3.11 | 3.9 | 0% | 0% | ? ... / ? ... |
| 2.7 vs 2.10 | 2.7 | 0% | 0% | ? ... / ? ... |
| 1.8 vs 1.12 | 1.8 | 0% | 0% | ? ... / ? ... |
| 5.6 vs 5.14 | 5.6 | 0% | 0% | ? ... / ? ... |

## 2. Mathematical Constants

| Test | Expected | Accuracy | Examples |
|------|----------|----------|----------|
| Is π (3.14) greater than 3.11?... | yes | 0% | ✗ ... / ✗ ... |
| Is e (2.71) greater than 2.8?... | no | 0% | ✗ ... / ✗ ... |
| Compare 3.14 with 3.2: which is larger?... | 3.2 | 20% | ✓ d decimal part of 3.... / ✓ n 3.2... |

## 3. Context-Dependent Comparisons

| Context | Expected | Accuracy | Examples |
|---------|----------|----------|----------|
| In mathematics, is 3.14 > 3.9?... | no | 0% | ✗ ... / ✗ ... |
| For Python versions, is 3.14 > 3.9?... | yes | 10% | ✗ rsion of Python, and... / ✗ numbers are strings,... |
| As decimal numbers, is 2.7 > 2.11?... | yes | 0% | ✗ mbers. Both numbers ... / ✗ 2.11 is also 2. Sinc... |
| As version numbers, is 2.7 > 2.11?... | no | 40% | ✗ mpare the major vers... / ✗ mpare the major vers... |

## 4. Mathematical Operations

| Operation | Expected | Accuracy | Examples |
|-----------|----------|----------|----------|
| 3.9 + 0.2 = ? | 4.1 | 0% | ✗ ... / ✗ ... |
| 2.7 + 0.01 = ? | 2.71 | 0% | ✗ ... / ✗ ... |
| 9.8 - 0.1 = ? | 9.7 | 0% | ✗ ... / ✗ ... |
| 3.14 + 0.06 = ? | 3.2 | 0% | ✗ ... / ✗ ... |

## Summary

**Overall Accuracy**: 4.4%

### Key Findings

- **No Clear Decimal Bug**: Low error rates on basic comparisons
- **Context Sensitivity**: Model shows poor context understanding
- **Mathematical Operations**: 0% accuracy on arithmetic
