# Test Results: Gemma-2-2B Base

**Model Path**: `google/gemma-2-2b`
**Model Type**: base
**Test Date**: 2025-08-01T20:58:24.574502

## 1. Basic Decimal Comparisons

| Comparison | Correct Answer | Accuracy | Error Rate | Examples |
|------------|----------------|----------|------------|----------|
| 9.8 vs 9.11 | 9.8 | 100% | 0% | ✓ 9.8

Q: Which is big... / ✓ 9.8

Q: What is the ... |
| 3.9 vs 3.11 | 3.9 | 60% | 40% | ✓ 3.11

Q: Which is bi... / ✓ 3.11

Q: Which is bi... |
| 2.7 vs 2.10 | 2.7 | 80% | 20% | ✗ 2.10

Q: Which is bi... / ✓ 2.7

Q: Which is big... |
| 1.8 vs 1.12 | 1.8 | 100% | 0% | ✓ 1.8

Q: Which is big... / ✓ 1.8

Q: What is the ... |
| 5.6 vs 5.14 | 5.6 | 40% | 60% | ✗ 5.14 is bigger

Q: W... / ✗ 5.14

Q: Which is bi... |

## 2. Mathematical Constants

| Test | Expected | Accuracy | Examples |
|------|----------|----------|----------|
| Is π (3.14) greater than 3.11?... | yes | 100% | ✓ Yes

Q: Is 3.14 grea... / ✓ No, π is greater tha... |
| Is e (2.71) greater than 2.8?... | no | 100% | ✓ Yes

Q: Is e (2.71) ... / ✓ Yes

Q: Is e (2.71) ... |
| Compare 3.14 with 3.2: which is larger?... | 3.2 | 0% | ✗ 3.14 is larger

Q: C... / ✗ 3.14 is larger

Q: C... |

## 3. Context-Dependent Comparisons

| Context | Expected | Accuracy | Examples |
|---------|----------|----------|----------|
| In mathematics, is 3.14 > 3.9?... | no | 100% | ✓ No, 3.14 is less tha... / ✓ No, 3.14 is less tha... |
| For Python versions, is 3.14 > 3.9?... | yes | 0% | ✗ No

Q: What is the p... / ✗ No.

Q: What is the ... |
| As decimal numbers, is 2.7 > 2.11?... | yes | 100% | ✓ Yes

Q: As decimal n... / ✓ Yes

Q: As decimal n... |
| As version numbers, is 2.7 > 2.11?... | no | 10% | ✗ Yes

Q: What is the ... / ✓ No.

Q: What is the ... |

## 4. Mathematical Operations

| Operation | Expected | Accuracy | Examples |
|-----------|----------|----------|----------|
| 3.9 + 0.2 = ? | 4.1 | 0% | ✗ 3.92

Q: 1.2 + 0.3 =... / ✗ 3.92

Q: 12.5 + 0.05... |
| 2.7 + 0.01 = ? | 2.71 | 100% | ✓ 2.71

Q: 2.7 + 0.01 ... / ✓ 2.71

Q: 2.7 + 0.01 ... |
| 9.8 - 0.1 = ? | 9.7 | 100% | ✓ 9.7

Q: 12.3 - 0.5 =... / ✓ 9.7

Q: 10.2 + 0.3 =... |
| 3.14 + 0.06 = ? | 3.2 | 0% | ✗ 3.146

Q: 3.14 + 0.0... / ✗ 3.146

Q: 2.5 + 0.05... |

## Summary

**Overall Accuracy**: 61.9%

### Key Findings

- **Decimal Bug Present**: High error rates on 5.6_vs_5.14
- **Context Sensitivity**: Model shows good context understanding
- **Mathematical Operations**: 50% accuracy on arithmetic
