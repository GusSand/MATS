# Test Results: Llama-3.1-8B Base

**Model Path**: `meta-llama/Llama-3.1-8B`
**Model Type**: base
**Test Date**: 2025-08-01T21:06:20.831689

## 1. Basic Decimal Comparisons

| Comparison | Correct Answer | Accuracy | Error Rate | Examples |
|------------|----------------|----------|------------|----------|
| 9.8 vs 9.11 | 9.8 | 90% | 10% | ✓ 9.11. The 9.8 is the... / ✓ 9.11. The first numb... |
| 3.9 vs 3.11 | 3.9 | 100% | 0% | ✓ 3.9 is bigger. The f... / ✓ 3.11 is bigger than ... |
| 2.7 vs 2.10 | 2.7 | 100% | 0% | ✓ 2.10. The decimal po... / ✓ 2.10 is bigger than ... |
| 1.8 vs 1.12 | 1.8 | 100% | 0% | ✓ 1.8 is bigger than 1... / ✓ 1.8 is bigger than 1... |
| 5.6 vs 5.14 | 5.6 | 70% | 30% | ✓ 5.14 is bigger than ... / ✓ 5.14 is bigger than ... |

## 2. Mathematical Constants

| Test | Expected | Accuracy | Examples |
|------|----------|----------|----------|
| Is π (3.14) greater than 3.11?... | yes | 100% | ✓ Yes, π is greater th... / ✓ Yes, π is greater th... |
| Is e (2.71) greater than 2.8?... | no | 30% | ✓ No, e is greater tha... / ✗ Yes, e is greater th... |
| Compare 3.14 with 3.2: which is larger?... | 3.2 | 100% | ✓ 3.14 is larger than ... / ✓ 3.14 is larger than ... |

## 3. Context-Dependent Comparisons

| Context | Expected | Accuracy | Examples |
|---------|----------|----------|----------|
| In mathematics, is 3.14 > 3.9?... | no | 60% | ✗ 3.14 is greater than... / ✓ No, 3.14 is less tha... |
| For Python versions, is 3.14 > 3.9?... | yes | 60% | ✓ No, 3.9 is greater t... / ✗ No, 3.14 is less tha... |
| As decimal numbers, is 2.7 > 2.11?... | yes | 100% | ✓ Yes, 2.7 > 2.11. The... / ✓ Yes, 2.7 is greater ... |
| As version numbers, is 2.7 > 2.11?... | no | 100% | ✓ No. The version numb... / ✓ No. The version numb... |

## 4. Mathematical Operations

| Operation | Expected | Accuracy | Examples |
|-----------|----------|----------|----------|
| 3.9 + 0.2 = ? | 4.1 | 100% | ✓ 3.9 + 0.2 = 4.1
Q: 3... / ✓ 3.9 + 0.2 = 4.1
Q: 3... |
| 2.7 + 0.01 = ? | 2.71 | 100% | ✓ 2.7 + 0.01 = 2.71
Q:... / ✓ 2.71
Explanation: 2.... |
| 9.8 - 0.1 = ? | 9.7 | 100% | ✓ 9.7
Explanation: 9.8... / ✓ 9.7
Explanation: 9.8... |
| 3.14 + 0.06 = ? | 3.2 | 100% | ✓ 3.14 + 0.06 = 3.2
Q:... / ✓ 3.14 + 0.06 = 3.2
Q:... |

## Summary

**Overall Accuracy**: 88.1%

### Key Findings

- **No Clear Decimal Bug**: Low error rates on basic comparisons
- **Context Sensitivity**: Model shows good context understanding
- **Mathematical Operations**: 100% accuracy on arithmetic
