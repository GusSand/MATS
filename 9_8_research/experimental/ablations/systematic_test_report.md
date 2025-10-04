# Systematic Contradiction Test - Comprehensive Report

**Test Date**: 2025-08-01T21:31:19.841409

## Executive Summary

This test was conducted to understand contradictions in decimal comparison results.

### Model Performance Summary

| Model | Best Format | 9.8 vs 9.11 Accuracy | Overall Performance |
|-------|-------------|---------------------|--------------------|
| Gemma-2-2B Base | Q&A format | 100.0% | 56.2% |
| Gemma-2-2B-IT | Simple question | 100.0% | 70.0% |
| Llama-3.1-8B-Instruct | Q&A format | 100.0% | 87.5% |

## Detailed Results by Model

### Gemma-2-2B Base

#### 9.8 vs 9.11 - Format Comparison

| Format | Accuracy | Error Rate | Unclear Rate | Sample Response |
|--------|----------|------------|--------------|------------------|
| Simple question | 60.0% | 0.0% | 40.0% | 

[Answer 1]

The answer is 9.11.

The d... |
| Q&A format | 100.0% | 0.0% | 0.0% |  9.8

Q: Which is bigger: 10.1 or 10.01?... |
| With Answer label | 100.0% | 0.0% | 0.0% |  9.8 is bigger.

Which is bigger: 10.1 o... |

#### Comprehensive Test Results


**Basic Comparisons**

| Comparison | Expected | Accuracy |
|------------|----------|----------|
| 3.9 vs 3.11 | 3.9 | 0.0% |
| 2.7 vs 2.10 | 2.7 | 100.0% |
| 1.8 vs 1.12 | 1.8 | 80.0% |
| 5.6 vs 5.14 | 5.6 | 20.0% |
| 10.8 vs 10.11 | 10.8 | 20.0% |

**Mathematical Constants**

- Is π (3.14) greater than 3.11?: 100.0% (expected: yes)
- Is e (2.71) greater than 2.8?: 100.0% (expected: no)
- Which is larger: 3.14 or 3.2?: 0.0% (expected: 3.2)

**Context Dependent**

- In mathematics, is 3.14 > 3.9?: 100.0% (expected: no)
- For Python versions, is 3.14 > 3.9?: 20.0% (expected: yes)
- As decimal numbers, is 2.7 > 2.11?: 80.0% (expected: yes)
- As version numbers, is 2.7 > 2.11?: 60.0% (expected: no)

**Mathematical Operations**

- 3.9 + 0.2 = ?: 20.0% (expected: 4.1)
- 2.7 + 0.01 = ?: 100.0% (expected: 2.71)
- 9.8 - 0.1 = ?: 100.0% (expected: 9.7)
- 3.14 + 0.06 = ?: 0.0% (expected: 3.2)

### Gemma-2-2B-IT

#### 9.8 vs 9.11 - Format Comparison

| Format | Accuracy | Error Rate | Unclear Rate | Sample Response |
|--------|----------|------------|--------------|------------------|
| Simple question | 100.0% | 0.0% | 0.0% | 

Here's how to figure it out:

* **Comp... |
| Q&A format | 100.0% | 0.0% | 0.0% |  9.8 is bigger. 

**Explanation:**

* **... |
| With Answer label | 100.0% | 0.0% | 0.0% |  9.8 is bigger. 
... |
| Gemma chat format | 0.0% | 0.0% | 100.0% | ... |
| Gemma chat with BOS | 0.0% | 0.0% | 100.0% | ... |
| Tokenizer chat template | 0.0% | 0.0% | 100.0% | ... |

#### Comprehensive Test Results


**Basic Comparisons**

| Comparison | Expected | Accuracy |
|------------|----------|----------|
| 3.9 vs 3.11 | 3.9 | 100.0% |
| 2.7 vs 2.10 | 2.7 | 100.0% |
| 1.8 vs 1.12 | 1.8 | 100.0% |
| 5.6 vs 5.14 | 5.6 | 100.0% |
| 10.8 vs 10.11 | 10.8 | 100.0% |

**Mathematical Constants**

- Is π (3.14) greater than 3.11?: 60.0% (expected: yes)
- Is e (2.71) greater than 2.8?: 40.0% (expected: no)
- Which is larger: 3.14 or 3.2?: 100.0% (expected: 3.2)

**Context Dependent**

- In mathematics, is 3.14 > 3.9?: 100.0% (expected: no)
- For Python versions, is 3.14 > 3.9?: 0.0% (expected: yes)
- As decimal numbers, is 2.7 > 2.11?: 100.0% (expected: yes)
- As version numbers, is 2.7 > 2.11?: 0.0% (expected: no)

**Mathematical Operations**

- 3.9 + 0.2 = ?: 20.0% (expected: 4.1)
- 2.7 + 0.01 = ?: 100.0% (expected: 2.71)
- 9.8 - 0.1 = ?: 100.0% (expected: 9.7)
- 3.14 + 0.06 = ?: 0.0% (expected: 3.2)

### Llama-3.1-8B-Instruct

#### 9.8 vs 9.11 - Format Comparison

| Format | Accuracy | Error Rate | Unclear Rate | Sample Response |
|--------|----------|------------|--------------|------------------|
| Simple question | 80.0% | 0.0% | 20.0% | ?
9.11 is bigger than 9.8.
9.11 is bigge... |
| Q&A format | 100.0% | 0.0% | 0.0% |  9.11 is bigger than 9.8.
Q: Which is bi... |
| With Answer label | 100.0% | 0.0% | 0.0% |  9.8 is bigger.
Explanation: 9.8 is bigg... |
| Llama chat (ablation format) | 0.0% | 0.0% | 100.0% | ... |
| Llama chat (begin_of_text) | 0.0% | 0.0% | 100.0% | ... |
| Tokenizer chat template | 0.0% | 0.0% | 100.0% | ... |

#### Comprehensive Test Results


**Basic Comparisons**

| Comparison | Expected | Accuracy |
|------------|----------|----------|
| 3.9 vs 3.11 | 3.9 | 100.0% |
| 2.7 vs 2.10 | 2.7 | 100.0% |
| 1.8 vs 1.12 | 1.8 | 100.0% |
| 5.6 vs 5.14 | 5.6 | 100.0% |
| 10.8 vs 10.11 | 10.8 | 100.0% |

**Mathematical Constants**

- Is π (3.14) greater than 3.11?: 100.0% (expected: yes)
- Is e (2.71) greater than 2.8?: 0.0% (expected: no)
- Which is larger: 3.14 or 3.2?: 100.0% (expected: 3.2)

**Context Dependent**

- In mathematics, is 3.14 > 3.9?: 100.0% (expected: no)
- For Python versions, is 3.14 > 3.9?: 0.0% (expected: yes)
- As decimal numbers, is 2.7 > 2.11?: 100.0% (expected: yes)
- As version numbers, is 2.7 > 2.11?: 100.0% (expected: no)

**Mathematical Operations**

- 3.9 + 0.2 = ?: 100.0% (expected: 4.1)
- 2.7 + 0.01 = ?: 100.0% (expected: 2.71)
- 9.8 - 0.1 = ?: 100.0% (expected: 9.7)
- 3.14 + 0.06 = ?: 100.0% (expected: 3.2)

## Key Findings

### 1. Format Sensitivity

Accuracy range across different prompt formats:

- **Gemma-2-2B Base**: 60.0% - 100.0%
- **Gemma-2-2B-IT**: 0.0% - 100.0%
- **Llama-3.1-8B-Instruct**: 0.0% - 100.0%

### 2. Contradiction Analysis

Comparing with earlier results:

- **Gemma-2B Base**: Earlier showed 90% error, now shows variable performance depending on format
- **Instruction models**: Chat formats often produce empty/unclear responses
- **Format matters more than model type**: Same model can go from 0% to 100% accuracy

### 3. Recommendations

1. **Always test multiple prompt formats** when evaluating model capabilities
2. **Document exact prompt format** when reporting results
3. **Be cautious with chat templates** - they may not work as expected
4. **Simple formats often work best** for numerical comparisons
