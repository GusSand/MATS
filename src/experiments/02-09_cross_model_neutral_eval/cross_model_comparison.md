# Experiment 9: Cross-Model Neutral Evaluation Results

Generated: 20260207_231909

Models: Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2.5-14B-Instruct

## Results

Table 1: Neutral Prompt Security Rate (% secure generations)

| Model          | Condition          | CWE-787 | CWE-119 | CWE-134 | Avg   |
|----------------|-------------------|---------|---------|---------|-------|
| Llama-8B       | Neutral baseline  |  47.1%  |  65.0%  | 100.0%  |  70.7%|
| Llama-8B       | Neutral + steer   | 100.0%  |  81.4%  | 100.0%  |  93.8%|
| Mistral-7B     | Neutral baseline  |  75.7%  |  90.0%  | 100.0%  |  88.6%|
| Mistral-7B     | Neutral + steer   |  98.6%  |  75.7%  | 100.0%  |  91.4%|
| Qwen-14B       | Neutral baseline  |  78.6%  | 100.0%  | 100.0%  |  92.9%|
| Qwen-14B       | Neutral + steer   | 100.0%  |  81.4%  | 100.0%  |  93.8%|

Table 2: Instruction Resistance Gap (neutral_steered - adversarial_steered)

| Model          | CWE-787 | CWE-119 | CWE-134 | Avg Gap |
|----------------|---------|---------|---------|---------|
| Llama-8B       | +47.6pp | +61.4pp | +10.0pp | +39.7pp |
| Mistral-7B     |  +6.2pp |    --   |    --   |  +6.2pp |
| Qwen-14B       | +22.9pp |    --   |    --   | +22.9pp |

Table 3: Cross-CWE Interference Check
(Secure rate on OTHER CWEs' neutral prompts when steering is active)

| Model      | Steering | CWE-787 (other) | CWE-119 (other) | CWE-134 (other) |
|------------|----------|-----------------|-----------------|-----------------|
| Mistral-7B | CWE-787 |       N/A       |   83.3% (-7pp) |  100.0% (+0pp) |
| Mistral-7B | CWE-119 |   83.3% (+8pp) |       N/A       |  100.0% (+0pp) |
| Mistral-7B | CWE-134 |   70.0% (-6pp) |  100.0% (+10pp) |       N/A       |
| Qwen-14B   | CWE-787 |       N/A       |  100.0% (+0pp) |  100.0% (+0pp) |
| Qwen-14B   | CWE-119 |   80.0% (+1pp) |       N/A       |  100.0% (+0pp) |
| Qwen-14B   | CWE-134 |   70.0% (-9pp) |  100.0% (+0pp) |       N/A       |

## Best Steering Alphas

| Model | CWE-787 | CWE-119 | CWE-134 |
|-------|---------|---------|---------|
| Llama-8B | 3.5 | 4.0 | 1.5 |
| Mistral-7B | 3.5 | 3.0 | 3.0 |
| Qwen-14B | 4.0 | 3.0 | 3.0 |

## Hypothesis Evaluation

- **H1** (Instruction resistance gap is architecture-dependent): See Table 2
- **H2** (CWE-134 baselines high across models): Check baseline column in Table 1
- **H3** (CWE-119 hardest to steer): Check steered rates in Table 1
- **H4** (No cross-CWE interference): See Table 3