# Experiment 3A Summary: Precision Steering Head-to-Head

**Timestamp**: 20260113_174901
**Model**: meta-llama/Meta-Llama-3.1-8B-Instruct
**Generations per prompt**: 3

## Methods Compared

| Method | Description |
|--------|-------------|
| M1 | Mean-diff at L31 |
| M2a | SAE L31:1895 (single feature) |
| M2b | SAE L30:10391 (single feature) |
| M3a | Top-5 SAE features |
| M3b | Top-10 SAE features |

## Aggregated Results (EXPANDED Scoring)

| Method | Setting | Secure% | Insecure% | Other% | Refusal% |
|--------|---------|---------|-----------|--------|----------|
| M1_mean_diff | 0.0 | 2.9% | 86.3% | 10.8% | 0.0% |
| M1_mean_diff | 0.5 | 3.8% | 82.9% | 13.3% | 0.0% |
| M1_mean_diff | 1.0 | 6.0% | 82.5% | 11.4% | 0.0% |
| M1_mean_diff | 1.5 | 9.8% | 82.2% | 7.9% | 0.0% |
| M1_mean_diff | 2.0 | 19.0% | 73.0% | 7.9% | 0.0% |
| M1_mean_diff | 2.5 | 27.9% | 63.8% | 8.3% | 0.0% |
| M1_mean_diff | 3.0 | 49.5% | 41.6% | 8.9% | 0.0% |
| M1_mean_diff | 3.5 | 53.0% | 26.3% | 20.6% | 0.0% |
| M2a_sae_L31_1895 | +1.0σ | 3.2% | 85.7% | 11.1% | 0.0% |
| M2a_sae_L31_1895 | +2.0σ | 1.3% | 85.4% | 13.3% | 0.0% |
| M2a_sae_L31_1895 | +3.0σ | 0.6% | 86.3% | 13.0% | 0.0% |
| M2b_sae_L30_10391 | +1.0σ | 1.3% | 87.9% | 10.8% | 0.0% |
| M2b_sae_L30_10391 | +2.0σ | 2.2% | 86.7% | 11.1% | 0.0% |
| M2b_sae_L30_10391 | +3.0σ | 2.2% | 84.8% | 13.0% | 0.0% |
| M3a_sae_top5 | +1.0σ | 3.2% | 86.3% | 10.5% | 0.0% |
| M3a_sae_top5 | +2.0σ | 1.0% | 90.2% | 8.9% | 0.0% |
| M3a_sae_top5 | +3.0σ | 1.9% | 84.4% | 13.7% | 0.0% |
| M3b_sae_top10 | +1.0σ | 1.6% | 85.7% | 12.7% | 0.0% |
| M3b_sae_top10 | +2.0σ | 1.9% | 88.6% | 9.5% | 0.0% |
| M3b_sae_top10 | +3.0σ | 2.2% | 87.6% | 10.2% | 0.0% |

## Best Operating Points (Other% ≤ 10%)

| Method | Setting | Secure% | Other% |
|--------|---------|---------|--------|
| M1_mean_diff | 3.0 | 49.5% | 8.9% |
| M3a_sae_top5 | +2.0σ | 1.0% | 8.9% |
| M3b_sae_top10 | +2.0σ | 1.9% | 9.5% |