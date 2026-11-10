# Evaluation results

fused = hybrid RRF only, plain = after rerank.

## 20261110_120002 · baseline · n=48

| level | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| fused_chunk | 0.65 | 0.97 | 0.97 | 0.78 |
| fused_page | 0.93 | 1.00 | 1.00 | 0.96 |
| chunk | 0.80 | 0.97 | 1.00 | 0.87 |
| page | 1.00 | 1.00 | 1.00 | 1.00 |

- answer_accuracy: 0.90
- answer_correct_or_partial: 1.00
- false_abstain_rate: 0.00
- abstain_accuracy_unanswerable: 0.75
- mean_supported_ratio: 0.85
- mean_latency_s: 6.46

## 20261110_134222 · v2 · n=46

| level | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| fused_chunk | 0.65 | 0.95 | 0.97 | 0.78 |
| fused_page | 0.90 | 1.00 | 1.00 | 0.95 |
| chunk | 0.78 | 0.97 | 1.00 | 0.86 |
| page | 1.00 | 1.00 | 1.00 | 1.00 |

- answer_accuracy: 0.88
- answer_correct_or_partial: 0.97
- false_abstain_rate: 0.00
- abstain_accuracy_unanswerable: 1.00
- mean_supported_ratio: 0.83
- mean_latency_s: 6.98

## 20261110_134822 · v2-repeat (RAG, noise check) · n=46

| level | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| fused_chunk | 0.65 | 0.95 | 0.97 | 0.78 |
| fused_page | 0.90 | 1.00 | 1.00 | 0.95 |
| chunk | 0.78 | 0.97 | 1.00 | 0.86 |
| page | 1.00 | 1.00 | 1.00 | 1.00 |

- answer_accuracy: 0.88
- answer_correct_or_partial: 0.97
- false_abstain_rate: 0.00
- abstain_accuracy_unanswerable: 1.00
- mean_supported_ratio: 0.84
- mean_latency_s: 6.16

## 20261110_141225 · agent · n=46

| level | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| fused_chunk | 0.65 | 0.95 | 0.97 | 0.78 |
| fused_page | 0.90 | 1.00 | 1.00 | 0.95 |
| chunk | 0.78 | 0.97 | 1.00 | 0.86 |
| page | 1.00 | 1.00 | 1.00 | 1.00 |

- answer_accuracy: 0.88
- answer_correct_or_partial: 1.00
- false_abstain_rate: 0.00
- abstain_accuracy_unanswerable: 1.00
- mean_supported_ratio: 0.85
- mean_latency_s: 28.50
- mean_agent_steps: 1.80

## 20261110_143513 · v3-blocks · n=46

| level | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| fused_chunk | 0.65 | 0.95 | 0.97 | 0.78 |
| fused_page | 0.90 | 1.00 | 1.00 | 0.95 |
| chunk | 0.78 | 0.97 | 1.00 | 0.86 |
| page | 1.00 | 1.00 | 1.00 | 1.00 |

- answer_accuracy: 0.90
- answer_correct_or_partial: 0.97
- false_abstain_rate: 0.03
- abstain_accuracy_unanswerable: 1.00
- mean_supported_ratio: 0.85
- mean_latency_s: 5.98
