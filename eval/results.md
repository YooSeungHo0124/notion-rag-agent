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
