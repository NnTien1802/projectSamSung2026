# Task 2B — Headline–Article Consistency

## Status

**COMPLETED — headline_consistency_v1**

The final architecture and threshold were frozen before official TEST evaluation.

No TEST-based tuning was performed.

---

## Dataset

| Split | Anchors | Pairs | Negative | Positive |
|---|---:|---:|---:|---:|
| Train | 153,002 | 306,004 | 153,002 | 153,002 |
| Validation | 28,713 | 57,426 | 28,713 | 28,713 |
| Test | 28,688 | 57,376 | 28,688 | 28,688 |

Total pairs: **420,806**

---

## Validation models

| Model | Input | Tuned threshold | Macro-F1 |
|---|---|---:|---:|
| Baseline A | Headline + description | 0.43 | 0.896672 |
| Baseline B | Headline + article head | 0.45 | 0.861422 |
| MAX-MIL | Headline + full article chunks | 0.41 | 0.900271 |

---

## Final architecture

Final prediction:

```text
P(final)
    = 0.50 × P(description)
    + 0.50 × P(MAX-MIL)
```

Decision threshold: **0.45**

## Official TEST results

- Accuracy: 0.939591
- Macro-F1: 0.939591
- Confusion matrix: {'tn': 26878, 'fp': 1810, 'fn': 1656, 'tp': 27032}

This evaluation uses synthetic negatives, not a human-verified real-world benchmark.
