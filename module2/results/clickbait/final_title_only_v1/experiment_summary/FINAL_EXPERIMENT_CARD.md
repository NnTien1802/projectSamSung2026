# Task 2A — Clickbait Detection

## Status

**COMPLETED — viclickbait_v1**

Final model configuration was frozen before official TEST evaluation.

No TEST-based tuning was performed.

---

## Dataset

| Split | Rows | NORMAL | CLICKBAIT | Clickbait rate |
|---|---:|---:|---:|---:|
| Train | 2,390 | 1,645 | 745 | 0.3117 |
| Validation | 512 | 352 | 160 | 0.3125 |
| Test | 512 | 352 | 160 | 0.3125 |

Total rows: **3,414**

Split strategy:

    70% train
    15% validation
    15% test

    stratified by label
    grouped by normalized title

---

## Diagnostic baselines

| Model | Tuned Macro-F1 | ROC-AUC |
|---|---:|---:|
| Majority | 0.407407 | 0.500000 |
| Source only | 0.627947 | 0.684721 |
| Source + category | 0.653469 | 0.711044 |
| Surface features | 0.715284 | 0.779403 |
| TF-IDF title | 0.741861 | 0.823881 |
| TF-IDF title + lead | 0.756958 | 0.856818 |

---

## Transformer comparison

| Model | Threshold | Validation Macro-F1 | ROC-AUC | AP |
|---|---:|---:|---:|---:|
| mBERT title-only | 0.61 | 0.792567 | 0.883256 | 0.755475 |
| mBERT title+lead | 0.23 | 0.792993 | 0.889675 | 0.764046 |

Title+lead minus title-only:

    Macro-F1 delta = +0.000426
    McNemar p      = 0.899076
    Bootstrap CI   = [-0.034690, +0.035357]
    P(delta > 0)   = 0.5191

The validation comparison did not show reliable superiority of title+lead. The simpler title-only architecture was therefore retained before TEST evaluation.

---

## Final model

    Architecture : bert-base-multilingual-cased
    Input        : title_only
    Max length   : 160
    Best epoch   : 4
    Threshold    : 0.61

Checkpoint:

    models\clickbait\mbert_title_only_v1\best_model.pt

Checkpoint SHA256:

    6c10d84f012cf05a0917e35abee4d8f006688315ab0570a3739bebea737efdf7

---

## Official TEST result

| Metric | Score |
|---|---:|
| Accuracy | 0.832031 |
| Precision — CLICKBAIT | 0.756944 |
| Recall — CLICKBAIT | 0.681250 |
| F1 — CLICKBAIT | 0.717105 |
| Macro-F1 | **0.798830** |
| ROC-AUC | 0.867418 |
| Average Precision | 0.802853 |

Confusion matrix:

    TN = 317
    FP = 35
    FN = 51
    TP = 109

---

## Lowest TEST sources

- **VnExpress**: Macro-F1=0.6883, n=101
- **Thanh Niên**: Macro-F1=0.7347, n=92
- **Tuổi Trẻ**: Macro-F1=0.7537, n=42
- **VietNamNet**: Macro-F1=0.7832, n=51
- **Báo Mới**: Macro-F1=0.8021, n=94

---

## Lowest TEST categories

- **Văn hóa & Nghệ thuật**: Macro-F1=0.4762, n=11
- **Kinh doanh & Kinh tế**: Macro-F1=0.6454, n=27
- **Giáo dục & Học đường**: Macro-F1=0.6667, n=27
- **Thể thao**: Macro-F1=0.6942, n=53
- **Tin tức tổng hợp**: Macro-F1=0.7315, n=29

---

## Methodological notes

1. Source-only and surface-feature baselines show that publisher/style shortcuts exist in the dataset.
2. Source and category metadata are not used as inputs to the final classifier.
3. Title-only was selected because title+lead did not show reliable overall Macro-F1 improvement on validation.
4. Small source/category subgroup results should be interpreted cautiously.
5. The TEST split was isolated from model development until architecture, checkpoint and threshold had been frozen.
6. Official TEST evaluation used the frozen decision threshold of 0.61.
7. No TEST-based threshold, checkpoint, architecture or input selection was performed.

---

## Final official score

    Macro-F1 = 0.798830
