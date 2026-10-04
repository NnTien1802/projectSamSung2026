# Unified Module 2 — Headline Integrity

## Status

**LOCKED — unified_module2_v1**

The unified inference pipeline passed regression checks against saved validation artifacts before being locked.

---

## Task 2A — Clickbait

- Architecture: `bert-base-multilingual-cased`
- Input: `title_only`
- Max length: `160`
- Decision threshold: `0.61`
- Regression max absolute error: `0.002441406`
- Prediction mismatches: `0`

---

## Task 2B — Headline/Article Consistency

Description branch:

- Weight: `0.50`
- Max length: `160`
- Regression max absolute error: `0.000488281`

MAX-MIL branch:

- Weight: `0.50`
- Chunk tokens: `120`
- Chunk overlap: `24`
- Chunk step: `96`
- Maximum chunks: `6`
- Sampling: `np.round(np.linspace(0, len(starts)-1, 6))`
- Regression max absolute error: `0.000488281`
- Token-count mismatches: `0`
- Chunk-count mismatches: `0`

Fusion:

- Description weight: `0.50`
- MAX-MIL weight: `0.50`
- Decision threshold: `0.45`
- Regression max absolute error: `0.000335693`
- Prediction mismatches: `0`

---

## Unified regression

| Branch | Result | Max abs error |
|---|---:|---:|
| Clickbait | PASS | 0.002441406 |
| Description | PASS | 0.000488281 |
| MAX-MIL | PASS | 0.000488281 |
| Fusion | PASS | 0.000335693 |

Tolerance: `0.005000`

**Unified pipeline regression: PASS**

---

## Pipeline SHA256

`ac3137209f676c2e1e279a967ab642b20b05c8cc64a1c0edb741be052045ace9`

## Important design decision

The module does not produce a single overall integrity score. Clickbait detection and headline/article consistency remain separate model outputs.
