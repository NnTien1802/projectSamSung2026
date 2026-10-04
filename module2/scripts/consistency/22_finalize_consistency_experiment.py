from pathlib import Path
import json

import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# PATHS
# ============================================================

MODULE_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    MODULE_ROOT
    / "data"
    / "processed"
    / "consistency"
    / "headline_consistency_v1"
)

TRAIN_PATH = (
    DATASET_DIR
    / "train.parquet"
)

VALIDATION_PATH = (
    DATASET_DIR
    / "validation.parquet"
)

TEST_PATH = (
    DATASET_DIR
    / "test.parquet"
)


# ------------------------------------------------------------
# Baseline A
# ------------------------------------------------------------

A_DIR = (
    MODULE_ROOT
    / "results"
    / "consistency"
    / "baseline_title_description_mbert"
)

A_METRICS_PATH = (
    A_DIR
    / "validation_metrics.json"
)


# ------------------------------------------------------------
# Baseline B
# ------------------------------------------------------------

B_DIR = (
    MODULE_ROOT
    / "results"
    / "consistency"
    / "baseline_title_article_head_mbert"
)

B_METRICS_PATH = (
    B_DIR
    / "validation_metrics.json"
)


# ------------------------------------------------------------
# Full article MAX-MIL
# ------------------------------------------------------------

C_DIR = (
    MODULE_ROOT
    / "results"
    / "consistency"
    / "full_article_max_mil_mbert"
)

C_METRICS_PATH = (
    C_DIR
    / "validation_metrics.json"
)


# ------------------------------------------------------------
# Validation fusion
# ------------------------------------------------------------

FUSION_DIR = (
    MODULE_ROOT
    / "results"
    / "consistency"
    / "description_mil_fusion_v1"
)

FUSION_REPORT_PATH = (
    FUSION_DIR
    / "fusion_report.json"
)


# ------------------------------------------------------------
# Final official model
# ------------------------------------------------------------

FINAL_DIR = (
    MODULE_ROOT
    / "results"
    / "consistency"
    / "final_hybrid_v1"
)

FINAL_LOCK_PATH = (
    FINAL_DIR
    / "final_model_lock.json"
)

OFFICIAL_TEST_METRICS_PATH = (
    FINAL_DIR
    / "official_test_metrics.json"
)

OFFICIAL_CATEGORY_PATH = (
    FINAL_DIR
    / "official_test_by_category.csv"
)

OFFICIAL_TIER_PATH = (
    FINAL_DIR
    / "official_test_negative_by_tier.csv"
)


# ------------------------------------------------------------
# Model checkpoints
# ------------------------------------------------------------

A_MODEL_PATH = (
    MODULE_ROOT
    / "models"
    / "consistency"
    / "baseline_title_description_mbert"
    / "best_model.pt"
)

B_MODEL_PATH = (
    MODULE_ROOT
    / "models"
    / "consistency"
    / "baseline_title_article_head_mbert"
    / "best_model.pt"
)

C_MODEL_PATH = (
    MODULE_ROOT
    / "models"
    / "consistency"
    / "full_article_max_mil_mbert"
    / "best_model.pt"
)


# ------------------------------------------------------------
# Step 22 outputs
# ------------------------------------------------------------

OUTPUT_DIR = (
    FINAL_DIR
    / "experiment_summary"
)

SUMMARY_JSON_PATH = (
    OUTPUT_DIR
    / "final_experiment_summary.json"
)

MODEL_COMPARISON_PATH = (
    OUTPUT_DIR
    / "model_comparison.csv"
)

DATASET_SUMMARY_PATH = (
    OUTPUT_DIR
    / "dataset_summary.csv"
)

EXPERIMENT_CARD_PATH = (
    OUTPUT_DIR
    / "FINAL_EXPERIMENT_CARD.md"
)


# ============================================================
# HELPERS
# ============================================================

def load_json(path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def save_json(
    data,
    path,
):

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def relative(path):

    return str(
        path.relative_to(
            MODULE_ROOT
        )
    )


def assert_exists(
    path,
):

    if not path.exists():

        raise FileNotFoundError(
            path
        )


# ============================================================
# DATASET SUMMARY
# ============================================================

def summarize_split(
    split,
    path,
):

    table = pq.read_table(
        path,
        columns=[
            "pair_id",
            "anchor_article_uid",
            "headline_article_uid",
            "label",
        ],
    )

    frame = table.to_pandas()

    labels = (
        frame[
            "label"
        ]
        .value_counts()
    )

    return {
        "split":
            split,

        "pairs":
            int(
                len(
                    frame
                )
            ),

        "anchors":
            int(
                frame[
                    "anchor_article_uid"
                ]
                .nunique()
            ),

        "headline_articles":
            int(
                frame[
                    "headline_article_uid"
                ]
                .nunique()
            ),

        "label_0":
            int(
                labels.get(
                    0,
                    0,
                )
            ),

        "label_1":
            int(
                labels.get(
                    1,
                    0,
                )
            ),

        "duplicate_pair_ids":
            int(
                frame[
                    "pair_id"
                ]
                .duplicated()
                .sum()
            ),
    }


# ============================================================
# VALIDATION MODEL INFO
# ============================================================

def extract_validation_result(
    name,
    architecture,
    metrics_path,
):

    data = load_json(
        metrics_path
    )

    metrics_05 = (
        data[
            "metrics_at_0_5"
        ]
    )

    tuned = (
        data[
            "metrics_at_tuned_threshold"
        ]
    )

    return {
        "model":
            name,

        "architecture":
            architecture,

        "split":
            "validation",

        "macro_f1_at_0_5":
            float(
                metrics_05[
                    "macro_f1"
                ]
            ),

        "accuracy_at_0_5":
            float(
                metrics_05[
                    "accuracy"
                ]
            ),

        "roc_auc":
            (
                float(
                    metrics_05[
                        "roc_auc"
                    ]
                )
                if metrics_05.get(
                    "roc_auc"
                )
                is not None
                else None
            ),

        "tuned_threshold":
            float(
                data[
                    "best_validation_threshold"
                ]
            ),

        "macro_f1_tuned":
            float(
                tuned[
                    "macro_f1"
                ]
            ),

        "accuracy_tuned":
            float(
                tuned[
                    "accuracy"
                ]
            ),

        "test_used":
            bool(
                data.get(
                    "test_used",
                    False,
                )
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "MODULE 2 - HEADLINE INTEGRITY"
    )
    print(
        "TASK 2B - HEADLINE CONSISTENCY"
    )
    print(
        "STEP 22 - FINALIZE EXPERIMENT"
    )
    print("=" * 80)

    print()
    print(
        "DOCUMENTATION / CONSOLIDATION ONLY."
    )

    print(
        "NO TRAINING."
    )

    print(
        "NO INFERENCE."
    )

    print(
        "NO TEST TUNING."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # FILE VALIDATION
    # ========================================================

    required = [
        TRAIN_PATH,
        VALIDATION_PATH,
        TEST_PATH,
        A_METRICS_PATH,
        B_METRICS_PATH,
        C_METRICS_PATH,
        FUSION_REPORT_PATH,
        FINAL_LOCK_PATH,
        OFFICIAL_TEST_METRICS_PATH,
        OFFICIAL_CATEGORY_PATH,
        OFFICIAL_TIER_PATH,
        A_MODEL_PATH,
        B_MODEL_PATH,
        C_MODEL_PATH,
    ]

    for path in required:

        assert_exists(
            path
        )

    # ========================================================
    # DATASET
    # ========================================================

    print()
    print("=" * 80)
    print(
        "DATASET SUMMARY"
    )
    print("=" * 80)

    dataset_rows = [
        summarize_split(
            "train",
            TRAIN_PATH,
        ),

        summarize_split(
            "validation",
            VALIDATION_PATH,
        ),

        summarize_split(
            "test",
            TEST_PATH,
        ),
    ]

    dataset_df = pd.DataFrame(
        dataset_rows
    )

    dataset_df.to_csv(
        DATASET_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    for row in dataset_rows:

        print(
            f"{row['split']:12s}: "
            f"{row['pairs']:,} pairs | "
            f"{row['anchors']:,} anchors | "
            f"label0={row['label_0']:,} | "
            f"label1={row['label_1']:,}"
        )

    total_pairs = sum(
        row[
            "pairs"
        ]
        for row in dataset_rows
    )

    total_anchors = sum(
        row[
            "anchors"
        ]
        for row in dataset_rows
    )

    # ========================================================
    # MODEL RESULTS
    # ========================================================

    print()
    print("=" * 80)
    print(
        "MODEL RESULTS"
    )
    print("=" * 80)

    baseline_a = (
        extract_validation_result(
            name=(
                "Baseline A"
            ),
            architecture=(
                "headline + description"
            ),
            metrics_path=(
                A_METRICS_PATH
            ),
        )
    )

    baseline_b = (
        extract_validation_result(
            name=(
                "Baseline B"
            ),
            architecture=(
                "headline + article head"
            ),
            metrics_path=(
                B_METRICS_PATH
            ),
        )
    )

    max_mil = (
        extract_validation_result(
            name=(
                "MAX-MIL"
            ),
            architecture=(
                "headline + full article chunks"
            ),
            metrics_path=(
                C_METRICS_PATH
            ),
        )
    )

    model_rows = [
        baseline_a,
        baseline_b,
        max_mil,
    ]

    model_df = pd.DataFrame(
        model_rows
    )

    model_df.to_csv(
        MODEL_COMPARISON_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    for row in model_rows:

        print(
            f"{row['model']:12s}: "
            f"Macro-F1 tuned="
            f"{row['macro_f1_tuned']:.6f} | "
            f"threshold="
            f"{row['tuned_threshold']:.2f}"
        )

    # ========================================================
    # FUSION VALIDATION
    # ========================================================

    fusion_report = load_json(
        FUSION_REPORT_PATH
    )

    fusion_calibration = (
        fusion_report[
            "calibration"
        ]
    )

    fusion_holdout = (
        fusion_report[
            "holdout"
        ]
    )

    validation_fusion_macro = float(
        fusion_holdout[
            "fusion"
        ][
            "macro_f1"
        ]
    )

    validation_description_macro = float(
        fusion_holdout[
            "description"
        ][
            "macro_f1"
        ]
    )

    validation_mil_macro = float(
        fusion_holdout[
            "mil"
        ][
            "macro_f1"
        ]
    )

    # ========================================================
    # FINAL MODEL LOCK
    # ========================================================

    lock = load_json(
        FINAL_LOCK_PATH
    )

    description_weight = float(
        lock[
            "description_weight"
        ]
    )

    mil_weight = float(
        lock[
            "mil_weight"
        ]
    )

    final_threshold = float(
        lock[
            "decision_threshold"
        ]
    )

    if (
        abs(
            description_weight
            -
            0.50
        )
        >
        1e-12
    ):

        raise RuntimeError(
            "Unexpected final "
            "description weight."
        )

    if (
        abs(
            mil_weight
            -
            0.50
        )
        >
        1e-12
    ):

        raise RuntimeError(
            "Unexpected final MIL weight."
        )

    if (
        abs(
            final_threshold
            -
            0.45
        )
        >
        1e-12
    ):

        raise RuntimeError(
            "Unexpected final threshold."
        )

    # ========================================================
    # OFFICIAL TEST
    # ========================================================

    official = load_json(
        OFFICIAL_TEST_METRICS_PATH
    )

    if (
        official.get(
            "test_tuning_performed"
        )
        is not False
    ):

        raise RuntimeError(
            "Official TEST report does not "
            "confirm test_tuning_performed=False."
        )

    if (
        official.get(
            "configuration_frozen_before_test"
        )
        is not True
    ):

        raise RuntimeError(
            "Official TEST report does not "
            "confirm frozen configuration."
        )

    official_metrics = (
        official[
            "official_final_metrics"
        ]
    )

    # ========================================================
    # TEST CATEGORY / TIER
    # ========================================================

    category_df = pd.read_csv(
        OFFICIAL_CATEGORY_PATH
    )

    tier_df = pd.read_csv(
        OFFICIAL_TIER_PATH
    )

    lowest_categories = (
        category_df
        .sort_values(
            "macro_f1"
        )
        .head(5)
        .to_dict(
            orient="records"
        )
    )

    tier_records = (
        tier_df
        .to_dict(
            orient="records"
        )
    )

    # ========================================================
    # FINAL SUMMARY JSON
    # ========================================================

    summary = {
        "project":
            "projectSamSung",

        "module":
            "Module 2 - Headline Integrity",

        "task":
            (
                "Task 2B - "
                "Headline-Article Consistency"
            ),

        "experiment_version":
            "headline_consistency_v1",

        "status":
            "COMPLETED",

        "dataset": {
            "train":
                dataset_rows[
                    0
                ],

            "validation":
                dataset_rows[
                    1
                ],

            "test":
                dataset_rows[
                    2
                ],

            "total_pairs":
                int(
                    total_pairs
                ),

            "total_anchors":
                int(
                    total_anchors
                ),

            "construction": {
                "positive":
                    (
                        "Original article title paired "
                        "with its own article."
                    ),

                "negative":
                    (
                        "Synthetic same-category hard "
                        "negative headline."
                    ),

                "negative_policy": {
                    "PRIMARY":
                        (
                            "candidate rank 4-10, "
                            "0.20 <= lexical similarity < 0.40"
                        ),

                    "FALLBACK":
                        (
                            "candidate rank 4-10, "
                            "0.15 <= lexical similarity < 0.20"
                        ),
                },
            },
        },

        "validation_models": {
            "baseline_a":
                baseline_a,

            "baseline_b":
                baseline_b,

            "max_mil":
                max_mil,
        },

        "validation_fusion": {
            "protocol":
                (
                    "Validation split divided by "
                    "anchor_article_uid into calibration "
                    "and holdout subsets."
                ),

            "description_macro_f1":
                validation_description_macro,

            "mil_macro_f1":
                validation_mil_macro,

            "fusion_macro_f1":
                validation_fusion_macro,

            "description_weight":
                float(
                    fusion_calibration[
                        "fusion_description_weight"
                    ]
                ),

            "mil_weight":
                float(
                    fusion_calibration[
                        "fusion_mil_weight"
                    ]
                ),

            "threshold":
                float(
                    fusion_calibration[
                        "fusion_threshold"
                    ]
                ),
        },

        "final_model": {
            "architecture":
                (
                    "Probability fusion of "
                    "description mBERT and "
                    "full-article MAX-MIL mBERT."
                ),

            "description_weight":
                description_weight,

            "mil_weight":
                mil_weight,

            "decision_threshold":
                final_threshold,

            "description_checkpoint":
                relative(
                    A_MODEL_PATH
                ),

            "mil_checkpoint":
                relative(
                    C_MODEL_PATH
                ),

            "configuration_lock":
                relative(
                    FINAL_LOCK_PATH
                ),
        },

        "official_test": {
            "pairs":
                int(
                    official[
                        "test_pairs"
                    ]
                ),

            "anchors":
                int(
                    official[
                        "test_anchors"
                    ]
                ),

            "accuracy":
                float(
                    official_metrics[
                        "accuracy"
                    ]
                ),

            "precision_consistent":
                float(
                    official_metrics[
                        "precision_consistent"
                    ]
                ),

            "recall_consistent":
                float(
                    official_metrics[
                        "recall_consistent"
                    ]
                ),

            "f1_consistent":
                float(
                    official_metrics[
                        "f1_consistent"
                    ]
                ),

            "macro_f1":
                float(
                    official_metrics[
                        "macro_f1"
                    ]
                ),

            "roc_auc":
                float(
                    official_metrics[
                        "roc_auc"
                    ]
                ),

            "average_precision":
                float(
                    official_metrics[
                        "average_precision"
                    ]
                ),

            "confusion_matrix":
                official_metrics[
                    "confusion_matrix"
                ],

            "configuration_frozen_before_test":
                True,

            "test_tuning_performed":
                False,

            "negative_tiers":
                tier_records,

            "lowest_5_categories":
                lowest_categories,
        },

        "methodological_limitations": [
            (
                "Original positive pairs are "
                "presumed consistent rather than "
                "individually human-verified."
            ),

            (
                "Negative pairs are synthetic hard "
                "negatives rather than naturally "
                "occurring misleading headlines."
            ),

            (
                "The official test set is an "
                "independent split of the same "
                "synthetic-negative construction "
                "pipeline, not a human-gold "
                "real-world benchmark."
            ),

            (
                "Manual candidate validation covered "
                "a limited sample; false-negative "
                "synthetic pairs may still exist."
            ),

            (
                "Reported official TEST performance "
                "must not be interpreted as direct "
                "real-world deployment accuracy."
            ),
        ],

        "recommended_next_evaluation": {
            "name":
                "human_verified_consistency_eval",

            "purpose":
                (
                    "Measure transfer from synthetic "
                    "evaluation to human-verified "
                    "headline/article consistency."
                ),

            "suggested_size":
                "Several hundred to approximately 1,000 pairs",
        },

        "artifacts": {
            "official_test_metrics":
                relative(
                    OFFICIAL_TEST_METRICS_PATH
                ),

            "official_test_predictions":
                relative(
                    FINAL_DIR
                    / "official_test_predictions.parquet"
                ),

            "official_test_categories":
                relative(
                    OFFICIAL_CATEGORY_PATH
                ),

            "official_test_negative_tiers":
                relative(
                    OFFICIAL_TIER_PATH
                ),
        },
    }

    save_json(
        summary,
        SUMMARY_JSON_PATH,
    )

    # ========================================================
    # EXPERIMENT CARD
    # ========================================================

    cm = (
        official_metrics[
            "confusion_matrix"
        ]
    )

    card = f"""# Task 2B — Headline–Article Consistency

## Status

**COMPLETED — headline_consistency_v1**

The final architecture and threshold were frozen before official TEST evaluation.

No TEST-based tuning was performed.

---

## Dataset

| Split | Anchors | Pairs | Negative | Positive |
|---|---:|---:|---:|---:|
| Train | {dataset_rows[0]['anchors']:,} | {dataset_rows[0]['pairs']:,} | {dataset_rows[0]['label_0']:,} | {dataset_rows[0]['label_1']:,} |
| Validation | {dataset_rows[1]['anchors']:,} | {dataset_rows[1]['pairs']:,} | {dataset_rows[1]['label_0']:,} | {dataset_rows[1]['label_1']:,} |
| Test | {dataset_rows[2]['anchors']:,} | {dataset_rows[2]['pairs']:,} | {dataset_rows[2]['label_0']:,} | {dataset_rows[2]['label_1']:,} |

Total pairs: **{total_pairs:,}**

---

## Validation models

| Model | Input | Tuned threshold | Macro-F1 |
|---|---|---:|---:|
| Baseline A | Headline + description | {baseline_a['tuned_threshold']:.2f} | {baseline_a['macro_f1_tuned']:.6f} |
| Baseline B | Headline + article head | {baseline_b['tuned_threshold']:.2f} | {baseline_b['macro_f1_tuned']:.6f} |
| MAX-MIL | Headline + full article chunks | {max_mil['tuned_threshold']:.2f} | {max_mil['macro_f1_tuned']:.6f} |

---

## Final architecture

Final prediction:

```text
P(final)
    = {description_weight:.2f} × P(description)
    + {mil_weight:.2f} × P(MAX-MIL)
```

Decision threshold: **{final_threshold:.2f}**

## Official TEST results

- Accuracy: {official_metrics['accuracy']:.6f}
- Macro-F1: {official_metrics['macro_f1']:.6f}
- Confusion matrix: {cm}

This evaluation uses synthetic negatives, not a human-verified real-world benchmark.
"""

    EXPERIMENT_CARD_PATH.write_text(
        card,
        encoding="utf-8",
    )

    print()
    print(f"Experiment summary: {OUTPUT_DIR}")
    print("STEP 22 FINISHED SUCCESSFULLY")


if __name__ == "__main__":
    main()
