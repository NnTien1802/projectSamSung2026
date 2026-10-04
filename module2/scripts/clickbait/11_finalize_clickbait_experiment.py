from pathlib import Path
import json

import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# PATHS
# ============================================================

MODULE_ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------

DATA_DIR = (
    MODULE_ROOT
    / "data"
    / "processed"
    / "clickbait"
    / "viclickbait_v1"
)

TRAIN_PATH = (
    DATA_DIR
    / "train.parquet"
)

VALIDATION_PATH = (
    DATA_DIR
    / "validation.parquet"
)

TEST_PATH = (
    DATA_DIR
    / "test.parquet"
)


# ------------------------------------------------------------
# STEP 05 - DIAGNOSTIC BASELINES
# ------------------------------------------------------------

DIAGNOSTIC_DIR = (
    MODULE_ROOT
    / "results"
    / "clickbait"
    / "diagnostic_baselines_v1"
)

DIAGNOSTIC_REPORT_PATH = (
    DIAGNOSTIC_DIR
    / "diagnostic_report.json"
)


# ------------------------------------------------------------
# STEP 06 - MBERT TITLE ONLY
# ------------------------------------------------------------

TITLE_DIR = (
    MODULE_ROOT
    / "results"
    / "clickbait"
    / "mbert_title_only_v1"
)

TITLE_METRICS_PATH = (
    TITLE_DIR
    / "validation_metrics.json"
)


# ------------------------------------------------------------
# STEP 07 - MBERT TITLE + LEAD
# ------------------------------------------------------------

TITLE_LEAD_DIR = (
    MODULE_ROOT
    / "results"
    / "clickbait"
    / "mbert_title_lead_v1"
)

TITLE_LEAD_METRICS_PATH = (
    TITLE_LEAD_DIR
    / "validation_metrics.json"
)


# ------------------------------------------------------------
# STEP 08 - CONTROLLED COMPARISON
# ------------------------------------------------------------

COMPARE_DIR = (
    MODULE_ROOT
    / "results"
    / "clickbait"
    / "compare_title_vs_title_lead_v1"
)

COMPARE_REPORT_PATH = (
    COMPARE_DIR
    / "comparison_report.json"
)


# ------------------------------------------------------------
# STEP 09 / 10 - FINAL MODEL
# ------------------------------------------------------------

FINAL_DIR = (
    MODULE_ROOT
    / "results"
    / "clickbait"
    / "final_title_only_v1"
)

LOCK_PATH = (
    FINAL_DIR
    / "final_model_lock.json"
)

OFFICIAL_METRICS_PATH = (
    FINAL_DIR
    / "official_test_metrics.json"
)

OFFICIAL_SOURCE_PATH = (
    FINAL_DIR
    / "official_test_by_source.csv"
)

OFFICIAL_CATEGORY_PATH = (
    FINAL_DIR
    / "official_test_by_category.csv"
)

OFFICIAL_PREDICTIONS_PATH = (
    FINAL_DIR
    / "official_test_predictions.parquet"
)


# ------------------------------------------------------------
# FINAL CHECKPOINT
# ------------------------------------------------------------

FINAL_CHECKPOINT_PATH = (
    MODULE_ROOT
    / "models"
    / "clickbait"
    / "mbert_title_only_v1"
    / "best_model.pt"
)


# ------------------------------------------------------------
# STEP 11 OUTPUTS
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
# EXPECTED FINAL CONFIG
# ============================================================

EXPECTED_TOTAL_ROWS = 3414

EXPECTED_MODEL = (
    "bert-base-multilingual-cased"
)

EXPECTED_INPUT = (
    "title_only"
)

EXPECTED_MAX_LENGTH = 160

EXPECTED_THRESHOLD = 0.61

EXPECTED_BEST_EPOCH = 4


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


def assert_exists(path):

    if not path.exists():

        raise FileNotFoundError(
            path
        )


# ============================================================
# DATASET SUMMARY
# ============================================================

def summarize_split(
    split_name,
    path,
):

    frame = (
        pq.read_table(
            path,
            columns=[
                "sample_id",
                "title",
                "lead",
                "source",
                "category",
                "label",
            ],
        )
        .to_pandas()
    )

    if (
        frame[
            "sample_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            f"{split_name}: duplicate sample_id."
        )

    labels = set(
        frame[
            "label"
        ].unique()
    )

    if not labels.issubset(
        {
            0,
            1,
        }
    ):

        raise RuntimeError(
            f"{split_name}: invalid labels "
            f"{labels}"
        )

    counts = (
        frame[
            "label"
        ]
        .value_counts()
    )

    normal = int(
        counts.get(
            0,
            0,
        )
    )

    clickbait = int(
        counts.get(
            1,
            0,
        )
    )

    rows = int(
        len(
            frame
        )
    )

    return {
        "split":
            split_name,

        "rows":
            rows,

        "normal":
            normal,

        "clickbait":
            clickbait,

        "clickbait_rate":
            float(
                clickbait
                /
                rows
            ),

        "empty_title":
            int(
                frame[
                    "title"
                ]
                .fillna("")
                .eq("")
                .sum()
            ),

        "empty_lead":
            int(
                frame[
                    "lead"
                ]
                .fillna("")
                .eq("")
                .sum()
            ),

        "unique_sources":
            int(
                frame[
                    "source"
                ]
                .fillna("")
                .nunique()
            ),

        "unique_categories":
            int(
                frame[
                    "category"
                ]
                .fillna("")
                .nunique()
            ),

        "duplicate_sample_ids":
            int(
                frame[
                    "sample_id"
                ]
                .duplicated()
                .sum()
            ),
    }


# ============================================================
# TRANSFORMER RESULT SUMMARY
# ============================================================

def extract_mbert_result(
    name,
    input_name,
    path,
):

    data = load_json(
        path
    )

    if (
        data.get(
            "test_used"
        )
        is not False
    ):

        raise RuntimeError(
            f"{name}: test_used is not False."
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

        "input":
            input_name,

        "split":
            "validation",

        "best_epoch":
            int(
                data[
                    "best_epoch"
                ]
            ),

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

        "precision_clickbait_at_0_5":
            float(
                metrics_05[
                    "precision_clickbait"
                ]
            ),

        "recall_clickbait_at_0_5":
            float(
                metrics_05[
                    "recall_clickbait"
                ]
            ),

        "f1_clickbait_at_0_5":
            float(
                metrics_05[
                    "f1_clickbait"
                ]
            ),

        "roc_auc":
            float(
                metrics_05[
                    "roc_auc"
                ]
            ),

        "average_precision":
            float(
                metrics_05[
                    "average_precision"
                ]
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

        "precision_clickbait_tuned":
            float(
                tuned[
                    "precision_clickbait"
                ]
            ),

        "recall_clickbait_tuned":
            float(
                tuned[
                    "recall_clickbait"
                ]
            ),

        "f1_clickbait_tuned":
            float(
                tuned[
                    "f1_clickbait"
                ]
            ),

        "test_used":
            False,
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
        "TASK 2A - CLICKBAIT DETECTION"
    )
    print(
        "STEP 11 - FINALIZE CLICKBAIT EXPERIMENT"
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
    # FILE CHECK
    # ========================================================

    required_files = [
        TRAIN_PATH,
        VALIDATION_PATH,
        TEST_PATH,
        DIAGNOSTIC_REPORT_PATH,
        TITLE_METRICS_PATH,
        TITLE_LEAD_METRICS_PATH,
        COMPARE_REPORT_PATH,
        LOCK_PATH,
        OFFICIAL_METRICS_PATH,
        OFFICIAL_SOURCE_PATH,
        OFFICIAL_CATEGORY_PATH,
        OFFICIAL_PREDICTIONS_PATH,
        FINAL_CHECKPOINT_PATH,
    ]

    for path in required_files:

        assert_exists(
            path
        )

    # ========================================================
    # DATASET SUMMARY
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

    total_rows = sum(
        row[
            "rows"
        ]
        for row in dataset_rows
    )

    if (
        total_rows
        !=
        EXPECTED_TOTAL_ROWS
    ):

        raise RuntimeError(
            f"Unexpected dataset size: "
            f"{total_rows:,}"
        )

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
            f"{row['rows']:,} rows | "
            f"NORMAL={row['normal']:,} | "
            f"CLICKBAIT={row['clickbait']:,} | "
            f"rate={row['clickbait_rate']:.4f}"
        )

    # ========================================================
    # DIAGNOSTIC BASELINES
    # ========================================================

    diagnostic = load_json(
        DIAGNOSTIC_REPORT_PATH
    )

    if (
        diagnostic.get(
            "test_loaded"
        )
        is not False
    ):

        raise RuntimeError(
            "Diagnostic Step 05 does not confirm "
            "test_loaded=False."
        )

    diagnostic_map = {
        row[
            "model"
        ]:
            row

        for row in diagnostic[
            "baselines"
        ]
    }

    expected_diagnostics = [
        "majority",
        "source_only",
        "source_category",
        "surface_features",
        "tfidf_title",
        "tfidf_title_lead",
    ]

    for name in expected_diagnostics:

        if name not in diagnostic_map:

            raise RuntimeError(
                f"Missing diagnostic baseline: "
                f"{name}"
            )

    # ========================================================
    # TRANSFORMER RESULTS
    # ========================================================

    title_model = extract_mbert_result(
        name="mBERT title-only",
        input_name="title",
        path=TITLE_METRICS_PATH,
    )

    title_lead_model = extract_mbert_result(
        name="mBERT title+lead",
        input_name="title + lead",
        path=TITLE_LEAD_METRICS_PATH,
    )

    # ========================================================
    # MODEL COMPARISON CSV
    # ========================================================

    model_rows = []

    for name in expected_diagnostics:

        row = diagnostic_map[
            name
        ]

        model_rows.append(
            {
                "model":
                    name,

                "type":
                    "diagnostic",

                "input":
                    name,

                "split":
                    "validation",

                "macro_f1_at_0_5":
                    float(
                        row[
                            "macro_f1_05"
                        ]
                    ),

                "macro_f1_tuned":
                    float(
                        row[
                            "macro_f1_tuned"
                        ]
                    ),

                "tuned_threshold":
                    float(
                        row[
                            "tuned_threshold"
                        ]
                    ),

                "roc_auc":
                    float(
                        row[
                            "roc_auc"
                        ]
                    ),

                "average_precision":
                    float(
                        row[
                            "average_precision"
                        ]
                    ),
            }
        )

    model_rows.append(
        {
            "model":
                title_model[
                    "model"
                ],

            "type":
                "transformer",

            "input":
                title_model[
                    "input"
                ],

            "split":
                "validation",

            "macro_f1_at_0_5":
                title_model[
                    "macro_f1_at_0_5"
                ],

            "macro_f1_tuned":
                title_model[
                    "macro_f1_tuned"
                ],

            "tuned_threshold":
                title_model[
                    "tuned_threshold"
                ],

            "roc_auc":
                title_model[
                    "roc_auc"
                ],

            "average_precision":
                title_model[
                    "average_precision"
                ],
        }
    )

    model_rows.append(
        {
            "model":
                title_lead_model[
                    "model"
                ],

            "type":
                "transformer",

            "input":
                title_lead_model[
                    "input"
                ],

            "split":
                "validation",

            "macro_f1_at_0_5":
                title_lead_model[
                    "macro_f1_at_0_5"
                ],

            "macro_f1_tuned":
                title_lead_model[
                    "macro_f1_tuned"
                ],

            "tuned_threshold":
                title_lead_model[
                    "tuned_threshold"
                ],

            "roc_auc":
                title_lead_model[
                    "roc_auc"
                ],

            "average_precision":
                title_lead_model[
                    "average_precision"
                ],
        }
    )

    model_df = pd.DataFrame(
        model_rows
    )

    model_df.to_csv(
        MODEL_COMPARISON_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # STEP 08 COMPARISON
    # ========================================================

    comparison = load_json(
        COMPARE_REPORT_PATH
    )

    if (
        comparison.get(
            "test_used"
        )
        is not False
    ):

        raise RuntimeError(
            "Step 08 does not confirm "
            "test_used=False."
        )

    macro_delta = float(
        comparison[
            "deltas_b_minus_a"
        ][
            "macro_f1_tuned"
        ]
    )

    mcnemar_p = float(
        comparison[
            "mcnemar_exact"
        ][
            "two_sided_p_value"
        ]
    )

    bootstrap_info = (
        comparison[
            "paired_bootstrap"
        ]
    )

    bootstrap_low = float(
        bootstrap_info[
            "ci95_low"
        ]
    )

    bootstrap_high = float(
        bootstrap_info[
            "ci95_high"
        ]
    )

    bootstrap_probability_b_better = float(
        bootstrap_info[
            "fraction_delta_greater_than_zero"
        ]
    )

    # ========================================================
    # FINAL LOCK
    # ========================================================

    lock = load_json(
        LOCK_PATH
    )

    if (
        lock.get(
            "status"
        )
        !=
        "LOCKED_BEFORE_TEST"
    ):

        raise RuntimeError(
            "Final configuration was not "
            "locked before TEST."
        )

    if (
        lock.get(
            "test_opened"
        )
        is not False
    ):

        raise RuntimeError(
            "Step 09 lock does not confirm "
            "test_opened=False."
        )

    if (
        lock.get(
            "test_metrics_computed"
        )
        is not False
    ):

        raise RuntimeError(
            "Step 09 lock does not confirm "
            "test_metrics_computed=False."
        )

    final_model = (
        lock[
            "final_model"
        ]
    )

    architecture = (
        final_model[
            "architecture"
        ]
    )

    final_input = (
        final_model[
            "input"
        ]
    )

    max_length = int(
        final_model[
            "max_length"
        ]
    )

    best_epoch = int(
        final_model[
            "best_epoch"
        ]
    )

    threshold = float(
        final_model[
            "decision_threshold"
        ]
    )

    if (
        architecture
        !=
        EXPECTED_MODEL
    ):

        raise RuntimeError(
            f"Unexpected final architecture: "
            f"{architecture}"
        )

    if (
        final_input
        !=
        EXPECTED_INPUT
    ):

        raise RuntimeError(
            f"Unexpected final input: "
            f"{final_input}"
        )

    if (
        max_length
        !=
        EXPECTED_MAX_LENGTH
    ):

        raise RuntimeError(
            f"Unexpected max_length: "
            f"{max_length}"
        )

    if (
        best_epoch
        !=
        EXPECTED_BEST_EPOCH
    ):

        raise RuntimeError(
            f"Unexpected best epoch: "
            f"{best_epoch}"
        )

    if (
        abs(
            threshold
            -
            EXPECTED_THRESHOLD
        )
        >
        1e-12
    ):

        raise RuntimeError(
            f"Unexpected threshold: "
            f"{threshold}"
        )

    # ========================================================
    # OFFICIAL TEST
    # ========================================================

    official = load_json(
        OFFICIAL_METRICS_PATH
    )

    if (
        official.get(
            "evaluation_status"
        )
        !=
        "OFFICIAL_TEST_COMPLETE"
    ):

        raise RuntimeError(
            "Official TEST evaluation is not complete."
        )

    if (
        official.get(
            "configuration_frozen_before_test"
        )
        is not True
    ):

        raise RuntimeError(
            "Official TEST report does not confirm "
            "configuration_frozen_before_test=True."
        )

    if (
        official.get(
            "test_tuning_performed"
        )
        is not False
    ):

        raise RuntimeError(
            "Official TEST report does not confirm "
            "test_tuning_performed=False."
        )

    official_config = (
        official[
            "frozen_configuration"
        ]
    )

    if (
        official_config[
            "architecture"
        ]
        !=
        architecture
    ):

        raise RuntimeError(
            "Official TEST architecture differs "
            "from final lock."
        )

    if (
        official_config[
            "input"
        ]
        !=
        final_input
    ):

        raise RuntimeError(
            "Official TEST input differs "
            "from final lock."
        )

    if (
        int(
            official_config[
                "max_length"
            ]
        )
        !=
        max_length
    ):

        raise RuntimeError(
            "Official TEST max_length differs "
            "from final lock."
        )

    if (
        abs(
            float(
                official_config[
                    "decision_threshold"
                ]
            )
            -
            threshold
        )
        >
        1e-12
    ):

        raise RuntimeError(
            "Official TEST threshold differs "
            "from final lock."
        )

    if (
        official_config[
            "checkpoint_sha256"
        ]
        !=
        final_model[
            "checkpoint_sha256"
        ]
    ):

        raise RuntimeError(
            "Official TEST checkpoint hash differs "
            "from final lock."
        )

    official_metrics = (
        official[
            "official_final_metrics"
        ]
    )

    # ========================================================
    # SUBGROUP FILES
    # ========================================================

    source_df = pd.read_csv(
        OFFICIAL_SOURCE_PATH
    )

    category_df = pd.read_csv(
        OFFICIAL_CATEGORY_PATH
    )

    if source_df.empty:

        lowest_sources = []

    else:

        lowest_sources = (
            source_df
            .sort_values(
                [
                    "macro_f1",
                    "n",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
            .head(5)
            .to_dict(
                orient="records"
            )
        )

    if category_df.empty:

        lowest_categories = []

    else:

        lowest_categories = (
            category_df
            .sort_values(
                [
                    "macro_f1",
                    "n",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
            .head(5)
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
            "Task 2A - Clickbait Detection",

        "dataset":
            "viclickbait_v1",

        "status":
            "COMPLETED",

        "dataset_summary": {
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

            "total_rows":
                int(
                    total_rows
                ),

            "split_policy":
                {
                    "train":
                        0.70,

                    "validation":
                        0.15,

                    "test":
                        0.15,

                    "stratified_by":
                        "label",

                    "grouped_by":
                        "normalized title",
                },
        },

        "diagnostic_baselines": {
            name:
                diagnostic_map[
                    name
                ]

            for name in (
                expected_diagnostics
            )
        },

        "transformer_validation": {
            "title_only":
                title_model,

            "title_plus_lead":
                title_lead_model,
        },

        "controlled_comparison": {
            "title_lead_minus_title_macro_f1":
                macro_delta,

            "mcnemar_exact_p_value":
                mcnemar_p,

            "paired_bootstrap_macro_f1_delta_ci95":
                [
                    bootstrap_low,
                    bootstrap_high,
                ],

            "bootstrap_fraction_delta_greater_than_zero":
                bootstrap_probability_b_better,

            "conclusion":
                (
                    "Title+lead did not show reliable "
                    "overall Macro-F1 superiority over "
                    "title-only on validation. "
                    "The simpler title-only model was "
                    "therefore selected before TEST."
                ),
        },

        "final_model": {
            "architecture":
                architecture,

            "input":
                final_input,

            "max_length":
                max_length,

            "best_epoch":
                best_epoch,

            "decision_threshold":
                threshold,

            "checkpoint":
                final_model[
                    "checkpoint"
                ],

            "checkpoint_sha256":
                final_model[
                    "checkpoint_sha256"
                ],
        },

        "official_test": {
            "samples":
                int(
                    official[
                        "test_samples"
                    ]
                ),

            "normal":
                int(
                    official[
                        "test_class_distribution"
                    ][
                        "normal"
                    ]
                ),

            "clickbait":
                int(
                    official[
                        "test_class_distribution"
                    ][
                        "clickbait"
                    ]
                ),

            "clickbait_rate":
                float(
                    official[
                        "test_class_distribution"
                    ][
                        "clickbait_rate"
                    ]
                ),

            "accuracy":
                float(
                    official_metrics[
                        "accuracy"
                    ]
                ),

            "precision_clickbait":
                float(
                    official_metrics[
                        "precision_clickbait"
                    ]
                ),

            "recall_clickbait":
                float(
                    official_metrics[
                        "recall_clickbait"
                    ]
                ),

            "f1_clickbait":
                float(
                    official_metrics[
                        "f1_clickbait"
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

            "lowest_5_sources":
                lowest_sources,

            "lowest_5_categories":
                lowest_categories,
        },

        "methodological_limitations": [
            (
                "The dataset exhibits publisher/source bias. "
                "A source-only diagnostic classifier achieved "
                "non-trivial validation performance."
            ),

            (
                "Surface characteristics such as quotation "
                "marks and question marks are associated "
                "with clickbait labels in the training data."
            ),

            (
                "Source and category metadata are not used "
                "as inputs to the final production classifier."
            ),

            (
                "The dataset contains only 3,414 examples, "
                "so subgroup estimates can be unstable."
            ),

            (
                "Some source/category subgroups in TEST "
                "contain relatively few examples; their "
                "individual Macro-F1 scores should be "
                "interpreted cautiously."
            ),

            (
                "The official TEST split is independent "
                "under the normalized-title grouped split, "
                "but comes from the same ViClickbait dataset "
                "collection as TRAIN and VALIDATION."
            ),
        ],

        "artifacts": {
            "model_lock":
                relative(
                    LOCK_PATH
                ),

            "checkpoint":
                relative(
                    FINAL_CHECKPOINT_PATH
                ),

            "official_test_metrics":
                relative(
                    OFFICIAL_METRICS_PATH
                ),

            "official_test_predictions":
                relative(
                    OFFICIAL_PREDICTIONS_PATH
                ),

            "official_test_by_source":
                relative(
                    OFFICIAL_SOURCE_PATH
                ),

            "official_test_by_category":
                relative(
                    OFFICIAL_CATEGORY_PATH
                ),

            "model_comparison":
                relative(
                    MODEL_COMPARISON_PATH
                ),

            "dataset_summary":
                relative(
                    DATASET_SUMMARY_PATH
                ),

            "experiment_card":
                relative(
                    EXPERIMENT_CARD_PATH
                ),
        },
    }

    save_json(
        summary,
        SUMMARY_JSON_PATH,
    )

    # ========================================================
    # EXPERIMENT CARD
    #
    # Deliberately constructed line-by-line.
    # No triple-quoted string is used here.
    # ========================================================

    cm = (
        official_metrics[
            "confusion_matrix"
        ]
    )

    card_lines = [
        "# Task 2A — Clickbait Detection",
        "",
        "## Status",
        "",
        "**COMPLETED — viclickbait_v1**",
        "",
        (
            "Final model configuration was frozen "
            "before official TEST evaluation."
        ),
        "",
        "No TEST-based tuning was performed.",
        "",
        "---",
        "",
        "## Dataset",
        "",
        (
            "| Split | Rows | NORMAL | CLICKBAIT | "
            "Clickbait rate |"
        ),
        "|---|---:|---:|---:|---:|",
        (
            f"| Train | "
            f"{dataset_rows[0]['rows']:,} | "
            f"{dataset_rows[0]['normal']:,} | "
            f"{dataset_rows[0]['clickbait']:,} | "
            f"{dataset_rows[0]['clickbait_rate']:.4f} |"
        ),
        (
            f"| Validation | "
            f"{dataset_rows[1]['rows']:,} | "
            f"{dataset_rows[1]['normal']:,} | "
            f"{dataset_rows[1]['clickbait']:,} | "
            f"{dataset_rows[1]['clickbait_rate']:.4f} |"
        ),
        (
            f"| Test | "
            f"{dataset_rows[2]['rows']:,} | "
            f"{dataset_rows[2]['normal']:,} | "
            f"{dataset_rows[2]['clickbait']:,} | "
            f"{dataset_rows[2]['clickbait_rate']:.4f} |"
        ),
        "",
        f"Total rows: **{total_rows:,}**",
        "",
        "Split strategy:",
        "",
        "    70% train",
        "    15% validation",
        "    15% test",
        "",
        "    stratified by label",
        "    grouped by normalized title",
        "",
        "---",
        "",
        "## Diagnostic baselines",
        "",
        "| Model | Tuned Macro-F1 | ROC-AUC |",
        "|---|---:|---:|",
        (
            f"| Majority | "
            f"{diagnostic_map['majority']['macro_f1_tuned']:.6f} | "
            f"{diagnostic_map['majority']['roc_auc']:.6f} |"
        ),
        (
            f"| Source only | "
            f"{diagnostic_map['source_only']['macro_f1_tuned']:.6f} | "
            f"{diagnostic_map['source_only']['roc_auc']:.6f} |"
        ),
        (
            f"| Source + category | "
            f"{diagnostic_map['source_category']['macro_f1_tuned']:.6f} | "
            f"{diagnostic_map['source_category']['roc_auc']:.6f} |"
        ),
        (
            f"| Surface features | "
            f"{diagnostic_map['surface_features']['macro_f1_tuned']:.6f} | "
            f"{diagnostic_map['surface_features']['roc_auc']:.6f} |"
        ),
        (
            f"| TF-IDF title | "
            f"{diagnostic_map['tfidf_title']['macro_f1_tuned']:.6f} | "
            f"{diagnostic_map['tfidf_title']['roc_auc']:.6f} |"
        ),
        (
            f"| TF-IDF title + lead | "
            f"{diagnostic_map['tfidf_title_lead']['macro_f1_tuned']:.6f} | "
            f"{diagnostic_map['tfidf_title_lead']['roc_auc']:.6f} |"
        ),
        "",
        "---",
        "",
        "## Transformer comparison",
        "",
        (
            "| Model | Threshold | Validation Macro-F1 | "
            "ROC-AUC | AP |"
        ),
        "|---|---:|---:|---:|---:|",
        (
            f"| mBERT title-only | "
            f"{title_model['tuned_threshold']:.2f} | "
            f"{title_model['macro_f1_tuned']:.6f} | "
            f"{title_model['roc_auc']:.6f} | "
            f"{title_model['average_precision']:.6f} |"
        ),
        (
            f"| mBERT title+lead | "
            f"{title_lead_model['tuned_threshold']:.2f} | "
            f"{title_lead_model['macro_f1_tuned']:.6f} | "
            f"{title_lead_model['roc_auc']:.6f} | "
            f"{title_lead_model['average_precision']:.6f} |"
        ),
        "",
        "Title+lead minus title-only:",
        "",
        (
            f"    Macro-F1 delta = "
            f"{macro_delta:+.6f}"
        ),
        (
            f"    McNemar p      = "
            f"{mcnemar_p:.6f}"
        ),
        (
            f"    Bootstrap CI   = "
            f"[{bootstrap_low:+.6f}, "
            f"{bootstrap_high:+.6f}]"
        ),
        (
            f"    P(delta > 0)   = "
            f"{bootstrap_probability_b_better:.4f}"
        ),
        "",
        (
            "The validation comparison did not show "
            "reliable superiority of title+lead. "
            "The simpler title-only architecture was "
            "therefore retained before TEST evaluation."
        ),
        "",
        "---",
        "",
        "## Final model",
        "",
        (
            f"    Architecture : "
            f"{architecture}"
        ),
        (
            f"    Input        : "
            f"{final_input}"
        ),
        (
            f"    Max length   : "
            f"{max_length}"
        ),
        (
            f"    Best epoch   : "
            f"{best_epoch}"
        ),
        (
            f"    Threshold    : "
            f"{threshold:.2f}"
        ),
        "",
        "Checkpoint:",
        "",
        (
            f"    "
            f"{final_model['checkpoint']}"
        ),
        "",
        "Checkpoint SHA256:",
        "",
        (
            f"    "
            f"{final_model['checkpoint_sha256']}"
        ),
        "",
        "---",
        "",
        "## Official TEST result",
        "",
        "| Metric | Score |",
        "|---|---:|",
        (
            f"| Accuracy | "
            f"{official_metrics['accuracy']:.6f} |"
        ),
        (
            f"| Precision — CLICKBAIT | "
            f"{official_metrics['precision_clickbait']:.6f} |"
        ),
        (
            f"| Recall — CLICKBAIT | "
            f"{official_metrics['recall_clickbait']:.6f} |"
        ),
        (
            f"| F1 — CLICKBAIT | "
            f"{official_metrics['f1_clickbait']:.6f} |"
        ),
        (
            f"| Macro-F1 | "
            f"**{official_metrics['macro_f1']:.6f}** |"
        ),
        (
            f"| ROC-AUC | "
            f"{official_metrics['roc_auc']:.6f} |"
        ),
        (
            f"| Average Precision | "
            f"{official_metrics['average_precision']:.6f} |"
        ),
        "",
        "Confusion matrix:",
        "",
        f"    TN = {cm['tn']:,}",
        f"    FP = {cm['fp']:,}",
        f"    FN = {cm['fn']:,}",
        f"    TP = {cm['tp']:,}",
        "",
        "---",
        "",
        "## Lowest TEST sources",
        "",
    ]

    if lowest_sources:

        for row in lowest_sources:

            card_lines.append(
                (
                    f"- **{row['source']}**: "
                    f"Macro-F1="
                    f"{float(row['macro_f1']):.4f}, "
                    f"n={int(row['n']):,}"
                )
            )

    else:

        card_lines.append(
            "No eligible source groups."
        )

    card_lines.extend(
        [
            "",
            "---",
            "",
            "## Lowest TEST categories",
            "",
        ]
    )

    if lowest_categories:

        for row in lowest_categories:

            card_lines.append(
                (
                    f"- **{row['category']}**: "
                    f"Macro-F1="
                    f"{float(row['macro_f1']):.4f}, "
                    f"n={int(row['n']):,}"
                )
            )

    else:

        card_lines.append(
            "No eligible category groups."
        )

    card_lines.extend(
        [
            "",
            "---",
            "",
            "## Methodological notes",
            "",
            (
                "1. Source-only and surface-feature "
                "baselines show that publisher/style "
                "shortcuts exist in the dataset."
            ),
            (
                "2. Source and category metadata are not "
                "used as inputs to the final classifier."
            ),
            (
                "3. Title-only was selected because "
                "title+lead did not show reliable overall "
                "Macro-F1 improvement on validation."
            ),
            (
                "4. Small source/category subgroup results "
                "should be interpreted cautiously."
            ),
            (
                "5. The TEST split was isolated from model "
                "development until architecture, checkpoint "
                "and threshold had been frozen."
            ),
            (
                f"6. Official TEST evaluation used the "
                f"frozen decision threshold of "
                f"{threshold:.2f}."
            ),
            (
                "7. No TEST-based threshold, checkpoint, "
                "architecture or input selection was performed."
            ),
            "",
            "---",
            "",
            "## Final official score",
            "",
            (
                f"    Macro-F1 = "
                f"{official_metrics['macro_f1']:.6f}"
            ),
            "",
        ]
    )

    card = "\n".join(
        card_lines
    )

    with EXPERIMENT_CARD_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            card
        )

    # ========================================================
    # PRINT FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print(
        "FINAL EXPERIMENT SUMMARY"
    )
    print("=" * 80)

    print()
    print(
        f"Dataset rows: "
        f"{total_rows:,}"
    )

    print()
    print(
        "Diagnostic validation:"
    )

    print(
        f"  Majority           : "
        f"{diagnostic_map['majority']['macro_f1_tuned']:.6f}"
    )

    print(
        f"  Source only        : "
        f"{diagnostic_map['source_only']['macro_f1_tuned']:.6f}"
    )

    print(
        f"  Surface features   : "
        f"{diagnostic_map['surface_features']['macro_f1_tuned']:.6f}"
    )

    print(
        f"  TF-IDF title       : "
        f"{diagnostic_map['tfidf_title']['macro_f1_tuned']:.6f}"
    )

    print(
        f"  TF-IDF title+lead  : "
        f"{diagnostic_map['tfidf_title_lead']['macro_f1_tuned']:.6f}"
    )

    print()
    print(
        "Transformer validation:"
    )

    print(
        f"  mBERT title-only   : "
        f"{title_model['macro_f1_tuned']:.6f}"
    )

    print(
        f"  mBERT title+lead   : "
        f"{title_lead_model['macro_f1_tuned']:.6f}"
    )

    print(
        f"  delta lead-title   : "
        f"{macro_delta:+.6f}"
    )

    print()
    print(
        "Final model:"
    )

    print(
        f"  architecture : "
        f"{architecture}"
    )

    print(
        f"  input        : "
        f"{final_input}"
    )

    print(
        f"  max length   : "
        f"{max_length}"
    )

    print(
        f"  best epoch   : "
        f"{best_epoch}"
    )

    print(
        f"  threshold    : "
        f"{threshold:.2f}"
    )

    print()
    print(
        "Official TEST:"
    )

    print(
        f"  Accuracy        : "
        f"{official_metrics['accuracy']:.6f}"
    )

    print(
        f"  Precision CLICK : "
        f"{official_metrics['precision_clickbait']:.6f}"
    )

    print(
        f"  Recall CLICK    : "
        f"{official_metrics['recall_clickbait']:.6f}"
    )

    print(
        f"  F1 CLICK        : "
        f"{official_metrics['f1_clickbait']:.6f}"
    )

    print(
        f"  Macro-F1        : "
        f"{official_metrics['macro_f1']:.6f}"
    )

    print(
        f"  ROC-AUC         : "
        f"{official_metrics['roc_auc']:.6f}"
    )

    print(
        f"  AP              : "
        f"{official_metrics['average_precision']:.6f}"
    )

    print()
    print(
        "Integrity checks:"
    )

    print(
        "  configuration frozen before TEST : YES"
    )

    print(
        "  TEST-based tuning performed       : NO"
    )

    print(
        "  final architecture verified       : YES"
    )

    print(
        "  final threshold verified          : YES"
    )

    print(
        "  checkpoint SHA256 verified by "
        "official TEST run               : YES"
    )

    print()
    print(
        "Outputs:"
    )

    print(
        f"  Summary JSON:\n"
        f"    {SUMMARY_JSON_PATH}"
    )

    print(
        f"  Model comparison:\n"
        f"    {MODEL_COMPARISON_PATH}"
    )

    print(
        f"  Dataset summary:\n"
        f"    {DATASET_SUMMARY_PATH}"
    )

    print(
        f"  Experiment card:\n"
        f"    {EXPERIMENT_CARD_PATH}"
    )

    print()
    print(
        "TASK 2A STATUS: COMPLETED"
    )

    print()
    print(
        "STEP 11 FINISHED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()