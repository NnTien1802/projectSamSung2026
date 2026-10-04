from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# PATHS
# ============================================================

MODULE_ROOT = Path(__file__).resolve().parents[2]

PAIR_PATH = (
    MODULE_ROOT
    / "data"
    / "processed"
    / "consistency"
    / "headline_consistency_v1"
    / "validation.parquet"
)

RESULT_DIR = (
    MODULE_ROOT
    / "results"
    / "consistency"
    / "baseline_title_description_mbert"
)

PREDICTION_PATH = (
    RESULT_DIR
    / "validation_predictions.parquet"
)

METRICS_PATH = (
    RESULT_DIR
    / "validation_metrics.json"
)

OUTPUT_DIR = (
    MODULE_ROOT
    / "results"
    / "consistency"
    / "baseline_title_description_mbert"
    / "error_analysis"
)

OVERALL_PATH = (
    OUTPUT_DIR
    / "overall.csv"
)

BY_CATEGORY_PATH = (
    OUTPUT_DIR
    / "by_category.csv"
)

BY_TIER_PATH = (
    OUTPUT_DIR
    / "by_negative_tier.csv"
)

BY_SIMILARITY_PATH = (
    OUTPUT_DIR
    / "by_negative_similarity.csv"
)

BY_RANK_PATH = (
    OUTPUT_DIR
    / "by_negative_rank.csv"
)

ERROR_SAMPLE_PATH = (
    OUTPUT_DIR
    / "error_samples.csv"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "analysis_report.json"
)


# ============================================================
# CONFIG
# ============================================================

SIMILARITY_BINS = [
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
]

SIMILARITY_LABELS = [
    "0.15-0.20",
    "0.20-0.25",
    "0.25-0.30",
    "0.30-0.35",
    "0.35-0.40",
]


# ============================================================
# HELPERS
# ============================================================

def save_json(data, path):

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


def classification_metrics(
    frame,
    prediction_column,
):

    y_true = (
        frame["label"]
        .astype(int)
        .to_numpy()
    )

    y_pred = (
        frame[prediction_column]
        .astype(int)
        .to_numpy()
    )

    return {
        "n":
            int(len(frame)),

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "precision":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "recall":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),
    }


def grouped_metrics(
    frame,
    group_column,
    prediction_column,
):

    rows = []

    for value, group in frame.groupby(
        group_column,
        dropna=False,
    ):

        metrics = classification_metrics(
            group,
            prediction_column,
        )

        rows.append(
            {
                group_column:
                    (
                        str(value)
                        if pd.notna(value)
                        else "NA"
                    ),

                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    )


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
        "STEP 12 - BASELINE A ERROR ANALYSIS"
    )
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in [
        PAIR_PATH,
        PREDICTION_PATH,
        METRICS_PATH,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

    # ========================================================
    # LOAD SELECTED THRESHOLD
    # ========================================================

    with METRICS_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        model_metrics = json.load(
            f
        )

    tuned_threshold = float(
        model_metrics[
            "best_validation_threshold"
        ]
    )

    print()
    print(
        f"Tuned validation threshold: "
        f"{tuned_threshold:.2f}"
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    pair_df = (
        pq.read_table(
            PAIR_PATH
        )
        .to_pandas()
    )

    prediction_df = (
        pq.read_table(
            PREDICTION_PATH
        )
        .to_pandas()
    )

    print()
    print(
        f"Validation pairs      : "
        f"{len(pair_df):,}"
    )

    print(
        f"Validation predictions: "
        f"{len(prediction_df):,}"
    )

    # ========================================================
    # JOIN BY PAIR ID
    # ========================================================

    if pair_df[
        "pair_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate pair_id "
            "in validation dataset."
        )

    if prediction_df[
        "pair_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate pair_id "
            "in prediction file."
        )

    frame = pair_df.merge(
        prediction_df[
            [
                "pair_id",
                "prob_consistent",
                "prediction_05",
                "prediction_tuned",
            ]
        ],
        on="pair_id",
        how="inner",
        validate="one_to_one",
    )

    if len(frame) != len(pair_df):

        raise RuntimeError(
            "Prediction join lost rows."
        )

    # ========================================================
    # ERROR TYPES
    # ========================================================

    frame[
        "error_type_05"
    ] = "CORRECT"

    frame.loc[
        (
            (frame["label"] == 0)
            &
            (
                frame[
                    "prediction_05"
                ]
                ==
                1
            )
        ),
        "error_type_05",
    ] = "FP"

    frame.loc[
        (
            (frame["label"] == 1)
            &
            (
                frame[
                    "prediction_05"
                ]
                ==
                0
            )
        ),
        "error_type_05",
    ] = "FN"

    frame[
        "error_type_tuned"
    ] = "CORRECT"

    frame.loc[
        (
            (frame["label"] == 0)
            &
            (
                frame[
                    "prediction_tuned"
                ]
                ==
                1
            )
        ),
        "error_type_tuned",
    ] = "FP"

    frame.loc[
        (
            (frame["label"] == 1)
            &
            (
                frame[
                    "prediction_tuned"
                ]
                ==
                0
            )
        ),
        "error_type_tuned",
    ] = "FN"

    # ========================================================
    # OVERALL
    # ========================================================

    metrics_05 = (
        classification_metrics(
            frame,
            "prediction_05",
        )
    )

    metrics_tuned = (
        classification_metrics(
            frame,
            "prediction_tuned",
        )
    )

    overall = pd.DataFrame(
        [
            {
                "setting":
                    "threshold_0.50",

                **metrics_05,
            },
            {
                "setting":
                    f"threshold_{tuned_threshold:.2f}",

                **metrics_tuned,
            },
        ]
    )

    overall.to_csv(
        OVERALL_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # CATEGORY
    # ========================================================

    by_category = grouped_metrics(
        frame,
        "anchor_category",
        "prediction_tuned",
    )

    by_category.sort_values(
        "macro_f1",
        inplace=True,
    )

    by_category.to_csv(
        BY_CATEGORY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # NEGATIVE-ONLY ANALYSIS
    # ========================================================

    negative = (
        frame[
            frame[
                "label"
            ]
            ==
            0
        ]
        .copy()
    )

    # For negatives:
    # predicted 0 = correct rejection
    # predicted 1 = false positive

    negative[
        "negative_correct"
    ] = (
        negative[
            "prediction_tuned"
        ]
        ==
        0
    )

    # --------------------------------------------------------
    # Tier
    # --------------------------------------------------------

    tier_rows = []

    for tier, group in negative.groupby(
        "selection_tier"
    ):

        total = len(
            group
        )

        correct = int(
            group[
                "negative_correct"
            ].sum()
        )

        fp = (
            total
            -
            correct
        )

        tier_rows.append(
            {
                "selection_tier":
                    tier,

                "n":
                    int(
                        total
                    ),

                "correct_negative":
                    correct,

                "false_positive":
                    int(
                        fp
                    ),

                "negative_recall":
                    float(
                        correct
                        /
                        total
                    ),
            }
        )

    tier_df = pd.DataFrame(
        tier_rows
    )

    tier_df.to_csv(
        BY_TIER_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Similarity
    # --------------------------------------------------------

    negative[
        "similarity_bin"
    ] = pd.cut(
        negative[
            "lexical_similarity"
        ],
        bins=SIMILARITY_BINS,
        labels=SIMILARITY_LABELS,
        right=False,
        include_lowest=True,
    )

    similarity_rows = []

    for sim_bin, group in negative.groupby(
        "similarity_bin",
        observed=True,
    ):

        total = len(
            group
        )

        correct = int(
            group[
                "negative_correct"
            ].sum()
        )

        similarity_rows.append(
            {
                "similarity_bin":
                    str(
                        sim_bin
                    ),

                "n":
                    int(
                        total
                    ),

                "correct_negative":
                    correct,

                "false_positive":
                    int(
                        total
                        -
                        correct
                    ),

                "negative_recall":
                    float(
                        correct
                        /
                        total
                    ),
            }
        )

    pd.DataFrame(
        similarity_rows
    ).to_csv(
        BY_SIMILARITY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Rank
    # --------------------------------------------------------

    rank_rows = []

    for rank, group in negative.groupby(
        "candidate_rank"
    ):

        total = len(
            group
        )

        correct = int(
            group[
                "negative_correct"
            ].sum()
        )

        rank_rows.append(
            {
                "candidate_rank":
                    int(
                        rank
                    ),

                "n":
                    int(
                        total
                    ),

                "correct_negative":
                    correct,

                "false_positive":
                    int(
                        total
                        -
                        correct
                    ),

                "negative_recall":
                    float(
                        correct
                        /
                        total
                    ),
            }
        )

    pd.DataFrame(
        rank_rows
    ).to_csv(
        BY_RANK_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # ERROR SAMPLES
    # ========================================================

    errors = (
        frame[
            frame[
                "error_type_tuned"
            ]
            !=
            "CORRECT"
        ]
        .copy()
    )

    errors[
        "confidence"
    ] = np.where(
        errors[
            "prediction_tuned"
        ]
        ==
        1,
        errors[
            "prob_consistent"
        ],
        1.0
        -
        errors[
            "prob_consistent"
        ],
    )

    # Keep strongest errors first.
    errors.sort_values(
        "confidence",
        ascending=False,
        inplace=True,
    )

    columns = [
        "pair_id",
        "label",
        "prediction_tuned",
        "prob_consistent",
        "error_type_tuned",
        "anchor_category",
        "selection_tier",
        "difficulty",
        "lexical_similarity",
        "candidate_rank",
        "anchor_article_uid",
        "headline_article_uid",
    ]

    errors[
        columns
    ].head(
        500
    ).to_csv(
        ERROR_SAMPLE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    fp = int(
        (
            frame[
                "error_type_tuned"
            ]
            ==
            "FP"
        ).sum()
    )

    fn = int(
        (
            frame[
                "error_type_tuned"
            ]
            ==
            "FN"
        ).sum()
    )

    report = {
        "baseline":
            "A_title_plus_description",

        "analysis_split":
            "validation",

        "test_used":
            False,

        "threshold":
            tuned_threshold,

        "overall":
            metrics_tuned,

        "false_positive":
            fp,

        "false_negative":
            fn,

        "outputs": {
            "overall":
                str(
                    OVERALL_PATH.relative_to(
                        MODULE_ROOT
                    )
                ),

            "by_category":
                str(
                    BY_CATEGORY_PATH.relative_to(
                        MODULE_ROOT
                    )
                ),

            "by_negative_tier":
                str(
                    BY_TIER_PATH.relative_to(
                        MODULE_ROOT
                    )
                ),

            "by_negative_similarity":
                str(
                    BY_SIMILARITY_PATH.relative_to(
                        MODULE_ROOT
                    )
                ),

            "by_negative_rank":
                str(
                    BY_RANK_PATH.relative_to(
                        MODULE_ROOT
                    )
                ),

            "error_samples":
                str(
                    ERROR_SAMPLE_PATH.relative_to(
                        MODULE_ROOT
                    )
                ),
        },
    }

    save_json(
        report,
        REPORT_PATH,
    )

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("=" * 80)
    print(
        "STEP 12 RESULTS"
    )
    print("=" * 80)

    print()
    print(
        f"Tuned threshold : "
        f"{tuned_threshold:.2f}"
    )

    print(
        f"Accuracy        : "
        f"{metrics_tuned['accuracy']:.6f}"
    )

    print(
        f"Macro-F1        : "
        f"{metrics_tuned['macro_f1']:.6f}"
    )

    print(
        f"FP              : "
        f"{fp:,}"
    )

    print(
        f"FN              : "
        f"{fn:,}"
    )

    print()
    print(
        "Negative performance by tier:"
    )

    for row in tier_df.itertuples(
        index=False
    ):

        print(
            f"  {row.selection_tier:10s}: "
            f"n={row.n:,}, "
            f"negative_recall="
            f"{row.negative_recall:.4f}, "
            f"FP={row.false_positive:,}"
        )

    print()
    print(
        "Lowest category Macro-F1:"
    )

    for row in (
        by_category
        .head(5)
        .itertuples(
            index=False
        )
    ):

        print(
            f"  "
            f"{row.anchor_category}: "
            f"{row.macro_f1:.4f} "
            f"(n={row.n:,})"
        )

    print()
    print(
        f"Output directory:\n"
        f"{OUTPUT_DIR}"
    )

    print()
    print(
        "TEST WAS NOT USED."
    )

    print(
        "STEP 12 FINISHED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()