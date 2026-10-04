from pathlib import Path
import json

import pandas as pd
import pyarrow.parquet as pq


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

A_DIR = (
    MODULE_ROOT
    / "results"
    / "consistency"
    / "baseline_title_description_mbert"
)

B_DIR = (
    MODULE_ROOT
    / "results"
    / "consistency"
    / "baseline_title_article_head_mbert"
)

A_PRED_PATH = (
    A_DIR
    / "validation_predictions.parquet"
)

B_PRED_PATH = (
    B_DIR
    / "validation_predictions.parquet"
)

A_METRICS_PATH = (
    A_DIR
    / "validation_metrics.json"
)

B_METRICS_PATH = (
    B_DIR
    / "validation_metrics.json"
)

OUTPUT_DIR = (
    MODULE_ROOT
    / "results"
    / "consistency"
    / "baseline_a_b_comparison"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "comparison_summary.json"
)

OVERLAP_PATH = (
    OUTPUT_DIR
    / "error_overlap.csv"
)

BY_CATEGORY_PATH = (
    OUTPUT_DIR
    / "by_category.csv"
)

BY_TIER_PATH = (
    OUTPUT_DIR
    / "by_negative_tier.csv"
)

DISAGREEMENT_PATH = (
    OUTPUT_DIR
    / "model_disagreements.csv"
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
        "STEP 14 - COMPARE BASELINES A AND B"
    )
    print("=" * 80)

    print()
    print(
        "VALIDATION ONLY."
    )

    print(
        "TEST IS NOT USED."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    required = [
        PAIR_PATH,
        A_PRED_PATH,
        B_PRED_PATH,
        A_METRICS_PATH,
        B_METRICS_PATH,
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

    # ========================================================
    # LOAD METRICS
    # ========================================================

    a_metrics = load_json(
        A_METRICS_PATH
    )

    b_metrics = load_json(
        B_METRICS_PATH
    )

    a_threshold = float(
        a_metrics[
            "best_validation_threshold"
        ]
    )

    b_threshold = float(
        b_metrics[
            "best_validation_threshold"
        ]
    )

    a_macro_f1 = float(
        a_metrics[
            "metrics_at_tuned_threshold"
        ][
            "macro_f1"
        ]
    )

    b_macro_f1 = float(
        b_metrics[
            "metrics_at_tuned_threshold"
        ][
            "macro_f1"
        ]
    )

    print()
    print(
        f"Baseline A threshold: "
        f"{a_threshold:.2f}"
    )

    print(
        f"Baseline B threshold: "
        f"{b_threshold:.2f}"
    )

    print(
        f"Baseline A Macro-F1 : "
        f"{a_macro_f1:.6f}"
    )

    print(
        f"Baseline B Macro-F1 : "
        f"{b_macro_f1:.6f}"
    )

    print(
        f"B - A               : "
        f"{b_macro_f1 - a_macro_f1:+.6f}"
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

    a_pred = (
        pq.read_table(
            A_PRED_PATH
        )
        .to_pandas()
    )

    b_pred = (
        pq.read_table(
            B_PRED_PATH
        )
        .to_pandas()
    )

    if pair_df[
        "pair_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate pair IDs "
            "in validation dataset."
        )

    if a_pred[
        "pair_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate pair IDs "
            "in Baseline A predictions."
        )

    if b_pred[
        "pair_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate pair IDs "
            "in Baseline B predictions."
        )

    # ========================================================
    # NORMALIZE PREDICTIONS
    # ========================================================

    a_pred = (
        a_pred[
            [
                "pair_id",
                "prob_consistent",
            ]
        ]
        .rename(
            columns={
                "prob_consistent":
                    "prob_a",
            }
        )
    )

    b_pred = (
        b_pred[
            [
                "pair_id",
                "prob_consistent",
            ]
        ]
        .rename(
            columns={
                "prob_consistent":
                    "prob_b",
            }
        )
    )

    frame = (
        pair_df
        .merge(
            a_pred,
            on="pair_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            b_pred,
            on="pair_id",
            how="inner",
            validate="one_to_one",
        )
    )

    if len(frame) != len(
        pair_df
    ):

        raise RuntimeError(
            "Prediction join lost rows."
        )

    frame[
        "pred_a"
    ] = (
        frame[
            "prob_a"
        ]
        >=
        a_threshold
    ).astype(int)

    frame[
        "pred_b"
    ] = (
        frame[
            "prob_b"
        ]
        >=
        b_threshold
    ).astype(int)

    frame[
        "correct_a"
    ] = (
        frame[
            "pred_a"
        ]
        ==
        frame[
            "label"
        ]
    )

    frame[
        "correct_b"
    ] = (
        frame[
            "pred_b"
        ]
        ==
        frame[
            "label"
        ]
    )

    # ========================================================
    # FOUR-WAY OVERLAP
    # ========================================================

    frame[
        "comparison_group"
    ] = "BOTH_WRONG"

    frame.loc[
        (
            frame[
                "correct_a"
            ]
            &
            frame[
                "correct_b"
            ]
        ),
        "comparison_group",
    ] = "BOTH_CORRECT"

    frame.loc[
        (
            frame[
                "correct_a"
            ]
            &
            ~frame[
                "correct_b"
            ]
        ),
        "comparison_group",
    ] = "A_ONLY_CORRECT"

    frame.loc[
        (
            ~frame[
                "correct_a"
            ]
            &
            frame[
                "correct_b"
            ]
        ),
        "comparison_group",
    ] = "B_ONLY_CORRECT"

    overlap = (
        frame[
            "comparison_group"
        ]
        .value_counts()
        .reindex(
            [
                "BOTH_CORRECT",
                "A_ONLY_CORRECT",
                "B_ONLY_CORRECT",
                "BOTH_WRONG",
            ],
            fill_value=0,
        )
    )

    overlap_rows = []

    for group, count in (
        overlap.items()
    ):

        overlap_rows.append(
            {
                "group":
                    group,

                "count":
                    int(
                        count
                    ),

                "rate":
                    float(
                        count
                        /
                        len(
                            frame
                        )
                    ),
            }
        )

    pd.DataFrame(
        overlap_rows
    ).to_csv(
        OVERLAP_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # CATEGORY COMPARISON
    # ========================================================

    category_rows = []

    for category, group in (
        frame.groupby(
            "anchor_category"
        )
    ):

        a_accuracy = float(
            group[
                "correct_a"
            ].mean()
        )

        b_accuracy = float(
            group[
                "correct_b"
            ].mean()
        )

        category_rows.append(
            {
                "category":
                    category,

                "n":
                    int(
                        len(
                            group
                        )
                    ),

                "accuracy_a":
                    a_accuracy,

                "accuracy_b":
                    b_accuracy,

                "delta_b_minus_a":
                    (
                        b_accuracy
                        -
                        a_accuracy
                    ),

                "a_only_correct":
                    int(
                        (
                            group[
                                "comparison_group"
                            ]
                            ==
                            "A_ONLY_CORRECT"
                        ).sum()
                    ),

                "b_only_correct":
                    int(
                        (
                            group[
                                "comparison_group"
                            ]
                            ==
                            "B_ONLY_CORRECT"
                        ).sum()
                    ),

                "both_wrong":
                    int(
                        (
                            group[
                                "comparison_group"
                            ]
                            ==
                            "BOTH_WRONG"
                        ).sum()
                    ),
            }
        )

    category_df = pd.DataFrame(
        category_rows
    )

    category_df.sort_values(
        "delta_b_minus_a",
        inplace=True,
    )

    category_df.to_csv(
        BY_CATEGORY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # NEGATIVE TIER COMPARISON
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

    tier_rows = []

    for tier, group in (
        negative.groupby(
            "selection_tier"
        )
    ):

        # negative is correct when prediction == 0

        recall_a = float(
            (
                group[
                    "pred_a"
                ]
                ==
                0
            ).mean()
        )

        recall_b = float(
            (
                group[
                    "pred_b"
                ]
                ==
                0
            ).mean()
        )

        tier_rows.append(
            {
                "tier":
                    tier,

                "n":
                    int(
                        len(
                            group
                        )
                    ),

                "negative_recall_a":
                    recall_a,

                "negative_recall_b":
                    recall_b,

                "delta_b_minus_a":
                    (
                        recall_b
                        -
                        recall_a
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

    # ========================================================
    # DISAGREEMENT SAMPLES
    # ========================================================

    disagreement = (
        frame[
            frame[
                "pred_a"
            ]
            !=
            frame[
                "pred_b"
            ]
        ]
        .copy()
    )

    disagreement[
        "prob_gap"
    ] = (
        disagreement[
            "prob_a"
        ]
        -
        disagreement[
            "prob_b"
        ]
    ).abs()

    disagreement.sort_values(
        [
            "comparison_group",
            "prob_gap",
        ],
        ascending=[
            True,
            False,
        ],
        inplace=True,
    )

    disagreement_columns = [
        "pair_id",
        "label",
        "anchor_category",
        "selection_tier",
        "lexical_similarity",
        "candidate_rank",
        "prob_a",
        "pred_a",
        "prob_b",
        "pred_b",
        "correct_a",
        "correct_b",
        "comparison_group",
        "prob_gap",
        "anchor_article_uid",
        "headline_article_uid",
    ]

    disagreement[
        disagreement_columns
    ].to_csv(
        DISAGREEMENT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # ORACLE ANALYSIS
    #
    # If either model is correct, an ideal selector could
    # classify the sample correctly. This is NOT a model score;
    # it only measures complementarity.
    # ========================================================

    oracle_correct = (
        frame[
            "correct_a"
        ]
        |
        frame[
            "correct_b"
        ]
    )

    oracle_accuracy = float(
        oracle_correct.mean()
    )

    both_wrong = int(
        (
            ~frame[
                "correct_a"
            ]
            &
            ~frame[
                "correct_b"
            ]
        ).sum()
    )

    a_only = int(
        (
            frame[
                "correct_a"
            ]
            &
            ~frame[
                "correct_b"
            ]
        ).sum()
    )

    b_only = int(
        (
            ~frame[
                "correct_a"
            ]
            &
            frame[
                "correct_b"
            ]
        ).sum()
    )

    # ========================================================
    # REPORT
    # ========================================================

    report = {
        "split":
            "validation",

        "test_used":
            False,

        "baseline_a": {
            "input":
                "headline + description",

            "threshold":
                a_threshold,

            "macro_f1":
                a_macro_f1,
        },

        "baseline_b": {
            "input":
                "headline + article head",

            "threshold":
                b_threshold,

            "macro_f1":
                b_macro_f1,
        },

        "macro_f1_delta_b_minus_a":
            float(
                b_macro_f1
                -
                a_macro_f1
            ),

        "error_overlap": {
            row[
                "group"
            ]: {
                "count":
                    row[
                        "count"
                    ],

                "rate":
                    row[
                        "rate"
                    ],
            }

            for row in overlap_rows
        },

        "a_only_correct":
            a_only,

        "b_only_correct":
            b_only,

        "both_wrong":
            both_wrong,

        "oracle_either_model_correct_rate":
            oracle_accuracy,

        "oracle_note":
            (
                "This is not a deployable model score. "
                "It measures prediction complementarity only."
            ),
    }

    save_json(
        report,
        SUMMARY_PATH,
    )

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("=" * 80)
    print(
        "STEP 14 RESULTS"
    )
    print("=" * 80)

    print()
    print(
        "Validation Macro-F1:"
    )

    print(
        f"  A description : "
        f"{a_macro_f1:.6f}"
    )

    print(
        f"  B article head: "
        f"{b_macro_f1:.6f}"
    )

    print(
        f"  B - A         : "
        f"{b_macro_f1 - a_macro_f1:+.6f}"
    )

    print()
    print(
        "Prediction overlap:"
    )

    for row in overlap_rows:

        print(
            f"  {row['group']:15s}: "
            f"{row['count']:,} "
            f"({row['rate']:.2%})"
        )

    print()
    print(
        "Complementarity:"
    )

    print(
        f"  A only correct : "
        f"{a_only:,}"
    )

    print(
        f"  B only correct : "
        f"{b_only:,}"
    )

    print(
        f"  Both wrong     : "
        f"{both_wrong:,}"
    )

    print(
        f"  Either correct : "
        f"{oracle_accuracy:.2%}"
    )

    print()
    print(
        "Negative tiers:"
    )

    for row in tier_df.itertuples(
        index=False
    ):

        print(
            f"  {row.tier:10s}: "
            f"A={row.negative_recall_a:.4f}, "
            f"B={row.negative_recall_b:.4f}, "
            f"delta="
            f"{row.delta_b_minus_a:+.4f}"
        )

    print()
    print(
        "Categories where B loses most:"
    )

    for row in (
        category_df
        .head(5)
        .itertuples(
            index=False
        )
    ):

        print(
            f"  {row.category}: "
            f"A={row.accuracy_a:.4f}, "
            f"B={row.accuracy_b:.4f}, "
            f"delta="
            f"{row.delta_b_minus_a:+.4f}"
        )

    print()
    print(
        "Categories where B gains most:"
    )

    for row in (
        category_df
        .tail(5)
        .sort_values(
            "delta_b_minus_a",
            ascending=False,
        )
        .itertuples(
            index=False
        )
    ):

        print(
            f"  {row.category}: "
            f"A={row.accuracy_a:.4f}, "
            f"B={row.accuracy_b:.4f}, "
            f"delta="
            f"{row.delta_b_minus_a:+.4f}"
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
        "STEP 14 FINISHED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()