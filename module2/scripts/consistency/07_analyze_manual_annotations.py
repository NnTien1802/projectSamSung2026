from pathlib import Path
import json

import pandas as pd


# ============================================================
# PATHS
# ============================================================

MODULE_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    MODULE_ROOT
    / "results"
    / "consistency"
    / "candidate_analysis_v1"
    / "manual_review_fulltext.csv"
)

OUTPUT_DIR = (
    MODULE_ROOT
    / "results"
    / "consistency"
    / "manual_annotation_analysis_v1"
)

OVERALL_PATH = (
    OUTPUT_DIR
    / "overall.csv"
)

BY_SIMILARITY_PATH = (
    OUTPUT_DIR
    / "by_similarity.csv"
)

BY_RANK_BUCKET_PATH = (
    OUTPUT_DIR
    / "by_rank_bucket.csv"
)

BY_SIMILARITY_RANK_PATH = (
    OUTPUT_DIR
    / "by_similarity_rank.csv"
)

BY_CATEGORY_PATH = (
    OUTPUT_DIR
    / "by_category.csv"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "analysis_report.json"
)


# ============================================================
# CONFIG
# ============================================================

VALID_LABELS = {
    "VALID_NEGATIVE",
    "FALSE_NEGATIVE",
    "AMBIGUOUS",
}

LABEL_ORDER = [
    "VALID_NEGATIVE",
    "FALSE_NEGATIVE",
    "AMBIGUOUS",
]


# ============================================================
# HELPERS
# ============================================================

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


def build_summary(
    df,
    group_columns,
):
    """
    Produce counts and percentages by arbitrary grouping.
    """

    grouped = (
        df.groupby(
            group_columns
            + ["manual_decision"],
            dropna=False,
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    for label in LABEL_ORDER:

        if label not in grouped.columns:

            grouped[
                label
            ] = 0

    grouped = grouped[
        LABEL_ORDER
    ]

    grouped[
        "total"
    ] = grouped.sum(
        axis=1
    )

    for label in LABEL_ORDER:

        grouped[
            f"{label}_rate"
        ] = (
            grouped[
                label
            ]
            /
            grouped[
                "total"
            ]
        )

    return (
        grouped
        .reset_index()
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
        "STEP 07 - ANALYZE MANUAL ANNOTATIONS"
    )
    print("=" * 80)

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            INPUT_PATH
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # LOAD
    # ========================================================

    df = pd.read_csv(
        INPUT_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )

    print()
    print(
        f"Rows loaded: "
        f"{len(df):,}"
    )

    required_columns = [
        "candidate_pair_id",
        "category",
        "similarity_bin",
        "rank_bucket",
        "candidate_rank",
        "lexical_similarity",
        "manual_decision",
    ]

    missing_columns = [
        column
        for column
        in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing columns: "
            f"{missing_columns}"
        )

    # ========================================================
    # CLEAN LABELS
    # ========================================================

    df[
        "manual_decision"
    ] = (
        df[
            "manual_decision"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    empty_count = int(
        (
            df[
                "manual_decision"
            ]
            ==
            ""
        ).sum()
    )

    if empty_count:

        raise ValueError(
            "Unannotated rows found: "
            f"{empty_count}"
        )

    observed_labels = set(
        df[
            "manual_decision"
        ].unique()
    )

    invalid_labels = sorted(
        observed_labels
        -
        VALID_LABELS
    )

    if invalid_labels:

        raise ValueError(
            "Invalid labels found: "
            f"{invalid_labels}"
        )

    duplicate_pair_ids = int(
        df[
            "candidate_pair_id"
        ]
        .duplicated()
        .sum()
    )

    if duplicate_pair_ids:

        raise ValueError(
            "Duplicate candidate_pair_id: "
            f"{duplicate_pair_ids}"
        )

    print(
        "Annotation validation: PASSED"
    )

    # ========================================================
    # OVERALL
    # ========================================================

    overall_counts = (
        df[
            "manual_decision"
        ]
        .value_counts()
    )

    overall_rows = []

    for label in LABEL_ORDER:

        count = int(
            overall_counts.get(
                label,
                0,
            )
        )

        overall_rows.append(
            {
                "label":
                    label,

                "count":
                    count,

                "rate":
                    (
                        count
                        /
                        len(df)
                    ),
            }
        )

    overall_df = pd.DataFrame(
        overall_rows
    )

    overall_df.to_csv(
        OVERALL_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # BY SIMILARITY
    # ========================================================

    by_similarity = build_summary(
        df,
        [
            "similarity_bin"
        ],
    )

    by_similarity.to_csv(
        BY_SIMILARITY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # BY RANK BUCKET
    # ========================================================

    by_rank = build_summary(
        df,
        [
            "rank_bucket"
        ],
    )

    by_rank.to_csv(
        BY_RANK_BUCKET_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # SIMILARITY × RANK
    # ========================================================

    by_similarity_rank = (
        build_summary(
            df,
            [
                "similarity_bin",
                "rank_bucket",
            ],
        )
    )

    by_similarity_rank.to_csv(
        BY_SIMILARITY_RANK_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # CATEGORY
    # ========================================================

    by_category = build_summary(
        df,
        [
            "category"
        ],
    )

    by_category.to_csv(
        BY_CATEGORY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # RECOMMENDED RULE ANALYSIS
    # ========================================================

    primary_mask = (
        (
            df[
                "rank_bucket"
            ]
            ==
            "rank_4_10"
        )
        &
        (
            df[
                "lexical_similarity"
            ]
            >=
            0.20
        )
        &
        (
            df[
                "lexical_similarity"
            ]
            <
            0.40
        )
    )

    fallback_mask = (
        (
            df[
                "rank_bucket"
            ]
            ==
            "rank_4_10"
        )
        &
        (
            df[
                "lexical_similarity"
            ]
            >=
            0.15
        )
        &
        (
            df[
                "lexical_similarity"
            ]
            <
            0.20
        )
    )

    primary_df = df[
        primary_mask
    ]

    fallback_df = df[
        fallback_mask
    ]

    def rule_stats(
        subset,
    ):

        total = len(
            subset
        )

        counts = (
            subset[
                "manual_decision"
            ]
            .value_counts()
        )

        return {
            "n":
                int(
                    total
                ),

            "valid_negative":
                int(
                    counts.get(
                        "VALID_NEGATIVE",
                        0,
                    )
                ),

            "false_negative":
                int(
                    counts.get(
                        "FALSE_NEGATIVE",
                        0,
                    )
                ),

            "ambiguous":
                int(
                    counts.get(
                        "AMBIGUOUS",
                        0,
                    )
                ),

            "valid_negative_rate":
                (
                    float(
                        (
                            subset[
                                "manual_decision"
                            ]
                            ==
                            "VALID_NEGATIVE"
                        ).mean()
                    )
                    if total
                    else None
                ),
        }

    primary_stats = (
        rule_stats(
            primary_df
        )
    )

    fallback_stats = (
        rule_stats(
            fallback_df
        )
    )

    # ========================================================
    # REPORT
    # ========================================================

    report = {
        "module":
            "module2_headline_integrity",

        "task":
            "headline_consistency",

        "stage":
            "07_analyze_manual_annotations",

        "rows":
            int(
                len(df)
            ),

        "validation": {
            "empty_annotations":
                empty_count,

            "duplicate_pair_ids":
                duplicate_pair_ids,

            "invalid_labels":
                invalid_labels,

            "passed":
                True,
        },

        "overall": {
            label: {
                "count":
                    int(
                        overall_counts.get(
                            label,
                            0,
                        )
                    ),

                "rate":
                    float(
                        (
                            df[
                                "manual_decision"
                            ]
                            ==
                            label
                        ).mean()
                    ),
            }

            for label
            in LABEL_ORDER
        },

        "candidate_policy_v1": {
            "primary": {
                "rank":
                    "4-10",

                "similarity_min":
                    0.20,

                "similarity_max_exclusive":
                    0.40,

                "manual_evidence":
                    primary_stats,
            },

            "fallback": {
                "rank":
                    "4-10",

                "similarity_min":
                    0.15,

                "similarity_max_exclusive":
                    0.20,

                "manual_evidence":
                    fallback_stats,
            },

            "forbidden_for_automatic_v1": {
                "rank_1":
                    True,

                "rank_2_3":
                    True,

                "similarity_0_40_0_55":
                    True,
            },
        },

        "notes": [
            (
                "Rules were designed using "
                "TRAIN manual annotations only."
            ),
            (
                "Do not infer category-specific "
                "rules because category sample "
                "sizes are small and unequal."
            ),
            (
                "Similarity x rank cells contain "
                "only about 10 reviewed examples; "
                "avoid overfitting fine-grained "
                "thresholds."
            ),
        ],
    }

    save_json(
        report,
        REPORT_PATH,
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print(
        "OVERALL ANNOTATION RESULTS"
    )
    print("=" * 80)

    for row in overall_rows:

        print(
            f"{row['label']:16s}: "
            f"{row['count']:3d} "
            f"({row['rate']:.2%})"
        )

    print()
    print(
        "Rank bucket:"
    )

    for row in (
        by_rank.itertuples(
            index=False
        )
    ):

        print(
            f"  {row.rank_bucket:12s}: "
            f"VALID="
            f"{row.VALID_NEGATIVE_rate:.2%}, "
            f"FALSE="
            f"{row.FALSE_NEGATIVE_rate:.2%}, "
            f"AMBIG="
            f"{row.AMBIGUOUS_rate:.2%}"
        )

    print()
    print(
        "Candidate policy v1:"
    )

    print(
        "  PRIMARY : "
        "rank 4-10, "
        "0.20 <= sim < 0.40"
    )

    print(
        "    manual validity: "
        f"{primary_stats['valid_negative']}/"
        f"{primary_stats['n']} "
        f"("
        f"{primary_stats['valid_negative_rate']:.2%}"
        f")"
    )

    print(
        "  FALLBACK: "
        "rank 4-10, "
        "0.15 <= sim < 0.20"
    )

    print(
        "    manual validity: "
        f"{fallback_stats['valid_negative']}/"
        f"{fallback_stats['n']} "
        f"("
        f"{fallback_stats['valid_negative_rate']:.2%}"
        f")"
    )

    print()
    print(
        f"Output directory:\n"
        f"{OUTPUT_DIR}"
    )

    print()
    print(
        "STEP 07 FINISHED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()