from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds


# ============================================================
# PATHS
# ============================================================

MODULE_ROOT = Path(__file__).resolve().parents[2]

ANALYSIS_DIR = (
    MODULE_ROOT
    / "results"
    / "consistency"
    / "candidate_analysis_v1"
)

INPUT_SAMPLE = (
    ANALYSIS_DIR
    / "manual_review_sample.csv"
)

TRAIN_ARTICLES = (
    MODULE_ROOT
    / "data"
    / "interim"
    / "consistency"
    / "articles_v1"
    / "train.parquet"
)

OUTPUT_PATH = (
    ANALYSIS_DIR
    / "manual_review_fulltext.csv"
)


# ============================================================
# CONFIG
# ============================================================

VALID_DECISIONS = [
    "VALID_NEGATIVE",
    "FALSE_NEGATIVE",
    "AMBIGUOUS",
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("MODULE 2 - HEADLINE INTEGRITY")
    print("TASK 2B - HEADLINE CONSISTENCY")
    print("STEP 06B - PREPARE FULL-TEXT MANUAL REVIEW")
    print("=" * 80)

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not INPUT_SAMPLE.exists():
        raise FileNotFoundError(
            INPUT_SAMPLE
        )

    if not TRAIN_ARTICLES.exists():
        raise FileNotFoundError(
            TRAIN_ARTICLES
        )

    sample_df = pd.read_csv(
        INPUT_SAMPLE,
        encoding="utf-8-sig",
        low_memory=False,
    )

    print()
    print(
        f"Sample rows: "
        f"{len(sample_df):,}"
    )

    required_columns = [
        "candidate_pair_id",
        "category",
        "similarity_bin",
        "rank_bucket",
        "candidate_rank",
        "lexical_similarity",
        "anchor_article_uid",
        "anchor_title_original",
        "anchor_description",
        "candidate_headline_article_uid",
        "candidate_headline",
    ]

    missing = [
        column
        for column in required_columns
        if column not in sample_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    # --------------------------------------------------------
    # Unique anchors
    # --------------------------------------------------------

    anchor_uids = (
        sample_df[
            "anchor_article_uid"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    print(
        f"Unique anchors: "
        f"{len(anchor_uids):,}"
    )

    # --------------------------------------------------------
    # Load full article text only for sampled anchors
    # --------------------------------------------------------

    dataset = ds.dataset(
        str(TRAIN_ARTICLES),
        format="parquet",
    )

    table = dataset.to_table(
        columns=[
            "article_uid",
            "content",
        ],
        filter=(
            ds.field(
                "article_uid"
            ).isin(
                anchor_uids
            )
        ),
    )

    article_df = (
        table
        .to_pandas()
    )

    print(
        f"Articles loaded: "
        f"{len(article_df):,}"
    )

    if (
        article_df[
            "article_uid"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate article_uid "
            "found in train corpus."
        )

    content_map = dict(
        zip(
            article_df[
                "article_uid"
            ],
            article_df[
                "content"
            ],
        )
    )

    missing_anchors = (
        set(
            anchor_uids
        )
        -
        set(
            content_map.keys()
        )
    )

    if missing_anchors:
        raise RuntimeError(
            "Missing sampled anchors: "
            f"{len(missing_anchors)}"
        )

    # --------------------------------------------------------
    # Build clean review table
    #
    # Important:
    # Candidate description is intentionally NOT included.
    # Reviewer should decide whether the candidate HEADLINE
    # is supported by the ANCHOR ARTICLE.
    # --------------------------------------------------------

    rows = []

    for row in sample_df.itertuples(
        index=False,
    ):

        anchor_uid = (
            row.anchor_article_uid
        )

        rows.append(
            {
                "candidate_pair_id":
                    row.candidate_pair_id,

                "category":
                    row.category,

                "similarity_bin":
                    row.similarity_bin,

                "rank_bucket":
                    row.rank_bucket,

                "candidate_rank":
                    int(
                        row.candidate_rank
                    ),

                "lexical_similarity":
                    float(
                        row.lexical_similarity
                    ),

                "anchor_article_uid":
                    anchor_uid,

                "ANCHOR_ORIGINAL_TITLE":
                    row.anchor_title_original,

                "ANCHOR_DESCRIPTION":
                    row.anchor_description,

                "ANCHOR_FULL_CONTENT":
                    content_map[
                        anchor_uid
                    ],

                "CANDIDATE_HEADLINE":
                    row.candidate_headline,

                # ============================================
                # HUMAN ANNOTATION
                # ============================================

                "manual_decision":
                    "",

                "manual_reason":
                    "",
            }
        )

    output_df = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Stable order
    # --------------------------------------------------------

    output_df.sort_values(
        by=[
            "similarity_bin",
            "rank_bucket",
            "category",
            "lexical_similarity",
        ],
        ascending=[
            True,
            True,
            True,
            False,
        ],
        inplace=True,
    )

    output_df.reset_index(
        drop=True,
        inplace=True,
    )

    output_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        f"Review rows: "
        f"{len(output_df):,}"
    )

    print()
    print(
        "Allowed manual_decision values:"
    )

    for value in VALID_DECISIONS:
        print(
            f"  {value}"
        )

    print()
    print(
        f"Output:\n"
        f"{OUTPUT_PATH}"
    )

    print()
    print(
        "STEP 06B FINISHED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()