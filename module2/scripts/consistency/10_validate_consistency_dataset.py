from pathlib import Path
from collections import Counter
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
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

ARTICLES_DIR = (
    MODULE_ROOT
    / "data"
    / "interim"
    / "consistency"
    / "articles_v1"
)

ARTICLE_INDEX_PATH = (
    ARTICLES_DIR
    / "article_index.parquet"
)

ARTICLE_FILES = {
    "train":
        ARTICLES_DIR
        / "train.parquet",

    "validation":
        ARTICLES_DIR
        / "validation.parquet",

    "test":
        ARTICLES_DIR
        / "test.parquet",
}

PAIR_FILES = {
    "train":
        DATASET_DIR
        / "train.parquet",

    "validation":
        DATASET_DIR
        / "validation.parquet",

    "test":
        DATASET_DIR
        / "test.parquet",
}

OUTPUT_DIR = (
    MODULE_ROOT
    / "results"
    / "consistency"
    / "final_dataset_validation_v1"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "validation_report.json"
)

SPLIT_STATS_PATH = (
    OUTPUT_DIR
    / "split_stats.csv"
)

CATEGORY_STATS_PATH = (
    OUTPUT_DIR
    / "category_stats.csv"
)

NEGATIVE_TIER_STATS_PATH = (
    OUTPUT_DIR
    / "negative_tier_stats.csv"
)

TRAIN_REVIEW_SAMPLE_PATH = (
    OUTPUT_DIR
    / "train_join_review_sample.csv"
)


# ============================================================
# CONFIG
# ============================================================

VALID_SPLITS = [
    "train",
    "validation",
    "test",
]

PRIMARY_MIN = 0.20
PRIMARY_MAX = 0.40

FALLBACK_MIN = 0.15
FALLBACK_MAX = 0.20

MIN_RANK = 4
MAX_RANK = 10

TRAIN_REVIEW_PER_TYPE = 10

CONTENT_PREVIEW_CHARS = 3500

RANDOM_SEED = 20260919


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


def deterministic_key(
    value,
):

    raw = (
        str(value)
        +
        str(RANDOM_SEED)
    )

    digest = hashlib.blake2b(
        raw.encode("utf-8"),
        digest_size=8,
    ).digest()

    return int.from_bytes(
        digest,
        byteorder="big",
        signed=False,
    )


def bool_value(
    value,
):

    return bool(
        value
    )


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_input_files():

    print()
    print("=" * 80)
    print(
        "VALIDATE INPUT FILES"
    )
    print("=" * 80)

    required = [
        ARTICLE_INDEX_PATH,
    ]

    required.extend(
        ARTICLE_FILES.values()
    )

    required.extend(
        PAIR_FILES.values()
    )

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

        print(
            f"FOUND: {path}"
        )

    print()
    print(
        "Input validation: PASSED"
    )


# ============================================================
# LOAD ARTICLE INDEX
# ============================================================

def load_article_index():

    print()
    print("=" * 80)
    print(
        "LOAD ARTICLE INDEX"
    )
    print("=" * 80)

    columns = [
        "article_uid",
        "dataset",
        "split",
        "title",
        "description",
        "category",
        "title_hash",
        "content_hash",
        "content_word_count",
    ]

    table = pq.read_table(
        ARTICLE_INDEX_PATH,
        columns=columns,
    )

    frame = table.to_pandas()

    if (
        frame[
            "article_uid"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate article_uid "
            "in article index."
        )

    if (
        frame[
            "article_uid"
        ]
        .isna()
        .any()
    ):

        raise RuntimeError(
            "Missing article_uid "
            "in article index."
        )

    metadata = (
        frame.set_index(
            "article_uid"
        )
    )

    print(
        f"Articles: "
        f"{len(metadata):,}"
    )

    print(
        "Article index validation: PASSED"
    )

    return metadata


# ============================================================
# VALIDATE ONE SPLIT
# ============================================================

def validate_split(
    split,
    metadata,
):

    print()
    print("=" * 80)
    print(
        f"VALIDATE SPLIT - "
        f"{split.upper()}"
    )
    print("=" * 80)

    path = (
        PAIR_FILES[
            split
        ]
    )

    frame = (
        pq.read_table(
            path
        )
        .to_pandas()
    )

    print(
        f"Rows: "
        f"{len(frame):,}"
    )

    required_columns = [
        "pair_id",
        "split",
        "anchor_article_uid",
        "headline_article_uid",
        "label",
        "label_name",
        "pair_type",
        "difficulty",
        "selection_tier",
        "anchor_category",
        "headline_category",
        "same_category",
        "anchor_title_hash",
        "headline_title_hash",
        "anchor_content_hash",
        "headline_content_hash",
        "lexical_similarity",
        "candidate_rank",
        "source_candidate_pair_id",
        "source_dataset",
    ]

    missing_columns = [
        column
        for column
        in required_columns
        if column not in frame.columns
    ]

    if missing_columns:

        raise RuntimeError(
            f"{split}: missing columns "
            f"{missing_columns}"
        )

    checks = {}

    # ========================================================
    # BASIC STRUCTURE
    # ========================================================

    checks[
        "pair_id_unique"
    ] = bool(
        ~frame[
            "pair_id"
        ]
        .duplicated()
        .any()
    )

    checks[
        "split_field_valid"
    ] = bool(
        (
            frame[
                "split"
            ]
            ==
            split
        ).all()
    )

    checks[
        "labels_exactly_0_1"
    ] = (
        set(
            frame[
                "label"
            ]
            .unique()
        )
        ==
        {
            0,
            1,
        }
    )

    expected_label_names = (
        frame[
            "label"
        ]
        .map(
            {
                0:
                    "INCONSISTENT",

                1:
                    "CONSISTENT",
            }
        )
    )

    checks[
        "label_name_valid"
    ] = bool(
        (
            expected_label_names
            ==
            frame[
                "label_name"
            ]
        ).all()
    )

    # ========================================================
    # REFERENTIAL INTEGRITY
    # ========================================================

    article_uids = set(
        metadata.index
    )

    anchor_exists = (
        frame[
            "anchor_article_uid"
        ]
        .isin(
            article_uids
        )
    )

    headline_exists = (
        frame[
            "headline_article_uid"
        ]
        .isin(
            article_uids
        )
    )

    checks[
        "all_anchor_articles_exist"
    ] = bool(
        anchor_exists.all()
    )

    checks[
        "all_headline_articles_exist"
    ] = bool(
        headline_exists.all()
    )

    if not checks[
        "all_anchor_articles_exist"
    ]:

        raise RuntimeError(
            f"{split}: missing anchor "
            "article references."
        )

    if not checks[
        "all_headline_articles_exist"
    ]:

        raise RuntimeError(
            f"{split}: missing headline "
            "article references."
        )

    # ========================================================
    # METADATA LOOKUP
    # ========================================================

    anchor_uid = (
        frame[
            "anchor_article_uid"
        ]
    )

    headline_uid = (
        frame[
            "headline_article_uid"
        ]
    )

    expected_anchor_split = (
        anchor_uid.map(
            metadata[
                "split"
            ]
        )
    )

    expected_headline_split = (
        headline_uid.map(
            metadata[
                "split"
            ]
        )
    )

    checks[
        "anchor_split_consistent"
    ] = bool(
        (
            expected_anchor_split
            ==
            split
        ).all()
    )

    checks[
        "headline_split_consistent"
    ] = bool(
        (
            expected_headline_split
            ==
            split
        ).all()
    )

    expected_anchor_category = (
        anchor_uid.map(
            metadata[
                "category"
            ]
        )
    )

    expected_headline_category = (
        headline_uid.map(
            metadata[
                "category"
            ]
        )
    )

    checks[
        "anchor_category_matches_index"
    ] = bool(
        (
            expected_anchor_category
            ==
            frame[
                "anchor_category"
            ]
        ).all()
    )

    checks[
        "headline_category_matches_index"
    ] = bool(
        (
            expected_headline_category
            ==
            frame[
                "headline_category"
            ]
        ).all()
    )

    expected_anchor_title_hash = (
        anchor_uid.map(
            metadata[
                "title_hash"
            ]
        )
    )

    expected_headline_title_hash = (
        headline_uid.map(
            metadata[
                "title_hash"
            ]
        )
    )

    checks[
        "anchor_title_hash_matches_index"
    ] = bool(
        (
            expected_anchor_title_hash
            ==
            frame[
                "anchor_title_hash"
            ]
        ).all()
    )

    checks[
        "headline_title_hash_matches_index"
    ] = bool(
        (
            expected_headline_title_hash
            ==
            frame[
                "headline_title_hash"
            ]
        ).all()
    )

    expected_anchor_content_hash = (
        anchor_uid.map(
            metadata[
                "content_hash"
            ]
        )
    )

    expected_headline_content_hash = (
        headline_uid.map(
            metadata[
                "content_hash"
            ]
        )
    )

    checks[
        "anchor_content_hash_matches_index"
    ] = bool(
        (
            expected_anchor_content_hash
            ==
            frame[
                "anchor_content_hash"
            ]
        ).all()
    )

    checks[
        "headline_content_hash_matches_index"
    ] = bool(
        (
            expected_headline_content_hash
            ==
            frame[
                "headline_content_hash"
            ]
        ).all()
    )

    # ========================================================
    # BALANCE + ANCHOR STRUCTURE
    # ========================================================

    label_counts = (
        frame[
            "label"
        ]
        .value_counts()
    )

    positive_count = int(
        label_counts.get(
            1,
            0,
        )
    )

    negative_count = int(
        label_counts.get(
            0,
            0,
        )
    )

    checks[
        "exact_class_balance"
    ] = (
        positive_count
        ==
        negative_count
    )

    anchor_sizes = (
        frame.groupby(
            "anchor_article_uid"
        )
        .size()
    )

    checks[
        "exactly_two_pairs_per_anchor"
    ] = bool(
        (
            anchor_sizes
            ==
            2
        ).all()
    )

    anchor_label_sums = (
        frame.groupby(
            "anchor_article_uid"
        )[
            "label"
        ]
        .sum()
    )

    checks[
        "one_positive_one_negative_per_anchor"
    ] = bool(
        (
            anchor_label_sums
            ==
            1
        ).all()
    )

    # ========================================================
    # POSITIVES
    # ========================================================

    positive = (
        frame[
            frame[
                "label"
            ]
            ==
            1
        ]
        .copy()
    )

    checks[
        "positive_self_pair"
    ] = bool(
        (
            positive[
                "anchor_article_uid"
            ]
            ==
            positive[
                "headline_article_uid"
            ]
        ).all()
    )

    checks[
        "positive_type_valid"
    ] = bool(
        (
            positive[
                "pair_type"
            ]
            ==
            "original_positive"
        ).all()
    )

    checks[
        "positive_tier_original"
    ] = bool(
        (
            positive[
                "selection_tier"
            ]
            ==
            "ORIGINAL"
        ).all()
    )

    checks[
        "positive_similarity_null"
    ] = bool(
        positive[
            "lexical_similarity"
        ]
        .isna()
        .all()
    )

    checks[
        "positive_rank_null"
    ] = bool(
        positive[
            "candidate_rank"
        ]
        .isna()
        .all()
    )

    checks[
        "positive_source_candidate_null"
    ] = bool(
        positive[
            "source_candidate_pair_id"
        ]
        .isna()
        .all()
    )

    # ========================================================
    # NEGATIVES
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

    checks[
        "negative_not_self"
    ] = bool(
        (
            negative[
                "anchor_article_uid"
            ]
            !=
            negative[
                "headline_article_uid"
            ]
        ).all()
    )

    checks[
        "negative_same_category"
    ] = bool(
        (
            negative[
                "anchor_category"
            ]
            ==
            negative[
                "headline_category"
            ]
        ).all()
        and
        negative[
            "same_category"
        ].all()
    )

    checks[
        "negative_title_different"
    ] = bool(
        (
            negative[
                "anchor_title_hash"
            ]
            !=
            negative[
                "headline_title_hash"
            ]
        ).all()
    )

    checks[
        "negative_content_different"
    ] = bool(
        (
            negative[
                "anchor_content_hash"
            ]
            !=
            negative[
                "headline_content_hash"
            ]
        ).all()
    )

    checks[
        "negative_source_candidate_present"
    ] = bool(
        negative[
            "source_candidate_pair_id"
        ]
        .notna()
        .all()
    )

    checks[
        "negative_similarity_present"
    ] = bool(
        negative[
            "lexical_similarity"
        ]
        .notna()
        .all()
    )

    checks[
        "negative_rank_present"
    ] = bool(
        negative[
            "candidate_rank"
        ]
        .notna()
        .all()
    )

    checks[
        "negative_rank_4_10"
    ] = bool(
        (
            (
                negative[
                    "candidate_rank"
                ]
                >=
                MIN_RANK
            )
            &
            (
                negative[
                    "candidate_rank"
                ]
                <=
                MAX_RANK
            )
        ).all()
    )

    tier_values = set(
        negative[
            "selection_tier"
        ]
        .unique()
    )

    checks[
        "negative_tiers_valid"
    ] = (
        tier_values
        .issubset(
            {
                "PRIMARY",
                "FALLBACK",
            }
        )
    )

    primary = (
        negative[
            negative[
                "selection_tier"
            ]
            ==
            "PRIMARY"
        ]
    )

    fallback = (
        negative[
            negative[
                "selection_tier"
            ]
            ==
            "FALLBACK"
        ]
    )

    checks[
        "primary_similarity_rule"
    ] = bool(
        (
            (
                primary[
                    "lexical_similarity"
                ]
                >=
                PRIMARY_MIN
            )
            &
            (
                primary[
                    "lexical_similarity"
                ]
                <
                PRIMARY_MAX
            )
        ).all()
    )

    checks[
        "fallback_similarity_rule"
    ] = bool(
        (
            (
                fallback[
                    "lexical_similarity"
                ]
                >=
                FALLBACK_MIN
            )
            &
            (
                fallback[
                    "lexical_similarity"
                ]
                <
                FALLBACK_MAX
            )
        ).all()
    )

    # ========================================================
    # CATEGORY BALANCE
    # ========================================================

    category_table = pd.crosstab(
        frame[
            "anchor_category"
        ],
        frame[
            "label"
        ],
    )

    category_balanced = True

    if (
        0 not in category_table.columns
        or
        1 not in category_table.columns
    ):

        category_balanced = False

    else:

        category_balanced = bool(
            (
                category_table[
                    0
                ]
                ==
                category_table[
                    1
                ]
            ).all()
        )

    checks[
        "category_balance_per_label"
    ] = (
        category_balanced
    )

    # ========================================================
    # HEADLINE REUSE
    # ========================================================

    negative_reuse = (
        negative[
            "headline_article_uid"
        ]
        .value_counts()
    )

    reuse_values = (
        negative_reuse
        .to_numpy()
    )

    reuse_stats = {
        "unique_headlines":
            int(
                len(
                    negative_reuse
                )
            ),

        "mean":
            float(
                reuse_values.mean()
            )
            if len(
                reuse_values
            )
            else 0.0,

        "median":
            float(
                np.median(
                    reuse_values
                )
            )
            if len(
                reuse_values
            )
            else 0.0,

        "p95":
            float(
                np.percentile(
                    reuse_values,
                    95,
                )
            )
            if len(
                reuse_values
            )
            else 0.0,

        "p99":
            float(
                np.percentile(
                    reuse_values,
                    99,
                )
            )
            if len(
                reuse_values
            )
            else 0.0,

        "max":
            int(
                reuse_values.max()
            )
            if len(
                reuse_values
            )
            else 0,
    }

    # ========================================================
    # SIMILARITY STATS
    # ========================================================

    similarities = (
        negative[
            "lexical_similarity"
        ]
        .astype(
            np.float32
        )
        .to_numpy()
    )

    similarity_stats = {
        "min":
            float(
                similarities.min()
            ),

        "p05":
            float(
                np.percentile(
                    similarities,
                    5,
                )
            ),

        "median":
            float(
                np.median(
                    similarities
                )
            ),

        "mean":
            float(
                similarities.mean()
            ),

        "p95":
            float(
                np.percentile(
                    similarities,
                    95,
                )
            ),

        "max":
            float(
                similarities.max()
            ),
    }

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print(
        f"Anchors  : "
        f"{len(anchor_sizes):,}"
    )

    print(
        f"Positive : "
        f"{positive_count:,}"
    )

    print(
        f"Negative : "
        f"{negative_count:,}"
    )

    print()
    print(
        "Negative tiers:"
    )

    tier_counts = (
        negative[
            "selection_tier"
        ]
        .value_counts()
    )

    for tier, count in (
        tier_counts.items()
    ):

        print(
            f"  {tier:10s}: "
            f"{count:,} "
            f"({count / len(negative):.2%})"
        )

    print()
    print(
        "Negative similarity:"
    )

    print(
        f"  min    : "
        f"{similarity_stats['min']:.4f}"
    )

    print(
        f"  median : "
        f"{similarity_stats['median']:.4f}"
    )

    print(
        f"  mean   : "
        f"{similarity_stats['mean']:.4f}"
    )

    print(
        f"  p95    : "
        f"{similarity_stats['p95']:.4f}"
    )

    print(
        f"  max    : "
        f"{similarity_stats['max']:.4f}"
    )

    print()
    print(
        "Headline reuse:"
    )

    print(
        f"  unique : "
        f"{reuse_stats['unique_headlines']:,}"
    )

    print(
        f"  p95    : "
        f"{reuse_stats['p95']:.1f}"
    )

    print(
        f"  p99    : "
        f"{reuse_stats['p99']:.1f}"
    )

    print(
        f"  max    : "
        f"{reuse_stats['max']}"
    )

    print()
    print(
        "Checks:"
    )

    passed = True

    for name, result in (
        checks.items()
    ):

        print(
            f"  {name:38s}: "
            f"{result}"
        )

        if not result:
            passed = False

    print()

    if passed:

        print(
            f"{split.upper()} VALIDATION: PASSED"
        )

    else:

        print(
            f"{split.upper()} VALIDATION: FAILED"
        )

    # ========================================================
    # STATS ROW
    # ========================================================

    split_stats = {
        "split":
            split,

        "pairs":
            int(
                len(frame)
            ),

        "anchors":
            int(
                len(
                    anchor_sizes
                )
            ),

        "positive":
            positive_count,

        "negative":
            negative_count,

        "primary":
            int(
                (
                    negative[
                        "selection_tier"
                    ]
                    ==
                    "PRIMARY"
                ).sum()
            ),

        "fallback":
            int(
                (
                    negative[
                        "selection_tier"
                    ]
                    ==
                    "FALLBACK"
                ).sum()
            ),

        "similarity_mean":
            similarity_stats[
                "mean"
            ],

        "similarity_median":
            similarity_stats[
                "median"
            ],

        "similarity_p95":
            similarity_stats[
                "p95"
            ],

        "headline_reuse_p95":
            reuse_stats[
                "p95"
            ],

        "headline_reuse_p99":
            reuse_stats[
                "p99"
            ],

        "headline_reuse_max":
            reuse_stats[
                "max"
            ],

        "validation_passed":
            passed,
    }

    # ========================================================
    # CATEGORY ROWS
    # ========================================================

    category_rows = []

    category_counts = (
        frame.groupby(
            [
                "anchor_category",
                "label",
            ]
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    for row in (
        category_counts.itertuples(
            index=False
        )
    ):

        category_rows.append(
            {
                "split":
                    split,

                "category":
                    row.anchor_category,

                "label":
                    int(
                        row.label
                    ),

                "label_name":
                    (
                        "CONSISTENT"
                        if row.label == 1
                        else "INCONSISTENT"
                    ),

                "count":
                    int(
                        row.count
                    ),
            }
        )

    # ========================================================
    # TIER ROWS
    # ========================================================

    tier_rows = []

    for tier, tier_frame in (
        negative.groupby(
            "selection_tier"
        )
    ):

        values = (
            tier_frame[
                "lexical_similarity"
            ]
            .astype(
                np.float32
            )
            .to_numpy()
        )

        tier_rows.append(
            {
                "split":
                    split,

                "tier":
                    tier,

                "count":
                    int(
                        len(
                            tier_frame
                        )
                    ),

                "rate":
                    float(
                        len(
                            tier_frame
                        )
                        /
                        len(
                            negative
                        )
                    ),

                "similarity_min":
                    float(
                        values.min()
                    ),

                "similarity_median":
                    float(
                        np.median(
                            values
                        )
                    ),

                "similarity_mean":
                    float(
                        values.mean()
                    ),

                "similarity_max":
                    float(
                        values.max()
                    ),
            }
        )

    return {
        "frame":
            frame,

        "checks":
            checks,

        "passed":
            passed,

        "split_stats":
            split_stats,

        "category_rows":
            category_rows,

        "tier_rows":
            tier_rows,

        "similarity_stats":
            similarity_stats,

        "reuse_stats":
            reuse_stats,
    }


# ============================================================
# CROSS-SPLIT VALIDATION
# ============================================================

def validate_cross_split(
    split_results,
):

    print()
    print("=" * 80)
    print(
        "CROSS-SPLIT VALIDATION"
    )
    print("=" * 80)

    used_article_sets = {}

    pair_id_sets = {}

    anchor_sets = {}

    for split in VALID_SPLITS:

        frame = (
            split_results[
                split
            ][
                "frame"
            ]
        )

        used_article_sets[
            split
        ] = (
            set(
                frame[
                    "anchor_article_uid"
                ]
            )
            |
            set(
                frame[
                    "headline_article_uid"
                ]
            )
        )

        pair_id_sets[
            split
        ] = set(
            frame[
                "pair_id"
            ]
        )

        anchor_sets[
            split
        ] = set(
            frame[
                "anchor_article_uid"
            ]
        )

    overlaps = {}

    passed = True

    for i in range(
        len(
            VALID_SPLITS
        )
    ):

        for j in range(
            i + 1,
            len(
                VALID_SPLITS
            ),
        ):

            a = (
                VALID_SPLITS[i]
            )

            b = (
                VALID_SPLITS[j]
            )

            article_overlap = len(
                used_article_sets[
                    a
                ]
                &
                used_article_sets[
                    b
                ]
            )

            anchor_overlap = len(
                anchor_sets[
                    a
                ]
                &
                anchor_sets[
                    b
                ]
            )

            pair_overlap = len(
                pair_id_sets[
                    a
                ]
                &
                pair_id_sets[
                    b
                ]
            )

            key = (
                f"{a}__{b}"
            )

            overlaps[
                key
            ] = {
                "article_uid":
                    int(
                        article_overlap
                    ),

                "anchor_uid":
                    int(
                        anchor_overlap
                    ),

                "pair_id":
                    int(
                        pair_overlap
                    ),
            }

            print(
                f"{a} ∩ {b}"
            )

            print(
                f"  article UID: "
                f"{article_overlap:,}"
            )

            print(
                f"  anchor UID : "
                f"{anchor_overlap:,}"
            )

            print(
                f"  pair ID    : "
                f"{pair_overlap:,}"
            )

            if any(
                [
                    article_overlap,
                    anchor_overlap,
                    pair_overlap,
                ]
            ):

                passed = False

    print()

    if passed:

        print(
            "CROSS-SPLIT VALIDATION: PASSED"
        )

    else:

        print(
            "CROSS-SPLIT VALIDATION: FAILED"
        )

    return (
        passed,
        overlaps,
    )


# ============================================================
# BUILD TRAIN-ONLY JOIN REVIEW SAMPLE
# ============================================================

def build_train_review_sample(
    train_frame,
    metadata,
):

    print()
    print("=" * 80)
    print(
        "BUILD TRAIN-ONLY JOIN REVIEW SAMPLE"
    )
    print("=" * 80)

    positive = (
        train_frame[
            train_frame[
                "label"
            ]
            ==
            1
        ]
        .copy()
    )

    negative = (
        train_frame[
            train_frame[
                "label"
            ]
            ==
            0
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Deterministic sample.
    # --------------------------------------------------------

    positive[
        "__sample_key"
    ] = (
        positive[
            "pair_id"
        ]
        .map(
            deterministic_key
        )
    )

    negative[
        "__sample_key"
    ] = (
        negative[
            "pair_id"
        ]
        .map(
            deterministic_key
        )
    )

    positive_sample = (
        positive.sort_values(
            "__sample_key"
        )
        .head(
            TRAIN_REVIEW_PER_TYPE
        )
        .copy()
    )

    # --------------------------------------------------------
    # Negative sample:
    # try to include PRIMARY and FALLBACK.
    # --------------------------------------------------------

    primary = (
        negative[
            negative[
                "selection_tier"
            ]
            ==
            "PRIMARY"
        ]
        .sort_values(
            "__sample_key"
        )
        .head(
            TRAIN_REVIEW_PER_TYPE
            //
            2
        )
    )

    fallback = (
        negative[
            negative[
                "selection_tier"
            ]
            ==
            "FALLBACK"
        ]
        .sort_values(
            "__sample_key"
        )
        .head(
            TRAIN_REVIEW_PER_TYPE
            -
            len(
                primary
            )
        )
    )

    negative_sample = pd.concat(
        [
            primary,
            fallback,
        ],
        ignore_index=True,
    )

    sample = pd.concat(
        [
            positive_sample,
            negative_sample,
        ],
        ignore_index=True,
    )

    anchor_uids = (
        sample[
            "anchor_article_uid"
        ]
        .unique()
        .tolist()
    )

    # --------------------------------------------------------
    # Load full TRAIN content only for sampled anchors.
    # --------------------------------------------------------

    article_dataset = ds.dataset(
        str(
            ARTICLE_FILES[
                "train"
            ]
        ),
        format="parquet",
    )

    table = article_dataset.to_table(
        columns=[
            "article_uid",
            "content",
        ],
        filter=(
            ds.field(
                "article_uid"
            )
            .isin(
                anchor_uids
            )
        ),
    )

    article_frame = (
        table.to_pandas()
    )

    content_map = dict(
        zip(
            article_frame[
                "article_uid"
            ],
            article_frame[
                "content"
            ],
        )
    )

    if (
        len(
            content_map
        )
        !=
        len(
            anchor_uids
        )
    ):

        raise RuntimeError(
            "Could not materialize all "
            "sampled TRAIN anchor contents."
        )

    rows = []

    for row in (
        sample.itertuples(
            index=False
        )
    ):

        anchor_uid = (
            row.anchor_article_uid
        )

        headline_uid = (
            row.headline_article_uid
        )

        anchor_meta = (
            metadata.loc[
                anchor_uid
            ]
        )

        headline_meta = (
            metadata.loc[
                headline_uid
            ]
        )

        content = (
            content_map[
                anchor_uid
            ]
        )

        rows.append(
            {
                "pair_id":
                    row.pair_id,

                "label":
                    int(
                        row.label
                    ),

                "label_name":
                    row.label_name,

                "selection_tier":
                    row.selection_tier,

                "candidate_rank":
                    (
                        int(
                            row.candidate_rank
                        )
                        if pd.notna(
                            row.candidate_rank
                        )
                        else None
                    ),

                "lexical_similarity":
                    (
                        float(
                            row.lexical_similarity
                        )
                        if pd.notna(
                            row.lexical_similarity
                        )
                        else None
                    ),

                "category":
                    row.anchor_category,

                "anchor_article_uid":
                    anchor_uid,

                "ANCHOR_ORIGINAL_TITLE":
                    anchor_meta[
                        "title"
                    ],

                "HEADLINE_USED_IN_PAIR":
                    headline_meta[
                        "title"
                    ],

                "ANCHOR_DESCRIPTION":
                    anchor_meta[
                        "description"
                    ],

                "ANCHOR_CONTENT_PREVIEW":
                    content[
                        :CONTENT_PREVIEW_CHARS
                    ],

                "anchor_content_words":
                    int(
                        anchor_meta[
                            "content_word_count"
                        ]
                    ),

                "join_check":
                    "",
            }
        )

    output = pd.DataFrame(
        rows
    )

    output.to_csv(
        TRAIN_REVIEW_SAMPLE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Rows: "
        f"{len(output):,}"
    )

    print(
        f"Saved:\n"
        f"{TRAIN_REVIEW_SAMPLE_PATH}"
    )

    return int(
        len(
            output
        )
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
        "STEP 10 - FINAL DATASET VALIDATION"
    )
    print("=" * 80)

    print()
    print(
        "Policy:"
    )

    print(
        "  Structural validation: "
        "TRAIN + VALIDATION + TEST"
    )

    print(
        "  Manual join sample: "
        "TRAIN ONLY"
    )

    print(
        "  Test semantics are NOT "
        "used for tuning."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_input_files()

    metadata = (
        load_article_index()
    )

    split_results = {}

    split_stats_rows = []

    category_rows = []

    tier_rows = []

    all_passed = True

    # ========================================================
    # VALIDATE ALL SPLITS
    # ========================================================

    for split in VALID_SPLITS:

        result = validate_split(
            split=split,
            metadata=metadata,
        )

        split_results[
            split
        ] = result

        split_stats_rows.append(
            result[
                "split_stats"
            ]
        )

        category_rows.extend(
            result[
                "category_rows"
            ]
        )

        tier_rows.extend(
            result[
                "tier_rows"
            ]
        )

        if not result[
            "passed"
        ]:

            all_passed = False

    # ========================================================
    # CROSS-SPLIT
    # ========================================================

    (
        cross_split_passed,
        cross_split_overlap,
    ) = validate_cross_split(
        split_results
    )

    if not cross_split_passed:

        all_passed = False

    # ========================================================
    # SAVE DISTRIBUTIONS
    # ========================================================

    pd.DataFrame(
        split_stats_rows
    ).to_csv(
        SPLIT_STATS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        category_rows
    ).to_csv(
        CATEGORY_STATS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        tier_rows
    ).to_csv(
        NEGATIVE_TIER_STATS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # TRAIN-ONLY MATERIALIZATION SAMPLE
    # ========================================================

    train_review_rows = (
        build_train_review_sample(
            train_frame=(
                split_results[
                    "train"
                ][
                    "frame"
                ]
            ),
            metadata=metadata,
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
            "10_final_dataset_validation",

        "dataset":
            "headline_consistency_v1",

        "semantic_review_policy": {
            "manual_join_sample_split":
                "train",

            "validation_semantics_used":
                False,

            "test_semantics_used":
                False,

            "reason":
                (
                    "Validation/test are kept "
                    "untouched for model selection "
                    "and final evaluation."
                ),
        },

        "splits": {
            split: {
                "stats":
                    split_results[
                        split
                    ][
                        "split_stats"
                    ],

                "checks":
                    {
                        key:
                            bool(value)

                        for key, value
                        in split_results[
                            split
                        ][
                            "checks"
                        ].items()
                    },

                "similarity":
                    split_results[
                        split
                    ][
                        "similarity_stats"
                    ],

                "headline_reuse":
                    split_results[
                        split
                    ][
                        "reuse_stats"
                    ],

                "passed":
                    bool(
                        split_results[
                            split
                        ][
                            "passed"
                        ]
                    ),
            }

            for split
            in VALID_SPLITS
        },

        "cross_split": {
            "passed":
                bool(
                    cross_split_passed
                ),

            "overlap":
                cross_split_overlap,
        },

        "train_join_review_rows":
            int(
                train_review_rows
            ),

        "overall_validation_passed":
            bool(
                all_passed
            ),

        "outputs": {
            "split_stats":
                str(
                    SPLIT_STATS_PATH
                    .relative_to(
                        MODULE_ROOT
                    )
                ),

            "category_stats":
                str(
                    CATEGORY_STATS_PATH
                    .relative_to(
                        MODULE_ROOT
                    )
                ),

            "negative_tier_stats":
                str(
                    NEGATIVE_TIER_STATS_PATH
                    .relative_to(
                        MODULE_ROOT
                    )
                ),

            "train_join_review_sample":
                str(
                    TRAIN_REVIEW_SAMPLE_PATH
                    .relative_to(
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
    # DROP LARGE DATAFRAMES BEFORE SUMMARY
    # ========================================================

    for split in VALID_SPLITS:

        split_results[
            split
        ].pop(
            "frame",
            None,
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print(
        "STEP 10 SUMMARY"
    )
    print("=" * 80)

    for row in (
        split_stats_rows
    ):

        print()
        print(
            f"{row['split']:12s}:"
        )

        print(
            f"  pairs     : "
            f"{row['pairs']:,}"
        )

        print(
            f"  anchors   : "
            f"{row['anchors']:,}"
        )

        print(
            f"  positive  : "
            f"{row['positive']:,}"
        )

        print(
            f"  negative  : "
            f"{row['negative']:,}"
        )

        print(
            f"  PRIMARY   : "
            f"{row['primary']:,}"
        )

        print(
            f"  FALLBACK  : "
            f"{row['fallback']:,}"
        )

        print(
            f"  sim median: "
            f"{row['similarity_median']:.4f}"
        )

        print(
            f"  validation: "
            f"{row['validation_passed']}"
        )

    print()
    print(
        f"TRAIN review sample: "
        f"{train_review_rows:,} rows"
    )

    print()
    print(
        f"Report:\n"
        f"{REPORT_PATH}"
    )

    print()

    if all_passed:

        print(
            "FINAL DATASET VALIDATION: PASSED"
        )

        print(
            "headline_consistency_v1 "
            "is structurally ready to lock."
        )

    else:

        raise RuntimeError(
            "headline_consistency_v1 "
            "failed final validation."
        )


if __name__ == "__main__":
    main()