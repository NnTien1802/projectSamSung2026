from pathlib import Path
from collections import Counter
import argparse
import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# PATHS
# ============================================================

MODULE_ROOT = Path(__file__).resolve().parents[2]

ARTICLE_INDEX_PATH = (
    MODULE_ROOT
    / "data"
    / "interim"
    / "consistency"
    / "articles_v1"
    / "article_index.parquet"
)

POSITIVE_DIR = (
    MODULE_ROOT
    / "data"
    / "interim"
    / "consistency"
    / "positive_pairs_v1"
)

NEGATIVE_DIR = (
    MODULE_ROOT
    / "data"
    / "interim"
    / "consistency"
    / "negative_pairs_v1"
)

OUTPUT_DIR = (
    MODULE_ROOT
    / "data"
    / "processed"
    / "consistency"
    / "headline_consistency_v1"
)

POSITIVE_FILES = {
    "train":
        POSITIVE_DIR / "train.parquet",

    "validation":
        POSITIVE_DIR / "validation.parquet",

    "test":
        POSITIVE_DIR / "test.parquet",
}

NEGATIVE_FILES = {
    "train":
        NEGATIVE_DIR / "train.parquet",

    "validation":
        NEGATIVE_DIR / "validation.parquet",

    "test":
        NEGATIVE_DIR / "test.parquet",
}

OUTPUT_FILES = {
    "train":
        OUTPUT_DIR / "train.parquet",

    "validation":
        OUTPUT_DIR / "validation.parquet",

    "test":
        OUTPUT_DIR / "test.parquet",
}

STATS_PATH = (
    OUTPUT_DIR
    / "dataset_stats.csv"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "dataset_manifest.json"
)


# ============================================================
# CONFIG
# ============================================================

VALID_SPLITS = [
    "train",
    "validation",
    "test",
]

LABEL_NAMES = {
    0: "INCONSISTENT",
    1: "CONSISTENT",
}


# ============================================================
# FINAL DATASET SCHEMA
# ============================================================

FINAL_SCHEMA = pa.schema(
    [
        (
            "pair_id",
            pa.string(),
        ),

        (
            "split",
            pa.string(),
        ),

        (
            "anchor_article_uid",
            pa.string(),
        ),

        (
            "headline_article_uid",
            pa.string(),
        ),

        (
            "label",
            pa.int8(),
        ),

        (
            "label_name",
            pa.string(),
        ),

        (
            "pair_type",
            pa.string(),
        ),

        (
            "difficulty",
            pa.string(),
        ),

        (
            "selection_tier",
            pa.string(),
        ),

        (
            "anchor_category",
            pa.string(),
        ),

        (
            "headline_category",
            pa.string(),
        ),

        (
            "same_category",
            pa.bool_(),
        ),

        (
            "anchor_title_hash",
            pa.string(),
        ),

        (
            "headline_title_hash",
            pa.string(),
        ),

        (
            "anchor_content_hash",
            pa.string(),
        ),

        (
            "headline_content_hash",
            pa.string(),
        ),

        (
            "lexical_similarity",
            pa.float32(),
        ),

        (
            "candidate_rank",
            pa.int16(),
        ),

        (
            "source_candidate_pair_id",
            pa.string(),
        ),

        (
            "source_dataset",
            pa.string(),
        ),
    ]
)


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


def human_size(
    num_bytes,
):

    value = float(
        num_bytes
    )

    for unit in [
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ]:

        if value < 1024:

            return (
                f"{value:.2f} "
                f"{unit}"
            )

        value /= 1024

    return (
        f"{value:.2f} PB"
    )


# ============================================================
# OUTPUT SAFETY
# ============================================================

def prepare_output_directory(
    overwrite=False,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated = (
        list(
            OUTPUT_FILES.values()
        )
        +
        [
            STATS_PATH,
            MANIFEST_PATH,
        ]
    )

    existing = [
        path
        for path in generated
        if path.exists()
    ]

    if (
        existing
        and
        not overwrite
    ):

        existing_text = "\n".join(
            str(path)
            for path in existing
        )

        raise RuntimeError(
            "Step 09 outputs already exist:\n"
            f"{existing_text}\n\n"
            "Use --overwrite to regenerate."
        )

    if overwrite:

        for path in existing:
            path.unlink()


# ============================================================
# VALIDATE INPUT PATHS
# ============================================================

def validate_input_paths():

    print()
    print("=" * 80)
    print(
        "VALIDATE INPUT FILES"
    )
    print("=" * 80)

    paths = [
        ARTICLE_INDEX_PATH,
    ]

    paths.extend(
        POSITIVE_FILES.values()
    )

    paths.extend(
        NEGATIVE_FILES.values()
    )

    for path in paths:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

        print(
            f"FOUND: {path}"
        )

    print()
    print(
        "Input file validation: PASSED"
    )


# ============================================================
# LOAD ARTICLE METADATA
# ============================================================

def load_article_metadata():

    print()
    print("=" * 80)
    print(
        "LOAD ARTICLE METADATA"
    )
    print("=" * 80)

    columns = [
        "article_uid",
        "dataset",
        "split",
        "category",
        "title_hash",
        "content_hash",
    ]

    table = pq.read_table(
        ARTICLE_INDEX_PATH,
        columns=columns,
    )

    df = table.to_pandas()

    if df[
        "article_uid"
    ].duplicated().any():

        raise ValueError(
            "Duplicate article_uid "
            "in article index."
        )

    metadata = {}

    for row in df.itertuples(
        index=False,
    ):

        metadata[
            row.article_uid
        ] = {
            "dataset":
                row.dataset,

            "split":
                row.split,

            "category":
                row.category,

            "title_hash":
                row.title_hash,

            "content_hash":
                row.content_hash,
        }

    print(
        f"Metadata loaded: "
        f"{len(metadata):,}"
    )

    return metadata


# ============================================================
# BUILD POSITIVE RECORD
# ============================================================

def make_positive_record(
    row,
    metadata,
    split,
):

    anchor_uid = (
        row.anchor_article_uid
    )

    headline_uid = (
        row.headline_article_uid
    )

    if anchor_uid != headline_uid:

        raise RuntimeError(
            "Positive pair is not a self pair."
        )

    if anchor_uid not in metadata:

        raise RuntimeError(
            f"Positive article missing: "
            f"{anchor_uid}"
        )

    article_meta = (
        metadata[
            anchor_uid
        ]
    )

    if (
        article_meta[
            "split"
        ]
        != split
    ):

        raise RuntimeError(
            "Positive split mismatch."
        )

    return {
        "pair_id":
            row.pair_id,

        "split":
            split,

        "anchor_article_uid":
            anchor_uid,

        "headline_article_uid":
            headline_uid,

        "label":
            1,

        "label_name":
            "CONSISTENT",

        "pair_type":
            "original_positive",

        "difficulty":
            "positive",

        "selection_tier":
            "ORIGINAL",

        "anchor_category":
            article_meta[
                "category"
            ],

        "headline_category":
            article_meta[
                "category"
            ],

        "same_category":
            True,

        "anchor_title_hash":
            article_meta[
                "title_hash"
            ],

        "headline_title_hash":
            article_meta[
                "title_hash"
            ],

        "anchor_content_hash":
            article_meta[
                "content_hash"
            ],

        "headline_content_hash":
            article_meta[
                "content_hash"
            ],

        "lexical_similarity":
            None,

        "candidate_rank":
            None,

        "source_candidate_pair_id":
            None,

        "source_dataset":
            article_meta[
                "dataset"
            ],
    }


# ============================================================
# BUILD NEGATIVE RECORD
# ============================================================

def make_negative_record(
    row,
    metadata,
    split,
):

    anchor_uid = (
        row.anchor_article_uid
    )

    headline_uid = (
        row.headline_article_uid
    )

    if anchor_uid not in metadata:

        raise RuntimeError(
            f"Negative anchor missing: "
            f"{anchor_uid}"
        )

    if headline_uid not in metadata:

        raise RuntimeError(
            f"Negative headline article missing: "
            f"{headline_uid}"
        )

    anchor_meta = (
        metadata[
            anchor_uid
        ]
    )

    headline_meta = (
        metadata[
            headline_uid
        ]
    )

    if (
        anchor_meta[
            "split"
        ]
        != split
    ):

        raise RuntimeError(
            "Negative anchor split mismatch."
        )

    if (
        headline_meta[
            "split"
        ]
        != split
    ):

        raise RuntimeError(
            "Negative headline split mismatch."
        )

    if anchor_uid == headline_uid:

        raise RuntimeError(
            "Negative self pair detected."
        )

    if (
        anchor_meta[
            "title_hash"
        ]
        ==
        headline_meta[
            "title_hash"
        ]
    ):

        raise RuntimeError(
            "Negative same-title pair detected."
        )

    if (
        anchor_meta[
            "content_hash"
        ]
        ==
        headline_meta[
            "content_hash"
        ]
    ):

        raise RuntimeError(
            "Negative same-content pair detected."
        )

    return {
        "pair_id":
            row.pair_id,

        "split":
            split,

        "anchor_article_uid":
            anchor_uid,

        "headline_article_uid":
            headline_uid,

        "label":
            0,

        "label_name":
            "INCONSISTENT",

        "pair_type":
            row.pair_type,

        "difficulty":
            row.difficulty,

        "selection_tier":
            row.selection_tier,

        "anchor_category":
            anchor_meta[
                "category"
            ],

        "headline_category":
            headline_meta[
                "category"
            ],

        "same_category":
            bool(
                anchor_meta[
                    "category"
                ]
                ==
                headline_meta[
                    "category"
                ]
            ),

        "anchor_title_hash":
            anchor_meta[
                "title_hash"
            ],

        "headline_title_hash":
            headline_meta[
                "title_hash"
            ],

        "anchor_content_hash":
            anchor_meta[
                "content_hash"
            ],

        "headline_content_hash":
            headline_meta[
                "content_hash"
            ],

        "lexical_similarity":
            float(
                row.lexical_similarity
            ),

        "candidate_rank":
            int(
                row.candidate_rank
            ),

        "source_candidate_pair_id":
            row.source_candidate_pair_id,

        "source_dataset":
            anchor_meta[
                "dataset"
            ],
    }


# ============================================================
# BUILD ONE SPLIT
# ============================================================

def build_split(
    split,
    metadata,
):

    print()
    print("=" * 80)
    print(
        f"BUILD FINAL SPLIT - "
        f"{split.upper()}"
    )
    print("=" * 80)

    # ========================================================
    # LOAD NEGATIVES
    # ========================================================

    negative_df = (
        pq.read_table(
            NEGATIVE_FILES[
                split
            ]
        )
        .to_pandas()
    )

    if negative_df[
        "anchor_article_uid"
    ].duplicated().any():

        raise RuntimeError(
            f"{split}: duplicate negative anchors."
        )

    if not (
        negative_df[
            "label"
        ]
        ==
        0
    ).all():

        raise RuntimeError(
            f"{split}: invalid negative label."
        )

    negative_anchors = set(
        negative_df[
            "anchor_article_uid"
        ]
    )

    print(
        f"Negative anchors: "
        f"{len(negative_anchors):,}"
    )

    # ========================================================
    # LOAD POSITIVES
    # ========================================================

    positive_df = (
        pq.read_table(
            POSITIVE_FILES[
                split
            ]
        )
        .to_pandas()
    )

    if positive_df[
        "anchor_article_uid"
    ].duplicated().any():

        raise RuntimeError(
            f"{split}: duplicate positive anchors."
        )

    if not (
        positive_df[
            "label"
        ]
        ==
        1
    ).all():

        raise RuntimeError(
            f"{split}: invalid positive label."
        )

    # --------------------------------------------------------
    # Keep exactly the positive corresponding to anchors
    # that successfully obtained a negative in Step 08.
    # --------------------------------------------------------

    positive_df = (
        positive_df[
            positive_df[
                "anchor_article_uid"
            ].isin(
                negative_anchors
            )
        ]
        .copy()
    )

    positive_anchors = set(
        positive_df[
            "anchor_article_uid"
        ]
    )

    if (
        positive_anchors
        !=
        negative_anchors
    ):

        missing_positive = (
            negative_anchors
            -
            positive_anchors
        )

        missing_negative = (
            positive_anchors
            -
            negative_anchors
        )

        raise RuntimeError(
            f"{split}: anchor mismatch. "
            f"missing_positive="
            f"{len(missing_positive)}, "
            f"missing_negative="
            f"{len(missing_negative)}"
        )

    print(
        f"Matched positives: "
        f"{len(positive_df):,}"
    )

    # ========================================================
    # BUILD COMMON RECORDS
    # ========================================================

    positive_map = {}

    for row in positive_df.itertuples(
        index=False,
    ):

        positive_map[
            row.anchor_article_uid
        ] = make_positive_record(
            row=row,
            metadata=metadata,
            split=split,
        )

    negative_map = {}

    for row in negative_df.itertuples(
        index=False,
    ):

        negative_map[
            row.anchor_article_uid
        ] = make_negative_record(
            row=row,
            metadata=metadata,
            split=split,
        )

    # ========================================================
    # INTERLEAVE POSITIVE + NEGATIVE PER ANCHOR
    #
    # Deterministic ordering:
    #
    # anchor A positive
    # anchor A negative
    # anchor B positive
    # anchor B negative
    #
    # Training DataLoader should still shuffle TRAIN.
    # ========================================================

    records = []

    for anchor_uid in sorted(
        negative_anchors
    ):

        records.append(
            positive_map[
                anchor_uid
            ]
        )

        records.append(
            negative_map[
                anchor_uid
            ]
        )

    # ========================================================
    # WRITE
    # ========================================================

    table = pa.Table.from_pylist(
        records,
        schema=FINAL_SCHEMA,
    )

    pq.write_table(
        table,
        OUTPUT_FILES[
            split
        ],
        compression="zstd",
        use_dictionary=True,
    )

    print(
        f"Final pairs: "
        f"{len(records):,}"
    )

    return {
        "anchors":
            int(
                len(
                    negative_anchors
                )
            ),

        "positive":
            int(
                len(
                    positive_df
                )
            ),

        "negative":
            int(
                len(
                    negative_df
                )
            ),

        "pairs":
            int(
                len(
                    records
                )
            ),
    }


# ============================================================
# FINAL SPLIT VALIDATION
# ============================================================

def validate_final_split(
    split,
):

    path = (
        OUTPUT_FILES[
            split
        ]
    )

    df = (
        pq.read_table(
            path
        )
        .to_pandas()
    )

    print()
    print(
        f"Validate {split}:"
    )

    passed = True

    # ========================================================
    # BASIC
    # ========================================================

    pair_ids_unique = bool(
        ~df[
            "pair_id"
        ]
        .duplicated()
        .any()
    )

    split_ok = bool(
        (
            df[
                "split"
            ]
            ==
            split
        ).all()
    )

    labels_valid = set(
        df[
            "label"
        ].unique()
    ) == {
        0,
        1,
    }

    # ========================================================
    # EXACT CLASS BALANCE
    # ========================================================

    label_counts = (
        df[
            "label"
        ]
        .value_counts()
        .to_dict()
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

    balanced = (
        positive_count
        ==
        negative_count
    )

    # ========================================================
    # EACH ANCHOR EXACTLY TWICE
    # ========================================================

    anchor_sizes = (
        df.groupby(
            "anchor_article_uid"
        )
        .size()
    )

    two_pairs_per_anchor = bool(
        (
            anchor_sizes
            ==
            2
        ).all()
    )

    # ========================================================
    # EACH ANCHOR HAS LABEL {0, 1}
    # ========================================================

    label_sum_by_anchor = (
        df.groupby(
            "anchor_article_uid"
        )[
            "label"
        ]
        .sum()
    )

    one_positive_per_anchor = bool(
        (
            label_sum_by_anchor
            ==
            1
        ).all()
    )

    # ========================================================
    # POSITIVE STRUCTURE
    # ========================================================

    positives = (
        df[
            df[
                "label"
            ]
            ==
            1
        ]
    )

    positive_self_pair = bool(
        (
            positives[
                "anchor_article_uid"
            ]
            ==
            positives[
                "headline_article_uid"
            ]
        ).all()
    )

    # ========================================================
    # NEGATIVE STRUCTURE
    # ========================================================

    negatives = (
        df[
            df[
                "label"
            ]
            ==
            0
        ]
    )

    negative_not_self = bool(
        (
            negatives[
                "anchor_article_uid"
            ]
            !=
            negatives[
                "headline_article_uid"
            ]
        ).all()
    )

    negative_title_different = bool(
        (
            negatives[
                "anchor_title_hash"
            ]
            !=
            negatives[
                "headline_title_hash"
            ]
        ).all()
    )

    negative_content_different = bool(
        (
            negatives[
                "anchor_content_hash"
            ]
            !=
            negatives[
                "headline_content_hash"
            ]
        ).all()
    )

    negative_same_category = bool(
        negatives[
            "same_category"
        ].all()
    )

    # ========================================================
    # PRINT
    # ========================================================

    print(
        f"  rows                  : "
        f"{len(df):,}"
    )

    print(
        f"  anchors               : "
        f"{len(anchor_sizes):,}"
    )

    print(
        f"  positive              : "
        f"{positive_count:,}"
    )

    print(
        f"  negative              : "
        f"{negative_count:,}"
    )

    checks = {
        "unique_pair_ids":
            pair_ids_unique,

        "split_ok":
            split_ok,

        "labels_valid":
            labels_valid,

        "balanced":
            balanced,

        "two_pairs_per_anchor":
            two_pairs_per_anchor,

        "one_positive_per_anchor":
            one_positive_per_anchor,

        "positive_self_pair":
            positive_self_pair,

        "negative_not_self":
            negative_not_self,

        "negative_title_different":
            negative_title_different,

        "negative_content_different":
            negative_content_different,

        "negative_same_category":
            negative_same_category,
    }

    for key, value in (
        checks.items()
    ):

        print(
            f"  {key:26s}: "
            f"{value}"
        )

        if not value:
            passed = False

    if passed:

        print(
            "  VALIDATION: PASSED"
        )

    else:

        print(
            "  VALIDATION: FAILED"
        )

    return {
        "rows":
            int(
                len(df)
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

        "checks":
            {
                key:
                    bool(value)

                for key, value
                in checks.items()
            },

        "passed":
            bool(
                passed
            ),
    }


# ============================================================
# CROSS-SPLIT VALIDATION
# ============================================================

def validate_cross_split():

    print()
    print("=" * 80)
    print(
        "CROSS-SPLIT VALIDATION"
    )
    print("=" * 80)

    anchor_sets = {}

    headline_sets = {}

    for split in VALID_SPLITS:

        table = pq.read_table(
            OUTPUT_FILES[
                split
            ],
            columns=[
                "anchor_article_uid",
                "headline_article_uid",
            ],
        )

        frame = (
            table.to_pandas()
        )

        anchor_sets[
            split
        ] = set(
            frame[
                "anchor_article_uid"
            ]
        )

        headline_sets[
            split
        ] = set(
            frame[
                "headline_article_uid"
            ]
        )

    passed = True

    overlaps = {}

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

            split_a = (
                VALID_SPLITS[i]
            )

            split_b = (
                VALID_SPLITS[j]
            )

            # ------------------------------------------------
            # Strong leakage test:
            #
            # ANY article used anywhere in split A must not
            # appear anywhere in split B.
            # ------------------------------------------------

            articles_a = (
                anchor_sets[
                    split_a
                ]
                |
                headline_sets[
                    split_a
                ]
            )

            articles_b = (
                anchor_sets[
                    split_b
                ]
                |
                headline_sets[
                    split_b
                ]
            )

            overlap = len(
                articles_a
                &
                articles_b
            )

            key = (
                f"{split_a}__"
                f"{split_b}"
            )

            overlaps[
                key
            ] = int(
                overlap
            )

            print(
                f"{split_a} ∩ "
                f"{split_b}: "
                f"{overlap:,} "
                f"article UIDs"
            )

            if overlap != 0:
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
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build final balanced "
            "headline_consistency_v1 dataset."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    print("=" * 80)
    print(
        "MODULE 2 - HEADLINE INTEGRITY"
    )
    print(
        "TASK 2B - HEADLINE CONSISTENCY"
    )
    print(
        "STEP 09 - BUILD FINAL DATASET"
    )
    print("=" * 80)

    print()
    print(
        "Dataset policy:"
    )

    print(
        "  Keep only anchors with "
        "a Step 08 negative."
    )

    print(
        "  Each anchor contributes:"
    )

    print(
        "    1 CONSISTENT pair"
    )

    print(
        "    1 INCONSISTENT pair"
    )

    print(
        "  Target class balance: 50 / 50"
    )

    prepare_output_directory(
        overwrite=args.overwrite
    )

    validate_input_paths()

    metadata = (
        load_article_metadata()
    )

    build_stats = {}

    validation_stats = {}

    # ========================================================
    # BUILD SPLITS
    # ========================================================

    for split in VALID_SPLITS:

        build_stats[
            split
        ] = build_split(
            split=split,
            metadata=metadata,
        )

    # ========================================================
    # VALIDATE EACH SPLIT
    # ========================================================

    print()
    print("=" * 80)
    print(
        "FINAL STRUCTURAL VALIDATION"
    )
    print("=" * 80)

    all_passed = True

    for split in VALID_SPLITS:

        validation = (
            validate_final_split(
                split
            )
        )

        validation_stats[
            split
        ] = validation

        if not validation[
            "passed"
        ]:

            all_passed = False

    # ========================================================
    # CROSS-SPLIT
    # ========================================================

    (
        cross_split_passed,
        cross_split_overlaps,
    ) = validate_cross_split()

    if not cross_split_passed:

        all_passed = False

    if not all_passed:

        raise RuntimeError(
            "Final dataset failed "
            "structural validation."
        )

    # ========================================================
    # STATS CSV
    # ========================================================

    stats_rows = []

    for split in VALID_SPLITS:

        stats = (
            validation_stats[
                split
            ]
        )

        stats_rows.append(
            {
                "split":
                    split,

                "anchors":
                    stats[
                        "anchors"
                    ],

                "positive":
                    stats[
                        "positive"
                    ],

                "negative":
                    stats[
                        "negative"
                    ],

                "pairs":
                    stats[
                        "rows"
                    ],

                "positive_rate":
                    (
                        stats[
                            "positive"
                        ]
                        /
                        stats[
                            "rows"
                        ]
                    ),

                "negative_rate":
                    (
                        stats[
                            "negative"
                        ]
                        /
                        stats[
                            "rows"
                        ]
                    ),
            }
        )

    pd.DataFrame(
        stats_rows
    ).to_csv(
        STATS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # MANIFEST
    # ========================================================

    total_pairs = sum(
        item[
            "rows"
        ]
        for item
        in validation_stats.values()
    )

    total_anchors = sum(
        item[
            "anchors"
        ]
        for item
        in validation_stats.values()
    )

    total_positive = sum(
        item[
            "positive"
        ]
        for item
        in validation_stats.values()
    )

    total_negative = sum(
        item[
            "negative"
        ]
        for item
        in validation_stats.values()
    )

    manifest = {
        "dataset_name":
            "headline_consistency_v1",

        "module":
            "module2_headline_integrity",

        "task":
            "headline_article_consistency",

        "version":
            "v1",

        "labels": {
            "0":
                "INCONSISTENT",

            "1":
                "CONSISTENT",
        },

        "construction": {
            "positive":
                (
                    "original headline paired "
                    "with its own original article"
                ),

            "negative":
                (
                    "synthetic same-category "
                    "headline mismatch selected "
                    "using TRAIN-derived candidate "
                    "policy v1"
                ),

            "balanced_per_anchor":
                True,

            "pairs_per_anchor":
                2,

            "positive_per_anchor":
                1,

            "negative_per_anchor":
                1,

            "full_text_materialized":
                False,

            "article_storage":
                str(
                    (
                        MODULE_ROOT
                        / "data"
                        / "interim"
                        / "consistency"
                        / "articles_v1"
                    )
                    .relative_to(
                        MODULE_ROOT
                    )
                ),
        },

        "negative_policy": {
            "primary":
                {
                    "candidate_rank":
                        "4-10",

                    "similarity":
                        "[0.20, 0.40)",
                },

            "fallback":
                {
                    "candidate_rank":
                        "4-10",

                    "similarity":
                        "[0.15, 0.20)",
                },

            "same_split":
                True,

            "same_category":
                True,

            "same_title_forbidden":
                True,

            "same_content_forbidden":
                True,
        },

        "manual_policy_evidence": {
            "source":
                (
                    "240 manually reviewed "
                    "TRAIN candidate pairs"
                ),

            "overall_valid_negative_rate":
                0.80,

            "rank_4_10_valid_negative_rate":
                0.9375,

            "primary_rule_valid_rate":
                0.9333,

            "fallback_rule_valid_rate":
                1.0,

            "note":
                (
                    "These estimates are based "
                    "on small stratified manual "
                    "samples and should not be "
                    "interpreted as exact population "
                    "error rates."
                ),
        },

        "splits": {
            split: {
                "anchors":
                    validation_stats[
                        split
                    ][
                        "anchors"
                    ],

                "positive":
                    validation_stats[
                        split
                    ][
                        "positive"
                    ],

                "negative":
                    validation_stats[
                        split
                    ][
                        "negative"
                    ],

                "pairs":
                    validation_stats[
                        split
                    ][
                        "rows"
                    ],

                "file":
                    str(
                        OUTPUT_FILES[
                            split
                        ]
                        .relative_to(
                            MODULE_ROOT
                        )
                    ),
            }

            for split
            in VALID_SPLITS
        },

        "total": {
            "anchors":
                int(
                    total_anchors
                ),

            "pairs":
                int(
                    total_pairs
                ),

            "positive":
                int(
                    total_positive
                ),

            "negative":
                int(
                    total_negative
                ),
        },

        "structural_validation": {
            "passed":
                True,

            "cross_split_article_overlap":
                cross_split_overlaps,
        },

        "evaluation_warning":
            (
                "The validation/test negatives "
                "are synthetic, not human-gold. "
                "Use them for controlled synthetic "
                "evaluation. A human-verified "
                "evaluation subset should be "
                "created before reporting final "
                "real-world performance."
            ),
    }

    save_json(
        manifest,
        MANIFEST_PATH,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print(
        "STEP 09 SUMMARY"
    )
    print("=" * 80)

    for split in VALID_SPLITS:

        stats = (
            validation_stats[
                split
            ]
        )

        print()
        print(
            f"{split:12s}:"
        )

        print(
            f"  anchors  : "
            f"{stats['anchors']:,}"
        )

        print(
            f"  positive : "
            f"{stats['positive']:,}"
        )

        print(
            f"  negative : "
            f"{stats['negative']:,}"
        )

        print(
            f"  pairs    : "
            f"{stats['rows']:,}"
        )

    print()
    print(
        f"TOTAL anchors : "
        f"{total_anchors:,}"
    )

    print(
        f"TOTAL pairs   : "
        f"{total_pairs:,}"
    )

    print(
        f"Positive      : "
        f"{total_positive:,}"
    )

    print(
        f"Negative      : "
        f"{total_negative:,}"
    )

    print()
    print(
        "Output files:"
    )

    for split in VALID_SPLITS:

        path = (
            OUTPUT_FILES[
                split
            ]
        )

        print(
            f"  {path.name:25s} "
            f"{human_size(path.stat().st_size)}"
        )

    print(
        f"  {STATS_PATH.name:25s} "
        f"{human_size(STATS_PATH.stat().st_size)}"
    )

    print(
        f"  {MANIFEST_PATH.name:25s} "
        f"{human_size(MANIFEST_PATH.stat().st_size)}"
    )

    print()
    print(
        f"Dataset directory:\n"
        f"{OUTPUT_DIR}"
    )

    print()
    print(
        "STRUCTURAL VALIDATION: PASSED"
    )

    print(
        "STEP 09 FINISHED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()