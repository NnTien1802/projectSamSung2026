from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json

import numpy as np
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

CANDIDATE_DIR = (
    MODULE_ROOT
    / "data"
    / "interim"
    / "consistency"
    / "candidate_pairs_v1"
)

OUTPUT_DIR = (
    MODULE_ROOT
    / "data"
    / "interim"
    / "consistency"
    / "negative_pairs_v1"
)

CANDIDATE_FILES = {
    "train":
        CANDIDATE_DIR
        / "train.parquet",

    "validation":
        CANDIDATE_DIR
        / "validation.parquet",

    "test":
        CANDIDATE_DIR
        / "test.parquet",
}

OUTPUT_FILES = {
    "train":
        OUTPUT_DIR
        / "train.parquet",

    "validation":
        OUTPUT_DIR
        / "validation.parquet",

    "test":
        OUTPUT_DIR
        / "test.parquet",
}

REUSE_STATS_PATH = (
    OUTPUT_DIR
    / "headline_reuse_stats.csv"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "build_report.json"
)


# ============================================================
# CONFIG
# ============================================================

VALID_SPLITS = [
    "train",
    "validation",
    "test",
]

LABEL_INCONSISTENT = 0

PAIR_TYPE = (
    "synthetic_hard_negative"
)

# ------------------------------------------------------------
# Policy decided from TRAIN manual review only.
# ------------------------------------------------------------

PRIMARY_MIN_SIM = 0.20
PRIMARY_MAX_SIM = 0.40

FALLBACK_MIN_SIM = 0.15
FALLBACK_MAX_SIM = 0.20

MIN_RANK = 4
MAX_RANK = 10


# ============================================================
# OUTPUT SCHEMA
# ============================================================

NEGATIVE_SCHEMA = pa.schema(
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

def stable_pair_id(
    anchor_uid,
    headline_uid,
    label,
):
    """
    Deterministic final pair ID.
    """

    raw = (
        f"{anchor_uid}"
        f"\u241f"
        f"{headline_uid}"
        f"\u241f"
        f"{label}"
    )

    digest = hashlib.blake2b(
        raw.encode("utf-8"),
        digest_size=12,
    ).hexdigest()

    return (
        f"pair_{digest}"
    )


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


def classify_candidate(
    rank,
    similarity,
):
    """
    Apply the frozen v1 candidate policy.

    Returns:
        (priority, tier, difficulty)

    Smaller priority is better.
    """

    if (
        rank < MIN_RANK
        or
        rank > MAX_RANK
    ):
        return None

    # --------------------------------------------------------
    # PRIMARY
    # --------------------------------------------------------

    if (
        similarity
        >= PRIMARY_MIN_SIM
        and
        similarity
        < PRIMARY_MAX_SIM
    ):

        return (
            0,
            "PRIMARY",
            "hard",
        )

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if (
        similarity
        >= FALLBACK_MIN_SIM
        and
        similarity
        < FALLBACK_MAX_SIM
    ):

        return (
            1,
            "FALLBACK",
            "medium",
        )

    return None


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
            REUSE_STATS_PATH,
            REPORT_PATH,
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

        text = "\n".join(
            str(path)
            for path in existing
        )

        raise RuntimeError(
            "Step 08 outputs already exist:\n"
            f"{text}\n\n"
            "Use --overwrite to "
            "regenerate Step 08."
        )

    if overwrite:

        for path in existing:
            path.unlink()


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

    if not ARTICLE_INDEX_PATH.exists():

        raise FileNotFoundError(
            ARTICLE_INDEX_PATH
        )

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

    split_counts = Counter()

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

        split_counts[
            row.split
        ] += 1

    print(
        f"Articles loaded: "
        f"{len(metadata):,}"
    )

    for split in VALID_SPLITS:

        print(
            f"  {split:12s}: "
            f"{split_counts[split]:,}"
        )

    print(
        "Article index validation: PASSED"
    )

    return (
        metadata,
        split_counts,
    )


# ============================================================
# SELECT ONE NEGATIVE PER ANCHOR
# ============================================================

def select_negatives_for_split(
    split,
    candidate_path,
):
    """
    Stream candidate file.

    For each anchor:
      PRIMARY preferred over FALLBACK.

    Inside the same tier:
      higher similarity preferred,
      then lower candidate rank,
      then stable candidate ID.
    """

    print()
    print("=" * 80)
    print(
        f"SELECT NEGATIVES - "
        f"{split.upper()}"
    )
    print("=" * 80)

    if not candidate_path.exists():

        raise FileNotFoundError(
            candidate_path
        )

    parquet_file = (
        pq.ParquetFile(
            candidate_path
        )
    )

    print(
        f"Candidate rows: "
        f"{parquet_file.metadata.num_rows:,}"
    )

    columns = [
        "candidate_pair_id",
        "split",
        "anchor_article_uid",
        "candidate_headline_article_uid",
        "category",
        "candidate_rank",
        "lexical_similarity",
        "anchor_title_hash",
        "candidate_title_hash",
        "anchor_content_hash",
        "candidate_content_hash",
    ]

    # anchor_uid -> selected candidate
    best = {}

    eligible_primary = 0
    eligible_fallback = 0

    for batch in (
        parquet_file.iter_batches(
            batch_size=16384,
            columns=columns,
        )
    ):

        frame = batch.to_pandas()

        for row in frame.itertuples(
            index=False,
        ):

            if row.split != split:

                raise RuntimeError(
                    "Candidate split mismatch."
                )

            rank = int(
                row.candidate_rank
            )

            similarity = float(
                row.lexical_similarity
            )

            policy = classify_candidate(
                rank=rank,
                similarity=similarity,
            )

            if policy is None:
                continue

            (
                priority,
                tier,
                difficulty,
            ) = policy

            if tier == "PRIMARY":
                eligible_primary += 1
            else:
                eligible_fallback += 1

            anchor_uid = (
                row.anchor_article_uid
            )

            # ------------------------------------------------
            # Selection key:
            #
            # 1. PRIMARY before FALLBACK
            # 2. higher similarity
            # 3. lower rank
            # 4. deterministic ID
            # ------------------------------------------------

            selection_key = (
                priority,
                -similarity,
                rank,
                str(
                    row.candidate_pair_id
                ),
            )

            previous = best.get(
                anchor_uid
            )

            if (
                previous is None
                or
                selection_key
                <
                previous[
                    "_selection_key"
                ]
            ):

                best[
                    anchor_uid
                ] = {
                    "_selection_key":
                        selection_key,

                    "source_candidate_pair_id":
                        row.candidate_pair_id,

                    "anchor_article_uid":
                        anchor_uid,

                    "headline_article_uid":
                        row.candidate_headline_article_uid,

                    "category":
                        row.category,

                    "candidate_rank":
                        rank,

                    "lexical_similarity":
                        similarity,

                    "anchor_title_hash":
                        row.anchor_title_hash,

                    "headline_title_hash":
                        row.candidate_title_hash,

                    "anchor_content_hash":
                        row.anchor_content_hash,

                    "headline_content_hash":
                        row.candidate_content_hash,

                    "selection_tier":
                        tier,

                    "difficulty":
                        difficulty,
                }

    print()
    print(
        f"Eligible PRIMARY candidates : "
        f"{eligible_primary:,}"
    )

    print(
        f"Eligible FALLBACK candidates: "
        f"{eligible_fallback:,}"
    )

    print(
        f"Anchors selected            : "
        f"{len(best):,}"
    )

    return (
        best,
        eligible_primary,
        eligible_fallback,
    )


# ============================================================
# BUILD OUTPUT RECORDS
# ============================================================

def build_negative_records(
    split,
    selected,
    metadata,
):

    records = []

    tier_counts = Counter()

    headline_reuse = Counter()

    similarity_values = []

    rank_counts = Counter()

    for anchor_uid in sorted(
        selected.keys()
    ):

        item = selected[
            anchor_uid
        ]

        headline_uid = (
            item[
                "headline_article_uid"
            ]
        )

        if anchor_uid not in metadata:

            raise RuntimeError(
                f"Anchor not found: "
                f"{anchor_uid}"
            )

        if headline_uid not in metadata:

            raise RuntimeError(
                f"Headline article "
                f"not found: "
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

        # ====================================================
        # SPLIT SAFETY
        # ====================================================

        if (
            anchor_meta[
                "split"
            ]
            != split
        ):

            raise RuntimeError(
                "Anchor split mismatch."
            )

        if (
            headline_meta[
                "split"
            ]
            != split
        ):

            raise RuntimeError(
                "Cross-split negative "
                "detected."
            )

        # ====================================================
        # SELF PAIR SAFETY
        # ====================================================

        if (
            anchor_uid
            ==
            headline_uid
        ):

            raise RuntimeError(
                "Self negative detected."
            )

        # ====================================================
        # CATEGORY SAFETY
        # ====================================================

        if (
            anchor_meta[
                "category"
            ]
            !=
            headline_meta[
                "category"
            ]
        ):

            raise RuntimeError(
                "Cross-category negative "
                "detected."
            )

        # ====================================================
        # HASH SAFETY
        # ====================================================

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
                "Same-title negative "
                "detected."
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
                "Same-content negative "
                "detected."
            )

        # ----------------------------------------------------
        # Verify candidate metadata has not changed.
        # ----------------------------------------------------

        if (
            item[
                "anchor_title_hash"
            ]
            !=
            anchor_meta[
                "title_hash"
            ]
        ):

            raise RuntimeError(
                "Anchor title hash mismatch."
            )

        if (
            item[
                "headline_title_hash"
            ]
            !=
            headline_meta[
                "title_hash"
            ]
        ):

            raise RuntimeError(
                "Headline title hash mismatch."
            )

        if (
            item[
                "anchor_content_hash"
            ]
            !=
            anchor_meta[
                "content_hash"
            ]
        ):

            raise RuntimeError(
                "Anchor content hash mismatch."
            )

        if (
            item[
                "headline_content_hash"
            ]
            !=
            headline_meta[
                "content_hash"
            ]
        ):

            raise RuntimeError(
                "Headline content hash mismatch."
            )

        pair_id = stable_pair_id(
            anchor_uid=anchor_uid,
            headline_uid=headline_uid,
            label=LABEL_INCONSISTENT,
        )

        records.append(
            {
                "pair_id":
                    pair_id,

                "split":
                    split,

                "anchor_article_uid":
                    anchor_uid,

                "headline_article_uid":
                    headline_uid,

                "label":
                    LABEL_INCONSISTENT,

                "pair_type":
                    PAIR_TYPE,

                "difficulty":
                    item[
                        "difficulty"
                    ],

                "selection_tier":
                    item[
                        "selection_tier"
                    ],

                "anchor_category":
                    anchor_meta[
                        "category"
                    ],

                "headline_category":
                    headline_meta[
                        "category"
                    ],

                "same_category":
                    True,

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
                        item[
                            "lexical_similarity"
                        ]
                    ),

                "candidate_rank":
                    int(
                        item[
                            "candidate_rank"
                        ]
                    ),

                "source_candidate_pair_id":
                    item[
                        "source_candidate_pair_id"
                    ],

                "source_dataset":
                    anchor_meta[
                        "dataset"
                    ],
            }
        )

        tier_counts[
            item[
                "selection_tier"
            ]
        ] += 1

        headline_reuse[
            headline_uid
        ] += 1

        similarity_values.append(
            item[
                "lexical_similarity"
            ]
        )

        rank_counts[
            item[
                "candidate_rank"
            ]
        ] += 1

    return {
        "records":
            records,

        "tier_counts":
            tier_counts,

        "headline_reuse":
            headline_reuse,

        "similarities":
            similarity_values,

        "rank_counts":
            rank_counts,
    }


# ============================================================
# WRITE ONE SPLIT
# ============================================================

def write_split(
    split,
    records,
):

    table = pa.Table.from_pylist(
        records,
        schema=NEGATIVE_SCHEMA,
    )

    pq.write_table(
        table,
        OUTPUT_FILES[
            split
        ],
        compression="zstd",
        use_dictionary=True,
    )


# ============================================================
# VALIDATE OUTPUT
# ============================================================

def validate_split_output(
    split,
):

    path = (
        OUTPUT_FILES[
            split
        ]
    )

    table = pq.read_table(
        path
    )

    df = table.to_pandas()

    checks = {}

    checks[
        "rows"
    ] = int(
        len(df)
    )

    checks[
        "label_ok"
    ] = bool(
        (
            df[
                "label"
            ]
            ==
            LABEL_INCONSISTENT
        ).all()
    )

    checks[
        "split_ok"
    ] = bool(
        (
            df[
                "split"
            ]
            ==
            split
        ).all()
    )

    checks[
        "unique_pair_ids"
    ] = bool(
        ~df[
            "pair_id"
        ]
        .duplicated()
        .any()
    )

    checks[
        "unique_anchors"
    ] = bool(
        ~df[
            "anchor_article_uid"
        ]
        .duplicated()
        .any()
    )

    checks[
        "no_self_pairs"
    ] = bool(
        (
            df[
                "anchor_article_uid"
            ]
            !=
            df[
                "headline_article_uid"
            ]
        ).all()
    )

    checks[
        "no_same_title"
    ] = bool(
        (
            df[
                "anchor_title_hash"
            ]
            !=
            df[
                "headline_title_hash"
            ]
        ).all()
    )

    checks[
        "no_same_content"
    ] = bool(
        (
            df[
                "anchor_content_hash"
            ]
            !=
            df[
                "headline_content_hash"
            ]
        ).all()
    )

    checks[
        "same_category"
    ] = bool(
        df[
            "same_category"
        ].all()
    )

    valid_tiers = set(
        df[
            "selection_tier"
        ].unique()
    ).issubset(
        {
            "PRIMARY",
            "FALLBACK",
        }
    )

    checks[
        "valid_tiers"
    ] = bool(
        valid_tiers
    )

    passed = all(
        value
        for key, value
        in checks.items()
        if key != "rows"
    )

    checks[
        "passed"
    ] = bool(
        passed
    )

    return checks


# ============================================================
# REUSE STATISTICS
# ============================================================

def summarize_reuse(
    split,
    headline_reuse,
):

    if not headline_reuse:

        return {
            "split":
                split,

            "unique_headlines_used":
                0,

            "mean_reuse":
                0.0,

            "median_reuse":
                0.0,

            "p95_reuse":
                0.0,

            "p99_reuse":
                0.0,

            "max_reuse":
                0,
        }

    values = np.asarray(
        list(
            headline_reuse.values()
        ),
        dtype=np.int32,
    )

    return {
        "split":
            split,

        "unique_headlines_used":
            int(
                len(values)
            ),

        "mean_reuse":
            float(
                values.mean()
            ),

        "median_reuse":
            float(
                np.median(
                    values
                )
            ),

        "p95_reuse":
            float(
                np.percentile(
                    values,
                    95,
                )
            ),

        "p99_reuse":
            float(
                np.percentile(
                    values,
                    99,
                )
            ),

        "max_reuse":
            int(
                values.max()
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build negative pairs using "
            "the frozen candidate policy v1."
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
        "STEP 08 - BUILD NEGATIVE PAIRS"
    )
    print("=" * 80)

    print()
    print(
        "Frozen policy:"
    )

    print(
        "  PRIMARY : "
        "rank 4-10, "
        "0.20 <= sim < 0.40"
    )

    print(
        "  FALLBACK: "
        "rank 4-10, "
        "0.15 <= sim < 0.20"
    )

    print(
        "  Maximum : "
        "1 negative / anchor"
    )

    prepare_output_directory(
        overwrite=args.overwrite
    )

    (
        metadata,
        article_split_counts,
    ) = load_article_index()

    split_reports = {}

    reuse_rows = []

    total_negative_pairs = 0

    # ========================================================
    # EACH SPLIT IS PROCESSED INDEPENDENTLY
    # ========================================================

    for split in VALID_SPLITS:

        (
            selected,
            eligible_primary,
            eligible_fallback,
        ) = select_negatives_for_split(
            split=split,
            candidate_path=(
                CANDIDATE_FILES[
                    split
                ]
            ),
        )

        built = build_negative_records(
            split=split,
            selected=selected,
            metadata=metadata,
        )

        records = (
            built[
                "records"
            ]
        )

        write_split(
            split=split,
            records=records,
        )

        validation = (
            validate_split_output(
                split
            )
        )

        if not validation[
            "passed"
        ]:

            raise RuntimeError(
                f"{split} failed "
                "structural validation."
            )

        reuse_stats = summarize_reuse(
            split=split,
            headline_reuse=(
                built[
                    "headline_reuse"
                ]
            ),
        )

        reuse_rows.append(
            reuse_stats
        )

        selected_count = len(
            records
        )

        total_anchors = int(
            article_split_counts[
                split
            ]
        )

        coverage = (
            selected_count
            /
            total_anchors
            if total_anchors
            else 0.0
        )

        tier_counts = (
            built[
                "tier_counts"
            ]
        )

        similarities = np.asarray(
            built[
                "similarities"
            ],
            dtype=np.float32,
        )

        similarity_stats = {}

        if len(
            similarities
        ) > 0:

            similarity_stats = {
                "mean":
                    float(
                        similarities.mean()
                    ),

                "median":
                    float(
                        np.median(
                            similarities
                        )
                    ),

                "p05":
                    float(
                        np.percentile(
                            similarities,
                            5,
                        )
                    ),

                "p95":
                    float(
                        np.percentile(
                            similarities,
                            95,
                        )
                    ),

                "min":
                    float(
                        similarities.min()
                    ),

                "max":
                    float(
                        similarities.max()
                    ),
            }

        split_reports[
            split
        ] = {
            "articles":
                total_anchors,

            "eligible_primary_candidates":
                int(
                    eligible_primary
                ),

            "eligible_fallback_candidates":
                int(
                    eligible_fallback
                ),

            "negative_pairs":
                int(
                    selected_count
                ),

            "coverage":
                float(
                    coverage
                ),

            "primary_selected":
                int(
                    tier_counts.get(
                        "PRIMARY",
                        0,
                    )
                ),

            "fallback_selected":
                int(
                    tier_counts.get(
                        "FALLBACK",
                        0,
                    )
                ),

            "anchors_without_negative":
                int(
                    total_anchors
                    -
                    selected_count
                ),

            "similarity":
                similarity_stats,

            "rank_distribution": {
                str(key):
                    int(value)

                for key, value
                in built[
                    "rank_counts"
                ].items()
            },

            "headline_reuse":
                reuse_stats,

            "validation":
                validation,
        }

        total_negative_pairs += (
            selected_count
        )

        print()
        print("-" * 80)

        print(
            f"{split.upper()} SUMMARY"
        )

        print("-" * 80)

        print(
            f"Articles               : "
            f"{total_anchors:,}"
        )

        print(
            f"Negative pairs         : "
            f"{selected_count:,}"
        )

        print(
            f"Coverage               : "
            f"{coverage:.2%}"
        )

        print(
            f"PRIMARY selected       : "
            f"{tier_counts.get('PRIMARY', 0):,}"
        )

        print(
            f"FALLBACK selected      : "
            f"{tier_counts.get('FALLBACK', 0):,}"
        )

        print(
            f"No negative             : "
            f"{total_anchors - selected_count:,}"
        )

        print(
            f"Unique headlines used  : "
            f"{reuse_stats['unique_headlines_used']:,}"
        )

        print(
            f"Headline reuse p95     : "
            f"{reuse_stats['p95_reuse']:.1f}"
        )

        print(
            f"Headline reuse p99     : "
            f"{reuse_stats['p99_reuse']:.1f}"
        )

        print(
            f"Headline max reuse     : "
            f"{reuse_stats['max_reuse']}"
        )

        print(
            "Structural validation  : PASSED"
        )

    # ========================================================
    # REUSE CSV
    # ========================================================

    pd.DataFrame(
        reuse_rows
    ).to_csv(
        REUSE_STATS_PATH,
        index=False,
        encoding="utf-8-sig",
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
            "08_build_negative_pairs",

        "label":
            {
                "value":
                    0,

                "meaning":
                    "INCONSISTENT",
            },

        "policy_source":
            (
                "Step 07 TRAIN-only "
                "manual annotation analysis"
            ),

        "policy": {
            "rank_min":
                MIN_RANK,

            "rank_max":
                MAX_RANK,

            "primary_similarity":
                {
                    "min_inclusive":
                        PRIMARY_MIN_SIM,

                    "max_exclusive":
                        PRIMARY_MAX_SIM,
                },

            "fallback_similarity":
                {
                    "min_inclusive":
                        FALLBACK_MIN_SIM,

                    "max_exclusive":
                        FALLBACK_MAX_SIM,
                },

            "max_negative_per_anchor":
                1,

            "primary_before_fallback":
                True,

            "within_tier_selection":
                (
                    "highest similarity, "
                    "then lowest rank, "
                    "then stable ID"
                ),

            "same_split_only":
                True,

            "same_category_only":
                True,

            "same_title_forbidden":
                True,

            "same_content_forbidden":
                True,
        },

        "splits":
            split_reports,

        "total_negative_pairs":
            int(
                total_negative_pairs
            ),

        "outputs": {
            split:
                str(
                    path.relative_to(
                        MODULE_ROOT
                    )
                )

            for split, path
            in OUTPUT_FILES.items()
        },

        "headline_reuse_stats":
            str(
                REUSE_STATS_PATH
                .relative_to(
                    MODULE_ROOT
                )
            ),
    }

    save_json(
        report,
        REPORT_PATH,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print(
        "STEP 08 SUMMARY"
    )
    print("=" * 80)

    for split in VALID_SPLITS:

        data = (
            split_reports[
                split
            ]
        )

        print()
        print(
            f"{split:12s}: "
            f"{data['negative_pairs']:,} "
            f"negative pairs / "
            f"{data['articles']:,} articles "
            f"({data['coverage']:.2%})"
        )

        print(
            f"  PRIMARY : "
            f"{data['primary_selected']:,}"
        )

        print(
            f"  FALLBACK: "
            f"{data['fallback_selected']:,}"
        )

        print(
            f"  Missing : "
            f"{data['anchors_without_negative']:,}"
        )

    print()
    print(
        f"Total negative pairs: "
        f"{total_negative_pairs:,}"
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
        f"  {REUSE_STATS_PATH.name:25s} "
        f"{human_size(REUSE_STATS_PATH.stat().st_size)}"
    )

    print(
        f"  {REPORT_PATH.name:25s} "
        f"{human_size(REPORT_PATH.stat().st_size)}"
    )

    print()
    print(
        f"Report:\n"
        f"{REPORT_PATH}"
    )

    print()
    print(
        "STEP 08 FINISHED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()