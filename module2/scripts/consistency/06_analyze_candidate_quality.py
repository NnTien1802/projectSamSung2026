from pathlib import Path
from collections import Counter, defaultdict
import argparse
import hashlib
import json
import random

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq


# ============================================================
# PATHS
# ============================================================

MODULE_ROOT = Path(__file__).resolve().parents[2]

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

TRAIN_ARTICLES_PATH = (
    ARTICLES_DIR
    / "train.parquet"
)

CANDIDATE_DIR = (
    MODULE_ROOT
    / "data"
    / "interim"
    / "consistency"
    / "candidate_pairs_v1"
)

TRAIN_CANDIDATES_PATH = (
    CANDIDATE_DIR
    / "train.parquet"
)

OUTPUT_DIR = (
    MODULE_ROOT
    / "results"
    / "consistency"
    / "candidate_analysis_v1"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "analysis_report.json"
)

SIMILARITY_DISTRIBUTION_PATH = (
    OUTPUT_DIR
    / "similarity_distribution.csv"
)

RANK_DISTRIBUTION_PATH = (
    OUTPUT_DIR
    / "rank_distribution.csv"
)

CATEGORY_DISTRIBUTION_PATH = (
    OUTPUT_DIR
    / "category_distribution.csv"
)

MANUAL_SAMPLE_PATH = (
    OUTPUT_DIR
    / "manual_review_sample.csv"
)


# ============================================================
# CONFIG
# ============================================================

RANDOM_SEED = 20260919

DEFAULT_SAMPLE_PER_STRATUM = 10

CONTENT_PREVIEW_CHARS = 2500

SIMILARITY_BINS = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.55,
    0.70,
    1.0001,
]

SIMILARITY_BIN_NAMES = [
    "0.10-0.15",
    "0.15-0.20",
    "0.20-0.25",
    "0.25-0.30",
    "0.30-0.40",
    "0.40-0.55",
    "0.55-0.70",
    "0.70-1.00",
]

RANK_BUCKETS = {
    "rank_1": (
        1,
        1,
    ),

    "rank_2_3": (
        2,
        3,
    ),

    "rank_4_10": (
        4,
        10,
    ),
}


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


def get_similarity_bin(
    value,
):

    for index in range(
        len(
            SIMILARITY_BINS
        )
        - 1
    ):

        lower = (
            SIMILARITY_BINS[
                index
            ]
        )

        upper = (
            SIMILARITY_BINS[
                index + 1
            ]
        )

        if (
            value >= lower
            and
            value < upper
        ):

            return (
                SIMILARITY_BIN_NAMES[
                    index
                ]
            )

    return None


def get_rank_bucket(
    rank,
):

    for name, (
        lower,
        upper,
    ) in RANK_BUCKETS.items():

        if (
            rank >= lower
            and
            rank <= upper
        ):

            return name

    return "other"


def stable_sample_key(
    pair_id,
):
    """
    Deterministic pseudo-random key.

    This makes sampling reproducible
    without depending on row order.
    """

    digest = hashlib.blake2b(
        (
            str(pair_id)
            +
            str(
                RANDOM_SEED
            )
        ).encode(
            "utf-8"
        ),
        digest_size=8,
    ).digest()

    return int.from_bytes(
        digest,
        byteorder="big",
        signed=False,
    )


# ============================================================
# VALIDATE INPUTS
# ============================================================

def validate_inputs():

    print()
    print("=" * 80)
    print(
        "VALIDATE INPUTS"
    )
    print("=" * 80)

    required = [
        ARTICLE_INDEX_PATH,
        TRAIN_ARTICLES_PATH,
        TRAIN_CANDIDATES_PATH,
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

        print(
            f"FOUND: {path}"
        )

    candidate_file = (
        pq.ParquetFile(
            TRAIN_CANDIDATES_PATH
        )
    )

    print()
    print(
        "Train candidate rows: "
        f"{candidate_file.metadata.num_rows:,}"
    )

    print(
        "Input validation: PASSED"
    )


# ============================================================
# ANALYZE TRAIN CANDIDATES
# ============================================================

def analyze_candidates(
    sample_per_stratum,
):

    print()
    print("=" * 80)
    print(
        "ANALYZE TRAIN CANDIDATES"
    )
    print("=" * 80)

    parquet_file = (
        pq.ParquetFile(
            TRAIN_CANDIDATES_PATH
        )
    )

    columns = [
        "candidate_pair_id",
        "anchor_article_uid",
        "candidate_headline_article_uid",
        "category",
        "candidate_rank",
        "lexical_similarity",
    ]

    # ========================================================
    # GLOBAL STATS
    # ========================================================

    similarities = []

    rank_counts = Counter()

    category_counts = Counter()

    category_similarity_sum = Counter()

    category_top1_count = Counter()

    category_top1_similarity_sum = (
        Counter()
    )

    anchor_candidate_counts = Counter()

    similarity_bin_counts = Counter()

    rank_bucket_counts = Counter()

    # ========================================================
    # RESERVOIR-LIKE SAMPLE
    #
    # Keep the smallest deterministic hash values
    # per similarity-bin × rank-bucket.
    # ========================================================

    samples_by_stratum = defaultdict(
        list
    )

    total_rows = 0

    for batch in (
        parquet_file.iter_batches(
            batch_size=8192,
            columns=columns,
        )
    ):

        frame = (
            batch.to_pandas()
        )

        total_rows += len(
            frame
        )

        batch_sims = (
            frame[
                "lexical_similarity"
            ]
            .astype(
                np.float32
            )
            .to_numpy()
        )

        similarities.append(
            batch_sims
        )

        for row in frame.itertuples(
            index=False,
        ):

            pair_id = (
                row.candidate_pair_id
            )

            anchor_uid = (
                row.anchor_article_uid
            )

            category = (
                row.category
            )

            rank = int(
                row.candidate_rank
            )

            similarity = float(
                row.lexical_similarity
            )

            # ----------------------------------------------
            # Counts
            # ----------------------------------------------

            rank_counts[
                rank
            ] += 1

            category_counts[
                category
            ] += 1

            category_similarity_sum[
                category
            ] += similarity

            anchor_candidate_counts[
                anchor_uid
            ] += 1

            if rank == 1:

                category_top1_count[
                    category
                ] += 1

                category_top1_similarity_sum[
                    category
                ] += similarity

            sim_bin = (
                get_similarity_bin(
                    similarity
                )
            )

            rank_bucket = (
                get_rank_bucket(
                    rank
                )
            )

            if sim_bin is not None:

                similarity_bin_counts[
                    sim_bin
                ] += 1

                rank_bucket_counts[
                    rank_bucket
                ] += 1

                stratum = (
                    sim_bin,
                    rank_bucket,
                )

                key = (
                    stable_sample_key(
                        pair_id
                    )
                )

                item = {
                    "sample_key":
                        key,

                    "candidate_pair_id":
                        pair_id,

                    "anchor_article_uid":
                        anchor_uid,

                    "candidate_headline_article_uid":
                        row.candidate_headline_article_uid,

                    "category":
                        category,

                    "candidate_rank":
                        rank,

                    "lexical_similarity":
                        similarity,

                    "similarity_bin":
                        sim_bin,

                    "rank_bucket":
                        rank_bucket,
                }

                bucket = (
                    samples_by_stratum[
                        stratum
                    ]
                )

                bucket.append(
                    item
                )

                bucket.sort(
                    key=lambda x:
                    x[
                        "sample_key"
                    ]
                )

                if (
                    len(
                        bucket
                    )
                    >
                    sample_per_stratum
                ):

                    bucket.pop()

    # ========================================================
    # SIMILARITY STATS
    # ========================================================

    similarities = np.concatenate(
        similarities
    )

    percentiles = [
        0,
        1,
        5,
        10,
        25,
        50,
        75,
        90,
        95,
        99,
        100,
    ]

    percentile_values = (
        np.percentile(
            similarities,
            percentiles,
        )
    )

    print()
    print(
        f"Total candidates: "
        f"{total_rows:,}"
    )

    print()
    print(
        "Similarity percentiles:"
    )

    percentile_report = {}

    for percentile, value in zip(
        percentiles,
        percentile_values,
    ):

        print(
            f"  p{percentile:>3}: "
            f"{value:.4f}"
        )

        percentile_report[
            str(percentile)
        ] = float(
            value
        )

    # ========================================================
    # CANDIDATES PER ANCHOR
    # ========================================================

    anchor_counts = np.asarray(
        list(
            anchor_candidate_counts
            .values()
        ),
        dtype=np.int16,
    )

    print()
    print(
        "Candidates per anchor:"
    )

    print(
        f"  anchors with candidates: "
        f"{len(anchor_counts):,}"
    )

    if len(
        anchor_counts
    ) > 0:

        print(
            f"  min    : "
            f"{anchor_counts.min()}"
        )

        print(
            f"  median : "
            f"{np.median(anchor_counts):.1f}"
        )

        print(
            f"  mean   : "
            f"{anchor_counts.mean():.3f}"
        )

        print(
            f"  max    : "
            f"{anchor_counts.max()}"
        )

    # ========================================================
    # SAVE SIMILARITY BIN DISTRIBUTION
    # ========================================================

    sim_rows = []

    for name in (
        SIMILARITY_BIN_NAMES
    ):

        count = int(
            similarity_bin_counts[
                name
            ]
        )

        sim_rows.append(
            {
                "similarity_bin":
                    name,

                "count":
                    count,

                "percentage":
                    (
                        count
                        /
                        total_rows
                    ),
            }
        )

    pd.DataFrame(
        sim_rows
    ).to_csv(
        SIMILARITY_DISTRIBUTION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # SAVE RANK DISTRIBUTION
    # ========================================================

    rank_rows = []

    for rank in sorted(
        rank_counts.keys()
    ):

        count = int(
            rank_counts[
                rank
            ]
        )

        rank_rows.append(
            {
                "rank":
                    rank,

                "count":
                    count,

                "percentage":
                    (
                        count
                        /
                        total_rows
                    ),
            }
        )

    pd.DataFrame(
        rank_rows
    ).to_csv(
        RANK_DISTRIBUTION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # SAVE CATEGORY DISTRIBUTION
    # ========================================================

    category_rows = []

    for category in sorted(
        category_counts.keys()
    ):

        count = int(
            category_counts[
                category
            ]
        )

        mean_similarity = (
            category_similarity_sum[
                category
            ]
            /
            count
        )

        top1_count = int(
            category_top1_count[
                category
            ]
        )

        mean_top1 = None

        if top1_count > 0:

            mean_top1 = (
                category_top1_similarity_sum[
                    category
                ]
                /
                top1_count
            )

        category_rows.append(
            {
                "category":
                    category,

                "candidate_count":
                    count,

                "mean_similarity":
                    mean_similarity,

                "top1_count":
                    top1_count,

                "mean_top1_similarity":
                    mean_top1,
            }
        )

    pd.DataFrame(
        category_rows
    ).to_csv(
        CATEGORY_DISTRIBUTION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # FLATTEN SAMPLE
    # ========================================================

    selected_samples = []

    for stratum in sorted(
        samples_by_stratum.keys()
    ):

        selected_samples.extend(
            samples_by_stratum[
                stratum
            ]
        )

    for sample in (
        selected_samples
    ):

        sample.pop(
            "sample_key",
            None,
        )

    print()
    print(
        "Manual-review candidates selected: "
        f"{len(selected_samples):,}"
    )

    return {
        "total_rows":
            int(
                total_rows
            ),

        "percentiles":
            percentile_report,

        "anchors_with_candidates":
            int(
                len(
                    anchor_candidate_counts
                )
            ),

        "similarity_bin_counts":
            {
                key:
                    int(value)

                for key, value
                in similarity_bin_counts.items()
            },

        "rank_counts":
            {
                str(key):
                    int(value)

                for key, value
                in rank_counts.items()
            },

        "selected_samples":
            selected_samples,
    }


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
        "title",
        "description",
        "category",
        "content_word_count",
    ]

    table = pq.read_table(
        ARTICLE_INDEX_PATH,
        columns=columns,
    )

    frame = (
        table.to_pandas()
    )

    metadata = {}

    for row in frame.itertuples(
        index=False,
    ):

        metadata[
            row.article_uid
        ] = {
            "title":
                row.title,

            "description":
                row.description,

            "category":
                row.category,

            "content_word_count":
                int(
                    row.content_word_count
                ),
        }

    print(
        f"Metadata loaded: "
        f"{len(metadata):,} articles"
    )

    return metadata


# ============================================================
# LOAD ONLY SAMPLED ANCHOR CONTENT
# ============================================================

def load_sampled_anchor_content(
    anchor_uids,
):

    print()
    print("=" * 80)
    print(
        "LOAD SAMPLED ANCHOR ARTICLES"
    )
    print("=" * 80)

    anchor_uids = sorted(
        set(
            anchor_uids
        )
    )

    print(
        f"Unique anchors requested: "
        f"{len(anchor_uids):,}"
    )

    dataset = ds.dataset(
        str(
            TRAIN_ARTICLES_PATH
        ),
        format="parquet",
    )

    filter_expression = (
        ds.field(
            "article_uid"
        ).isin(
            anchor_uids
        )
    )

    table = dataset.to_table(
        columns=[
            "article_uid",
            "content",
        ],
        filter=(
            filter_expression
        ),
    )

    frame = table.to_pandas()

    content_map = {}

    for row in frame.itertuples(
        index=False,
    ):

        content_map[
            row.article_uid
        ] = row.content

    print(
        f"Articles found: "
        f"{len(content_map):,}"
    )

    missing = (
        set(
            anchor_uids
        )
        -
        set(
            content_map.keys()
        )
    )

    if missing:

        raise RuntimeError(
            "Some sampled anchors "
            "were not found in train.parquet: "
            f"{len(missing)}"
        )

    return content_map


# ============================================================
# BUILD MANUAL REVIEW FILE
# ============================================================

def build_manual_review(
    selected_samples,
):

    print()
    print("=" * 80)
    print(
        "BUILD MANUAL REVIEW SAMPLE"
    )
    print("=" * 80)

    metadata = (
        load_article_metadata()
    )

    anchor_uids = [
        sample[
            "anchor_article_uid"
        ]
        for sample
        in selected_samples
    ]

    content_map = (
        load_sampled_anchor_content(
            anchor_uids
        )
    )

    rows = []

    for sample in (
        selected_samples
    ):

        anchor_uid = (
            sample[
                "anchor_article_uid"
            ]
        )

        candidate_uid = (
            sample[
                "candidate_headline_article_uid"
            ]
        )

        anchor_metadata = (
            metadata[
                anchor_uid
            ]
        )

        candidate_metadata = (
            metadata[
                candidate_uid
            ]
        )

        content = (
            content_map[
                anchor_uid
            ]
        )

        content_preview = (
            content[
                :CONTENT_PREVIEW_CHARS
            ]
        )

        rows.append(
            {
                "candidate_pair_id":
                    sample[
                        "candidate_pair_id"
                    ],

                "category":
                    sample[
                        "category"
                    ],

                "similarity_bin":
                    sample[
                        "similarity_bin"
                    ],

                "rank_bucket":
                    sample[
                        "rank_bucket"
                    ],

                "candidate_rank":
                    sample[
                        "candidate_rank"
                    ],

                "lexical_similarity":
                    sample[
                        "lexical_similarity"
                    ],

                "anchor_article_uid":
                    anchor_uid,

                "anchor_title_original":
                    anchor_metadata[
                        "title"
                    ],

                "anchor_description":
                    anchor_metadata[
                        "description"
                    ],

                "anchor_content_preview":
                    content_preview,

                "anchor_content_words":
                    anchor_metadata[
                        "content_word_count"
                    ],

                "candidate_headline_article_uid":
                    candidate_uid,

                "candidate_headline":
                    candidate_metadata[
                        "title"
                    ],

                "candidate_description":
                    candidate_metadata[
                        "description"
                    ],

                # ============================================
                # MANUAL REVIEW FIELDS
                # ============================================

                "manual_decision":
                    "",

                "manual_reason":
                    "",
            }
        )

    review_df = pd.DataFrame(
        rows
    )

    review_df.sort_values(
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

    review_df.to_csv(
        MANUAL_SAMPLE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        f"Manual review rows: "
        f"{len(review_df):,}"
    )

    print(
        f"Saved to:\n"
        f"{MANUAL_SAMPLE_PATH}"
    )

    return int(
        len(
            review_df
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Analyze Step 05 candidate "
            "quality using TRAIN only."
        )
    )

    parser.add_argument(
        "--sample-per-stratum",
        type=int,
        default=(
            DEFAULT_SAMPLE_PER_STRATUM
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    if (
        args.sample_per_stratum
        < 1
    ):

        raise ValueError(
            "--sample-per-stratum "
            "must be >= 1"
        )

    print("=" * 80)
    print(
        "MODULE 2 - HEADLINE INTEGRITY"
    )
    print(
        "TASK 2B - HEADLINE CONSISTENCY"
    )
    print(
        "STEP 06 - ANALYZE CANDIDATE QUALITY"
    )
    print("=" * 80)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "  Only TRAIN candidates "
        "are inspected in this step."
    )

    print(
        "  Validation/test are not "
        "used to design negative rules."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = [
        REPORT_PATH,
        SIMILARITY_DISTRIBUTION_PATH,
        RANK_DISTRIBUTION_PATH,
        CATEGORY_DISTRIBUTION_PATH,
        MANUAL_SAMPLE_PATH,
    ]

    existing = [
        path
        for path in outputs
        if path.exists()
    ]

    if (
        existing
        and
        not args.overwrite
    ):

        existing_text = "\n".join(
            str(path)
            for path in existing
        )

        raise RuntimeError(
            "Step 06 outputs already exist:\n"
            f"{existing_text}\n\n"
            "Use --overwrite to regenerate."
        )

    if args.overwrite:

        for path in existing:
            path.unlink()

    validate_inputs()

    analysis = analyze_candidates(
        sample_per_stratum=(
            args.sample_per_stratum
        )
    )

    review_rows = (
        build_manual_review(
            analysis[
                "selected_samples"
            ]
        )
    )

    report = {
        "module":
            "module2_headline_integrity",

        "task":
            "headline_consistency",

        "stage":
            "06_analyze_candidate_quality",

        "analysis_split":
            "train",

        "test_used":
            False,

        "validation_used":
            False,

        "candidate_rows":
            analysis[
                "total_rows"
            ],

        "anchors_with_candidates":
            analysis[
                "anchors_with_candidates"
            ],

        "similarity_percentiles":
            analysis[
                "percentiles"
            ],

        "similarity_bin_counts":
            analysis[
                "similarity_bin_counts"
            ],

        "rank_counts":
            analysis[
                "rank_counts"
            ],

        "manual_review_rows":
            review_rows,

        "manual_review_instruction": {
            "VALID_NEGATIVE":
                (
                    "candidate headline is clearly "
                    "not supported by anchor article"
                ),

            "FALSE_NEGATIVE":
                (
                    "candidate headline is actually "
                    "compatible with anchor article"
                ),

            "AMBIGUOUS":
                (
                    "cannot confidently determine "
                    "from the article/context"
                ),
        },

        "outputs": {
            "similarity_distribution":
                str(
                    SIMILARITY_DISTRIBUTION_PATH
                    .relative_to(
                        MODULE_ROOT
                    )
                ),

            "rank_distribution":
                str(
                    RANK_DISTRIBUTION_PATH
                    .relative_to(
                        MODULE_ROOT
                    )
                ),

            "category_distribution":
                str(
                    CATEGORY_DISTRIBUTION_PATH
                    .relative_to(
                        MODULE_ROOT
                    )
                ),

            "manual_review":
                str(
                    MANUAL_SAMPLE_PATH
                    .relative_to(
                        MODULE_ROOT
                    )
                ),
        },
    }

    # Do not store the full selected sample
    # inside the compact JSON report.
    save_json(
        report,
        REPORT_PATH,
    )

    print()
    print("=" * 80)
    print(
        "STEP 06 SUMMARY"
    )
    print("=" * 80)

    print()
    print(
        f"Train candidates analyzed: "
        f"{analysis['total_rows']:,}"
    )

    print(
        f"Manual review rows: "
        f"{review_rows:,}"
    )

    print()
    print(
        "Outputs:"
    )

    print(
        f"  {SIMILARITY_DISTRIBUTION_PATH}"
    )

    print(
        f"  {RANK_DISTRIBUTION_PATH}"
    )

    print(
        f"  {CATEGORY_DISTRIBUTION_PATH}"
    )

    print(
        f"  {MANUAL_SAMPLE_PATH}"
    )

    print(
        f"  {REPORT_PATH}"
    )

    print()
    print(
        "STEP 06 FINISHED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()