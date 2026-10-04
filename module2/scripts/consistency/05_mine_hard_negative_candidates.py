from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import math

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)


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

OUTPUT_DIR = (
    MODULE_ROOT
    / "data"
    / "interim"
    / "consistency"
    / "candidate_pairs_v1"
)

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

GROUP_STATS_PATH = (
    OUTPUT_DIR
    / "group_stats.csv"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "mining_report.json"
)


# ============================================================
# CONFIG
# ============================================================

VALID_SPLITS = [
    "train",
    "validation",
    "test",
]

DEFAULT_TOP_K = 10

DEFAULT_BATCH_SIZE = 256

DEFAULT_MAX_FEATURES = 120_000

DEFAULT_MIN_SIMILARITY = 0.10

# We first retrieve more than TOP_K because some candidates
# will later be removed by same-title / self filters.
CANDIDATE_BUFFER_MULTIPLIER = 20

MIN_CANDIDATE_BUFFER = 100

TITLE_REPEAT = 2

RETRIEVAL_METHOD = (
    "same_split_same_category_"
    "tfidf_word_1_2"
)


# ============================================================
# OUTPUT SCHEMA
# ============================================================

CANDIDATE_SCHEMA = pa.schema(
    [
        (
            "candidate_pair_id",
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
            "candidate_headline_article_uid",
            pa.string(),
        ),

        (
            "category",
            pa.string(),
        ),

        (
            "candidate_rank",
            pa.int16(),
        ),

        (
            "lexical_similarity",
            pa.float32(),
        ),

        (
            "anchor_title_hash",
            pa.string(),
        ),

        (
            "candidate_title_hash",
            pa.string(),
        ),

        (
            "anchor_content_hash",
            pa.string(),
        ),

        (
            "candidate_content_hash",
            pa.string(),
        ),

        (
            "retrieval_method",
            pa.string(),
        ),
    ]
)


# ============================================================
# HELPERS
# ============================================================

def stable_candidate_id(
    anchor_uid,
    candidate_uid,
):
    """
    Deterministic ID for one anchor-candidate pair.
    """

    raw = (
        f"{anchor_uid}"
        f"\u241f"
        f"{candidate_uid}"
    )

    digest = hashlib.blake2b(
        raw.encode("utf-8"),
        digest_size=12,
    ).hexdigest()

    return (
        f"candidate_{digest}"
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


def normalize_for_retrieval(
    value,
):

    if value is None:
        return ""

    if (
        isinstance(
            value,
            float,
        )
        and
        math.isnan(value)
    ):
        return ""

    return str(
        value
    ).strip()


def build_retrieval_text(
    title,
    description,
):
    """
    Give headline slightly higher lexical weight by
    repeating it TITLE_REPEAT times.
    """

    title = normalize_for_retrieval(
        title
    )

    description = (
        normalize_for_retrieval(
            description
        )
    )

    pieces = []

    for _ in range(
        TITLE_REPEAT
    ):

        if title:
            pieces.append(
                title
            )

    if description:

        pieces.append(
            description
        )

    return " ".join(
        pieces
    ).strip()


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

    generated_files = (
        list(
            OUTPUT_FILES.values()
        )
        +
        [
            GROUP_STATS_PATH,
            REPORT_PATH,
        ]
    )

    existing = [
        path
        for path in generated_files
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
            "Step 05 output files "
            "already exist:\n"
            f"{existing_text}\n\n"
            "Use --overwrite if you "
            "want to regenerate them."
        )

    if overwrite:

        for path in existing:
            path.unlink()


# ============================================================
# INPUT VALIDATION
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

    required_columns = [
        "article_uid",
        "dataset",
        "split",
        "title",
        "description",
        "category",
        "title_hash",
        "content_hash",
    ]

    table = pq.read_table(
        ARTICLE_INDEX_PATH,
        columns=required_columns,
    )

    df = table.to_pandas()

    print(
        f"Rows loaded: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Structural validation
    # --------------------------------------------------------

    if df[
        "article_uid"
    ].isna().any():

        raise ValueError(
            "Missing article_uid."
        )

    if df[
        "article_uid"
    ].duplicated().any():

        duplicate_count = int(
            df[
                "article_uid"
            ]
            .duplicated()
            .sum()
        )

        raise ValueError(
            "Duplicate article_uid: "
            f"{duplicate_count:,}"
        )

    unknown_splits = (
        set(
            df[
                "split"
            ].unique()
        )
        -
        set(
            VALID_SPLITS
        )
    )

    if unknown_splits:

        raise ValueError(
            "Unknown split(s): "
            f"{sorted(unknown_splits)}"
        )

    if df[
        "title_hash"
    ].isna().any():

        raise ValueError(
            "Missing title_hash."
        )

    if df[
        "content_hash"
    ].isna().any():

        raise ValueError(
            "Missing content_hash."
        )

    print(
        "Article index validation: PASSED"
    )

    return df


# ============================================================
# VECTORIZE ONE GROUP
# ============================================================

def build_tfidf_matrix(
    texts,
    max_features,
):
    """
    Word-level TF-IDF.

    Word 1-2 grams provide a relatively interpretable
    first-stage retrieval signal.

    This is NOT the final semantic score.
    """

    n_documents = len(
        texts
    )

    if n_documents < 2:

        return (
            None,
            None,
        )

    # --------------------------------------------------------
    # For normal groups use min_df=2 to remove one-off noise.
    # Tiny groups fall back to min_df=1.
    # --------------------------------------------------------

    min_df = (
        2
        if n_documents >= 10
        else 1
    )

    max_df = (
        0.95
        if n_documents >= 20
        else 1.0
    )

    vectorizer = (
        TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(
                1,
                2,
            ),
            min_df=min_df,
            max_df=max_df,
            max_features=(
                max_features
            ),
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
            token_pattern=(
                r"(?u)\b\w+\b"
            ),
        )
    )

    try:

        matrix = (
            vectorizer
            .fit_transform(
                texts
            )
            .tocsr()
        )

    except ValueError:

        # ----------------------------------------------------
        # Safe fallback for unusual tiny groups.
        # ----------------------------------------------------

        vectorizer = (
            TfidfVectorizer(
                lowercase=True,
                analyzer="word",
                ngram_range=(
                    1,
                    2,
                ),
                min_df=1,
                max_df=1.0,
                max_features=(
                    max_features
                ),
                sublinear_tf=True,
                norm="l2",
                dtype=np.float32,
                token_pattern=(
                    r"(?u)\b\w+\b"
                ),
            )
        )

        matrix = (
            vectorizer
            .fit_transform(
                texts
            )
            .tocsr()
        )

    return (
        vectorizer,
        matrix,
    )


# ============================================================
# TOP CANDIDATE EXTRACTION
# ============================================================

def select_top_candidates(
    similarity_indices,
    similarity_values,
    anchor_local_index,
    title_hashes,
    content_hashes,
    top_k,
    min_similarity,
):
    """
    Select top candidate local indices after structural filters.
    """

    if len(
        similarity_indices
    ) == 0:

        return []

    buffer_size = max(
        top_k
        * CANDIDATE_BUFFER_MULTIPLIER,
        MIN_CANDIDATE_BUFFER,
    )

    buffer_size = min(
        buffer_size,
        len(
            similarity_values
        ),
    )

    # --------------------------------------------------------
    # Keep only the strongest lexical candidates before
    # full filtering.
    # --------------------------------------------------------

    if (
        len(
            similarity_values
        )
        >
        buffer_size
    ):

        strongest_positions = (
            np.argpartition(
                similarity_values,
                -buffer_size,
            )[
                -buffer_size:
            ]
        )

    else:

        strongest_positions = (
            np.arange(
                len(
                    similarity_values
                )
            )
        )

    strongest_positions = (
        strongest_positions[
            np.argsort(
                similarity_values[
                    strongest_positions
                ]
            )[
                ::-1
            ]
        ]
    )

    anchor_title_hash = (
        title_hashes[
            anchor_local_index
        ]
    )

    anchor_content_hash = (
        content_hashes[
            anchor_local_index
        ]
    )

    selected = []

    seen_candidates = set()

    for position in (
        strongest_positions
    ):

        candidate_index = int(
            similarity_indices[
                position
            ]
        )

        similarity = float(
            similarity_values[
                position
            ]
        )

        # ----------------------------------------------------
        # Ignore weak/no-overlap candidates.
        # ----------------------------------------------------

        if (
            similarity
            <
            min_similarity
        ):
            continue

        # ----------------------------------------------------
        # Never pair an article with itself.
        # ----------------------------------------------------

        if (
            candidate_index
            ==
            anchor_local_index
        ):
            continue

        # ----------------------------------------------------
        # Same normalized headline is forbidden.
        # ----------------------------------------------------

        if (
            title_hashes[
                candidate_index
            ]
            ==
            anchor_title_hash
        ):
            continue

        # ----------------------------------------------------
        # Same content is also forbidden, even though Step 03
        # already established global content uniqueness.
        # ----------------------------------------------------

        if (
            content_hashes[
                candidate_index
            ]
            ==
            anchor_content_hash
        ):
            continue

        if (
            candidate_index
            in
            seen_candidates
        ):
            continue

        seen_candidates.add(
            candidate_index
        )

        selected.append(
            (
                candidate_index,
                similarity,
            )
        )

        if (
            len(
                selected
            )
            >=
            top_k
        ):
            break

    return selected


# ============================================================
# MINE ONE CATEGORY
# ============================================================

def mine_group(
    group_df,
    split,
    category,
    writer,
    top_k,
    batch_size,
    max_features,
    min_similarity,
):

    group_df = (
        group_df
        .reset_index(
            drop=True
        )
    )

    n_articles = len(
        group_df
    )

    stats = {
        "split":
            split,

        "category":
            category,

        "articles":
            int(
                n_articles
            ),

        "vocabulary_size":
            0,

        "candidates":
            0,

        "anchors_with_candidates":
            0,

        "anchors_without_candidates":
            0,

        "mean_candidates_per_anchor":
            0.0,

        "mean_top1_similarity":
            None,

        "min_similarity":
            None,

        "max_similarity":
            None,
    }

    if n_articles < 2:

        stats[
            "anchors_without_candidates"
        ] = int(
            n_articles
        )

        return stats

    # ========================================================
    # RETRIEVAL TEXT
    # ========================================================

    texts = [
        build_retrieval_text(
            title,
            description,
        )

        for title, description
        in zip(
            group_df[
                "title"
            ],
            group_df[
                "description"
            ],
        )
    ]

    # ========================================================
    # TF-IDF
    # ========================================================

    (
        vectorizer,
        matrix,
    ) = build_tfidf_matrix(
        texts=texts,
        max_features=(
            max_features
        ),
    )

    if matrix is None:

        stats[
            "anchors_without_candidates"
        ] = int(
            n_articles
        )

        return stats

    stats[
        "vocabulary_size"
    ] = int(
        len(
            vectorizer.vocabulary_
        )
    )

    # Since TF-IDF vectors are L2-normalized:
    #
    # X @ X.T == cosine similarity.
    #
    # Sparse multiplication means we do not create a
    # full dense NxN matrix.
    # ========================================================

    article_uids = (
        group_df[
            "article_uid"
        ]
        .to_numpy()
    )

    title_hashes = (
        group_df[
            "title_hash"
        ]
        .to_numpy()
    )

    content_hashes = (
        group_df[
            "content_hash"
        ]
        .to_numpy()
    )

    candidate_count = 0

    anchors_with_candidates = 0

    anchors_without_candidates = 0

    all_candidate_similarities = []

    top1_similarities = []

    # ========================================================
    # QUERY IN BATCHES
    # ========================================================

    for batch_start in range(
        0,
        n_articles,
        batch_size,
    ):

        batch_end = min(
            batch_start
            +
            batch_size,
            n_articles,
        )

        query_matrix = (
            matrix[
                batch_start:
                batch_end
            ]
        )

        similarity_matrix = (
            query_matrix
            @
            matrix.T
        ).tocsr()

        records = []

        for row_offset in range(
            batch_end
            -
            batch_start
        ):

            anchor_index = (
                batch_start
                +
                row_offset
            )

            row_start = (
                similarity_matrix
                .indptr[
                    row_offset
                ]
            )

            row_end = (
                similarity_matrix
                .indptr[
                    row_offset
                    +
                    1
                ]
            )

            similarity_indices = (
                similarity_matrix
                .indices[
                    row_start:
                    row_end
                ]
            )

            similarity_values = (
                similarity_matrix
                .data[
                    row_start:
                    row_end
                ]
            )

            selected = (
                select_top_candidates(
                    similarity_indices=(
                        similarity_indices
                    ),
                    similarity_values=(
                        similarity_values
                    ),
                    anchor_local_index=(
                        anchor_index
                    ),
                    title_hashes=(
                        title_hashes
                    ),
                    content_hashes=(
                        content_hashes
                    ),
                    top_k=top_k,
                    min_similarity=(
                        min_similarity
                    ),
                )
            )

            if not selected:

                anchors_without_candidates += 1

                continue

            anchors_with_candidates += 1

            top1_similarities.append(
                selected[0][1]
            )

            anchor_uid = (
                article_uids[
                    anchor_index
                ]
            )

            for rank, (
                candidate_index,
                similarity,
            ) in enumerate(
                selected,
                start=1,
            ):

                candidate_uid = (
                    article_uids[
                        candidate_index
                    ]
                )

                candidate_pair_id = (
                    stable_candidate_id(
                        anchor_uid,
                        candidate_uid,
                    )
                )

                records.append(
                    {
                        "candidate_pair_id":
                            candidate_pair_id,

                        "split":
                            split,

                        "anchor_article_uid":
                            anchor_uid,

                        "candidate_headline_article_uid":
                            candidate_uid,

                        "category":
                            category,

                        "candidate_rank":
                            rank,

                        "lexical_similarity":
                            float(
                                similarity
                            ),

                        "anchor_title_hash":
                            title_hashes[
                                anchor_index
                            ],

                        "candidate_title_hash":
                            title_hashes[
                                candidate_index
                            ],

                        "anchor_content_hash":
                            content_hashes[
                                anchor_index
                            ],

                        "candidate_content_hash":
                            content_hashes[
                                candidate_index
                            ],

                        "retrieval_method":
                            RETRIEVAL_METHOD,
                    }
                )

                candidate_count += 1

                all_candidate_similarities.append(
                    similarity
                )

        if records:

            table = (
                pa.Table.from_pylist(
                    records,
                    schema=(
                        CANDIDATE_SCHEMA
                    ),
                )
            )

            writer.write_table(
                table
            )

    # ========================================================
    # GROUP STATS
    # ========================================================

    stats[
        "candidates"
    ] = int(
        candidate_count
    )

    stats[
        "anchors_with_candidates"
    ] = int(
        anchors_with_candidates
    )

    stats[
        "anchors_without_candidates"
    ] = int(
        anchors_without_candidates
    )

    stats[
        "mean_candidates_per_anchor"
    ] = (
        float(
            candidate_count
            /
            n_articles
        )
        if n_articles
        else 0.0
    )

    if top1_similarities:

        stats[
            "mean_top1_similarity"
        ] = float(
            np.mean(
                top1_similarities
            )
        )

    if all_candidate_similarities:

        stats[
            "min_similarity"
        ] = float(
            np.min(
                all_candidate_similarities
            )
        )

        stats[
            "max_similarity"
        ] = float(
            np.max(
                all_candidate_similarities
            )
        )

    return stats


# ============================================================
# MINE ALL SPLITS
# ============================================================

def mine_candidates(
    df,
    top_k,
    batch_size,
    max_features,
    min_similarity,
):

    print()
    print("=" * 80)
    print(
        "MINE HARD-NEGATIVE CANDIDATES"
    )
    print("=" * 80)

    all_group_stats = []

    split_candidate_counts = Counter()

    split_anchor_counts = Counter()

    # ========================================================
    # SPLITS ARE COMPLETELY INDEPENDENT
    # ========================================================

    for split in VALID_SPLITS:

        split_df = (
            df[
                df[
                    "split"
                ]
                ==
                split
            ]
            .copy()
        )

        split_anchor_counts[
            split
        ] = len(
            split_df
        )

        print()
        print(
            "#" * 80
        )

        print(
            f"SPLIT: "
            f"{split.upper()}"
        )

        print(
            f"Articles: "
            f"{len(split_df):,}"
        )

        print(
            "#" * 80
        )

        output_path = (
            OUTPUT_FILES[
                split
            ]
        )

        writer = (
            pq.ParquetWriter(
                output_path,
                CANDIDATE_SCHEMA,
                compression="zstd",
                use_dictionary=True,
            )
        )

        try:

            categories = sorted(
                split_df[
                    "category"
                ]
                .unique()
                .tolist()
            )

            for category_index, category in enumerate(
                categories,
                start=1,
            ):

                group_df = (
                    split_df[
                        split_df[
                            "category"
                        ]
                        ==
                        category
                    ]
                    .copy()
                )

                print()
                print(
                    f"[{category_index}/"
                    f"{len(categories)}] "
                    f"{category}"
                )

                print(
                    f"  Articles: "
                    f"{len(group_df):,}"
                )

                stats = mine_group(
                    group_df=group_df,
                    split=split,
                    category=category,
                    writer=writer,
                    top_k=top_k,
                    batch_size=batch_size,
                    max_features=(
                        max_features
                    ),
                    min_similarity=(
                        min_similarity
                    ),
                )

                all_group_stats.append(
                    stats
                )

                split_candidate_counts[
                    split
                ] += (
                    stats[
                        "candidates"
                    ]
                )

                print(
                    f"  Vocabulary: "
                    f"{stats['vocabulary_size']:,}"
                )

                print(
                    f"  Candidates: "
                    f"{stats['candidates']:,}"
                )

                print(
                    "  Anchors with candidates: "
                    f"{stats['anchors_with_candidates']:,}"
                )

                print(
                    "  Anchors without candidates: "
                    f"{stats['anchors_without_candidates']:,}"
                )

                if (
                    stats[
                        "mean_top1_similarity"
                    ]
                    is not None
                ):

                    print(
                        "  Mean top-1 similarity: "
                        f"{stats['mean_top1_similarity']:.4f}"
                    )

        finally:

            writer.close()

        print()
        print(
            f"{split} total candidates: "
            f"{split_candidate_counts[split]:,}"
        )

    return (
        all_group_stats,
        split_candidate_counts,
        split_anchor_counts,
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

def validate_outputs():

    print()
    print("=" * 80)
    print(
        "FINAL STRUCTURAL VALIDATION"
    )
    print("=" * 80)

    passed = True

    row_counts = {}

    for split in VALID_SPLITS:

        path = (
            OUTPUT_FILES[
                split
            ]
        )

        if not path.exists():

            print(
                f"{split}: "
                "output missing"
            )

            passed = False

            continue

        parquet_file = (
            pq.ParquetFile(
                path
            )
        )

        total_rows = (
            parquet_file
            .metadata
            .num_rows
        )

        row_counts[
            split
        ] = int(
            total_rows
        )

        invalid_split = 0
        self_pairs = 0
        same_title = 0
        same_content = 0
        invalid_similarity = 0
        invalid_rank = 0

        seen_pair_ids = set()

        duplicate_pair_ids = 0

        columns = [
            "candidate_pair_id",
            "split",
            "anchor_article_uid",
            "candidate_headline_article_uid",
            "candidate_rank",
            "lexical_similarity",
            "anchor_title_hash",
            "candidate_title_hash",
            "anchor_content_hash",
            "candidate_content_hash",
        ]

        for batch in (
            parquet_file
            .iter_batches(
                batch_size=8192,
                columns=columns,
            )
        ):

            frame = (
                batch
                .to_pandas()
            )

            invalid_split += int(
                (
                    frame[
                        "split"
                    ]
                    !=
                    split
                ).sum()
            )

            self_pairs += int(
                (
                    frame[
                        "anchor_article_uid"
                    ]
                    ==
                    frame[
                        "candidate_headline_article_uid"
                    ]
                ).sum()
            )

            same_title += int(
                (
                    frame[
                        "anchor_title_hash"
                    ]
                    ==
                    frame[
                        "candidate_title_hash"
                    ]
                ).sum()
            )

            same_content += int(
                (
                    frame[
                        "anchor_content_hash"
                    ]
                    ==
                    frame[
                        "candidate_content_hash"
                    ]
                ).sum()
            )

            invalid_similarity += int(
                (
                    (
                        frame[
                            "lexical_similarity"
                        ]
                        <= 0
                    )
                    |
                    (
                        frame[
                            "lexical_similarity"
                        ]
                        > 1.0001
                    )
                ).sum()
            )

            invalid_rank += int(
                (
                    frame[
                        "candidate_rank"
                    ]
                    < 1
                ).sum()
            )

            for pair_id in frame[
                "candidate_pair_id"
            ]:

                if (
                    pair_id
                    in
                    seen_pair_ids
                ):

                    duplicate_pair_ids += 1

                else:

                    seen_pair_ids.add(
                        pair_id
                    )

        print()
        print(
            f"{split}: "
            f"{total_rows:,} candidates"
        )

        print(
            f"  invalid split : "
            f"{invalid_split:,}"
        )

        print(
            f"  self pairs    : "
            f"{self_pairs:,}"
        )

        print(
            f"  same title    : "
            f"{same_title:,}"
        )

        print(
            f"  same content  : "
            f"{same_content:,}"
        )

        print(
            f"  invalid sim   : "
            f"{invalid_similarity:,}"
        )

        print(
            f"  invalid rank  : "
            f"{invalid_rank:,}"
        )

        print(
            f"  duplicate IDs : "
            f"{duplicate_pair_ids:,}"
        )

        if any(
            [
                invalid_split,
                self_pairs,
                same_title,
                same_content,
                invalid_similarity,
                invalid_rank,
                duplicate_pair_ids,
            ]
        ):

            passed = False

    print()

    if passed:

        print(
            "STRUCTURAL VALIDATION: PASSED"
        )

    else:

        print(
            "STRUCTURAL VALIDATION: FAILED"
        )

    return (
        passed,
        row_counts,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Mine same-category lexical "
            "headline candidates for "
            "headline consistency."
        )
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=(
            DEFAULT_TOP_K
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=(
            DEFAULT_BATCH_SIZE
        ),
    )

    parser.add_argument(
        "--max-features",
        type=int,
        default=(
            DEFAULT_MAX_FEATURES
        ),
    )

    parser.add_argument(
        "--min-similarity",
        type=float,
        default=(
            DEFAULT_MIN_SIMILARITY
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    if args.top_k < 1:

        raise ValueError(
            "--top-k must be >= 1"
        )

    if args.batch_size < 1:

        raise ValueError(
            "--batch-size must be >= 1"
        )

    if not (
        0.0
        <=
        args.min_similarity
        <=
        1.0
    ):

        raise ValueError(
            "--min-similarity "
            "must be in [0, 1]."
        )

    print("=" * 80)
    print(
        "MODULE 2 - HEADLINE INTEGRITY"
    )
    print(
        "TASK 2B - HEADLINE CONSISTENCY"
    )
    print(
        "STEP 05 - MINE HARD-NEGATIVE CANDIDATES"
    )
    print("=" * 80)

    print()
    print(
        f"Module root:\n"
        f"{MODULE_ROOT}"
    )

    print()
    print(
        "Mining configuration:"
    )

    print(
        f"  top_k          : "
        f"{args.top_k}"
    )

    print(
        f"  batch_size     : "
        f"{args.batch_size}"
    )

    print(
        f"  max_features   : "
        f"{args.max_features:,}"
    )

    print(
        f"  min_similarity : "
        f"{args.min_similarity}"
    )

    print(
        f"  title_repeat   : "
        f"{TITLE_REPEAT}"
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    prepare_output_directory(
        overwrite=(
            args.overwrite
        )
    )

    # ========================================================
    # INPUT
    # ========================================================

    df = load_article_index()

    # ========================================================
    # MINE
    # ========================================================

    (
        group_stats,
        split_candidate_counts,
        split_anchor_counts,
    ) = mine_candidates(
        df=df,
        top_k=args.top_k,
        batch_size=(
            args.batch_size
        ),
        max_features=(
            args.max_features
        ),
        min_similarity=(
            args.min_similarity
        ),
    )

    # ========================================================
    # SAVE GROUP STATS
    # ========================================================

    group_stats_df = (
        pd.DataFrame(
            group_stats
        )
    )

    group_stats_df.to_csv(
        GROUP_STATS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # VALIDATE
    # ========================================================

    (
        validation_passed,
        validated_row_counts,
    ) = validate_outputs()

    # ========================================================
    # REPORT
    # ========================================================

    report = {
        "module":
            "module2_headline_integrity",

        "task":
            "headline_consistency",

        "stage":
            (
                "05_mine_hard_negative_"
                "candidates"
            ),

        "important_note":
            (
                "These records are candidate "
                "mismatches only. They are NOT "
                "yet gold negative labels."
            ),

        "retrieval": {
            "method":
                RETRIEVAL_METHOD,

            "scope":
                (
                    "same split + "
                    "same category"
                ),

            "text":
                (
                    "title repeated twice "
                    "+ description"
                ),

            "vectorizer":
                "TF-IDF",

            "analyzer":
                "word",

            "ngram_range":
                [
                    1,
                    2,
                ],

            "top_k":
                int(
                    args.top_k
                ),

            "max_features":
                int(
                    args.max_features
                ),

            "min_similarity":
                float(
                    args.min_similarity
                ),

            "title_repeat":
                int(
                    TITLE_REPEAT
                ),
        },

        "filters": {
            "same_article_forbidden":
                True,

            "same_title_hash_forbidden":
                True,

            "same_content_hash_forbidden":
                True,

            "cross_split_candidates":
                False,

            "cross_category_candidates":
                False,
        },

        "anchor_counts": {
            split:
                int(
                    split_anchor_counts[
                        split
                    ]
                )

            for split
            in VALID_SPLITS
        },

        "candidate_counts": {
            split:
                int(
                    split_candidate_counts[
                        split
                    ]
                )

            for split
            in VALID_SPLITS
        },

        "validated_row_counts":
            validated_row_counts,

        "total_candidates":
            int(
                sum(
                    split_candidate_counts
                    .values()
                )
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

        "group_stats":
            str(
                GROUP_STATS_PATH
                .relative_to(
                    MODULE_ROOT
                )
            ),

        "validation_passed":
            bool(
                validation_passed
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
        "STEP 05 SUMMARY"
    )
    print("=" * 80)

    print()

    for split in (
        VALID_SPLITS
    ):

        print(
            f"{split:12s}: "
            f"{split_candidate_counts[split]:,} "
            f"candidates for "
            f"{split_anchor_counts[split]:,} "
            f"articles"
        )

    print()
    print(
        "Total candidates: "
        f"{sum(split_candidate_counts.values()):,}"
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
        f"  {GROUP_STATS_PATH.name:25s} "
        f"{human_size(GROUP_STATS_PATH.stat().st_size)}"
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

    if validation_passed:

        print(
            "STEP 05 FINISHED SUCCESSFULLY"
        )

    else:

        raise RuntimeError(
            "Candidate mining output "
            "failed structural validation."
        )


if __name__ == "__main__":
    main()