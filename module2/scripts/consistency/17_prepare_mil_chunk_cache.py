from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from transformers import AutoTokenizer


# ============================================================
# PATHS
# ============================================================

MODULE_ROOT = Path(__file__).resolve().parents[2]

PAIR_DIR = (
    MODULE_ROOT
    / "data"
    / "processed"
    / "consistency"
    / "headline_consistency_v1"
)

ARTICLE_DIR = (
    MODULE_ROOT
    / "data"
    / "interim"
    / "consistency"
    / "articles_v1"
)

PAIR_FILES = {
    "train":
        PAIR_DIR
        / "train.parquet",

    "validation":
        PAIR_DIR
        / "validation.parquet",
}

ARTICLE_FILES = {
    "train":
        ARTICLE_DIR
        / "train.parquet",

    "validation":
        ARTICLE_DIR
        / "validation.parquet",
}

OUTPUT_DIR = (
    MODULE_ROOT
    / "data"
    / "interim"
    / "consistency"
    / "mil_chunk_cache_mbert_v1"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "cache_report.json"
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = (
    "bert-base-multilingual-cased"
)

CHUNK_TOKENS = 120

CHUNK_OVERLAP = 24

MAX_CHUNKS = 6


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


# ============================================================
# CHUNKING
# ============================================================

def build_chunks(
    content,
    tokenizer,
):
    """
    Same full-document sampling policy as Step 16.

    1. Tokenize complete article.
    2. Sliding windows of 120 tokens, overlap 24.
    3. If there are >6 candidate windows,
       uniformly select 6 positions across the article.
    """

    token_ids = tokenizer.encode(
        str(
            content
        ),
        add_special_tokens=False,
        verbose=False,
    )

    total_tokens = len(
        token_ids
    )

    if total_tokens == 0:

        return (
            [
                []
            ],
            0,
            [
                0
            ],
        )

    if (
        total_tokens
        <=
        CHUNK_TOKENS
    ):

        return (
            [
                token_ids
            ],
            total_tokens,
            [
                0
            ],
        )

    step = (
        CHUNK_TOKENS
        -
        CHUNK_OVERLAP
    )

    if step <= 0:

        raise ValueError(
            "CHUNK_OVERLAP must be "
            "smaller than CHUNK_TOKENS."
        )

    starts = list(
        range(
            0,
            total_tokens,
            step,
        )
    )

    # --------------------------------------------------------
    # Uniformly cover the full article.
    # --------------------------------------------------------

    if (
        len(
            starts
        )
        >
        MAX_CHUNKS
    ):

        indices = np.linspace(
            0,
            len(
                starts
            )
            -
            1,
            MAX_CHUNKS,
        )

        indices = (
            np.round(
                indices
            )
            .astype(
                int
            )
            .tolist()
        )

        # Remove duplicates caused by rounding.
        indices = list(
            dict.fromkeys(
                indices
            )
        )

        starts = [
            starts[
                index
            ]
            for index
            in indices
        ]

    chunks = []

    used_starts = []

    for start in starts:

        chunk = token_ids[
            start:
            start
            +
            CHUNK_TOKENS
        ]

        if chunk:

            chunks.append(
                chunk
            )

            used_starts.append(
                start
            )

    if not chunks:

        chunks = [
            token_ids[
                :CHUNK_TOKENS
            ]
        ]

        used_starts = [
            0
        ]

    return (
        chunks,
        total_tokens,
        used_starts,
    )


# ============================================================
# LOAD REQUIRED ANCHORS
# ============================================================

def load_required_anchors(
    split,
):

    pair_path = (
        PAIR_FILES[
            split
        ]
    )

    frame = (
        pq.read_table(
            pair_path,
            columns=[
                "anchor_article_uid",
            ],
        )
        .to_pandas()
    )

    anchors = (
        frame[
            "anchor_article_uid"
        ]
        .drop_duplicates()
        .astype(str)
        .sort_values()
        .tolist()
    )

    return anchors


# ============================================================
# PREPARE ONE SPLIT
# ============================================================

def prepare_split(
    split,
    tokenizer,
):

    print()
    print("=" * 80)
    print(
        f"PREPARE CHUNK CACHE - "
        f"{split.upper()}"
    )
    print("=" * 80)

    required_anchors = (
        load_required_anchors(
            split
        )
    )

    print(
        f"Required anchors: "
        f"{len(required_anchors):,}"
    )

    # ========================================================
    # LOAD FULL CONTENT
    # ========================================================

    print(
        "Loading article content..."
    )

    article_frame = (
        pq.read_table(
            ARTICLE_FILES[
                split
            ],
            columns=[
                "article_uid",
                "content",
            ],
        )
        .to_pandas()
    )

    article_frame[
        "article_uid"
    ] = (
        article_frame[
            "article_uid"
        ]
        .astype(str)
    )

    if (
        article_frame[
            "article_uid"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            f"{split}: duplicate "
            "article_uid."
        )

    article_frame = (
        article_frame[
            article_frame[
                "article_uid"
            ]
            .isin(
                required_anchors
            )
        ]
        .copy()
    )

    if (
        len(
            article_frame
        )
        !=
        len(
            required_anchors
        )
    ):

        found = set(
            article_frame[
                "article_uid"
            ]
        )

        missing = (
            set(
                required_anchors
            )
            -
            found
        )

        raise RuntimeError(
            f"{split}: missing "
            f"{len(missing):,} "
            "anchor articles."
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

    # Free dataframe memory after map creation.
    del article_frame

    # ========================================================
    # OUTPUT FILES
    # ========================================================

    chunk_ids_path = (
        OUTPUT_DIR
        / f"{split}_chunk_ids.npy"
    )

    chunk_lengths_path = (
        OUTPUT_DIR
        / f"{split}_chunk_lengths.npy"
    )

    chunk_starts_path = (
        OUTPUT_DIR
        / f"{split}_chunk_starts.npy"
    )

    chunk_counts_path = (
        OUTPUT_DIR
        / f"{split}_chunk_counts.npy"
    )

    article_index_path = (
        OUTPUT_DIR
        / f"{split}_article_index.parquet"
    )

    n_articles = len(
        required_anchors
    )

    # ========================================================
    # MEMORY-MAPPED ARRAYS
    # ========================================================

    print()
    print(
        "Creating memory-mapped arrays..."
    )

    chunk_ids = (
        np.lib.format.open_memmap(
            chunk_ids_path,
            mode="w+",
            dtype=np.int32,
            shape=(
                n_articles,
                MAX_CHUNKS,
                CHUNK_TOKENS,
            ),
        )
    )

    chunk_lengths = (
        np.lib.format.open_memmap(
            chunk_lengths_path,
            mode="w+",
            dtype=np.uint16,
            shape=(
                n_articles,
                MAX_CHUNKS,
            ),
        )
    )

    chunk_starts = (
        np.lib.format.open_memmap(
            chunk_starts_path,
            mode="w+",
            dtype=np.int32,
            shape=(
                n_articles,
                MAX_CHUNKS,
            ),
        )
    )

    chunk_counts = (
        np.lib.format.open_memmap(
            chunk_counts_path,
            mode="w+",
            dtype=np.uint8,
            shape=(
                n_articles,
            ),
        )
    )

    # Initialize.
    chunk_ids[:] = 0
    chunk_lengths[:] = 0
    chunk_starts[:] = -1
    chunk_counts[:] = 0

    # ========================================================
    # TOKENIZE ARTICLES
    # ========================================================

    index_rows = []

    all_token_counts = []

    all_chunk_counts = []

    all_chunk_lengths = []

    print()
    print(
        "Tokenizing full articles..."
    )

    for row_idx, article_uid in enumerate(
        required_anchors
    ):

        content = (
            content_map[
                article_uid
            ]
        )

        (
            chunks,
            token_count,
            starts,
        ) = build_chunks(
            content=content,
            tokenizer=tokenizer,
        )

        if (
            len(
                chunks
            )
            >
            MAX_CHUNKS
        ):

            raise RuntimeError(
                f"{split}: article "
                f"{article_uid} generated "
                f"{len(chunks)} chunks."
            )

        chunk_counts[
            row_idx
        ] = len(
            chunks
        )

        all_token_counts.append(
            token_count
        )

        all_chunk_counts.append(
            len(
                chunks
            )
        )

        for chunk_index, chunk in enumerate(
            chunks
        ):

            length = len(
                chunk
            )

            if (
                length
                >
                CHUNK_TOKENS
            ):

                raise RuntimeError(
                    "Chunk exceeds "
                    "CHUNK_TOKENS."
                )

            if length > 0:

                chunk_ids[
                    row_idx,
                    chunk_index,
                    :length,
                ] = np.asarray(
                    chunk,
                    dtype=np.int32,
                )

            chunk_lengths[
                row_idx,
                chunk_index,
            ] = length

            chunk_starts[
                row_idx,
                chunk_index,
            ] = starts[
                chunk_index
            ]

            all_chunk_lengths.append(
                length
            )

        index_rows.append(
            {
                "row_idx":
                    int(
                        row_idx
                    ),

                "article_uid":
                    article_uid,

                "token_count":
                    int(
                        token_count
                    ),

                "chunk_count":
                    int(
                        len(
                            chunks
                        )
                    ),
            }
        )

        if (
            (
                row_idx
                +
                1
            )
            %
            5000
            ==
            0
        ):

            print(
                f"  processed "
                f"{row_idx + 1:,}/"
                f"{n_articles:,}"
            )

    # ========================================================
    # FLUSH ARRAYS
    # ========================================================

    chunk_ids.flush()
    chunk_lengths.flush()
    chunk_starts.flush()
    chunk_counts.flush()

    # ========================================================
    # SAVE INDEX
    # ========================================================

    index_frame = pd.DataFrame(
        index_rows
    )

    index_frame.to_parquet(
        article_index_path,
        index=False,
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    print()
    print(
        "Validating cache..."
    )

    if (
        index_frame[
            "article_uid"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            f"{split}: duplicate cached "
            "article_uid."
        )

    if (
        len(
            index_frame
        )
        !=
        n_articles
    ):

        raise RuntimeError(
            f"{split}: index row count "
            "mismatch."
        )

    if not (
        index_frame[
            "row_idx"
        ]
        .to_numpy()
        ==
        np.arange(
            n_articles
        )
    ).all():

        raise RuntimeError(
            f"{split}: row_idx mismatch."
        )

    if not (
        (
            chunk_counts
            >=
            1
        )
        &
        (
            chunk_counts
            <=
            MAX_CHUNKS
        )
    ).all():

        raise RuntimeError(
            f"{split}: invalid "
            "chunk counts."
        )

    # ========================================================
    # STATS
    # ========================================================

    token_values = np.asarray(
        all_token_counts,
        dtype=np.int64,
    )

    count_values = np.asarray(
        all_chunk_counts,
        dtype=np.int64,
    )

    length_values = np.asarray(
        all_chunk_lengths,
        dtype=np.int64,
    )

    stats = {
        "split":
            split,

        "articles":
            int(
                n_articles
            ),

        "chunk_slots":
            int(
                n_articles
                *
                MAX_CHUNKS
            ),

        "actual_chunks":
            int(
                count_values.sum()
            ),

        "chunks_per_article": {
            "min":
                int(
                    count_values.min()
                ),

            "mean":
                float(
                    count_values.mean()
                ),

            "median":
                float(
                    np.median(
                        count_values
                    )
                ),

            "p95":
                float(
                    np.percentile(
                        count_values,
                        95,
                    )
                ),

            "max":
                int(
                    count_values.max()
                ),
        },

        "article_tokens": {
            "min":
                int(
                    token_values.min()
                ),

            "median":
                float(
                    np.median(
                        token_values
                    )
                ),

            "mean":
                float(
                    token_values.mean()
                ),

            "p95":
                float(
                    np.percentile(
                        token_values,
                        95,
                    )
                ),

            "max":
                int(
                    token_values.max()
                ),
        },

        "chunk_lengths": {
            "min":
                int(
                    length_values.min()
                ),

            "median":
                float(
                    np.median(
                        length_values
                    )
                ),

            "mean":
                float(
                    length_values.mean()
                ),

            "max":
                int(
                    length_values.max()
                ),
        },

        "files": {
            "chunk_ids":
                str(
                    chunk_ids_path
                    .relative_to(
                        MODULE_ROOT
                    )
                ),

            "chunk_lengths":
                str(
                    chunk_lengths_path
                    .relative_to(
                        MODULE_ROOT
                    )
                ),

            "chunk_starts":
                str(
                    chunk_starts_path
                    .relative_to(
                        MODULE_ROOT
                    )
                ),

            "chunk_counts":
                str(
                    chunk_counts_path
                    .relative_to(
                        MODULE_ROOT
                    )
                ),

            "article_index":
                str(
                    article_index_path
                    .relative_to(
                        MODULE_ROOT
                    )
                ),
        },
    }

    print()
    print(
        f"{split} cache:"
    )

    print(
        f"  articles      : "
        f"{n_articles:,}"
    )

    print(
        f"  actual chunks : "
        f"{stats['actual_chunks']:,}"
    )

    print(
        f"  chunks mean   : "
        f"{stats['chunks_per_article']['mean']:.2f}"
    )

    print(
        f"  chunks median : "
        f"{stats['chunks_per_article']['median']:.1f}"
    )

    print(
        f"  token median  : "
        f"{stats['article_tokens']['median']:.1f}"
    )

    print(
        f"  token p95     : "
        f"{stats['article_tokens']['p95']:.1f}"
    )

    print(
        f"  token max     : "
        f"{stats['article_tokens']['max']:,}"
    )

    print(
        "  validation    : PASSED"
    )

    # Drop maps/memmaps explicitly.
    del content_map
    del chunk_ids
    del chunk_lengths
    del chunk_starts
    del chunk_counts

    return (
        stats,
        set(
            required_anchors
        ),
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
        "STEP 17 - PREPARE HYBRID MIL CHUNK CACHE"
    )
    print("=" * 80)

    print()
    print(
        "TRAIN + VALIDATION ONLY."
    )

    print(
        "TEST IS NOT LOADED."
    )

    print()
    print(
        f"Tokenizer     : "
        f"{MODEL_NAME}"
    )

    print(
        f"Chunk tokens  : "
        f"{CHUNK_TOKENS}"
    )

    print(
        f"Chunk overlap : "
        f"{CHUNK_OVERLAP}"
    )

    print(
        f"Max chunks    : "
        f"{MAX_CHUNKS}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # FILE CHECK
    # ========================================================

    required_files = [
        PAIR_FILES[
            "train"
        ],
        PAIR_FILES[
            "validation"
        ],
        ARTICLE_FILES[
            "train"
        ],
        ARTICLE_FILES[
            "validation"
        ],
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

    # ========================================================
    # TOKENIZER
    # ========================================================

    print()
    print(
        "Loading tokenizer..."
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_NAME
        )
    )

    if (
        tokenizer.pad_token_id
        is None
    ):

        raise RuntimeError(
            "Tokenizer has no PAD token."
        )

    # ========================================================
    # PREPARE TRAIN + VALIDATION
    # ========================================================

    train_stats, train_uids = (
        prepare_split(
            split="train",
            tokenizer=tokenizer,
        )
    )

    validation_stats, validation_uids = (
        prepare_split(
            split="validation",
            tokenizer=tokenizer,
        )
    )

    # ========================================================
    # CROSS-SPLIT VALIDATION
    # ========================================================

    overlap = (
        train_uids
        &
        validation_uids
    )

    print()
    print("=" * 80)
    print(
        "CROSS-SPLIT CACHE VALIDATION"
    )
    print("=" * 80)

    print(
        f"train ∩ validation: "
        f"{len(overlap):,} article UIDs"
    )

    if overlap:

        raise RuntimeError(
            "Chunk cache has cross-split "
            "article overlap."
        )

    print(
        "CROSS-SPLIT VALIDATION: PASSED"
    )

    # ========================================================
    # REPORT
    # ========================================================

    report = {
        "stage":
            "17_prepare_mil_chunk_cache",

        "model_name":
            MODEL_NAME,

        "test_used":
            False,

        "chunk_policy": {
            "chunk_tokens":
                CHUNK_TOKENS,

            "chunk_overlap":
                CHUNK_OVERLAP,

            "max_chunks":
                MAX_CHUNKS,

            "sampling":
                (
                    "uniform coverage across "
                    "full article"
                ),
        },

        "train":
            train_stats,

        "validation":
            validation_stats,

        "cross_split_article_uid_overlap":
            int(
                len(
                    overlap
                )
            ),
    }

    save_json(
        report,
        REPORT_PATH,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print(
        "STEP 17 SUMMARY"
    )
    print("=" * 80)

    print()
    print(
        "TRAIN:"
    )

    print(
        f"  anchors       : "
        f"{train_stats['articles']:,}"
    )

    print(
        f"  chunks        : "
        f"{train_stats['actual_chunks']:,}"
    )

    print(
        f"  mean chunks   : "
        f"{train_stats['chunks_per_article']['mean']:.2f}"
    )

    print()
    print(
        "VALIDATION:"
    )

    print(
        f"  anchors       : "
        f"{validation_stats['articles']:,}"
    )

    print(
        f"  chunks        : "
        f"{validation_stats['actual_chunks']:,}"
    )

    print(
        f"  mean chunks   : "
        f"{validation_stats['chunks_per_article']['mean']:.2f}"
    )

    print()
    print(
        f"Cache directory:\n"
        f"{OUTPUT_DIR}"
    )

    print()
    print(
        f"Report:\n"
        f"{REPORT_PATH}"
    )

    print()
    print(
        "TEST WAS NOT USED."
    )

    print(
        "STEP 17 FINISHED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()