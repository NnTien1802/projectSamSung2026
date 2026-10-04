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

PAIR_PATH = (
    MODULE_ROOT
    / "data"
    / "processed"
    / "consistency"
    / "headline_consistency_v1"
    / "test.parquet"
)

ARTICLE_PATH = (
    MODULE_ROOT
    / "data"
    / "interim"
    / "consistency"
    / "articles_v1"
    / "test.parquet"
)

CACHE_DIR = (
    MODULE_ROOT
    / "data"
    / "interim"
    / "consistency"
    / "mil_chunk_cache_mbert_v1"
)

LOCK_PATH = (
    MODULE_ROOT
    / "results"
    / "consistency"
    / "final_hybrid_v1"
    / "final_model_lock.json"
)

REPORT_PATH = (
    CACHE_DIR
    / "test_cache_report.json"
)


# ============================================================
# FROZEN CONFIG
# ============================================================

MODEL_NAME = (
    "bert-base-multilingual-cased"
)

CHUNK_TOKENS = 120
CHUNK_OVERLAP = 24
MAX_CHUNKS = 6

FINAL_DESCRIPTION_WEIGHT = 0.50
FINAL_MIL_WEIGHT = 0.50
FINAL_THRESHOLD = 0.45


# ============================================================
# HELPERS
# ============================================================

def save_json(
    data,
    path,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    token_ids = tokenizer.encode(
        str(content),
        add_special_tokens=False,
        verbose=False,
    )

    total_tokens = len(
        token_ids
    )

    if total_tokens == 0:

        return (
            [[]],
            0,
            [0],
        )

    if (
        total_tokens
        <=
        CHUNK_TOKENS
    ):

        return (
            [token_ids],
            total_tokens,
            [0],
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

    if (
        len(starts)
        >
        MAX_CHUNKS
    ):

        indices = np.linspace(
            0,
            len(starts) - 1,
            MAX_CHUNKS,
        )

        indices = (
            np.round(indices)
            .astype(int)
            .tolist()
        )

        indices = list(
            dict.fromkeys(
                indices
            )
        )

        starts = [
            starts[index]
            for index
            in indices
        ]

    chunks = []
    used_starts = []

    for start in starts:

        chunk = token_ids[
            start:
            start + CHUNK_TOKENS
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

        used_starts = [0]

    return (
        chunks,
        total_tokens,
        used_starts,
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
        "STEP 20 - PREPARE TEST MIL CHUNK CACHE"
    )
    print("=" * 80)

    print()
    print(
        "FINAL ARCHITECTURE IS NOW FROZEN."
    )

    print()
    print(
        f"Description weight : "
        f"{FINAL_DESCRIPTION_WEIGHT:.2f}"
    )

    print(
        f"MIL weight         : "
        f"{FINAL_MIL_WEIGHT:.2f}"
    )

    print(
        f"Final threshold     : "
        f"{FINAL_THRESHOLD:.2f}"
    )

    print()
    print(
        "NO TEST-BASED TUNING IS ALLOWED."
    )

    for path in [
        PAIR_PATH,
        ARTICLE_PATH,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # SAVE FINAL LOCK BEFORE TEST PREPROCESSING
    # ========================================================

    lock = {
        "architecture":
            (
                "description_mbert "
                "+ full_article_max_mil_mbert"
            ),

        "status":
            "FROZEN_BEFORE_TEST_EVALUATION",

        "description_weight":
            FINAL_DESCRIPTION_WEIGHT,

        "mil_weight":
            FINAL_MIL_WEIGHT,

        "decision_threshold":
            FINAL_THRESHOLD,

        "model_name":
            MODEL_NAME,

        "chunk_policy": {
            "chunk_tokens":
                CHUNK_TOKENS,

            "chunk_overlap":
                CHUNK_OVERLAP,

            "max_chunks":
                MAX_CHUNKS,

            "sampling":
                (
                    "uniform positions across "
                    "full article"
                ),
        },

        "selection_source":
            (
                "validation calibration/holdout "
                "Steps 15-19"
            ),

        "test_tuning_allowed":
            False,
    }

    save_json(
        lock,
        LOCK_PATH,
    )

    print()
    print(
        f"Final configuration locked:\n"
        f"{LOCK_PATH}"
    )

    # ========================================================
    # LOAD TEST ANCHORS
    # ========================================================

    pair_frame = (
        pq.read_table(
            PAIR_PATH,
            columns=[
                "anchor_article_uid",
            ],
        )
        .to_pandas()
    )

    required_anchors = (
        pair_frame[
            "anchor_article_uid"
        ]
        .drop_duplicates()
        .astype(str)
        .sort_values()
        .tolist()
    )

    print()
    print(
        f"TEST anchors required: "
        f"{len(required_anchors):,}"
    )

    # Expected from final dataset:
    # 28,688 anchors.

    # ========================================================
    # LOAD TEST ARTICLES
    # ========================================================

    print(
        "Loading TEST article content..."
    )

    article_frame = (
        pq.read_table(
            ARTICLE_PATH,
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
            "TEST contains duplicate article_uid."
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

    found = set(
        article_frame[
            "article_uid"
        ]
    )

    missing = (
        set(required_anchors)
        -
        found
    )

    if missing:

        raise RuntimeError(
            f"Missing TEST anchors: "
            f"{len(missing):,}"
        )

    if (
        len(article_frame)
        !=
        len(required_anchors)
    ):

        raise RuntimeError(
            "TEST article count mismatch."
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

    del article_frame

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

    # ========================================================
    # OUTPUTS
    # ========================================================

    chunk_ids_path = (
        CACHE_DIR
        / "test_chunk_ids.npy"
    )

    chunk_lengths_path = (
        CACHE_DIR
        / "test_chunk_lengths.npy"
    )

    chunk_starts_path = (
        CACHE_DIR
        / "test_chunk_starts.npy"
    )

    chunk_counts_path = (
        CACHE_DIR
        / "test_chunk_counts.npy"
    )

    article_index_path = (
        CACHE_DIR
        / "test_article_index.parquet"
    )

    n_articles = len(
        required_anchors
    )

    # ========================================================
    # MEMORY MAPS
    # ========================================================

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

    chunk_ids[:] = 0
    chunk_lengths[:] = 0
    chunk_starts[:] = -1
    chunk_counts[:] = 0

    # ========================================================
    # TOKENIZE
    # ========================================================

    print()
    print(
        "Building TEST chunk cache..."
    )

    index_rows = []

    token_counts = []
    count_values = []

    for row_idx, article_uid in enumerate(
        required_anchors
    ):

        (
            chunks,
            token_count,
            starts,
        ) = build_chunks(
            content_map[
                article_uid
            ],
            tokenizer,
        )

        if (
            len(chunks)
            >
            MAX_CHUNKS
        ):

            raise RuntimeError(
                "Generated too many chunks."
            )

        chunk_counts[
            row_idx
        ] = len(
            chunks
        )

        token_counts.append(
            token_count
        )

        count_values.append(
            len(chunks)
        )

        for (
            chunk_index,
            chunk,
        ) in enumerate(
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
                    "Chunk too long."
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

        index_rows.append(
            {
                "row_idx":
                    int(row_idx),

                "article_uid":
                    article_uid,

                "token_count":
                    int(token_count),

                "chunk_count":
                    int(len(chunks)),
            }
        )

        if (
            (row_idx + 1)
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
    # FLUSH
    # ========================================================

    chunk_ids.flush()
    chunk_lengths.flush()
    chunk_starts.flush()
    chunk_counts.flush()

    # ========================================================
    # INDEX
    # ========================================================

    index_frame = pd.DataFrame(
        index_rows
    )

    index_frame.to_parquet(
        article_index_path,
        index=False,
    )

    # ========================================================
    # VALIDATE
    # ========================================================

    if (
        len(index_frame)
        !=
        n_articles
    ):

        raise RuntimeError(
            "TEST cache index mismatch."
        )

    if (
        index_frame[
            "article_uid"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate cached TEST UID."
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
            "Invalid TEST chunk count."
        )

    token_values = np.asarray(
        token_counts,
        dtype=np.int64,
    )

    chunk_count_values = np.asarray(
        count_values,
        dtype=np.int64,
    )

    report = {
        "stage":
            "20_prepare_test_mil_chunk_cache",

        "test_preprocessing_started":
            True,

        "test_metrics_computed":
            False,

        "final_model_locked":
            True,

        "articles":
            int(
                n_articles
            ),

        "actual_chunks":
            int(
                chunk_count_values.sum()
            ),

        "chunks_per_article": {
            "min":
                int(
                    chunk_count_values.min()
                ),

            "mean":
                float(
                    chunk_count_values.mean()
                ),

            "median":
                float(
                    np.median(
                        chunk_count_values
                    )
                ),

            "max":
                int(
                    chunk_count_values.max()
                ),
        },

        "article_tokens": {
            "median":
                float(
                    np.median(
                        token_values
                    )
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

        "frozen_fusion": {
            "description_weight":
                FINAL_DESCRIPTION_WEIGHT,

            "mil_weight":
                FINAL_MIL_WEIGHT,

            "threshold":
                FINAL_THRESHOLD,
        },
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
        "STEP 20 SUMMARY"
    )
    print("=" * 80)

    print()
    print(
        f"TEST anchors      : "
        f"{n_articles:,}"
    )

    print(
        f"TEST chunks       : "
        f"{report['actual_chunks']:,}"
    )

    print(
        f"Mean chunks       : "
        f"{report['chunks_per_article']['mean']:.2f}"
    )

    print(
        f"Median chunks     : "
        f"{report['chunks_per_article']['median']:.1f}"
    )

    print(
        f"Median tokens     : "
        f"{report['article_tokens']['median']:.1f}"
    )

    print(
        f"P95 tokens        : "
        f"{report['article_tokens']['p95']:.1f}"
    )

    print()
    print(
        "FINAL CONFIG REMAINS:"
    )

    print(
        "  description = 0.50"
    )

    print(
        "  MIL         = 0.50"
    )

    print(
        "  threshold   = 0.45"
    )

    print()
    print(
        "NO TEST METRICS HAVE BEEN "
        "COMPUTED YET."
    )

    print()
    print(
        f"Cache directory:\n"
        f"{CACHE_DIR}"
    )

    print()
    print(
        "STEP 20 FINISHED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()