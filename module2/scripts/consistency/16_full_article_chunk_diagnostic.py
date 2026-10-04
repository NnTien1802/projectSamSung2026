from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import torch
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


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

ARTICLE_INDEX_PATH = (
    MODULE_ROOT
    / "data"
    / "interim"
    / "consistency"
    / "articles_v1"
    / "article_index.parquet"
)

ARTICLE_PATH = (
    MODULE_ROOT
    / "data"
    / "interim"
    / "consistency"
    / "articles_v1"
    / "validation.parquet"
)

A_PRED_PATH = (
    MODULE_ROOT
    / "results"
    / "consistency"
    / "baseline_title_description_mbert"
    / "validation_predictions.parquet"
)

B_MODEL_PATH = (
    MODULE_ROOT
    / "models"
    / "consistency"
    / "baseline_title_article_head_mbert"
    / "best_model.pt"
)

OUTPUT_DIR = (
    MODULE_ROOT
    / "results"
    / "consistency"
    / "full_article_chunk_diagnostic"
)

PAIR_SCORE_PATH = (
    OUTPUT_DIR
    / "validation_chunk_scores.parquet"
)

SEARCH_PATH = (
    OUTPUT_DIR
    / "fusion_search.csv"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "diagnostic_report.json"
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "bert-base-multilingual-cased"

SEED = 20260920

# Same token budget as Baseline A/B.
MAX_LENGTH = 160

# Raw article tokens per chunk.
#
# Headline + [CLS]/[SEP]/[SEP] still consume part
# of MAX_LENGTH, so the collator defensively truncates
# the article chunk if required.
CHUNK_TOKENS = 120

CHUNK_OVERLAP = 24

# Maximum sampled windows from one full article.
MAX_CHUNKS = 6

INFERENCE_BATCH_SIZE = 32

ALPHA_VALUES = np.arange(
    0.0,
    1.001,
    0.05,
)

THRESHOLD_VALUES = np.arange(
    0.05,
    0.951,
    0.01,
)

AGGREGATIONS = [
    "max",
    "mean",
    "top2_mean",
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


def stable_key(
    value,
):
    """
    Must remain consistent with Step 15 so that
    calibration/holdout anchor partitions are identical.
    """

    raw = (
        f"{value}|{SEED}"
    ).encode(
        "utf-8"
    )

    digest = hashlib.blake2b(
        raw,
        digest_size=8,
    ).digest()

    return int.from_bytes(
        digest,
        byteorder="big",
        signed=False,
    )


# ============================================================
# METRICS
# ============================================================

def metrics(
    labels,
    probabilities,
    threshold,
):

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    predictions = (
        probabilities
        >=
        threshold
    ).astype(
        np.int64
    )

    cm = confusion_matrix(
        labels,
        predictions,
        labels=[
            0,
            1,
        ],
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    return {
        "threshold":
            float(
                threshold
            ),

        "accuracy":
            float(
                accuracy_score(
                    labels,
                    predictions,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    labels,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),

        "f1_consistent":
            float(
                f1_score(
                    labels,
                    predictions,
                    zero_division=0,
                )
            ),

        "confusion_matrix": {
            "tn":
                int(
                    tn
                ),

            "fp":
                int(
                    fp
                ),

            "fn":
                int(
                    fn
                ),

            "tp":
                int(
                    tp
                ),
        },
    }


# ============================================================
# ARTICLE CHUNKING
# ============================================================

def build_chunks(
    content,
    tokenizer,
):
    """
    Tokenize the full article once, then generate overlapping
    article windows.

    For long articles, sample at most MAX_CHUNKS windows
    uniformly across the document so evidence near the middle
    or end is not ignored.
    """

    token_ids = tokenizer.encode(
        str(
            content
        ),
        add_special_tokens=False,
        verbose=False,
    )

    if not token_ids:

        return [
            []
        ]

    # --------------------------------------------------------
    # Article fits in one raw content chunk.
    # --------------------------------------------------------

    if len(
        token_ids
    ) <= CHUNK_TOKENS:

        return [
            token_ids
        ]

    step = (
        CHUNK_TOKENS
        -
        CHUNK_OVERLAP
    )

    if step <= 0:

        raise ValueError(
            "CHUNK_OVERLAP must be smaller "
            "than CHUNK_TOKENS."
        )

    starts = list(
        range(
            0,
            len(
                token_ids
            ),
            step,
        )
    )

    # --------------------------------------------------------
    # If too many possible chunks exist, sample positions
    # uniformly across the complete document.
    # --------------------------------------------------------

    if len(
        starts
    ) > MAX_CHUNKS:

        indices = np.linspace(
            0,
            len(
                starts
            )
            -
            1,
            MAX_CHUNKS,
        )

        indices = np.round(
            indices
        ).astype(
            int
        )

        # Avoid duplicated rounded indices.
        indices = list(
            dict.fromkeys(
                indices.tolist()
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

    if not chunks:

        chunks = [
            token_ids[
                :CHUNK_TOKENS
            ]
        ]

    return chunks


# ============================================================
# DATASET
# ============================================================

class ChunkPairDataset(
    Dataset
):

    def __init__(
        self,
        records,
    ):

        self.records = records

    def __len__(
        self,
    ):

        return len(
            self.records
        )

    def __getitem__(
        self,
        index,
    ):

        return self.records[
            index
        ]


# ============================================================
# COLLATOR
# ============================================================

class ChunkCollator:
    """
    Build BERT sentence pairs manually.

    Layout:

        [CLS] headline [SEP] article_chunk [SEP]

    token_type_ids:

        0     headline      0      article      1
    """

    def __init__(
        self,
        tokenizer,
    ):

        self.tokenizer = tokenizer

        if (
            tokenizer.cls_token_id
            is None
            or
            tokenizer.sep_token_id
            is None
            or
            tokenizer.pad_token_id
            is None
        ):

            raise RuntimeError(
                "Tokenizer is missing "
                "CLS/SEP/PAD token IDs."
            )

        self.cls_id = int(
            tokenizer.cls_token_id
        )

        self.sep_id = int(
            tokenizer.sep_token_id
        )

        self.pad_id = int(
            tokenizer.pad_token_id
        )

    def __call__(
        self,
        batch,
    ):

        input_id_rows = []
        token_type_rows = []
        attention_rows = []

        pair_ids = []
        chunk_indices = []

        special_tokens = 3

        for item in batch:

            # =================================================
            # HEADLINE TOKENS
            # =================================================

            headline_ids = (
                self.tokenizer.encode(
                    str(
                        item[
                            "headline"
                        ]
                    ),
                    add_special_tokens=False,
                    verbose=False,
                )
            )

            # Keep at least one token position available
            # for the article side.
            max_headline_tokens = (
                MAX_LENGTH
                -
                special_tokens
                -
                1
            )

            if (
                len(
                    headline_ids
                )
                >
                max_headline_tokens
            ):

                headline_ids = (
                    headline_ids[
                        :max_headline_tokens
                    ]
                )

            # =================================================
            # AVAILABLE ARTICLE TOKENS
            # =================================================

            remaining_for_chunk = (
                MAX_LENGTH
                -
                len(
                    headline_ids
                )
                -
                special_tokens
            )

            remaining_for_chunk = max(
                0,
                remaining_for_chunk,
            )

            chunk_ids = list(
                item[
                    "chunk_ids"
                ]
            )[
                :remaining_for_chunk
            ]

            # =================================================
            # BUILD
            #
            # [CLS] headline [SEP] chunk [SEP]
            # =================================================

            input_ids = (
                [
                    self.cls_id
                ]
                +
                headline_ids
                +
                [
                    self.sep_id
                ]
                +
                chunk_ids
                +
                [
                    self.sep_id
                ]
            )

            token_type_ids = (
                [
                    0
                ]
                *
                (
                    len(
                        headline_ids
                    )
                    +
                    2
                )
                +
                [
                    1
                ]
                *
                (
                    len(
                        chunk_ids
                    )
                    +
                    1
                )
            )

            attention_mask = (
                [
                    1
                ]
                *
                len(
                    input_ids
                )
            )

            # =================================================
            # SAFETY
            # =================================================

            if (
                len(
                    input_ids
                )
                >
                MAX_LENGTH
            ):

                raise RuntimeError(
                    "Constructed BERT input "
                    "exceeds MAX_LENGTH."
                )

            if (
                len(
                    input_ids
                )
                !=
                len(
                    token_type_ids
                )
            ):

                raise RuntimeError(
                    "input_ids/token_type_ids "
                    "length mismatch."
                )

            if (
                len(
                    input_ids
                )
                !=
                len(
                    attention_mask
                )
            ):

                raise RuntimeError(
                    "input_ids/attention_mask "
                    "length mismatch."
                )

            input_id_rows.append(
                input_ids
            )

            token_type_rows.append(
                token_type_ids
            )

            attention_rows.append(
                attention_mask
            )

            pair_ids.append(
                item[
                    "pair_id"
                ]
            )

            chunk_indices.append(
                int(
                    item[
                        "chunk_index"
                    ]
                )
            )

        # =====================================================
        # DYNAMIC PADDING
        # =====================================================

        batch_max_length = max(
            len(
                row
            )
            for row
            in input_id_rows
        )

        if (
            batch_max_length
            >
            MAX_LENGTH
        ):

            raise RuntimeError(
                "Batch sequence exceeds MAX_LENGTH."
            )

        for i in range(
            len(
                input_id_rows
            )
        ):

            padding_length = (
                batch_max_length
                -
                len(
                    input_id_rows[
                        i
                    ]
                )
            )

            if padding_length > 0:

                input_id_rows[
                    i
                ].extend(
                    [
                        self.pad_id
                    ]
                    *
                    padding_length
                )

                # Padding segment IDs conventionally use 0.
                token_type_rows[
                    i
                ].extend(
                    [
                        0
                    ]
                    *
                    padding_length
                )

                attention_rows[
                    i
                ].extend(
                    [
                        0
                    ]
                    *
                    padding_length
                )

        return {
            "input_ids":
                torch.tensor(
                    input_id_rows,
                    dtype=torch.long,
                ),

            "token_type_ids":
                torch.tensor(
                    token_type_rows,
                    dtype=torch.long,
                ),

            "attention_mask":
                torch.tensor(
                    attention_rows,
                    dtype=torch.long,
                ),

            "pair_ids":
                pair_ids,

            "chunk_indices":
                chunk_indices,
        }


# ============================================================
# BUILD VALIDATION CHUNK RECORDS
# ============================================================

def prepare_records(
    pairs,
    title_map,
    content_map,
    tokenizer,
):

    print()
    print("=" * 80)
    print(
        "BUILD VALIDATION CHUNKS"
    )
    print("=" * 80)

    # Tokenize every anchor article once.
    anchor_chunks = {}

    unique_anchors = (
        pairs[
            "anchor_article_uid"
        ]
        .unique()
        .tolist()
    )

    print(
        f"Unique anchors: "
        f"{len(unique_anchors):,}"
    )

    chunk_counts = []

    for i, anchor_uid in enumerate(
        unique_anchors,
        start=1,
    ):

        chunks = build_chunks(
            content_map[
                anchor_uid
            ],
            tokenizer,
        )

        anchor_chunks[
            anchor_uid
        ] = chunks

        chunk_counts.append(
            len(
                chunks
            )
        )

        if i % 5000 == 0:

            print(
                f"  tokenized "
                f"{i:,}/"
                f"{len(unique_anchors):,}"
            )

    print()

    print(
        f"Mean chunks/article  : "
        f"{np.mean(chunk_counts):.2f}"
    )

    print(
        f"Median chunks/article: "
        f"{np.median(chunk_counts):.1f}"
    )

    print(
        f"Min chunks/article   : "
        f"{min(chunk_counts)}"
    )

    print(
        f"Max chunks/article   : "
        f"{max(chunk_counts)}"
    )

    records = []

    for row in pairs.itertuples(
        index=False,
    ):

        headline = (
            title_map[
                row.headline_article_uid
            ]
        )

        chunks = (
            anchor_chunks[
                row.anchor_article_uid
            ]
        )

        for (
            chunk_index,
            chunk_ids,
        ) in enumerate(
            chunks
        ):

            records.append(
                {
                    "pair_id":
                        row.pair_id,

                    "headline":
                        headline,

                    "chunk_ids":
                        chunk_ids,

                    "chunk_index":
                        chunk_index,
                }
            )

    print()

    print(
        f"Chunk-pair instances: "
        f"{len(records):,}"
    )

    return records


# ============================================================
# CHUNK INFERENCE
# ============================================================

@torch.no_grad()
def score_chunks(
    model,
    loader,
    device,
):

    model.eval()

    rows = []

    use_amp = (
        device.type
        ==
        "cuda"
    )

    total_batches = len(
        loader
    )

    for (
        batch_index,
        batch,
    ) in enumerate(
        loader,
        start=1,
    ):

        pair_ids = batch.pop(
            "pair_ids"
        )

        chunk_indices = batch.pop(
            "chunk_indices"
        )

        batch = {
            key:
                value.to(
                    device,
                    non_blocking=True,
                )

            for key, value
            in batch.items()
        }

        with torch.amp.autocast(
            device_type=(
                device.type
            ),
            enabled=use_amp,
        ):

            output = model(
                **batch
            )

        probabilities = (
            torch.softmax(
                output.logits,
                dim=-1,
            )[
                :,
                1
            ]
            .detach()
            .cpu()
            .numpy()
        )

        for (
            pair_id,
            chunk_index,
            probability,
        ) in zip(
            pair_ids,
            chunk_indices,
            probabilities,
        ):

            rows.append(
                {
                    "pair_id":
                        pair_id,

                    "chunk_index":
                        int(
                            chunk_index
                        ),

                    "prob_consistent":
                        float(
                            probability
                        ),
                }
            )

        if (
            batch_index
            %
            500
            ==
            0
            or
            batch_index
            ==
            total_batches
        ):

            print(
                f"  inference batch "
                f"{batch_index:,}/"
                f"{total_batches:,}"
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# AGGREGATE CHUNKS
# ============================================================

def aggregate_chunk_scores(
    chunk_scores,
):

    rows = []

    for (
        pair_id,
        group,
    ) in chunk_scores.groupby(
        "pair_id",
        sort=False,
    ):

        values = (
            group[
                "prob_consistent"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        sorted_values = np.sort(
            values
        )[
            ::-1
        ]

        top2 = sorted_values[
            :min(
                2,
                len(
                    sorted_values
                ),
            )
        ]

        rows.append(
            {
                "pair_id":
                    pair_id,

                "chunk_count":
                    int(
                        len(
                            values
                        )
                    ),

                "chunk_max":
                    float(
                        values.max()
                    ),

                "chunk_mean":
                    float(
                        values.mean()
                    ),

                "chunk_top2_mean":
                    float(
                        top2.mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# DESCRIPTION + CHUNK FUSION SEARCH
# ============================================================

def tune_fusion(
    calibration,
):

    labels = (
        calibration[
            "label"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    prob_description = (
        calibration[
            "prob_a"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    best = None

    search_rows = []

    for aggregation in (
        AGGREGATIONS
    ):

        chunk_column = {
            "max":
                "chunk_max",

            "mean":
                "chunk_mean",

            "top2_mean":
                "chunk_top2_mean",
        }[
            aggregation
        ]

        chunk_probabilities = (
            calibration[
                chunk_column
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        for alpha in (
            ALPHA_VALUES
        ):

            fused = (
                alpha
                *
                prob_description
                +
                (
                    1.0
                    -
                    alpha
                )
                *
                chunk_probabilities
            )

            for threshold in (
                THRESHOLD_VALUES
            ):

                predictions = (
                    fused
                    >=
                    threshold
                ).astype(
                    np.int64
                )

                macro_f1 = float(
                    f1_score(
                        labels,
                        predictions,
                        average="macro",
                        zero_division=0,
                    )
                )

                accuracy = float(
                    accuracy_score(
                        labels,
                        predictions,
                    )
                )

                row = {
                    "aggregation":
                        aggregation,

                    "alpha_description":
                        float(
                            alpha
                        ),

                    "alpha_chunks":
                        float(
                            1.0
                            -
                            alpha
                        ),

                    "threshold":
                        float(
                            threshold
                        ),

                    "macro_f1":
                        macro_f1,

                    "accuracy":
                        accuracy,
                }

                search_rows.append(
                    row
                )

                # ------------------------------------------------
                # Tie-break:
                # 1. Macro-F1
                # 2. Accuracy
                # 3. Weight closer to equal fusion
                # 4. Threshold closer to 0.5
                # ------------------------------------------------

                key = (
                    macro_f1,
                    accuracy,
                    -abs(
                        alpha
                        -
                        0.5
                    ),
                    -abs(
                        threshold
                        -
                        0.5
                    ),
                )

                if best is None:

                    best = (
                        row,
                        key,
                    )

                elif (
                    key
                    >
                    best[
                        1
                    ]
                ):

                    best = (
                        row,
                        key,
                    )

    return (
        best[
            0
        ],
        pd.DataFrame(
            search_rows
        ),
    )


# ============================================================
# DESCRIPTION-ONLY THRESHOLD
# ============================================================

def tune_description_threshold(
    calibration,
):

    best = None

    for threshold in (
        THRESHOLD_VALUES
    ):

        candidate = metrics(
            calibration[
                "label"
            ],
            calibration[
                "prob_a"
            ],
            float(
                threshold
            ),
        )

        key = (
            candidate[
                "macro_f1"
            ],
            candidate[
                "accuracy"
            ],
            -abs(
                threshold
                -
                0.5
            ),
        )

        if best is None:

            best = (
                candidate,
                key,
            )

        elif (
            key
            >
            best[
                1
            ]
        ):

            best = (
                candidate,
                key,
            )

    return best[
        0
    ]


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
        "STEP 16 - FULL ARTICLE CHUNK DIAGNOSTIC"
    )
    print("=" * 80)

    print()
    print(
        "VALIDATION ONLY."
    )

    print(
        "TEST IS NOT LOADED."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # REQUIRED FILES
    # ========================================================

    for path in [
        PAIR_PATH,
        ARTICLE_INDEX_PATH,
        ARTICLE_PATH,
        A_PRED_PATH,
        B_MODEL_PATH,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    print()
    print(
        f"Device: "
        f"{device}"
    )

    if (
        device.type
        ==
        "cuda"
    ):

        print(
            "GPU   : "
            f"{torch.cuda.get_device_name(0)}"
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

    # ========================================================
    # ARTICLE INDEX
    # ========================================================

    article_index = (
        pq.read_table(
            ARTICLE_INDEX_PATH,
            columns=[
                "article_uid",
                "title",
                "split",
            ],
        )
        .to_pandas()
    )

    if (
        article_index[
            "article_uid"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate article_uid "
            "in article index."
        )

    title_map = dict(
        zip(
            article_index[
                "article_uid"
            ],
            article_index[
                "title"
            ],
        )
    )

    split_map = dict(
        zip(
            article_index[
                "article_uid"
            ],
            article_index[
                "split"
            ],
        )
    )

    # ========================================================
    # VALIDATION PAIRS
    # ========================================================

    pairs = (
        pq.read_table(
            PAIR_PATH,
            columns=[
                "pair_id",
                "anchor_article_uid",
                "headline_article_uid",
                "anchor_category",
                "label",
            ],
        )
        .to_pandas()
    )

    if (
        pairs[
            "pair_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate validation pair_id."
        )

    if not (
        pairs[
            "anchor_article_uid"
        ]
        .map(
            split_map
        )
        ==
        "validation"
    ).all():

        raise RuntimeError(
            "Anchor split leakage."
        )

    if not (
        pairs[
            "headline_article_uid"
        ]
        .map(
            split_map
        )
        ==
        "validation"
    ).all():

        raise RuntimeError(
            "Headline split leakage."
        )

    print()
    print(
        f"Validation pairs: "
        f"{len(pairs):,}"
    )

    print(
        f"Validation anchors: "
        f"{pairs['anchor_article_uid'].nunique():,}"
    )

    # ========================================================
    # VALIDATION ARTICLE CONTENT
    # ========================================================

    print()
    print(
        "Loading validation articles..."
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

    if (
        article_frame[
            "article_uid"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate validation article_uid."
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

    required_anchors = set(
        pairs[
            "anchor_article_uid"
        ]
    )

    missing = (
        required_anchors
        -
        set(
            content_map
        )
    )

    if missing:

        raise RuntimeError(
            f"Missing anchor content: "
            f"{len(missing):,}"
        )

    # ========================================================
    # BUILD CHUNK RECORDS
    # ========================================================

    records = prepare_records(
        pairs=pairs,
        title_map=title_map,
        content_map=content_map,
        tokenizer=tokenizer,
    )

    dataset = ChunkPairDataset(
        records
    )

    collator = ChunkCollator(
        tokenizer
    )

    loader = DataLoader(
        dataset,
        batch_size=(
            INFERENCE_BATCH_SIZE
        ),
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type
            ==
            "cuda"
        ),
        collate_fn=collator,
    )

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print(
        "Loading Baseline B checkpoint..."
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=2,
        )
    )

    checkpoint = torch.load(
        B_MODEL_PATH,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(
        device
    )

    model.eval()

    # ========================================================
    # CHUNK INFERENCE
    # ========================================================

    print()
    print("=" * 80)
    print(
        "CHUNK INFERENCE"
    )
    print("=" * 80)

    print()
    print(
        f"Inference batch size: "
        f"{INFERENCE_BATCH_SIZE}"
    )

    print(
        f"Total instances     : "
        f"{len(dataset):,}"
    )

    print(
        f"Total batches       : "
        f"{len(loader):,}"
    )

    chunk_scores = score_chunks(
        model=model,
        loader=loader,
        device=device,
    )

    print()
    print(
        f"Scored chunk pairs: "
        f"{len(chunk_scores):,}"
    )

    if (
        len(
            chunk_scores
        )
        !=
        len(
            records
        )
    ):

        raise RuntimeError(
            "Chunk inference row count mismatch."
        )

    # ========================================================
    # AGGREGATION
    # ========================================================

    pair_scores = (
        aggregate_chunk_scores(
            chunk_scores
        )
    )

    if (
        len(
            pair_scores
        )
        !=
        len(
            pairs
        )
    ):

        raise RuntimeError(
            "Aggregated pair count mismatch."
        )

    pair_scores.to_parquet(
        PAIR_SCORE_PATH,
        index=False,
    )

    # ========================================================
    # BASELINE A DESCRIPTION PREDICTIONS
    # ========================================================

    a_pred = (
        pq.read_table(
            A_PRED_PATH,
            columns=[
                "pair_id",
                "prob_consistent",
            ],
        )
        .to_pandas()
        .rename(
            columns={
                "prob_consistent":
                    "prob_a",
            }
        )
    )

    if (
        a_pred[
            "pair_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate Baseline A pair_id."
        )

    # ========================================================
    # JOIN
    # ========================================================

    frame = (
        pairs
        .merge(
            a_pred,
            on="pair_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            pair_scores,
            on="pair_id",
            how="inner",
            validate="one_to_one",
        )
    )

    if (
        len(
            frame
        )
        !=
        len(
            pairs
        )
    ):

        raise RuntimeError(
            "Pair-score join lost rows."
        )

    # ========================================================
    # SAME INTERNAL SPLIT AS STEP 15
    # ========================================================

    anchors = (
        frame[
            [
                "anchor_article_uid"
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    anchors[
        "__key"
    ] = (
        anchors[
            "anchor_article_uid"
        ]
        .map(
            stable_key
        )
    )

    anchors.sort_values(
        "__key",
        inplace=True,
    )

    midpoint = (
        len(
            anchors
        )
        //
        2
    )

    calibration_anchors = set(
        anchors.iloc[
            :midpoint
        ][
            "anchor_article_uid"
        ]
    )

    holdout_anchors = set(
        anchors.iloc[
            midpoint:
        ][
            "anchor_article_uid"
        ]
    )

    if (
        calibration_anchors
        &
        holdout_anchors
    ):

        raise RuntimeError(
            "Calibration/holdout "
            "anchor overlap."
        )

    calibration = (
        frame[
            frame[
                "anchor_article_uid"
            ]
            .isin(
                calibration_anchors
            )
        ]
        .copy()
    )

    holdout = (
        frame[
            frame[
                "anchor_article_uid"
            ]
            .isin(
                holdout_anchors
            )
        ]
        .copy()
    )

    print()
    print(
        "Internal split:"
    )

    print(
        f"  calibration anchors: "
        f"{len(calibration_anchors):,}"
    )

    print(
        f"  calibration pairs  : "
        f"{len(calibration):,}"
    )

    print(
        f"  holdout anchors    : "
        f"{len(holdout_anchors):,}"
    )

    print(
        f"  holdout pairs      : "
        f"{len(holdout):,}"
    )

    # Each anchor has one positive and one negative.
    for (
        subset_name,
        subset,
    ) in [
        (
            "calibration",
            calibration,
        ),
        (
            "holdout",
            holdout,
        ),
    ]:

        counts = (
            subset[
                "label"
            ]
            .value_counts()
        )

        n0 = int(
            counts.get(
                0,
                0,
            )
        )

        n1 = int(
            counts.get(
                1,
                0,
            )
        )

        if n0 != n1:

            raise RuntimeError(
                f"{subset_name} imbalance: "
                f"{n0:,} vs {n1:,}"
            )

    # ========================================================
    # SEARCH BEST FUSION ON CALIBRATION ONLY
    # ========================================================

    (
        best,
        search_frame,
    ) = tune_fusion(
        calibration
    )

    search_frame.to_csv(
        SEARCH_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    aggregation = (
        best[
            "aggregation"
        ]
    )

    chunk_column = {
        "max":
            "chunk_max",

        "mean":
            "chunk_mean",

        "top2_mean":
            "chunk_top2_mean",
    }[
        aggregation
    ]

    alpha_description = float(
        best[
            "alpha_description"
        ]
    )

    alpha_chunks = float(
        best[
            "alpha_chunks"
        ]
    )

    fusion_threshold = float(
        best[
            "threshold"
        ]
    )

    # ========================================================
    # FAIR DESCRIPTION-ONLY BASELINE
    #
    # Tune threshold on the same calibration subset.
    # ========================================================

    best_description = (
        tune_description_threshold(
            calibration
        )
    )

    # ========================================================
    # HOLDOUT EVALUATION
    # ========================================================

    holdout_description_prob = (
        holdout[
            "prob_a"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    holdout_chunk_prob = (
        holdout[
            chunk_column
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    holdout_fused_prob = (
        alpha_description
        *
        holdout_description_prob
        +
        alpha_chunks
        *
        holdout_chunk_prob
    )

    holdout_fusion_metrics = (
        metrics(
            holdout[
                "label"
            ],
            holdout_fused_prob,
            fusion_threshold,
        )
    )

    holdout_description_metrics = (
        metrics(
            holdout[
                "label"
            ],
            holdout_description_prob,
            best_description[
                "threshold"
            ],
        )
    )

    delta = (
        holdout_fusion_metrics[
            "macro_f1"
        ]
        -
        holdout_description_metrics[
            "macro_f1"
        ]
    )

    # ========================================================
    # REPORT
    # ========================================================

    report = {
        "stage":
            "16_full_article_chunk_diagnostic",

        "dataset":
            "headline_consistency_v1",

        "analysis_split":
            "validation",

        "test_used":
            False,

        "model_source":
            (
                "Step 13 Baseline B "
                "headline + article-head checkpoint"
            ),

        "chunking": {
            "max_length":
                MAX_LENGTH,

            "chunk_tokens":
                CHUNK_TOKENS,

            "chunk_overlap":
                CHUNK_OVERLAP,

            "max_chunks":
                MAX_CHUNKS,

            "sampling":
                (
                    "uniform positions across "
                    "the full anchor article"
                ),

            "chunk_instances":
                int(
                    len(
                        chunk_scores
                    )
                ),

            "mean_chunks_per_article":
                float(
                    pair_scores[
                        "chunk_count"
                    ].mean()
                ),
        },

        "internal_validation": {
            "split_unit":
                "anchor_article_uid",

            "calibration_anchors":
                int(
                    len(
                        calibration_anchors
                    )
                ),

            "holdout_anchors":
                int(
                    len(
                        holdout_anchors
                    )
                ),

            "calibration_pairs":
                int(
                    len(
                        calibration
                    )
                ),

            "holdout_pairs":
                int(
                    len(
                        holdout
                    )
                ),
        },

        "calibration_best": {
            "aggregation":
                aggregation,

            "description_weight":
                alpha_description,

            "chunk_weight":
                alpha_chunks,

            "threshold":
                fusion_threshold,

            "macro_f1":
                float(
                    best[
                        "macro_f1"
                    ]
                ),
        },

        "description_baseline": {
            "calibration_threshold":
                float(
                    best_description[
                        "threshold"
                    ]
                ),
        },

        "holdout": {
            "description_only":
                holdout_description_metrics,

            "description_plus_chunks":
                holdout_fusion_metrics,

            "delta_macro_f1":
                float(
                    delta
                ),
        },

        "warning":
            (
                "Diagnostic on validation only. "
                "Test remains untouched."
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
        "STEP 16 RESULTS"
    )
    print("=" * 80)

    print()
    print(
        "Best calibration configuration:"
    )

    print(
        f"  aggregation        : "
        f"{aggregation}"
    )

    print(
        f"  description weight : "
        f"{alpha_description:.2f}"
    )

    print(
        f"  chunk weight       : "
        f"{alpha_chunks:.2f}"
    )

    print(
        f"  threshold          : "
        f"{fusion_threshold:.2f}"
    )

    print(
        f"  calibration Macro-F1: "
        f"{best['macro_f1']:.6f}"
    )

    print()
    print(
        "HOLDOUT:"
    )

    print(
        f"  Description threshold: "
        f"{best_description['threshold']:.2f}"
    )

    print(
        f"  Description only    : "
        f"{holdout_description_metrics['macro_f1']:.6f}"
    )

    print(
        f"  Description + chunks: "
        f"{holdout_fusion_metrics['macro_f1']:.6f}"
    )

    print(
        f"  Delta               : "
        f"{delta:+.6f}"
    )

    print()
    print(
        "Fusion confusion:"
    )

    cm = (
        holdout_fusion_metrics[
            "confusion_matrix"
        ]
    )

    print(
        f"  TN={cm['tn']:,} "
        f"FP={cm['fp']:,}"
    )

    print(
        f"  FN={cm['fn']:,} "
        f"TP={cm['tp']:,}"
    )

    print()
    print(
        "Outputs:"
    )

    print(
        f"  Pair scores:\n"
        f"    {PAIR_SCORE_PATH}"
    )

    print(
        f"  Search:\n"
        f"    {SEARCH_PATH}"
    )

    print(
        f"  Report:\n"
        f"    {REPORT_PATH}"
    )

    print()
    print(
        "TEST WAS NOT USED."
    )

    print(
        "STEP 16 FINISHED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()