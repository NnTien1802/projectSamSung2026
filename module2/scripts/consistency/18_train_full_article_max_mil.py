from pathlib import Path
import argparse
import json
import math
import random
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)


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

CACHE_DIR = (
    MODULE_ROOT
    / "data"
    / "interim"
    / "consistency"
    / "mil_chunk_cache_mbert_v1"
)

ARTICLE_INDEX_PATH = (
    MODULE_ROOT
    / "data"
    / "interim"
    / "consistency"
    / "articles_v1"
    / "article_index.parquet"
)

PAIR_FILES = {
    "train":
        PAIR_DIR
        / "train.parquet",

    "validation":
        PAIR_DIR
        / "validation.parquet",
}

B_MODEL_PATH = (
    MODULE_ROOT
    / "models"
    / "consistency"
    / "baseline_title_article_head_mbert"
    / "best_model.pt"
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "bert-base-multilingual-cased"

MAX_LENGTH = 160

CACHE_MAX_CHUNKS = 6

DEFAULT_BATCH_SIZE = 2

DEFAULT_EVAL_BATCH_SIZE = 4

DEFAULT_GRAD_ACCUM = 8

DEFAULT_EPOCHS = 1

# Initialized from already fine-tuned Baseline B,
# therefore use a slightly smaller learning rate.
DEFAULT_LR = 1e-5

DEFAULT_WEIGHT_DECAY = 0.01

DEFAULT_WARMUP_RATIO = 0.06

DEFAULT_MAX_GRAD_NORM = 1.0

DEFAULT_SEED = 20260920

DEFAULT_NUM_WORKERS = 0


# ============================================================
# GENERAL HELPERS
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
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


def human_time(seconds):

    seconds = int(
        seconds
    )

    hours = (
        seconds
        //
        3600
    )

    minutes = (
        (
            seconds
            %
            3600
        )
        //
        60
    )

    seconds = (
        seconds
        %
        60
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    labels,
    probabilities,
    threshold=0.5,
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

    try:

        roc_auc = roc_auc_score(
            labels,
            probabilities,
        )

    except ValueError:

        roc_auc = None

    try:

        ap = average_precision_score(
            labels,
            probabilities,
        )

    except ValueError:

        ap = None

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

        "precision_consistent":
            float(
                precision_score(
                    labels,
                    predictions,
                    zero_division=0,
                )
            ),

        "recall_consistent":
            float(
                recall_score(
                    labels,
                    predictions,
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

        "macro_f1":
            float(
                f1_score(
                    labels,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),

        "roc_auc":
            (
                float(
                    roc_auc
                )
                if roc_auc is not None
                else None
            ),

        "average_precision":
            (
                float(
                    ap
                )
                if ap is not None
                else None
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


def tune_threshold(
    labels,
    probabilities,
):

    best = None

    rows = []

    for threshold in np.arange(
        0.05,
        0.951,
        0.01,
    ):

        result = compute_metrics(
            labels=labels,
            probabilities=probabilities,
            threshold=float(
                threshold
            ),
        )

        rows.append(
            {
                "threshold":
                    result[
                        "threshold"
                    ],

                "accuracy":
                    result[
                        "accuracy"
                    ],

                "macro_f1":
                    result[
                        "macro_f1"
                    ],

                "f1_consistent":
                    result[
                        "f1_consistent"
                    ],
            }
        )

        key = (
            result[
                "macro_f1"
            ],
            result[
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
                result,
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
                result,
                key,
            )

    return (
        best[
            0
        ],
        pd.DataFrame(
            rows
        ),
    )


# ============================================================
# LOAD ARTICLE TITLES
# ============================================================

def load_title_data():

    print()
    print("=" * 80)
    print(
        "LOAD ARTICLE TITLES"
    )
    print("=" * 80)

    frame = (
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

    title_map = dict(
        zip(
            frame[
                "article_uid"
            ],
            frame[
                "title"
            ],
        )
    )

    split_map = dict(
        zip(
            frame[
                "article_uid"
            ],
            frame[
                "split"
            ],
        )
    )

    print(
        f"Articles indexed: "
        f"{len(title_map):,}"
    )

    return (
        title_map,
        split_map,
    )


# ============================================================
# LOAD PAIRS
# ============================================================

def load_pairs(
    split,
    split_map,
):

    frame = (
        pq.read_table(
            PAIR_FILES[
                split
            ],
            columns=[
                "pair_id",
                "anchor_article_uid",
                "headline_article_uid",
                "label",
            ],
        )
        .to_pandas()
    )

    frame[
        "anchor_article_uid"
    ] = (
        frame[
            "anchor_article_uid"
        ]
        .astype(str)
    )

    frame[
        "headline_article_uid"
    ] = (
        frame[
            "headline_article_uid"
        ]
        .astype(str)
    )

    if (
        frame[
            "pair_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            f"{split}: duplicate pair_id."
        )

    anchor_split = (
        frame[
            "anchor_article_uid"
        ]
        .map(
            split_map
        )
    )

    headline_split = (
        frame[
            "headline_article_uid"
        ]
        .map(
            split_map
        )
    )

    if not (
        anchor_split
        ==
        split
    ).all():

        raise RuntimeError(
            f"{split}: anchor leakage."
        )

    if not (
        headline_split
        ==
        split
    ).all():

        raise RuntimeError(
            f"{split}: headline leakage."
        )

    counts = (
        frame[
            "label"
        ]
        .value_counts()
    )

    print(
        f"{split:12s}: "
        f"{len(frame):,} pairs"
    )

    print(
        f"  label 0: "
        f"{int(counts.get(0, 0)):,}"
    )

    print(
        f"  label 1: "
        f"{int(counts.get(1, 0)):,}"
    )

    return frame


# ============================================================
# SMOKE SUBSET BY ANCHOR
# ============================================================

def anchor_subset(
    frame,
    max_anchors,
    seed,
):

    anchors = (
        frame[
            "anchor_article_uid"
        ]
        .drop_duplicates()
        .sample(
            n=min(
                max_anchors,
                frame[
                    "anchor_article_uid"
                ].nunique(),
            ),
            random_state=seed,
        )
    )

    anchor_set = set(
        anchors.tolist()
    )

    subset = (
        frame[
            frame[
                "anchor_article_uid"
            ]
            .isin(
                anchor_set
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    return subset


# ============================================================
# CACHE LOADER
# ============================================================

class ChunkCache:

    def __init__(
        self,
        split,
    ):

        index_path = (
            CACHE_DIR
            / f"{split}_article_index.parquet"
        )

        chunk_ids_path = (
            CACHE_DIR
            / f"{split}_chunk_ids.npy"
        )

        chunk_lengths_path = (
            CACHE_DIR
            / f"{split}_chunk_lengths.npy"
        )

        chunk_counts_path = (
            CACHE_DIR
            / f"{split}_chunk_counts.npy"
        )

        for path in [
            index_path,
            chunk_ids_path,
            chunk_lengths_path,
            chunk_counts_path,
        ]:

            if not path.exists():

                raise FileNotFoundError(
                    path
                )

        index_frame = (
            pq.read_table(
                index_path,
                columns=[
                    "row_idx",
                    "article_uid",
                    "chunk_count",
                ],
            )
            .to_pandas()
        )

        index_frame[
            "article_uid"
        ] = (
            index_frame[
                "article_uid"
            ]
            .astype(str)
        )

        if (
            index_frame[
                "article_uid"
            ]
            .duplicated()
            .any()
        ):

            raise RuntimeError(
                f"{split}: duplicate cache UID."
            )

        self.uid_to_row = dict(
            zip(
                index_frame[
                    "article_uid"
                ],
                index_frame[
                    "row_idx"
                ].astype(int),
            )
        )

        self.chunk_ids = np.load(
            chunk_ids_path,
            mmap_mode="r",
        )

        self.chunk_lengths = np.load(
            chunk_lengths_path,
            mmap_mode="r",
        )

        self.chunk_counts = np.load(
            chunk_counts_path,
            mmap_mode="r",
        )

        if (
            self.chunk_ids.shape[
                1
            ]
            !=
            CACHE_MAX_CHUNKS
        ):

            raise RuntimeError(
                f"{split}: unexpected "
                "MAX_CHUNKS."
            )

        print(
            f"{split} cache loaded:"
        )

        print(
            f"  articles: "
            f"{len(self.uid_to_row):,}"
        )

        print(
            f"  shape   : "
            f"{self.chunk_ids.shape}"
        )


# ============================================================
# PRETOKENIZE HEADLINES
# ============================================================

def build_headline_token_map(
    pair_frames,
    title_map,
    tokenizer,
):

    print()
    print("=" * 80)
    print(
        "PRETOKENIZE HEADLINES"
    )
    print("=" * 80)

    unique_uids = set()

    for frame in pair_frames:

        unique_uids.update(
            frame[
                "headline_article_uid"
            ]
            .astype(str)
            .tolist()
        )

    print(
        f"Unique headline articles: "
        f"{len(unique_uids):,}"
    )

    token_map = {}

    for i, uid in enumerate(
        sorted(
            unique_uids
        ),
        start=1,
    ):

        if uid not in title_map:

            raise RuntimeError(
                f"Missing title for {uid}"
            )

        tokens = tokenizer.encode(
            str(
                title_map[
                    uid
                ]
            ),
            add_special_tokens=False,
            verbose=False,
        )

        token_map[
            uid
        ] = tokens

        if (
            i
            %
            50000
            ==
            0
        ):

            print(
                f"  tokenized "
                f"{i:,}/"
                f"{len(unique_uids):,}"
            )

    print(
        "Headline tokenization: PASSED"
    )

    return token_map


# ============================================================
# DATASET
# ============================================================

class MILPairDataset(
    Dataset
):

    def __init__(
        self,
        frame,
        cache,
        headline_token_map,
    ):

        self.pair_ids = (
            frame[
                "pair_id"
            ]
            .astype(str)
            .tolist()
        )

        self.anchor_uids = (
            frame[
                "anchor_article_uid"
            ]
            .astype(str)
            .tolist()
        )

        self.headline_uids = (
            frame[
                "headline_article_uid"
            ]
            .astype(str)
            .tolist()
        )

        self.labels = (
            frame[
                "label"
            ]
            .astype(
                np.int64
            )
            .tolist()
        )

        self.cache = cache

        self.headline_token_map = (
            headline_token_map
        )

        self.cache_rows = []

        for uid in (
            self.anchor_uids
        ):

            row_idx = (
                self.cache
                .uid_to_row
                .get(
                    uid
                )
            )

            if row_idx is None:

                raise RuntimeError(
                    f"Anchor {uid} "
                    "missing from chunk cache."
                )

            self.cache_rows.append(
                int(
                    row_idx
                )
            )

    def __len__(
        self,
    ):

        return len(
            self.labels
        )

    def __getitem__(
        self,
        index,
    ):

        row_idx = (
            self.cache_rows[
                index
            ]
        )

        headline_uid = (
            self.headline_uids[
                index
            ]
        )

        return {
            "pair_id":
                self.pair_ids[
                    index
                ],

            "label":
                self.labels[
                    index
                ],

            "headline_ids":
                self.headline_token_map[
                    headline_uid
                ],

            "cache_row":
                row_idx,
        }


# ============================================================
# MIL COLLATOR
# ============================================================

class MILCollator:

    def __init__(
        self,
        tokenizer,
        cache,
    ):

        self.cache = cache

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

        input_rows = []

        type_rows = []

        attention_rows = []

        bag_indices = []

        pair_ids = []

        labels = []

        chunk_counts = []

        # [CLS] headline [SEP] chunk [SEP]
        special_tokens = 3

        for bag_idx, item in enumerate(
            batch
        ):

            headline_ids = list(
                item[
                    "headline_ids"
                ]
            )

            max_headline = (
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
                max_headline
            ):

                headline_ids = (
                    headline_ids[
                        :max_headline
                    ]
                )

            remaining = (
                MAX_LENGTH
                -
                len(
                    headline_ids
                )
                -
                special_tokens
            )

            cache_row = int(
                item[
                    "cache_row"
                ]
            )

            count = int(
                self.cache
                .chunk_counts[
                    cache_row
                ]
            )

            if (
                count
                <
                1
                or
                count
                >
                CACHE_MAX_CHUNKS
            ):

                raise RuntimeError(
                    "Invalid chunk count."
                )

            pair_ids.append(
                item[
                    "pair_id"
                ]
            )

            labels.append(
                item[
                    "label"
                ]
            )

            chunk_counts.append(
                count
            )

            for chunk_index in range(
                count
            ):

                length = int(
                    self.cache
                    .chunk_lengths[
                        cache_row,
                        chunk_index,
                    ]
                )

                chunk_ids = (
                    self.cache
                    .chunk_ids[
                        cache_row,
                        chunk_index,
                        :length,
                    ]
                    .astype(
                        np.int64
                    )
                    .tolist()
                )

                chunk_ids = (
                    chunk_ids[
                        :remaining
                    ]
                )

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

                if (
                    len(
                        input_ids
                    )
                    >
                    MAX_LENGTH
                ):

                    raise RuntimeError(
                        "Input exceeds MAX_LENGTH."
                    )

                input_rows.append(
                    input_ids
                )

                type_rows.append(
                    token_type_ids
                )

                attention_rows.append(
                    attention_mask
                )

                bag_indices.append(
                    bag_idx
                )

        max_batch_length = max(
            len(
                row
            )
            for row
            in input_rows
        )

        for index in range(
            len(
                input_rows
            )
        ):

            pad = (
                max_batch_length
                -
                len(
                    input_rows[
                        index
                    ]
                )
            )

            if pad > 0:

                input_rows[
                    index
                ].extend(
                    [
                        self.pad_id
                    ]
                    *
                    pad
                )

                type_rows[
                    index
                ].extend(
                    [
                        0
                    ]
                    *
                    pad
                )

                attention_rows[
                    index
                ].extend(
                    [
                        0
                    ]
                    *
                    pad
                )

        return {
            "input_ids":
                torch.tensor(
                    input_rows,
                    dtype=torch.long,
                ),

            "token_type_ids":
                torch.tensor(
                    type_rows,
                    dtype=torch.long,
                ),

            "attention_mask":
                torch.tensor(
                    attention_rows,
                    dtype=torch.long,
                ),

            "bag_indices":
                torch.tensor(
                    bag_indices,
                    dtype=torch.long,
                ),

            "labels":
                torch.tensor(
                    labels,
                    dtype=torch.float32,
                ),

            "pair_ids":
                pair_ids,

            "chunk_counts":
                chunk_counts,
        }


# ============================================================
# MAX MIL
# ============================================================

def max_mil_forward(
    model,
    batch,
):
    """
    Returns one bag-level logit per headline/article pair.

    For each chunk:
        binary score = logit(CONSISTENT) - logit(INCONSISTENT)

    Bag:
        max(chunk binary score)

    sigmoid(bag_score) = P(CONSISTENT)
    """

    bag_indices = (
        batch[
            "bag_indices"
        ]
    )

    labels = (
        batch[
            "labels"
        ]
    )

    model_inputs = {
        "input_ids":
            batch[
                "input_ids"
            ],

        "token_type_ids":
            batch[
                "token_type_ids"
            ],

        "attention_mask":
            batch[
                "attention_mask"
            ],
    }

    output = model(
        **model_inputs
    )

    logits = (
        output.logits
    )

    if (
        logits.ndim
        !=
        2
        or
        logits.shape[
            1
        ]
        !=
        2
    ):

        raise RuntimeError(
            "Expected model logits "
            "with shape [N, 2]."
        )

    chunk_scores = (
        logits[
            :,
            1
        ]
        -
        logits[
            :,
            0
        ]
    )

    bag_scores = []

    bag_count = int(
        labels.shape[
            0
        ]
    )

    for bag_idx in range(
        bag_count
    ):

        mask = (
            bag_indices
            ==
            bag_idx
        )

        if not mask.any():

            raise RuntimeError(
                "MIL bag has no chunks."
            )

        bag_score = (
            chunk_scores[
                mask
            ]
            .max()
        )

        bag_scores.append(
            bag_score
        )

    bag_scores = torch.stack(
        bag_scores
    )

    return (
        bag_scores,
        labels,
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    use_amp,
):

    model.eval()

    loss_fn = (
        torch.nn
        .BCEWithLogitsLoss(
            reduction="sum"
        )
    )

    total_loss = 0.0

    total_examples = 0

    all_labels = []

    all_probabilities = []

    all_pair_ids = []

    for batch in loader:

        pair_ids = batch.pop(
            "pair_ids"
        )

        batch.pop(
            "chunk_counts"
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

            (
                bag_scores,
                labels,
            ) = max_mil_forward(
                model,
                batch,
            )

            loss = loss_fn(
                bag_scores,
                labels,
            )

        probabilities = (
            torch.sigmoid(
                bag_scores
            )
            .detach()
            .cpu()
            .numpy()
        )

        labels_np = (
            labels
            .detach()
            .cpu()
            .numpy()
        )

        total_loss += float(
            loss.item()
        )

        total_examples += int(
            len(
                labels_np
            )
        )

        all_probabilities.extend(
            probabilities.tolist()
        )

        all_labels.extend(
            labels_np.tolist()
        )

        all_pair_ids.extend(
            pair_ids
        )

    return {
        "loss":
            float(
                total_loss
                /
                total_examples
            ),

        "labels":
            np.asarray(
                all_labels,
                dtype=np.int64,
            ),

        "probabilities":
            np.asarray(
                all_probabilities,
                dtype=np.float32,
            ),

        "pair_ids":
            all_pair_ids,
    }


# ============================================================
# TRAINING
# ============================================================

def train_model(
    model,
    train_loader,
    validation_loader,
    device,
    args,
    best_model_path,
    history_path,
):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=(
            args.weight_decay
        ),
    )

    optimizer_steps_per_epoch = (
        math.ceil(
            len(
                train_loader
            )
            /
            args.grad_accum
        )
    )

    total_optimizer_steps = (
        optimizer_steps_per_epoch
        *
        args.epochs
    )

    warmup_steps = int(
        total_optimizer_steps
        *
        args.warmup_ratio
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=(
                warmup_steps
            ),
            num_training_steps=(
                total_optimizer_steps
            ),
        )
    )

    use_amp = (
        device.type
        ==
        "cuda"
        and
        not args.no_amp
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    loss_fn = (
        torch.nn
        .BCEWithLogitsLoss()
    )

    best_macro_f1 = (
        -1.0
    )

    history = []

    optimizer_step = 0

    optimizer.zero_grad(
        set_to_none=True
    )

    print()
    print("=" * 80)
    print(
        "TRAIN FULL-ARTICLE MAX-MIL"
    )
    print("=" * 80)

    print(
        f"Physical pair batch   : "
        f"{args.batch_size}"
    )

    print(
        f"Gradient accumulation : "
        f"{args.grad_accum}"
    )

    print(
        f"Effective pair batch  : "
        f"{args.batch_size * args.grad_accum}"
    )

    print(
        f"Train batches/epoch   : "
        f"{len(train_loader):,}"
    )

    print(
        f"Optimizer steps/epoch : "
        f"{optimizer_steps_per_epoch:,}"
    )

    print(
        f"Total optimizer steps : "
        f"{total_optimizer_steps:,}"
    )

    print(
        f"Warmup steps          : "
        f"{warmup_steps:,}"
    )

    print(
        f"Mixed precision       : "
        f"{use_amp}"
    )

    for epoch in range(
        1,
        args.epochs
        +
        1,
    ):

        start_time = time.time()

        model.train()

        running_loss = 0.0

        seen_pairs = 0

        for step, batch in enumerate(
            train_loader,
            start=1,
        ):

            batch.pop(
                "pair_ids"
            )

            batch.pop(
                "chunk_counts"
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

            pair_count = int(
                batch[
                    "labels"
                ].shape[
                    0
                ]
            )

            with torch.amp.autocast(
                device_type=(
                    device.type
                ),
                enabled=use_amp,
            ):

                (
                    bag_scores,
                    labels,
                ) = max_mil_forward(
                    model,
                    batch,
                )

                raw_loss = loss_fn(
                    bag_scores,
                    labels,
                )

                loss = (
                    raw_loss
                    /
                    args.grad_accum
                )

            scaler.scale(
                loss
            ).backward()

            running_loss += (
                float(
                    raw_loss.item()
                )
                *
                pair_count
            )

            seen_pairs += (
                pair_count
            )

            should_step = (
                step
                %
                args.grad_accum
                ==
                0
                or
                step
                ==
                len(
                    train_loader
                )
            )

            if should_step:

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.max_grad_norm,
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                optimizer_step += 1

            if (
                step
                %
                args.log_every
                ==
                0
            ):

                print(
                    f"Epoch "
                    f"{epoch}/"
                    f"{args.epochs} "
                    f"| batch "
                    f"{step:,}/"
                    f"{len(train_loader):,} "
                    f"| loss="
                    f"{running_loss / seen_pairs:.5f} "
                    f"| opt_step="
                    f"{optimizer_step:,}"
                )

        train_loss = (
            running_loss
            /
            seen_pairs
        )

        print()
        print(
            "Evaluating validation..."
        )

        validation = evaluate(
            model=model,
            loader=(
                validation_loader
            ),
            device=device,
            use_amp=use_amp,
        )

        metrics_05 = compute_metrics(
            labels=(
                validation[
                    "labels"
                ]
            ),
            probabilities=(
                validation[
                    "probabilities"
                ]
            ),
            threshold=0.5,
        )

        elapsed = (
            time.time()
            -
            start_time
        )

        history.append(
            {
                "epoch":
                    int(
                        epoch
                    ),

                "train_loss":
                    float(
                        train_loss
                    ),

                "validation_loss":
                    float(
                        validation[
                            "loss"
                        ]
                    ),

                "validation_accuracy_05":
                    float(
                        metrics_05[
                            "accuracy"
                        ]
                    ),

                "validation_macro_f1_05":
                    float(
                        metrics_05[
                            "macro_f1"
                        ]
                    ),

                "validation_f1_consistent_05":
                    float(
                        metrics_05[
                            "f1_consistent"
                        ]
                    ),

                "validation_roc_auc":
                    metrics_05[
                        "roc_auc"
                    ],

                "elapsed_seconds":
                    float(
                        elapsed
                    ),
            }
        )

        pd.DataFrame(
            history
        ).to_csv(
            history_path,
            index=False,
            encoding="utf-8-sig",
        )

        print()
        print(
            f"Epoch {epoch}:"
        )

        print(
            f"  train loss : "
            f"{train_loss:.6f}"
        )

        print(
            f"  val loss   : "
            f"{validation['loss']:.6f}"
        )

        print(
            f"  accuracy   : "
            f"{metrics_05['accuracy']:.6f}"
        )

        print(
            f"  Macro-F1   : "
            f"{metrics_05['macro_f1']:.6f}"
        )

        print(
            f"  ROC-AUC    : "
            f"{metrics_05['roc_auc']:.6f}"
        )

        print(
            f"  elapsed    : "
            f"{human_time(elapsed)}"
        )

        if (
            metrics_05[
                "macro_f1"
            ]
            >
            best_macro_f1
        ):

            best_macro_f1 = (
                metrics_05[
                    "macro_f1"
                ]
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "epoch":
                        int(
                            epoch
                        ),

                    "validation_macro_f1_05":
                        float(
                            best_macro_f1
                        ),

                    "model_name":
                        MODEL_NAME,

                    "architecture":
                        "max_mil",

                    "max_length":
                        MAX_LENGTH,

                    "cache_max_chunks":
                        CACHE_MAX_CHUNKS,
                },
                best_model_path,
            )

            print(
                "  → best checkpoint saved"
            )

        print()

    return history


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train full-document MAX-MIL "
            "headline consistency model."
        )
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=(
            DEFAULT_BATCH_SIZE
        ),
    )

    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=(
            DEFAULT_EVAL_BATCH_SIZE
        ),
    )

    parser.add_argument(
        "--grad-accum",
        type=int,
        default=(
            DEFAULT_GRAD_ACCUM
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=(
            DEFAULT_EPOCHS
        ),
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=(
            DEFAULT_LR
        ),
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=(
            DEFAULT_WEIGHT_DECAY
        ),
    )

    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=(
            DEFAULT_WARMUP_RATIO
        ),
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=(
            DEFAULT_MAX_GRAD_NORM
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=(
            DEFAULT_SEED
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=(
            DEFAULT_NUM_WORKERS
        ),
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
    )

    args = parser.parse_args()

    if args.batch_size < 1:

        raise ValueError(
            "--batch-size must be >= 1"
        )

    if args.eval_batch_size < 1:

        raise ValueError(
            "--eval-batch-size must be >= 1"
        )

    if args.grad_accum < 1:

        raise ValueError(
            "--grad-accum must be >= 1"
        )

    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    run_name = (
        "full_article_max_mil_mbert"
    )

    if args.smoke_test:

        run_name += "_smoke"

    MODEL_DIR = (
        MODULE_ROOT
        / "models"
        / "consistency"
        / run_name
    )

    RESULT_DIR = (
        MODULE_ROOT
        / "results"
        / "consistency"
        / run_name
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    BEST_MODEL_PATH = (
        MODEL_DIR
        / "best_model.pt"
    )

    HISTORY_PATH = (
        RESULT_DIR
        / "training_history.csv"
    )

    METRICS_PATH = (
        RESULT_DIR
        / "validation_metrics.json"
    )

    PREDICTION_PATH = (
        RESULT_DIR
        / "validation_predictions.parquet"
    )

    THRESHOLD_PATH = (
        RESULT_DIR
        / "threshold_search.csv"
    )

    CONFIG_PATH = (
        RESULT_DIR
        / "run_config.json"
    )

    # ========================================================
    # HEADER
    # ========================================================

    print("=" * 80)
    print(
        "MODULE 2 - HEADLINE INTEGRITY"
    )
    print(
        "TASK 2B - HEADLINE CONSISTENCY"
    )
    print(
        "STEP 18 - FULL ARTICLE MAX-MIL"
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
        f"Model initialization : "
        f"Baseline B checkpoint"
    )

    print(
        f"Pooling              : MAX"
    )

    print(
        f"Max chunks/article   : "
        f"{CACHE_MAX_CHUNKS}"
    )

    print(
        f"Max input length     : "
        f"{MAX_LENGTH}"
    )

    print(
        f"Pair batch size      : "
        f"{args.batch_size}"
    )

    print(
        f"Eval pair batch      : "
        f"{args.eval_batch_size}"
    )

    print(
        f"Grad accumulation    : "
        f"{args.grad_accum}"
    )

    print(
        f"Effective pair batch : "
        f"{args.batch_size * args.grad_accum}"
    )

    print(
        f"Epochs               : "
        f"{args.epochs}"
    )

    print(
        f"Learning rate        : "
        f"{args.lr}"
    )

    print(
        f"Smoke test           : "
        f"{args.smoke_test}"
    )

    # ========================================================
    # REPRODUCIBILITY
    # ========================================================

    set_seed(
        args.seed
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
    # REQUIRED FILES
    # ========================================================

    required = [
        ARTICLE_INDEX_PATH,
        PAIR_FILES[
            "train"
        ],
        PAIR_FILES[
            "validation"
        ],
        B_MODEL_PATH,
    ]

    for path in required:

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

    # ========================================================
    # TITLES + PAIRS
    # ========================================================

    (
        title_map,
        split_map,
    ) = load_title_data()

    train_frame = load_pairs(
        "train",
        split_map,
    )

    validation_frame = load_pairs(
        "validation",
        split_map,
    )

    # ========================================================
    # SMOKE TEST
    # ========================================================

    if args.smoke_test:

        # Each anchor has 2 pairs.
        train_frame = anchor_subset(
            frame=train_frame,
            max_anchors=1000,
            seed=args.seed,
        )

        validation_frame = anchor_subset(
            frame=validation_frame,
            max_anchors=500,
            seed=(
                args.seed
                +
                100
            ),
        )

        print()
        print(
            "SMOKE TEST SUBSET:"
        )

        print(
            f"  train pairs      : "
            f"{len(train_frame):,}"
        )

        print(
            f"  train anchors    : "
            f"{train_frame['anchor_article_uid'].nunique():,}"
        )

        print(
            f"  validation pairs : "
            f"{len(validation_frame):,}"
        )

        print(
            f"  validation anchors: "
            f"{validation_frame['anchor_article_uid'].nunique():,}"
        )

    # ========================================================
    # CHUNK CACHE
    # ========================================================

    print()
    print("=" * 80)
    print(
        "LOAD CHUNK CACHE"
    )
    print("=" * 80)

    train_cache = ChunkCache(
        "train"
    )

    validation_cache = ChunkCache(
        "validation"
    )

    # ========================================================
    # PRETOKENIZE HEADLINES
    # ========================================================

    headline_token_map = (
        build_headline_token_map(
            pair_frames=[
                train_frame,
                validation_frame,
            ],
            title_map=title_map,
            tokenizer=tokenizer,
        )
    )

    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = MILPairDataset(
        frame=train_frame,
        cache=train_cache,
        headline_token_map=(
            headline_token_map
        ),
    )

    validation_dataset = MILPairDataset(
        frame=validation_frame,
        cache=validation_cache,
        headline_token_map=(
            headline_token_map
        ),
    )

    train_collator = MILCollator(
        tokenizer=tokenizer,
        cache=train_cache,
    )

    validation_collator = MILCollator(
        tokenizer=tokenizer,
        cache=validation_cache,
    )

    generator = torch.Generator()

    generator.manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=(
            args.batch_size
        ),
        shuffle=True,
        collate_fn=(
            train_collator
        ),
        num_workers=(
            args.num_workers
        ),
        pin_memory=(
            device.type
            ==
            "cuda"
        ),
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=(
            args.eval_batch_size
        ),
        shuffle=False,
        collate_fn=(
            validation_collator
        ),
        num_workers=(
            args.num_workers
        ),
        pin_memory=(
            device.type
            ==
            "cuda"
        ),
    )

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print("=" * 80)
    print(
        "LOAD BASELINE B CHECKPOINT"
    )
    print("=" * 80)

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

    print(
        f"Baseline B epoch: "
        f"{checkpoint.get('epoch')}"
    )

    print(
        "Checkpoint load: PASSED"
    )

    if args.gradient_checkpointing:

        model.gradient_checkpointing_enable()

        if hasattr(
            model.config,
            "use_cache",
        ):

            model.config.use_cache = False

        print(
            "Gradient checkpointing: ENABLED"
        )

    model.to(
        device
    )

    # ========================================================
    # SAVE CONFIG
    # ========================================================

    save_json(
        {
            "stage":
                "18_train_full_article_max_mil",

            "dataset":
                "headline_consistency_v1",

            "model":
                MODEL_NAME,

            "initialization":
                (
                    "baseline_title_article_head_mbert/"
                    "best_model.pt"
                ),

            "pooling":
                "max",

            "loss":
                "BCEWithLogitsLoss",

            "chunk_binary_score":
                (
                    "logit_consistent "
                    "- logit_inconsistent"
                ),

            "train_pairs":
                int(
                    len(
                        train_frame
                    )
                ),

            "validation_pairs":
                int(
                    len(
                        validation_frame
                    )
                ),

            "test_loaded":
                False,

            "max_length":
                MAX_LENGTH,

            "max_chunks":
                CACHE_MAX_CHUNKS,

            "pair_batch_size":
                args.batch_size,

            "eval_pair_batch_size":
                args.eval_batch_size,

            "grad_accum":
                args.grad_accum,

            "effective_pair_batch":
                (
                    args.batch_size
                    *
                    args.grad_accum
                ),

            "epochs":
                args.epochs,

            "learning_rate":
                args.lr,

            "weight_decay":
                args.weight_decay,

            "warmup_ratio":
                args.warmup_ratio,

            "seed":
                args.seed,

            "amp":
                (
                    device.type
                    ==
                    "cuda"
                    and
                    not args.no_amp
                ),

            "gradient_checkpointing":
                args.gradient_checkpointing,

            "smoke_test":
                args.smoke_test,
        },
        CONFIG_PATH,
    )

    # ========================================================
    # TRAIN
    # ========================================================

    train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=(
            validation_loader
        ),
        device=device,
        args=args,
        best_model_path=(
            BEST_MODEL_PATH
        ),
        history_path=(
            HISTORY_PATH
        ),
    )

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print()
    print("=" * 80)
    print(
        "LOAD BEST MIL CHECKPOINT"
    )
    print("=" * 80)

    best_checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(
        best_checkpoint[
            "model_state_dict"
        ]
    )

    print(
        f"Best epoch: "
        f"{best_checkpoint['epoch']}"
    )

    print(
        "Validation Macro-F1 @0.5: "
        f"{best_checkpoint['validation_macro_f1_05']:.6f}"
    )

    # ========================================================
    # FINAL VALIDATION INFERENCE
    # ========================================================

    use_amp = (
        device.type
        ==
        "cuda"
        and
        not args.no_amp
    )

    print()
    print(
        "Running final validation "
        "inference..."
    )

    validation = evaluate(
        model=model,
        loader=validation_loader,
        device=device,
        use_amp=use_amp,
    )

    metrics_05 = compute_metrics(
        labels=(
            validation[
                "labels"
            ]
        ),
        probabilities=(
            validation[
                "probabilities"
            ]
        ),
        threshold=0.5,
    )

    (
        tuned_metrics,
        threshold_frame,
    ) = tune_threshold(
        labels=(
            validation[
                "labels"
            ]
        ),
        probabilities=(
            validation[
                "probabilities"
            ]
        ),
    )

    threshold_frame.to_csv(
        THRESHOLD_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # SAVE VALIDATION PREDICTIONS
    # ========================================================

    predictions = pd.DataFrame(
        {
            "pair_id":
                validation[
                    "pair_ids"
                ],

            "label":
                validation[
                    "labels"
                ],

            "prob_consistent":
                validation[
                    "probabilities"
                ],
        }
    )

    predictions[
        "prediction_05"
    ] = (
        predictions[
            "prob_consistent"
        ]
        >=
        0.5
    ).astype(
        np.int8
    )

    tuned_threshold = (
        tuned_metrics[
            "threshold"
        ]
    )

    predictions[
        "prediction_tuned"
    ] = (
        predictions[
            "prob_consistent"
        ]
        >=
        tuned_threshold
    ).astype(
        np.int8
    )

    predictions.to_parquet(
        PREDICTION_PATH,
        index=False,
    )

    # ========================================================
    # METRICS REPORT
    # ========================================================

    save_json(
        {
            "stage":
                "18_train_full_article_max_mil",

            "dataset":
                "headline_consistency_v1",

            "model":
                MODEL_NAME,

            "architecture":
                "full_article_max_mil",

            "test_used":
                False,

            "validation_loss":
                validation[
                    "loss"
                ],

            "metrics_at_0_5":
                metrics_05,

            "best_validation_threshold":
                tuned_threshold,

            "metrics_at_tuned_threshold":
                tuned_metrics,

            "note":
                (
                    "Threshold tuned on validation only. "
                    "Test remains untouched."
                ),
        },
        METRICS_PATH,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print(
        "STEP 18 VALIDATION RESULTS"
    )
    print("=" * 80)

    print()
    print(
        "Threshold = 0.50"
    )

    print(
        f"  Accuracy : "
        f"{metrics_05['accuracy']:.6f}"
    )

    print(
        f"  Precision: "
        f"{metrics_05['precision_consistent']:.6f}"
    )

    print(
        f"  Recall   : "
        f"{metrics_05['recall_consistent']:.6f}"
    )

    print(
        f"  F1       : "
        f"{metrics_05['f1_consistent']:.6f}"
    )

    print(
        f"  Macro-F1 : "
        f"{metrics_05['macro_f1']:.6f}"
    )

    print(
        f"  ROC-AUC  : "
        f"{metrics_05['roc_auc']:.6f}"
    )

    cm = (
        metrics_05[
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
        "Tuned validation threshold:"
    )

    print(
        f"  threshold: "
        f"{tuned_threshold:.2f}"
    )

    print(
        f"  Accuracy : "
        f"{tuned_metrics['accuracy']:.6f}"
    )

    print(
        f"  Macro-F1 : "
        f"{tuned_metrics['macro_f1']:.6f}"
    )

    print(
        f"  F1       : "
        f"{tuned_metrics['f1_consistent']:.6f}"
    )

    tuned_cm = (
        tuned_metrics[
            "confusion_matrix"
        ]
    )

    print(
        f"  TN={tuned_cm['tn']:,} "
        f"FP={tuned_cm['fp']:,}"
    )

    print(
        f"  FN={tuned_cm['fn']:,} "
        f"TP={tuned_cm['tp']:,}"
    )

    print()
    print(
        "TEST WAS NOT USED."
    )

    print()
    print(
        f"Best model:\n"
        f"{BEST_MODEL_PATH}"
    )

    print()
    print(
        f"Results:\n"
        f"{RESULT_DIR}"
    )

    print()
    print(
        "STEP 18 FINISHED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()