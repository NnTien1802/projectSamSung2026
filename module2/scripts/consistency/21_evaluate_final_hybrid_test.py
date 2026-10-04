from pathlib import Path
import json
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

ARTICLE_INDEX_PATH = (
    MODULE_ROOT
    / "data"
    / "interim"
    / "consistency"
    / "articles_v1"
    / "article_index.parquet"
)

CACHE_DIR = (
    MODULE_ROOT
    / "data"
    / "interim"
    / "consistency"
    / "mil_chunk_cache_mbert_v1"
)

DESCRIPTION_MODEL_PATH = (
    MODULE_ROOT
    / "models"
    / "consistency"
    / "baseline_title_description_mbert"
    / "best_model.pt"
)

MIL_MODEL_PATH = (
    MODULE_ROOT
    / "models"
    / "consistency"
    / "full_article_max_mil_mbert"
    / "best_model.pt"
)

OUTPUT_DIR = (
    MODULE_ROOT
    / "results"
    / "consistency"
    / "final_hybrid_v1"
)

LOCK_PATH = (
    OUTPUT_DIR
    / "final_model_lock.json"
)

PREDICTION_PATH = (
    OUTPUT_DIR
    / "official_test_predictions.parquet"
)

METRICS_PATH = (
    OUTPUT_DIR
    / "official_test_metrics.json"
)

CATEGORY_PATH = (
    OUTPUT_DIR
    / "official_test_by_category.csv"
)

TIER_PATH = (
    OUTPUT_DIR
    / "official_test_negative_by_tier.csv"
)


# ============================================================
# FROZEN MODEL CONFIG
# ============================================================

MODEL_NAME = (
    "bert-base-multilingual-cased"
)

MAX_LENGTH = 160

MAX_CHUNKS = 6

DESCRIPTION_BATCH_SIZE = 32

MIL_BATCH_SIZE = 4


# ============================================================
# HELPERS
# ============================================================

def load_json(path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


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

    seconds = int(seconds)

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
            float(
                roc_auc_score(
                    labels,
                    probabilities,
                )
            ),

        "average_precision":
            float(
                average_precision_score(
                    labels,
                    probabilities,
                )
            ),

        "confusion_matrix": {
            "tn":
                int(tn),

            "fp":
                int(fp),

            "fn":
                int(fn),

            "tp":
                int(tp),
        },
    }


# ============================================================
# LOAD FROZEN CONFIG
# ============================================================

def load_and_validate_lock():

    if not LOCK_PATH.exists():

        raise FileNotFoundError(
            LOCK_PATH
        )

    lock = load_json(
        LOCK_PATH
    )

    required = [
        "description_weight",
        "mil_weight",
        "decision_threshold",
    ]

    for key in required:

        if key not in lock:

            raise RuntimeError(
                f"Final lock is missing: {key}"
            )

    description_weight = float(
        lock[
            "description_weight"
        ]
    )

    mil_weight = float(
        lock[
            "mil_weight"
        ]
    )

    threshold = float(
        lock[
            "decision_threshold"
        ]
    )

    # --------------------------------------------------------
    # These values were frozen by Step 20.
    # Abort rather than silently evaluate a different model.
    # --------------------------------------------------------

    if not np.isclose(
        description_weight,
        0.50,
    ):

        raise RuntimeError(
            "Unexpected frozen "
            "description weight."
        )

    if not np.isclose(
        mil_weight,
        0.50,
    ):

        raise RuntimeError(
            "Unexpected frozen MIL weight."
        )

    if not np.isclose(
        threshold,
        0.45,
    ):

        raise RuntimeError(
            "Unexpected frozen threshold."
        )

    if not np.isclose(
        description_weight
        +
        mil_weight,
        1.0,
    ):

        raise RuntimeError(
            "Fusion weights do not sum to 1."
        )

    return (
        lock,
        description_weight,
        mil_weight,
        threshold,
    )


# ============================================================
# DESCRIPTION DATASET
# ============================================================

class DescriptionDataset(
    Dataset
):

    def __init__(
        self,
        frame,
        title_map,
        description_map,
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

        self.title_map = (
            title_map
        )

        self.description_map = (
            description_map
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

        anchor_uid = (
            self.anchor_uids[
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

            "headline":
                self.title_map[
                    headline_uid
                ],

            "description":
                self.description_map[
                    anchor_uid
                ],

            "label":
                self.labels[
                    index
                ],
        }


class DescriptionCollator:

    def __init__(
        self,
        tokenizer,
    ):

        self.tokenizer = tokenizer

    def __call__(
        self,
        batch,
    ):

        headlines = [
            item[
                "headline"
            ]
            for item in batch
        ]

        descriptions = [
            item[
                "description"
            ]
            for item in batch
        ]

        pair_ids = [
            item[
                "pair_id"
            ]
            for item in batch
        ]

        labels = torch.tensor(
            [
                item[
                    "label"
                ]
                for item in batch
            ],
            dtype=torch.long,
        )

        encoded = (
            self.tokenizer(
                headlines,
                descriptions,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
        )

        encoded[
            "pair_ids"
        ] = pair_ids

        encoded[
            "labels"
        ] = labels

        return encoded


# ============================================================
# DESCRIPTION INFERENCE
# ============================================================

@torch.no_grad()
def predict_description(
    model,
    loader,
    device,
):

    model.eval()

    pair_ids = []
    labels = []
    probabilities = []

    use_amp = (
        device.type
        ==
        "cuda"
    )

    total_batches = len(
        loader
    )

    for batch_index, batch in enumerate(
        loader,
        start=1,
    ):

        batch_pair_ids = (
            batch.pop(
                "pair_ids"
            )
        )

        batch_labels = (
            batch.pop(
                "labels"
            )
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

        probs = (
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

        pair_ids.extend(
            batch_pair_ids
        )

        labels.extend(
            batch_labels
            .numpy()
            .tolist()
        )

        probabilities.extend(
            probs.tolist()
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
                f"  description batch "
                f"{batch_index:,}/"
                f"{total_batches:,}"
            )

    return pd.DataFrame(
        {
            "pair_id":
                pair_ids,

            "label_description":
                np.asarray(
                    labels,
                    dtype=np.int64,
                ),

            "prob_description":
                np.asarray(
                    probabilities,
                    dtype=np.float32,
                ),
        }
    )


# ============================================================
# MIL CACHE
# ============================================================

class TestChunkCache:

    def __init__(
        self,
    ):

        index_path = (
            CACHE_DIR
            / "test_article_index.parquet"
        )

        chunk_ids_path = (
            CACHE_DIR
            / "test_chunk_ids.npy"
        )

        chunk_lengths_path = (
            CACHE_DIR
            / "test_chunk_lengths.npy"
        )

        chunk_counts_path = (
            CACHE_DIR
            / "test_chunk_counts.npy"
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
                "Duplicate TEST cache UID."
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
            MAX_CHUNKS
        ):

            raise RuntimeError(
                "Unexpected TEST cache "
                "MAX_CHUNKS."
            )

        print(
            f"TEST cache articles: "
            f"{len(self.uid_to_row):,}"
        )

        print(
            f"TEST cache shape   : "
            f"{self.chunk_ids.shape}"
        )


# ============================================================
# HEADLINE TOKEN MAP FOR MIL
# ============================================================

def build_headline_token_map(
    frame,
    title_map,
    tokenizer,
):

    unique_uids = (
        frame[
            "headline_article_uid"
        ]
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    print(
        f"Unique TEST headline articles: "
        f"{len(unique_uids):,}"
    )

    result = {}

    for i, uid in enumerate(
        unique_uids,
        start=1,
    ):

        result[
            uid
        ] = tokenizer.encode(
            str(
                title_map[
                    uid
                ]
            ),
            add_special_tokens=False,
            verbose=False,
        )

        if (
            i
            %
            10000
            ==
            0
        ):

            print(
                f"  tokenized headlines "
                f"{i:,}/"
                f"{len(unique_uids):,}"
            )

    return result


# ============================================================
# MIL DATASET
# ============================================================

class MILTestDataset(
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

        for uid in self.anchor_uids:

            row = (
                cache
                .uid_to_row
                .get(
                    uid
                )
            )

            if row is None:

                raise RuntimeError(
                    f"TEST anchor missing "
                    f"from cache: {uid}"
                )

            self.cache_rows.append(
                int(row)
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
                    self.headline_uids[
                        index
                    ]
                ],

            "cache_row":
                self.cache_rows[
                    index
                ],
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
                len(headline_ids)
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
                len(headline_ids)
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
                count < 1
                or
                count > MAX_CHUNKS
            ):

                raise RuntimeError(
                    "Invalid TEST chunk count."
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
                    [self.cls_id]
                    +
                    headline_ids
                    +
                    [self.sep_id]
                    +
                    chunk_ids
                    +
                    [self.sep_id]
                )

                token_type_ids = (
                    [0]
                    *
                    (
                        len(headline_ids)
                        +
                        2
                    )
                    +
                    [1]
                    *
                    (
                        len(chunk_ids)
                        +
                        1
                    )
                )

                attention_mask = (
                    [1]
                    *
                    len(input_ids)
                )

                if (
                    len(input_ids)
                    >
                    MAX_LENGTH
                ):

                    raise RuntimeError(
                        "TEST MIL input exceeds "
                        "MAX_LENGTH."
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
            len(row)
            for row in input_rows
        )

        for i in range(
            len(input_rows)
        ):

            pad = (
                max_batch_length
                -
                len(
                    input_rows[
                        i
                    ]
                )
            )

            if pad > 0:

                input_rows[
                    i
                ].extend(
                    [self.pad_id]
                    *
                    pad
                )

                type_rows[
                    i
                ].extend(
                    [0]
                    *
                    pad
                )

                attention_rows[
                    i
                ].extend(
                    [0]
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
                    dtype=torch.long,
                ),

            "pair_ids":
                pair_ids,
        }


# ============================================================
# MIL FORWARD
# ============================================================

def max_mil_probabilities(
    model,
    batch,
):

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

    output = model(
        input_ids=(
            batch[
                "input_ids"
            ]
        ),
        token_type_ids=(
            batch[
                "token_type_ids"
            ]
        ),
        attention_mask=(
            batch[
                "attention_mask"
            ]
        ),
    )

    logits = (
        output.logits
    )

    # Same binary score definition as Step 18.
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

    for bag_idx in range(
        labels.shape[
            0
        ]
    ):

        mask = (
            bag_indices
            ==
            bag_idx
        )

        if not mask.any():

            raise RuntimeError(
                "TEST MIL bag has no chunks."
            )

        bag_scores.append(
            chunk_scores[
                mask
            ].max()
        )

    bag_scores = torch.stack(
        bag_scores
    )

    probabilities = torch.sigmoid(
        bag_scores
    )

    return probabilities


# ============================================================
# MIL INFERENCE
# ============================================================

@torch.no_grad()
def predict_mil(
    model,
    loader,
    device,
):

    model.eval()

    pair_ids = []
    labels = []
    probabilities = []

    use_amp = (
        device.type
        ==
        "cuda"
    )

    total_batches = len(
        loader
    )

    for batch_index, batch in enumerate(
        loader,
        start=1,
    ):

        batch_pair_ids = (
            batch.pop(
                "pair_ids"
            )
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

        labels_tensor = (
            batch[
                "labels"
            ]
        )

        with torch.amp.autocast(
            device_type=(
                device.type
            ),
            enabled=use_amp,
        ):

            probs = (
                max_mil_probabilities(
                    model,
                    batch,
                )
            )

        pair_ids.extend(
            batch_pair_ids
        )

        labels.extend(
            labels_tensor
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        probabilities.extend(
            probs
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        if (
            batch_index
            %
            1000
            ==
            0
            or
            batch_index
            ==
            total_batches
        ):

            print(
                f"  MIL batch "
                f"{batch_index:,}/"
                f"{total_batches:,}"
            )

    return pd.DataFrame(
        {
            "pair_id":
                pair_ids,

            "label_mil":
                np.asarray(
                    labels,
                    dtype=np.int64,
                ),

            "prob_mil":
                np.asarray(
                    probabilities,
                    dtype=np.float32,
                ),
        }
    )


# ============================================================
# LOAD MODEL CHECKPOINT
# ============================================================

def load_model(
    checkpoint_path,
    device,
):

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=2,
        )
    )

    checkpoint = torch.load(
        checkpoint_path,
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

    return model


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
        "STEP 21 - OFFICIAL FINAL TEST EVALUATION"
    )
    print("=" * 80)

    print()
    print(
        "THIS IS THE OFFICIAL TEST RUN."
    )

    print(
        "NO WEIGHT OR THRESHOLD SEARCH "
        "WILL BE PERFORMED."
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
        ARTICLE_PATH,
        ARTICLE_INDEX_PATH,
        DESCRIPTION_MODEL_PATH,
        MIL_MODEL_PATH,
        LOCK_PATH,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

    # ========================================================
    # FROZEN CONFIG
    # ========================================================

    (
        lock,
        description_weight,
        mil_weight,
        final_threshold,
    ) = load_and_validate_lock()

    print()
    print(
        "FROZEN CONFIG:"
    )

    print(
        f"  Description weight : "
        f"{description_weight:.2f}"
    )

    print(
        f"  MIL weight         : "
        f"{mil_weight:.2f}"
    )

    print(
        f"  Decision threshold : "
        f"{final_threshold:.2f}"
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
        f"Device: {device}"
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

    article_index[
        "article_uid"
    ] = (
        article_index[
            "article_uid"
        ]
        .astype(str)
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
    # TEST PAIRS
    # ========================================================

    pairs = (
        pq.read_table(
            PAIR_PATH,
            columns=[
                "pair_id",
                "anchor_article_uid",
                "headline_article_uid",
                "anchor_category",
                "selection_tier",
                "lexical_similarity",
                "candidate_rank",
                "label",
            ],
        )
        .to_pandas()
    )

    pairs[
        "anchor_article_uid"
    ] = (
        pairs[
            "anchor_article_uid"
        ]
        .astype(str)
    )

    pairs[
        "headline_article_uid"
    ] = (
        pairs[
            "headline_article_uid"
        ]
        .astype(str)
    )

    if (
        pairs[
            "pair_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate TEST pair_id."
        )

    if not (
        pairs[
            "anchor_article_uid"
        ]
        .map(
            split_map
        )
        ==
        "test"
    ).all():

        raise RuntimeError(
            "TEST anchor split mismatch."
        )

    if not (
        pairs[
            "headline_article_uid"
        ]
        .map(
            split_map
        )
        ==
        "test"
    ).all():

        raise RuntimeError(
            "TEST headline split mismatch."
        )

    print()
    print(
        f"TEST pairs   : "
        f"{len(pairs):,}"
    )

    print(
        f"TEST anchors : "
        f"{pairs['anchor_article_uid'].nunique():,}"
    )

    counts = (
        pairs[
            "label"
        ]
        .value_counts()
    )

    print(
        f"Label 0      : "
        f"{int(counts.get(0, 0)):,}"
    )

    print(
        f"Label 1      : "
        f"{int(counts.get(1, 0)):,}"
    )

    # ========================================================
    # TEST DESCRIPTION MAP
    # ========================================================

    article_frame = (
        pq.read_table(
            ARTICLE_PATH,
            columns=[
                "article_uid",
                "description",
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
            "Duplicate TEST article_uid."
        )

    description_map = dict(
        zip(
            article_frame[
                "article_uid"
            ],
            article_frame[
                "description"
            ],
        )
    )

    missing_description = (
        set(
            pairs[
                "anchor_article_uid"
            ]
        )
        -
        set(
            description_map
        )
    )

    if missing_description:

        raise RuntimeError(
            f"Missing TEST descriptions: "
            f"{len(missing_description):,}"
        )

    # ========================================================
    # DESCRIPTION BRANCH
    # ========================================================

    print()
    print("=" * 80)
    print(
        "TEST INFERENCE - DESCRIPTION BRANCH"
    )
    print("=" * 80)

    description_dataset = (
        DescriptionDataset(
            frame=pairs,
            title_map=title_map,
            description_map=(
                description_map
            ),
        )
    )

    description_loader = DataLoader(
        description_dataset,
        batch_size=(
            DESCRIPTION_BATCH_SIZE
        ),
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type
            ==
            "cuda"
        ),
        collate_fn=(
            DescriptionCollator(
                tokenizer
            )
        ),
    )

    start = time.time()

    description_model = load_model(
        DESCRIPTION_MODEL_PATH,
        device,
    )

    description_predictions = (
        predict_description(
            model=description_model,
            loader=description_loader,
            device=device,
        )
    )

    print(
        f"Description inference time: "
        f"{human_time(time.time() - start)}"
    )

    # Free GPU before MIL model.
    del description_model

    if (
        device.type
        ==
        "cuda"
    ):

        torch.cuda.empty_cache()

    # ========================================================
    # MIL BRANCH
    # ========================================================

    print()
    print("=" * 80)
    print(
        "TEST INFERENCE - FULL ARTICLE MAX-MIL"
    )
    print("=" * 80)

    cache = TestChunkCache()

    headline_token_map = (
        build_headline_token_map(
            frame=pairs,
            title_map=title_map,
            tokenizer=tokenizer,
        )
    )

    mil_dataset = MILTestDataset(
        frame=pairs,
        cache=cache,
        headline_token_map=(
            headline_token_map
        ),
    )

    mil_loader = DataLoader(
        mil_dataset,
        batch_size=(
            MIL_BATCH_SIZE
        ),
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type
            ==
            "cuda"
        ),
        collate_fn=(
            MILCollator(
                tokenizer=tokenizer,
                cache=cache,
            )
        ),
    )

    start = time.time()

    mil_model = load_model(
        MIL_MODEL_PATH,
        device,
    )

    mil_predictions = (
        predict_mil(
            model=mil_model,
            loader=mil_loader,
            device=device,
        )
    )

    print(
        f"MIL inference time: "
        f"{human_time(time.time() - start)}"
    )

    # ========================================================
    # JOIN BRANCH OUTPUTS
    # ========================================================

    result = (
        pairs
        .merge(
            description_predictions,
            on="pair_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            mil_predictions,
            on="pair_id",
            how="inner",
            validate="one_to_one",
        )
    )

    if (
        len(result)
        !=
        len(pairs)
    ):

        raise RuntimeError(
            "Official TEST prediction "
            "join lost rows."
        )

    if not (
        result[
            "label"
        ].to_numpy()
        ==
        result[
            "label_description"
        ].to_numpy()
    ).all():

        raise RuntimeError(
            "Description label mismatch."
        )

    if not (
        result[
            "label"
        ].to_numpy()
        ==
        result[
            "label_mil"
        ].to_numpy()
    ).all():

        raise RuntimeError(
            "MIL label mismatch."
        )

    # ========================================================
    # FROZEN FUSION
    # ========================================================

    result[
        "prob_final"
    ] = (
        description_weight
        *
        result[
            "prob_description"
        ]
        +
        mil_weight
        *
        result[
            "prob_mil"
        ]
    )

    result[
        "prediction_final"
    ] = (
        result[
            "prob_final"
        ]
        >=
        final_threshold
    ).astype(
        np.int8
    )

    # Branch predictions shown for analysis only.
    # These are NOT used to tune final config.

    result[
        "prediction_description_045"
    ] = (
        result[
            "prob_description"
        ]
        >=
        final_threshold
    ).astype(
        np.int8
    )

    result[
        "prediction_mil_045"
    ] = (
        result[
            "prob_mil"
        ]
        >=
        final_threshold
    ).astype(
        np.int8
    )

    # ========================================================
    # OFFICIAL FINAL METRICS
    # ========================================================

    labels = (
        result[
            "label"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    final_metrics = compute_metrics(
        labels=labels,
        probabilities=(
            result[
                "prob_final"
            ]
        ),
        threshold=(
            final_threshold
        ),
    )

    # For reference only.
    # Fixed threshold, no test search.
    description_reference = (
        compute_metrics(
            labels=labels,
            probabilities=(
                result[
                    "prob_description"
                ]
            ),
            threshold=(
                final_threshold
            ),
        )
    )

    mil_reference = (
        compute_metrics(
            labels=labels,
            probabilities=(
                result[
                    "prob_mil"
                ]
            ),
            threshold=(
                final_threshold
            ),
        )
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    save_columns = [
        "pair_id",
        "anchor_article_uid",
        "headline_article_uid",
        "anchor_category",
        "selection_tier",
        "lexical_similarity",
        "candidate_rank",
        "label",
        "prob_description",
        "prob_mil",
        "prob_final",
        "prediction_final",
    ]

    result[
        save_columns
    ].to_parquet(
        PREDICTION_PATH,
        index=False,
    )

    # ========================================================
    # CATEGORY ANALYSIS
    # ========================================================

    category_rows = []

    for category, group in (
        result.groupby(
            "anchor_category"
        )
    ):

        category_metrics = (
            compute_metrics(
                labels=(
                    group[
                        "label"
                    ]
                ),
                probabilities=(
                    group[
                        "prob_final"
                    ]
                ),
                threshold=(
                    final_threshold
                ),
            )
        )

        category_rows.append(
            {
                "category":
                    category,

                "n":
                    int(
                        len(group)
                    ),

                "accuracy":
                    category_metrics[
                        "accuracy"
                    ],

                "macro_f1":
                    category_metrics[
                        "macro_f1"
                    ],

                "f1_consistent":
                    category_metrics[
                        "f1_consistent"
                    ],
            }
        )

    category_df = pd.DataFrame(
        category_rows
    )

    category_df.sort_values(
        "macro_f1",
        ascending=True,
        inplace=True,
    )

    category_df.to_csv(
        CATEGORY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # NEGATIVE TIER ANALYSIS
    # ========================================================

    negative = (
        result[
            result[
                "label"
            ]
            ==
            0
        ]
        .copy()
    )

    tier_rows = []

    for tier, group in (
        negative.groupby(
            "selection_tier"
        )
    ):

        predictions = (
            group[
                "prediction_final"
            ]
            .to_numpy()
        )

        negative_recall = float(
            (
                predictions
                ==
                0
            ).mean()
        )

        tier_rows.append(
            {
                "tier":
                    tier,

                "n":
                    int(
                        len(group)
                    ),

                "negative_recall":
                    negative_recall,

                "false_positives":
                    int(
                        (
                            predictions
                            ==
                            1
                        ).sum()
                    ),
            }
        )

    tier_df = pd.DataFrame(
        tier_rows
    )

    tier_df.to_csv(
        TIER_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # SAVE OFFICIAL REPORT
    # ========================================================

    report = {
        "stage":
            "21_official_final_test_evaluation",

        "dataset":
            "headline_consistency_v1",

        "split":
            "test",

        "evaluation_status":
            "OFFICIAL_TEST_EVALUATED",

        "configuration_frozen_before_test":
            True,

        "test_tuning_performed":
            False,

        "frozen_configuration": {
            "description_weight":
                description_weight,

            "mil_weight":
                mil_weight,

            "decision_threshold":
                final_threshold,

            "description_model":
                (
                    "baseline_title_description_mbert"
                ),

            "mil_model":
                (
                    "full_article_max_mil_mbert"
                ),
        },

        "test_pairs":
            int(
                len(result)
            ),

        "test_anchors":
            int(
                result[
                    "anchor_article_uid"
                ].nunique()
            ),

        "official_final_metrics":
            final_metrics,

        "branch_reference_metrics_fixed_at_final_threshold": {
            "description":
                description_reference,

            "mil":
                mil_reference,
        },

        "methodological_warning":
            (
                "This test set uses presumed-consistent "
                "original positives and synthetic hard "
                "negatives. It is not a human-gold "
                "real-world consistency benchmark."
            ),
    }

    save_json(
        report,
        METRICS_PATH,
    )

    # ========================================================
    # PRINT OFFICIAL RESULT
    # ========================================================

    print()
    print("=" * 80)
    print(
        "OFFICIAL TEST RESULT"
    )
    print("=" * 80)

    print()
    print(
        "Frozen final model:"
    )

    print(
        f"  description weight : "
        f"{description_weight:.2f}"
    )

    print(
        f"  MIL weight         : "
        f"{mil_weight:.2f}"
    )

    print(
        f"  threshold          : "
        f"{final_threshold:.2f}"
    )

    print()
    print(
        f"TEST pairs   : "
        f"{len(result):,}"
    )

    print(
        f"TEST anchors : "
        f"{result['anchor_article_uid'].nunique():,}"
    )

    print()
    print(
        "FINAL HYBRID:"
    )

    print(
        f"  Accuracy   : "
        f"{final_metrics['accuracy']:.6f}"
    )

    print(
        f"  Precision  : "
        f"{final_metrics['precision_consistent']:.6f}"
    )

    print(
        f"  Recall     : "
        f"{final_metrics['recall_consistent']:.6f}"
    )

    print(
        f"  F1         : "
        f"{final_metrics['f1_consistent']:.6f}"
    )

    print(
        f"  Macro-F1   : "
        f"{final_metrics['macro_f1']:.6f}"
    )

    print(
        f"  ROC-AUC    : "
        f"{final_metrics['roc_auc']:.6f}"
    )

    print(
        f"  AP         : "
        f"{final_metrics['average_precision']:.6f}"
    )

    cm = (
        final_metrics[
            "confusion_matrix"
        ]
    )

    print()
    print(
        "Confusion matrix:"
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
        "Negative tiers:"
    )

    for row in tier_df.itertuples(
        index=False
    ):

        print(
            f"  {row.tier:10s}: "
            f"n={row.n:,}, "
            f"negative_recall="
            f"{row.negative_recall:.4f}, "
            f"FP={row.false_positives:,}"
        )

    print()
    print(
        "Lowest 5 categories:"
    )

    for row in (
        category_df
        .head(5)
        .itertuples(
            index=False
        )
    ):

        print(
            f"  {row.category}: "
            f"Macro-F1="
            f"{row.macro_f1:.4f} "
            f"(n={row.n:,})"
        )

    print()
    print(
        "NO TEST-BASED TUNING WAS PERFORMED."
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "  This is a synthetic-negative TEST, "
        "not a human-gold real-world benchmark."
    )

    print()
    print(
        f"Predictions:\n"
        f"{PREDICTION_PATH}"
    )

    print()
    print(
        f"Metrics:\n"
        f"{METRICS_PATH}"
    )

    print()
    print(
        "STEP 21 FINISHED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()