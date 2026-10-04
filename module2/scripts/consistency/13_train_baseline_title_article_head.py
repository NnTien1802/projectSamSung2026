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

ARTICLE_DIR = (
    MODULE_ROOT
    / "data"
    / "interim"
    / "consistency"
    / "articles_v1"
)

ARTICLE_INDEX_PATH = (
    ARTICLE_DIR
    / "article_index.parquet"
)

PAIR_FILES = {
    "train":
        PAIR_DIR / "train.parquet",

    "validation":
        PAIR_DIR / "validation.parquet",
}

ARTICLE_FILES = {
    "train":
        ARTICLE_DIR / "train.parquet",

    "validation":
        ARTICLE_DIR / "validation.parquet",
}


# ============================================================
# CONFIG
# ============================================================

DEFAULT_MODEL_NAME = (
    "bert-base-multilingual-cased"
)

DEFAULT_MAX_LENGTH = 160

DEFAULT_BATCH_SIZE = 8

DEFAULT_GRAD_ACCUM = 4

DEFAULT_EPOCHS = 1

DEFAULT_LR = 2e-5

DEFAULT_WEIGHT_DECAY = 0.01

DEFAULT_WARMUP_RATIO = 0.06

DEFAULT_SEED = 20260920

DEFAULT_NUM_WORKERS = 0

DEFAULT_MAX_GRAD_NORM = 1.0

# Avoid tokenizing an extremely long full article when
# the model can only consume 160 tokens anyway.
#
# 3000 chars is much larger than the actual token window,
# while keeping CPU tokenization manageable.
DEFAULT_ARTICLE_CHAR_LIMIT = 3000


# ============================================================
# HELPERS
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(data, path):

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

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

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

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = (
        matrix.ravel()
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
            float(threshold),

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
                float(roc_auc)
                if roc_auc is not None
                else None
            ),

        "average_precision":
            (
                float(ap)
                if ap is not None
                else None
            ),

        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def tune_threshold(
    labels,
    probabilities,
):

    rows = []

    best = None

    for threshold in np.arange(
        0.05,
        0.951,
        0.01,
    ):

        metrics = compute_metrics(
            labels,
            probabilities,
            threshold=float(
                threshold
            ),
        )

        rows.append(
            {
                "threshold":
                    metrics["threshold"],

                "accuracy":
                    metrics["accuracy"],

                "macro_f1":
                    metrics["macro_f1"],

                "f1_consistent":
                    metrics[
                        "f1_consistent"
                    ],
            }
        )

        if best is None:

            best = metrics
            continue

        if (
            metrics["macro_f1"]
            >
            best["macro_f1"]
            +
            1e-12
        ):

            best = metrics

        elif (
            abs(
                metrics["macro_f1"]
                -
                best["macro_f1"]
            )
            <=
            1e-12
        ):

            if (
                abs(
                    threshold
                    -
                    0.5
                )
                <
                abs(
                    best["threshold"]
                    -
                    0.5
                )
            ):

                best = metrics

    return (
        best,
        pd.DataFrame(rows),
    )


# ============================================================
# LOAD TITLE MAP
# ============================================================

def load_title_map():

    print()
    print("=" * 80)
    print("LOAD ARTICLE TITLES")
    print("=" * 80)

    table = pq.read_table(
        ARTICLE_INDEX_PATH,
        columns=[
            "article_uid",
            "title",
            "split",
        ],
    )

    frame = table.to_pandas()

    if frame[
        "article_uid"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate article_uid "
            "in article index."
        )

    title_map = dict(
        zip(
            frame["article_uid"],
            frame["title"],
        )
    )

    split_map = dict(
        zip(
            frame["article_uid"],
            frame["split"],
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
# PAIRS
# ============================================================

def load_pairs(
    split,
    path,
    split_map,
):

    frame = (
        pq.read_table(
            path,
            columns=[
                "pair_id",
                "anchor_article_uid",
                "headline_article_uid",
                "label",
            ],
        )
        .to_pandas()
    )

    if frame[
        "pair_id"
    ].duplicated().any():

        raise RuntimeError(
            f"{split}: duplicate pair_id."
        )

    anchor_splits = (
        frame[
            "anchor_article_uid"
        ]
        .map(
            split_map
        )
    )

    headline_splits = (
        frame[
            "headline_article_uid"
        ]
        .map(
            split_map
        )
    )

    if not (
        anchor_splits
        ==
        split
    ).all():

        raise RuntimeError(
            f"{split}: anchor leakage."
        )

    if not (
        headline_splits
        ==
        split
    ).all():

        raise RuntimeError(
            f"{split}: headline leakage."
        )

    print(
        f"{split:12s}: "
        f"{len(frame):,} pairs"
    )

    return frame


# ============================================================
# BALANCED SMOKE SUBSET
# ============================================================

def balanced_subset(
    frame,
    max_rows,
    seed,
):

    if len(frame) <= max_rows:

        return (
            frame
            .reset_index(
                drop=True
            )
        )

    per_class = (
        max_rows
        //
        2
    )

    parts = []

    for label in [0, 1]:

        group = (
            frame[
                frame["label"]
                ==
                label
            ]
        )

        part = group.sample(
            n=per_class,
            random_state=(
                seed
                +
                label
            ),
        )

        parts.append(part)

    return (
        pd.concat(
            parts,
            ignore_index=True,
        )
        .sample(
            frac=1.0,
            random_state=seed,
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# LOAD CONTENT
# ============================================================

def load_content_map(
    split,
    needed_anchor_uids,
):

    print()
    print(
        f"Loading {split} article content..."
    )

    table = pq.read_table(
        ARTICLE_FILES[
            split
        ],
        columns=[
            "article_uid",
            "content",
        ],
    )

    frame = (
        table.to_pandas()
    )

    # Only keep anchors actually used by the pair dataset.
    frame = frame[
        frame[
            "article_uid"
        ].isin(
            needed_anchor_uids
        )
    ]

    content_map = dict(
        zip(
            frame[
                "article_uid"
            ],
            frame[
                "content"
            ],
        )
    )

    missing = (
        set(
            needed_anchor_uids
        )
        -
        set(
            content_map
        )
    )

    if missing:

        raise RuntimeError(
            f"{split}: missing "
            f"{len(missing):,} "
            f"anchor contents."
        )

    print(
        f"  content loaded: "
        f"{len(content_map):,}"
    )

    return content_map


# ============================================================
# DATASET
# ============================================================

class ArticleHeadPairDataset(
    Dataset
):

    def __init__(
        self,
        pair_frame,
        title_map,
        content_map,
        article_char_limit,
    ):

        self.pair_ids = (
            pair_frame[
                "pair_id"
            ]
            .astype(str)
            .tolist()
        )

        self.anchor_uids = (
            pair_frame[
                "anchor_article_uid"
            ]
            .astype(str)
            .tolist()
        )

        self.headline_uids = (
            pair_frame[
                "headline_article_uid"
            ]
            .astype(str)
            .tolist()
        )

        self.labels = (
            pair_frame[
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

        self.content_map = (
            content_map
        )

        self.article_char_limit = (
            article_char_limit
        )

    def __len__(self):

        return len(
            self.labels
        )

    def __getitem__(
        self,
        index,
    ):

        headline_uid = (
            self.headline_uids[
                index
            ]
        )

        anchor_uid = (
            self.anchor_uids[
                index
            ]
        )

        headline = (
            self.title_map[
                headline_uid
            ]
        )

        content = (
            self.content_map[
                anchor_uid
            ]
        )

        if (
            self.article_char_limit
            is not None
        ):

            content = content[
                :self.article_char_limit
            ]

        return {
            "pair_id":
                self.pair_ids[
                    index
                ],

            "headline":
                headline,

            "content":
                content,

            "label":
                self.labels[
                    index
                ],
        }


# ============================================================
# COLLATOR
# ============================================================

class ArticleHeadCollator:

    def __init__(
        self,
        tokenizer,
        max_length,
    ):

        self.tokenizer = (
            tokenizer
        )

        self.max_length = (
            max_length
        )

    def __call__(
        self,
        batch,
    ):

        headlines = [
            item["headline"]
            for item in batch
        ]

        contents = [
            item["content"]
            for item in batch
        ]

        labels = torch.tensor(
            [
                item["label"]
                for item in batch
            ],
            dtype=torch.long,
        )

        pair_ids = [
            item["pair_id"]
            for item in batch
        ]

        encoded = (
            self.tokenizer(
                headlines,
                contents,
                padding=True,
                truncation=True,
                max_length=(
                    self.max_length
                ),
                return_tensors="pt",
            )
        )

        encoded[
            "labels"
        ] = labels

        encoded[
            "pair_ids"
        ] = pair_ids

        return encoded


# ============================================================
# EVALUATE
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    use_amp,
):

    model.eval()

    total_loss = 0.0
    total_examples = 0

    all_labels = []
    all_probabilities = []
    all_pair_ids = []

    for batch in loader:

        pair_ids = batch.pop(
            "pair_ids"
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

        labels = (
            batch[
                "labels"
            ]
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=use_amp,
        ):

            outputs = model(
                **batch
            )

        probabilities = (
            torch.softmax(
                outputs.logits,
                dim=-1,
            )[
                :,
                1
            ]
        )

        batch_size = (
            labels.size(0)
        )

        total_loss += (
            outputs.loss.item()
            *
            batch_size
        )

        total_examples += (
            batch_size
        )

        all_labels.extend(
            labels
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        all_probabilities.extend(
            probabilities
            .detach()
            .cpu()
            .numpy()
            .tolist()
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
# TRAIN
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

    steps_per_epoch = math.ceil(
        len(train_loader)
        /
        args.grad_accum
    )

    total_steps = (
        steps_per_epoch
        *
        args.epochs
    )

    warmup_steps = int(
        total_steps
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
                total_steps
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

    history = []

    best_macro_f1 = (
        -1.0
    )

    optimizer_step = 0

    print()
    print("=" * 80)
    print("TRAINING")
    print("=" * 80)

    print(
        f"Physical batch       : "
        f"{args.batch_size}"
    )

    print(
        f"Gradient accumulation: "
        f"{args.grad_accum}"
    )

    print(
        f"Effective batch      : "
        f"{args.batch_size * args.grad_accum}"
    )

    print(
        f"Optimizer steps      : "
        f"{total_steps:,}"
    )

    print(
        f"Mixed precision      : "
        f"{use_amp}"
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        start_time = (
            time.time()
        )

        model.train()

        running_loss = 0.0
        seen_examples = 0

        for step, batch in enumerate(
            train_loader,
            start=1,
        ):

            batch.pop(
                "pair_ids"
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

            batch_size = (
                batch[
                    "labels"
                ]
                .size(0)
            )

            with torch.amp.autocast(
                device_type="cuda",
                enabled=use_amp,
            ):

                outputs = model(
                    **batch
                )

                raw_loss = (
                    outputs.loss
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
                raw_loss.item()
                *
                batch_size
            )

            seen_examples += (
                batch_size
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
                    f"| train_loss="
                    f"{running_loss / seen_examples:.5f} "
                    f"| opt_step="
                    f"{optimizer_step:,}"
                )

        train_loss = (
            running_loss
            /
            seen_examples
        )

        print()
        print(
            "Evaluating validation..."
        )

        validation = evaluate(
            model,
            validation_loader,
            device,
            use_amp,
        )

        metrics = compute_metrics(
            validation["labels"],
            validation[
                "probabilities"
            ],
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
                    epoch,

                "train_loss":
                    train_loss,

                "validation_loss":
                    validation["loss"],

                "validation_accuracy_05":
                    metrics["accuracy"],

                "validation_macro_f1_05":
                    metrics["macro_f1"],

                "validation_f1_consistent_05":
                    metrics[
                        "f1_consistent"
                    ],

                "validation_roc_auc":
                    metrics[
                        "roc_auc"
                    ],

                "elapsed_seconds":
                    elapsed,
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
            f"{metrics['accuracy']:.6f}"
        )

        print(
            f"  Macro-F1   : "
            f"{metrics['macro_f1']:.6f}"
        )

        print(
            f"  ROC-AUC    : "
            f"{metrics['roc_auc']:.6f}"
        )

        print(
            f"  elapsed    : "
            f"{human_time(elapsed)}"
        )

        if (
            metrics[
                "macro_f1"
            ]
            >
            best_macro_f1
        ):

            best_macro_f1 = (
                metrics[
                    "macro_f1"
                ]
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "epoch":
                        epoch,

                    "validation_macro_f1_05":
                        best_macro_f1,

                    "model_name":
                        args.model_name,

                    "max_length":
                        args.max_length,

                    "article_char_limit":
                        args.article_char_limit,
                },
                best_model_path,
            )

            print(
                "  → best checkpoint saved"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-name",
        type=str,
        default=(
            DEFAULT_MODEL_NAME
        ),
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=(
            DEFAULT_MAX_LENGTH
        ),
    )

    parser.add_argument(
        "--article-char-limit",
        type=int,
        default=(
            DEFAULT_ARTICLE_CHAR_LIMIT
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
        "--max-grad-norm",
        type=float,
        default=(
            DEFAULT_MAX_GRAD_NORM
        ),
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
    )

    args = parser.parse_args()

    # ========================================================
    # OUTPUTS
    # ========================================================

    run_name = (
        "baseline_title_article_head_mbert"
    )

    if args.smoke_test:

        run_name += "_smoke"

    model_dir = (
        MODULE_ROOT
        / "models"
        / "consistency"
        / run_name
    )

    result_dir = (
        MODULE_ROOT
        / "results"
        / "consistency"
        / run_name
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_model_path = (
        model_dir
        / "best_model.pt"
    )

    history_path = (
        result_dir
        / "training_history.csv"
    )

    metrics_path = (
        result_dir
        / "validation_metrics.json"
    )

    predictions_path = (
        result_dir
        / "validation_predictions.parquet"
    )

    threshold_path = (
        result_dir
        / "threshold_search.csv"
    )

    config_path = (
        result_dir
        / "run_config.json"
    )

    print("=" * 80)
    print(
        "MODULE 2 - HEADLINE INTEGRITY"
    )
    print(
        "TASK 2B - HEADLINE CONSISTENCY"
    )
    print(
        "STEP 13 - BASELINE B"
    )
    print(
        "INPUT: HEADLINE + ARTICLE HEAD"
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
        f"Model              : "
        f"{args.model_name}"
    )

    print(
        f"Max tokens         : "
        f"{args.max_length}"
    )

    print(
        f"Article char limit : "
        f"{args.article_char_limit}"
    )

    print(
        f"Batch size         : "
        f"{args.batch_size}"
    )

    print(
        f"Grad accum         : "
        f"{args.grad_accum}"
    )

    print(
        f"Effective batch    : "
        f"{args.batch_size * args.grad_accum}"
    )

    set_seed(
        args.seed
    )

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        print(
            "GPU   : "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ========================================================
    # FILE CHECK
    # ========================================================

    required = [
        ARTICLE_INDEX_PATH,
        PAIR_FILES["train"],
        PAIR_FILES["validation"],
        ARTICLE_FILES["train"],
        ARTICLE_FILES["validation"],
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

    # ========================================================
    # TITLES + PAIRS
    # ========================================================

    (
        title_map,
        split_map,
    ) = load_title_map()

    train_pairs = load_pairs(
        "train",
        PAIR_FILES["train"],
        split_map,
    )

    validation_pairs = load_pairs(
        "validation",
        PAIR_FILES[
            "validation"
        ],
        split_map,
    )

    if args.smoke_test:

        train_pairs = balanced_subset(
            train_pairs,
            4000,
            args.seed,
        )

        validation_pairs = balanced_subset(
            validation_pairs,
            2000,
            args.seed + 100,
        )

        print()
        print(
            "SMOKE TEST:"
        )

        print(
            f"  train      : "
            f"{len(train_pairs):,}"
        )

        print(
            f"  validation : "
            f"{len(validation_pairs):,}"
        )

    # ========================================================
    # CONTENT MAPS
    # ========================================================

    train_anchor_uids = set(
        train_pairs[
            "anchor_article_uid"
        ]
    )

    validation_anchor_uids = set(
        validation_pairs[
            "anchor_article_uid"
        ]
    )

    train_content_map = (
        load_content_map(
            "train",
            train_anchor_uids,
        )
    )

    validation_content_map = (
        load_content_map(
            "validation",
            validation_anchor_uids,
        )
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
            args.model_name
        )
    )

    collator = (
        ArticleHeadCollator(
            tokenizer,
            args.max_length,
        )
    )

    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = (
        ArticleHeadPairDataset(
            train_pairs,
            title_map,
            train_content_map,
            args.article_char_limit,
        )
    )

    validation_dataset = (
        ArticleHeadPairDataset(
            validation_pairs,
            title_map,
            validation_content_map,
            args.article_char_limit,
        )
    )

    generator = (
        torch.Generator()
    )

    generator.manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=(
            args.batch_size
        ),
        shuffle=True,
        collate_fn=collator,
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
            args.batch_size
        ),
        shuffle=False,
        collate_fn=collator,
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
    print(
        "Loading model..."
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            args.model_name,
            num_labels=2,
        )
    )

    model.to(
        device
    )

    # ========================================================
    # CONFIG
    # ========================================================

    save_json(
        {
            "stage":
                "13_baseline_title_article_head",

            "model_name":
                args.model_name,

            "input":
                (
                    "headline_used_in_pair "
                    "+ beginning_of_anchor_article"
                ),

            "max_length":
                args.max_length,

            "article_char_limit":
                args.article_char_limit,

            "train_rows":
                len(
                    train_pairs
                ),

            "validation_rows":
                len(
                    validation_pairs
                ),

            "test_loaded":
                False,

            "batch_size":
                args.batch_size,

            "grad_accum":
                args.grad_accum,

            "effective_batch":
                (
                    args.batch_size
                    *
                    args.grad_accum
                ),

            "epochs":
                args.epochs,

            "learning_rate":
                args.lr,

            "seed":
                args.seed,

            "smoke_test":
                args.smoke_test,
        },
        config_path,
    )

    # ========================================================
    # TRAIN
    # ========================================================

    train_model(
        model,
        train_loader,
        validation_loader,
        device,
        args,
        best_model_path,
        history_path,
    )

    # ========================================================
    # BEST MODEL
    # ========================================================

    print()
    print("=" * 80)
    print(
        "LOAD BEST CHECKPOINT"
    )
    print("=" * 80)

    checkpoint = torch.load(
        best_model_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print(
        f"Best epoch: "
        f"{checkpoint['epoch']}"
    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    use_amp = (
        device.type
        ==
        "cuda"
        and
        not args.no_amp
    )

    validation = evaluate(
        model,
        validation_loader,
        device,
        use_amp,
    )

    metrics_05 = compute_metrics(
        validation["labels"],
        validation[
            "probabilities"
        ],
        threshold=0.5,
    )

    (
        tuned_metrics,
        threshold_frame,
    ) = tune_threshold(
        validation["labels"],
        validation[
            "probabilities"
        ],
    )

    threshold_frame.to_csv(
        threshold_path,
        index=False,
        encoding="utf-8-sig",
    )

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
        predictions_path,
        index=False,
    )

    save_json(
        {
            "baseline":
                "B_title_plus_article_head",

            "model":
                args.model_name,

            "dataset":
                "headline_consistency_v1",

            "test_used":
                False,

            "validation_loss":
                validation["loss"],

            "metrics_at_0_5":
                metrics_05,

            "best_validation_threshold":
                tuned_threshold,

            "metrics_at_tuned_threshold":
                tuned_metrics,
        },
        metrics_path,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print(
        "STEP 13 VALIDATION RESULTS"
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
        "Best validation threshold:"
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

    print()
    print(
        "TEST WAS NOT USED."
    )

    print()
    print(
        f"Best model:\n"
        f"{best_model_path}"
    )

    print()
    print(
        f"Results:\n"
        f"{result_dir}"
    )

    print()
    print(
        "STEP 13 FINISHED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()