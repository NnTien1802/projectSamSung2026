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
from torch.utils.data import (
    Dataset,
    DataLoader,
)

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

ARTICLE_INDEX_PATH = (
    MODULE_ROOT
    / "data"
    / "interim"
    / "consistency"
    / "articles_v1"
    / "article_index.parquet"
)

TRAIN_PAIR_PATH = (
    PAIR_DIR
    / "train.parquet"
)

VALIDATION_PAIR_PATH = (
    PAIR_DIR
    / "validation.parquet"
)


# ============================================================
# DEFAULT CONFIG
# ============================================================

DEFAULT_MODEL_NAME = (
    "bert-base-multilingual-cased"
)

DEFAULT_MAX_LENGTH = 160

DEFAULT_BATCH_SIZE = 16

DEFAULT_GRAD_ACCUM = 2

DEFAULT_EPOCHS = 1

DEFAULT_LR = 2e-5

DEFAULT_WEIGHT_DECAY = 0.01

DEFAULT_WARMUP_RATIO = 0.06

DEFAULT_SEED = 20260920

DEFAULT_NUM_WORKERS = 0

DEFAULT_MAX_GRAD_NORM = 1.0


# ============================================================
# GENERAL HELPERS
# ============================================================

def set_seed(
    seed,
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

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


def human_time(
    seconds,
):

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

    secs = (
        seconds
        %
        60
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d}"
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

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0,
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[
            0,
            1,
        ],
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

        average_precision = (
            average_precision_score(
                labels,
                probabilities,
            )
        )

    except ValueError:

        average_precision = None

    return {
        "threshold":
            float(
                threshold
            ),

        "accuracy":
            float(
                accuracy
            ),

        "precision_consistent":
            float(
                precision
            ),

        "recall_consistent":
            float(
                recall
            ),

        "f1_consistent":
            float(
                f1
            ),

        "macro_f1":
            float(
                macro_f1
            ),

        "roc_auc":
            (
                float(
                    roc_auc
                )
                if roc_auc
                is not None
                else None
            ),

        "average_precision":
            (
                float(
                    average_precision
                )
                if average_precision
                is not None
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


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def tune_threshold(
    labels,
    probabilities,
):

    candidates = np.arange(
        0.05,
        0.951,
        0.01,
    )

    rows = []

    best = None

    for threshold in candidates:

        metrics = compute_metrics(
            labels=labels,
            probabilities=probabilities,
            threshold=float(
                threshold
            ),
        )

        rows.append(
            {
                "threshold":
                    metrics[
                        "threshold"
                    ],

                "macro_f1":
                    metrics[
                        "macro_f1"
                    ],

                "accuracy":
                    metrics[
                        "accuracy"
                    ],

                "f1_consistent":
                    metrics[
                        "f1_consistent"
                    ],
            }
        )

        if best is None:

            best = metrics

            continue

        current_score = (
            metrics[
                "macro_f1"
            ]
        )

        best_score = (
            best[
                "macro_f1"
            ]
        )

        if (
            current_score
            >
            best_score
            +
            1e-12
        ):

            best = metrics

        elif (
            abs(
                current_score
                -
                best_score
            )
            <=
            1e-12
        ):

            # ----------------------------------------------
            # Tie-break:
            # prefer threshold closer to 0.5.
            # ----------------------------------------------

            if (
                abs(
                    threshold
                    -
                    0.5
                )
                <
                abs(
                    best[
                        "threshold"
                    ]
                    -
                    0.5
                )
            ):

                best = metrics

    return (
        best,
        pd.DataFrame(
            rows
        ),
    )


# ============================================================
# DATASET
# ============================================================

class PairDataset(
    Dataset
):

    def __init__(
        self,
        frame,
    ):

        self.pair_ids = (
            frame[
                "pair_id"
            ]
            .astype(str)
            .tolist()
        )

        self.headlines = (
            frame[
                "headline_text"
            ]
            .astype(str)
            .tolist()
        )

        self.descriptions = (
            frame[
                "anchor_description"
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

            "headline":
                self.headlines[
                    index
                ],

            "description":
                self.descriptions[
                    index
                ],

            "label":
                self.labels[
                    index
                ],
        }


# ============================================================
# BATCH COLLATOR
# ============================================================

class PairCollator:

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
            item[
                "headline"
            ]
            for item
            in batch
        ]

        descriptions = [
            item[
                "description"
            ]
            for item
            in batch
        ]

        labels = torch.tensor(
            [
                item[
                    "label"
                ]
                for item
                in batch
            ],
            dtype=torch.long,
        )

        pair_ids = [
            item[
                "pair_id"
            ]
            for item
            in batch
        ]

        # ----------------------------------------------------
        # Encode as sentence pair:
        #
        # headline [SEP] anchor_description
        #
        # Dynamic padding saves GPU memory.
        # ----------------------------------------------------

        encoded = (
            self.tokenizer(
                headlines,
                descriptions,
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
# LOAD ARTICLE METADATA
# ============================================================

def load_article_metadata():

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

    table = pq.read_table(
        ARTICLE_INDEX_PATH,
        columns=[
            "article_uid",
            "split",
            "title",
            "description",
        ],
    )

    frame = (
        table.to_pandas()
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

    metadata = (
        frame.set_index(
            "article_uid"
        )
    )

    print(
        f"Articles loaded: "
        f"{len(metadata):,}"
    )

    return metadata


# ============================================================
# MATERIALIZE ONE SPLIT
# ============================================================

def materialize_split(
    split,
    pair_path,
    metadata,
):

    print()
    print(
        f"Materializing {split}..."
    )

    pair_frame = (
        pq.read_table(
            pair_path,
            columns=[
                "pair_id",
                "split",
                "anchor_article_uid",
                "headline_article_uid",
                "label",
            ],
        )
        .to_pandas()
    )

    if not (
        pair_frame[
            "split"
        ]
        ==
        split
    ).all():

        raise RuntimeError(
            f"{split}: split mismatch."
        )

    # --------------------------------------------------------
    # Pair IDs.
    # --------------------------------------------------------

    if (
        pair_frame[
            "pair_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            f"{split}: duplicate pair_id."
        )

    # --------------------------------------------------------
    # Lookup headline from headline article.
    # --------------------------------------------------------

    pair_frame[
        "headline_text"
    ] = (
        pair_frame[
            "headline_article_uid"
        ]
        .map(
            metadata[
                "title"
            ]
        )
    )

    # --------------------------------------------------------
    # Lookup description from ANCHOR article.
    # --------------------------------------------------------

    pair_frame[
        "anchor_description"
    ] = (
        pair_frame[
            "anchor_article_uid"
        ]
        .map(
            metadata[
                "description"
            ]
        )
    )

    anchor_split = (
        pair_frame[
            "anchor_article_uid"
        ]
        .map(
            metadata[
                "split"
            ]
        )
    )

    headline_split = (
        pair_frame[
            "headline_article_uid"
        ]
        .map(
            metadata[
                "split"
            ]
        )
    )

    if (
        pair_frame[
            "headline_text"
        ]
        .isna()
        .any()
    ):

        raise RuntimeError(
            f"{split}: missing headline lookup."
        )

    if (
        pair_frame[
            "anchor_description"
        ]
        .isna()
        .any()
    ):

        raise RuntimeError(
            f"{split}: missing description lookup."
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

    # --------------------------------------------------------
    # Empty text check.
    # --------------------------------------------------------

    empty_headline = (
        pair_frame[
            "headline_text"
        ]
        .astype(str)
        .str.strip()
        ==
        ""
    )

    empty_description = (
        pair_frame[
            "anchor_description"
        ]
        .astype(str)
        .str.strip()
        ==
        ""
    )

    if empty_headline.any():

        raise RuntimeError(
            f"{split}: empty headline."
        )

    if empty_description.any():

        raise RuntimeError(
            f"{split}: empty description."
        )

    label_counts = (
        pair_frame[
            "label"
        ]
        .value_counts()
        .to_dict()
    )

    print(
        f"  rows     : "
        f"{len(pair_frame):,}"
    )

    print(
        f"  label 0  : "
        f"{label_counts.get(0, 0):,}"
    )

    print(
        f"  label 1  : "
        f"{label_counts.get(1, 0):,}"
    )

    return pair_frame


# ============================================================
# BALANCED SMOKE SUBSET
# ============================================================

def balanced_subset(
    frame,
    max_rows,
    seed,
):

    if (
        max_rows is None
        or
        len(frame)
        <=
        max_rows
    ):

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

    for label in [
        0,
        1,
    ]:

        subset = (
            frame[
                frame[
                    "label"
                ]
                ==
                label
            ]
            .sample(
                n=min(
                    per_class,
                    int(
                        (
                            frame[
                                "label"
                            ]
                            ==
                            label
                        ).sum()
                    ),
                ),
                random_state=(
                    seed
                    +
                    label
                ),
            )
        )

        parts.append(
            subset
        )

    result = (
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

    return result


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

    total_loss = 0.0

    total_examples = 0

    all_probabilities = []

    all_labels = []

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

        labels = batch[
            "labels"
        ]

        with torch.cuda.amp.autocast(
            enabled=use_amp,
        ):

            outputs = model(
                **batch
            )

            loss = (
                outputs.loss
            )

        logits = (
            outputs.logits
        )

        probabilities = (
            torch.softmax(
                logits,
                dim=-1,
            )[
                :,
                1
            ]
        )

        batch_size = (
            labels.size(
                0
            )
        )

        total_loss += (
            loss.item()
            *
            batch_size
        )

        total_examples += (
            batch_size
        )

        all_probabilities.extend(
            probabilities
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        all_labels.extend(
            labels
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        all_pair_ids.extend(
            pair_ids
        )

    average_loss = (
        total_loss
        /
        total_examples
    )

    return {
        "loss":
            float(
                average_loss
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

    scaler = (
        torch.cuda.amp.GradScaler(
            enabled=use_amp
        )
    )

    history = []

    best_macro_f1 = (
        -1.0
    )

    global_optimizer_step = 0

    print()
    print("=" * 80)
    print(
        "TRAINING"
    )
    print("=" * 80)

    print(
        f"Epochs                 : "
        f"{args.epochs}"
    )

    print(
        f"Physical batch size    : "
        f"{args.batch_size}"
    )

    print(
        f"Gradient accumulation  : "
        f"{args.grad_accum}"
    )

    print(
        f"Effective batch size   : "
        f"{args.batch_size * args.grad_accum}"
    )

    print(
        f"Optimizer steps/epoch  : "
        f"{optimizer_steps_per_epoch:,}"
    )

    print(
        f"Total optimizer steps  : "
        f"{total_optimizer_steps:,}"
    )

    print(
        f"Warmup steps           : "
        f"{warmup_steps:,}"
    )

    print(
        f"Mixed precision        : "
        f"{use_amp}"
    )

    model.zero_grad(
        set_to_none=True
    )

    for epoch in range(
        1,
        args.epochs
        +
        1,
    ):

        epoch_start = (
            time.time()
        )

        model.train()

        running_loss = (
            0.0
        )

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

            labels = (
                batch[
                    "labels"
                ]
            )

            batch_size = (
                labels.size(
                    0
                )
            )

            with torch.cuda.amp.autocast(
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
                (
                    step
                    %
                    args.grad_accum
                    ==
                    0
                )
                or
                (
                    step
                    ==
                    len(
                        train_loader
                    )
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

                global_optimizer_step += 1

            if (
                step
                %
                args.log_every
                ==
                0
            ):

                avg_loss = (
                    running_loss
                    /
                    seen_examples
                )

                print(
                    f"Epoch "
                    f"{epoch}/"
                    f"{args.epochs} "
                    f"| batch "
                    f"{step:,}/"
                    f"{len(train_loader):,} "
                    f"| train_loss="
                    f"{avg_loss:.5f} "
                    f"| opt_step="
                    f"{global_optimizer_step:,}"
                )

        train_loss = (
            running_loss
            /
            seen_examples
        )

        print()
        print(
            f"Evaluating validation "
            f"after epoch {epoch}..."
        )

        validation = evaluate(
            model=model,
            loader=(
                validation_loader
            ),
            device=device,
            use_amp=use_amp,
        )

        metrics_05 = (
            compute_metrics(
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
        )

        elapsed = (
            time.time()
            -
            epoch_start
        )

        epoch_row = {
            "epoch":
                epoch,

            "train_loss":
                float(
                    train_loss
                ),

            "validation_loss":
                validation[
                    "loss"
                ],

            "validation_accuracy_05":
                metrics_05[
                    "accuracy"
                ],

            "validation_macro_f1_05":
                metrics_05[
                    "macro_f1"
                ],

            "validation_f1_consistent_05":
                metrics_05[
                    "f1_consistent"
                ],

            "validation_roc_auc":
                metrics_05[
                    "roc_auc"
                ],

            "validation_ap":
                metrics_05[
                    "average_precision"
                ],

            "elapsed_seconds":
                float(
                    elapsed
                ),
        }

        history.append(
            epoch_row
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
            f"Epoch {epoch} result:"
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
            f"  macro F1   : "
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

        current_score = (
            metrics_05[
                "macro_f1"
            ]
        )

        if (
            current_score
            >
            best_macro_f1
        ):

            best_macro_f1 = (
                current_score
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "epoch":
                        epoch,

                    "validation_macro_f1_05":
                        current_score,

                    "model_name":
                        args.model_name,

                    "max_length":
                        args.max_length,
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
            "Baseline A for headline consistency: "
            "headline + anchor description."
        )
    )

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
        help=(
            "Use 4,000 train pairs and "
            "2,000 validation pairs."
        ),
    )

    args = parser.parse_args()

    if args.batch_size < 1:

        raise ValueError(
            "--batch-size must be >= 1"
        )

    if args.grad_accum < 1:

        raise ValueError(
            "--grad-accum must be >= 1"
        )

    if args.epochs < 1:

        raise ValueError(
            "--epochs must be >= 1"
        )

    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    run_name = (
        "baseline_title_description_mbert"
    )

    if args.smoke_test:

        run_name += (
            "_smoke"
        )

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

    PREDICTIONS_PATH = (
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
        "STEP 11 - BASELINE A"
    )
    print(
        "INPUT: HEADLINE + DESCRIPTION"
    )
    print("=" * 80)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "  TRAIN + VALIDATION only."
    )

    print(
        "  TEST split is never loaded "
        "by this script."
    )

    print()
    print(
        f"Model      : "
        f"{args.model_name}"
    )

    print(
        f"Max length : "
        f"{args.max_length}"
    )

    print(
        f"Batch size : "
        f"{args.batch_size}"
    )

    print(
        f"Grad accum : "
        f"{args.grad_accum}"
    )

    print(
        f"Epochs     : "
        f"{args.epochs}"
    )

    print(
        f"LR         : "
        f"{args.lr}"
    )

    print(
        f"Smoke test : "
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

    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

        gpu_name = (
            torch.cuda.get_device_name(
                0
            )
        )

    else:

        device = torch.device(
            "cpu"
        )

        gpu_name = None

    print()
    print(
        f"Device: "
        f"{device}"
    )

    if gpu_name:

        print(
            f"GPU   : "
            f"{gpu_name}"
        )

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    for path in [
        ARTICLE_INDEX_PATH,
        TRAIN_PAIR_PATH,
        VALIDATION_PAIR_PATH,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

    # ========================================================
    # MATERIALIZE INPUT TEXT
    # ========================================================

    metadata = (
        load_article_metadata()
    )

    train_frame = (
        materialize_split(
            split="train",
            pair_path=(
                TRAIN_PAIR_PATH
            ),
            metadata=metadata,
        )
    )

    validation_frame = (
        materialize_split(
            split="validation",
            pair_path=(
                VALIDATION_PAIR_PATH
            ),
            metadata=metadata,
        )
    )

    if args.smoke_test:

        train_frame = (
            balanced_subset(
                train_frame,
                max_rows=4000,
                seed=args.seed,
            )
        )

        validation_frame = (
            balanced_subset(
                validation_frame,
                max_rows=2000,
                seed=(
                    args.seed
                    +
                    100
                ),
            )
        )

        print()
        print(
            "SMOKE TEST SUBSET:"
        )

        print(
            f"  train      : "
            f"{len(train_frame):,}"
        )

        print(
            f"  validation : "
            f"{len(validation_frame):,}"
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

    collator = PairCollator(
        tokenizer=tokenizer,
        max_length=(
            args.max_length
        ),
    )

    # ========================================================
    # DATALOADERS
    # ========================================================

    train_dataset = PairDataset(
        train_frame
    )

    validation_dataset = (
        PairDataset(
            validation_frame
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
    # SAVE RUN CONFIG
    # ========================================================

    config = {
        "stage":
            "11_baseline_title_description",

        "model_name":
            args.model_name,

        "input":
            (
                "headline_used_in_pair "
                "+ anchor_description"
            ),

        "train_rows":
            int(
                len(
                    train_frame
                )
            ),

        "validation_rows":
            int(
                len(
                    validation_frame
                )
            ),

        "test_loaded":
            False,

        "max_length":
            int(
                args.max_length
            ),

        "batch_size":
            int(
                args.batch_size
            ),

        "gradient_accumulation":
            int(
                args.grad_accum
            ),

        "effective_batch_size":
            int(
                args.batch_size
                *
                args.grad_accum
            ),

        "epochs":
            int(
                args.epochs
            ),

        "learning_rate":
            float(
                args.lr
            ),

        "weight_decay":
            float(
                args.weight_decay
            ),

        "warmup_ratio":
            float(
                args.warmup_ratio
            ),

        "seed":
            int(
                args.seed
            ),

        "amp":
            bool(
                device.type
                ==
                "cuda"
                and
                not args.no_amp
            ),

        "device":
            str(
                device
            ),

        "gpu":
            gpu_name,

        "smoke_test":
            bool(
                args.smoke_test
            ),
    }

    save_json(
        config,
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
    # LOAD BEST CHECKPOINT
    # ========================================================

    print()
    print("=" * 80)
    print(
        "LOAD BEST CHECKPOINT"
    )
    print("=" * 80)

    checkpoint = torch.load(
        BEST_MODEL_PATH,
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

    print(
        "Validation Macro-F1 @0.5: "
        f"{checkpoint['validation_macro_f1_05']:.6f}"
    )

    # ========================================================
    # FINAL VALIDATION PREDICTIONS
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

    # ========================================================
    # VALIDATION THRESHOLD TUNING
    # ========================================================

    (
        best_threshold_metrics,
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
    # SAVE PREDICTIONS
    # ========================================================

    prediction_frame = pd.DataFrame(
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

    prediction_frame[
        "prediction_05"
    ] = (
        prediction_frame[
            "prob_consistent"
        ]
        >=
        0.5
    ).astype(
        np.int8
    )

    selected_threshold = (
        best_threshold_metrics[
            "threshold"
        ]
    )

    prediction_frame[
        "prediction_tuned"
    ] = (
        prediction_frame[
            "prob_consistent"
        ]
        >=
        selected_threshold
    ).astype(
        np.int8
    )

    prediction_frame.to_parquet(
        PREDICTIONS_PATH,
        index=False,
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics_report = {
        "dataset":
            "headline_consistency_v1",

        "model":
            args.model_name,

        "baseline":
            (
                "A_title_plus_description"
            ),

        "test_used":
            False,

        "checkpoint_selection":
            (
                "best validation Macro-F1 "
                "at threshold 0.5"
            ),

        "validation_loss":
            validation[
                "loss"
            ],

        "metrics_at_0_5":
            metrics_05,

        "best_validation_threshold":
            selected_threshold,

        "metrics_at_tuned_threshold":
            best_threshold_metrics,

        "important_note":
            (
                "Threshold was tuned on "
                "validation only. "
                "Do not tune on test."
            ),
    }

    save_json(
        metrics_report,
        METRICS_PATH,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print(
        "STEP 11 VALIDATION RESULTS"
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

    print(
        f"  AP       : "
        f"{metrics_05['average_precision']:.6f}"
    )

    cm = (
        metrics_05[
            "confusion_matrix"
        ]
    )

    print(
        "  Confusion:"
    )

    print(
        f"    TN={cm['tn']:,} "
        f"FP={cm['fp']:,}"
    )

    print(
        f"    FN={cm['fn']:,} "
        f"TP={cm['tp']:,}"
    )

    print()
    print(
        "Best validation threshold:"
    )

    print(
        f"  threshold: "
        f"{best_threshold_metrics['threshold']:.2f}"
    )

    print(
        f"  Accuracy : "
        f"{best_threshold_metrics['accuracy']:.6f}"
    )

    print(
        f"  Macro-F1 : "
        f"{best_threshold_metrics['macro_f1']:.6f}"
    )

    print(
        f"  F1       : "
        f"{best_threshold_metrics['f1_consistent']:.6f}"
    )

    tuned_cm = (
        best_threshold_metrics[
            "confusion_matrix"
        ]
    )

    print(
        "  Confusion:"
    )

    print(
        f"    TN={tuned_cm['tn']:,} "
        f"FP={tuned_cm['fp']:,}"
    )

    print(
        f"    FN={tuned_cm['fn']:,} "
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
        "STEP 11 FINISHED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()