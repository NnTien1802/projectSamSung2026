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

DATA_DIR = (
    MODULE_ROOT
    / "data"
    / "processed"
    / "clickbait"
    / "viclickbait_v1"
)

TRAIN_PATH = (
    DATA_DIR
    / "train.parquet"
)

VALIDATION_PATH = (
    DATA_DIR
    / "validation.parquet"
)

MODEL_DIR = (
    MODULE_ROOT
    / "models"
    / "clickbait"
    / "mbert_title_lead_v1"
)

RESULT_DIR = (
    MODULE_ROOT
    / "results"
    / "clickbait"
    / "mbert_title_lead_v1"
)

BEST_MODEL_PATH = (
    MODEL_DIR
    / "best_model.pt"
)

HISTORY_PATH = (
    RESULT_DIR
    / "training_history.csv"
)

PREDICTION_PATH = (
    RESULT_DIR
    / "validation_predictions.parquet"
)

METRICS_PATH = (
    RESULT_DIR
    / "validation_metrics.json"
)

THRESHOLD_PATH = (
    RESULT_DIR
    / "threshold_search.csv"
)

CONFIG_PATH = (
    RESULT_DIR
    / "run_config.json"
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "bert-base-multilingual-cased"

MAX_LENGTH = 160

DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 5
DEFAULT_PATIENCE = 2

DEFAULT_LR = 2e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.10
DEFAULT_MAX_GRAD_NORM = 1.0

DEFAULT_SEED = 20260921
DEFAULT_NUM_WORKERS = 0

THRESHOLDS = np.arange(
    0.05,
    0.951,
    0.01,
)


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
# DATASET
# ============================================================

class ClickbaitDataset(Dataset):

    def __init__(
        self,
        frame,
    ):

        self.sample_ids = (
            frame[
                "sample_id"
            ]
            .astype(str)
            .tolist()
        )

        self.titles = (
            frame[
                "title"
            ]
            .fillna("")
            .astype(str)
            .tolist()
        )

        self.leads = (
            frame[
                "lead"
            ]
            .fillna("")
            .astype(str)
            .tolist()
        )

        self.labels = (
            frame[
                "label"
            ]
            .astype(int)
            .tolist()
        )

    def __len__(self):

        return len(
            self.labels
        )

    def __getitem__(
        self,
        index,
    ):

        return {
            "sample_id":
                self.sample_ids[
                    index
                ],

            "title":
                self.titles[
                    index
                ],

            "lead":
                self.leads[
                    index
                ],

            "label":
                self.labels[
                    index
                ],
        }


# ============================================================
# COLLATOR
# ============================================================

class TitleLeadCollator:

    def __init__(
        self,
        tokenizer,
    ):

        self.tokenizer = tokenizer

    def __call__(
        self,
        batch,
    ):

        titles = [
            item[
                "title"
            ]
            for item in batch
        ]

        leads = [
            item[
                "lead"
            ]
            for item in batch
        ]

        sample_ids = [
            item[
                "sample_id"
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

        # Preserve the title.
        # Truncate only the lead when necessary.
        encoded = self.tokenizer(
            titles,
            leads,
            padding=True,
            truncation="only_second",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        encoded[
            "labels"
        ] = labels

        encoded[
            "sample_ids"
        ] = sample_ids

        return encoded


# ============================================================
# LOAD DATA
# ============================================================

def load_split(
    name,
    path,
):

    if not path.exists():

        raise FileNotFoundError(
            path
        )

    frame = (
        pq.read_table(
            path,
            columns=[
                "sample_id",
                "title",
                "lead",
                "label",
            ],
        )
        .to_pandas()
    )

    if (
        frame[
            "sample_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            f"{name}: duplicate sample_id."
        )

    frame[
        "title"
    ] = (
        frame[
            "title"
        ]
        .fillna("")
        .astype(str)
    )

    frame[
        "lead"
    ] = (
        frame[
            "lead"
        ]
        .fillna("")
        .astype(str)
    )

    if (
        frame[
            "title"
        ]
        .eq("")
        .any()
    ):

        raise RuntimeError(
            f"{name}: empty title."
        )

    labels = set(
        frame[
            "label"
        ].unique()
    )

    if not labels.issubset(
        {
            0,
            1,
        }
    ):

        raise RuntimeError(
            f"{name}: invalid labels."
        )

    print(
        f"{name:12s}: "
        f"{len(frame):,} rows"
    )

    print(
        f"  NORMAL   : "
        f"{int((frame['label'] == 0).sum()):,}"
    )

    print(
        f"  CLICKBAIT: "
        f"{int((frame['label'] == 1).sum()):,}"
    )

    print(
        f"  empty lead: "
        f"{int(frame['lead'].eq('').sum()):,}"
    )

    return frame


# ============================================================
# CLASS WEIGHTS
# ============================================================

def compute_class_weights(
    labels,
):

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    counts = np.bincount(
        labels,
        minlength=2,
    )

    if np.any(
        counts == 0
    ):

        raise RuntimeError(
            "A class has zero samples."
        )

    total = int(
        counts.sum()
    )

    weights = (
        total
        /
        (
            2.0
            *
            counts.astype(
                np.float64
            )
        )
    )

    return (
        counts,
        weights,
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

    try:

        roc_auc = float(
            roc_auc_score(
                labels,
                probabilities,
            )
        )

    except ValueError:

        roc_auc = None

    try:

        ap = float(
            average_precision_score(
                labels,
                probabilities,
            )
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

        "precision_clickbait":
            float(
                precision_score(
                    labels,
                    predictions,
                    zero_division=0,
                )
            ),

        "recall_clickbait":
            float(
                recall_score(
                    labels,
                    predictions,
                    zero_division=0,
                )
            ),

        "f1_clickbait":
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
            roc_auc,

        "average_precision":
            ap,

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


def tune_threshold(
    labels,
    probabilities,
):

    rows = []

    best_metrics = None
    best_key = None

    for threshold in THRESHOLDS:

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

                "accuracy":
                    metrics[
                        "accuracy"
                    ],

                "macro_f1":
                    metrics[
                        "macro_f1"
                    ],

                "precision_clickbait":
                    metrics[
                        "precision_clickbait"
                    ],

                "recall_clickbait":
                    metrics[
                        "recall_clickbait"
                    ],

                "f1_clickbait":
                    metrics[
                        "f1_clickbait"
                    ],
            }
        )

        key = (
            metrics[
                "macro_f1"
            ],

            metrics[
                "accuracy"
            ],

            -abs(
                threshold
                -
                0.5
            ),
        )

        if (
            best_key is None
            or
            key
            >
            best_key
        ):

            best_key = key
            best_metrics = metrics

    return (
        best_metrics,
        pd.DataFrame(
            rows
        ),
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    loss_fn,
    use_amp,
):

    model.eval()

    total_loss = 0.0
    total_examples = 0

    all_sample_ids = []
    all_labels = []
    all_probabilities = []

    for batch in loader:

        sample_ids = (
            batch.pop(
                "sample_ids"
            )
        )

        labels = (
            batch.pop(
                "labels"
            )
            .to(
                device,
                non_blocking=True,
            )
        )

        inputs = {
            key:
                value.to(
                    device,
                    non_blocking=True,
                )

            for key, value
            in batch.items()
        }

        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):

            outputs = model(
                **inputs
            )

            loss = loss_fn(
                outputs.logits,
                labels,
            )

        probabilities = (
            torch.softmax(
                outputs.logits,
                dim=-1,
            )[
                :,
                1
            ]
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

        batch_size = int(
            len(
                labels_np
            )
        )

        total_loss += (
            float(
                loss.item()
            )
            *
            batch_size
        )

        total_examples += (
            batch_size
        )

        all_sample_ids.extend(
            sample_ids
        )

        all_labels.extend(
            labels_np.tolist()
        )

        all_probabilities.extend(
            probabilities.tolist()
        )

    return {
        "loss":
            float(
                total_loss
                /
                total_examples
            ),

        "sample_ids":
            all_sample_ids,

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
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )

    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=DEFAULT_WARMUP_RATIO,
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=DEFAULT_MAX_GRAD_NORM,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    args = parser.parse_args()

    # ========================================================
    # HEADER
    # ========================================================

    print("=" * 80)
    print(
        "MODULE 2 - HEADLINE INTEGRITY"
    )
    print(
        "TASK 2A - CLICKBAIT DETECTION"
    )
    print(
        "STEP 07 - MBERT TITLE + LEAD"
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
        f"Model          : "
        f"{MODEL_NAME}"
    )

    print(
        "Input          : title + lead"
    )

    print(
        "Truncation     : only_second "
        "(preserve title)"
    )

    print(
        f"Max length     : "
        f"{MAX_LENGTH}"
    )

    print(
        f"Batch size     : "
        f"{args.batch_size}"
    )

    print(
        f"Max epochs     : "
        f"{args.epochs}"
    )

    print(
        f"Patience       : "
        f"{args.patience}"
    )

    print(
        f"Learning rate  : "
        f"{args.lr}"
    )

    # ========================================================
    # DIRECTORIES
    # ========================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # SEED / DEVICE
    # ========================================================

    set_seed(
        args.seed
    )

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

    use_amp = (
        device.type
        ==
        "cuda"
        and
        not args.no_amp
    )

    # ========================================================
    # DATA
    # ========================================================

    print()
    print("=" * 80)
    print(
        "LOAD DATA"
    )
    print("=" * 80)

    train_frame = load_split(
        "train",
        TRAIN_PATH,
    )

    validation_frame = load_split(
        "validation",
        VALIDATION_PATH,
    )

    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    (
        class_counts,
        class_weights,
    ) = compute_class_weights(
        train_frame[
            "label"
        ]
    )

    print()
    print(
        "Class weights:"
    )

    print(
        f"  NORMAL    n={class_counts[0]:,} "
        f"weight={class_weights[0]:.6f}"
    )

    print(
        f"  CLICKBAIT n={class_counts[1]:,} "
        f"weight={class_weights[1]:.6f}"
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

    train_dataset = ClickbaitDataset(
        train_frame
    )

    validation_dataset = ClickbaitDataset(
        validation_frame
    )

    collator = TitleLeadCollator(
        tokenizer
    )

    generator = torch.Generator()

    generator.manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=(
            device.type
            ==
            "cuda"
        ),
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
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
            MODEL_NAME,
            num_labels=2,
        )
    )

    model.to(
        device
    )

    loss_weights_tensor = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )

    loss_fn = torch.nn.CrossEntropyLoss(
        weight=loss_weights_tensor
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = (
        len(
            train_loader
        )
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

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    print()
    print(
        f"Train batches/epoch : "
        f"{len(train_loader):,}"
    )

    print(
        f"Total steps         : "
        f"{total_steps:,}"
    )

    print(
        f"Warmup steps        : "
        f"{warmup_steps:,}"
    )

    print(
        f"Mixed precision     : "
        f"{use_amp}"
    )

    # ========================================================
    # SAVE CONFIG
    # ========================================================

    save_json(
        {
            "stage":
                "07_train_mbert_title_lead",

            "task":
                "2A_clickbait_detection",

            "dataset":
                "viclickbait_v1",

            "model":
                MODEL_NAME,

            "input":
                "title_plus_lead",

            "truncation":
                "only_second",

            "max_length":
                MAX_LENGTH,

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

            "batch_size":
                args.batch_size,

            "epochs":
                args.epochs,

            "early_stopping_patience":
                args.patience,

            "learning_rate":
                args.lr,

            "weight_decay":
                args.weight_decay,

            "warmup_ratio":
                args.warmup_ratio,

            "seed":
                args.seed,

            "class_counts": {
                "normal":
                    int(
                        class_counts[
                            0
                        ]
                    ),

                "clickbait":
                    int(
                        class_counts[
                            1
                        ]
                    ),
            },

            "class_weights": {
                "normal":
                    float(
                        class_weights[
                            0
                        ]
                    ),

                "clickbait":
                    float(
                        class_weights[
                            1
                        ]
                    ),
            },

            "controlled_comparison_with_step06":
                True,
        },
        CONFIG_PATH,
    )

    # ========================================================
    # TRAIN
    # ========================================================

    print()
    print("=" * 80)
    print(
        "TRAIN"
    )
    print("=" * 80)

    history = []

    best_macro_f1 = -1.0
    best_val_loss = math.inf

    patience_counter = 0

    training_start = time.time()

    for epoch in range(
        1,
        args.epochs
        +
        1,
    ):

        epoch_start = time.time()

        model.train()

        running_loss = 0.0
        seen_examples = 0

        for batch in train_loader:

            batch.pop(
                "sample_ids"
            )

            labels = (
                batch.pop(
                    "labels"
                )
                .to(
                    device,
                    non_blocking=True,
                )
            )

            inputs = {
                key:
                    value.to(
                        device,
                        non_blocking=True,
                    )

                for key, value
                in batch.items()
            }

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):

                outputs = model(
                    **inputs
                )

                loss = loss_fn(
                    outputs.logits,
                    labels,
                )

            scaler.scale(
                loss
            ).backward()

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

            batch_size = int(
                labels.shape[
                    0
                ]
            )

            running_loss += (
                float(
                    loss.item()
                )
                *
                batch_size
            )

            seen_examples += (
                batch_size
            )

        train_loss = (
            running_loss
            /
            seen_examples
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        validation = evaluate(
            model=model,
            loader=validation_loader,
            device=device,
            loss_fn=loss_fn,
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
            epoch_start
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

                "validation_precision_clickbait_05":
                    float(
                        metrics_05[
                            "precision_clickbait"
                        ]
                    ),

                "validation_recall_clickbait_05":
                    float(
                        metrics_05[
                            "recall_clickbait"
                        ]
                    ),

                "validation_f1_clickbait_05":
                    float(
                        metrics_05[
                            "f1_clickbait"
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
            HISTORY_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        print()
        print(
            f"Epoch {epoch}/{args.epochs}"
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
            f"  CLICK F1   : "
            f"{metrics_05['f1_clickbait']:.6f}"
        )

        print(
            f"  ROC-AUC    : "
            f"{metrics_05['roc_auc']:.6f}"
        )

        print(
            f"  elapsed    : "
            f"{human_time(elapsed)}"
        )

        # ====================================================
        # SAME CHECKPOINT POLICY AS STEP 06
        # ====================================================

        current_macro = (
            metrics_05[
                "macro_f1"
            ]
        )

        current_val_loss = (
            validation[
                "loss"
            ]
        )

        improved = (
            current_macro
            >
            best_macro_f1
            +
            1e-12
        )

        tied_but_lower_loss = (
            abs(
                current_macro
                -
                best_macro_f1
            )
            <=
            1e-12
            and
            current_val_loss
            <
            best_val_loss
        )

        if (
            improved
            or
            tied_but_lower_loss
        ):

            best_macro_f1 = (
                current_macro
            )

            best_val_loss = (
                current_val_loss
            )

            patience_counter = 0

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
                            current_macro
                        ),

                    "validation_loss":
                        float(
                            current_val_loss
                        ),

                    "model_name":
                        MODEL_NAME,

                    "input":
                        "title_plus_lead",

                    "max_length":
                        MAX_LENGTH,

                    "truncation":
                        "only_second",
                },
                BEST_MODEL_PATH,
            )

            print(
                "  -> best checkpoint saved"
            )

        else:

            patience_counter += 1

            print(
                f"  no improvement "
                f"({patience_counter}/"
                f"{args.patience})"
            )

            if (
                patience_counter
                >=
                args.patience
            ):

                print()
                print(
                    "Early stopping triggered."
                )

                break

    total_training_time = (
        time.time()
        -
        training_start
    )

    # ========================================================
    # LOAD BEST
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

    print()
    print(
        f"Best epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Best validation Macro-F1 @0.5: "
        f"{checkpoint['validation_macro_f1_05']:.6f}"
    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    final_validation = evaluate(
        model=model,
        loader=validation_loader,
        device=device,
        loss_fn=loss_fn,
        use_amp=use_amp,
    )

    metrics_05 = compute_metrics(
        labels=(
            final_validation[
                "labels"
            ]
        ),
        probabilities=(
            final_validation[
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
            final_validation[
                "labels"
            ]
        ),
        probabilities=(
            final_validation[
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

    predictions = pd.DataFrame(
        {
            "sample_id":
                final_validation[
                    "sample_ids"
                ],

            "label":
                final_validation[
                    "labels"
                ],

            "prob_clickbait":
                final_validation[
                    "probabilities"
                ],
        }
    )

    predictions[
        "prediction_05"
    ] = (
        predictions[
            "prob_clickbait"
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
            "prob_clickbait"
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
    # SAVE METRICS
    # ========================================================

    save_json(
        {
            "stage":
                "07_train_mbert_title_lead",

            "task":
                "2A_clickbait_detection",

            "dataset":
                "viclickbait_v1",

            "model":
                MODEL_NAME,

            "input":
                "title_plus_lead",

            "truncation":
                "only_second",

            "test_used":
                False,

            "best_epoch":
                int(
                    checkpoint[
                        "epoch"
                    ]
                ),

            "total_training_seconds":
                float(
                    total_training_time
                ),

            "validation_loss":
                float(
                    final_validation[
                        "loss"
                    ]
                ),

            "metrics_at_0_5":
                metrics_05,

            "best_validation_threshold":
                float(
                    tuned_threshold
                ),

            "metrics_at_tuned_threshold":
                tuned_metrics,

            "controlled_comparison_step06":
                {
                    "same_model":
                        True,

                    "same_max_length":
                        True,

                    "same_batch_size":
                        True,

                    "same_learning_rate":
                        True,

                    "same_class_weight_policy":
                        True,

                    "same_checkpoint_policy":
                        True,

                    "changed_variable":
                        "lead_added_as_second_sequence",
                },

            "note":
                (
                    "Checkpoint selected using "
                    "validation Macro-F1 @0.5. "
                    "Threshold tuned on validation "
                    "after checkpoint selection. "
                    "TEST remains untouched."
                ),
        },
        METRICS_PATH,
    )

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("=" * 80)
    print(
        "STEP 07 VALIDATION RESULTS"
    )
    print("=" * 80)

    print()
    print(
        "Threshold = 0.50"
    )

    print(
        f"  Accuracy        : "
        f"{metrics_05['accuracy']:.6f}"
    )

    print(
        f"  Precision CLICK : "
        f"{metrics_05['precision_clickbait']:.6f}"
    )

    print(
        f"  Recall CLICK    : "
        f"{metrics_05['recall_clickbait']:.6f}"
    )

    print(
        f"  F1 CLICK        : "
        f"{metrics_05['f1_clickbait']:.6f}"
    )

    print(
        f"  Macro-F1        : "
        f"{metrics_05['macro_f1']:.6f}"
    )

    print(
        f"  ROC-AUC         : "
        f"{metrics_05['roc_auc']:.6f}"
    )

    print(
        f"  AP              : "
        f"{metrics_05['average_precision']:.6f}"
    )

    cm = (
        metrics_05[
            "confusion_matrix"
        ]
    )

    print()
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
        "Tuned validation threshold"
    )

    print(
        f"  threshold       : "
        f"{tuned_threshold:.2f}"
    )

    print(
        f"  Accuracy        : "
        f"{tuned_metrics['accuracy']:.6f}"
    )

    print(
        f"  Precision CLICK : "
        f"{tuned_metrics['precision_clickbait']:.6f}"
    )

    print(
        f"  Recall CLICK    : "
        f"{tuned_metrics['recall_clickbait']:.6f}"
    )

    print(
        f"  F1 CLICK        : "
        f"{tuned_metrics['f1_clickbait']:.6f}"
    )

    print(
        f"  Macro-F1        : "
        f"{tuned_metrics['macro_f1']:.6f}"
    )

    tuned_cm = (
        tuned_metrics[
            "confusion_matrix"
        ]
    )

    print()
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
        f"Total training time: "
        f"{human_time(total_training_time)}"
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
        "STEP 07 FINISHED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()