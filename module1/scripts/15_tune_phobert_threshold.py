from pathlib import Path
from collections import Counter
from contextlib import nullcontext
import csv
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
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

from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vn_newsspell_v1"
)

VALIDATION_PATH = (
    DATA_DIR
    / "validation.jsonl"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "module1_phobert_mil_full"
)

BEST_MODEL_PATH = (
    MODEL_DIR
    / "best_model.pt"
)

TOKENIZER_DIR = (
    MODEL_DIR
    / "tokenizer"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "module1_phobert_mil_full"
    / "threshold_tuning"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

PREDICTIONS_PATH = (
    RESULT_DIR
    / "validation_probabilities.csv"
)

THRESHOLD_SWEEP_PATH = (
    RESULT_DIR
    / "threshold_sweep.csv"
)

BEST_THRESHOLD_PATH = (
    RESULT_DIR
    / "best_threshold.json"
)


# ============================================================
# THRESHOLD CONFIG
# ============================================================

THRESHOLD_MIN = 0.10
THRESHOLD_MAX = 0.90
THRESHOLD_STEP = 0.01

RECALL_TARGETS = [
    0.90,
    0.92,
]


# ============================================================
# INFERENCE CONFIG
# ============================================================

NUM_WORKERS = 2
PREFETCH_FACTOR = 2

RANDOM_SEED = 20260918


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

USE_AMP = (
    DEVICE.type == "cuda"
)


# ============================================================
# CUDA OPTIMIZATION
# ============================================================

if torch.cuda.is_available():

    torch.backends.cuda.matmul.allow_tf32 = True

    torch.backends.cudnn.allow_tf32 = True

    torch.backends.cudnn.benchmark = True

    torch.set_float32_matmul_precision(
        "high"
    )


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint():

    print(
        f"\nLoading checkpoint:\n"
        f"{BEST_MODEL_PATH}"
    )

    try:

        checkpoint = torch.load(
            BEST_MODEL_PATH,
            map_location="cpu",
            weights_only=False
        )

    except TypeError:

        checkpoint = torch.load(
            BEST_MODEL_PATH,
            map_location="cpu"
        )

    required_keys = [
        "model_name",
        "text_field",
        "max_length",
        "stride",
        "max_chunks_per_document",
        "state_dict",
    ]

    missing = [
        key
        for key in required_keys
        if key not in checkpoint
    ]

    if missing:

        raise RuntimeError(
            "Checkpoint missing keys: "
            + ", ".join(missing)
        )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch')}"
    )

    print(
        f"Model name: "
        f"{checkpoint['model_name']}"
    )

    print(
        f"Text field: "
        f"{checkpoint['text_field']}"
    )

    print(
        f"MAX_LENGTH: "
        f"{checkpoint['max_length']}"
    )

    print(
        f"STRIDE: "
        f"{checkpoint['stride']}"
    )

    print(
        f"MAX_CHUNKS: "
        f"{checkpoint['max_chunks_per_document']}"
    )

    return checkpoint


# ============================================================
# LOAD VALIDATION RECORDS
# ============================================================

def load_records(
    path,
    text_field
):

    print(
        f"\nLoading validation:\n"
        f"{path}"
    )

    records = []

    label_counter = Counter()

    start_time = time.time()

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            row = json.loads(
                line
            )

            label = int(
                row["label"]
            )

            text = row.get(
                text_field,
                ""
            )

            if text is None:
                text = ""

            records.append({
                "text":
                    str(text),

                "label":
                    label,

                "source_id":
                    row.get(
                        "source_id",
                        ""
                    ),

                "variant_id":
                    row.get(
                        "variant_id",
                        ""
                    ),

                "severity":
                    row.get(
                        "severity",
                        "clean"
                    ),

                "error_count":
                    int(
                        row.get(
                            "error_count",
                            0
                        )
                    ),
            })

            label_counter[
                label
            ] += 1

    elapsed = (
        time.time()
        - start_time
    )

    print(
        f"Records: "
        f"{len(records):,}"
    )

    print(
        f"Clean  : "
        f"{label_counter[0]:,}"
    )

    print(
        f"Noisy  : "
        f"{label_counter[1]:,}"
    )

    print(
        f"Load time: "
        f"{elapsed:.2f}s"
    )

    return records


# ============================================================
# DATASET
# ============================================================

class NewsDataset(Dataset):

    def __init__(
        self,
        records
    ):

        self.records = records

    def __len__(self):

        return len(
            self.records
        )

    def __getitem__(
        self,
        index
    ):

        return self.records[
            index
        ]


# ============================================================
# CHUNK SELECTION
# ============================================================

def evenly_spaced_indices(
    total_chunks,
    max_chunks
):

    if total_chunks <= max_chunks:

        return list(
            range(
                total_chunks
            )
        )

    indices = np.linspace(
        0,
        total_chunks - 1,
        max_chunks,
        dtype=int
    )

    return sorted(
        set(
            indices.tolist()
        )
    )


# ============================================================
# COLLATOR
# ============================================================

class DocumentCollator:

    def __init__(
        self,
        tokenizer,
        max_length,
        stride,
        max_chunks
    ):

        self.tokenizer = tokenizer

        self.max_length = int(
            max_length
        )

        self.stride = int(
            stride
        )

        self.max_chunks = int(
            max_chunks
        )

        self.special_tokens = (
            tokenizer
            .num_special_tokens_to_add(
                pair=False
            )
        )

        self.content_length = (
            self.max_length
            - self.special_tokens
        )

        self.step_size = (
            self.content_length
            - self.stride
        )

        if self.step_size <= 0:

            raise ValueError(
                "STRIDE must be smaller "
                "than usable chunk length."
            )

    # ========================================================
    # CHUNK TOKEN IDS
    # ========================================================

    def chunk_token_ids(
        self,
        token_ids
    ):

        chunks = []

        if len(token_ids) == 0:

            chunks.append(
                self.tokenizer
                .build_inputs_with_special_tokens(
                    []
                )
            )

            return chunks

        start = 0

        while True:

            content = token_ids[
                start:
                start
                + self.content_length
            ]

            full_ids = (
                self.tokenizer
                .build_inputs_with_special_tokens(
                    content
                )
            )

            chunks.append(
                full_ids
            )

            if (
                start
                + self.content_length
                >= len(token_ids)
            ):

                break

            start += (
                self.step_size
            )

        selected_indices = (
            evenly_spaced_indices(
                len(chunks),
                self.max_chunks
            )
        )

        return [
            chunks[index]
            for index
            in selected_indices
        ]

    # ========================================================
    # COLLATE
    # ========================================================

    def __call__(
        self,
        batch
    ):

        texts = [
            item[
                "text"
            ]
            for item
            in batch
        ]

        encoded = self.tokenizer(
            texts,

            add_special_tokens=False,

            truncation=False,

            padding=False,

            return_attention_mask=False,
        )

        all_token_ids = encoded[
            "input_ids"
        ]

        chunk_features = []

        doc_indices = []

        labels = []

        metadata = []

        for doc_index, (
            item,
            token_ids
        ) in enumerate(
            zip(
                batch,
                all_token_ids
            )
        ):

            chunks = self.chunk_token_ids(
                token_ids
            )

            for chunk_ids in chunks:

                chunk_features.append({
                    "input_ids":
                        chunk_ids,

                    "attention_mask":
                        [1]
                        * len(
                            chunk_ids
                        ),
                })

                doc_indices.append(
                    doc_index
                )

            labels.append(
                item[
                    "label"
                ]
            )

            metadata.append({
                "source_id":
                    item[
                        "source_id"
                    ],

                "variant_id":
                    item[
                        "variant_id"
                    ],

                "severity":
                    item[
                        "severity"
                    ],

                "error_count":
                    item[
                        "error_count"
                    ],
            })

        padded = self.tokenizer.pad(
            chunk_features,

            padding=True,

            return_tensors="pt",
        )

        return {
            "input_ids":
                padded[
                    "input_ids"
                ],

            "attention_mask":
                padded[
                    "attention_mask"
                ],

            "doc_indices":
                torch.tensor(
                    doc_indices,
                    dtype=torch.long
                ),

            "labels":
                torch.tensor(
                    labels,
                    dtype=torch.float32
                ),

            "metadata":
                metadata,
        }


# ============================================================
# MODEL
# ============================================================

class PhoBERTMIL(nn.Module):

    def __init__(
        self,
        model_name
    ):

        super().__init__()

        # ----------------------------------------------------
        # Chỉ dựng architecture.
        #
        # Toàn bộ trained weights sẽ được load từ
        # best_model.pt ngay sau đó.
        # ----------------------------------------------------

        config = (
            AutoConfig
            .from_pretrained(
                model_name
            )
        )

        self.encoder = (
            AutoModel
            .from_config(
                config
            )
        )

        hidden_size = (
            self.encoder
            .config
            .hidden_size
        )

        self.dropout = nn.Dropout(
            0.1
        )

        self.classifier = nn.Linear(
            hidden_size,
            1
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        doc_indices,
        num_documents
    ):

        outputs = self.encoder(
            input_ids=
                input_ids,

            attention_mask=
                attention_mask,
        )

        # ====================================================
        # EXACTLY THE SAME AS TRAINING SCRIPT
        #
        # FIRST TOKEN REPRESENTATION
        # ====================================================

        cls_embeddings = (
            outputs
            .last_hidden_state[
                :,
                0,
                :
            ]
        )

        chunk_logits = (
            self.classifier(
                self.dropout(
                    cls_embeddings
                )
            )
            .squeeze(-1)
        )

        # ====================================================
        # MIL MAX POOLING
        # ====================================================

        document_logits = []

        for doc_index in range(
            num_documents
        ):

            mask = (
                doc_indices
                == doc_index
            )

            logits_for_doc = (
                chunk_logits[
                    mask
                ]
            )

            document_logits.append(
                torch.max(
                    logits_for_doc
                )
            )

        return torch.stack(
            document_logits
        )


# ============================================================
# AMP
# ============================================================

def autocast_context():

    if USE_AMP:

        return torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )

    return nullcontext()


# ============================================================
# INFERENCE
# ============================================================

def run_validation_inference(
    model,
    loader
):

    model.eval()

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    labels_all = []

    probabilities_all = []

    metadata_all = []

    total_loss = 0.0

    start_time = time.time()

    with torch.inference_mode():

        progress = tqdm(
            loader,
            desc="Validation inference"
        )

        for batch in progress:

            input_ids = (
                batch[
                    "input_ids"
                ]
                .to(
                    DEVICE,
                    non_blocking=True
                )
            )

            attention_mask = (
                batch[
                    "attention_mask"
                ]
                .to(
                    DEVICE,
                    non_blocking=True
                )
            )

            doc_indices = (
                batch[
                    "doc_indices"
                ]
                .to(
                    DEVICE,
                    non_blocking=True
                )
            )

            labels = (
                batch[
                    "labels"
                ]
                .to(
                    DEVICE,
                    non_blocking=True
                )
            )

            with autocast_context():

                logits = model(
                    input_ids=
                        input_ids,

                    attention_mask=
                        attention_mask,

                    doc_indices=
                        doc_indices,

                    num_documents=
                        len(labels),
                )

                loss = criterion(
                    logits,
                    labels
                )

            probabilities = (
                torch.sigmoid(
                    logits
                )
                .float()
                .cpu()
                .numpy()
            )

            total_loss += (
                loss.item()
            )

            labels_all.extend(
                labels
                .cpu()
                .numpy()
                .astype(
                    np.int64
                )
                .tolist()
            )

            probabilities_all.extend(
                probabilities.tolist()
            )

            metadata_all.extend(
                batch[
                    "metadata"
                ]
            )

    elapsed = (
        time.time()
        - start_time
    )

    average_loss = (
        total_loss
        / max(
            1,
            len(loader)
        )
    )

    print(
        f"\nInference time: "
        f"{elapsed:.2f}s"
    )

    print(
        f"Validation loss: "
        f"{average_loss:.6f}"
    )

    return (
        np.asarray(
            labels_all,
            dtype=np.int64
        ),
        np.asarray(
            probabilities_all,
            dtype=np.float64
        ),
        metadata_all,
        float(
            average_loss
        ),
        float(
            elapsed
        ),
    )


# ============================================================
# THRESHOLD METRICS
# ============================================================

def calculate_threshold_metrics(
    labels,
    probabilities,
    metadata,
    threshold
):

    predictions = (
        probabilities
        >= threshold
    ).astype(
        np.int64
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        pos_label=1,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        pos_label=1,
        zero_division=0
    )

    error_f1 = f1_score(
        labels,
        predictions,
        pos_label=1,
        zero_division=0
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    cm = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    )

    tn = int(
        cm[0, 0]
    )

    fp = int(
        cm[0, 1]
    )

    fn = int(
        cm[1, 0]
    )

    tp = int(
        cm[1, 1]
    )

    specificity = (
        tn
        / (tn + fp)
        if (
            tn + fp
        ) > 0
        else 0.0
    )

    severity_recall = {}

    for severity in [
        "minor",
        "moderate",
        "severe",
    ]:

        indices = [
            index
            for index, item
            in enumerate(
                metadata
            )
            if (
                labels[index] == 1
                and
                item[
                    "severity"
                ]
                == severity
            )
        ]

        if not indices:

            severity_recall[
                severity
            ] = None

            continue

        severity_recall[
            severity
        ] = float(
            np.mean(
                predictions[
                    indices
                ]
                == 1
            )
        )

    return {
        "threshold":
            float(
                threshold
            ),

        "accuracy":
            float(
                accuracy
            ),

        "error_precision":
            float(
                precision
            ),

        "error_recall":
            float(
                recall
            ),

        "error_f1":
            float(
                error_f1
            ),

        "macro_f1":
            float(
                macro_f1
            ),

        "specificity":
            float(
                specificity
            ),

        "tn":
            tn,

        "fp":
            fp,

        "fn":
            fn,

        "tp":
            tp,

        "minor_recall":
            severity_recall[
                "minor"
            ],

        "moderate_recall":
            severity_recall[
                "moderate"
            ],

        "severe_recall":
            severity_recall[
                "severe"
            ],
    }


# ============================================================
# BUILD THRESHOLDS
# ============================================================

def build_thresholds():

    thresholds = np.arange(
        THRESHOLD_MIN,
        THRESHOLD_MAX
        + THRESHOLD_STEP / 2,
        THRESHOLD_STEP
    )

    thresholds = np.round(
        thresholds,
        4
    )

    # Luôn đảm bảo 0.5 có mặt.
    thresholds = np.unique(
        np.append(
            thresholds,
            0.5
        )
    )

    return thresholds


# ============================================================
# SWEEP
# ============================================================

def sweep_thresholds(
    labels,
    probabilities,
    metadata
):

    print("\n" + "=" * 80)
    print("THRESHOLD SWEEP")
    print("=" * 80)

    thresholds = (
        build_thresholds()
    )

    print(
        f"\nThreshold range: "
        f"{thresholds[0]:.2f} "
        f"-> "
        f"{thresholds[-1]:.2f}"
    )

    print(
        f"Number of thresholds: "
        f"{len(thresholds):,}"
    )

    rows = []

    for threshold in tqdm(
        thresholds,
        desc="Threshold sweep"
    ):

        metrics = (
            calculate_threshold_metrics(
                labels,
                probabilities,
                metadata,
                threshold
            )
        )

        rows.append(
            metrics
        )

    return rows


# ============================================================
# SAVE PROBABILITIES
# ============================================================

def save_validation_probabilities(
    labels,
    probabilities,
    metadata
):

    with open(
        PREDICTIONS_PATH,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        fieldnames = [
            "source_id",
            "variant_id",
            "label",
            "severity",
            "error_count",
            "probability_error",
            "prediction_at_0_5",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for (
            label,
            probability,
            item
        ) in zip(
            labels,
            probabilities,
            metadata
        ):

            writer.writerow({
                "source_id":
                    item[
                        "source_id"
                    ],

                "variant_id":
                    item[
                        "variant_id"
                    ],

                "label":
                    int(
                        label
                    ),

                "severity":
                    item[
                        "severity"
                    ],

                "error_count":
                    item[
                        "error_count"
                    ],

                "probability_error":
                    float(
                        probability
                    ),

                "prediction_at_0_5":
                    int(
                        probability
                        >= 0.5
                    ),
            })


# ============================================================
# SAVE SWEEP CSV
# ============================================================

def save_threshold_sweep(
    rows
):

    fieldnames = [
        "threshold",
        "accuracy",
        "error_precision",
        "error_recall",
        "error_f1",
        "macro_f1",
        "specificity",
        "tn",
        "fp",
        "fn",
        "tp",
        "minor_recall",
        "moderate_recall",
        "severe_recall",
    ]

    with open(
        THRESHOLD_SWEEP_PATH,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                row
            )


# ============================================================
# SELECT BEST THRESHOLDS
# ============================================================

def select_best_thresholds(
    rows
):

    default_metrics = min(
        rows,
        key=lambda row:
            abs(
                row["threshold"]
                - 0.5
            )
    )

    best_macro_f1 = max(
        rows,
        key=lambda row: (
            row[
                "macro_f1"
            ],
            row[
                "error_f1"
            ],
            row[
                "error_recall"
            ],
            -abs(
                row[
                    "threshold"
                ]
                - 0.5
            ),
        )
    )

    best_error_f1 = max(
        rows,
        key=lambda row: (
            row[
                "error_f1"
            ],
            row[
                "macro_f1"
            ],
            row[
                "error_recall"
            ],
        )
    )

    recall_operating_points = {}

    for target in RECALL_TARGETS:

        candidates = [
            row
            for row in rows
            if (
                row[
                    "error_recall"
                ]
                >= target
            )
        ]

        key_name = (
            f"recall_at_least_"
            f"{int(target * 100)}"
        )

        if not candidates:

            recall_operating_points[
                key_name
            ] = None

            continue

        # ----------------------------------------------------
        # Trong các threshold đạt recall target:
        #
        # ưu tiên precision cao nhất.
        #
        # Điều này tránh hạ threshold quá mức cần thiết.
        # ----------------------------------------------------

        selected = max(
            candidates,
            key=lambda row: (
                row[
                    "error_precision"
                ],
                row[
                    "error_f1"
                ],
                row[
                    "macro_f1"
                ],
                row[
                    "threshold"
                ],
            )
        )

        recall_operating_points[
            key_name
        ] = selected

    return {
        "default_threshold_0_5":
            default_metrics,

        "best_macro_f1":
            best_macro_f1,

        "best_error_f1":
            best_error_f1,

        **recall_operating_points,
    }


# ============================================================
# PRINT ONE OPERATING POINT
# ============================================================

def print_operating_point(
    title,
    metrics
):

    print("\n" + "-" * 80)

    print(
        title
    )

    print("-" * 80)

    if metrics is None:

        print(
            "No threshold in current sweep "
            "satisfies this criterion."
        )

        return

    print(
        f"Threshold       : "
        f"{metrics['threshold']:.4f}"
    )

    print(
        f"Accuracy        : "
        f"{metrics['accuracy']:.6f}"
    )

    print(
        f"ERROR Precision : "
        f"{metrics['error_precision']:.6f}"
    )

    print(
        f"ERROR Recall    : "
        f"{metrics['error_recall']:.6f}"
    )

    print(
        f"ERROR F1        : "
        f"{metrics['error_f1']:.6f}"
    )

    print(
        f"Macro F1        : "
        f"{metrics['macro_f1']:.6f}"
    )

    print(
        f"Specificity     : "
        f"{metrics['specificity']:.6f}"
    )

    print(
        "\nConfusion Matrix"
    )

    print(
        f"TN: {metrics['tn']:,}"
    )

    print(
        f"FP: {metrics['fp']:,}"
    )

    print(
        f"FN: {metrics['fn']:,}"
    )

    print(
        f"TP: {metrics['tp']:,}"
    )

    print(
        "\nRecall by severity"
    )

    for (
        severity,
        key
    ) in [
        (
            "minor",
            "minor_recall"
        ),
        (
            "moderate",
            "moderate_recall"
        ),
        (
            "severe",
            "severe_recall"
        ),
    ]:

        value = metrics[
            key
        ]

        if value is None:

            print(
                f"  {severity:10s}: N/A"
            )

        else:

            print(
                f"  {severity:10s}: "
                f"{value:.6f}"
            )


# ============================================================
# SANITY CHECK AGAINST TRAINING CHECKPOINT
# ============================================================

def compare_with_checkpoint(
    checkpoint,
    default_metrics
):

    original_metrics = (
        checkpoint.get(
            "metrics",
            {}
        )
    )

    original_macro_f1 = (
        original_metrics.get(
            "macro_f1"
        )
    )

    if original_macro_f1 is None:
        return

    current_macro_f1 = (
        default_metrics[
            "macro_f1"
        ]
    )

    delta = (
        current_macro_f1
        - float(
            original_macro_f1
        )
    )

    print("\n" + "=" * 80)
    print("CHECKPOINT REPRODUCTION CHECK")
    print("=" * 80)

    print(
        f"\nTraining-time Macro-F1 @0.5 : "
        f"{float(original_macro_f1):.6f}"
    )

    print(
        f"Current Macro-F1 @0.5       : "
        f"{current_macro_f1:.6f}"
    )

    print(
        f"Difference                  : "
        f"{delta:+.8f}"
    )

    if abs(delta) <= 0.001:

        print(
            "\nReproduction check: PASSED"
        )

    else:

        print(
            "\nWARNING: Validation result differs "
            "noticeably from training-time metrics."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("VN-NEWSSPELL - PHOBERT MIL THRESHOLD TUNING")
    print("=" * 80)

    print(
        f"\nDevice: "
        f"{DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Validation set ONLY."
    )

    print(
        "Test set is NOT loaded or used."
    )

    set_seed(
        RANDOM_SEED
    )

    # ========================================================
    # CHECKPOINT
    # ========================================================

    checkpoint = (
        load_checkpoint()
    )

    model_name = checkpoint[
        "model_name"
    ]

    text_field = checkpoint[
        "text_field"
    ]

    max_length = int(
        checkpoint[
            "max_length"
        ]
    )

    stride = int(
        checkpoint[
            "stride"
        ]
    )

    max_chunks = int(
        checkpoint[
            "max_chunks_per_document"
        ]
    )

    validation_batch_size = int(
        checkpoint.get(
            "validation_document_batch_size",
            16
        )
    )

    # ========================================================
    # DATA
    # ========================================================

    records = load_records(
        VALIDATION_PATH,
        text_field
    )

    dataset = NewsDataset(
        records
    )

    # ========================================================
    # TOKENIZER
    # ========================================================

    print(
        "\nLoading saved tokenizer..."
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            TOKENIZER_DIR,
            use_fast=False
        )
    )

    print(
        f"Tokenizer fast: "
        f"{tokenizer.is_fast}"
    )

    collator = DocumentCollator(
        tokenizer=
            tokenizer,

        max_length=
            max_length,

        stride=
            stride,

        max_chunks=
            max_chunks,
    )

    # ========================================================
    # LOADER
    # ========================================================

    loader_kwargs = {
        "batch_size":
            validation_batch_size,

        "shuffle":
            False,

        "collate_fn":
            collator,

        "num_workers":
            NUM_WORKERS,

        "pin_memory":
            USE_AMP,
    }

    if NUM_WORKERS > 0:

        loader_kwargs[
            "persistent_workers"
        ] = True

        loader_kwargs[
            "prefetch_factor"
        ] = PREFETCH_FACTOR

    loader = DataLoader(
        dataset,
        **loader_kwargs
    )

    print(
        f"\nValidation batch size: "
        f"{validation_batch_size}"
    )

    print(
        f"Validation batches: "
        f"{len(loader):,}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    print(
        "\nBuilding PhoBERT MIL architecture..."
    )

    model = PhoBERTMIL(
        model_name
    )

    print(
        "\nLoading trained state_dict..."
    )

    load_result = (
        model.load_state_dict(
            checkpoint[
                "state_dict"
            ],
            strict=True
        )
    )

    print(
        f"Missing keys    : "
        f"{len(load_result.missing_keys)}"
    )

    print(
        f"Unexpected keys : "
        f"{len(load_result.unexpected_keys)}"
    )

    model.to(
        DEVICE
    )

    model.eval()

    # ========================================================
    # RUN INFERENCE ONCE
    # ========================================================

    (
        labels,
        probabilities,
        metadata,
        validation_loss,
        inference_seconds,
    ) = run_validation_inference(
        model,
        loader
    )

    # ========================================================
    # GLOBAL RANKING METRICS
    # ========================================================

    roc_auc = roc_auc_score(
        labels,
        probabilities
    )

    average_precision = (
        average_precision_score(
            labels,
            probabilities
        )
    )

    print("\n" + "=" * 80)
    print("THRESHOLD-INDEPENDENT METRICS")
    print("=" * 80)

    print(
        f"\nROC-AUC          : "
        f"{roc_auc:.6f}"
    )

    print(
        f"Average Precision: "
        f"{average_precision:.6f}"
    )

    # ========================================================
    # SAVE PROBABILITIES
    # ========================================================

    save_validation_probabilities(
        labels,
        probabilities,
        metadata
    )

    # ========================================================
    # THRESHOLD SWEEP
    # ========================================================

    rows = sweep_thresholds(
        labels,
        probabilities,
        metadata
    )

    save_threshold_sweep(
        rows
    )

    selections = (
        select_best_thresholds(
            rows
        )
    )

    # ========================================================
    # REPRODUCTION CHECK
    # ========================================================

    compare_with_checkpoint(
        checkpoint,
        selections[
            "default_threshold_0_5"
        ]
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n" + "=" * 80)
    print("SELECTED OPERATING POINTS")
    print("=" * 80)

    print_operating_point(
        "DEFAULT THRESHOLD = 0.50",
        selections[
            "default_threshold_0_5"
        ]
    )

    print_operating_point(
        "BEST VALIDATION MACRO-F1",
        selections[
            "best_macro_f1"
        ]
    )

    print_operating_point(
        "BEST VALIDATION ERROR-F1",
        selections[
            "best_error_f1"
        ]
    )

    print_operating_point(
        "RECALL >= 0.90 WITH HIGHEST PRECISION",
        selections[
            "recall_at_least_90"
        ]
    )

    print_operating_point(
        "RECALL >= 0.92 WITH HIGHEST PRECISION",
        selections[
            "recall_at_least_92"
        ]
    )

    # ========================================================
    # PRIMARY THRESHOLD
    # ========================================================
    #
    # Dùng best Macro-F1 làm threshold mặc định
    # cho evaluation chính thức.
    #
    # Đồng thời lưu các operating point khác để
    # sau này có thể phân tích precision/recall trade-off.
    # ========================================================

    primary = selections[
        "best_macro_f1"
    ]

    output = {
        "model":
            model_name,

        "checkpoint":
            str(
                BEST_MODEL_PATH
            ),

        "checkpoint_epoch":
            checkpoint.get(
                "epoch"
            ),

        "text_field":
            text_field,

        "max_length":
            max_length,

        "stride":
            stride,

        "max_chunks_per_document":
            max_chunks,

        "validation_records":
            int(
                len(labels)
            ),

        "validation_loss":
            validation_loss,

        "inference_seconds":
            inference_seconds,

        "roc_auc":
            float(
                roc_auc
            ),

        "average_precision":
            float(
                average_precision
            ),

        "threshold_sweep": {
            "minimum":
                THRESHOLD_MIN,

            "maximum":
                THRESHOLD_MAX,

            "step":
                THRESHOLD_STEP,
        },

        "primary_selection_rule":
            "maximum_validation_macro_f1",

        "selected_threshold":
            float(
                primary[
                    "threshold"
                ]
            ),

        "selected_metrics":
            primary,

        "operating_points":
            selections,

        "test_set_used":
            False,
    }

    with open(
        BEST_THRESHOLD_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ========================================================
    # FINISH
    # ========================================================

    print("\n" + "=" * 80)
    print("THRESHOLD TUNING FINISHED")
    print("=" * 80)

    print(
        f"\nPRIMARY THRESHOLD: "
        f"{primary['threshold']:.4f}"
    )

    print(
        "Selection rule: "
        "maximum validation Macro-F1"
    )

    print(
        f"\nValidation probabilities:\n"
        f"{PREDICTIONS_PATH}"
    )

    print(
        f"\nThreshold sweep:\n"
        f"{THRESHOLD_SWEEP_PATH}"
    )

    print(
        f"\nBest threshold:\n"
        f"{BEST_THRESHOLD_PATH}"
    )

    print(
        "\nTest set was NOT loaded or used."
    )


if __name__ == "__main__":
    main()