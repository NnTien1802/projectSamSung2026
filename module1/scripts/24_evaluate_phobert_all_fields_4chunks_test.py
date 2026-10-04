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

TEST_PATH = (
    DATA_DIR
    / "test.jsonl"
)


MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "module1_phobert_mil_all_fields_4chunks"
)

BEST_MODEL_PATH = (
    MODEL_DIR
    / "best_model.pt"
)

TOKENIZER_DIR = (
    MODEL_DIR
    / "tokenizer"
)


THRESHOLD_PATH = (
    PROJECT_ROOT
    / "results"
    / "module1_phobert_mil_all_fields_4chunks"
    / "threshold_tuning"
    / "best_threshold.json"
)


RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "module1_phobert_mil_all_fields_4chunks"
    / "test_evaluation"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

TEST_METRICS_PATH = (
    RESULT_DIR
    / "official_test_metrics.json"
)

TEST_PREDICTIONS_PATH = (
    RESULT_DIR
    / "test_predictions.csv"
)

ERROR_ANALYSIS_PATH = (
    RESULT_DIR
    / "test_error_analysis.json"
)


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
        "input_mode",
        "text_fields",
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
        f"Model: "
        f"{checkpoint['model_name']}"
    )

    print(
        f"Input mode: "
        f"{checkpoint['input_mode']}"
    )

    print(
        "Text fields:"
    )

    for field in checkpoint[
        "text_fields"
    ]:

        print(
            f"  - {field}"
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
# LOAD LOCKED THRESHOLD
# ============================================================

def load_locked_threshold():

    print(
        f"\nLoading locked threshold:\n"
        f"{THRESHOLD_PATH}"
    )

    with open(
        THRESHOLD_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    if "selected_threshold" not in data:

        raise RuntimeError(
            "selected_threshold not found."
        )

    threshold = float(
        data[
            "selected_threshold"
        ]
    )

    rule = data.get(
        "primary_selection_rule",
        "unknown"
    )

    print(
        f"Locked threshold: "
        f"{threshold:.4f}"
    )

    print(
        f"Selection rule: "
        f"{rule}"
    )

    print(
        "Threshold was selected using "
        "VALIDATION only."
    )

    return (
        threshold,
        data
    )


# ============================================================
# BUILD DOCUMENT TEXT
# ============================================================

def build_document_text(
    row,
    text_fields,
    field_separator
):

    parts = []

    for field in text_fields:

        text = row.get(
            field,
            ""
        )

        if text is None:
            text = ""

        text = str(
            text
        ).strip()

        if text:

            parts.append(
                text
            )

    return field_separator.join(
        parts
    )


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_records(
    path,
    text_fields,
    field_separator
):

    print(
        f"\nLoading TEST set:\n"
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
                row[
                    "label"
                ]
            )

            text = build_document_text(
                row,
                text_fields,
                field_separator
            )

            records.append({
                "text":
                    text,

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

                "error_types":
                    row.get(
                        "error_types",
                        []
                    ),

                "error_sources":
                    row.get(
                        "error_sources",
                        []
                    ),

                "errors":
                    row.get(
                        "errors",
                        []
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

        all_token_ids = (
            encoded[
                "input_ids"
            ]
        )

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

            chunks = (
                self.chunk_token_ids(
                    token_ids
                )
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

                "error_types":
                    item[
                        "error_types"
                    ],

                "error_sources":
                    item[
                        "error_sources"
                    ],

                "errors":
                    item[
                        "errors"
                    ],
            })

        padded = (
            self.tokenizer.pad(
                chunk_features,
                padding=True,
                return_tensors="pt",
            )
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
# TEST INFERENCE
# ============================================================

def run_test_inference(
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
            desc="TEST inference"
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

            total_loss += (
                loss.item()
            )

            probabilities = (
                torch.sigmoid(
                    logits
                )
                .float()
                .cpu()
                .numpy()
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
        f"\nTest inference time: "
        f"{elapsed:.2f}s"
    )

    print(
        f"Test BCE loss: "
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
# METRICS
# ============================================================

def calculate_metrics(
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

    severity_results = {}

    for severity in [
        "minor",
        "moderate",
        "severe",
    ]:

        indices = [
            i
            for i, item
            in enumerate(
                metadata
            )
            if (
                labels[i] == 1
                and
                item[
                    "severity"
                ]
                == severity
            )
        ]

        if not indices:

            severity_results[
                severity
            ] = {
                "count": 0,
                "recall": None,
            }

            continue

        severity_results[
            severity
        ] = {
            "count":
                len(indices),

            "recall":
                float(
                    np.mean(
                        predictions[
                            indices
                        ]
                        == 1
                    )
                ),
        }

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

        "roc_auc":
            float(
                roc_auc
            ),

        "average_precision":
            float(
                average_precision
            ),

        "confusion_matrix": {
            "tn":
                tn,

            "fp":
                fp,

            "fn":
                fn,

            "tp":
                tp,
        },

        "severity_recall":
            severity_results,
    }


# ============================================================
# ERROR ANALYSIS
# ============================================================

def build_error_analysis(
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

    false_negative_indices = np.where(
        (labels == 1)
        &
        (predictions == 0)
    )[0]

    false_positive_indices = np.where(
        (labels == 0)
        &
        (predictions == 1)
    )[0]

    fn_severity = Counter()

    fn_error_count = Counter()

    fn_error_types = Counter()

    fn_error_sources = Counter()

    fn_error_fields = Counter()

    for index in false_negative_indices:

        item = metadata[
            index
        ]

        fn_severity[
            item[
                "severity"
            ]
        ] += 1

        fn_error_count[
            item[
                "error_count"
            ]
        ] += 1

        for error_type in (
            item[
                "error_types"
            ]
        ):

            fn_error_types[
                error_type
            ] += 1

        for error_source in (
            item[
                "error_sources"
            ]
        ):

            fn_error_sources[
                error_source
            ] += 1

        fields_seen = set()

        for error in item[
            "errors"
        ]:

            field = error.get(
                "field",
                "unknown"
            )

            fields_seen.add(
                field
            )

        for field in fields_seen:

            fn_error_fields[
                field
            ] += 1

    clean_probabilities = (
        probabilities[
            labels == 0
        ]
    )

    noisy_probabilities = (
        probabilities[
            labels == 1
        ]
    )

    probability_stats = {
        "clean": {
            "mean":
                float(
                    np.mean(
                        clean_probabilities
                    )
                ),

            "median":
                float(
                    np.median(
                        clean_probabilities
                    )
                ),

            "p95":
                float(
                    np.percentile(
                        clean_probabilities,
                        95
                    )
                ),
        },

        "noisy": {
            "mean":
                float(
                    np.mean(
                        noisy_probabilities
                    )
                ),

            "median":
                float(
                    np.median(
                        noisy_probabilities
                    )
                ),

            "p05":
                float(
                    np.percentile(
                        noisy_probabilities,
                        5
                    )
                ),
        },
    }

    return {
        "threshold":
            float(
                threshold
            ),

        "false_negative_count":
            int(
                len(
                    false_negative_indices
                )
            ),

        "false_positive_count":
            int(
                len(
                    false_positive_indices
                )
            ),

        "false_negative_by_severity":
            dict(
                fn_severity
            ),

        "false_negative_by_error_count":
            {
                str(key):
                    value
                for key, value
                in sorted(
                    fn_error_count.items()
                )
            },

        "false_negative_fields":
            dict(
                fn_error_fields
            ),

        "false_negative_error_types":
            dict(
                fn_error_types.most_common()
            ),

        "false_negative_error_sources":
            dict(
                fn_error_sources
            ),

        "probability_statistics":
            probability_stats,
    }


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
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

    with open(
        TEST_PREDICTIONS_PATH,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        fieldnames = [
            "source_id",
            "variant_id",
            "label",
            "prediction",
            "probability_error",
            "threshold",
            "correct",
            "severity",
            "error_count",
            "error_types",
            "error_sources",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for (
            label,
            prediction,
            probability,
            item
        ) in zip(
            labels,
            predictions,
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

                "prediction":
                    int(
                        prediction
                    ),

                "probability_error":
                    float(
                        probability
                    ),

                "threshold":
                    float(
                        threshold
                    ),

                "correct":
                    int(
                        prediction
                        == label
                    ),

                "severity":
                    item[
                        "severity"
                    ],

                "error_count":
                    item[
                        "error_count"
                    ],

                "error_types":
                    "|".join(
                        item[
                            "error_types"
                        ]
                    ),

                "error_sources":
                    "|".join(
                        item[
                            "error_sources"
                        ]
                    ),
            })


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(
    metrics
):

    print("\n" + "=" * 80)
    print("OFFICIAL ALL-FIELDS TEST RESULTS")
    print("=" * 80)

    print(
        f"\nLocked threshold : "
        f"{metrics['threshold']:.4f}"
    )

    print(
        f"Accuracy         : "
        f"{metrics['accuracy']:.6f}"
    )

    print(
        f"ERROR Precision  : "
        f"{metrics['error_precision']:.6f}"
    )

    print(
        f"ERROR Recall     : "
        f"{metrics['error_recall']:.6f}"
    )

    print(
        f"ERROR F1         : "
        f"{metrics['error_f1']:.6f}"
    )

    print(
        f"Macro F1         : "
        f"{metrics['macro_f1']:.6f}"
    )

    print(
        f"Specificity      : "
        f"{metrics['specificity']:.6f}"
    )

    print(
        f"ROC-AUC          : "
        f"{metrics['roc_auc']:.6f}"
    )

    print(
        f"Average Precision: "
        f"{metrics['average_precision']:.6f}"
    )

    cm = metrics[
        "confusion_matrix"
    ]

    print(
        "\nConfusion Matrix"
    )

    print(
        f"TN: {cm['tn']:,}"
    )

    print(
        f"FP: {cm['fp']:,}"
    )

    print(
        f"FN: {cm['fn']:,}"
    )

    print(
        f"TP: {cm['tp']:,}"
    )

    print(
        "\nRecall by severity"
    )

    for severity, values in (
        metrics[
            "severity_recall"
        ].items()
    ):

        recall = values[
            "recall"
        ]

        count = values[
            "count"
        ]

        if recall is None:

            print(
                f"  {severity:10s}: N/A"
            )

        else:

            print(
                f"  {severity:10s}: "
                f"{recall:.6f} "
                f"(n={count:,})"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
    "VN-NEWSSPELL - PHOBERT MIL ALL-FIELDS "
    "4-CHUNKS OFFICIAL TEST"
)
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
        "Threshold is LOCKED from validation."
    )

    print(
        "No threshold tuning is performed "
        "on test."
    )

    set_seed(
        RANDOM_SEED
    )

    # ========================================================
    # CHECKPOINT + THRESHOLD
    # ========================================================

    checkpoint = (
        load_checkpoint()
    )

    (
        threshold,
        threshold_metadata
    ) = load_locked_threshold()

    model_name = checkpoint[
        "model_name"
    ]

    input_mode = checkpoint[
        "input_mode"
    ]

    text_fields = checkpoint[
        "text_fields"
    ]

    field_separator = checkpoint.get(
        "field_separator",
        " . "
    )

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

    batch_size = int(
        checkpoint.get(
            "validation_document_batch_size",
            16
        )
    )

    # ========================================================
    # THRESHOLD SAFETY CHECK
    # ========================================================

    selection_rule = (
        threshold_metadata.get(
            "primary_selection_rule"
        )
    )

    if (
        selection_rule
        != "maximum_validation_macro_f1"
    ):

        raise RuntimeError(
            "Unexpected threshold "
            "selection rule."
        )

    print(
        f"\nInput mode confirmed: "
        f"{input_mode}"
    )

    # ========================================================
    # LOAD TEST
    # ========================================================

    records = load_records(
        TEST_PATH,
        text_fields,
        field_separator
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
            batch_size,

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
        f"\nTest batch size: "
        f"{batch_size}"
    )

    print(
        f"Test batches: "
        f"{len(loader):,}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    print(
        "\nBuilding model architecture..."
    )

    model = PhoBERTMIL(
        model_name
    )

    print(
        "Loading trained state_dict..."
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

    if (
        load_result.missing_keys
        or load_result.unexpected_keys
    ):

        raise RuntimeError(
            "Checkpoint state_dict mismatch."
        )

    model.to(
        DEVICE
    )

    model.eval()

    # ========================================================
    # TEST INFERENCE
    # ========================================================

    (
        labels,
        probabilities,
        metadata,
        test_loss,
        inference_seconds,
    ) = run_test_inference(
        model,
        loader
    )

    # ========================================================
    # METRICS
    # ========================================================

    metrics = calculate_metrics(
        labels,
        probabilities,
        metadata,
        threshold
    )

    metrics[
        "test_loss"
    ] = test_loss

    metrics[
        "inference_seconds"
    ] = inference_seconds

    metrics[
        "test_records"
    ] = int(
        len(labels)
    )

    metrics[
        "clean_records"
    ] = int(
        np.sum(
            labels == 0
        )
    )

    metrics[
        "noisy_records"
    ] = int(
        np.sum(
            labels == 1
        )
    )

    metrics[
        "model_name"
    ] = model_name

    metrics[
        "checkpoint_epoch"
    ] = checkpoint.get(
        "epoch"
    )

    metrics[
        "input_mode"
    ] = input_mode

    metrics[
        "text_fields"
    ] = text_fields

    metrics[
        "field_separator"
    ] = field_separator

    metrics[
        "max_length"
    ] = max_length

    metrics[
        "stride"
    ] = stride

    metrics[
        "max_chunks_per_document"
    ] = max_chunks

    metrics[
        "threshold_selection_rule"
    ] = selection_rule

    metrics[
        "threshold_selected_on"
    ] = "validation"

    metrics[
        "threshold_tuned_on_test"
    ] = False

    # ========================================================
    # ERROR ANALYSIS
    # ========================================================

    error_analysis = build_error_analysis(
        labels,
        probabilities,
        metadata,
        threshold
    )

    # ========================================================
    # SAVE
    # ========================================================

    save_predictions(
        labels,
        probabilities,
        metadata,
        threshold
    )

    with open(
        TEST_METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=2
        )

    with open(
        ERROR_ANALYSIS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            error_analysis,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ========================================================
    # PRINT
    # ========================================================

    print_metrics(
        metrics
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "ERROR ANALYSIS SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        f"\nFalse positives: "
        f"{error_analysis['false_positive_count']:,}"
    )

    print(
        f"False negatives: "
        f"{error_analysis['false_negative_count']:,}"
    )

    print(
        "\nFalse negatives by severity:"
    )

    for severity, count in (
        error_analysis[
            "false_negative_by_severity"
        ].items()
    ):

        print(
            f"  {severity:10s}: "
            f"{count:,}"
        )

    print(
        "\nFalse negatives by error count:"
    )

    for error_count, count in (
        error_analysis[
            "false_negative_by_error_count"
        ].items()
    ):

        print(
            f"  {error_count:>3s} errors: "
            f"{count:,}"
        )

    print(
        "\nFalse-negative fields:"
    )

    for field, count in (
        error_analysis[
            "false_negative_fields"
        ].items()
    ):

        print(
            f"  {field:10s}: "
            f"{count:,}"
        )

    print(
        "\nTop false-negative error types:"
    )

    for error_type, count in list(
        error_analysis[
            "false_negative_error_types"
        ].items()
    )[:10]:

        print(
            f"  {error_type:25s}: "
            f"{count:,}"
        )

    # ========================================================
    # FINISH
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "OFFICIAL ALL-FIELDS TEST "
        "EVALUATION FINISHED"
    )

    print(
        "=" * 80
    )

    print(
        f"\nLocked threshold: "
        f"{threshold:.4f}"
    )

    print(
        "\nFiles:"
    )

    print(
        f"  Metrics:\n"
        f"  {TEST_METRICS_PATH}"
    )

    print(
        f"\n  Predictions:\n"
        f"  {TEST_PREDICTIONS_PATH}"
    )

    print(
        f"\n  Error analysis:\n"
        f"  {ERROR_ANALYSIS_PATH}"
    )

    print(
        "\nTest threshold was NOT tuned."
    )


if __name__ == "__main__":
    main()