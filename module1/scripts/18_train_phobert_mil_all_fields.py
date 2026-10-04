from pathlib import Path
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
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
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vn_newsspell_v1"
)

TRAIN_PATH = (
    DATA_DIR
    / "train.jsonl"
)

VALIDATION_PATH = (
    DATA_DIR
    / "validation.jsonl"
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "vinai/phobert-base-v2"


# ============================================================
# INPUT
# ============================================================

# Experiment 2:
#
# Không chỉ đọc article.
#
# Ghép:
#   title
#   abstract
#   article
#
# Mục tiêu:
# model thực sự nhìn thấy tất cả field mà generator
# có thể chèn lỗi vào.
TEXT_FIELDS = [
    "title_segmented",
    "abstract_segmented",
    "article_segmented",
]

INPUT_MODE = "title_abstract_article"

# Không thêm special token mới vào vocabulary.
# Dùng dấu chấm để ngăn cách các field.
FIELD_SEPARATOR = " . "


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "module1_phobert_mil_all_fields"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "module1_phobert_mil_all_fields"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL_PATH = (
    OUTPUT_DIR
    / "best_model.pt"
)

METRICS_PATH = (
    RESULT_DIR
    / "best_validation_metrics.json"
)

HISTORY_PATH = (
    RESULT_DIR
    / "training_history.json"
)

CONFIG_PATH = (
    RESULT_DIR
    / "training_config.json"
)


# ============================================================
# FULL TRAINING
# ============================================================

PILOT_MODE = False

# Chỉ dùng khi PILOT_MODE = True
PILOT_TRAIN_RECORDS = 20_000
PILOT_VALIDATION_RECORDS = 4_000


# ============================================================
# CHUNK CONFIG
# ============================================================

# Giữ nguyên baseline để so sánh công bằng.
MAX_LENGTH = 256

# Overlap giữa hai chunk liên tiếp.
STRIDE = 64

# Nếu document dài:
#
# lấy tối đa 3 chunk trải đều:
#
# đầu - giữa - cuối
#
# Vì title + abstract nằm ở đầu document,
# chunk đầu sẽ nhìn thấy chúng.
MAX_CHUNKS_PER_DOCUMENT = 3


# ============================================================
# TRAIN CONFIG
# ============================================================

# Giữ nguyên baseline.
EPOCHS = 2

DOCUMENT_BATCH_SIZE = 12

VALIDATION_DOCUMENT_BATCH_SIZE = 16

GRADIENT_ACCUMULATION_STEPS = 1


# ============================================================
# LEARNING RATE
# ============================================================

ENCODER_LR = 2e-5

HEAD_LR = 1e-4

WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.10

MAX_GRAD_NORM = 1.0


# ============================================================
# MODEL FREEZING
# ============================================================

# PhoBERT-base có 12 Transformer layers.
#
# Freeze 6 layer dưới.
# Fine-tune 6 layer trên.
FREEZE_BOTTOM_LAYERS = 6


# ============================================================
# GRADIENT CHECKPOINTING
# ============================================================

USE_GRADIENT_CHECKPOINTING = False


# ============================================================
# DATA LOADER
# ============================================================

NUM_WORKERS = 2

PREFETCH_FACTOR = 2


# ============================================================
# RANDOM SEED
# ============================================================

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
# BUILD DOCUMENT TEXT
# ============================================================

def build_document_text(row):
    """
    Ghép:

        title_segmented
        abstract_segmented
        article_segmented

    Ví dụ:

        <title> . <abstract> . <article>

    Không tạo special token mới để tránh phải resize
    embedding của PhoBERT.
    """

    parts = []

    for field in TEXT_FIELDS:

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

    return FIELD_SEPARATOR.join(
        parts
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_records(
    path,
    limit=None
):

    print(
        f"\nLoading:\n{path}"
    )

    records = []

    label_counts = {
        0: 0,
        1: 0,
    }

    if limit is not None:

        target_per_label = (
            limit // 2
        )

    else:

        target_per_label = None

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

            # =================================================
            # PILOT BALANCING
            # =================================================

            if (
                target_per_label
                is not None
            ):

                if (
                    label_counts[label]
                    >= target_per_label
                ):

                    if (
                        label_counts[0]
                        >= target_per_label
                        and
                        label_counts[1]
                        >= target_per_label
                    ):

                        break

                    continue

            # =================================================
            # ALL-FIELDS INPUT
            # =================================================

            text = build_document_text(
                row
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
            })

            label_counts[
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
        f"{label_counts[0]:,}"
    )

    print(
        f"Noisy  : "
        f"{label_counts[1]:,}"
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
        tokenizer
    ):

        self.tokenizer = tokenizer

        self.special_tokens = (
            tokenizer
            .num_special_tokens_to_add(
                pair=False
            )
        )

        self.content_length = (
            MAX_LENGTH
            - self.special_tokens
        )

        self.step_size = (
            self.content_length
            - STRIDE
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

        # Trường hợp document rỗng.
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

        # ====================================================
        # Nếu document tạo quá nhiều chunk:
        #
        # lấy các chunk trải đều toàn document.
        #
        # đầu - giữa - cuối
        # ====================================================

        selected_indices = (
            evenly_spaced_indices(
                len(chunks),
                MAX_CHUNKS_PER_DOCUMENT
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

        # ====================================================
        # TOKENIZE WHOLE DOCUMENT BATCH
        # ====================================================

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

        # ====================================================
        # CREATE DOCUMENT CHUNKS
        # ====================================================

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
            })

        # ====================================================
        # PAD ALL CHUNKS IN CURRENT DOCUMENT BATCH
        # ====================================================

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

        self.encoder = (
            AutoModel
            .from_pretrained(
                model_name
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
        # FIRST TOKEN REPRESENTATION
        #
        # Không dùng pooler_output.
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
        # MULTIPLE INSTANCE LEARNING
        #
        # document score =
        # max(chunk score)
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
# FREEZE LOWER LAYERS
# ============================================================

def freeze_lower_layers(
    model,
    number_of_layers
):

    if number_of_layers <= 0:

        print(
            "\nAll PhoBERT layers trainable."
        )

        return

    # ========================================================
    # EMBEDDINGS
    # ========================================================

    if hasattr(
        model.encoder,
        "embeddings"
    ):

        for parameter in (
            model.encoder
            .embeddings
            .parameters()
        ):

            parameter.requires_grad = False

    # ========================================================
    # TRANSFORMER LAYERS
    # ========================================================

    layers = (
        model.encoder
        .encoder
        .layer
    )

    freeze_count = min(
        number_of_layers,
        len(layers)
    )

    for layer in layers[
        :freeze_count
    ]:

        for parameter in (
            layer.parameters()
        ):

            parameter.requires_grad = False

    print(
        f"\nFrozen bottom layers: "
        f"{freeze_count}"
    )

    print(
        f"Trainable upper layers: "
        f"{len(layers) - freeze_count}"
    )


# ============================================================
# MODEL PARAMETER STATS
# ============================================================

def print_parameter_stats(
    model
):

    total = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    frozen = (
        total
        - trainable
    )

    print(
        f"\nTotal parameters     : "
        f"{total:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable:,}"
    )

    print(
        f"Frozen parameters    : "
        f"{frozen:,}"
    )

    print(
        f"Trainable ratio      : "
        f"{trainable / total * 100:.2f}%"
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    labels,
    probabilities,
    metadata
):

    labels = np.asarray(
        labels,
        dtype=np.int64
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64
    )

    predictions = (
        probabilities
        >= 0.5
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

    # ========================================================
    # RECALL BY SEVERITY
    # ========================================================

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
            ] = {
                "count": 0,
                "recall": None,
            }

            continue

        severity_predictions = (
            predictions[
                indices
            ]
        )

        severity_recall[
            severity
        ] = {
            "count":
                len(indices),

            "recall":
                float(
                    np.mean(
                        severity_predictions
                        == 1
                    )
                ),
        }

    return {
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
                int(
                    cm[0, 0]
                ),

            "fp":
                int(
                    cm[0, 1]
                ),

            "fn":
                int(
                    cm[1, 0]
                ),

            "tp":
                int(
                    cm[1, 1]
                ),
        },

        "severity_recall":
            severity_recall,
    }


# ============================================================
# AMP
# ============================================================

def autocast_context():

    return torch.amp.autocast(
        device_type="cuda",

        dtype=torch.float16,

        enabled=USE_AMP,
    )


# ============================================================
# GPU MEMORY
# ============================================================

def print_gpu_memory():

    if not torch.cuda.is_available():
        return

    allocated = (
        torch.cuda.memory_allocated()
        / 1024**3
    )

    reserved = (
        torch.cuda.memory_reserved()
        / 1024**3
    )

    peak = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    print(
        f"\nGPU memory allocated : "
        f"{allocated:.2f} GB"
    )

    print(
        f"GPU memory reserved  : "
        f"{reserved:.2f} GB"
    )

    print(
        f"GPU peak allocated   : "
        f"{peak:.2f} GB"
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    labels_all = []

    probabilities_all = []

    metadata_all = []

    validation_start = time.time()

    with torch.inference_mode():

        progress = tqdm(
            loader,
            desc="Validation"
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

    validation_seconds = (
        time.time()
        - validation_start
    )

    metrics = calculate_metrics(
        labels_all,
        probabilities_all,
        metadata_all
    )

    metrics[
        "validation_loss"
    ] = (
        total_loss
        / max(
            1,
            len(loader)
        )
    )

    metrics[
        "validation_seconds"
    ] = float(
        validation_seconds
    )

    return metrics


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(
    epoch,
    metrics
):

    print("\n" + "=" * 80)

    print(
        f"EPOCH {epoch} - VALIDATION"
    )

    print("=" * 80)

    print(
        f"Loss            : "
        f"{metrics['validation_loss']:.6f}"
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
        f"ROC-AUC         : "
        f"{metrics['roc_auc']:.6f}"
    )

    print(
        f"AveragePrecision: "
        f"{metrics['average_precision']:.6f}"
    )

    print(
        f"Validation time : "
        f"{metrics['validation_seconds']:.2f}s"
    )

    cm = metrics[
        "confusion_matrix"
    ]

    print("\nConfusion Matrix")

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
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    tokenizer,
    metrics,
    epoch
):

    checkpoint = {
        "model_name":
            MODEL_NAME,

        "input_mode":
            INPUT_MODE,

        "text_fields":
            TEXT_FIELDS,

        "field_separator":
            FIELD_SEPARATOR,

        "epoch":
            epoch,

        "max_length":
            MAX_LENGTH,

        "stride":
            STRIDE,

        "max_chunks_per_document":
            MAX_CHUNKS_PER_DOCUMENT,

        "document_batch_size":
            DOCUMENT_BATCH_SIZE,

        "validation_document_batch_size":
            VALIDATION_DOCUMENT_BATCH_SIZE,

        "gradient_accumulation_steps":
            GRADIENT_ACCUMULATION_STEPS,

        "freeze_bottom_layers":
            FREEZE_BOTTOM_LAYERS,

        "gradient_checkpointing":
            USE_GRADIENT_CHECKPOINTING,

        "metrics":
            metrics,

        "state_dict":
            model.state_dict(),
    }

    torch.save(
        checkpoint,
        BEST_MODEL_PATH
    )

    tokenizer.save_pretrained(
        OUTPUT_DIR
        / "tokenizer"
    )

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"\nBest model saved:\n"
        f"{BEST_MODEL_PATH}"
    )


# ============================================================
# SAVE CONFIG
# ============================================================

def save_training_config():

    config = {
        "model_name":
            MODEL_NAME,

        "input_mode":
            INPUT_MODE,

        "text_fields":
            TEXT_FIELDS,

        "field_separator":
            FIELD_SEPARATOR,

        "pilot_mode":
            PILOT_MODE,

        "epochs":
            EPOCHS,

        "max_length":
            MAX_LENGTH,

        "stride":
            STRIDE,

        "max_chunks_per_document":
            MAX_CHUNKS_PER_DOCUMENT,

        "document_batch_size":
            DOCUMENT_BATCH_SIZE,

        "validation_document_batch_size":
            VALIDATION_DOCUMENT_BATCH_SIZE,

        "gradient_accumulation_steps":
            GRADIENT_ACCUMULATION_STEPS,

        "effective_document_batch_size":
            (
                DOCUMENT_BATCH_SIZE
                * GRADIENT_ACCUMULATION_STEPS
            ),

        "encoder_lr":
            ENCODER_LR,

        "head_lr":
            HEAD_LR,

        "weight_decay":
            WEIGHT_DECAY,

        "warmup_ratio":
            WARMUP_RATIO,

        "freeze_bottom_layers":
            FREEZE_BOTTOM_LAYERS,

        "gradient_checkpointing":
            USE_GRADIENT_CHECKPOINTING,

        "num_workers":
            NUM_WORKERS,

        "random_seed":
            RANDOM_SEED,
    }

    with open(
        CONFIG_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("VN-NEWSSPELL - PHOBERT MIL ALL FIELDS")
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
        f"\nModel: "
        f"{MODEL_NAME}"
    )

    print(
        f"Input mode: "
        f"{INPUT_MODE}"
    )

    print(
        "Text fields:"
    )

    for field in TEXT_FIELDS:

        print(
            f"  - {field}"
        )

    print(
        f"Field separator: "
        f"{repr(FIELD_SEPARATOR)}"
    )

    print(
        f"Pilot mode: "
        f"{PILOT_MODE}"
    )

    print(
        f"\nMAX_LENGTH: "
        f"{MAX_LENGTH}"
    )

    print(
        f"STRIDE: "
        f"{STRIDE}"
    )

    print(
        f"MAX_CHUNKS: "
        f"{MAX_CHUNKS_PER_DOCUMENT}"
    )

    print(
        f"\nTrain document batch: "
        f"{DOCUMENT_BATCH_SIZE}"
    )

    print(
        f"Validation batch: "
        f"{VALIDATION_DOCUMENT_BATCH_SIZE}"
    )

    print(
        f"Gradient accumulation: "
        f"{GRADIENT_ACCUMULATION_STEPS}"
    )

    effective_batch = (
        DOCUMENT_BATCH_SIZE
        * GRADIENT_ACCUMULATION_STEPS
    )

    print(
        f"Effective document batch: "
        f"{effective_batch}"
    )

    print(
        f"DataLoader workers: "
        f"{NUM_WORKERS}"
    )

    print(
        f"Gradient checkpointing: "
        f"{USE_GRADIENT_CHECKPOINTING}"
    )

    print(
        "\nTest set is NOT used."
    )

    set_seed(
        RANDOM_SEED
    )

    save_training_config()

    # ========================================================
    # LOAD DATA
    # ========================================================

    if PILOT_MODE:

        train_limit = (
            PILOT_TRAIN_RECORDS
        )

        validation_limit = (
            PILOT_VALIDATION_RECORDS
        )

    else:

        train_limit = None

        validation_limit = None

    train_records = load_records(
        TRAIN_PATH,
        limit=train_limit
    )

    validation_records = load_records(
        VALIDATION_PATH,
        limit=validation_limit
    )

    train_dataset = NewsDataset(
        train_records
    )

    validation_dataset = (
        NewsDataset(
            validation_records
        )
    )

    # ========================================================
    # TOKENIZER
    # ========================================================

    print(
        "\nLoading tokenizer..."
    )

    try:

        tokenizer = (
            AutoTokenizer
            .from_pretrained(
                MODEL_NAME,
                use_fast=True
            )
        )

        print(
            f"Tokenizer fast: "
            f"{tokenizer.is_fast}"
        )

    except Exception as error:

        print(
            "\nFast tokenizer unavailable."
        )

        print(
            f"Reason: {error}"
        )

        print(
            "Falling back to slow tokenizer..."
        )

        tokenizer = (
            AutoTokenizer
            .from_pretrained(
                MODEL_NAME,
                use_fast=False
            )
        )

    collator = DocumentCollator(
        tokenizer
    )

    # ========================================================
    # DATA LOADERS
    # ========================================================

    loader_common = {
        "num_workers":
            NUM_WORKERS,

        "pin_memory":
            USE_AMP,
    }

    if NUM_WORKERS > 0:

        loader_common[
            "persistent_workers"
        ] = True

        loader_common[
            "prefetch_factor"
        ] = PREFETCH_FACTOR

    train_loader = DataLoader(
        train_dataset,

        batch_size=
            DOCUMENT_BATCH_SIZE,

        shuffle=True,

        collate_fn=
            collator,

        **loader_common,
    )

    validation_loader = DataLoader(
        validation_dataset,

        batch_size=
            VALIDATION_DOCUMENT_BATCH_SIZE,

        shuffle=False,

        collate_fn=
            collator,

        **loader_common,
    )

    # ========================================================
    # MODEL
    # ========================================================

    print(
        "\nLoading PhoBERT..."
    )

    model = PhoBERTMIL(
        MODEL_NAME
    )

    freeze_lower_layers(
        model,
        FREEZE_BOTTOM_LAYERS
    )

    # ========================================================
    # GRADIENT CHECKPOINTING
    # ========================================================

    if (
        USE_GRADIENT_CHECKPOINTING
        and hasattr(
            model.encoder,
            "gradient_checkpointing_enable"
        )
    ):

        model.encoder.gradient_checkpointing_enable()

        print(
            "Gradient checkpointing: ENABLED"
        )

    else:

        if hasattr(
            model.encoder,
            "gradient_checkpointing_disable"
        ):

            model.encoder.gradient_checkpointing_disable()

        print(
            "Gradient checkpointing: DISABLED"
        )

    print_parameter_stats(
        model
    )

    model.to(
        DEVICE
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    encoder_parameters = [
        parameter
        for parameter
        in model.encoder.parameters()
        if parameter.requires_grad
    ]

    head_parameters = list(
        model.classifier.parameters()
    )

    optimizer_groups = [
        {
            "params":
                encoder_parameters,

            "lr":
                ENCODER_LR,
        },
        {
            "params":
                head_parameters,

            "lr":
                HEAD_LR,
        },
    ]

    # ========================================================
    # FUSED ADAMW
    # ========================================================

    try:

        optimizer = torch.optim.AdamW(
            optimizer_groups,

            weight_decay=
                WEIGHT_DECAY,

            fused=(
                DEVICE.type
                == "cuda"
            ),
        )

        if DEVICE.type == "cuda":

            print(
                "\nAdamW fused: ENABLED"
            )

    except TypeError:

        optimizer = torch.optim.AdamW(
            optimizer_groups,

            weight_decay=
                WEIGHT_DECAY,
        )

        print(
            "\nAdamW fused: unavailable"
        )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    # ========================================================
    # SCHEDULER
    # ========================================================

    optimizer_steps_per_epoch = (
        math.ceil(
            len(train_loader)
            / GRADIENT_ACCUMULATION_STEPS
        )
    )

    total_optimizer_steps = (
        optimizer_steps_per_epoch
        * EPOCHS
    )

    warmup_steps = int(
        total_optimizer_steps
        * WARMUP_RATIO
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,

            num_warmup_steps=
                warmup_steps,

            num_training_steps=
                total_optimizer_steps,
        )
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP
    )

    print(
        f"\nTraining batches/epoch: "
        f"{len(train_loader):,}"
    )

    print(
        f"Optimizer steps/epoch: "
        f"{optimizer_steps_per_epoch:,}"
    )

    print(
        f"Total optimizer steps: "
        f"{total_optimizer_steps:,}"
    )

    print(
        f"Warmup steps: "
        f"{warmup_steps:,}"
    )

    # ========================================================
    # TRAIN
    # ========================================================

    best_macro_f1 = -1.0

    training_history = []

    optimizer.zero_grad(
        set_to_none=True
    )

    overall_start = time.time()

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        print("\n" + "#" * 80)

        print(
            f"EPOCH {epoch}/{EPOCHS}"
        )

        print("#" * 80)

        model.train()

        running_loss = 0.0

        epoch_start = time.time()

        if torch.cuda.is_available():

            torch.cuda.reset_peak_memory_stats()

        progress = tqdm(
            enumerate(
                train_loader,
                start=1
            ),

            total=
                len(
                    train_loader
                ),

            desc="Training"
        )

        for step, batch in progress:

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

            # =================================================
            # FORWARD
            # =================================================

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

                scaled_loss = (
                    loss
                    / GRADIENT_ACCUMULATION_STEPS
                )

            # =================================================
            # BACKWARD
            # =================================================

            scaler.scale(
                scaled_loss
            ).backward()

            running_loss += (
                loss.item()
            )

            should_step = (
                step
                % GRADIENT_ACCUMULATION_STEPS
                == 0
                or
                step
                == len(train_loader)
            )

            if should_step:

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    MAX_GRAD_NORM
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

            progress.set_postfix({
                "loss":
                    f"{running_loss / step:.4f}",

                "lr":
                    f"{scheduler.get_last_lr()[0]:.2e}",
            })

        # ====================================================
        # TRAIN TIME
        # ====================================================

        epoch_seconds = (
            time.time()
            - epoch_start
        )

        docs_per_second = (
            len(
                train_dataset
            )
            / epoch_seconds
        )

        average_train_loss = (
            running_loss
            / max(
                1,
                len(train_loader)
            )
        )

        print(
            f"\nAverage train loss: "
            f"{average_train_loss:.6f}"
        )

        print(
            f"Epoch training time: "
            f"{epoch_seconds:.2f}s "
            f"({epoch_seconds / 60:.2f} min)"
        )

        print(
            f"Documents/sec: "
            f"{docs_per_second:.2f}"
        )

        print_gpu_memory()

        # ====================================================
        # VALIDATION
        # ====================================================

        metrics = evaluate(
            model,
            validation_loader,
            criterion
        )

        metrics[
            "epoch"
        ] = epoch

        metrics[
            "average_train_loss"
        ] = float(
            average_train_loss
        )

        metrics[
            "epoch_training_seconds"
        ] = float(
            epoch_seconds
        )

        metrics[
            "documents_per_second"
        ] = float(
            docs_per_second
        )

        metrics[
            "pilot_mode"
        ] = PILOT_MODE

        metrics[
            "input_mode"
        ] = INPUT_MODE

        metrics[
            "text_fields"
        ] = TEXT_FIELDS

        metrics[
            "field_separator"
        ] = FIELD_SEPARATOR

        metrics[
            "max_length"
        ] = MAX_LENGTH

        metrics[
            "stride"
        ] = STRIDE

        metrics[
            "max_chunks_per_document"
        ] = (
            MAX_CHUNKS_PER_DOCUMENT
        )

        metrics[
            "document_batch_size"
        ] = (
            DOCUMENT_BATCH_SIZE
        )

        metrics[
            "gradient_accumulation_steps"
        ] = (
            GRADIENT_ACCUMULATION_STEPS
        )

        print_metrics(
            epoch,
            metrics
        )

        # ====================================================
        # HISTORY
        # ====================================================

        training_history.append(
            metrics
        )

        with open(
            HISTORY_PATH,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                training_history,
                f,
                ensure_ascii=False,
                indent=2
            )

        # ====================================================
        # SAVE BEST MODEL BY VALIDATION MACRO-F1 @ 0.5
        # ====================================================

        if (
            metrics[
                "macro_f1"
            ]
            > best_macro_f1
        ):

            best_macro_f1 = (
                metrics[
                    "macro_f1"
                ]
            )

            save_checkpoint(
                model,
                tokenizer,
                metrics,
                epoch
            )

    # ========================================================
    # FINISH
    # ========================================================

    total_seconds = (
        time.time()
        - overall_start
    )

    print("\n" + "=" * 80)
    print("PHOBERT MIL ALL-FIELDS TRAINING FINISHED")
    print("=" * 80)

    print(
        f"\nBest validation Macro-F1: "
        f"{best_macro_f1:.6f}"
    )

    print(
        f"Total training + validation time: "
        f"{total_seconds / 60:.2f} minutes"
    )

    print(
        f"\nBest model:\n"
        f"{BEST_MODEL_PATH}"
    )

    print(
        f"\nTraining history:\n"
        f"{HISTORY_PATH}"
    )

    print(
        f"\nBest validation metrics:\n"
        f"{METRICS_PATH}"
    )

    print(
        "\nTest set was NOT used."
    )


if __name__ == "__main__":
    main()