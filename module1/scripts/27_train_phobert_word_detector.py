from pathlib import Path
from collections import Counter
from bisect import bisect_left
from contextlib import nullcontext
import json
import math
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
    get_linear_schedule_with_warmup,
)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
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
    / "vn_newsspell_wordloc_v1"
)

TRAIN_PATH = (
    DATA_DIR
    / "train.jsonl"
)

VALIDATION_PATH = (
    DATA_DIR
    / "validation.jsonl"
)

DATASET_SUMMARY_PATH = (
    DATA_DIR
    / "dataset_summary.json"
)


# ============================================================
# SOURCE MODEL
# ============================================================

SOURCE_MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "module1_phobert_mil_all_fields_4chunks"
)

SOURCE_CHECKPOINT_PATH = (
    SOURCE_MODEL_DIR
    / "best_model.pt"
)

SOURCE_TOKENIZER_DIR = (
    SOURCE_MODEL_DIR
    / "tokenizer"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "module1_phobert_word_detector_v1"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "module1_phobert_word_detector_v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


BEST_MODEL_PATH = (
    OUTPUT_DIR
    / "best_model.pt"
)

TOKENIZER_OUTPUT_DIR = (
    OUTPUT_DIR
    / "tokenizer"
)

HISTORY_PATH = (
    RESULT_DIR
    / "training_history.json"
)

CONFIG_PATH = (
    RESULT_DIR
    / "training_config.json"
)

BEST_METRICS_PATH = (
    RESULT_DIR
    / "best_validation_metrics.json"
)

BEST_VALIDATION_PROBS_PATH = (
    RESULT_DIR
    / "best_validation_word_probabilities.npz"
)


# ============================================================
# TRAIN CONFIG
# ============================================================

EPOCHS = 2

TRAIN_DOCUMENT_BATCH_SIZE = 12

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
# FREEZING
# ============================================================

FREEZE_BOTTOM_LAYERS = 6


# ============================================================
# DATALOADER
# ============================================================

NUM_WORKERS = 2

PREFETCH_FACTOR = 2


# ============================================================
# RANDOM
# ============================================================

RANDOM_SEED = 20260922


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
# CUDA
# ============================================================

if torch.cuda.is_available():

    torch.backends.cuda.matmul.allow_tf32 = True

    torch.backends.cudnn.allow_tf32 = True

    torch.backends.cudnn.benchmark = True

    torch.set_float32_matmul_precision(
        "high"
    )


# ============================================================
# CONSTANT CODES
# ============================================================

FIELD_TO_CODE = {
    "title": 1,
    "abstract": 2,
    "article": 3,
}

CODE_TO_FIELD = {
    value: key
    for key, value
    in FIELD_TO_CODE.items()
}

SEVERITY_TO_CODE = {
    "minor": 1,
    "moderate": 2,
    "severe": 3,
}

CODE_TO_SEVERITY = {
    value: key
    for key, value
    in SEVERITY_TO_CODE.items()
}


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
# LAZY JSONL DATASET
#
# Không load 52 triệu words vào RAM.
# Chỉ lưu byte offset của từng dòng JSONL.
# ============================================================

class JsonlOffsetDataset(Dataset):

    def __init__(
        self,
        path,
    ):

        self.path = Path(
            path
        )

        self.offsets = []

        self._file = None

        print(
            f"\nIndexing:\n"
            f"{self.path}"
        )

        start_time = time.time()

        with open(
            self.path,
            "rb",
        ) as f:

            while True:

                offset = f.tell()

                line = f.readline()

                if not line:
                    break

                if line.strip():

                    self.offsets.append(
                        offset
                    )

        elapsed = (
            time.time()
            - start_time
        )

        print(
            f"Records: "
            f"{len(self.offsets):,}"
        )

        print(
            f"Index time: "
            f"{elapsed:.2f}s"
        )

    def __len__(
        self,
    ):

        return len(
            self.offsets
        )

    def _get_file(
        self,
    ):

        if self._file is None:

            self._file = open(
                self.path,
                "rb",
            )

        return self._file

    def __getitem__(
        self,
        index,
    ):

        f = self._get_file()

        f.seek(
            self.offsets[
                index
            ]
        )

        line = f.readline()

        return json.loads(
            line.decode(
                "utf-8"
            )
        )

    def __getstate__(
        self,
    ):

        state = (
            self.__dict__
            .copy()
        )

        state[
            "_file"
        ] = None

        return state

    def __del__(
        self,
    ):

        try:

            if self._file is not None:

                self._file.close()

        except Exception:

            pass


# ============================================================
# SOURCE CHECKPOINT
# ============================================================

def load_source_checkpoint():

    print(
        f"\nLoading source checkpoint:\n"
        f"{SOURCE_CHECKPOINT_PATH}"
    )

    try:

        checkpoint = torch.load(
            SOURCE_CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:

        checkpoint = torch.load(
            SOURCE_CHECKPOINT_PATH,
            map_location="cpu",
        )

    required = [
        "model_name",
        "max_length",
        "stride",
        "max_chunks_per_document",
        "state_dict",
    ]

    missing = [
        key
        for key in required
        if key not in checkpoint
    ]

    if missing:

        raise RuntimeError(
            "Source checkpoint missing keys: "
            + ", ".join(
                missing
            )
        )

    print(
        f"Source epoch  : "
        f"{checkpoint.get('epoch')}"
    )

    print(
        f"Model         : "
        f"{checkpoint['model_name']}"
    )

    print(
        f"MAX_LENGTH    : "
        f"{checkpoint['max_length']}"
    )

    print(
        f"STRIDE        : "
        f"{checkpoint['stride']}"
    )

    print(
        f"MAX_CHUNKS    : "
        f"{checkpoint['max_chunks_per_document']}"
    )

    return checkpoint


# ============================================================
# POSITIVE WEIGHT
# ============================================================

def calculate_positive_weight():

    with open(
        DATASET_SUMMARY_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        summary = json.load(f)

    train_summary = summary[
        "train"
    ]

    total_words = int(
        train_summary[
            "total_word_positions"
        ]
    )

    positive_words = int(
        train_summary[
            "changed_word_positions"
        ]
    )

    negative_words = (
        total_words
        - positive_words
    )

    raw_ratio = (
        negative_words
        / positive_words
    )

    positive_weight = math.sqrt(
        raw_ratio
    )

    return {
        "total_words":
            total_words,

        "positive_words":
            positive_words,

        "negative_words":
            negative_words,

        "raw_negative_positive_ratio":
            raw_ratio,

        "positive_weight":
            positive_weight,
    }


# ============================================================
# CHUNK SELECTION
# ============================================================

def evenly_spaced_indices(
    total_chunks,
    max_chunks,
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
        dtype=int,
    )

    return sorted(
        set(
            indices.tolist()
        )
    )


def get_selected_chunk_ranges(
    total_tokens,
    content_length,
    stride,
    max_chunks,
):

    step_size = (
        content_length
        - stride
    )

    if step_size <= 0:

        raise ValueError(
            "Invalid chunk stride."
        )

    if total_tokens == 0:

        return [
            {
                "chunk_index": 0,
                "start": 0,
                "end": 0,
            }
        ]

    starts = []

    start = 0

    while True:

        starts.append(
            start
        )

        if (
            start
            + content_length
            >= total_tokens
        ):

            break

        start += (
            step_size
        )

    selected_indices = (
        evenly_spaced_indices(
            len(
                starts
            ),
            max_chunks,
        )
    )

    result = []

    for chunk_index in (
        selected_indices
    ):

        chunk_start = (
            starts[
                chunk_index
            ]
        )

        chunk_end = min(
            chunk_start
            + content_length,
            total_tokens,
        )

        result.append({
            "chunk_index":
                int(
                    chunk_index
                ),

            "start":
                int(
                    chunk_start
                ),

            "end":
                int(
                    chunk_end
                ),
        })

    return result


# ============================================================
# FIND CONTENT OFFSET AFTER SPECIAL TOKENS
# ============================================================

def find_content_offset(
    full_ids,
    content_ids,
):

    if not content_ids:

        return 0

    max_start = (
        len(
            full_ids
        )
        - len(
            content_ids
        )
    )

    for start in range(
        max_start + 1
    ):

        if (
            full_ids[
                start:
                start
                + len(
                    content_ids
                )
            ]
            == content_ids
        ):

            return start

    raise RuntimeError(
        "Could not locate content tokens "
        "inside special-token sequence."
    )


# ============================================================
# COLLATOR
# ============================================================

class WordDetectorCollator:

    def __init__(
        self,
        tokenizer,
        max_length,
        stride,
        max_chunks,
        deduplicate_training_targets,
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

        self.deduplicate_training_targets = (
            deduplicate_training_targets
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

    def __call__(
        self,
        batch,
    ):

        # ====================================================
        # TOKENIZE ALL WORDS IN ONE TOKENIZER CALL
        # ====================================================

        flat_words = []

        document_slices = []

        for row in batch:

            start = len(
                flat_words
            )

            words = row[
                "words"
            ]

            flat_words.extend(
                words
            )

            end = len(
                flat_words
            )

            document_slices.append(
                (
                    start,
                    end,
                )
            )

        encoded_words = (
            self.tokenizer(
                flat_words,
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )[
                "input_ids"
            ]
        )

        chunk_features = []

        target_chunk_indices = []
        target_token_positions = []
        target_labels = []
        target_doc_indices = []
        target_word_indices = []

        metadata = []

        # ====================================================
        # EACH DOCUMENT
        # ====================================================

        for doc_index, row in enumerate(
            batch
        ):

            slice_start, slice_end = (
                document_slices[
                    doc_index
                ]
            )

            word_token_ids = (
                encoded_words[
                    slice_start:
                    slice_end
                ]
            )

            words = row[
                "words"
            ]

            word_fields = row[
                "word_fields"
            ]

            if (
                len(
                    word_token_ids
                )
                != len(
                    words
                )
            ):

                raise RuntimeError(
                    "Word/tokenized-word count mismatch."
                )

            # =================================================
            # BUILD FULL TOKEN SEQUENCE + WORD START POSITIONS
            # =================================================

            full_document_ids = []

            word_token_starts = []

            for word_index, ids in enumerate(
                word_token_ids
            ):

                if (
                    word_fields[
                        word_index
                    ]
                    != "separator"
                    and
                    len(ids) == 0
                ):

                    raise RuntimeError(
                        f"Empty tokenization for word: "
                        f"{words[word_index]!r}"
                    )

                word_token_starts.append(
                    len(
                        full_document_ids
                    )
                )

                full_document_ids.extend(
                    ids
                )

            error_indices = set(
                int(index)
                for index
                in row[
                    "error_word_indices"
                ]
            )

            selected_ranges = (
                get_selected_chunk_ranges(
                    total_tokens=
                        len(
                            full_document_ids
                        ),

                    content_length=
                        self.content_length,

                    stride=
                        self.stride,

                    max_chunks=
                        self.max_chunks,
                )
            )

            seen_training_words = set()

            # =================================================
            # CHUNKS
            # =================================================

            for chunk in selected_ranges:

                start = chunk[
                    "start"
                ]

                end = chunk[
                    "end"
                ]

                content_ids = (
                    full_document_ids[
                        start:end
                    ]
                )

                chunk_ids = (
                    self.tokenizer
                    .build_inputs_with_special_tokens(
                        content_ids
                    )
                )

                content_offset = (
                    find_content_offset(
                        chunk_ids,
                        content_ids,
                    )
                )

                chunk_global_index = (
                    len(
                        chunk_features
                    )
                )

                chunk_features.append({
                    "input_ids":
                        chunk_ids,

                    "attention_mask":
                        [1]
                        * len(
                            chunk_ids
                        ),
                })

                # =============================================
                # Find words whose FIRST SUBTOKEN is inside
                # this chunk.
                # =============================================

                left = bisect_left(
                    word_token_starts,
                    start,
                )

                right = bisect_left(
                    word_token_starts,
                    end,
                )

                for word_index in range(
                    left,
                    right,
                ):

                    # Separator is never a classification target.
                    if (
                        word_fields[
                            word_index
                        ]
                        == "separator"
                    ):

                        continue

                    if (
                        self.deduplicate_training_targets
                        and
                        word_index
                        in
                        seen_training_words
                    ):

                        continue

                    global_token_start = (
                        word_token_starts[
                            word_index
                        ]
                    )

                    local_token_position = (
                        content_offset
                        +
                        (
                            global_token_start
                            - start
                        )
                    )

                    target_chunk_indices.append(
                        chunk_global_index
                    )

                    target_token_positions.append(
                        local_token_position
                    )

                    target_labels.append(
                        1.0
                        if word_index
                        in error_indices
                        else 0.0
                    )

                    target_doc_indices.append(
                        doc_index
                    )

                    target_word_indices.append(
                        word_index
                    )

                    if (
                        self.deduplicate_training_targets
                    ):

                        seen_training_words.add(
                            word_index
                        )

            metadata.append({
                "source_id":
                    row[
                        "source_id"
                    ],

                "variant_id":
                    row[
                        "variant_id"
                    ],

                "severity":
                    row[
                        "severity"
                    ],

                "word_count":
                    len(
                        words
                    ),

                "word_fields":
                    word_fields,

                "error_word_indices":
                    list(
                        error_indices
                    ),

                "selected_chunk_count":
                    len(
                        selected_ranges
                    ),
            })

        # ====================================================
        # PAD CHUNKS
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

            "target_chunk_indices":
                torch.tensor(
                    target_chunk_indices,
                    dtype=torch.long,
                ),

            "target_token_positions":
                torch.tensor(
                    target_token_positions,
                    dtype=torch.long,
                ),

            "target_labels":
                torch.tensor(
                    target_labels,
                    dtype=torch.float32,
                ),

            "target_doc_indices":
                torch.tensor(
                    target_doc_indices,
                    dtype=torch.long,
                ),

            "target_word_indices":
                torch.tensor(
                    target_word_indices,
                    dtype=torch.long,
                ),

            "metadata":
                metadata,
        }


# ============================================================
# MODEL
# ============================================================

class PhoBERTWordDetector(nn.Module):

    def __init__(
        self,
        model_name,
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

        self.word_classifier = (
            nn.Linear(
                hidden_size,
                1,
            )
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        target_chunk_indices,
        target_token_positions,
    ):

        outputs = self.encoder(
            input_ids=
                input_ids,

            attention_mask=
                attention_mask,
        )

        hidden = (
            outputs
            .last_hidden_state
        )

        word_embeddings = (
            hidden[
                target_chunk_indices,
                target_token_positions,
            ]
        )

        logits = (
            self.word_classifier(
                self.dropout(
                    word_embeddings
                )
            )
            .squeeze(-1)
        )

        return logits


# ============================================================
# LOAD ENCODER FROM MIL MODEL
# ============================================================

def load_encoder_from_source(
    model,
    checkpoint,
):

    source_state = checkpoint[
        "state_dict"
    ]

    encoder_state = {}

    prefix = "encoder."

    for key, value in (
        source_state.items()
    ):

        if key.startswith(
            prefix
        ):

            encoder_state[
                key[
                    len(
                        prefix
                    ):
                ]
            ] = value

    if not encoder_state:

        raise RuntimeError(
            "No encoder weights found "
            "in source checkpoint."
        )

    result = (
        model.encoder
        .load_state_dict(
            encoder_state,
            strict=True,
        )
    )

    print(
        f"\nTransferred encoder weights."
    )

    print(
        f"Encoder missing keys    : "
        f"{len(result.missing_keys)}"
    )

    print(
        f"Encoder unexpected keys : "
        f"{len(result.unexpected_keys)}"
    )


# ============================================================
# FREEZE LOWER LAYERS
# ============================================================

def freeze_lower_layers(
    model,
    count,
):

    if hasattr(
        model.encoder,
        "embeddings",
    ):

        for parameter in (
            model.encoder
            .embeddings
            .parameters()
        ):

            parameter.requires_grad = False

    layers = (
        model.encoder
        .encoder
        .layer
    )

    freeze_count = min(
        count,
        len(
            layers
        ),
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
# PARAMETER STATS
# ============================================================

def print_parameter_stats(
    model,
):

    total = sum(
        p.numel()
        for p
        in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p
        in model.parameters()
        if p.requires_grad
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
        f"{total - trainable:,}"
    )

    print(
        f"Trainable ratio      : "
        f"{trainable / total * 100:.2f}%"
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
# GPU MEMORY
# ============================================================

def print_gpu_memory():

    if not torch.cuda.is_available():
        return

    print(
        f"\nGPU allocated : "
        f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    )

    print(
        f"GPU reserved  : "
        f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
    )

    print(
        f"GPU peak      : "
        f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
    )


# ============================================================
# VALIDATION
# ============================================================

def evaluate(
    model,
    loader,
    positive_weight,
):

    model.eval()

    # Loss này chỉ để tham khảo.
    criterion_sum = (
        nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                positive_weight,
                dtype=torch.float32,
                device=DEVICE,
            ),
            reduction="sum",
        )
    )

    total_loss = 0.0
    total_occurrence_targets = 0

    probability_batches = []
    label_batches = []
    covered_batches = []
    severity_batches = []
    field_batches = []

    documents = 0
    documents_any_error_detected = 0
    documents_all_errors_detected = 0

    start_time = time.time()

    with torch.inference_mode():

        progress = tqdm(
            loader,
            desc="Validation",
        )

        for batch in progress:

            input_ids = (
                batch[
                    "input_ids"
                ]
                .to(
                    DEVICE,
                    non_blocking=True,
                )
            )

            attention_mask = (
                batch[
                    "attention_mask"
                ]
                .to(
                    DEVICE,
                    non_blocking=True,
                )
            )

            target_chunk_indices = (
                batch[
                    "target_chunk_indices"
                ]
                .to(
                    DEVICE,
                    non_blocking=True,
                )
            )

            target_token_positions = (
                batch[
                    "target_token_positions"
                ]
                .to(
                    DEVICE,
                    non_blocking=True,
                )
            )

            occurrence_labels = (
                batch[
                    "target_labels"
                ]
                .to(
                    DEVICE,
                    non_blocking=True,
                )
            )

            with autocast_context():

                logits = model(
                    input_ids=
                        input_ids,

                    attention_mask=
                        attention_mask,

                    target_chunk_indices=
                        target_chunk_indices,

                    target_token_positions=
                        target_token_positions,
                )

                loss_sum = (
                    criterion_sum(
                        logits,
                        occurrence_labels,
                    )
                )

            total_loss += float(
                loss_sum.item()
            )

            total_occurrence_targets += int(
                occurrence_labels.numel()
            )

            occurrence_probs = (
                torch.sigmoid(
                    logits
                )
                .float()
                .cpu()
                .numpy()
            )

            occurrence_doc_indices = (
                batch[
                    "target_doc_indices"
                ]
                .numpy()
            )

            occurrence_word_indices = (
                batch[
                    "target_word_indices"
                ]
                .numpy()
            )

            # =================================================
            # DEDUP SAME WORD ACROSS OVERLAPPING CHUNKS
            # =================================================

            for doc_index, metadata in enumerate(
                batch[
                    "metadata"
                ]
            ):

                word_count = int(
                    metadata[
                        "word_count"
                    ]
                )

                doc_probs = np.zeros(
                    word_count,
                    dtype=np.float32,
                )

                doc_covered = np.zeros(
                    word_count,
                    dtype=bool,
                )

                mask = (
                    occurrence_doc_indices
                    == doc_index
                )

                words_for_doc = (
                    occurrence_word_indices[
                        mask
                    ]
                )

                probs_for_doc = (
                    occurrence_probs[
                        mask
                    ]
                )

                if len(
                    words_for_doc
                ) > 0:

                    np.maximum.at(
                        doc_probs,
                        words_for_doc,
                        probs_for_doc,
                    )

                    doc_covered[
                        words_for_doc
                    ] = True

                doc_labels = np.zeros(
                    word_count,
                    dtype=np.uint8,
                )

                error_indices = np.asarray(
                    metadata[
                        "error_word_indices"
                    ],
                    dtype=np.int64,
                )

                if len(
                    error_indices
                ) > 0:

                    doc_labels[
                        error_indices
                    ] = 1

                field_codes = np.asarray(
                    [
                        FIELD_TO_CODE.get(
                            field,
                            0,
                        )
                        for field
                        in metadata[
                            "word_fields"
                        ]
                    ],
                    dtype=np.uint8,
                )

                # separator = code 0
                eval_mask = (
                    field_codes
                    != 0
                )

                severity_code = (
                    SEVERITY_TO_CODE.get(
                        metadata[
                            "severity"
                        ],
                        0,
                    )
                )

                eval_count = int(
                    np.sum(
                        eval_mask
                    )
                )

                severity_codes = np.full(
                    eval_count,
                    severity_code,
                    dtype=np.uint8,
                )

                probability_batches.append(
                    doc_probs[
                        eval_mask
                    ]
                )

                label_batches.append(
                    doc_labels[
                        eval_mask
                    ]
                )

                covered_batches.append(
                    doc_covered[
                        eval_mask
                    ]
                    .astype(
                        np.uint8
                    )
                )

                severity_batches.append(
                    severity_codes
                )

                field_batches.append(
                    field_codes[
                        eval_mask
                    ]
                )

                # =================================================
                # DOCUMENT-LEVEL WORD LOCALIZATION SUCCESS
                # =================================================

                documents += 1

                if len(
                    error_indices
                ) > 0:

                    error_probs = (
                        doc_probs[
                            error_indices
                        ]
                    )

                    error_predictions = (
                        error_probs
                        >= 0.5
                    )

                    if np.any(
                        error_predictions
                    ):

                        documents_any_error_detected += 1

                    if np.all(
                        error_predictions
                    ):

                        documents_all_errors_detected += 1

    elapsed = (
        time.time()
        - start_time
    )

    probabilities = np.concatenate(
        probability_batches
    ).astype(
        np.float32
    )

    labels = np.concatenate(
        label_batches
    ).astype(
        np.uint8
    )

    covered = np.concatenate(
        covered_batches
    ).astype(
        np.uint8
    )

    severity_codes = np.concatenate(
        severity_batches
    ).astype(
        np.uint8
    )

    field_codes = np.concatenate(
        field_batches
    ).astype(
        np.uint8
    )

    predictions = (
        probabilities
        >= 0.5
    ).astype(
        np.uint8
    )

    # ========================================================
    # CORE METRICS
    # ========================================================

    precision = precision_score(
        labels,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    average_precision = (
        average_precision_score(
            labels,
            probabilities,
        )
    )

    roc_auc = roc_auc_score(
        labels,
        probabilities,
    )

    positive_count = int(
        np.sum(
            labels == 1
        )
    )

    negative_count = int(
        np.sum(
            labels == 0
        )
    )

    tp = int(
        np.sum(
            (labels == 1)
            &
            (predictions == 1)
        )
    )

    fp = int(
        np.sum(
            (labels == 0)
            &
            (predictions == 1)
        )
    )

    fn = int(
        np.sum(
            (labels == 1)
            &
            (predictions == 0)
        )
    )

    covered_words = int(
        np.sum(
            covered == 1
        )
    )

    covered_error_words = int(
        np.sum(
            (labels == 1)
            &
            (covered == 1)
        )
    )

    # ========================================================
    # AP ONLY ON WORDS ACTUALLY SEEN BY MODEL
    # ========================================================

    covered_mask = (
        covered == 1
    )

    covered_ap = (
        average_precision_score(
            labels[
                covered_mask
            ],
            probabilities[
                covered_mask
            ],
        )
        if np.any(
            covered_mask
        )
        else None
    )

    # ========================================================
    # RECALL BY SEVERITY
    # ========================================================

    severity_results = {}

    for severity_name, severity_code in (
        SEVERITY_TO_CODE.items()
    ):

        mask = (
            (labels == 1)
            &
            (
                severity_codes
                == severity_code
            )
        )

        count = int(
            np.sum(
                mask
            )
        )

        if count == 0:

            recall_value = None

        else:

            recall_value = float(
                np.mean(
                    predictions[
                        mask
                    ]
                    == 1
                )
            )

        severity_results[
            severity_name
        ] = {
            "error_words":
                count,

            "recall":
                recall_value,
        }

    # ========================================================
    # RECALL BY FIELD
    # ========================================================

    field_results = {}

    for field_name, field_code in (
        FIELD_TO_CODE.items()
    ):

        mask = (
            (labels == 1)
            &
            (
                field_codes
                == field_code
            )
        )

        count = int(
            np.sum(
                mask
            )
        )

        if count == 0:

            recall_value = None

        else:

            recall_value = float(
                np.mean(
                    predictions[
                        mask
                    ]
                    == 1
                )
            )

        field_results[
            field_name
        ] = {
            "error_words":
                count,

            "recall":
                recall_value,
        }

    metrics = {
        "threshold":
            0.5,

        "word_positions":
            int(
                len(
                    labels
                )
            ),

        "error_words":
            positive_count,

        "clean_words":
            negative_count,

        "word_error_rate":
            float(
                positive_count
                / len(
                    labels
                )
            ),

        "word_precision":
            float(
                precision
            ),

        "word_recall":
            float(
                recall
            ),

        "word_f1":
            float(
                f1
            ),

        "average_precision":
            float(
                average_precision
            ),

        "covered_word_average_precision":
            (
                float(
                    covered_ap
                )
                if covered_ap
                is not None
                else None
            ),

        "roc_auc":
            float(
                roc_auc
            ),

        "tp":
            tp,

        "fp":
            fp,

        "fn":
            fn,

        "word_coverage":
            float(
                covered_words
                / len(
                    labels
                )
            ),

        "error_word_coverage":
            float(
                covered_error_words
                / positive_count
            ),

        "covered_words":
            covered_words,

        "covered_error_words":
            covered_error_words,

        "documents":
            documents,

        "documents_with_any_error_word_detected":
            documents_any_error_detected,

        "documents_with_all_error_words_detected":
            documents_all_errors_detected,

        "document_any_error_recall":
            float(
                documents_any_error_detected
                / documents
            ),

        "document_all_errors_recall":
            float(
                documents_all_errors_detected
                / documents
            ),

        "severity_recall":
            severity_results,

        "field_recall":
            field_results,

        "validation_loss_per_occurrence":
            float(
                total_loss
                / max(
                    1,
                    total_occurrence_targets
                )
            ),

        "validation_seconds":
            float(
                elapsed
            ),
    }

    arrays = {
        "probabilities":
            probabilities,

        "labels":
            labels,

        "covered":
            covered,

        "severity_codes":
            severity_codes,

        "field_codes":
            field_codes,
    }

    return (
        metrics,
        arrays,
    )


# ============================================================
# PRINT VALIDATION
# ============================================================

def print_validation_metrics(
    epoch,
    metrics,
):

    print(
        "\n" + "=" * 80
    )

    print(
        f"EPOCH {epoch} - "
        f"WORD-LEVEL VALIDATION"
    )

    print(
        "=" * 80
    )

    print(
        f"\nWord positions    : "
        f"{metrics['word_positions']:,}"
    )

    print(
        f"Error words       : "
        f"{metrics['error_words']:,}"
    )

    print(
        f"Error rate        : "
        f"{metrics['word_error_rate'] * 100:.4f}%"
    )

    print(
        f"\nWord Precision    : "
        f"{metrics['word_precision']:.6f}"
    )

    print(
        f"Word Recall       : "
        f"{metrics['word_recall']:.6f}"
    )

    print(
        f"Word F1           : "
        f"{metrics['word_f1']:.6f}"
    )

    print(
        f"Average Precision : "
        f"{metrics['average_precision']:.6f}"
    )

    print(
        f"ROC-AUC           : "
        f"{metrics['roc_auc']:.6f}"
    )

    print(
        f"\nWord coverage     : "
        f"{metrics['word_coverage']:.6f}"
    )

    print(
        f"Error coverage    : "
        f"{metrics['error_word_coverage']:.6f}"
    )

    print(
        f"\nTP: {metrics['tp']:,}"
    )

    print(
        f"FP: {metrics['fp']:,}"
    )

    print(
        f"FN: {metrics['fn']:,}"
    )

    print(
        "\nDocument localization:"
    )

    print(
        f"  >=1 error word detected : "
        f"{metrics['document_any_error_recall']:.6f}"
    )

    print(
        f"  all error words detected: "
        f"{metrics['document_all_errors_recall']:.6f}"
    )

    print(
        "\nRecall by severity:"
    )

    for severity, values in (
        metrics[
            "severity_recall"
        ].items()
    ):

        print(
            f"  {severity:10s}: "
            f"{values['recall']:.6f} "
            f"(n={values['error_words']:,})"
        )

    print(
        "\nRecall by field:"
    )

    for field, values in (
        metrics[
            "field_recall"
        ].items()
    ):

        print(
            f"  {field:10s}: "
            f"{values['recall']:.6f} "
            f"(n={values['error_words']:,})"
        )

    print(
        f"\nValidation time: "
        f"{metrics['validation_seconds']:.2f}s"
    )


# ============================================================
# SAVE BEST
# ============================================================

def save_best(
    model,
    tokenizer,
    metrics,
    arrays,
    epoch,
    source_checkpoint,
    positive_weight_info,
):

    checkpoint = {
        "model_name":
            source_checkpoint[
                "model_name"
            ],

        "task":
            "word_error_detection",

        "epoch":
            epoch,

        "source_checkpoint":
            str(
                SOURCE_CHECKPOINT_PATH
            ),

        "source_checkpoint_epoch":
            source_checkpoint.get(
                "epoch"
            ),

        "max_length":
            int(
                source_checkpoint[
                    "max_length"
                ]
            ),

        "stride":
            int(
                source_checkpoint[
                    "stride"
                ]
            ),

        "max_chunks_per_document":
            int(
                source_checkpoint[
                    "max_chunks_per_document"
                ]
            ),

        "freeze_bottom_layers":
            FREEZE_BOTTOM_LAYERS,

        "positive_weight":
            float(
                positive_weight_info[
                    "positive_weight"
                ]
            ),

        "checkpoint_selection_metric":
            "validation_average_precision",

        "metrics":
            metrics,

        "state_dict":
            model.state_dict(),
    }

    torch.save(
        checkpoint,
        BEST_MODEL_PATH,
    )

    tokenizer.save_pretrained(
        TOKENIZER_OUTPUT_DIR
    )

    with open(
        BEST_METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=2,
        )

    np.savez_compressed(
        BEST_VALIDATION_PROBS_PATH,

        probabilities=
            arrays[
                "probabilities"
            ].astype(
                np.float32
            ),

        labels=
            arrays[
                "labels"
            ].astype(
                np.uint8
            ),

        covered=
            arrays[
                "covered"
            ].astype(
                np.uint8
            ),

        severity_codes=
            arrays[
                "severity_codes"
            ].astype(
                np.uint8
            ),

        field_codes=
            arrays[
                "field_codes"
            ].astype(
                np.uint8
            ),
    )

    print(
        f"\nBest word detector saved:\n"
        f"{BEST_MODEL_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "VN-NEWSSPELL - PHOBERT "
        "WORD ERROR DETECTOR"
    )

    print("=" * 80)

    print(
        "\nTrain + validation ONLY."
    )

    print(
        "Test set is NOT read or used."
    )

    print(
        f"\nDevice: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    set_seed(
        RANDOM_SEED
    )

    # ========================================================
    # SOURCE
    # ========================================================

    source_checkpoint = (
        load_source_checkpoint()
    )

    model_name = (
        source_checkpoint[
            "model_name"
        ]
    )

    max_length = int(
        source_checkpoint[
            "max_length"
        ]
    )

    stride = int(
        source_checkpoint[
            "stride"
        ]
    )

    max_chunks = int(
        source_checkpoint[
            "max_chunks_per_document"
        ]
    )

    # ========================================================
    # CLASS IMBALANCE
    # ========================================================

    positive_weight_info = (
        calculate_positive_weight()
    )

    positive_weight = float(
        positive_weight_info[
            "positive_weight"
        ]
    )

    print(
        "\nClass imbalance:"
    )

    print(
        f"Positive words : "
        f"{positive_weight_info['positive_words']:,}"
    )

    print(
        f"Negative words : "
        f"{positive_weight_info['negative_words']:,}"
    )

    print(
        f"Negative/positive ratio: "
        f"{positive_weight_info['raw_negative_positive_ratio']:.2f}"
    )

    print(
        f"sqrt ratio POS_WEIGHT: "
        f"{positive_weight:.4f}"
    )

    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = (
        JsonlOffsetDataset(
            TRAIN_PATH
        )
    )

    validation_dataset = (
        JsonlOffsetDataset(
            VALIDATION_PATH
        )
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
            SOURCE_TOKENIZER_DIR,
            use_fast=False,
        )
    )

    print(
        f"Tokenizer fast: "
        f"{tokenizer.is_fast}"
    )

    train_collator = (
        WordDetectorCollator(
            tokenizer=
                tokenizer,

            max_length=
                max_length,

            stride=
                stride,

            max_chunks=
                max_chunks,

            deduplicate_training_targets=
                True,
        )
    )

    validation_collator = (
        WordDetectorCollator(
            tokenizer=
                tokenizer,

            max_length=
                max_length,

            stride=
                stride,

            max_chunks=
                max_chunks,

            deduplicate_training_targets=
                False,
        )
    )

    # ========================================================
    # LOADERS
    # ========================================================

    common_loader = {
        "num_workers":
            NUM_WORKERS,

        "pin_memory":
            USE_AMP,
    }

    if NUM_WORKERS > 0:

        common_loader[
            "persistent_workers"
        ] = True

        common_loader[
            "prefetch_factor"
        ] = PREFETCH_FACTOR

    train_loader = DataLoader(
        train_dataset,

        batch_size=
            TRAIN_DOCUMENT_BATCH_SIZE,

        shuffle=True,

        collate_fn=
            train_collator,

        **common_loader,
    )

    validation_loader = DataLoader(
        validation_dataset,

        batch_size=
            VALIDATION_DOCUMENT_BATCH_SIZE,

        shuffle=False,

        collate_fn=
            validation_collator,

        **common_loader,
    )

    print(
        f"\nTrain batches      : "
        f"{len(train_loader):,}"
    )

    print(
        f"Validation batches : "
        f"{len(validation_loader):,}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    print(
        "\nBuilding word detector..."
    )

    model = (
        PhoBERTWordDetector(
            model_name
        )
    )

    load_encoder_from_source(
        model,
        source_checkpoint,
    )

    freeze_lower_layers(
        model,
        FREEZE_BOTTOM_LAYERS,
    )

    print_parameter_stats(
        model
    )

    model.to(
        DEVICE
    )

    # ========================================================
    # LOSS
    # ========================================================

    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                positive_weight,
                dtype=torch.float32,
                device=DEVICE,
            )
        )
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
        model
        .word_classifier
        .parameters()
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

    # ========================================================
    # SCHEDULER
    # ========================================================

    optimizer_steps_per_epoch = (
        math.ceil(
            len(
                train_loader
            )
            /
            GRADIENT_ACCUMULATION_STEPS
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
        enabled=USE_AMP,
    )

    # ========================================================
    # SAVE CONFIG
    # ========================================================

    config = {
        "model_name":
            model_name,

        "task":
            "word_error_detection",

        "source_checkpoint":
            str(
                SOURCE_CHECKPOINT_PATH
            ),

        "epochs":
            EPOCHS,

        "max_length":
            max_length,

        "stride":
            stride,

        "max_chunks":
            max_chunks,

        "train_batch_size":
            TRAIN_DOCUMENT_BATCH_SIZE,

        "validation_batch_size":
            VALIDATION_DOCUMENT_BATCH_SIZE,

        "encoder_lr":
            ENCODER_LR,

        "head_lr":
            HEAD_LR,

        "freeze_bottom_layers":
            FREEZE_BOTTOM_LAYERS,

        "positive_weight_method":
            "sqrt_negative_positive_ratio",

        "positive_weight":
            positive_weight,

        "checkpoint_selection_metric":
            "validation_average_precision",

        "random_seed":
            RANDOM_SEED,
    }

    with open(
        CONFIG_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # TRAIN
    # ========================================================

    best_average_precision = -1.0

    history = []

    overall_start = time.time()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        print(
            "\n" + "#" * 80
        )

        print(
            f"EPOCH {epoch}/{EPOCHS}"
        )

        print(
            "#" * 80
        )

        model.train()

        if torch.cuda.is_available():

            torch.cuda.reset_peak_memory_stats()

        running_loss = 0.0
        running_targets = 0
        running_positive_targets = 0

        optimizer.zero_grad(
            set_to_none=True
        )

        epoch_start = time.time()

        progress = tqdm(
            enumerate(
                train_loader,
                start=1,
            ),

            total=
                len(
                    train_loader
                ),

            desc="Training",
        )

        for step, batch in progress:

            input_ids = (
                batch[
                    "input_ids"
                ]
                .to(
                    DEVICE,
                    non_blocking=True,
                )
            )

            attention_mask = (
                batch[
                    "attention_mask"
                ]
                .to(
                    DEVICE,
                    non_blocking=True,
                )
            )

            target_chunk_indices = (
                batch[
                    "target_chunk_indices"
                ]
                .to(
                    DEVICE,
                    non_blocking=True,
                )
            )

            target_token_positions = (
                batch[
                    "target_token_positions"
                ]
                .to(
                    DEVICE,
                    non_blocking=True,
                )
            )

            target_labels = (
                batch[
                    "target_labels"
                ]
                .to(
                    DEVICE,
                    non_blocking=True,
                )
            )

            if (
                target_labels.numel()
                == 0
            ):

                continue

            with autocast_context():

                logits = model(
                    input_ids=
                        input_ids,

                    attention_mask=
                        attention_mask,

                    target_chunk_indices=
                        target_chunk_indices,

                    target_token_positions=
                        target_token_positions,
                )

                loss = criterion(
                    logits,
                    target_labels,
                )

                scaled_loss = (
                    loss
                    /
                    GRADIENT_ACCUMULATION_STEPS
                )

            scaler.scale(
                scaled_loss
            ).backward()

            running_loss += float(
                loss.item()
            )

            running_targets += int(
                target_labels.numel()
            )

            running_positive_targets += int(
                torch.sum(
                    target_labels
                ).item()
            )

            should_step = (
                step
                %
                GRADIENT_ACCUMULATION_STEPS
                == 0
                or
                step
                == len(
                    train_loader
                )
            )

            if should_step:

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    MAX_GRAD_NORM,
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

                "positive":
                    (
                        f"{running_positive_targets / max(1, running_targets) * 100:.3f}%"
                    ),

                "lr":
                    f"{scheduler.get_last_lr()[0]:.2e}",
            })

        epoch_seconds = (
            time.time()
            - epoch_start
        )

        average_train_loss = (
            running_loss
            / max(
                1,
                len(
                    train_loader
                )
            )
        )

        print(
            f"\nAverage train loss: "
            f"{average_train_loss:.6f}"
        )

        print(
            f"Training targets: "
            f"{running_targets:,}"
        )

        print(
            f"Positive targets: "
            f"{running_positive_targets:,}"
        )

        print(
            f"Positive rate: "
            f"{running_positive_targets / max(1, running_targets) * 100:.4f}%"
        )

        print(
            f"Training time: "
            f"{epoch_seconds / 60:.2f} min"
        )

        print_gpu_memory()

        # ====================================================
        # VALIDATION
        # ====================================================

        (
            metrics,
            arrays,
        ) = evaluate(
            model,
            validation_loader,
            positive_weight,
        )

        metrics[
            "epoch"
        ] = epoch

        metrics[
            "average_train_loss"
        ] = average_train_loss

        metrics[
            "epoch_training_seconds"
        ] = epoch_seconds

        print_validation_metrics(
            epoch,
            metrics,
        )

        history.append(
            metrics
        )

        with open(
            HISTORY_PATH,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                history,
                f,
                ensure_ascii=False,
                indent=2,
            )

        # ====================================================
        # SAVE BEST BY THRESHOLD-INDEPENDENT AP
        # ====================================================

        current_ap = (
            metrics[
                "average_precision"
            ]
        )

        if (
            current_ap
            >
            best_average_precision
        ):

            best_average_precision = (
                current_ap
            )

            save_best(
                model=
                    model,

                tokenizer=
                    tokenizer,

                metrics=
                    metrics,

                arrays=
                    arrays,

                epoch=
                    epoch,

                source_checkpoint=
                    source_checkpoint,

                positive_weight_info=
                    positive_weight_info,
            )

        # Free large validation arrays before next epoch.
        del arrays

    # ========================================================
    # FINISH
    # ========================================================

    total_seconds = (
        time.time()
        - overall_start
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "WORD ERROR DETECTOR TRAINING FINISHED"
    )

    print(
        "=" * 80
    )

    print(
        f"\nBest validation Average Precision: "
        f"{best_average_precision:.6f}"
    )

    print(
        f"Total time: "
        f"{total_seconds / 60:.2f} minutes"
    )

    print(
        f"\nBest model:\n"
        f"{BEST_MODEL_PATH}"
    )

    print(
        f"\nBest validation probabilities:\n"
        f"{BEST_VALIDATION_PROBS_PATH}"
    )

    print(
        f"\nTraining history:\n"
        f"{HISTORY_PATH}"
    )

    print(
        "\nTest set was NOT read or used."
    )


if __name__ == "__main__":

    main()