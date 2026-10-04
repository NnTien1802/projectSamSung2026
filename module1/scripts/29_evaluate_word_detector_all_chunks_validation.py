from pathlib import Path
from bisect import bisect_left
from contextlib import nullcontext
import json
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

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vn_newsspell_wordloc_v1"
    / "validation.jsonl"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "module1_phobert_word_detector_v1"
)

CHECKPOINT_PATH = (
    MODEL_DIR
    / "best_model.pt"
)

TOKENIZER_DIR = (
    MODEL_DIR
    / "tokenizer"
)

FOUR_CHUNK_THRESHOLD_PATH = (
    PROJECT_ROOT
    / "results"
    / "module1_phobert_word_detector_v1"
    / "threshold_tuning"
    / "best_threshold.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "module1_phobert_word_detector_v1"
    / "all_chunks_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

METRICS_PATH = (
    OUTPUT_DIR
    / "validation_all_chunks_metrics_at_locked_threshold.json"
)

PROBABILITIES_PATH = (
    OUTPUT_DIR
    / "validation_word_probabilities_all_chunks.npz"
)


# ============================================================
# CONFIG
# ============================================================

VALIDATION_BATCH_SIZE = 16

NUM_WORKERS = 2
PREFETCH_FACTOR = 2

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

USE_AMP = (
    DEVICE.type == "cuda"
)


# ============================================================
# CODES
# ============================================================

FIELD_TO_CODE = {
    "title": 1,
    "abstract": 2,
    "article": 3,
}

SEVERITY_TO_CODE = {
    "minor": 1,
    "moderate": 2,
    "severe": 3,
}


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
# DATASET
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

        print(
            f"Records: "
            f"{len(self.offsets):,}"
        )

        print(
            f"Index time: "
            f"{time.time() - start_time:.2f}s"
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
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint():

    print(
        f"\nLoading checkpoint:\n"
        f"{CHECKPOINT_PATH}"
    )

    try:

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
        )

    required = [
        "model_name",
        "max_length",
        "stride",
        "state_dict",
    ]

    missing = [
        key
        for key in required
        if key not in checkpoint
    ]

    if missing:

        raise RuntimeError(
            "Checkpoint missing: "
            + ", ".join(
                missing
            )
        )

    return checkpoint


# ============================================================
# LOAD LOCKED 4-CHUNK THRESHOLD
# ============================================================

def load_locked_threshold():

    with open(
        FOUR_CHUNK_THRESHOLD_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    threshold = float(
        data[
            "selected_threshold"
        ]
    )

    baseline_metrics = (
        data.get(
            "selected_metrics",
            {},
        )
    )

    return (
        threshold,
        baseline_metrics,
    )


# ============================================================
# ALL CHUNK RANGES
# ============================================================

def get_all_chunk_ranges(
    total_tokens,
    content_length,
    stride,
):

    step_size = (
        content_length
        - stride
    )

    if step_size <= 0:

        raise ValueError(
            "STRIDE must be smaller "
            "than usable content length."
        )

    if total_tokens == 0:

        return [
            {
                "chunk_index": 0,
                "start": 0,
                "end": 0,
            }
        ]

    ranges = []

    start = 0
    chunk_index = 0

    while True:

        end = min(
            start
            + content_length,
            total_tokens,
        )

        ranges.append({
            "chunk_index":
                chunk_index,

            "start":
                start,

            "end":
                end,
        })

        if end >= total_tokens:

            break

        start += (
            step_size
        )

        chunk_index += 1

    return ranges


# ============================================================
# FIND CONTENT OFFSET
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
        "Could not locate content inside "
        "special-token sequence."
    )


# ============================================================
# COLLATOR
# ============================================================

class AllChunksWordCollator:

    def __init__(
        self,
        tokenizer,
        max_length,
        stride,
    ):

        self.tokenizer = tokenizer

        self.max_length = int(
            max_length
        )

        self.stride = int(
            stride
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
        # TOKENIZE ALL WORDS
        # ====================================================

        flat_words = []

        document_slices = []

        for row in batch:

            start = len(
                flat_words
            )

            flat_words.extend(
                row[
                    "words"
                ]
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

        target_doc_indices = []
        target_word_indices = []

        metadata = []

        chunk_counts = []

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
                    "Word tokenization length mismatch."
                )

            # =================================================
            # FULL DOCUMENT TOKEN IDS
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
                        "Empty tokenization for word: "
                        + repr(
                            words[
                                word_index
                            ]
                        )
                    )

                word_token_starts.append(
                    len(
                        full_document_ids
                    )
                )

                full_document_ids.extend(
                    ids
                )

            # =================================================
            # IMPORTANT:
            # ALL CHUNKS, NO max_chunks LIMIT
            # =================================================

            all_ranges = (
                get_all_chunk_ranges(
                    total_tokens=
                        len(
                            full_document_ids
                        ),

                    content_length=
                        self.content_length,

                    stride=
                        self.stride,
                )
            )

            chunk_counts.append(
                len(
                    all_ranges
                )
            )

            # =================================================
            # BUILD CHUNKS
            # =================================================

            for chunk in all_ranges:

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

                global_chunk_index = (
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
                # First subtoken of words inside this chunk
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

                    if (
                        word_fields[
                            word_index
                        ]
                        == "separator"
                    ):

                        continue

                    global_token_start = (
                        word_token_starts[
                            word_index
                        ]
                    )

                    local_position = (
                        content_offset
                        +
                        (
                            global_token_start
                            - start
                        )
                    )

                    target_chunk_indices.append(
                        global_chunk_index
                    )

                    target_token_positions.append(
                        local_position
                    )

                    target_doc_indices.append(
                        doc_index
                    )

                    target_word_indices.append(
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
                    [
                        int(index)
                        for index
                        in row[
                            "error_word_indices"
                        ]
                    ],

                "all_chunk_count":
                    len(
                        all_ranges
                    ),
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

            "chunk_counts":
                chunk_counts,
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
# SAFE DIVIDE
# ============================================================

def safe_divide(
    numerator,
    denominator,
):

    if denominator == 0:

        return 0.0

    return (
        numerator
        / denominator
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "VN-NEWSSPELL - WORD DETECTOR "
        "ALL-CHUNKS VALIDATION"
    )

    print("=" * 80)

    print(
        "\nValidation ONLY."
    )

    print(
        "No training is performed."
    )

    print(
        "Test set is NOT read or used."
    )

    print(
        f"\nDevice: "
        f"{DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    checkpoint = (
        load_checkpoint()
    )

    model_name = (
        checkpoint[
            "model_name"
        ]
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

    print(
        f"\nCheckpoint epoch : "
        f"{checkpoint.get('epoch')}"
    )

    print(
        f"Model            : "
        f"{model_name}"
    )

    print(
        f"Task             : "
        f"{checkpoint.get('task')}"
    )

    print(
        f"MAX_LENGTH       : "
        f"{max_length}"
    )

    print(
        f"STRIDE           : "
        f"{stride}"
    )

    print(
        "\nMAX_CHUNKS       : ALL"
    )

    # ========================================================
    # LOCKED THRESHOLD
    # ========================================================

    (
        locked_threshold,
        four_chunk_baseline,
    ) = load_locked_threshold()

    print(
        f"\nLocked 4-chunk threshold: "
        f"{locked_threshold:.4f}"
    )

    print(
        "Threshold is NOT tuned in this script."
    )

    # ========================================================
    # DATA
    # ========================================================

    validation_dataset = (
        JsonlOffsetDataset(
            VALIDATION_PATH
        )
    )

    # ========================================================
    # TOKENIZER
    # ========================================================

    print(
        "\nLoading tokenizer..."
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            TOKENIZER_DIR,
            use_fast=False,
        )
    )

    print(
        f"Tokenizer fast: "
        f"{tokenizer.is_fast}"
    )

    collator = (
        AllChunksWordCollator(
            tokenizer=
                tokenizer,

            max_length=
                max_length,

            stride=
                stride,
        )
    )

    loader_kwargs = {
        "batch_size":
            VALIDATION_BATCH_SIZE,

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

    validation_loader = (
        DataLoader(
            validation_dataset,
            **loader_kwargs,
        )
    )

    print(
        f"Validation batches: "
        f"{len(validation_loader):,}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    print(
        "\nBuilding model..."
    )

    model = (
        PhoBERTWordDetector(
            model_name
        )
    )

    load_result = (
        model.load_state_dict(
            checkpoint[
                "state_dict"
            ],
            strict=True,
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
    # OUTPUT ARRAYS
    # ========================================================

    probability_batches = []
    label_batches = []
    covered_batches = []
    severity_batches = []
    field_batches = []

    total_documents = 0

    total_chunks = 0

    max_chunks_in_document = 0

    chunk_count_histogram = {}

    documents_any_error_detected = 0
    documents_all_errors_detected = 0

    start_time = time.time()

    # ========================================================
    # INFERENCE
    # ========================================================

    with torch.inference_mode():

        progress = tqdm(
            validation_loader,
            desc="All-chunks validation",
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
            # CHUNK STATS
            # =================================================

            for count in batch[
                "chunk_counts"
            ]:

                count = int(
                    count
                )

                total_chunks += (
                    count
                )

                max_chunks_in_document = max(
                    max_chunks_in_document,
                    count,
                )

                chunk_count_histogram[
                    count
                ] = (
                    chunk_count_histogram.get(
                        count,
                        0,
                    )
                    + 1
                )

            # =================================================
            # EACH DOCUMENT
            # =================================================

            for doc_index, metadata in enumerate(
                batch[
                    "metadata"
                ]
            ):

                total_documents += 1

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

                occurrence_mask = (
                    occurrence_doc_indices
                    == doc_index
                )

                words_for_doc = (
                    occurrence_word_indices[
                        occurrence_mask
                    ]
                )

                probs_for_doc = (
                    occurrence_probs[
                        occurrence_mask
                    ]
                )

                # =============================================
                # SAME WORD MAY APPEAR IN OVERLAPPING CHUNKS
                #
                # Aggregate with MAX probability.
                # =============================================

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

                # =============================================
                # LABELS
                # =============================================

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

                # =============================================
                # FIELD
                # =============================================

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

                # Exclude synthetic "." separators.
                eval_mask = (
                    field_codes
                    != 0
                )

                # =============================================
                # SEVERITY
                # =============================================

                severity_code = (
                    SEVERITY_TO_CODE.get(
                        metadata[
                            "severity"
                        ],
                        0,
                    )
                )

                evaluated_word_count = int(
                    np.sum(
                        eval_mask
                    )
                )

                severity_codes = np.full(
                    evaluated_word_count,
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
                    ].astype(
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

                # =============================================
                # DOCUMENT LOCALIZATION @ LOCKED THRESHOLD
                # =============================================

                if len(
                    error_indices
                ) > 0:

                    error_probs = (
                        doc_probs[
                            error_indices
                        ]
                    )

                    predicted_errors = (
                        error_probs
                        >= locked_threshold
                    )

                    if np.any(
                        predicted_errors
                    ):

                        documents_any_error_detected += 1

                    if np.all(
                        predicted_errors
                    ):

                        documents_all_errors_detected += 1

    elapsed = (
        time.time()
        - start_time
    )

    # ========================================================
    # CONCAT
    # ========================================================

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
        >= locked_threshold
    )

    positive = (
        labels == 1
    )

    negative = (
        labels == 0
    )

    # ========================================================
    # METRICS
    # ========================================================

    tp = int(
        np.sum(
            predictions
            &
            positive
        )
    )

    fp = int(
        np.sum(
            predictions
            &
            negative
        )
    )

    fn = int(
        np.sum(
            (~predictions)
            &
            positive
        )
    )

    tn = int(
        np.sum(
            (~predictions)
            &
            negative
        )
    )

    precision = (
        precision_score(
            labels,
            predictions,
            pos_label=1,
            zero_division=0,
        )
    )

    recall = (
        recall_score(
            labels,
            predictions,
            pos_label=1,
            zero_division=0,
        )
    )

    word_f1 = (
        f1_score(
            labels,
            predictions,
            pos_label=1,
            zero_division=0,
        )
    )

    roc_auc = (
        roc_auc_score(
            labels,
            probabilities,
        )
    )

    average_precision = (
        average_precision_score(
            labels,
            probabilities,
        )
    )

    specificity = (
        safe_divide(
            tn,
            tn + fp,
        )
    )

    # ========================================================
    # COVERAGE
    # ========================================================

    covered_word_count = int(
        np.sum(
            covered == 1
        )
    )

    error_word_count = int(
        np.sum(
            positive
        )
    )

    covered_error_word_count = int(
        np.sum(
            positive
            &
            (
                covered == 1
            )
        )
    )

    uncovered_error_word_count = (
        error_word_count
        - covered_error_word_count
    )

    word_coverage = safe_divide(
        covered_word_count,
        len(
            labels
        ),
    )

    error_word_coverage = safe_divide(
        covered_error_word_count,
        error_word_count,
    )

    # ========================================================
    # COVERED ERROR RECALL
    # ========================================================

    covered_error_mask = (
        positive
        &
        (
            covered == 1
        )
    )

    covered_error_recall = (
        safe_divide(
            int(
                np.sum(
                    predictions
                    &
                    covered_error_mask
                )
            ),
            int(
                np.sum(
                    covered_error_mask
                )
            ),
        )
    )

    # ========================================================
    # RECALL BY SEVERITY
    # ========================================================

    severity_results = {}

    for name, code in (
        SEVERITY_TO_CODE.items()
    ):

        mask = (
            positive
            &
            (
                severity_codes
                == code
            )
        )

        count = int(
            np.sum(
                mask
            )
        )

        value = (
            float(
                np.mean(
                    predictions[
                        mask
                    ]
                )
            )
            if count > 0
            else None
        )

        severity_results[
            name
        ] = {
            "error_words":
                count,

            "recall":
                value,
        }

    # ========================================================
    # RECALL BY FIELD
    # ========================================================

    field_results = {}

    for name, code in (
        FIELD_TO_CODE.items()
    ):

        mask = (
            positive
            &
            (
                field_codes
                == code
            )
        )

        count = int(
            np.sum(
                mask
            )
        )

        value = (
            float(
                np.mean(
                    predictions[
                        mask
                    ]
                )
            )
            if count > 0
            else None
        )

        field_results[
            name
        ] = {
            "error_words":
                count,

            "recall":
                value,
        }

    # ========================================================
    # CHUNK STATS
    # ========================================================

    average_chunks = (
        total_chunks
        / total_documents
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "ALL-CHUNKS COVERAGE"
    )

    print(
        "=" * 80
    )

    print(
        f"\nDocuments             : "
        f"{total_documents:,}"
    )

    print(
        f"Total chunks          : "
        f"{total_chunks:,}"
    )

    print(
        f"Average chunks/doc    : "
        f"{average_chunks:.4f}"
    )

    print(
        f"Max chunks/document   : "
        f"{max_chunks_in_document:,}"
    )

    print(
        f"\nWord positions        : "
        f"{len(labels):,}"
    )

    print(
        f"Word coverage         : "
        f"{word_coverage:.6f}"
    )

    print(
        f"Error words           : "
        f"{error_word_count:,}"
    )

    print(
        f"Covered error words   : "
        f"{covered_error_word_count:,}"
    )

    print(
        f"Uncovered error words : "
        f"{uncovered_error_word_count:,}"
    )

    print(
        f"Error-word coverage   : "
        f"{error_word_coverage:.6f}"
    )

    print(
        "\nChunk count distribution:"
    )

    for chunk_count in sorted(
        chunk_count_histogram
    ):

        print(
            f"  {chunk_count:2d} chunks: "
            f"{chunk_count_histogram[chunk_count]:,}"
        )

    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "ALL-CHUNKS METRICS @ LOCKED 4-CHUNK THRESHOLD"
    )

    print(
        "=" * 80
    )

    print(
        f"\nThreshold             : "
        f"{locked_threshold:.4f}"
    )

    print(
        f"Word Precision        : "
        f"{precision:.6f}"
    )

    print(
        f"Word Recall           : "
        f"{recall:.6f}"
    )

    print(
        f"Word F1               : "
        f"{word_f1:.6f}"
    )

    print(
        f"Specificity           : "
        f"{specificity:.6f}"
    )

    print(
        f"Covered-error Recall  : "
        f"{covered_error_recall:.6f}"
    )

    print(
        f"ROC-AUC               : "
        f"{roc_auc:.6f}"
    )

    print(
        f"Average Precision     : "
        f"{average_precision:.6f}"
    )

    print(
        f"\nTP: {tp:,}"
    )

    print(
        f"FP: {fp:,}"
    )

    print(
        f"FN: {fn:,}"
    )

    print(
        f"TN: {tn:,}"
    )

    # ========================================================

    print(
        "\nDocument localization:"
    )

    print(
        f"  >=1 error word detected : "
        f"{documents_any_error_detected / total_documents:.6f}"
    )

    print(
        f"  all error words detected: "
        f"{documents_all_errors_detected / total_documents:.6f}"
    )

    # ========================================================

    print(
        "\nRecall by severity:"
    )

    for name, values in (
        severity_results.items()
    ):

        print(
            f"  {name:10s}: "
            f"{values['recall']:.6f} "
            f"(n={values['error_words']:,})"
        )

    print(
        "\nRecall by field:"
    )

    for name, values in (
        field_results.items()
    ):

        print(
            f"  {name:10s}: "
            f"{values['recall']:.6f} "
            f"(n={values['error_words']:,})"
        )

    # ========================================================
    # FAIR COMPARISON AGAINST 4 CHUNKS
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "4-CHUNK vs ALL-CHUNKS @ SAME THRESHOLD"
    )

    print(
        "=" * 80
    )

    if four_chunk_baseline:

        baseline_precision = float(
            four_chunk_baseline.get(
                "precision",
                0.0,
            )
        )

        baseline_recall = float(
            four_chunk_baseline.get(
                "recall",
                0.0,
            )
        )

        baseline_f1 = float(
            four_chunk_baseline.get(
                "f1",
                0.0,
            )
        )

        baseline_fp = int(
            four_chunk_baseline.get(
                "fp",
                0,
            )
        )

        baseline_fn = int(
            four_chunk_baseline.get(
                "fn",
                0,
            )
        )

        print(
            f"\n{'Metric':20s}"
            f"{'4 chunks':>14s}"
            f"{'All chunks':>14s}"
            f"{'Delta':>14s}"
        )

        print(
            "-" * 62
        )

        print(
            f"{'Precision':20s}"
            f"{baseline_precision:14.6f}"
            f"{precision:14.6f}"
            f"{precision - baseline_precision:+14.6f}"
        )

        print(
            f"{'Recall':20s}"
            f"{baseline_recall:14.6f}"
            f"{recall:14.6f}"
            f"{recall - baseline_recall:+14.6f}"
        )

        print(
            f"{'F1':20s}"
            f"{baseline_f1:14.6f}"
            f"{word_f1:14.6f}"
            f"{word_f1 - baseline_f1:+14.6f}"
        )

        print(
            f"{'FP':20s}"
            f"{baseline_fp:14,d}"
            f"{fp:14,d}"
            f"{fp - baseline_fp:+14,d}"
        )

        print(
            f"{'FN':20s}"
            f"{baseline_fn:14,d}"
            f"{fn:14,d}"
            f"{fn - baseline_fn:+14,d}"
        )

    else:

        print(
            "\n4-chunk baseline metrics "
            "not found in threshold JSON."
        )

    # ========================================================
    # SAVE PROBABILITIES
    #
    # Bước 30 sẽ tune threshold từ file này.
    # ========================================================

    np.savez_compressed(
        PROBABILITIES_PATH,

        probabilities=
            probabilities.astype(
                np.float32
            ),

        labels=
            labels.astype(
                np.uint8
            ),

        covered=
            covered.astype(
                np.uint8
            ),

        severity_codes=
            severity_codes.astype(
                np.uint8
            ),

        field_codes=
            field_codes.astype(
                np.uint8
            ),
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics = {
        "dataset":
            "validation",

        "test_used":
            False,

        "training_performed":
            False,

        "mode":
            "all_chunks",

        "checkpoint_epoch":
            checkpoint.get(
                "epoch"
            ),

        "locked_threshold":
            locked_threshold,

        "threshold_source":
            "4_chunk_validation_best_word_f1",

        "max_length":
            max_length,

        "stride":
            stride,

        "max_chunks":
            "all",

        "documents":
            total_documents,

        "total_chunks":
            total_chunks,

        "average_chunks_per_document":
            float(
                average_chunks
            ),

        "max_chunks_per_document_observed":
            max_chunks_in_document,

        "chunk_count_distribution": {
            str(key):
                value
            for key, value
            in sorted(
                chunk_count_histogram.items()
            )
        },

        "word_positions":
            int(
                len(
                    labels
                )
            ),

        "error_words":
            error_word_count,

        "covered_error_words":
            covered_error_word_count,

        "uncovered_error_words":
            uncovered_error_word_count,

        "word_coverage":
            float(
                word_coverage
            ),

        "error_word_coverage":
            float(
                error_word_coverage
            ),

        "precision":
            float(
                precision
            ),

        "recall":
            float(
                recall
            ),

        "f1":
            float(
                word_f1
            ),

        "specificity":
            float(
                specificity
            ),

        "covered_error_recall":
            float(
                covered_error_recall
            ),

        "roc_auc":
            float(
                roc_auc
            ),

        "average_precision":
            float(
                average_precision
            ),

        "tp":
            tp,

        "fp":
            fp,

        "fn":
            fn,

        "tn":
            tn,

        "document_any_error_recall":
            float(
                documents_any_error_detected
                / total_documents
            ),

        "document_all_errors_recall":
            float(
                documents_all_errors_detected
                / total_documents
            ),

        "severity_recall":
            severity_results,

        "field_recall":
            field_results,

        "runtime_seconds":
            float(
                elapsed
            ),
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # FINISH
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "ALL-CHUNKS VALIDATION FINISHED"
    )

    print(
        "=" * 80
    )

    print(
        f"\nRuntime: "
        f"{elapsed:.2f}s"
    )

    print(
        f"\nMetrics:\n"
        f"{METRICS_PATH}"
    )

    print(
        f"\nProbabilities for Step 30:\n"
        f"{PROBABILITIES_PATH}"
    )

    print(
        "\nNo training was performed."
    )

    print(
        "Threshold was NOT tuned."
    )

    print(
        "Test set was NOT read or used."
    )


if __name__ == "__main__":

    main()