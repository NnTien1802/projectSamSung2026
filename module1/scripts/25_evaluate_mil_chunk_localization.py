from pathlib import Path
from collections import Counter, defaultdict
from contextlib import nullcontext
from difflib import SequenceMatcher
import csv
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
    / "vn_newsspell_v1"
    / "validation.jsonl"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "module1_phobert_mil_all_fields_4chunks"
)

CHECKPOINT_PATH = (
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
    / "chunk_localization"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_JSON_PATH = (
    RESULT_DIR
    / "validation_chunk_localization_metrics.json"
)

DETAIL_CSV_PATH = (
    RESULT_DIR
    / "validation_chunk_localization_details.csv"
)


# ============================================================
# CONFIG
# ============================================================

NUM_WORKERS = 2
PREFETCH_FACTOR = 2
BATCH_SIZE = 16

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
# CHECKPOINT
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
        "text_fields",
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
            "Missing checkpoint keys: "
            + ", ".join(missing)
        )

    return checkpoint


# ============================================================
# THRESHOLD
# ============================================================

def load_threshold():

    with open(
        THRESHOLD_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    return float(
        data[
            "selected_threshold"
        ]
    )


# ============================================================
# VALIDATION CLEAN / NOISY PAIRS
# ============================================================

def load_validation_pairs():

    groups = {}

    with open(
        VALIDATION_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            row = json.loads(
                line
            )

            source_id = row[
                "source_id"
            ]

            label = int(
                row[
                    "label"
                ]
            )

            if source_id not in groups:

                groups[
                    source_id
                ] = {}

            if label == 0:

                groups[
                    source_id
                ][
                    "clean"
                ] = row

            elif label == 1:

                groups[
                    source_id
                ][
                    "noisy"
                ] = row

    pairs = []

    invalid = 0

    for source_id, group in (
        groups.items()
    ):

        if (
            "clean" not in group
            or
            "noisy" not in group
        ):

            invalid += 1
            continue

        pairs.append({
            "source_id":
                source_id,

            "clean":
                group[
                    "clean"
                ],

            "noisy":
                group[
                    "noisy"
                ],
        })

    if invalid:

        raise RuntimeError(
            f"Invalid clean/noisy groups: "
            f"{invalid:,}"
        )

    return pairs


# ============================================================
# BUILD EXACT DOCUMENT INPUT
# ============================================================

def build_document_text(
    row,
    text_fields,
    field_separator,
):

    parts = []

    for field in text_fields:

        text = row.get(
            field,
            "",
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


# ============================================================
# GET CHUNKS
# ============================================================

def get_chunk_ranges(
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
            "Invalid stride."
        )

    if total_tokens == 0:

        all_ranges = [
            {
                "chunk_index": 0,
                "start_token": 0,
                "end_token": 0,
            }
        ]

        return (
            all_ranges,
            all_ranges,
        )

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

    all_ranges = []

    for chunk_index, start in enumerate(
        starts
    ):

        all_ranges.append({
            "chunk_index":
                int(
                    chunk_index
                ),

            "start_token":
                int(
                    start
                ),

            "end_token":
                int(
                    min(
                        start
                        + content_length,
                        total_tokens,
                    )
                ),
        })

    selected_indices = (
        evenly_spaced_indices(
            len(
                all_ranges
            ),
            max_chunks,
        )
    )

    selected_ranges = [
        all_ranges[
            index
        ]
        for index in selected_indices
    ]

    return (
        all_ranges,
        selected_ranges,
    )


# ============================================================
# TOKEN DIFF
# ============================================================

def get_token_differences(
    clean_ids,
    noisy_ids,
):

    matcher = SequenceMatcher(
        a=clean_ids,
        b=noisy_ids,
        autojunk=False,
    )

    changes = []

    for (
        tag,
        i1,
        i2,
        j1,
        j2,
    ) in matcher.get_opcodes():

        if tag == "equal":
            continue

        changes.append({
            "tag":
                tag,

            "clean_start":
                int(
                    i1
                ),

            "clean_end":
                int(
                    i2
                ),

            "noisy_start":
                int(
                    j1
                ),

            "noisy_end":
                int(
                    j2
                ),
        })

    return changes


# ============================================================
# DOES CHANGE OVERLAP CHUNK?
# ============================================================

def change_overlaps_chunk(
    change,
    chunk,
    noisy_token_count,
):

    start = change[
        "noisy_start"
    ]

    end = change[
        "noisy_end"
    ]

    chunk_start = chunk[
        "start_token"
    ]

    chunk_end = chunk[
        "end_token"
    ]

    # --------------------------------------------------------
    # Replace / insert / token sequence change
    # --------------------------------------------------------

    if end > start:

        return (
            start < chunk_end
            and
            end > chunk_start
        )

    # --------------------------------------------------------
    # Deletion represented by zero-length noisy span.
    #
    # Use boundary around deleted position.
    # --------------------------------------------------------

    if noisy_token_count == 0:

        return True

    boundary_start = max(
        0,
        start - 1,
    )

    boundary_end = min(
        noisy_token_count,
        start + 1,
    )

    if boundary_end <= boundary_start:

        boundary_end = min(
            noisy_token_count,
            boundary_start + 1,
        )

    return (
        boundary_start < chunk_end
        and
        boundary_end > chunk_start
    )


# ============================================================
# DATASET
# ============================================================

class PairDataset(Dataset):

    def __init__(
        self,
        pairs,
    ):

        self.pairs = pairs

    def __len__(self):

        return len(
            self.pairs
        )

    def __getitem__(
        self,
        index,
    ):

        return self.pairs[
            index
        ]


# ============================================================
# COLLATOR
# ============================================================

class LocalizationCollator:

    def __init__(
        self,
        tokenizer,
        text_fields,
        field_separator,
        max_length,
        stride,
        max_chunks,
    ):

        self.tokenizer = tokenizer

        self.text_fields = (
            text_fields
        )

        self.field_separator = (
            field_separator
        )

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

    def __call__(
        self,
        batch,
    ):

        clean_texts = []

        noisy_texts = []

        for item in batch:

            clean_texts.append(
                build_document_text(
                    item[
                        "clean"
                    ],
                    self.text_fields,
                    self.field_separator,
                )
            )

            noisy_texts.append(
                build_document_text(
                    item[
                        "noisy"
                    ],
                    self.text_fields,
                    self.field_separator,
                )
            )

        # ====================================================
        # TOKENIZE CLEAN + NOISY
        # ====================================================

        clean_encoded = (
            self.tokenizer(
                clean_texts,
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )
        )

        noisy_encoded = (
            self.tokenizer(
                noisy_texts,
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )
        )

        clean_ids_batch = (
            clean_encoded[
                "input_ids"
            ]
        )

        noisy_ids_batch = (
            noisy_encoded[
                "input_ids"
            ]
        )

        chunk_features = []

        doc_indices = []

        chunk_ground_truth = []

        metadata = []

        # ====================================================
        # EACH DOCUMENT
        # ====================================================

        for doc_index, item in enumerate(
            batch
        ):

            clean_ids = (
                clean_ids_batch[
                    doc_index
                ]
            )

            noisy_ids = (
                noisy_ids_batch[
                    doc_index
                ]
            )

            changes = (
                get_token_differences(
                    clean_ids,
                    noisy_ids,
                )
            )

            (
                all_ranges,
                selected_ranges,
            ) = get_chunk_ranges(
                total_tokens=
                    len(
                        noisy_ids
                    ),

                content_length=
                    self.content_length,

                stride=
                    self.stride,

                max_chunks=
                    self.max_chunks,
            )

            gt_mask = []

            # =================================================
            # CREATE SELECTED CHUNKS
            # =================================================

            for selected_order, chunk in enumerate(
                selected_ranges
            ):

                start = chunk[
                    "start_token"
                ]

                end = chunk[
                    "end_token"
                ]

                content = noisy_ids[
                    start:end
                ]

                full_ids = (
                    self.tokenizer
                    .build_inputs_with_special_tokens(
                        content
                    )
                )

                contains_change = any(
                    change_overlaps_chunk(
                        change,
                        chunk,
                        len(
                            noisy_ids
                        ),
                    )
                    for change in changes
                )

                gt_mask.append(
                    bool(
                        contains_change
                    )
                )

                chunk_features.append({
                    "input_ids":
                        full_ids,

                    "attention_mask":
                        [1]
                        * len(
                            full_ids
                        ),
                })

                doc_indices.append(
                    doc_index
                )

                chunk_ground_truth.append(
                    int(
                        contains_change
                    )
                )

            noisy_row = item[
                "noisy"
            ]

            errors = noisy_row.get(
                "errors",
                [],
            )

            if errors is None:

                errors = []

            metadata.append({
                "source_id":
                    item[
                        "source_id"
                    ],

                "variant_id":
                    noisy_row.get(
                        "variant_id",
                        "",
                    ),

                "severity":
                    noisy_row.get(
                        "severity",
                        "unknown",
                    ),

                "error_count":
                    int(
                        noisy_row.get(
                            "error_count",
                            len(
                                errors
                            ),
                        )
                    ),

                "errors":
                    errors,

                "clean_token_count":
                    len(
                        clean_ids
                    ),

                "noisy_token_count":
                    len(
                        noisy_ids
                    ),

                "total_possible_chunks":
                    len(
                        all_ranges
                    ),

                "selected_ranges":
                    selected_ranges,

                "ground_truth_mask":
                    gt_mask,

                "changes":
                    changes,
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
                    dtype=torch.long,
                ),

            "chunk_ground_truth":
                torch.tensor(
                    chunk_ground_truth,
                    dtype=torch.long,
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

        self.classifier = nn.Linear(
            hidden_size,
            1,
        )

    def score_chunks(
        self,
        input_ids,
        attention_mask,
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

        return chunk_logits


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
# GROUP STATISTICS
# ============================================================

def update_group(
    stats,
    key,
    top1_hit,
    top2_hit,
):

    stats[
        key
    ][
        "total"
    ] += 1

    if top1_hit:

        stats[
            key
        ][
            "top1"
        ] += 1

    if top2_hit:

        stats[
            key
        ][
            "top2"
        ] += 1


def finalize_group(
    stats,
):

    output = {}

    for key, values in stats.items():

        total = values[
            "total"
        ]

        output[
            str(
                key
            )
        ] = {
            "total":
                total,

            "top1_hits":
                values[
                    "top1"
                ],

            "top1_accuracy":
                (
                    values[
                        "top1"
                    ]
                    / total
                    if total > 0
                    else None
                ),

            "top2_hits":
                values[
                    "top2"
                ],

            "top2_recall":
                (
                    values[
                        "top2"
                    ]
                    / total
                    if total > 0
                    else None
                ),
        }

    return output


# ============================================================
# PERCENTAGE
# ============================================================

def pct(
    value,
    total,
):

    if total == 0:

        return 0.0

    return (
        value
        / total
        * 100.0
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "PHOBERT MIL 4-CHUNKS - "
        "VALIDATION CHUNK LOCALIZATION"
    )

    print("=" * 80)

    print(
        "\nValidation ONLY."
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

    # ========================================================
    # LOAD CONFIG
    # ========================================================

    checkpoint = (
        load_checkpoint()
    )

    threshold = (
        load_threshold()
    )

    model_name = checkpoint[
        "model_name"
    ]

    text_fields = checkpoint[
        "text_fields"
    ]

    field_separator = (
        checkpoint.get(
            "field_separator",
            " . ",
        )
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

    print(
        f"\nCheckpoint epoch : "
        f"{checkpoint.get('epoch')}"
    )

    print(
        f"Model            : "
        f"{model_name}"
    )

    print(
        f"Locked threshold : "
        f"{threshold:.4f}"
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
        f"MAX_CHUNKS       : "
        f"{max_chunks}"
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    print(
        "\nLoading validation clean/noisy pairs..."
    )

    pairs = (
        load_validation_pairs()
    )

    print(
        f"Validation source pairs: "
        f"{len(pairs):,}"
    )

    dataset = PairDataset(
        pairs
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

    collator = LocalizationCollator(
        tokenizer=
            tokenizer,

        text_fields=
            text_fields,

        field_separator=
            field_separator,

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
            BATCH_SIZE,

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
        **loader_kwargs,
    )

    print(
        f"Validation batches: "
        f"{len(loader):,}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    print(
        "\nBuilding model..."
    )

    model = PhoBERTMIL(
        model_name
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
    # COUNTERS
    # ========================================================

    total_docs = 0

    visible_docs = 0
    invisible_docs = 0

    top1_hits = 0
    top2_hits = 0

    reciprocal_ranks = []

    competitive_docs = 0
    competitive_top1 = 0
    competitive_top2 = 0
    competitive_rr = []

    detected_visible_docs = 0
    detected_visible_top1 = 0

    missed_visible_docs = 0
    missed_visible_top1 = 0

    visible_detected = 0
    invisible_detected = 0

    by_severity = defaultdict(
        lambda: {
            "total": 0,
            "top1": 0,
            "top2": 0,
        }
    )

    by_error_type = defaultdict(
        lambda: {
            "total": 0,
            "top1": 0,
            "top2": 0,
        }
    )

    by_field = defaultdict(
        lambda: {
            "total": 0,
            "top1": 0,
            "top2": 0,
        }
    )

    by_error_count = defaultdict(
        lambda: {
            "total": 0,
            "top1": 0,
            "top2": 0,
        }
    )

    by_selected_chunks = defaultdict(
        lambda: {
            "total": 0,
            "top1": 0,
            "top2": 0,
        }
    )

    positive_chunk_scores = []
    negative_chunk_scores = []

    all_chunk_labels = []
    all_chunk_scores = []

    best_positive_scores = []
    max_negative_scores = []
    localization_margins = []

    detail_rows = []

    start_time = time.time()

    # ========================================================
    # INFERENCE
    # ========================================================

    with torch.inference_mode():

        progress = tqdm(
            loader,
            desc="Chunk localization",
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

            doc_indices_cpu = (
                batch[
                    "doc_indices"
                ]
                .numpy()
            )

            gt_cpu = (
                batch[
                    "chunk_ground_truth"
                ]
                .numpy()
            )

            with autocast_context():

                chunk_logits = (
                    model.score_chunks(
                        input_ids=
                            input_ids,

                        attention_mask=
                            attention_mask,
                    )
                )

            chunk_probs = (
                torch.sigmoid(
                    chunk_logits
                )
                .float()
                .cpu()
                .numpy()
            )

            # =================================================
            # EACH DOCUMENT IN BATCH
            # =================================================

            for doc_index, metadata in enumerate(
                batch[
                    "metadata"
                ]
            ):

                mask = (
                    doc_indices_cpu
                    == doc_index
                )

                doc_scores = (
                    chunk_probs[
                        mask
                    ]
                )

                doc_gt = (
                    gt_cpu[
                        mask
                    ].astype(
                        bool
                    )
                )

                selected_count = len(
                    doc_scores
                )

                total_docs += 1

                all_chunk_scores.extend(
                    doc_scores.tolist()
                )

                all_chunk_labels.extend(
                    doc_gt.astype(
                        int
                    ).tolist()
                )

                positive_chunk_scores.extend(
                    doc_scores[
                        doc_gt
                    ].tolist()
                )

                negative_chunk_scores.extend(
                    doc_scores[
                        ~doc_gt
                    ].tolist()
                )

                visible = bool(
                    np.any(
                        doc_gt
                    )
                )

                document_probability = float(
                    np.max(
                        doc_scores
                    )
                )

                detected = (
                    document_probability
                    >= threshold
                )

                if not visible:

                    invisible_docs += 1

                    if detected:

                        invisible_detected += 1

                    detail_rows.append({
                        "source_id":
                            metadata[
                                "source_id"
                            ],

                        "variant_id":
                            metadata[
                                "variant_id"
                            ],

                        "severity":
                            metadata[
                                "severity"
                            ],

                        "error_count":
                            metadata[
                                "error_count"
                            ],

                        "selected_chunks":
                            selected_count,

                        "visible_change":
                            0,

                        "document_probability":
                            document_probability,

                        "document_detected":
                            int(
                                detected
                            ),

                        "top1_hit":
                            "",

                        "top2_hit":
                            "",

                        "first_positive_rank":
                            "",

                        "chunk_scores":
                            json.dumps(
                                doc_scores.tolist()
                            ),

                        "ground_truth_mask":
                            json.dumps(
                                doc_gt.astype(
                                    int
                                ).tolist()
                            ),

                        "selected_ranges":
                            json.dumps(
                                metadata[
                                    "selected_ranges"
                                ],
                                ensure_ascii=False,
                            ),
                    })

                    continue

                # =================================================
                # VISIBLE DOCUMENT
                # =================================================

                visible_docs += 1

                if detected:

                    visible_detected += 1

                order = np.argsort(
                    -doc_scores
                )

                top1_index = int(
                    order[0]
                )

                top2_indices = (
                    order[
                        :min(
                            2,
                            len(
                                order
                            ),
                        )
                    ]
                )

                top1_hit = bool(
                    doc_gt[
                        top1_index
                    ]
                )

                top2_hit = bool(
                    np.any(
                        doc_gt[
                            top2_indices
                        ]
                    )
                )

                first_positive_rank = None

                for rank, chunk_index in enumerate(
                    order,
                    start=1,
                ):

                    if doc_gt[
                        chunk_index
                    ]:

                        first_positive_rank = (
                            rank
                        )

                        break

                if first_positive_rank is None:

                    raise RuntimeError(
                        "Visible document has no "
                        "positive rank."
                    )

                top1_hits += int(
                    top1_hit
                )

                top2_hits += int(
                    top2_hit
                )

                reciprocal_ranks.append(
                    1.0
                    / first_positive_rank
                )

                # =================================================
                # COMPETITIVE LOCALIZATION
                #
                # At least one positive chunk AND at least one
                # negative chunk.
                #
                # This removes trivial cases where every selected
                # chunk contains the changed region.
                # =================================================

                has_negative_chunk = bool(
                    np.any(
                        ~doc_gt
                    )
                )

                if has_negative_chunk:

                    competitive_docs += 1

                    competitive_top1 += int(
                        top1_hit
                    )

                    competitive_top2 += int(
                        top2_hit
                    )

                    competitive_rr.append(
                        1.0
                        / first_positive_rank
                    )

                # =================================================
                # DOCUMENT DETECTION x LOCALIZATION
                # =================================================

                if detected:

                    detected_visible_docs += 1

                    detected_visible_top1 += int(
                        top1_hit
                    )

                else:

                    missed_visible_docs += 1

                    missed_visible_top1 += int(
                        top1_hit
                    )

                # =================================================
                # SCORE ANALYSIS
                # =================================================

                pos_scores = (
                    doc_scores[
                        doc_gt
                    ]
                )

                neg_scores = (
                    doc_scores[
                        ~doc_gt
                    ]
                )

                best_positive = float(
                    np.max(
                        pos_scores
                    )
                )

                best_positive_scores.append(
                    best_positive
                )

                if len(
                    neg_scores
                ) > 0:

                    max_negative = float(
                        np.max(
                            neg_scores
                        )
                    )

                    max_negative_scores.append(
                        max_negative
                    )

                    localization_margins.append(
                        best_positive
                        - max_negative
                    )

                # =================================================
                # GROUPS
                # =================================================

                update_group(
                    by_severity,
                    metadata[
                        "severity"
                    ],
                    top1_hit,
                    top2_hit,
                )

                update_group(
                    by_error_count,
                    metadata[
                        "error_count"
                    ],
                    top1_hit,
                    top2_hit,
                )

                update_group(
                    by_selected_chunks,
                    selected_count,
                    top1_hit,
                    top2_hit,
                )

                errors = metadata[
                    "errors"
                ]

                error_types = set()

                error_fields = set()

                for error in errors:

                    error_types.add(
                        error.get(
                            "error_type",
                            "unknown",
                        )
                    )

                    error_fields.add(
                        error.get(
                            "field",
                            "unknown",
                        )
                    )

                for error_type in error_types:

                    update_group(
                        by_error_type,
                        error_type,
                        top1_hit,
                        top2_hit,
                    )

                for field in error_fields:

                    update_group(
                        by_field,
                        field,
                        top1_hit,
                        top2_hit,
                    )

                # =================================================
                # DETAILS
                # =================================================

                detail_rows.append({
                    "source_id":
                        metadata[
                            "source_id"
                        ],

                    "variant_id":
                        metadata[
                            "variant_id"
                        ],

                    "severity":
                        metadata[
                            "severity"
                        ],

                    "error_count":
                        metadata[
                            "error_count"
                        ],

                    "selected_chunks":
                        selected_count,

                    "visible_change":
                        1,

                    "document_probability":
                        document_probability,

                    "document_detected":
                        int(
                            detected
                        ),

                    "top1_hit":
                        int(
                            top1_hit
                        ),

                    "top2_hit":
                        int(
                            top2_hit
                        ),

                    "first_positive_rank":
                        first_positive_rank,

                    "chunk_scores":
                        json.dumps(
                            doc_scores.tolist()
                        ),

                    "ground_truth_mask":
                        json.dumps(
                            doc_gt.astype(
                                int
                            ).tolist()
                        ),

                    "selected_ranges":
                        json.dumps(
                            metadata[
                                "selected_ranges"
                            ],
                            ensure_ascii=False,
                        ),
                })

    elapsed = (
        time.time()
        - start_time
    )

    # ========================================================
    # METRICS
    # ========================================================

    top1_accuracy = (
        top1_hits
        / visible_docs
    )

    top2_recall = (
        top2_hits
        / visible_docs
    )

    mrr = float(
        np.mean(
            reciprocal_ranks
        )
    )

    competitive_top1_accuracy = (
        competitive_top1
        / competitive_docs
        if competitive_docs > 0
        else None
    )

    competitive_top2_recall = (
        competitive_top2
        / competitive_docs
        if competitive_docs > 0
        else None
    )

    competitive_mrr = (
        float(
            np.mean(
                competitive_rr
            )
        )
        if competitive_rr
        else None
    )

    chunk_auc = roc_auc_score(
        all_chunk_labels,
        all_chunk_scores,
    )

    chunk_ap = average_precision_score(
        all_chunk_labels,
        all_chunk_scores,
    )

    # ========================================================
    # PRINT
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "LOCALIZATION COVERAGE"
    )

    print(
        "=" * 80
    )

    print(
        f"\nNoisy validation documents : "
        f"{total_docs:,}"
    )

    print(
        f"Visible changed region      : "
        f"{visible_docs:,} "
        f"({pct(visible_docs, total_docs):.2f}%)"
    )

    print(
        f"No visible changed region   : "
        f"{invisible_docs:,} "
        f"({pct(invisible_docs, total_docs):.2f}%)"
    )

    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "CHUNK LOCALIZATION"
    )

    print(
        "=" * 80
    )

    print(
        f"\nTop-1 localization accuracy : "
        f"{top1_accuracy:.6f}"
    )

    print(
        f"Top-2 localization recall   : "
        f"{top2_recall:.6f}"
    )

    print(
        f"Mean Reciprocal Rank        : "
        f"{mrr:.6f}"
    )

    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "NON-TRIVIAL / COMPETITIVE DOCUMENTS"
    )

    print(
        "=" * 80
    )

    print(
        "\nThese documents contain both "
        "error and non-error selected chunks."
    )

    print(
        f"\nCompetitive documents       : "
        f"{competitive_docs:,}"
    )

    if competitive_top1_accuracy is not None:

        print(
            f"Top-1 accuracy              : "
            f"{competitive_top1_accuracy:.6f}"
        )

    if competitive_top2_recall is not None:

        print(
            f"Top-2 recall                : "
            f"{competitive_top2_recall:.6f}"
        )

    if competitive_mrr is not None:

        print(
            f"MRR                         : "
            f"{competitive_mrr:.6f}"
        )

    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "DOCUMENT DETECTION x LOCALIZATION"
    )

    print(
        "=" * 80
    )

    print(
        f"\nVisible docs detected as ERROR : "
        f"{detected_visible_docs:,}"
    )

    if detected_visible_docs > 0:

        print(
            f"Top-1 localization accuracy   : "
            f"{detected_visible_top1 / detected_visible_docs:.6f}"
        )

    print(
        f"\nVisible docs missed as CLEAN   : "
        f"{missed_visible_docs:,}"
    )

    if missed_visible_docs > 0:

        print(
            f"Top-1 localization accuracy   : "
            f"{missed_visible_top1 / missed_visible_docs:.6f}"
        )

    if invisible_docs > 0:

        print(
            f"\nInvisible docs still detected  : "
            f"{invisible_detected:,}/"
            f"{invisible_docs:,}"
        )

    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "CHUNK SCORE QUALITY"
    )

    print(
        "=" * 80
    )

    print(
        f"\nChunk-level ROC-AUC          : "
        f"{chunk_auc:.6f}"
    )

    print(
        f"Chunk-level AveragePrecision : "
        f"{chunk_ap:.6f}"
    )

    print(
        f"\nMean error-chunk probability : "
        f"{np.mean(positive_chunk_scores):.6f}"
    )

    print(
        f"Mean clean-chunk probability : "
        f"{np.mean(negative_chunk_scores):.6f}"
    )

    print(
        f"\nMean best error-chunk score  : "
        f"{np.mean(best_positive_scores):.6f}"
    )

    if max_negative_scores:

        print(
            f"Mean max clean-chunk score   : "
            f"{np.mean(max_negative_scores):.6f}"
        )

        print(
            f"Mean localization margin     : "
            f"{np.mean(localization_margins):.6f}"
        )

    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "LOCALIZATION BY SEVERITY"
    )

    print(
        "=" * 80
    )

    for severity, values in (
        finalize_group(
            by_severity
        ).items()
    ):

        print(
            f"\n{severity}"
        )

        print(
            f"  n     : "
            f"{values['total']:,}"
        )

        print(
            f"  Top-1 : "
            f"{values['top1_accuracy']:.6f}"
        )

        print(
            f"  Top-2 : "
            f"{values['top2_recall']:.6f}"
        )

    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "LOCALIZATION BY SELECTED CHUNK COUNT"
    )

    print(
        "=" * 80
    )

    selected_stats = (
        finalize_group(
            by_selected_chunks
        )
    )

    for count in sorted(
        selected_stats,
        key=int,
    ):

        values = (
            selected_stats[
                count
            ]
        )

        print(
            f"\n{count} selected chunks"
        )

        print(
            f"  n     : "
            f"{values['total']:,}"
        )

        print(
            f"  Top-1 : "
            f"{values['top1_accuracy']:.6f}"
        )

        print(
            f"  Top-2 : "
            f"{values['top2_recall']:.6f}"
        )

    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "LOCALIZATION BY FIELD"
    )

    print(
        "=" * 80
    )

    field_stats = (
        finalize_group(
            by_field
        )
    )

    for field, values in (
        field_stats.items()
    ):

        print(
            f"\n{field}"
        )

        print(
            f"  n     : "
            f"{values['total']:,}"
        )

        print(
            f"  Top-1 : "
            f"{values['top1_accuracy']:.6f}"
        )

        print(
            f"  Top-2 : "
            f"{values['top2_recall']:.6f}"
        )

    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "LOCALIZATION BY ERROR TYPE"
    )

    print(
        "=" * 80
    )

    error_type_stats = (
        finalize_group(
            by_error_type
        )
    )

    ordered_error_types = sorted(
        error_type_stats.items(),
        key=lambda item:
            item[1][
                "total"
            ],
        reverse=True,
    )

    for error_type, values in (
        ordered_error_types
    ):

        print(
            f"\n{error_type}"
        )

        print(
            f"  n     : "
            f"{values['total']:,}"
        )

        print(
            f"  Top-1 : "
            f"{values['top1_accuracy']:.6f}"
        )

        print(
            f"  Top-2 : "
            f"{values['top2_recall']:.6f}"
        )

    # ========================================================
    # SAVE CSV
    # ========================================================

    with open(
        DETAIL_CSV_PATH,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        fieldnames = [
            "source_id",
            "variant_id",
            "severity",
            "error_count",
            "selected_chunks",
            "visible_change",
            "document_probability",
            "document_detected",
            "top1_hit",
            "top2_hit",
            "first_positive_rank",
            "chunk_scores",
            "ground_truth_mask",
            "selected_ranges",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in detail_rows:

            writer.writerow(
                row
            )

    # ========================================================
    # SAVE JSON
    # ========================================================

    result = {
        "dataset":
            "validation",

        "test_used":
            False,

        "model":
            model_name,

        "checkpoint_epoch":
            checkpoint.get(
                "epoch"
            ),

        "threshold":
            threshold,

        "max_length":
            max_length,

        "stride":
            stride,

        "max_chunks":
            max_chunks,

        "documents": {
            "total_noisy":
                total_docs,

            "visible":
                visible_docs,

            "invisible":
                invisible_docs,

            "visible_percentage":
                pct(
                    visible_docs,
                    total_docs
                ),
        },

        "localization": {
            "top1_hits":
                top1_hits,

            "top1_accuracy":
                top1_accuracy,

            "top2_hits":
                top2_hits,

            "top2_recall":
                top2_recall,

            "mrr":
                mrr,
        },

        "competitive_localization": {
            "documents":
                competitive_docs,

            "top1_accuracy":
                competitive_top1_accuracy,

            "top2_recall":
                competitive_top2_recall,

            "mrr":
                competitive_mrr,
        },

        "detection_and_localization": {
            "visible_detected":
                detected_visible_docs,

            "visible_detected_top1_accuracy":
                (
                    detected_visible_top1
                    / detected_visible_docs
                    if detected_visible_docs
                    else None
                ),

            "visible_missed":
                missed_visible_docs,

            "visible_missed_top1_accuracy":
                (
                    missed_visible_top1
                    / missed_visible_docs
                    if missed_visible_docs
                    else None
                ),

            "invisible_detected":
                invisible_detected,
        },

        "chunk_score_quality": {
            "roc_auc":
                float(
                    chunk_auc
                ),

            "average_precision":
                float(
                    chunk_ap
                ),

            "mean_positive_chunk_probability":
                float(
                    np.mean(
                        positive_chunk_scores
                    )
                ),

            "mean_negative_chunk_probability":
                float(
                    np.mean(
                        negative_chunk_scores
                    )
                ),

            "mean_best_positive_probability":
                float(
                    np.mean(
                        best_positive_scores
                    )
                ),

            "mean_max_negative_probability":
                (
                    float(
                        np.mean(
                            max_negative_scores
                        )
                    )
                    if max_negative_scores
                    else None
                ),

            "mean_localization_margin":
                (
                    float(
                        np.mean(
                            localization_margins
                        )
                    )
                    if localization_margins
                    else None
                ),
        },

        "by_severity":
            finalize_group(
                by_severity
            ),

        "by_error_count":
            finalize_group(
                by_error_count
            ),

        "by_selected_chunks":
            finalize_group(
                by_selected_chunks
            ),

        "by_field":
            finalize_group(
                by_field
            ),

        "by_error_type":
            finalize_group(
                by_error_type
            ),

        "inference_seconds":
            elapsed,
    }

    with open(
        RESULT_JSON_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
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
        "CHUNK LOCALIZATION FINISHED"
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
        f"{RESULT_JSON_PATH}"
    )

    print(
        f"\nDetails:\n"
        f"{DETAIL_CSV_PATH}"
    )

    print(
        "\nTest set was NOT read or used."
    )


if __name__ == "__main__":
    main()