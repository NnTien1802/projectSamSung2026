from pathlib import Path
from bisect import bisect_left
from contextlib import nullcontext
import argparse
import json
import math
import unicodedata

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
)


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# DOCUMENT DETECTOR
# ============================================================

DOCUMENT_MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "module1_phobert_mil_all_fields_4chunks"
)

DOCUMENT_CHECKPOINT_PATH = (
    DOCUMENT_MODEL_DIR
    / "best_model.pt"
)

DOCUMENT_TOKENIZER_DIR = (
    DOCUMENT_MODEL_DIR
    / "tokenizer"
)

DOCUMENT_THRESHOLD_PATH = (
    PROJECT_ROOT
    / "results"
    / "module1_phobert_mil_all_fields_4chunks"
    / "threshold_tuning"
    / "best_threshold.json"
)


# ============================================================
# WORD DETECTOR
# ============================================================

WORD_MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "module1_phobert_word_detector_v1"
)

WORD_CHECKPOINT_PATH = (
    WORD_MODEL_DIR
    / "best_model.pt"
)

WORD_TOKENIZER_DIR = (
    WORD_MODEL_DIR
    / "tokenizer"
)

WORD_THRESHOLD_PATH = (
    PROJECT_ROOT
    / "results"
    / "module1_phobert_word_detector_v1"
    / "all_chunks_threshold_tuning"
    / "best_threshold.json"
)


# ============================================================
# CORRECTION LOOKUP
# ============================================================

CORRECTION_LOOKUP_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vn_newsspell_correction_v1"
    / "correction_lookup.json"
)


# ============================================================
# CONTEXTUAL RERANKER
# ============================================================

RERANKER_SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "module1_correction_reranker_v1"
    / "validation_reranking_summary.json"
)

MLM_MODEL_NAME = "vinai/phobert-base-v2"

MLM_MAX_LENGTH = 256

CONTEXT_WINDOW = 8


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
# FIELDS
# ============================================================

TEXT_FIELDS = [
    "title_segmented",
    "abstract_segmented",
    "article_segmented",
]

FIELD_NAMES = {
    "title_segmented": "title",
    "abstract_segmented": "abstract",
    "article_segmented": "article",
}

FIELD_SEPARATOR = " . "


# ============================================================
# NORMALIZATION
# ============================================================

def nfc(text):

    return unicodedata.normalize(
        "NFC",
        str(text),
    )


def canonical(text):

    return (
        nfc(text)
        .casefold()
    )


# ============================================================
# RESTORE CASE
# ============================================================

def restore_case(
    source,
    candidate,
):

    if not source:

        return candidate

    if source.isupper():

        return candidate.upper()

    if (
        len(source) > 1
        and
        source[0].isupper()
        and
        source[1:].islower()
    ):

        return (
            candidate[:1].upper()
            + candidate[1:]
        )

    if (
        len(source) == 1
        and
        source.isupper()
    ):

        return candidate.upper()

    return candidate


# ============================================================
# PUNCTUATION / CORE SPLIT
# ============================================================

def is_core_character(char):

    category = (
        unicodedata.category(
            char
        )
    )

    return (
        category.startswith("L")
        or
        category.startswith("M")
        or
        category.startswith("N")
    )


def split_affixes(text):

    text = nfc(
        text
    )

    if not text:

        return (
            "",
            "",
            "",
        )

    start = 0

    while (
        start < len(text)
        and
        not is_core_character(
            text[start]
        )
    ):

        start += 1

    end = len(
        text
    )

    while (
        end > start
        and
        not is_core_character(
            text[
                end - 1
            ]
        )
    ):

        end -= 1

    return (
        text[:start],
        text[start:end],
        text[end:],
    )


# ============================================================
# LOAD TORCH CHECKPOINT
# ============================================================

def load_torch_checkpoint(path):

    try:

        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:

        return torch.load(
            path,
            map_location="cpu",
        )


# ============================================================
# LOAD THRESHOLD
# ============================================================

def load_threshold(path):

    with open(
        path,
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
# LOAD RERANKER LAMBDA
# ============================================================

def load_reranker_lambda():

    with open(
        RERANKER_SUMMARY_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    return float(
        data[
            "selected_lambda"
        ]
    )


# ============================================================
# LOAD CORRECTION LOOKUP
# ============================================================

def load_correction_lookup():

    with open(
        CORRECTION_LOOKUP_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    return (
        data[
            "exact_lookup"
        ],
        data[
            "component_lookup"
        ],
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
# DOCUMENT MODEL
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
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls_embeddings = (
            outputs
            .last_hidden_state[
                :,
                0,
                :
            ]
        )

        logits = (
            self.classifier(
                self.dropout(
                    cls_embeddings
                )
            )
            .squeeze(-1)
        )

        return logits


# ============================================================
# WORD MODEL
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
            input_ids=input_ids,
            attention_mask=attention_mask,
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
# DOCUMENT CHUNK SELECTION
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
# BUILD DOCUMENT TEXT
# ============================================================

def build_document_text(data):

    parts = []

    for field in TEXT_FIELDS:

        text = data.get(
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

    return FIELD_SEPARATOR.join(
        parts
    )


# ============================================================
# DOCUMENT CHUNKS
# ============================================================

def build_document_chunks(
    text,
    tokenizer,
    max_length,
    stride,
    max_chunks,
):

    token_ids = (
        tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )[
            "input_ids"
        ]
    )

    special_tokens = (
        tokenizer
        .num_special_tokens_to_add(
            pair=False
        )
    )

    content_length = (
        max_length
        - special_tokens
    )

    step_size = (
        content_length
        - stride
    )

    if step_size <= 0:

        raise ValueError(
            "Invalid STRIDE."
        )

    chunks = []

    if len(token_ids) == 0:

        chunks.append(
            tokenizer
            .build_inputs_with_special_tokens(
                []
            )
        )

    else:

        start = 0

        while True:

            content = token_ids[
                start:
                start
                + content_length
            ]

            chunk_ids = (
                tokenizer
                .build_inputs_with_special_tokens(
                    content
                )
            )

            chunks.append(
                chunk_ids
            )

            if (
                start
                + content_length
                >= len(token_ids)
            ):

                break

            start += (
                step_size
            )

    selected_indices = (
        evenly_spaced_indices(
            len(
                chunks
            ),
            max_chunks,
        )
    )

    return [
        chunks[index]
        for index in selected_indices
    ]


# ============================================================
# DOCUMENT PREDICTION
# ============================================================

def predict_document(
    data,
    model,
    tokenizer,
    checkpoint,
    threshold,
):

    text = build_document_text(
        data
    )

    max_chunks = int(
        checkpoint.get(
            "max_chunks_per_document",
            checkpoint.get(
                "max_chunks",
                4,
            ),
        )
    )

    chunks = build_document_chunks(
        text=text,

        tokenizer=tokenizer,

        max_length=int(
            checkpoint[
                "max_length"
            ]
        ),

        stride=int(
            checkpoint[
                "stride"
            ]
        ),

        max_chunks=max_chunks,
    )

    features = [
        {
            "input_ids":
                chunk,

            "attention_mask":
                [1] * len(
                    chunk
                ),
        }
        for chunk in chunks
    ]

    padded = tokenizer.pad(
        features,
        padding=True,
        return_tensors="pt",
    )

    input_ids = (
        padded[
            "input_ids"
        ]
        .to(
            DEVICE
        )
    )

    attention_mask = (
        padded[
            "attention_mask"
        ]
        .to(
            DEVICE
        )
    )

    with torch.inference_mode():

        with autocast_context():

            logits = model.score_chunks(
                input_ids=
                    input_ids,

                attention_mask=
                    attention_mask,
            )

    probabilities = (
        torch.sigmoid(
            logits
        )
        .float()
        .cpu()
        .numpy()
    )

    document_probability = float(
        np.max(
            probabilities
        )
    )

    predicted_error = (
        document_probability
        >= threshold
    )

    return {
        "probability":
            document_probability,

        "prediction":
            (
                "ERROR"
                if predicted_error
                else "CLEAN"
            ),

        "is_error":
            bool(
                predicted_error
            ),

        "chunk_probabilities":
            [
                float(value)
                for value
                in probabilities
            ],

        "chunks_used":
            len(
                probabilities
            ),
    }


# ============================================================
# BUILD WORDS
# ============================================================

def build_words(data):

    words = []

    fields = []

    local_indices = []

    nonempty_fields = []

    for field in TEXT_FIELDS:

        text = data.get(
            field,
            "",
        )

        if text is None:

            text = ""

        text = str(
            text
        ).strip()

        if not text:

            continue

        field_words = (
            text.split()
        )

        if field_words:

            nonempty_fields.append(
                (
                    field,
                    field_words,
                )
            )

    for field_number, (
        field,
        field_words,
    ) in enumerate(
        nonempty_fields
    ):

        if field_number > 0:

            words.append(
                "."
            )

            fields.append(
                "separator"
            )

            local_indices.append(
                -1
            )

        simple_field = (
            FIELD_NAMES[
                field
            ]
        )

        for local_index, word in enumerate(
            field_words
        ):

            words.append(
                word
            )

            fields.append(
                simple_field
            )

            local_indices.append(
                local_index
            )

    return (
        words,
        fields,
        local_indices,
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
            "Invalid STRIDE."
        )

    if total_tokens == 0:

        return []

    ranges = []

    start = 0

    while True:

        end = min(
            start
            + content_length,
            total_tokens,
        )

        ranges.append(
            (
                start,
                end,
            )
        )

        if end >= total_tokens:

            break

        start += (
            step_size
        )

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
        "Could not locate content tokens."
    )


# ============================================================
# LOCAL CONTEXT
# ============================================================

def get_local_context(
    words,
    fields,
    target_index,
    window,
):

    target_field = (
        fields[
            target_index
        ]
    )

    left = target_index

    count = 0

    while (
        left > 0
        and
        count < window
    ):

        previous_index = (
            left - 1
        )

        if (
            fields[
                previous_index
            ]
            != target_field
        ):

            break

        left -= 1
        count += 1

    right = (
        target_index + 1
    )

    count = 0

    while (
        right
        < len(
            words
        )
        and
        count < window
    ):

        if (
            fields[
                right
            ]
            != target_field
        ):

            break

        right += 1
        count += 1

    context_words = list(
        words[
            left:right
        ]
    )

    target_position = (
        target_index
        - left
    )

    return (
        context_words,
        target_position,
    )


# ============================================================
# CONTEXT DISPLAY
# ============================================================

def build_context_display(
    words,
    fields,
    target_index,
    window=5,
):

    (
        context_words,
        _
    ) = get_local_context(
        words=
            words,

        fields=
            fields,

        target_index=
            target_index,

        window=
            window,
    )

    return " ".join(
        context_words
    )


# ============================================================
# WORD PREDICTION
# ============================================================

def predict_words(
    data,
    model,
    tokenizer,
    checkpoint,
    threshold,
    top_k,
):

    (
        words,
        fields,
        local_indices,
    ) = build_words(
        data
    )

    if not words:

        return {
            "words": [],
            "fields": [],
            "predicted_errors": [],
            "top_candidates": [],
            "chunks_used": 0,
            "words_scanned": 0,
        }

    encoded_words = (
        tokenizer(
            words,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )[
            "input_ids"
        ]
    )

    full_document_ids = []

    word_token_starts = []

    for word_index, ids in enumerate(
        encoded_words
    ):

        if (
            fields[
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

    special_tokens = (
        tokenizer
        .num_special_tokens_to_add(
            pair=False
        )
    )

    content_length = (
        max_length
        - special_tokens
    )

    ranges = get_all_chunk_ranges(
        total_tokens=
            len(
                full_document_ids
            ),

        content_length=
            content_length,

        stride=
            stride,
    )

    chunk_features = []

    target_chunk_indices = []

    target_token_positions = []

    target_word_indices = []

    for chunk_index, (
        start,
        end,
    ) in enumerate(
        ranges
    ):

        content_ids = (
            full_document_ids[
                start:end
            ]
        )

        chunk_ids = (
            tokenizer
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

        chunk_features.append({
            "input_ids":
                chunk_ids,

            "attention_mask":
                [1] * len(
                    chunk_ids
                ),
        })

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
                fields[
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
                chunk_index
            )

            target_token_positions.append(
                local_position
            )

            target_word_indices.append(
                word_index
            )

    padded = tokenizer.pad(
        chunk_features,
        padding=True,
        return_tensors="pt",
    )

    input_ids = (
        padded[
            "input_ids"
        ]
        .to(
            DEVICE
        )
    )

    attention_mask = (
        padded[
            "attention_mask"
        ]
        .to(
            DEVICE
        )
    )

    target_chunk_indices_tensor = (
        torch.tensor(
            target_chunk_indices,
            dtype=torch.long,
            device=DEVICE,
        )
    )

    target_token_positions_tensor = (
        torch.tensor(
            target_token_positions,
            dtype=torch.long,
            device=DEVICE,
        )
    )

    with torch.inference_mode():

        with autocast_context():

            logits = model(
                input_ids=
                    input_ids,

                attention_mask=
                    attention_mask,

                target_chunk_indices=
                    target_chunk_indices_tensor,

                target_token_positions=
                    target_token_positions_tensor,
            )

    occurrence_probabilities = (
        torch.sigmoid(
            logits
        )
        .float()
        .cpu()
        .numpy()
    )

    word_probabilities = np.zeros(
        len(
            words
        ),
        dtype=np.float32,
    )

    target_word_indices_np = np.asarray(
        target_word_indices,
        dtype=np.int64,
    )

    np.maximum.at(
        word_probabilities,
        target_word_indices_np,
        occurrence_probabilities,
    )

    candidates = []

    for word_index, word in enumerate(
        words
    ):

        if (
            fields[
                word_index
            ]
            == "separator"
        ):

            continue

        probability = float(
            word_probabilities[
                word_index
            ]
        )

        context = (
            build_context_display(
                words=
                    words,

                fields=
                    fields,

                target_index=
                    word_index,

                window=5,
            )
        )

        candidates.append({
            "field":
                fields[
                    word_index
                ],

            "word_index_global":
                word_index,

            "word_index_in_field":
                local_indices[
                    word_index
                ],

            "word_segmented":
                word,

            "word_display":
                word.replace(
                    "_",
                    " ",
                ),

            "probability":
                probability,

            "is_error":
                bool(
                    probability
                    >= threshold
                ),

            "context_segmented":
                context,

            "context_display":
                context.replace(
                    "_",
                    " ",
                ),
        })

    ranked = sorted(
        candidates,
        key=lambda item:
            item[
                "probability"
            ],
        reverse=True,
    )

    predicted_errors = [
        item
        for item
        in ranked
        if item[
            "probability"
        ]
        >= threshold
    ]

    return {
        "words":
            words,

        "fields":
            fields,

        "predicted_errors":
            predicted_errors,

        "top_candidates":
            ranked[
                :top_k
            ],

        "chunks_used":
            len(
                ranges
            ),

        "words_scanned":
            len(
                candidates
            ),
    }


# ============================================================
# COMPONENT CANDIDATES
# ============================================================

def get_component_candidates(
    noisy_word,
    component_lookup,
):

    parts = nfc(
        noisy_word
    ).split(
        "_"
    )

    candidates = {}

    for component_index, part in enumerate(
        parts
    ):

        (
            prefix,
            core,
            suffix,
        ) = split_affixes(
            part
        )

        if not core:

            continue

        key = canonical(
            core
        )

        entries = (
            component_lookup.get(
                key
            )
        )

        if not entries:

            continue

        for entry in entries:

            corrected_core = (
                restore_case(
                    core,
                    entry[
                        "candidate"
                    ],
                )
            )

            corrected_part = (
                prefix
                + corrected_core
                + suffix
            )

            corrected_parts = list(
                parts
            )

            corrected_parts[
                component_index
            ] = corrected_part

            candidate = "_".join(
                corrected_parts
            )

            if (
                canonical(
                    candidate
                )
                ==
                canonical(
                    noisy_word
                )
            ):

                continue

            value = {
                "candidate":
                    nfc(
                        candidate
                    ),

                "method":
                    "component",

                "component_index":
                    component_index,

                "support":
                    int(
                        entry[
                            "count"
                        ]
                    ),

                "lookup_probability":
                    float(
                        entry[
                            "probability"
                        ]
                    ),
            }

            previous = candidates.get(
                candidate
            )

            if previous is None:

                candidates[
                    candidate
                ] = value

            elif (
                value[
                    "lookup_probability"
                ]
                >
                previous[
                    "lookup_probability"
                ]
            ):

                candidates[
                    candidate
                ] = value

            elif (
                value[
                    "lookup_probability"
                ]
                ==
                previous[
                    "lookup_probability"
                ]
                and
                value[
                    "support"
                ]
                >
                previous[
                    "support"
                ]
            ):

                candidates[
                    candidate
                ] = value

    result = list(
        candidates.values()
    )

    result.sort(
        key=lambda item: (
            item[
                "lookup_probability"
            ],
            item[
                "support"
            ],
        ),
        reverse=True,
    )

    return result[:5]


# ============================================================
# CORRECTION CANDIDATES
# ============================================================

def get_correction_candidates(
    noisy_word,
    exact_lookup,
    component_lookup,
):

    noisy_word = nfc(
        noisy_word
    )

    key = canonical(
        noisy_word
    )

    exact_entries = (
        exact_lookup.get(
            key
        )
    )

    if exact_entries:

        candidates = []

        for entry in exact_entries:

            candidate = (
                restore_case(
                    noisy_word,
                    entry[
                        "candidate"
                    ],
                )
            )

            candidates.append({
                "candidate":
                    candidate,

                "method":
                    "exact",

                "component_index":
                    None,

                "support":
                    int(
                        entry[
                            "count"
                        ]
                    ),

                "lookup_probability":
                    float(
                        entry[
                            "probability"
                        ]
                    ),

                "mlm_score":
                    None,

                "final_score":
                    None,
            })

        return candidates

    return get_component_candidates(
        noisy_word=
            noisy_word,

        component_lookup=
            component_lookup,
    )


# ============================================================
# BUILD MLM FEATURE
# ============================================================

def build_mlm_feature(
    prefix_ids,
    suffix_ids,
    candidate,
    tokenizer,
):

    candidate_ids = (
        tokenizer(
            candidate,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )[
            "input_ids"
        ]
    )

    if not candidate_ids:

        return None

    mask_count = len(
        candidate_ids
    )

    special_tokens = (
        tokenizer
        .num_special_tokens_to_add(
            pair=False
        )
    )

    content_length = (
        MLM_MAX_LENGTH
        - special_tokens
    )

    available_context = (
        content_length
        - mask_count
    )

    if available_context < 0:

        return None

    left_budget = (
        available_context
        // 2
    )

    right_budget = (
        available_context
        - left_budget
    )

    left_ids = (
        prefix_ids[
            -left_budget:
        ]
        if left_budget > 0
        else []
    )

    right_ids = (
        suffix_ids[
            :right_budget
        ]
        if right_budget > 0
        else []
    )

    unused_left = (
        left_budget
        - len(
            left_ids
        )
    )

    unused_right = (
        right_budget
        - len(
            right_ids
        )
    )

    if (
        unused_left > 0
        and
        len(
            suffix_ids
        )
        >
        len(
            right_ids
        )
    ):

        extra = suffix_ids[
            len(
                right_ids
            ):
            len(
                right_ids
            )
            + unused_left
        ]

        right_ids = (
            right_ids
            + extra
        )

    if (
        unused_right > 0
        and
        len(
            prefix_ids
        )
        >
        len(
            left_ids
        )
    ):

        desired = min(
            len(
                prefix_ids
            ),
            len(
                left_ids
            )
            + unused_right,
        )

        left_ids = prefix_ids[
            -desired:
        ]

    mask_ids = [
        tokenizer.mask_token_id
    ] * mask_count

    content_ids = (
        left_ids
        + mask_ids
        + right_ids
    )

    full_ids = (
        tokenizer
        .build_inputs_with_special_tokens(
            content_ids
        )
    )

    content_offset = (
        find_content_offset(
            full_ids,
            content_ids,
        )
    )

    mask_start = (
        content_offset
        + len(
            left_ids
        )
    )

    mask_positions = list(
        range(
            mask_start,
            mask_start
            + mask_count,
        )
    )

    return {
        "input_ids":
            full_ids,

        "attention_mask":
            [1] * len(
                full_ids
            ),

        "candidate_ids":
            candidate_ids,

        "mask_positions":
            mask_positions,
    }


# ============================================================
# SCORE ONE COMPONENT CANDIDATE
# ============================================================

def score_component_candidate(
    candidate,
    context_words,
    target_position,
    tokenizer,
    model,
):

    encoded_words = (
        tokenizer(
            context_words,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )[
            "input_ids"
        ]
    )

    prefix_ids = []

    for ids in encoded_words[
        :target_position
    ]:

        prefix_ids.extend(
            ids
        )

    suffix_ids = []

    for ids in encoded_words[
        target_position
        + 1:
    ]:

        suffix_ids.extend(
            ids
        )

    feature = build_mlm_feature(
        prefix_ids=
            prefix_ids,

        suffix_ids=
            suffix_ids,

        candidate=
            candidate,

        tokenizer=
            tokenizer,
    )

    if feature is None:

        return float(
            "-inf"
        )

    input_ids = torch.tensor(
        [
            feature[
                "input_ids"
            ]
        ],
        dtype=torch.long,
        device=DEVICE,
    )

    attention_mask = torch.tensor(
        [
            feature[
                "attention_mask"
            ]
        ],
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.inference_mode():

        with autocast_context():

            output = model(
                input_ids=
                    input_ids,

                attention_mask=
                    attention_mask,
            )

    logits = (
        output.logits[
            0
        ]
    )

    mask_positions = (
        feature[
            "mask_positions"
        ]
    )

    candidate_ids = (
        feature[
            "candidate_ids"
        ]
    )

    selected_logits = logits[
        mask_positions,
        :
    ]

    log_probs = F.log_softmax(
        selected_logits.float(),
        dim=-1,
    )

    target_ids = torch.tensor(
        candidate_ids,
        dtype=torch.long,
        device=DEVICE,
    )

    token_scores = log_probs[
        torch.arange(
            len(
                candidate_ids
            ),
            device=DEVICE,
        ),
        target_ids,
    ]

    return float(
        token_scores
        .mean()
        .detach()
        .cpu()
    )


# ============================================================
# CORRECT ONE ERROR WORD
# ============================================================

def correct_error_word(
    error_item,
    words,
    fields,
    exact_lookup,
    component_lookup,
    mlm_tokenizer,
    mlm_model,
    reranker_lambda,
):

    noisy_word = (
        error_item[
            "word_segmented"
        ]
    )

    candidates = (
        get_correction_candidates(
            noisy_word=
                noisy_word,

            exact_lookup=
                exact_lookup,

            component_lookup=
                component_lookup,
        )
    )

    if not candidates:

        return {
            "suggestion":
                None,

            "method":
                "none",

            "lookup_probability":
                None,

            "support":
                None,

            "mlm_score":
                None,

            "final_score":
                None,

            "candidates":
                [],
        }

    # ========================================================
    # EXACT:
    #
    # Keep original lookup ordering.
    # Do not apply MLM.
    # ========================================================

    if (
        candidates[
            0
        ][
            "method"
        ]
        == "exact"
    ):

        best = candidates[
            0
        ]

        return {
            "suggestion":
                best[
                    "candidate"
                ],

            "method":
                "exact_lookup",

            "lookup_probability":
                best[
                    "lookup_probability"
                ],

            "support":
                best[
                    "support"
                ],

            "mlm_score":
                None,

            "final_score":
                None,

            "candidates":
                candidates,
        }

    # ========================================================
    # COMPONENT:
    #
    # PhoBERT contextual reranking.
    # ========================================================

    target_index = int(
        error_item[
            "word_index_global"
        ]
    )

    (
        context_words,
        target_position,
    ) = get_local_context(
        words=
            words,

        fields=
            fields,

        target_index=
            target_index,

        window=
            CONTEXT_WINDOW,
    )

    reranked = []

    for candidate in candidates:

        mlm_score = (
            score_component_candidate(
                candidate=
                    candidate[
                        "candidate"
                    ],

                context_words=
                    context_words,

                target_position=
                    target_position,

                tokenizer=
                    mlm_tokenizer,

                model=
                    mlm_model,
            )
        )

        lookup_probability = max(
            candidate[
                "lookup_probability"
            ],
            1e-12,
        )

        lookup_log_score = math.log(
            lookup_probability
        )

        if np.isfinite(
            mlm_score
        ):

            final_score = (
                lookup_log_score
                +
                reranker_lambda
                * mlm_score
            )

        else:

            final_score = float(
                "-inf"
            )

        reranked.append({
            **candidate,

            "lookup_log_score":
                float(
                    lookup_log_score
                ),

            "mlm_score":
                float(
                    mlm_score
                ),

            "final_score":
                float(
                    final_score
                ),
        })

    reranked.sort(
        key=lambda item:
            item[
                "final_score"
            ],
        reverse=True,
    )

    best = reranked[
        0
    ]

    return {
        "suggestion":
            best[
                "candidate"
            ],

        "method":
            "component_contextual_mlm",

        "lookup_probability":
            best[
                "lookup_probability"
            ],

        "support":
            best[
                "support"
            ],

        "mlm_score":
            best[
                "mlm_score"
            ],

        "final_score":
            best[
                "final_score"
            ],

        "candidates":
            reranked,
    }


# ============================================================
# INPUT
# ============================================================

def load_input(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    aliases = {
        "title":
            "title_segmented",

        "abstract":
            "abstract_segmented",

        "article":
            "article_segmented",
    }

    for source, target in (
        aliases.items()
    ):

        if (
            target not in data
            and
            source in data
        ):

            data[
                target
            ] = data[
                source
            ]

    for field in TEXT_FIELDS:

        if field not in data:

            data[
                field
            ] = ""

    return data


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Vietnamese spelling error detection, "
            "localization and correction suggestion."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Input JSON containing segmented "
            "title/abstract/article."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output JSON."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--force-word-scan",
        action="store_true",
    )

    args = parser.parse_args()

    print("=" * 80)

    print(
        "VN-NEWSSPELL - FULL SPELLING PIPELINE"
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

    # ========================================================
    # INPUT
    # ========================================================

    input_path = Path(
        args.input
    )

    data = load_input(
        input_path
    )

    if not any(
        str(
            data.get(
                field,
                "",
            )
        ).strip()
        for field in TEXT_FIELDS
    ):

        raise ValueError(
            "Input document is empty."
        )

    # ========================================================
    # LOCKED CONFIG
    # ========================================================

    document_threshold = (
        load_threshold(
            DOCUMENT_THRESHOLD_PATH
        )
    )

    word_threshold = (
        load_threshold(
            WORD_THRESHOLD_PATH
        )
    )

    reranker_lambda = (
        load_reranker_lambda()
    )

    print(
        "\nLocked configuration:"
    )

    print(
        f"  document threshold : "
        f"{document_threshold:.4f}"
    )

    print(
        f"  word threshold     : "
        f"{word_threshold:.4f}"
    )

    print(
        f"  correction lambda  : "
        f"{reranker_lambda:.4f}"
    )

    print(
        f"  correction context : "
        f"{CONTEXT_WINDOW} words/side"
    )

    # ========================================================
    # DOCUMENT MODEL
    # ========================================================

    print(
        "\nLoading document detector..."
    )

    document_checkpoint = (
        load_torch_checkpoint(
            DOCUMENT_CHECKPOINT_PATH
        )
    )

    document_tokenizer = (
        AutoTokenizer
        .from_pretrained(
            DOCUMENT_TOKENIZER_DIR,
            use_fast=False,
        )
    )

    document_model = (
        PhoBERTMIL(
            document_checkpoint[
                "model_name"
            ]
        )
    )

    load_result = (
        document_model
        .load_state_dict(
            document_checkpoint[
                "state_dict"
            ],
            strict=True,
        )
    )

    print(
        f"Document missing keys    : "
        f"{len(load_result.missing_keys)}"
    )

    print(
        f"Document unexpected keys : "
        f"{len(load_result.unexpected_keys)}"
    )

    document_model.to(
        DEVICE
    )

    document_model.eval()

    # ========================================================
    # STAGE 1
    # ========================================================

    document_result = (
        predict_document(
            data=
                data,

            model=
                document_model,

            tokenizer=
                document_tokenizer,

            checkpoint=
                document_checkpoint,

            threshold=
                document_threshold,
        )
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "STAGE 1 - DOCUMENT DETECTION"
    )

    print(
        "=" * 80
    )

    print(
        f"\nThreshold  : "
        f"{document_threshold:.4f}"
    )

    print(
        f"Probability: "
        f"{document_result['probability']:.6f}"
    )

    print(
        f"Prediction : "
        f"{document_result['prediction']}"
    )

    print(
        f"Chunks used: "
        f"{document_result['chunks_used']}"
    )

    final_result = {
        "input":
            str(
                input_path
            ),

        "configuration": {
            "document_threshold":
                document_threshold,

            "word_threshold":
                word_threshold,

            "correction_lambda":
                reranker_lambda,

            "correction_context_window":
                CONTEXT_WINDOW,
        },

        "document_detector":
            document_result,

        "word_detector":
            None,

        "corrections":
            [],
    }

    should_scan_words = (
        document_result[
            "is_error"
        ]
        or
        args.force_word_scan
    )

    if not should_scan_words:

        print(
            "\nDocument predicted CLEAN."
        )

        print(
            "Word detector and correction "
            "were skipped."
        )

    else:

        # ====================================================
        # RELEASE DOCUMENT MODEL
        # ====================================================

        del document_model

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        # ====================================================
        # WORD MODEL
        # ====================================================

        print(
            "\nLoading word detector..."
        )

        word_checkpoint = (
            load_torch_checkpoint(
                WORD_CHECKPOINT_PATH
            )
        )

        word_tokenizer = (
            AutoTokenizer
            .from_pretrained(
                WORD_TOKENIZER_DIR,
                use_fast=False,
            )
        )

        word_model = (
            PhoBERTWordDetector(
                word_checkpoint[
                    "model_name"
                ]
            )
        )

        load_result = (
            word_model
            .load_state_dict(
                word_checkpoint[
                    "state_dict"
                ],
                strict=True,
            )
        )

        print(
            f"Word missing keys    : "
            f"{len(load_result.missing_keys)}"
        )

        print(
            f"Word unexpected keys : "
            f"{len(load_result.unexpected_keys)}"
        )

        word_model.to(
            DEVICE
        )

        word_model.eval()

        # ====================================================
        # STAGE 2
        # ====================================================

        word_result = (
            predict_words(
                data=
                    data,

                model=
                    word_model,

                tokenizer=
                    word_tokenizer,

                checkpoint=
                    word_checkpoint,

                threshold=
                    word_threshold,

                top_k=
                    max(
                        1,
                        args.top_k,
                    ),
            )
        )

        words = (
            word_result[
                "words"
            ]
        )

        fields = (
            word_result[
                "fields"
            ]
        )

        final_result[
            "word_detector"
        ] = {
            key:
                value
            for key, value
            in word_result.items()
            if key not in {
                "words",
                "fields",
            }
        }

        print(
            "\n" + "=" * 80
        )

        print(
            "STAGE 2 - WORD ERROR LOCALIZATION"
        )

        print(
            "=" * 80
        )

        print(
            f"\nThreshold    : "
            f"{word_threshold:.4f}"
        )

        print(
            f"Chunks used  : "
            f"{word_result['chunks_used']}"
        )

        print(
            f"Words scanned: "
            f"{word_result['words_scanned']:,}"
        )

        print(
            f"Errors found : "
            f"{len(word_result['predicted_errors'])}"
        )

        # ====================================================
        # RELEASE WORD MODEL
        # ====================================================

        del word_model

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        # ====================================================
        # CORRECTION LOOKUP
        # ====================================================

        print(
            "\nLoading correction lookup..."
        )

        (
            exact_lookup,
            component_lookup,
        ) = load_correction_lookup()

        # ====================================================
        # Determine whether MLM is needed.
        # ====================================================

        preliminary = []

        need_mlm = False

        for error_item in (
            word_result[
                "predicted_errors"
            ]
        ):

            candidates = (
                get_correction_candidates(
                    noisy_word=
                        error_item[
                            "word_segmented"
                        ],

                    exact_lookup=
                        exact_lookup,

                    component_lookup=
                        component_lookup,
                )
            )

            preliminary.append(
                candidates
            )

            if (
                candidates
                and
                candidates[
                    0
                ][
                    "method"
                ]
                == "component"
            ):

                need_mlm = True

        # ====================================================
        # LOAD MLM ONLY IF REQUIRED
        # ====================================================

        mlm_tokenizer = None

        mlm_model = None

        if need_mlm:

            print(
                "\nLoading PhoBERT contextual "
                "correction reranker..."
            )

            mlm_tokenizer = (
                AutoTokenizer
                .from_pretrained(
                    WORD_TOKENIZER_DIR,
                    use_fast=False,
                )
            )

            mlm_model = (
                AutoModelForMaskedLM
                .from_pretrained(
                    MLM_MODEL_NAME
                )
            )

            mlm_model.to(
                DEVICE
            )

            mlm_model.eval()

        # ====================================================
        # STAGE 3
        # ====================================================

        print(
            "\n" + "=" * 80
        )

        print(
            "STAGE 3 - CORRECTION SUGGESTIONS"
        )

        print(
            "=" * 80
        )

        print(
            f"\nReranker lambda : "
            f"{reranker_lambda:.4f}"
        )

        print(
            f"Context window   : "
            f"{CONTEXT_WINDOW}"
        )

        corrections = []

        for rank, error_item in enumerate(
            word_result[
                "predicted_errors"
            ],
            start=1,
        ):

            correction = (
                correct_error_word(
                    error_item=
                        error_item,

                    words=
                        words,

                    fields=
                        fields,

                    exact_lookup=
                        exact_lookup,

                    component_lookup=
                        component_lookup,

                    mlm_tokenizer=
                        mlm_tokenizer,

                    mlm_model=
                        mlm_model,

                    reranker_lambda=
                        reranker_lambda,
                )
            )

            result_item = {
                **error_item,

                **correction,
            }

            corrections.append(
                result_item
            )

            print(
                "\n"
                + "-" * 75
            )

            print(
                f"#{rank}"
            )

            print(
                f"Field       : "
                f"{error_item['field']}"
            )

            print(
                f"Word        : "
                f"{error_item['word_segmented']}"
            )

            print(
                f"Error prob  : "
                f"{error_item['probability']:.6f}"
            )

            print(
                f"Context     : "
                f"{error_item['context_segmented']}"
            )

            if (
                correction[
                    "suggestion"
                ]
                is None
            ):

                print(
                    "Suggestion  : NONE"
                )

                print(
                    "Method      : no candidate"
                )

            else:

                print(
                    f"Suggestion  : "
                    f"{correction['suggestion']}"
                )

                print(
                    f"Display     : "
                    f"{correction['suggestion'].replace('_', ' ')}"
                )

                print(
                    f"Method      : "
                    f"{correction['method']}"
                )

                print(
                    f"Lookup prob : "
                    f"{correction['lookup_probability']:.6f}"
                )

                print(
                    f"Support     : "
                    f"{correction['support']}"
                )

                if (
                    correction[
                        "mlm_score"
                    ]
                    is not None
                ):

                    print(
                        f"MLM score   : "
                        f"{correction['mlm_score']:.6f}"
                    )

                    print(
                        f"Final score : "
                        f"{correction['final_score']:.6f}"
                    )

        final_result[
            "corrections"
        ] = corrections

        # ====================================================
        # TOP WORD CANDIDATES
        # ====================================================

        print(
            "\n" + "=" * 80
        )

        print(
            f"TOP "
            f"{len(word_result['top_candidates'])} "
            f"WORD CANDIDATES"
        )

        print(
            "=" * 80
        )

        for rank, item in enumerate(
            word_result[
                "top_candidates"
            ],
            start=1,
        ):

            status = (
                "ERROR"
                if item[
                    "is_error"
                ]
                else "below threshold"
            )

            print(
                f"\n{rank:2d}. "
                f"[{item['field']}] "
                f"{item['word_segmented']} "
                f"| p="
                f"{item['probability']:.6f} "
                f"| {status}"
            )

        # ====================================================
        # RELEASE MLM
        # ====================================================

        if mlm_model is not None:

            del mlm_model

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

    # ========================================================
    # SAVE OUTPUT
    # ========================================================

    if args.output is not None:

        output_path = Path(
            args.output
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                final_result,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(
            f"\nOutput saved:\n"
            f"{output_path}"
        )

    print(
        "\n" + "=" * 80
    )

    print(
        "FULL SPELLING PIPELINE FINISHED"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()