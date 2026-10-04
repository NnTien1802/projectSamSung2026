from pathlib import Path
from bisect import bisect_left
from contextlib import nullcontext
import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from transformers import (
    AutoConfig,
    AutoModel,
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
# INPUT FIELDS
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
# EVEN CHUNK SELECTION FOR DOCUMENT MODEL
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

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )

    token_ids = (
        encoded[
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

            full_ids = (
                tokenizer
                .build_inputs_with_special_tokens(
                    content
                )
            )

            chunks.append(
                full_ids
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
        chunks[
            index
        ]
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

    text = (
        build_document_text(
            data
        )
    )

    chunks = (
        build_document_chunks(
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

            max_chunks=int(
                checkpoint[
                    "max_chunks_per_document"
                ]
            ),
        )
    )

    features = [
        {
            "input_ids":
                chunk,

            "attention_mask":
                [1]
                * len(
                    chunk
                ),
        }
        for chunk in chunks
    ]

    padded = (
        tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
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

            logits = (
                model.score_chunks(
                    input_ids=
                        input_ids,

                    attention_mask=
                        attention_mask,
                )
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
            "ERROR"
            if predicted_error
            else "CLEAN",

        "is_error":
            bool(
                predicted_error
            ),

        "chunk_probabilities":
            [
                float(
                    value
                )
                for value
                in probabilities
            ],

        "chunks_used":
            len(
                probabilities
            ),
    }


# ============================================================
# BUILD WORD SEQUENCE
# ============================================================

def build_words(
    data,
):

    words = []

    fields = []

    local_indices = []

    nonempty = []

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

            nonempty.append(
                (
                    field,
                    field_words,
                )
            )

    for field_number, (
        field,
        field_words,
    ) in enumerate(
        nonempty
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
        "Could not locate content "
        "inside chunk."
    )


# ============================================================
# CONTEXT
# ============================================================

def build_context(
    words,
    fields,
    word_index,
    window=5,
):

    field = (
        fields[
            word_index
        ]
    )

    left = word_index

    count = 0

    while (
        left > 0
        and
        count < window
    ):

        if (
            fields[
                left - 1
            ]
            != field
        ):

            break

        left -= 1
        count += 1

    right = (
        word_index + 1
    )

    count = 0

    while (
        right < len(
            words
        )
        and
        count < window
    ):

        if (
            fields[
                right
            ]
            != field
        ):

            break

        right += 1
        count += 1

    context_words = (
        words[
            left:right
        ]
    )

    return " ".join(
        context_words
    )


# ============================================================
# DISPLAY WORD
# ============================================================

def display_text(text):

    return str(
        text
    ).replace(
        "_",
        " ",
    )


# ============================================================
# WORD DETECTION
# ============================================================

def predict_words(
    data,
    model,
    tokenizer,
    checkpoint,
    threshold,
    top_k=10,
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
            "predicted_errors": [],
            "top_candidates": [],
            "chunks_used": 0,
            "words_scanned": 0,
        }

    # ========================================================
    # TOKENIZE EACH WORD
    # ========================================================

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
            len(
                ids
            )
            == 0
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

    ranges = (
        get_all_chunk_ranges(
            total_tokens=
                len(
                    full_document_ids
                ),

            content_length=
                content_length,

            stride=
                stride,
        )
    )

    # ========================================================
    # BUILD MODEL INPUTS
    # ========================================================

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
                [1]
                * len(
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

    if not chunk_features:

        return {
            "predicted_errors": [],
            "top_candidates": [],
            "chunks_used": 0,
            "words_scanned": 0,
        }

    padded = (
        tokenizer.pad(
            chunk_features,
            padding=True,
            return_tensors="pt",
        )
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

    # ========================================================
    # MODEL
    # ========================================================

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

    # ========================================================
    # MAX OVER OVERLAPPING CHUNKS
    # ========================================================

    word_probabilities = (
        np.zeros(
            len(
                words
            ),
            dtype=np.float32,
        )
    )

    target_word_indices_np = (
        np.asarray(
            target_word_indices,
            dtype=np.int64,
        )
    )

    np.maximum.at(
        word_probabilities,
        target_word_indices_np,
        occurrence_probabilities,
    )

    # ========================================================
    # CANDIDATES
    # ========================================================

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

        context_segmented = (
            build_context(
                words=
                    words,

                fields=
                    fields,

                word_index=
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
                display_text(
                    word
                ),

            "probability":
                probability,

            "is_error":
                bool(
                    probability
                    >= threshold
                ),

            "context_segmented":
                context_segmented,

            "context_display":
                display_text(
                    context_segmented
                ),
        })

    # ========================================================
    # SORT
    # ========================================================

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

    top_candidates = (
        ranked[
            :top_k
        ]
    )

    return {
        "predicted_errors":
            predicted_errors,

        "top_candidates":
            top_candidates,

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
# LOAD INPUT JSON
# ============================================================

def load_input(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    # Allow both:
    #
    # title_segmented
    #
    # or
    #
    # title
    #
    # But title/abstract/article here are still expected
    # to already be word-segmented.

    aliases = {
        "title": "title_segmented",
        "abstract": "abstract_segmented",
        "article": "article_segmented",
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
            "Predict Vietnamese spelling errors "
            "using document + word-level PhoBERT."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to input JSON containing "
            "segmented title/abstract/article."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output JSON path."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help=(
            "Number of highest-probability "
            "words to show."
        ),
    )

    parser.add_argument(
        "--force-word-scan",
        action="store_true",
        help=(
            "Run word detector even when "
            "document detector predicts CLEAN."
        ),
    )

    args = parser.parse_args()

    print("=" * 80)

    print(
        "VN-NEWSSPELL - SPELLING ERROR PREDICTION"
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

    data = (
        load_input(
            input_path
        )
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
    # DOCUMENT CHECKPOINT
    # ========================================================

    print(
        "\nLoading document detector..."
    )

    document_checkpoint = (
        load_torch_checkpoint(
            DOCUMENT_CHECKPOINT_PATH
        )
    )

    document_threshold = (
        load_threshold(
            DOCUMENT_THRESHOLD_PATH
        )
    )

    document_model_name = (
        document_checkpoint[
            "model_name"
        ]
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
            document_model_name
        )
    )

    result = (
        document_model
        .load_state_dict(
            document_checkpoint[
                "state_dict"
            ],
            strict=True,
        )
    )

    print(
        f"Document model missing keys    : "
        f"{len(result.missing_keys)}"
    )

    print(
        f"Document model unexpected keys : "
        f"{len(result.unexpected_keys)}"
    )

    document_model.to(
        DEVICE
    )

    document_model.eval()

    # ========================================================
    # DOCUMENT PREDICTION
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

    # ========================================================
    # RESULT CONTAINER
    # ========================================================

    final_result = {
        "input":
            str(
                input_path
            ),

        "document_detector": {
            "threshold":
                document_threshold,

            **document_result,
        },

        "word_detector":
            None,
    }

    # ========================================================
    # DOCUMENT GATE
    # ========================================================

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
            "Word-level detector was skipped."
        )

        print(
            "\nUse --force-word-scan if you "
            "want to inspect words anyway."
        )

    else:

        # ====================================================
        # FREE DOCUMENT MODEL GPU MEMORY
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

        word_threshold = (
            load_threshold(
                WORD_THRESHOLD_PATH
            )
        )

        word_model_name = (
            word_checkpoint[
                "model_name"
            ]
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
                word_model_name
            )
        )

        result = (
            word_model
            .load_state_dict(
                word_checkpoint[
                    "state_dict"
                ],
                strict=True,
            )
        )

        print(
            f"Word model missing keys    : "
            f"{len(result.missing_keys)}"
        )

        print(
            f"Word model unexpected keys : "
            f"{len(result.unexpected_keys)}"
        )

        word_model.to(
            DEVICE
        )

        word_model.eval()

        # ====================================================
        # WORD PREDICTION
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

        final_result[
            "word_detector"
        ] = {
            "threshold":
                word_threshold,

            **word_result,
        }

        # ====================================================
        # PRINT
        # ====================================================

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

        predicted_errors = (
            word_result[
                "predicted_errors"
            ]
        )

        print(
            f"Errors found : "
            f"{len(predicted_errors)}"
        )

        if predicted_errors:

            print(
                "\nPredicted spelling errors:"
            )

            for rank, item in enumerate(
                predicted_errors,
                start=1,
            ):

                print(
                    "\n"
                    + "-" * 70
                )

                print(
                    f"#{rank}"
                )

                print(
                    f"Field       : "
                    f"{item['field']}"
                )

                print(
                    f"Word        : "
                    f"{item['word_segmented']}"
                )

                print(
                    f"Display     : "
                    f"{item['word_display']}"
                )

                print(
                    f"Field index : "
                    f"{item['word_index_in_field']}"
                )

                print(
                    f"Probability : "
                    f"{item['probability']:.6f}"
                )

                print(
                    f"Context     : "
                    f"{item['context_segmented']}"
                )

        else:

            print(
                "\nNo word crossed the "
                "word-error threshold."
            )

        # ====================================================
        # TOP CANDIDATES
        # ====================================================

        print(
            "\n" + "=" * 80
        )

        print(
            f"TOP {len(word_result['top_candidates'])} "
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
                f"| p={item['probability']:.6f} "
                f"| {status}"
            )

    # ========================================================
    # OUTPUT JSON
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

    # ========================================================
    # FINISH
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "PREDICTION FINISHED"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()