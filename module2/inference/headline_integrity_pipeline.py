from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# ============================================================
# PATHS
# ============================================================

MODULE_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------
# TASK 2A - CLICKBAIT
# ------------------------------------------------------------

CLICKBAIT_LOCK_PATH = (
    MODULE_ROOT
    / "results"
    / "clickbait"
    / "final_title_only_v1"
    / "final_model_lock.json"
)


# ------------------------------------------------------------
# TASK 2B - CONSISTENCY
# ------------------------------------------------------------

CONSISTENCY_LOCK_PATH = (
    MODULE_ROOT
    / "results"
    / "consistency"
    / "final_hybrid_v1"
    / "final_model_lock.json"
)

DESCRIPTION_CHECKPOINT_PATH = (
    MODULE_ROOT
    / "models"
    / "consistency"
    / "baseline_title_description_mbert"
    / "best_model.pt"
)

MIL_CHECKPOINT_PATH = (
    MODULE_ROOT
    / "models"
    / "consistency"
    / "full_article_max_mil_mbert"
    / "best_model.pt"
)


# ============================================================
# EXPECTED FROZEN CONFIGURATION
# ============================================================

MODEL_NAME = (
    "bert-base-multilingual-cased"
)


# ------------------------------------------------------------
# TASK 2A
# ------------------------------------------------------------

CLICKBAIT_MAX_LENGTH = 160
CLICKBAIT_THRESHOLD = 0.61


# ------------------------------------------------------------
# TASK 2B
# ------------------------------------------------------------

CONSISTENCY_MAX_LENGTH = 160

DESCRIPTION_WEIGHT = 0.50
MIL_WEIGHT = 0.50

CONSISTENCY_THRESHOLD = 0.45

CHUNK_TOKENS = 120
CHUNK_OVERLAP = 24
MAX_CHUNKS = 6

CHUNK_STEP = (
    CHUNK_TOKENS
    -
    CHUNK_OVERLAP
)


# ============================================================
# HELPERS
# ============================================================

def load_json(
    path,
):

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(
            f
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


def sha256_file(
    path,
):

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:

                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def load_checkpoint_state(
    path,
):

    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    if (
        isinstance(
            checkpoint,
            dict,
        )
        and
        "model_state_dict"
        in
        checkpoint
    ):

        return checkpoint[
            "model_state_dict"
        ]

    if (
        isinstance(
            checkpoint,
            dict,
        )
        and
        "state_dict"
        in
        checkpoint
    ):

        return checkpoint[
            "state_dict"
        ]

    return checkpoint


def require_file(
    path,
):

    if not path.exists():

        raise FileNotFoundError(
            path
        )


def deep_find(
    obj,
    key,
):
    """
    Find a key recursively inside a JSON-like object.

    Used because Task 2A and Task 2B lock files were created
    by separate pipelines and do not necessarily have exactly
    the same JSON nesting structure.
    """

    if isinstance(
        obj,
        dict,
    ):

        if key in obj:

            return obj[
                key
            ]

        for value in obj.values():

            result = deep_find(
                value,
                key,
            )

            if result is not None:

                return result

    elif isinstance(
        obj,
        list,
    ):

        for value in obj:

            result = deep_find(
                value,
                key,
            )

            if result is not None:

                return result

    return None


# ============================================================
# MAIN PIPELINE
# ============================================================

class HeadlineIntegrityPipeline:

    def __init__(
        self,
        device=None,
    ):

        # ====================================================
        # REQUIRED FILES
        # ====================================================

        required = [
            CLICKBAIT_LOCK_PATH,
            CONSISTENCY_LOCK_PATH,
            DESCRIPTION_CHECKPOINT_PATH,
            MIL_CHECKPOINT_PATH,
        ]

        for path in required:

            require_file(
                path
            )

        # ====================================================
        # DEVICE
        # ====================================================

        if device is None:

            device = (
                "cuda"
                if torch.cuda.is_available()
                else
                "cpu"
            )

        self.device = torch.device(
            device
        )

        self.use_amp = (
            self.device.type
            ==
            "cuda"
        )

        print(
            f"Device: "
            f"{self.device}"
        )

        if (
            self.device.type
            ==
            "cuda"
        ):

            print(
                "GPU   : "
                f"{torch.cuda.get_device_name(0)}"
            )

        # ====================================================
        # LOAD LOCK FILES
        # ====================================================

        print()
        print(
            "Loading final model locks..."
        )

        self.clickbait_lock = (
            load_json(
                CLICKBAIT_LOCK_PATH
            )
        )

        self.consistency_lock = (
            load_json(
                CONSISTENCY_LOCK_PATH
            )
        )

        self._verify_clickbait_lock()
        self._verify_consistency_lock()

        # ====================================================
        # TOKENIZER
        # ====================================================

        print()
        print(
            "Loading shared mBERT tokenizer..."
        )

        self.tokenizer = (
            AutoTokenizer
            .from_pretrained(
                MODEL_NAME
            )
        )

        # ====================================================
        # MODELS
        # ====================================================

        self.clickbait_model = None
        self.description_model = None
        self.mil_model = None

        self._load_models()

        print()
        print(
            "HeadlineIntegrityPipeline READY."
        )

    # ========================================================
    # LOCK VALIDATION
    # ========================================================

    def _verify_clickbait_lock(
        self,
    ):

        final_model = (
            self.clickbait_lock[
                "final_model"
            ]
        )

        architecture = (
            final_model[
                "architecture"
            ]
        )

        input_type = (
            final_model[
                "input"
            ]
        )

        max_length = int(
            final_model[
                "max_length"
            ]
        )

        threshold = float(
            final_model[
                "decision_threshold"
            ]
        )

        if (
            architecture
            !=
            MODEL_NAME
        ):

            raise RuntimeError(
                "Clickbait architecture mismatch."
            )

        if (
            input_type
            !=
            "title_only"
        ):

            raise RuntimeError(
                "Clickbait input must be title_only."
            )

        if (
            max_length
            !=
            CLICKBAIT_MAX_LENGTH
        ):

            raise RuntimeError(
                "Clickbait max_length mismatch."
            )

        if (
            abs(
                threshold
                -
                CLICKBAIT_THRESHOLD
            )
            >
            1e-12
        ):

            raise RuntimeError(
                "Clickbait threshold mismatch."
            )

        checkpoint_relative_path = (
            final_model["checkpoint"]
            .replace("\\", "/")
)

        checkpoint_path = (
            MODULE_ROOT
            /
        Path(checkpoint_relative_path)
)

        require_file(
            checkpoint_path
        )

        expected_hash = (
            final_model[
                "checkpoint_sha256"
            ]
        )

        actual_hash = (
            sha256_file(
                checkpoint_path
            )
        )

        if (
            actual_hash
            !=
            expected_hash
        ):

            raise RuntimeError(
                "Clickbait checkpoint SHA256 mismatch."
            )

        self.clickbait_checkpoint_path = (
            checkpoint_path
        )

        print(
            "Clickbait lock: VERIFIED"
        )

    def _verify_consistency_lock(
        self,
    ):

        description_weight = (
            deep_find(
                self.consistency_lock,
                "description_weight",
            )
        )

        mil_weight = (
            deep_find(
                self.consistency_lock,
                "mil_weight",
            )
        )

        threshold = (
            deep_find(
                self.consistency_lock,
                "decision_threshold",
            )
        )

        if (
            description_weight
            is None
        ):

            raise RuntimeError(
                "Cannot find description_weight "
                "in consistency lock."
            )

        if (
            mil_weight
            is None
        ):

            raise RuntimeError(
                "Cannot find mil_weight "
                "in consistency lock."
            )

        if (
            threshold
            is None
        ):

            raise RuntimeError(
                "Cannot find decision_threshold "
                "in consistency lock."
            )

        description_weight = float(
            description_weight
        )

        mil_weight = float(
            mil_weight
        )

        threshold = float(
            threshold
        )

        if (
            abs(
                description_weight
                -
                DESCRIPTION_WEIGHT
            )
            >
            1e-12
        ):

            raise RuntimeError(
                "Consistency description weight mismatch."
            )

        if (
            abs(
                mil_weight
                -
                MIL_WEIGHT
            )
            >
            1e-12
        ):

            raise RuntimeError(
                "Consistency MIL weight mismatch."
            )

        if (
            abs(
                threshold
                -
                CONSISTENCY_THRESHOLD
            )
            >
            1e-12
        ):

            raise RuntimeError(
                "Consistency threshold mismatch."
            )

        print(
            "Consistency lock: VERIFIED"
        )

    # ========================================================
    # LOAD MODELS
    # ========================================================

    def _create_classifier(
        self,
    ):

        model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                MODEL_NAME,
                num_labels=2,
            )
        )

        return model

    def _load_models(
        self,
    ):

        # ----------------------------------------------------
        # Clickbait
        # ----------------------------------------------------

        print()
        print(
            "Loading Clickbait model..."
        )

        self.clickbait_model = (
            self._create_classifier()
        )

        clickbait_state = (
            load_checkpoint_state(
                self.clickbait_checkpoint_path
            )
        )

        self.clickbait_model.load_state_dict(
            clickbait_state,
            strict=True,
        )

        self.clickbait_model.to(
            self.device
        )

        self.clickbait_model.eval()

        # ----------------------------------------------------
        # Description consistency branch
        # ----------------------------------------------------

        print(
            "Loading Consistency Description model..."
        )

        self.description_model = (
            self._create_classifier()
        )

        description_state = (
            load_checkpoint_state(
                DESCRIPTION_CHECKPOINT_PATH
            )
        )

        self.description_model.load_state_dict(
            description_state,
            strict=True,
        )

        self.description_model.to(
            self.device
        )

        self.description_model.eval()

        # ----------------------------------------------------
        # MAX-MIL branch
        # ----------------------------------------------------

        print(
            "Loading Consistency MAX-MIL model..."
        )

        self.mil_model = (
            self._create_classifier()
        )

        mil_state = (
            load_checkpoint_state(
                MIL_CHECKPOINT_PATH
            )
        )

        self.mil_model.load_state_dict(
            mil_state,
            strict=True,
        )

        self.mil_model.to(
            self.device
        )

        self.mil_model.eval()

    # ========================================================
    # TASK 2A - CLICKBAIT
    # ========================================================

    @torch.inference_mode()
    def predict_clickbait(
        self,
        title,
    ):

        encoded = self.tokenizer(
            title,
            padding=False,
            truncation=True,
            max_length=(
                CLICKBAIT_MAX_LENGTH
            ),
            return_tensors="pt",
        )

        encoded = {
            key:
                value.to(
                    self.device
                )

            for key, value
            in encoded.items()
        }

        with torch.amp.autocast(
            device_type=(
                self.device.type
            ),
            enabled=(
                self.use_amp
            ),
        ):

            outputs = (
                self.clickbait_model(
                    **encoded
                )
            )

        probability = float(
            torch.softmax(
                outputs.logits,
                dim=-1,
            )[
                0,
                1,
            ]
            .detach()
            .cpu()
            .item()
        )

        prediction = int(
            probability
            >=
            CLICKBAIT_THRESHOLD
        )

        label = (
            "CLICKBAIT"
            if prediction == 1
            else
            "NORMAL"
        )

        return {
            "probability_clickbait":
                probability,

            "threshold":
                CLICKBAIT_THRESHOLD,

            "prediction":
                prediction,

            "label":
                label,
        }

    # ========================================================
    # TASK 2B - DESCRIPTION BRANCH
    # ========================================================

    @torch.inference_mode()
    def predict_description_consistency(
        self,
        title,
        lead,
    ):

        encoded = self.tokenizer(
            title,
            lead,
            padding=False,
            truncation=True,
            max_length=(
                CONSISTENCY_MAX_LENGTH
            ),
            return_tensors="pt",
        )

        encoded = {
            key:
                value.to(
                    self.device
                )

            for key, value
            in encoded.items()
        }

        with torch.amp.autocast(
            device_type=(
                self.device.type
            ),
            enabled=(
                self.use_amp
            ),
        ):

            outputs = (
                self.description_model(
                    **encoded
                )
            )

        probability = float(
            torch.softmax(
                outputs.logits,
                dim=-1,
            )[
                0,
                1,
            ]
            .detach()
            .cpu()
            .item()
        )

        return probability

    # ========================================================
    # ARTICLE CHUNKING
    # ========================================================

    def _build_article_chunks(
        self,
        content,
    ):

        # ----------------------------------------------------
        # Tokenize the complete article once.
        #
        # The original Step 17 / Step 20 MIL cache builders
        # use the complete article token sequence with:
        #
        #   chunk size = 120
        #   overlap    = 24
        #   step       = 96
        #   max chunks = 6
        #
        # When >6 candidate windows exist, the original
        # implementation uses:
        #
        #   np.linspace(...)
        #   np.round(...)
        #   astype(int)
        #
        # This exact policy must be preserved here so online
        # inference reproduces the frozen MIL benchmark.
        # ----------------------------------------------------

        old_model_max_length = (
            self.tokenizer.model_max_length
        )

        try:

            # Prevent the cosmetic HuggingFace warning for
            # article sequences longer than BERT's 512-token
            # model input limit.
            #
            # The full sequence is NOT passed directly into
            # BERT. It is only tokenized here and subsequently
            # divided into 120-token chunks.
            self.tokenizer.model_max_length = (
                10**9
            )

            article_ids = (
                self.tokenizer.encode(
                    str(
                        content
                    ),
                    add_special_tokens=False,
                    truncation=False,
                )
            )

        finally:

            self.tokenizer.model_max_length = (
                old_model_max_length
            )

        total_tokens = int(
            len(
                article_ids
            )
        )

        if total_tokens == 0:

            raise ValueError(
                "Article content produced zero tokens."
            )

        # ----------------------------------------------------
        # Short article
        # ----------------------------------------------------

        if (
            total_tokens
            <=
            CHUNK_TOKENS
        ):

            return (
                article_ids,
                [
                    {
                        "start_token":
                            0,

                        "token_ids":
                            article_ids,
                    }
                ],
            )

        # ----------------------------------------------------
        # Sliding-window stride
        # ----------------------------------------------------

        step = (
            CHUNK_TOKENS
            -
            CHUNK_OVERLAP
        )

        if step <= 0:

            raise ValueError(
                "CHUNK_OVERLAP must be "
                "smaller than CHUNK_TOKENS."
            )

        starts = list(
            range(
                0,
                total_tokens,
                step,
            )
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Exact policy used by:
        #
        #   17_prepare_mil_chunk_cache.py
        #   20_prepare_test_mil_chunk_cache.py
        #
        # Do NOT replace np.round with dtype=int / floor.
        # ----------------------------------------------------

        if (
            len(
                starts
            )
            >
            MAX_CHUNKS
        ):

            indices = np.linspace(
                0,
                len(
                    starts
                )
                -
                1,
                MAX_CHUNKS,
            )

            indices = (
                np.round(
                    indices
                )
                .astype(
                    int
                )
                .tolist()
            )

            # Defensive uniqueness check.
            #
            # With MAX_CHUNKS=6 and len(starts)>6 this should
            # already be unique, but keep behavior safe.
            indices = list(
                dict.fromkeys(
                    indices
                )
            )

            starts = [
                starts[
                    index
                ]

                for index
                in indices
            ]

        # ----------------------------------------------------
        # Build actual chunks
        # ----------------------------------------------------

        chunks = []

        for start in starts:

            chunk = article_ids[
                start:
                start
                +
                CHUNK_TOKENS
            ]

            if not chunk:

                continue

            chunks.append(
                {
                    "start_token":
                        int(
                            start
                        ),

                    "token_ids":
                        chunk,
                }
            )

        # ----------------------------------------------------
        # Defensive fallback
        # ----------------------------------------------------

        if not chunks:

            chunks = [
                {
                    "start_token":
                        0,

                    "token_ids":
                        article_ids[
                            :CHUNK_TOKENS
                        ],
                }
            ]

        if (
            len(
                chunks
            )
            >
            MAX_CHUNKS
        ):

            raise RuntimeError(
                "Generated more than MAX_CHUNKS "
                "article chunks."
            )

        return (
            article_ids,
            chunks,
        )

    # ========================================================
    # MANUAL TITLE + CHUNK PAIR
    #
    # Same conceptual construction used in MAX-MIL:
    #
    # [CLS] headline [SEP] article_chunk [SEP]
    #
    # chunk token budget is preserved.
    # headline is truncated only if required.
    # ========================================================

    def _build_mil_pair(
        self,
        headline_ids,
        chunk_ids,
    ):

        cls_id = (
            self.tokenizer.cls_token_id
        )

        sep_id = (
            self.tokenizer.sep_token_id
        )

        pad_id = (
            self.tokenizer.pad_token_id
        )

        if (
            cls_id is None
            or
            sep_id is None
            or
            pad_id is None
        ):

            raise RuntimeError(
                "Tokenizer is missing "
                "CLS/SEP/PAD token IDs."
            )

        chunk_ids = list(
            chunk_ids
        )

        # With 120 chunk tokens and max_length 160,
        # up to 37 headline tokens are available:
        #
        # 1 CLS
        # headline
        # 1 SEP
        # chunk
        # 1 SEP
        #
        # => headline budget =
        # 160 - chunk_len - 3
        headline_budget = (
            CONSISTENCY_MAX_LENGTH
            -
            len(
                chunk_ids
            )
            -
            3
        )

        if headline_budget < 1:

            # Defensive fallback.
            max_chunk = (
                CONSISTENCY_MAX_LENGTH
                -
                4
            )

            chunk_ids = (
                chunk_ids[
                    :max_chunk
                ]
            )

            headline_budget = 1

        used_headline_ids = (
            headline_ids[
                :headline_budget
            ]
        )

        input_ids = (
            [
                cls_id
            ]
            +
            used_headline_ids
            +
            [
                sep_id
            ]
            +
            chunk_ids
            +
            [
                sep_id
            ]
        )

        token_type_ids = (
            [
                0
            ]
            *
            (
                len(
                    used_headline_ids
                )
                +
                2
            )
            +
            [
                1
            ]
            *
            (
                len(
                    chunk_ids
                )
                +
                1
            )
        )

        attention_mask = [
            1
        ] * len(
            input_ids
        )

        padding_length = (
            CONSISTENCY_MAX_LENGTH
            -
            len(
                input_ids
            )
        )

        if padding_length < 0:

            raise RuntimeError(
                "MIL pair exceeds MAX_LENGTH."
            )

        if padding_length > 0:

            input_ids += (
                [
                    pad_id
                ]
                *
                padding_length
            )

            token_type_ids += (
                [
                    0
                ]
                *
                padding_length
            )

            attention_mask += (
                [
                    0
                ]
                *
                padding_length
            )

        return (
            input_ids,
            attention_mask,
            token_type_ids,
        )

    # ========================================================
    # TASK 2B - MAX-MIL BRANCH
    # ========================================================

    @torch.inference_mode()
    def predict_mil_consistency(
        self,
        title,
        content,
    ):

        (
            article_ids,
            chunks,
        ) = self._build_article_chunks(
            content
        )

        headline_ids = (
            self.tokenizer(
                title,
                add_special_tokens=False,
                truncation=False,
            )[
                "input_ids"
            ]
        )

        batch_input_ids = []
        batch_attention = []
        batch_token_types = []

        chunk_starts = []

        for chunk in chunks:

            (
                input_ids,
                attention_mask,
                token_type_ids,
            ) = self._build_mil_pair(
                headline_ids=(
                    headline_ids
                ),
                chunk_ids=(
                    chunk[
                        "token_ids"
                    ]
                ),
            )

            batch_input_ids.append(
                input_ids
            )

            batch_attention.append(
                attention_mask
            )

            batch_token_types.append(
                token_type_ids
            )

            chunk_starts.append(
                chunk[
                    "start_token"
                ]
            )

        input_ids_tensor = torch.tensor(
            batch_input_ids,
            dtype=torch.long,
            device=self.device,
        )

        attention_tensor = torch.tensor(
            batch_attention,
            dtype=torch.long,
            device=self.device,
        )

        token_type_tensor = torch.tensor(
            batch_token_types,
            dtype=torch.long,
            device=self.device,
        )

        with torch.amp.autocast(
            device_type=(
                self.device.type
            ),
            enabled=(
                self.use_amp
            ),
        ):

            outputs = self.mil_model(
                input_ids=(
                    input_ids_tensor
                ),
                attention_mask=(
                    attention_tensor
                ),
                token_type_ids=(
                    token_type_tensor
                ),
            )

        logits = (
            outputs.logits
        )

        # ----------------------------------------------------
        # Binary chunk score used by the trained MAX-MIL:
        #
        # score = logit(label=1) - logit(label=0)
        # ----------------------------------------------------

        chunk_scores = (
            logits[
                :,
                1
            ]
            -
            logits[
                :,
                0
            ]
        )

        # Hard MAX-MIL
        bag_logit = torch.max(
            chunk_scores
        )

        probability = float(
            torch.sigmoid(
                bag_logit
            )
            .detach()
            .cpu()
            .item()
        )

        best_chunk_index = int(
            torch.argmax(
                chunk_scores
            )
            .detach()
            .cpu()
            .item()
        )

        chunk_scores_list = (
            chunk_scores
            .detach()
            .float()
            .cpu()
            .tolist()
        )

        return {
            "probability":
                probability,

            "article_token_count":
                int(
                    len(
                        article_ids
                    )
                ),

            "chunk_count":
                int(
                    len(
                        chunks
                    )
                ),

            "chunk_starts":
                [
                    int(
                        value
                    )
                    for value
                    in chunk_starts
                ],

            "chunk_scores":
                [
                    float(
                        value
                    )
                    for value
                    in chunk_scores_list
                ],

            "max_chunk_index":
                best_chunk_index,

            "max_chunk_start_token":
                int(
                    chunk_starts[
                        best_chunk_index
                    ]
                ),
        }

    # ========================================================
    # TASK 2B - FINAL FUSION
    # ========================================================

    def predict_consistency(
        self,
        title,
        lead,
        content,
    ):

        description_probability = (
            self.predict_description_consistency(
                title=title,
                lead=lead,
            )
        )

        mil_result = (
            self.predict_mil_consistency(
                title=title,
                content=content,
            )
        )

        mil_probability = float(
            mil_result[
                "probability"
            ]
        )

        final_probability = float(
            DESCRIPTION_WEIGHT
            *
            description_probability
            +
            MIL_WEIGHT
            *
            mil_probability
        )

        prediction = int(
            final_probability
            >=
            CONSISTENCY_THRESHOLD
        )

        label = (
            "CONSISTENT"
            if prediction == 1
            else
            "INCONSISTENT"
        )

        return {
            "probability_consistent":
                final_probability,

            "threshold":
                CONSISTENCY_THRESHOLD,

            "prediction":
                prediction,

            "label":
                label,

            "branches": {
                "description": {
                    "weight":
                        DESCRIPTION_WEIGHT,

                    "probability_consistent":
                        float(
                            description_probability
                        ),
                },

                "max_mil": {
                    "weight":
                        MIL_WEIGHT,

                    "probability_consistent":
                        mil_probability,

                    "article_token_count":
                        mil_result[
                            "article_token_count"
                        ],

                    "chunk_count":
                        mil_result[
                            "chunk_count"
                        ],

                    "chunk_starts":
                        mil_result[
                            "chunk_starts"
                        ],

                    "chunk_scores":
                        mil_result[
                            "chunk_scores"
                        ],

                    "max_chunk_index":
                        mil_result[
                            "max_chunk_index"
                        ],

                    "max_chunk_start_token":
                        mil_result[
                            "max_chunk_start_token"
                        ],
                },
            },
        }

    # ========================================================
    # FULL MODULE 2
    # ========================================================

    def predict(
        self,
        title,
        lead,
        content,
    ):

        if title is None:

            raise ValueError(
                "title is required."
            )

        if content is None:

            raise ValueError(
                "content is required."
            )

        title = str(
            title
        ).strip()

        lead = (
            ""
            if lead is None
            else
            str(
                lead
            ).strip()
        )

        content = str(
            content
        ).strip()

        if not title:

            raise ValueError(
                "title cannot be empty."
            )

        if not content:

            raise ValueError(
                "content cannot be empty."
            )

        start_time = time.time()

        clickbait_result = (
            self.predict_clickbait(
                title=title
            )
        )

        consistency_result = (
            self.predict_consistency(
                title=title,
                lead=lead,
                content=content,
            )
        )

        elapsed = float(
            time.time()
            -
            start_time
        )

        return {
            "module":
                "Module 2 - Headline Integrity",

            "input": {
                "title":
                    title,

                "lead_available":
                    bool(
                        lead
                    ),

                "content_chars":
                    int(
                        len(
                            content
                        )
                    ),
            },

            "clickbait":
                clickbait_result,

            "headline_consistency":
                consistency_result,

            # ------------------------------------------------
            # IMPORTANT:
            #
            # No single overall integrity score is produced.
            # 2A and 2B were trained and validated separately.
            # ------------------------------------------------

            "overall_integrity_score":
                None,

            "elapsed_seconds":
                elapsed,

            "model_versions": {
                "clickbait":
                    "final_title_only_v1",

                "consistency":
                    "final_hybrid_v1",
            },
        }


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Unified Module 2 - "
            "Headline Integrity inference pipeline"
        )
    )

    parser.add_argument(
        "--input-json",
        type=str,
        required=True,
        help=(
            "JSON containing title, lead, content"
        ),
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help=(
            "Optional output JSON path"
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=[
            "cpu",
            "cuda",
        ],
    )

    args = parser.parse_args()

    input_path = Path(
        args.input_json
    )

    if not input_path.exists():

        raise FileNotFoundError(
            input_path
        )

    input_data = load_json(
        input_path
    )

    required_keys = [
        "title",
        "content",
    ]

    for key in required_keys:

        if key not in input_data:

            raise RuntimeError(
                f"Missing input key: "
                f"{key}"
            )

    title = input_data[
        "title"
    ]

    lead = input_data.get(
        "lead",
        input_data.get(
            "description",
            "",
        ),
    )

    content = input_data[
        "content"
    ]

    print("=" * 80)
    print(
        "MODULE 2 - HEADLINE INTEGRITY"
    )
    print(
        "UNIFIED INFERENCE PIPELINE"
    )
    print("=" * 80)

    pipeline = (
        HeadlineIntegrityPipeline(
            device=args.device
        )
    )

    print()
    print("=" * 80)
    print(
        "RUN INFERENCE"
    )
    print("=" * 80)

    result = pipeline.predict(
        title=title,
        lead=lead,
        content=content,
    )

    print()
    print("=" * 80)
    print(
        "RESULT"
    )
    print("=" * 80)

    print()
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )

    if (
        args.output_json
        is not None
    ):

        output_path = Path(
            args.output_json
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_json(
            result,
            output_path,
        )

        print()
        print(
            f"Saved:\n"
            f"{output_path}"
        )


if __name__ == "__main__":
    main()
